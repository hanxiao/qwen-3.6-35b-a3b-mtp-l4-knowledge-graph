"""In-process job scheduler for KG extraction.

Single-slot serial queue with foreground preemption and auto-backfill, modeled
on the searchbox scheduler but implemented entirely with asyncio (no subprocess).

States:
  queued   - foreground, waiting for the slot (fresh submit or explicit resume)
  running  - on the slot now
  paused   - preemptible idle pool; auto-backfill may resume it. A preempted
             foreground job carries preempted=True so it outranks bulk paused.
  held     - USER-paused; sticky, NOT auto-backfilled. Explicit resume only.
  done     - finished all work
  failed   - errored out

Everything is persisted to data/jobs/<id>/ so the list + jsonl reload + resume
survive restarts and container rebuilds (mount data/ as a volume).
"""

import os, json, time, asyncio, uuid, shutil, zipfile, hashlib
from pathlib import Path
from typing import Optional

JOBS_DIR = Path(os.environ.get("JOBS_DIR", str(Path(__file__).parent / "data" / "jobs"))).resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Foreground preemption: a fresh submit / explicit resume pauses ANY running job
# and takes the slot. Auto-backfill runs are always preemptible.
PREEMPT_FOREGROUND = os.environ.get("PREEMPT_FOREGROUND", "1") != "0"
AUTO_BACKFILL = os.environ.get("AUTO_BACKFILL", "1") != "0"

# In-memory job registry: job_id -> meta dict
_jobs: dict[str, dict] = {}
_queue: list[str] = []                  # FIFO of foreground 'queued' job ids
_current = {"job_id": None}             # the running job id
_cond = asyncio.Condition()             # guards _jobs/_queue/_current + wakes worker
_subscribers: dict[str, list] = {}      # job_id -> list[asyncio.Queue] (live SSE listeners)
_pause_flags: dict[str, bool] = {}      # job_id -> cooperative pause request


# ---------- persistence ----------
def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id

def _meta_path(job_id: str) -> Path:
    return job_dir(job_id) / "meta.json"

def jsonl_path(job_id: str) -> Path:
    return job_dir(job_id) / "facts.jsonl"

def save_meta(job_id: str):
    try:
        d = job_dir(job_id); d.mkdir(parents=True, exist_ok=True)
        # never persist transient subscriber-only keys
        meta = {k: v for k, v in _jobs.get(job_id, {}).items() if not k.startswith("_")}
        _meta_path(job_id).write_text(json.dumps(meta))
    except Exception:
        pass

def load_meta(job_id: str) -> dict:
    try:
        return json.loads(_meta_path(job_id).read_text())
    except Exception:
        return {}


def public_meta(m: dict) -> dict:
    """Trimmed job dict for the list/status API."""
    return {
        "job_id": m.get("job_id"),
        "status": m.get("status"),
        "title": m.get("title") or m.get("source_name") or "(job)",
        "source_kind": m.get("source_kind"),
        "source_name": m.get("source_name"),
        "num_files": m.get("num_files"),
        "files_done": m.get("files_done", 0),
        "total_facts": m.get("total_facts", 0),
        "unique_facts": m.get("unique_facts", 0),
        "duplicate_facts": m.get("duplicate_facts", 0),
        "k": m.get("k", 1),
        "dedup_model": m.get("dedup_model", ""),
        "preempted": bool(m.get("preempted")),
        "submitted": m.get("submitted"),
        "finished": m.get("finished"),
        "error": m.get("error"),
    }


# ---------- broadcast ----------
def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=10000)
    _subscribers.setdefault(job_id, []).append(q)
    return q

def unsubscribe(job_id: str, q: asyncio.Queue):
    lst = _subscribers.get(job_id)
    if lst and q in lst:
        lst.remove(q)

def _emit(job_id: str, event: dict):
    for q in list(_subscribers.get(job_id, [])):
        try:
            q.put_nowait(event)
        except Exception:
            pass


# ---------- startup reconcile ----------
def reconcile_on_startup():
    """A fresh process owns no running jobs. Load all jobs; collapse any mid-flight
    (running/queued/pausing) job into resumable 'paused' so it rejoins the backfill pool."""
    if not JOBS_DIR.exists():
        return
    for p in sorted(JOBS_DIR.iterdir()):
        if not p.is_dir():
            continue
        m = load_meta(p.name)
        if not m:
            continue
        m["job_id"] = p.name
        st = m.get("status")
        if st in ("running", "queued", "pausing"):
            m["status"] = "paused"
        # 'held'/'paused'/'done'/'failed' keep their state
        _jobs[p.name] = m
        save_meta(p.name)


# ---------- selection ----------
def _next_foreground() -> Optional[str]:
    for j in _queue:
        if _jobs.get(j, {}).get("status") == "queued":
            return j
    return None

def _paused_pool() -> list:
    """Backfill candidates in resume priority: preempted foreground first, then oldest paused."""
    cands = [m for m in _jobs.values() if m.get("status") == "paused"]
    cands.sort(key=lambda m: (0 if m.get("preempted") else 1, m.get("submitted") or 0))
    return [m["job_id"] for m in cands]

def _select_next():
    """Return (job_id, is_backfill) or (None, False). Foreground 'queued' wins."""
    fg = _next_foreground()
    if fg is not None:
        if fg in _queue:
            _queue.remove(fg)
        return fg, False
    if AUTO_BACKFILL:
        for cand in _paused_pool():
            m = _jobs.get(cand, {})
            if m.get("status") != "paused":
                continue
            m["status"] = "queued"; m["preempted"] = False
            _jobs[cand] = m; save_meta(cand)
            return cand, True
    return None, False

def _preempt_running():
    """Flag the running job to pause so the slot frees for fresh foreground work."""
    jid = _current["job_id"]
    if not jid:
        return
    cur = _jobs.get(jid, {})
    if cur.get("status") != "running":
        return
    is_auto = bool(cur.get("auto"))
    if not is_auto and not PREEMPT_FOREGROUND:
        return
    _pause_flags[jid] = True
    cur["status"] = "pausing"
    if not is_auto:
        cur["preempted"] = True
    save_meta(jid)


# ---------- public API used by the FastAPI layer ----------
async def create_job(meta: dict, source_bytes: bytes, source_filename: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    d = job_dir(job_id); d.mkdir(parents=True, exist_ok=True)
    if meta.get("source_kind") == "zip":
        (d / "input.zip").write_bytes(source_bytes)
    else:
        (d / "input.json").write_text(source_bytes.decode("utf-8", "ignore"))
    jsonl_path(job_id).write_text("")
    meta.update({"job_id": job_id, "status": "queued", "submitted": time.time(),
                 "files_done": 0, "total_facts": 0, "unique_facts": 0,
                 "duplicate_facts": 0, "finished": None, "auto": False})
    async with _cond:
        _jobs[job_id] = meta
        _queue.append(job_id)
        _preempt_running()           # fresh submit preempts the running job
        _cond.notify_all()
    save_meta(job_id)
    return job_id

async def pause_job(job_id: str) -> dict:
    """User pause: sticky 'held', not auto-backfilled."""
    async with _cond:
        m = _jobs.get(job_id) or load_meta(job_id)
        if not m:
            return {"error": "no such job"}
        if m.get("status") in ("done", "failed"):
            return public_meta(m)
        running = _current["job_id"] == job_id
        if running:
            _pause_flags[job_id] = True
            m["status"] = "pausing"
        else:
            m["status"] = "held"
        m["preempted"] = False
        if job_id in _queue:
            _queue.remove(job_id)
        _jobs[job_id] = m; save_meta(job_id)
        _cond.notify_all()
    return public_meta(m)

async def resume_job(job_id: str) -> dict:
    """Re-enqueue a held/paused job as foreground (preempts running)."""
    async with _cond:
        m = _jobs.get(job_id) or load_meta(job_id)
        if not m:
            return {"error": "no such job"}
        if m.get("status") in ("running", "queued", "pausing"):
            return public_meta(m)
        m["job_id"] = job_id
        m["status"] = "queued"; m["preempted"] = False
        _jobs[job_id] = m
        if job_id not in _queue:
            _queue.append(job_id)
        _preempt_running()
        _cond.notify_all()
    save_meta(job_id)
    return public_meta(m)

async def delete_job(job_id: str) -> dict:
    async with _cond:
        running = _current["job_id"] == job_id
        if running:
            _pause_flags[job_id] = True       # stop it; it will exit, then we purge
        if job_id in _queue:
            _queue.remove(job_id)
        m = _jobs.pop(job_id, None)
        _cond.notify_all()
    if not running:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
    else:
        _jobs_pending_delete.add(job_id)
    return {"ok": True}

_jobs_pending_delete: set = set()
_user_held: dict = {}     # job_id -> True when the pending pause is a sticky USER hold

def list_jobs() -> list:
    out = [public_meta(m) for m in _jobs.values()]
    out.sort(key=lambda m: -(m.get("submitted") or 0))
    return out

def get_job(job_id: str) -> Optional[dict]:
    m = _jobs.get(job_id)
    return public_meta(m) if m else None

def read_jsonl(job_id: str) -> str:
    try:
        return jsonl_path(job_id).read_text()
    except Exception:
        return ""

def is_paused_requested(job_id: str) -> bool:
    return _pause_flags.get(job_id, False)
