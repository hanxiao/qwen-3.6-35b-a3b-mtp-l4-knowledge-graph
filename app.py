#!/usr/bin/env python3
"""Knowledge Graph Extractor - LLM triple extraction (Qwen3.6 MTP) + jina-v5-nano semantic dedup."""

import json, time, hashlib, re, random, os, io, zipfile, tempfile, shutil, asyncio, numpy as np
from typing import AsyncGenerator
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import httpx
import uvicorn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only for embedding model

app = FastAPI()

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080")
JINA_KEY = os.environ.get("JINA_API_KEY", "")
CTX_SIZE = 16384
# The 16384-token ctx must hold the whole request (prompt + doc) AND the output
# budget (max_tokens=8192). So the input side must stay <= ~8192 tokens. With the
# instruction prompt ~1.4K tokens, that leaves ~6.5K tokens (~24K chars) for the
# doc per LLM call. Instead of truncating oversized docs, we CHUNK them so the
# whole text gets extracted (see chunk_text).
MAX_INPUT_CHARS = 24000


def chunk_text(text: str, limit: int = MAX_INPUT_CHARS) -> list:
    """Split an oversized doc into <=limit-char chunks so the FULL text is
    extracted, never truncated.

    Strategy per Han: don't cut off -- back off the chunk size by halving
    exponentially until a piece fits, take that piece, then apply the SAME
    strategy to the remainder. In practice: walk the text, and for each step
    take the largest prefix that fits within `limit`, preferring to break on a
    paragraph/sentence/word boundary near the cut (found by halving back from
    the hard limit) so chunks stay coherent. Returns a list of chunk strings
    whose concatenation equals the original text (modulo the split points).
    """
    text = text or ""
    if len(text) <= limit:
        return [text] if text.strip() else []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        if n - i <= limit:
            piece = text[i:]
            if piece.strip():
                chunks.append(piece)
            break
        # hard window [i, i+limit]; find a clean break by backing off (halving)
        hard = i + limit
        cut = -1
        # exponential back-off: try boundaries within the last 1/2, 1/4, 1/8 ...
        # of the window, looking for a paragraph break, then sentence, then space.
        back = limit
        while back >= 64:
            lo = hard - back
            for seps in ("\n\n", "\n", ". ", "? ", "! ", " "):
                pos = text.rfind(seps, lo, hard)
                if pos > i:
                    cut = pos + len(seps)   # include the separator in this chunk
                    break
            if cut > i:
                break
            back //= 2
        if cut <= i:
            cut = hard  # no boundary found; hard cut (still no data loss)
        piece = text[i:cut]
        if piece.strip():
            chunks.append(piece)
        i = cut
    return chunks

DEDUP_MODEL = None
DEDUP_MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"

def get_dedup_model():
    global DEDUP_MODEL
    if DEDUP_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("Loading dedup model on CPU...")
        DEDUP_MODEL = SentenceTransformer(DEDUP_MODEL_NAME, device="cpu", trust_remote_code=True)
        print("Dedup model loaded.")
    return DEDUP_MODEL

def embed_fact(fact: dict, field: str = "triple") -> np.ndarray:
    model = get_dedup_model()
    if field == "triple":
        text = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
    elif field == "title":
        text = fact.get('title', '')
    elif field == "description":
        text = fact.get('description', '')
    elif field == "title+desc":
        text = f"{fact.get('title', '')} {fact.get('description', '')}"
    elif field == "triple+title":
        text = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')} {fact.get('title', '')}"
    else:  # all
        text = f"{fact.get('title', '')} {fact.get('description', '')} {fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
    emb = model.encode([text], task="text-matching", normalize_embeddings=True)
    return emb[0]

def check_duplicate(new_emb: np.ndarray, existing_embs: list, existing_indices: list, threshold: float) -> tuple[bool, float, list]:
    """Returns (is_dup, max_sim, dup_of_list) where dup_of_list has {index, similarity}
    for all matches at/above threshold. Vectorized (single BLAS matvec) so it stays
    fast even with thousands of accumulated embeddings."""
    if not existing_embs:
        return False, 0.0, []
    sims = np.asarray(existing_embs) @ new_emb   # (n,) cosine sims (embs are normalized)
    max_sim = float(sims.max())
    if max_sim < threshold:
        return False, max_sim, []
    hits = np.nonzero(sims >= threshold)[0]
    dup_of = sorted(
        ({"index": existing_indices[i], "similarity": round(float(sims[i]), 4)} for i in hits),
        key=lambda x: -x["similarity"],
    )
    return True, max_sim, dup_of


DEFAULT_PROMPT = """Extract a knowledge graph from the document. Return a JSON object with key
"facts" containing 0-15 atomic relationship facts. Each fact is ONE edge:
a (subject) --[predicate]--> (object) triple plus human-readable context.
Long, dense documents (articles, papers, profiles) typically warrant 8-15
facts; short or generic pages 0-3.

The single most important rule -- THIS DRIVES GRAPH CONNECTIVITY:
subject and object MUST be canonical ENTITIES or short atomic VALUES, never
prose. They are graph nodes: the same entity must come out identical every
time so edges connect. Put narrative, evidence and nuance in the description,
NOT in subject/object.

subject / object rules:
- Use the shortest canonical name of a real entity: a person, organisation,
 product/model, dataset, method, paper, place, technology, or a concrete
 atomic value (a date, a number+unit, a version, a metric score).
- Strip articles, roles, and qualifiers: "the Jina AI team" -> "Jina AI";
 "a model called jina-embeddings-v3" -> "jina-embeddings-v3".
- Use the canonical surface form, not a pronoun or paraphrase. Reuse the exact
 same string for the same entity across every fact (this is how nodes merge).
- Never put a sentence or clause in subject/object. If the value is inherently
 descriptive (e.g. "trained on 2B multilingual tokens"), make object the
 atomic value ("2B multilingual tokens") and explain in description.
- Prefer relationships that link TWO named entities (entity-entity edges) --
 these are what make the graph rich. Entity-value edges are fine too.

Each fact:
 {
 "title": "<one natural sentence <=140 chars stating the fact, ending with the value when possible>",
 "description": "<2-3 sentences <=350 chars carrying the answer + evidence: entities, relation, value, date/number/source detail, and an inline verbatim quote when it disambiguates. Avoid restating the title verbatim.>",
 "subject": "<canonical entity name, short>",
 "predicate": "<precise snake_case relation, <=32 chars>",
 "object": "<canonical entity name OR short atomic value>",
 "evidence_span": "<verbatim 1-3 sentence quote, substring of the doc text above>",
 "confidence": <0..100 integer>,
 "tags": ["<entity/topic/year tags, lowercase, alphanumeric+hyphen>", ...]
 }
Coverage priorities -- extract a fact for EACH of the following when grounded in the text:
- Every named person + their role / position / affiliation (even if named once).
- Every named organisation, product, model, dataset, or method + how it relates
 to other named entities (built_by, based_on, trained_on, outperforms, part_of).
- Every concrete date/version + the event or release it marks.
- Every named place + what happened or is located there.
- Every quantitative result: metric scores, sizes, token counts, speedups,
 prices -- as entity --[has_metric/scored]--> value edges.
- Every cross-entity relationship: X built Y, X based on Y, X collaborated with
 Y, X acquired Y, X cites Y, X compared against Y.
Anti-patterns -- do NOT do these:
- Don't put descriptive sentences into subject or object. That creates dead-end
 nodes that never connect. Keep nodes short and canonical.
- Don't only extract facts about the dominant entity. Secondary entities named
 once still warrant their own edge.
- Don't fill the budget with generic boilerplate (tagline, copyright, nav) at
 the expense of specific, connectable relationships deeper in the body.
Predicate guidance:
- Always choose the MOST precise snake_case predicate (<=32 chars) for the actual
 relation -- accuracy matters more than reusing a known term. Coin a new one
 freely whenever it fits better.
- The following are only EXAMPLES of the style/granularity (NOT a fixed list,
 NOT a menu to pick from): built_by, based_on, trained_on, fine_tuned_from,
 released_on, outperforms, evaluated_on, scored, integrates_with, acquired_by,
 authored_by, cites, position_held, successor_of, used_for. Do not force a
 relation into one of these if a more specific predicate describes it better.
- AVOID vague catch-alls like affiliated_with or related_to.
Title and description constraints (CRITICAL -- items violating these are dropped):
- title and description MUST read as natural standalone fact statements.
- They MUST NOT mention the document, the dataset, or this task.
Fact constraints:
- Favor specificity (proper nouns, versions, numbers) over generic claims.
- Skip the doc entirely (return empty facts list) for navigation pages, login
 walls, error pages, very short or generic content.
- evidence_span must be a verbatim substring of the doc text supplied above.

Output ONLY the JSON object."""

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "evidence_span": {"type": "string"},
                    "confidence": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "description", "subject", "predicate", "object", "evidence_span", "confidence", "tags"]
            }
        }
    },
    "required": ["facts"]
}


def parse_facts_incremental(text: str, already_parsed: set) -> list:
    new_facts = []
    pattern = re.compile(r'\{\s*"title"\s*:')
    for m in pattern.finditer(text):
        start = m.start()
        depth = 0
        i = start
        while i < len(text):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    obj_str = text[start:i+1]
                    obj_hash = hash(obj_str)
                    if obj_hash not in already_parsed:
                        try:
                            fact = json.loads(obj_str)
                            if "title" in fact and "subject" in fact:
                                already_parsed.add(obj_hash)
                                new_facts.append(fact)
                        except json.JSONDecodeError:
                            pass
                    break
            i += 1
    return new_facts


async def fetch_url(url: str) -> tuple[str, str, bool]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {JINA_KEY}", "Accept": "text/markdown"},
        )
        r.raise_for_status()
        text = r.text
        title = ""
        for line in text.strip().split("\n")[:5]:
            if line.startswith("Title:"):
                title = line[6:].strip()
                break
            elif line.startswith("# "):
                title = line[2:].strip()
                break
        # Return the FULL text; chunking happens at extraction time so nothing
        # is dropped. 'truncated' kept for the UI 'chars' note only.
        return text, title, False


def read_file_text(path: str, name: str) -> str:
    """Best-effort extract plain text/markdown from an uploaded file."""
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                return "\n\n".join((p.extract_text() or "") for p in reader.pages)
            except Exception:
                return ""
        if ext in (".html", ".htm"):
            raw = open(path, "r", encoding="utf-8", errors="ignore").read()
            raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
            raw = re.sub(r"(?s)<[^>]+>", " ", raw)
            raw = re.sub(r"&nbsp;", " ", raw)
            return re.sub(r"[ \t]+", " ", raw)
        if ext == ".docx":
            try:
                import zipfile as _zf
                with _zf.ZipFile(path) as z:
                    xml = z.read("word/document.xml").decode("utf-8", "ignore")
                xml = re.sub(r"(?s)<w:p[ >]", "\n", xml)
                return re.sub(r"(?s)<[^>]+>", "", xml)
            except Exception:
                return ""
        # default: treat as utf-8 text (txt, md, json, csv, code, etc.)
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""


TEXT_EXTS = {".txt", ".md", ".markdown", ".text", ".rst", ".json", ".jsonl",
             ".csv", ".tsv", ".log", ".html", ".htm", ".pdf", ".docx",
             ".xml", ".yaml", ".yml", ".py", ".js", ".ts"}


def list_zip_files(zip_path: str, extract_dir: str) -> list:
    """Extract a zip safely and return [{name, path}] for supported text files."""
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            base = os.path.basename(name)
            if not base or base.startswith(".") or "__MACOSX" in name:
                continue
            ext = os.path.splitext(base)[1].lower()
            if ext not in TEXT_EXTS:
                continue
            # safe extract (no path traversal)
            target = os.path.join(extract_dir, os.path.relpath(os.path.join(extract_dir, name), extract_dir))
            if not os.path.abspath(target).startswith(os.path.abspath(extract_dir)):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append({"name": name, "path": target})
    out.sort(key=lambda x: x["name"])
    return out


async def stream_single_extraction(
    body_text: str, extraction_prompt: str, url: str, docid: str,
    round_num: int, seed: int, fact_offset: int,
    existing_embs: list, existing_indices: list, dedup_threshold: float, dedup_field: str = "triple",
    dedup_enabled: bool = True, source_file: str = ""
) -> AsyncGenerator[str, None]:
    full_prompt = f"{extraction_prompt}\n\nDocument:\n  docid: {docid}\n  url: {url or 'n/a'}\n  text: {body_text}"
    prompt_tokens_est = len(full_prompt) // 4
    system_tokens_est = len(extraction_prompt) // 4
    doc_tokens_est = len(body_text) // 4

    payload = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 8192,
        "stream": True,
        "seed": seed,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "response_format": {"type": "json_schema", "json_schema": {"name": "ki_facts", "strict": True, "schema": FACT_SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    yield f"data: {json.dumps({'type': 'round_start', 'round': round_num, 'seed': seed, 'prompt_tokens_est': prompt_tokens_est, 'system_tokens_est': system_tokens_est, 'doc_tokens_est': doc_tokens_est})}\n\n"

    start_time = time.time()
    content_buf = ""
    token_count = 0
    round_facts = 0
    round_dupes = 0
    parsed_hashes: set = set()

    timeout = httpx.Timeout(connect=10, read=300, write=30, pool=300)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{LLAMA_URL}/v1/chat/completions",
            json=payload, headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                msg = body.decode("utf-8", "ignore")[:300]
                yield f"data: {json.dumps({'type': 'llm_error', 'round': round_num, 'status': response.status_code, 'message': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'round_end', 'round': round_num, 'round_facts': round_facts, 'round_dupes': round_dupes, 'tokens': 0, 'elapsed': 0, 'tps': 0, 'note': 'llm error'})}\n\n"
                return
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if not content:
                    continue

                content_buf += content
                token_count += 1
                elapsed = time.time() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                if token_count % 10 == 0:
                    yield f"data: {json.dumps({'type': 'metrics', 'round': round_num, 'tokens': token_count, 'elapsed': round(elapsed, 1), 'tps': round(tps, 1), 'prompt_tokens_est': prompt_tokens_est, 'system_tokens_est': system_tokens_est, 'doc_tokens_est': doc_tokens_est})}\n\n"

                if token_count % 5 == 0:
                    new_facts = parse_facts_incremental(content_buf, parsed_hashes)
                    for fact in new_facts:
                        if source_file:
                            fact["source_file"] = source_file
                        round_facts += 1
                        total_idx = fact_offset + round_facts
                        if dedup_enabled:
                            emb = embed_fact(fact, dedup_field)
                            is_dup, max_sim, dup_of = check_duplicate(emb, existing_embs, existing_indices, dedup_threshold)
                            if is_dup:
                                round_dupes += 1
                            else:
                                existing_embs.append(emb)
                                existing_indices.append(total_idx)
                        else:
                            is_dup, max_sim, dup_of = False, 0.0, []
                        yield f"data: {json.dumps({'type': 'fact', 'round': round_num, 'index': total_idx, 'fact': fact, 'is_duplicate': is_dup, 'max_similarity': round(max_sim, 4), 'dup_of': dup_of, 'tokens': token_count, 'elapsed': round(elapsed, 1), 'tps': round(tps, 1)})}\n\n"

    elapsed = time.time() - start_time
    tps = token_count / elapsed if elapsed > 0 else 0
    new_facts = parse_facts_incremental(content_buf, parsed_hashes)
    for fact in new_facts:
        if source_file:
            fact["source_file"] = source_file
        round_facts += 1
        total_idx = fact_offset + round_facts
        if dedup_enabled:
            emb = embed_fact(fact, dedup_field)
            is_dup, max_sim, dup_of = check_duplicate(emb, existing_embs, existing_indices, dedup_threshold)
            if is_dup:
                round_dupes += 1
            else:
                existing_embs.append(emb)
                existing_indices.append(total_idx)
        else:
            is_dup, max_sim, dup_of = False, 0.0, []
        yield f"data: {json.dumps({'type': 'fact', 'round': round_num, 'index': total_idx, 'fact': fact, 'is_duplicate': is_dup, 'max_similarity': round(max_sim, 4), 'dup_of': dup_of, 'tokens': token_count, 'elapsed': round(elapsed, 1), 'tps': round(tps, 1)})}\n\n"

    cached_note = "(prompt cached)" if round_num > 1 else ""
    yield f"data: {json.dumps({'type': 'round_end', 'round': round_num, 'round_facts': round_facts, 'round_dupes': round_dupes, 'tokens': token_count, 'elapsed': round(elapsed, 1), 'tps': round(tps, 1), 'note': cached_note})}\n\n"


import jobs as J

# ---------- job runner: does the actual extraction for one job, cooperatively ----------
async def run_job(job_id: str):
    """Run (or resume) one extraction job. Emits SSE-style events via J._emit and
    appends each fact to facts.jsonl. Honors cooperative pause at file/round edges:
    on pause it persists progress and returns without marking done."""
    meta = J._jobs.get(job_id, {})
    k_rounds = max(1, min(10, int(meta.get("k", 1))))
    extraction_prompt = (meta.get("prompt") or DEFAULT_PROMPT).strip()
    dedup_threshold = float(meta.get("dedup_threshold", 0.90))
    dedup_field = meta.get("dedup_field", "triple")
    dedup_enabled = bool(meta.get("dedup_model"))
    d = J.job_dir(job_id)

    def emit(ev):
        J._emit(job_id, ev)

    # ---- rebuild dedup state + counters from prior facts.jsonl (resume) ----
    existing_embs: list = []
    existing_indices: list = []
    total_facts = int(meta.get("total_facts", 0))
    total_dupes = int(meta.get("duplicate_facts", 0))
    done_files = set(meta.get("done_files", []))
    prior_lines = []
    jp = J.jsonl_path(job_id)
    if jp.exists() and total_facts:
        for line in jp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            prior_lines.append(rec)
            if dedup_enabled and not rec.get("is_duplicate"):
                emb = embed_fact(rec, dedup_field)
                existing_embs.append(emb)
                existing_indices.append(len(existing_indices) + 1)

    jsonl_fh = open(jp, "a", encoding="utf-8")

    def persist_progress(status):
        meta["total_facts"] = total_facts
        meta["duplicate_facts"] = total_dupes
        meta["unique_facts"] = total_facts - total_dupes
        meta["done_files"] = sorted(done_files)
        meta["files_done"] = len(done_files)
        # 'keep' = persist counters only, don't touch status (scheduler owns it on pause).
        # Also never clobber a pause/stop requested mid-run with 'running'.
        if status == "keep":
            pass
        elif status == "running" and (paused_requested() or meta.get("status") in ("pausing", "held", "paused")):
            pass
        else:
            meta["status"] = status
        J._jobs[job_id] = meta
        J.save_meta(job_id)

    def paused_requested():
        return J.is_paused_requested(job_id)

    try:
        # replay prior facts to any (re)connected viewer so the graph rebuilds
        if prior_lines:
            emit({"type": "replay", "facts": prior_lines})

        # ---- build the work item list (zip files, or single url/text) ----
        if meta.get("source_kind") == "zip":
            tmpdir = tempfile.mkdtemp(prefix="kizip_")
            extract_dir = os.path.join(tmpdir, "ex"); os.makedirs(extract_dir, exist_ok=True)
            try:
                files = list_zip_files(str(d / "input.zip"), extract_dir)
            except Exception:
                files = []
            if not files:
                emit({"type": "error", "message": "No supported text files in zip"})
                persist_progress("failed"); meta["finished"] = time.time(); J.save_meta(job_id)
                jsonl_fh.close(); shutil.rmtree(tmpdir, ignore_errors=True)
                return
            meta["num_files"] = len(files)
            emit({"type": "filelist", "files": [f["name"] for f in files], "done_files": sorted(done_files)})
            # On resume, only read files we still need (skip already-done ones).
            work = [(f["name"], None if f["name"] in done_files else read_file_text(f["path"], f["name"])) for f in files]
            cleanup = lambda: shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            payload = json.loads((d / "input.json").read_text())
            url = (payload.get("url") or "").strip()
            text = (payload.get("text") or "").strip()
            if url:
                emit({"type": "status", "message": "Fetching via Jina Reader..."})
                doc_text, title, truncated = await fetch_url(url)
                emit({"type": "fetched", "title": title, "chars": len(doc_text), "truncated": truncated})
            else:
                doc_text = text
                emit({"type": "fetched", "title": "Pasted text", "chars": len(doc_text), "truncated": False})
            meta["num_files"] = 1
            work = [(url or "pasted", doc_text)]
            cleanup = lambda: None

        # ---- iterate work items, skipping already-done ones (resume) ----
        for fi, (fname, doc_text) in enumerate(work):
            if fname in done_files:
                continue
            if paused_requested():
                persist_progress("keep"); cleanup(); jsonl_fh.close(); return  # scheduler finalizes state
            emit({"type": "file_start", "file_index": fi, "file": fname})
            if not (doc_text or "").strip():
                done_files.add(fname)
                emit({"type": "file_end", "file_index": fi, "file": fname, "file_facts": 0, "skipped": True})
                persist_progress("running")
                continue
            # Split oversized docs so the FULL text is extracted (no truncation).
            chunks = chunk_text(doc_text, MAX_INPUT_CHARS)
            if len(chunks) > 1:
                emit({"type": "status", "message": f"Doc is large ({len(doc_text):,} chars) -> {len(chunks)} chunks"})
            file_facts = 0
            file_llm_error = None
            for ci, chunk in enumerate(chunks):
                if paused_requested():
                    break
                docid = hashlib.md5(chunk[:500].encode()).hexdigest()[:8]
                for r in range(1, k_rounds + 1):
                    seed = random.randint(1, 999999)
                    async for event in stream_single_extraction(
                        chunk, extraction_prompt, fname if meta.get("source_kind") != "zip" else "",
                        docid, round_num=r, seed=seed, fact_offset=total_facts,
                        existing_embs=existing_embs, existing_indices=existing_indices,
                        dedup_threshold=dedup_threshold, dedup_field=dedup_field,
                        dedup_enabled=dedup_enabled,
                        source_file=fname if meta.get("source_kind") == "zip" else "",
                    ):
                        if not event.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(event[6:])
                        except Exception:
                            continue
                        if ev.get("type") == "fact":
                            rec = {**ev["fact"], "is_duplicate": ev["is_duplicate"]}
                            jsonl_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            jsonl_fh.flush()
                            emit({"type": "fact", "fact": rec, "is_duplicate": ev["is_duplicate"], "tps": ev.get("tps", 0)})
                        elif ev.get("type") == "round_end":
                            total_facts += ev["round_facts"]
                            total_dupes += ev["round_dupes"]
                            file_facts += ev["round_facts"]
                        elif ev.get("type") == "llm_error":
                            file_llm_error = f"llama {ev.get('status')}: {ev.get('message','')}"
                            emit({"type": "warn", "file": fname, "message": file_llm_error})
                        elif ev.get("type") in ("metrics", "round_start"):
                            emit(ev)
                # persist after each chunk so the live counter advances on long docs
                persist_progress("running")
            # A single-document job (url/text) with an LLM error and no facts is a
            # real failure -- surface it instead of a fake 'done, 0 facts'.
            if file_llm_error and meta.get("source_kind") != "zip" and file_facts == 0:
                cleanup(); jsonl_fh.close()
                meta["error"] = file_llm_error; persist_progress("failed")
                meta["finished"] = time.time(); J.save_meta(job_id)
                emit({"type": "error", "message": file_llm_error})
                return
            done_files.add(fname)
            emit({"type": "file_end", "file_index": fi, "file": fname, "file_facts": file_facts, "warn": file_llm_error})
            persist_progress("running")
            if paused_requested():
                persist_progress("keep"); cleanup(); jsonl_fh.close(); return

        # ---- finished all work ----
        cleanup()
        jsonl_fh.close()
        persist_progress("done")
        meta["finished"] = time.time()
        J.save_meta(job_id)
        emit({"type": "done", "unique_facts": total_facts - total_dupes,
              "total_facts": total_facts, "duplicate_facts": total_dupes,
              "num_files": meta.get("num_files")})
    except Exception as e:
        import traceback; traceback.print_exc()
        try: jsonl_fh.close()
        except Exception: pass
        meta["error"] = str(e)[:300]; persist_progress("failed")
        meta["finished"] = time.time(); J.save_meta(job_id)
        emit({"type": "error", "message": str(e)})


async def _worker():
    while True:
        async with J._cond:
            job_id, backfill = J._select_next()
            if job_id is None:
                await J._cond.wait()
                continue
            J._jobs[job_id]["status"] = "running"
            J._jobs[job_id]["auto"] = backfill
            J._current["job_id"] = job_id
            J._pause_flags.pop(job_id, None)
            J.save_meta(job_id)
        # run outside the lock
        try:
            await run_job(job_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            J._jobs.get(job_id, {})["status"] = "failed"
            J.save_meta(job_id)
        async with J._cond:
            J._current["job_id"] = None
            m = J._jobs.get(job_id, {})
            # If run_job returned while a pause/preempt was pending (or status not terminal),
            # finalize: user pause -> sticky 'held', system preempt -> backfillable 'paused'.
            pause_pending = J._pause_flags.pop(job_id, False)
            if m.get("status") not in ("done", "failed"):
                if m.get("status") == "pausing" or pause_pending:
                    m["status"] = "held" if J._user_held.get(job_id) else "paused"
                    J.save_meta(job_id)
            J._user_held.pop(job_id, None)
            # purge a job deleted while running
            if job_id in J._jobs_pending_delete:
                J._jobs.pop(job_id, None)
                J._jobs_pending_delete.discard(job_id)
                shutil.rmtree(J.job_dir(job_id), ignore_errors=True)
            J._cond.notify_all()


@app.post("/api/jobs")
async def api_create_job(request: Request):
    body = await request.json()
    meta = {
        "source_kind": "url" if body.get("url") else "text",
        "source_name": (body.get("url") or "pasted text")[:200],
        "title": (body.get("url") or "pasted text")[:80],
        "k": int(body.get("k", 1)),
        "prompt": body.get("prompt", DEFAULT_PROMPT),
        "dedup_threshold": float(body.get("dedup_threshold", 0.90)),
        "dedup_field": body.get("dedup_field", "triple"),
        "dedup_model": body.get("dedup_model", ""),
    }
    src = json.dumps({"url": body.get("url", ""), "text": body.get("text", "")}).encode()
    job_id = await J.create_job(meta, src, "input.json")
    return {"job_id": job_id}


@app.post("/api/jobs-text")
async def api_create_job_text(
    file: UploadFile = File(...),
    k: int = Form(1),
    prompt: str = Form(DEFAULT_PROMPT),
    dedup_threshold: float = Form(0.90),
    dedup_field: str = Form("triple"),
    dedup_model: str = Form(""),
):
    # Large pasted text arrives as a plain-text file upload (multipart), avoiding
    # JSON body size/encoding limits. Stored exactly like the JSON text path.
    raw = await file.read()
    text = raw.decode("utf-8", "ignore")
    meta = {
        "source_kind": "text",
        "source_name": "pasted text",
        "title": "pasted text",
        "k": int(k),
        "prompt": prompt,
        "dedup_threshold": float(dedup_threshold),
        "dedup_field": dedup_field,
        "dedup_model": dedup_model,
    }
    src = json.dumps({"url": "", "text": text}).encode()
    job_id = await J.create_job(meta, src, "input.json")
    return {"job_id": job_id}


@app.post("/api/jobs-zip")
async def api_create_job_zip(
    file: UploadFile = File(...),
    k: int = Form(1),
    prompt: str = Form(DEFAULT_PROMPT),
    dedup_threshold: float = Form(0.90),
    dedup_field: str = Form("triple"),
    dedup_model: str = Form(""),
):
    raw = await file.read()
    meta = {
        "source_kind": "zip",
        "source_name": file.filename,
        "title": file.filename,
        "k": int(k),
        "prompt": prompt,
        "dedup_threshold": float(dedup_threshold),
        "dedup_field": dedup_field,
        "dedup_model": dedup_model,
    }
    job_id = await J.create_job(meta, raw, file.filename)
    return {"job_id": job_id}


@app.get("/api/jobs")
async def api_list_jobs():
    return {"jobs": J.list_jobs(), "current": J._current["job_id"]}


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    m = J.get_job(job_id)
    if not m:
        return {"error": "no such job"}
    return m


@app.get("/api/jobs/{job_id}/jsonl")
async def api_job_jsonl(job_id: str):
    return {"jsonl": J.read_jsonl(job_id), "meta": J.get_job(job_id)}


@app.post("/api/jobs/{job_id}/pause")
async def api_pause_job(job_id: str):
    J._user_held[job_id] = True
    return await J.pause_job(job_id)


@app.post("/api/jobs/{job_id}/resume")
async def api_resume_job(job_id: str):
    return await J.resume_job(job_id)


@app.delete("/api/jobs/{job_id}")
async def api_delete_job(job_id: str):
    # Guard: never delete a job that produced extracted data (a good result).
    m = J.get_job(job_id)
    if m and (m.get("unique_facts") or 0) > 0:
        return {"error": "job has extracted data and cannot be deleted"}
    return await J.delete_job(job_id)


@app.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str):
    """Subscribe to live events for a job. Replays prior facts first so a late
    viewer rebuilds the full graph, then streams new events."""
    async def gen():
        q = J.subscribe(job_id)
        # initial snapshot: replay existing jsonl + status
        prior = J.read_jsonl(job_id)
        if prior.strip():
            facts = [json.loads(l) for l in prior.splitlines() if l.strip()]
            yield f"data: {json.dumps({'type': 'replay', 'facts': facts})}\n\n"
        m = J.get_job(job_id)
        if m:
            yield f"data: {json.dumps({'type': 'job_status', 'meta': m})}\n\n"
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                    if ev.get("type") in ("done", "error"):
                        # keep open briefly so client gets it, then end
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            J.unsubscribe(job_id, q)
    return StreamingResponse(gen(), media_type="text/event-stream")


DEFAULT_ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jina-corpus.zip")
DEFAULT_ZIP_NAME = "jina-corpus.zip"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

@app.get("/assets/{name}")
async def serve_asset(name: str):
    safe = os.path.basename(name)
    path = os.path.join(ASSETS_DIR, safe)
    if not os.path.isfile(path):
        return HTMLResponse("not found", status_code=404)
    return FileResponse(path, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/favicon.ico")
async def favicon_ico():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico")
    if os.path.isfile(p):
        return FileResponse(p)
    return HTMLResponse("not found", status_code=404)

@app.get("/api/default-prompt")
async def get_default_prompt():
    return {"prompt": DEFAULT_PROMPT}

@app.get("/api/default-zip-info")
async def default_zip_info():
    if not os.path.exists(DEFAULT_ZIP_PATH):
        return {"available": False}
    try:
        with zipfile.ZipFile(DEFAULT_ZIP_PATH) as zf:
            n = sum(1 for i in zf.infolist() if not i.is_dir())
    except Exception:
        n = 0
    return {"available": True, "name": DEFAULT_ZIP_NAME, "files": n,
            "size_mb": round(os.path.getsize(DEFAULT_ZIP_PATH) / 1024 / 1024, 1)}

@app.get("/api/default-zip")
async def default_zip():
    if not os.path.exists(DEFAULT_ZIP_PATH):
        return {"available": False}
    return FileResponse(DEFAULT_ZIP_PATH, media_type="application/zip", filename=DEFAULT_ZIP_NAME)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.on_event("startup")
async def startup():
    get_dedup_model()
    J.reconcile_on_startup()
    asyncio.create_task(_worker())



HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Graph Extractor &mdash; Qwen3.6-35B-A3B-MTP on a single L4</title>
<meta name="description" content="Turn any document or a whole zip into an interactive knowledge graph, using a self-hosted Qwen3.6-35B-A3B-MTP LLM on a single NVIDIA L4. Entity-linked triples, semantic dedup, live force-directed graph, JSONL export.">
<meta name="keywords" content="knowledge graph, LLM, information extraction, entity extraction, triples, Qwen3, llama.cpp, embeddings, jina, force graph, NVIDIA L4, self-hosted">
<meta name="author" content="Han Xiao">
<link rel="canonical" href="https://hanxiao.io/knowledge-graph">
<meta name="theme-color" content="#1a1a1a">
<!-- favicons -->
<link rel="icon" type="image/png" sizes="32x32" href="https://hanxiao.io/knowledge-graph/assets/favicon-32.png">
<link rel="icon" type="image/png" sizes="16x16" href="https://hanxiao.io/knowledge-graph/assets/favicon-16.png">
<link rel="apple-touch-icon" sizes="180x180" href="https://hanxiao.io/knowledge-graph/assets/favicon-180.png">
<link rel="icon" href="https://hanxiao.io/knowledge-graph/assets/favicon.png">
<!-- Open Graph -->
<meta property="og:type" content="website">
<meta property="og:site_name" content="hanxiao.io">
<meta property="og:title" content="Knowledge Graph Extractor">
<meta property="og:description" content="Turn any document or a whole zip into an interactive knowledge graph, on a single NVIDIA L4. Entity-linked triples, semantic dedup, live force-directed graph, JSONL export.">
<meta property="og:url" content="https://hanxiao.io/knowledge-graph">
<meta property="og:image" content="https://hanxiao.io/knowledge-graph/assets/og-banner.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Knowledge Graph Extractor &mdash; interactive entity graph extracted from a document">
<!-- Twitter -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@hxiao">
<meta name="twitter:creator" content="@hxiao">
<meta name="twitter:title" content="Knowledge Graph Extractor">
<meta name="twitter:description" content="Any document &rarr; an interactive knowledge graph, on a single NVIDIA L4. Qwen3.6-35B-A3B-MTP.">
<meta name="twitter:image" content="https://hanxiao.io/knowledge-graph/assets/og-banner.png">
<script src="https://unpkg.com/force-graph@1.43.5/dist/force-graph.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#fff;--bg2:#fbfbfa;--bg3:#f0f0ee;
  --border:#1a1a1a;--border-soft:#dcdcd8;
  --text:#1a1a1a;--text2:#555;--text3:#999;
  --black:#1a1a1a;
  --mono:'SF Mono','JetBrains Mono','Fira Code','DejaVu Sans Mono','Consolas',monospace;
  --sans:var(--mono);
}
body{font-family:var(--mono);background:var(--bg);color:var(--text);height:100vh;overflow:hidden;font-size:13px;line-height:1.5;-webkit-font-smoothing:antialiased}

nav{border-bottom:1px solid var(--border);padding:0 18px;display:flex;align-items:center;gap:10px;height:42px}
nav .logo{font-weight:700;font-size:14px;font-family:var(--mono);color:var(--black);letter-spacing:-.3px}
nav .sep{color:var(--text3)}
nav .tag{font-size:11px;color:var(--text3);font-family:var(--mono)}
nav .gh{margin-left:auto;color:var(--text2);display:flex;align-items:center;transition:color .12s}
nav .gh:hover{color:var(--text)}

.layout{display:flex;height:calc(100vh - 42px);position:relative}
.sidebar{width:330px;min-width:330px;border-right:1px solid var(--border);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:14px;background:var(--bg2);transition:margin-left .2s ease}
.sidebar.collapsed{margin-left:-331px}
.main{flex:1;position:relative;background:var(--bg);overflow:hidden;min-width:0}
/* collapse handle, sits on the sidebar's right edge */
.sidebar-toggle{position:absolute;top:8px;left:302px;z-index:40;width:22px;height:30px;border:1px solid var(--border);background:var(--bg);color:var(--text2);cursor:pointer;font-family:var(--mono);font-size:13px;line-height:1;display:flex;align-items:center;justify-content:center;padding:0;transition:left .2s ease}
.sidebar-toggle:hover{background:var(--bg3);color:var(--text)}
/* collapsed: handle hugs the far-left edge */
.layout.collapsed .sidebar-toggle{left:0;border-left:none}

.section-title{font-size:10px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.newjob-box{position:relative;border:1px solid var(--border);padding:14px 12px 12px;display:flex;flex-direction:column;gap:14px;margin-top:4px}
.newjob-title{position:absolute;top:-8px;left:10px;background:var(--bg2);padding:0 6px;font-size:10px;font-weight:700;color:var(--text);text-transform:uppercase;letter-spacing:1px}
.prompt-toggle{cursor:pointer;user-select:none}
.prompt-toggle:hover{color:var(--text)}
.prompt-toggle span{display:inline-block;width:10px}
input[type="text"],textarea{width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:7px 9px;border-radius:0;font-size:12px;font-family:var(--mono)}
input:focus,textarea:focus{outline:none;box-shadow:inset 0 0 0 1px var(--border)}
textarea{font-family:var(--mono);font-size:11px;line-height:1.5;resize:vertical}
input[type="number"]{width:52px;text-align:center;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:5px 4px;border-radius:0;font-size:12px;font-family:var(--mono)}
.hidden{display:none}

.tabs{display:flex;gap:0;margin-bottom:8px;border:1px solid var(--border)}
.tab{flex:1;text-align:center;padding:5px 0;cursor:pointer;color:var(--text3);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;border-right:1px solid var(--border)}
.tab:last-child{border-right:none}
.tab:hover{color:var(--text2);background:var(--bg3)}
.tab.active{color:var(--bg);background:var(--black)}

.btn{background:var(--black);color:#fff;border:1px solid var(--black);padding:9px 0;border-radius:0;font-size:12px;cursor:pointer;font-weight:600;text-transform:uppercase;letter-spacing:.5px;width:100%;font-family:var(--mono)}
.btn:hover{background:#000}
.btn:disabled{background:var(--bg3);color:var(--text3);border-color:var(--border-soft);cursor:not-allowed}
.btn-sm{background:var(--bg);border:1px solid var(--border);color:var(--text2);padding:3px 9px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;border-radius:0;cursor:pointer;font-family:var(--mono)}
.btn-sm:hover{background:var(--black);color:#fff}
.btn-dl{background:var(--bg);color:var(--text);border:1px solid var(--border);padding:8px 0;border-radius:0;font-size:11px;cursor:pointer;font-weight:600;text-transform:uppercase;letter-spacing:.5px;width:100%;font-family:var(--mono)}
.btn-dl:hover:not(:disabled){background:var(--black);color:#fff}
.btn-dl:disabled{background:var(--bg3);color:var(--text3);border-color:var(--border-soft);cursor:not-allowed}

.param-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.param-row label{font-size:11px;color:var(--text2);min-width:62px}
.param-hint{font-size:10px;color:var(--text3);font-family:var(--mono)}
.threshold-row{display:flex;align-items:center;gap:8px}
.threshold-val{font-size:11px;font-family:var(--mono);color:var(--text);font-weight:600;min-width:30px}
input[type="range"]{-webkit-appearance:none;width:100%;height:2px;background:var(--border);border-radius:0;outline:none;border:none}
input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:0;background:var(--black);cursor:pointer;border:none}
input[type="range"]:disabled{background:var(--border-soft)}
input[type="range"]:disabled::-webkit-slider-thumb{background:var(--text3)}
.dedup-select{flex:1;min-width:0;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:5px 7px;border-radius:0;font-size:11px;font-family:var(--mono)}
.dedup-select:disabled{color:var(--text3);border-color:var(--border-soft);background:var(--bg3)}

.dropzone{border:1px dashed var(--border);border-radius:0;padding:18px 12px;text-align:center;cursor:pointer;color:var(--text3);font-size:11px;transition:all .12s;background:var(--bg)}
.dropzone:hover,.dropzone.drag{border-style:solid;color:var(--text);background:var(--bg3)}
.dropzone b{color:var(--text);font-weight:700}

/* JOBS LIST */
.jobs-list{display:flex;flex-direction:column;gap:0;max-height:200px;overflow-y:auto;border:1px solid var(--border);background:var(--bg)}
.jb{display:flex;align-items:center;gap:6px;padding:5px 7px;font-size:11px;font-family:var(--mono);border-bottom:1px solid var(--border-soft);cursor:pointer}
.jb:last-child{border-bottom:none}
.jb:hover{background:var(--bg3)}
.jb.active{background:var(--black);color:#fff}
.jb.active .jb-meta,.jb.active .jb-st{color:#fff}
.jb-st{width:12px;flex-shrink:0;text-align:center;color:var(--text2)}
.jb-st.running{color:var(--text);animation:jbpulse 1.1s ease-in-out infinite}
.jb.active .jb-st.running{color:#fff}
@keyframes jbpulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.78)}}
.jb-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.jb-meta{color:var(--text3);font-size:10px;flex-shrink:0}
.jb-btns{display:flex;gap:2px;flex-shrink:0}
.jb-x{background:none;border:none;color:inherit;opacity:.55;cursor:pointer;font-size:11px;padding:0 2px;line-height:1}
.jb-x:hover{opacity:1}

/* FILE LIST */
.filelist{display:flex;flex-direction:column;gap:0;max-height:200px;overflow-y:auto;border:1px solid var(--border);border-radius:0;background:var(--bg)}
.fl-item{display:flex;align-items:center;gap:7px;padding:4px 7px;font-size:11px;font-family:var(--mono);border-bottom:1px solid var(--border-soft)}
.fl-item:last-child{border-bottom:none}
.fl-item.active{background:var(--bg3)}
.fl-stat{width:14px;height:14px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:11px}
.fl-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text2)}
.fl-count{color:var(--text3);font-size:10px}
.fl-spin{display:inline-block;width:9px;height:9px;border:1.5px solid var(--border-soft);border-top-color:var(--black);border-radius:50%;animation:spin .7s linear infinite}
.fl-check{color:var(--black);font-weight:700}
.fl-wait{color:var(--text3)}
.fl-skip{color:var(--text3)}
@keyframes spin{to{transform:rotate(360deg)}}

.stats-mini{display:flex;gap:0;background:var(--bg);border:1px solid var(--border);border-radius:0;overflow:hidden}
.sm{flex:1;padding:7px 4px;text-align:center;border-right:1px solid var(--border-soft)}
.sm:last-child{border-right:none}
.smv{font-size:15px;font-weight:700;font-family:var(--mono)}
.sml{font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:1px}

.status-msg{padding:7px 10px;background:var(--bg);border:1px solid var(--border);border-radius:0;color:var(--text2);font-size:11px}
.status-msg.err{border-color:var(--border);color:var(--text);background:var(--bg3)}
.status-msg.ok{border-color:var(--border);color:var(--text);background:var(--bg3)}
.spinner{display:inline-block;width:9px;height:9px;border:1.5px solid var(--border-soft);border-top-color:var(--black);border-radius:50%;animation:spin .7s linear infinite;margin-right:6px;vertical-align:middle}

/* GRAPH */
#graph{width:100%;height:100%}
.graph-empty{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--text3);font-size:12px;text-align:center;text-transform:uppercase;letter-spacing:1px}
.graph-overlay{position:absolute;top:12px;left:12px;display:flex;gap:6px;align-items:center;z-index:10;flex-wrap:wrap;max-width:calc(100% - 24px)}
.path-bar{position:absolute;top:44px;left:12px;right:12px;z-index:9;background:var(--bg);border:1px solid var(--border);padding:7px 10px;font-family:var(--mono);font-size:11px;color:var(--text);line-height:1.7;max-height:90px;overflow-y:auto;box-shadow:3px 3px 0 rgba(26,26,26,.10)}
.path-bar .pn{font-weight:700}
.path-bar .pe{color:var(--text2)}
.path-bar .parrow{color:var(--text3);margin:0 2px}
.path-bar .ptitle{display:block;font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}
.graph-badge{background:var(--bg);border:1px solid var(--border);border-radius:0;padding:4px 9px;font-size:10px;font-family:var(--mono);color:var(--text);text-transform:uppercase;letter-spacing:.5px}
.graph-badge.on{background:var(--black);color:var(--bg);border-color:var(--black)}

/* HOVER CARD */
.hovercard{position:absolute;z-index:50;background:var(--bg);border:1px solid var(--border);border-radius:0;padding:11px;width:320px;box-shadow:4px 4px 0 rgba(26,26,26,.12);pointer-events:none;font-size:12px}
.hc-title{font-size:12px;font-weight:700;color:var(--text);margin-bottom:5px;line-height:1.4}
.hc-triple{display:flex;align-items:center;gap:4px;font-family:var(--mono);font-size:10px;color:var(--text3);margin-bottom:6px;flex-wrap:wrap}
.hc-triple .s{color:var(--text);font-weight:700}.hc-triple .p{color:var(--text2);font-weight:600}.hc-triple .o{color:var(--text);font-weight:700}
.hc-desc{font-size:11px;color:var(--text2);margin-bottom:6px;line-height:1.5}
.hc-meta{display:flex;justify-content:space-between;align-items:center;font-size:10px;color:var(--text3);font-family:var(--mono);margin-bottom:5px}
.hc-tags{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:5px}
.hc-tag{background:var(--bg3);color:var(--text2);padding:1px 6px;border-radius:0;font-size:9px;border:1px solid var(--border-soft)}
.hc-src{font-size:9px;color:var(--text3);font-family:var(--mono);border-top:1px solid var(--border-soft);padding-top:5px;margin-top:4px;word-break:break-all}
.hc-ev{margin-top:5px;padding:5px 8px;background:var(--bg2);border-left:2px solid var(--border);font-size:10px;color:var(--text2);font-style:italic;line-height:1.4}

.busy-banner{position:fixed;top:42px;left:0;right:0;background:var(--bg3);border-bottom:1px solid var(--border);color:var(--text);font-size:11px;padding:5px 18px;z-index:99;display:flex;align-items:center;gap:8px}
.busy-dot{width:7px;height:7px;border-radius:0;background:var(--black);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Mobile: sidebar overlays the graph instead of squishing it */
@media (max-width:760px){
  .sidebar{position:absolute;top:0;bottom:0;left:0;z-index:30;width:86vw;min-width:0;max-width:340px;box-shadow:4px 0 16px rgba(0,0,0,.18)}
  .sidebar.collapsed{margin-left:calc(-86vw - 2px)}
  .sidebar-toggle{left:calc(min(86vw, 340px) - 24px)}
  .layout.collapsed .sidebar-toggle{left:0}
}
</style>
</head>
<body>

<nav>
  <div class="logo">KG</div>
  <span class="sep">|</span>
  <span class="tag">qwen3.6-35b-a3b-mtp / nvidia l4 24gb</span>
  <a class="gh" href="https://github.com/hanxiao/knowledge-graph-extractor" target="_blank" rel="noopener" title="Source on GitHub" aria-label="GitHub">
    <svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
  </a>
</nav>

<div class="layout">
  <button class="sidebar-toggle" id="sidebar-toggle" onclick="toggleSidebar()" title="Toggle panel" aria-label="Toggle panel">‹</button>
  <div class="sidebar" id="sidebar">
    <div class="newjob-box">
    <div class="newjob-title">New Job</div>
    <div class="section">
      <div class="section-title">Source</div>
      <div class="tabs">
        <div class="tab active" data-tab="url">URL</div>
        <div class="tab" data-tab="text">Paste</div>
        <div class="tab" data-tab="zip">Zip</div>
      </div>
      <div id="p-url"><input type="text" id="url" placeholder="https://..." value="https://jina.ai/news"></div>
      <div id="p-text" class="hidden"><textarea id="text-paste" rows="5" placeholder="Paste markdown or plain text..."></textarea></div>
      <div id="p-zip" class="hidden">
        <div class="dropzone" id="dropzone">
          <div><b>Click to choose</b> or drop a .zip</div>
          <div id="zip-name" style="margin-top:4px;font-family:var(--mono);font-size:10px"></div>
        </div>
        <input type="file" id="zip-file" accept=".zip" class="hidden">
      </div>
    </div>

    <div class="section">
      <div class="section-title prompt-toggle" id="prompt-toggle" onclick="togglePrompt()"><span id="prompt-caret">▸</span> Extraction Prompt</div>
      <div id="prompt-body" class="hidden">
        <textarea id="prompt-edit" rows="5"></textarea>
        <div style="margin-top:4px;text-align:right"><button class="btn-sm" onclick="resetPrompt()">Reset</button></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Parameters</div>
      <div class="param-row">
        <label>Rounds</label>
        <input type="number" id="k-input" value="1" min="1" max="10">
        <span class="param-hint">per file</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Deduplication</div>
      <div class="param-row" style="margin:0;margin-bottom:6px">
        <label>Model</label>
        <select id="dedup-model" class="dedup-select">
          <option value="v5-nano" selected>v5-nano</option>
          <option value="">Off</option>
        </select>
      </div>
      <div class="param-row" style="margin:0;margin-bottom:6px">
        <label>Field</label>
        <select id="dedup-field" class="dedup-select">
          <option value="triple" selected>Triple (S->P->O)</option>
          <option value="title">Title</option>
          <option value="description">Description</option>
          <option value="title+desc">Title + Description</option>
          <option value="triple+title">Triple + Title</option>
          <option value="all">All fields</option>
        </select>
      </div>
      <div class="param-row" style="margin:0">
        <label>Threshold</label>
        <div class="threshold-row" style="flex:1">
          <input type="range" id="dedup-slider" min="0.5" max="0.99" step="0.01" value="0.90" oninput="document.getElementById('dedup-val').textContent=this.value">
          <span class="threshold-val" id="dedup-val">0.90</span>
        </div>
      </div>
    </div>

    <button class="btn" id="extract-btn" onclick="extract()">Extract</button>
    </div>

    <div class="section hidden" id="jobs-section">
      <div class="section-title">Jobs</div>
      <div class="jobs-list" id="jobs-list"></div>
    </div>

    <div class="section hidden" id="files-section">
      <div class="section-title" style="display:flex;justify-content:space-between"><span>Files</span><span id="files-progress" style="font-family:var(--mono);color:var(--text2)"></span></div>
      <div class="filelist" id="filelist"></div>
    </div>

    <div class="section hidden" id="results-section">
      <div class="stats-mini" style="margin-bottom:8px">
        <div class="sm"><div class="smv" id="s-edges">0</div><div class="sml">Edges</div></div>
        <div class="sm"><div class="smv" id="s-nodes">0</div><div class="sml">Nodes</div></div>
        <div class="sm"><div class="smv" id="s-tps">0</div><div class="sml">tok/s</div></div>
      </div>
      <button class="btn-dl" id="dl-btn" onclick="downloadJsonl()" disabled>Download JSONL</button>
    </div>

    <div id="status-area"></div>
  </div>

  <div class="main">
    <div id="graph"></div>
    <div class="graph-empty" id="graph-empty">Extract facts to build the relation graph</div>
    <div class="graph-overlay hidden" id="graph-overlay">
      <span class="graph-badge" id="graph-stats">0 edges</span>
      <span class="graph-badge" style="cursor:pointer" onclick="zoomFit()">fit</span>
      <span class="graph-badge on" style="cursor:pointer" id="node-labels-toggle" onclick="toggleNodeLabels()">node labels</span>
      <span class="graph-badge" style="cursor:pointer" id="edge-labels-toggle" onclick="toggleEdgeLabels()">edge labels</span>
      <span class="graph-badge" style="cursor:pointer" id="path-toggle" onclick="toggleLongestPath()">longest path</span>
    </div>
    <div class="path-bar hidden" id="path-bar"></div>
    <div class="hovercard hidden" id="hovercard"></div>
  </div>
</div>

<script>
// API base: works at the L4 root ("/") and behind a reverse proxy subpath
// (e.g. hanxiao.io/knowledge-graph). Derived from where index.html was served.
const API_BASE=(location.pathname.replace(/\/+$/,'')||'');
let currentTab='url', defaultPrompt='', zipFileObj=null;
let jsonlLines=[];           // raw jsonl strings for download
let graphNodes=new Map();    // id -> node
let graphLinks=[];           // link objects, each carries .fact
let Graph=null;
let showNodeLabels=true;     // toggle node text labels
let showEdgeLabels=false;    // toggle edge (predicate) text labels
let _pathNodes=new Set();    // node ids on the highlighted longest path
let _pathLinks=new Set();    // link objects on the highlighted longest path
let pathOn=false;

fetch(API_BASE+'/api/default-prompt').then(r=>r.json()).then(d=>{
  defaultPrompt=d.prompt;
  document.getElementById('prompt-edit').value=d.prompt;
});

// Default URL: randomly rotate among the latest 10 jina.ai/news posts (hardcoded).
const NEWS_POSTS=[
  'https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video',
  'https://jina.ai/news/jina-embeddings-v5-text-distilling-4b-quality-into-sub-1b-multilingual-embeddings',
  'https://jina.ai/news/jina-vlm-small-multilingual-vision-language-model',
  'https://jina.ai/news/bootstrapping-audio-embeddings-from-multimodal-llms',
  'https://jina.ai/news/identifying-embedding-models-from-raw-numerical-values',
  'https://jina.ai/news/jina-reranker-v3-0-6b-listwise-reranker-for-sota-multilingual-retrieval',
  'https://jina.ai/news/multimodal-embeddings-in-llama-cpp-and-gguf',
  'https://jina.ai/news/jina-code-embeddings-sota-code-retrieval-at-0-5b-and-1-5b',
  'https://jina.ai/news/agentic-workflow-with-jina-remote-mcp-server',
  'https://jina.ai/news/optimizing-ggufs-for-decoder-only-embedding-models'
];
(function(){
  const el=document.getElementById('url');
  el.value=NEWS_POSTS[Math.floor(Math.random()*NEWS_POSTS.length)];
})();

// Preload the bundled default corpus zip so the Zip tab works out of the box.
let defaultZipLoaded=false;
fetch(API_BASE+'/api/default-zip-info').then(r=>r.json()).then(d=>{
  if(!d.available)return;
  const zn=document.getElementById('zip-name');
  zn.textContent='loading '+d.name+' ('+d.files+' files, '+d.size_mb+' MB)...';
  fetch(API_BASE+'/api/default-zip').then(r=>r.blob()).then(b=>{
    const f=new File([b],d.name,{type:'application/zip'});
    zipFileObj=f;defaultZipLoaded=true;
    zn.textContent=d.name+' · '+d.files+' files · '+d.size_mb+' MB (default)';
  }).catch(()=>{zn.textContent=''});
}).catch(()=>{});

document.getElementById('dedup-model').addEventListener('change',function(){
  const on=this.value!=='';
  document.getElementById('dedup-field').disabled=!on;
  document.getElementById('dedup-slider').disabled=!on;
});

function resetPrompt(){document.getElementById('prompt-edit').value=defaultPrompt}
function togglePrompt(){
  const b=document.getElementById('prompt-body');
  const open=b.classList.toggle('hidden')===false;
  document.getElementById('prompt-caret').textContent=open?'▾':'▸';
}

function toggleSidebar(){
  const sb=document.getElementById('sidebar');
  const collapsed=sb.classList.toggle('collapsed');
  document.querySelector('.layout').classList.toggle('collapsed',collapsed);
  document.getElementById('sidebar-toggle').textContent=collapsed?'›':'‹';
  // graph reflows to the freed space
  setTimeout(()=>{if(Graph){const m=document.querySelector('.main');Graph.width(m.clientWidth).height(m.clientHeight);zoomFit();}},220);
}
// Default-collapse on small screens so the graph isn't squeezed.
if(window.matchMedia('(max-width:760px)').matches){
  const sb=document.getElementById('sidebar');sb.classList.add('collapsed');
  document.querySelector('.layout').classList.add('collapsed');
  document.getElementById('sidebar-toggle').textContent='›';
}

function switchTab(tab){
  currentTab=tab;
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===tab));
  document.getElementById('p-url').classList.toggle('hidden',tab!=='url');
  document.getElementById('p-text').classList.toggle('hidden',tab!=='text');
  document.getElementById('p-zip').classList.toggle('hidden',tab!=='zip');
}
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>switchTab(t.dataset.tab)));

// Zip picker + dnd
const dz=document.getElementById('dropzone'), zf=document.getElementById('zip-file');
dz.addEventListener('click',()=>zf.click());
zf.addEventListener('change',()=>{ if(zf.files[0]) setZip(zf.files[0]); });
['dragover','dragenter'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add('drag')}));
['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove('drag')}));
dz.addEventListener('drop',e=>{const f=e.dataTransfer.files[0]; if(f&&f.name.endsWith('.zip'))setZip(f)});
function setZip(f){zipFileObj=f;defaultZipLoaded=false;document.getElementById('zip-name').textContent=f.name+' ('+(f.size/1024/1024).toFixed(1)+' MB)'}

function esc(s){const d=document.createElement('div');d.textContent=s==null?'':s;return d.innerHTML}

// ---------- Graph ----------
function initGraph(){
  if(Graph)return;
  const el=document.getElementById('graph');
  Graph=ForceGraph()(el)
    .width(el.clientWidth).height(el.clientHeight)
    .backgroundColor('#ffffff')
    .nodeLabel(n=>n.label||n.id)
    .nodeVal(n=>Math.min(10,1.5+n.deg))
    .nodeColor(()=> '#1a1a1a')
    .nodeRelSize(4)
    .nodePointerAreaPaint((n,color,ctx)=>{
      const r=Math.min(10,1.5+n.deg);
      ctx.fillStyle=color;ctx.beginPath();ctx.arc(n.x,n.y,r,0,2*Math.PI);ctx.fill();
    })
    .linkColor(l=> _pathLinks.has(l) ? '#1a1a1a' : 'rgba(26,26,26,0.18)')
    .linkWidth(l=> _pathLinks.has(l) ? 2.5 : 1)
    .linkDirectionalArrowLength(3.5)
    .linkDirectionalArrowColor(l=> _pathLinks.has(l) ? '#1a1a1a' : 'rgba(26,26,26,0.45)')
    .linkDirectionalArrowRelPos(1)
    .linkHoverPrecision(8)
    .nodeCanvasObjectMode(()=> 'replace')
    .nodeCanvasObject((n,ctx,scale)=>{
      const r=Math.min(10,1.5+n.deg);
      const onPath=_pathNodes.has(n.id);
      // hollow node: white fill + black ring (filled black when on longest path)
      ctx.beginPath();ctx.arc(n.x,n.y,r,0,2*Math.PI);
      ctx.fillStyle=onPath?'#1a1a1a':'#ffffff';ctx.fill();
      ctx.lineWidth=(onPath?2:1.3)/scale;ctx.strokeStyle='#1a1a1a';ctx.stroke();
      const showThis = showNodeLabels && (scale>1.0 || onPath);
      if(showThis){
        const disp=n.label||n.id;
        const label=disp.length>30?disp.slice(0,29)+'…':disp;
        ctx.font=`${onPath?'bold ':''}${Math.max(2.5,10/scale)}px 'SF Mono',monospace`;
        ctx.fillStyle='#1a1a1a';ctx.textAlign='center';ctx.textBaseline='top';
        ctx.fillText(label,n.x,n.y+r+1.5/scale);
      }
    })
    .linkCanvasObjectMode(()=> showEdgeLabels ? 'after' : undefined)
    .linkCanvasObject((l,ctx,scale)=>{
      if(!showEdgeLabels)return;
      const onPath=_pathLinks.has(l);
      if(scale<=1.2 && !onPath)return;
      const s=(typeof l.source==='object')?l.source:null;
      const t=(typeof l.target==='object')?l.target:null;
      if(!s||!t)return;
      const f=l.fact||{};const pred=f.predicate||'';
      if(!pred)return;
      const mx=(s.x+t.x)/2, my=(s.y+t.y)/2;
      const disp=pred.length>24?pred.slice(0,23)+'…':pred;
      ctx.font=`${onPath?'bold ':''}${Math.max(2,8/scale)}px 'SF Mono',monospace`;
      ctx.textAlign='center';ctx.textBaseline='middle';
      const w=ctx.measureText(disp).width, pad=2/scale;
      ctx.fillStyle='rgba(255,255,255,0.82)';
      ctx.fillRect(mx-w/2-pad,my-(5/scale),w+pad*2,10/scale);
      ctx.fillStyle=onPath?'#1a1a1a':'rgba(26,26,26,0.65)';
      ctx.fillText(disp,mx,my);
    })
    .cooldownTicks(90)
    .onEngineStop(()=>zoomFit())
    .onLinkHover(link=>{ showHover(link); })
    .onNodeHover(n=>{ if(!n) hideHover(); });
  // Tame the physics so many small/disconnected clusters stay compact.
  // Default charge has no distance cap -> separate components fly apart ->
  // zoomToFit then shrinks the whole graph to fit them. Cap charge range and
  // add a mild centering pull so isolated edges don't drift to infinity.
  try{ Graph.d3Force('charge').strength(-20).distanceMax(60); }catch(e){}
  try{ Graph.d3Force('link').distance(24).strength(1); }catch(e){}
  const centerForce=(function(){
    let nodes=[];
    function f(alpha){ const k=alpha*0.08; for(const n of nodes){ n.vx-=n.x*k; n.vy-=n.y*k; } }
    f.initialize=n=>{ nodes=n; };
    return f;
  })();
  try{ Graph.d3Force('center2', centerForce); }catch(e){}
  // hide hover when leaving canvas
  el.addEventListener('mouseleave',hideHover);
  el.addEventListener('mousemove',e=>{ window._mx=e.clientX; window._my=e.clientY; positionHover(); });
}

let _fitPending=null;
function refreshGraph(){
  const nodes=[...graphNodes.values()];
  Graph.graphData({nodes, links:graphLinks});
  document.getElementById('s-nodes').textContent=nodes.length;
  document.getElementById('s-edges').textContent=graphLinks.length;
  document.getElementById('graph-stats').textContent=graphLinks.length+' edges · '+nodes.length+' nodes';
  // keep whole graph framed during streaming (debounced)
  if(_fitPending)clearTimeout(_fitPending);
  _fitPending=setTimeout(()=>{ zoomFit(); if(pathOn){ const b=computeLongestPath(); _pathNodes=new Set(b.nodes); _pathLinks=new Set(b.links);} },250);
}

// Normalize entity names so case/whitespace/punctuation variants map to one node.
function normKey(s){
  return (s||'').toLowerCase()
    .normalize('NFKC')
    .replace(/[\s\u00a0]+/g,' ')      // collapse whitespace
    .replace(/^[\s"'`([{]+|[\s"'`)\]}.,;:!?]+$/g,'') // strip wrapping/trailing punct
    .trim();
}
function addFactEdge(fact){
  const sRaw=(fact.subject||'').trim()||'?';
  const oRaw=(fact.object||'').trim()||'?';
  const sKey=normKey(sRaw)||'?', oKey=normKey(oRaw)||'?';
  for(const [key,raw] of [[sKey,sRaw],[oKey,oRaw]]){
    if(!graphNodes.has(key)) graphNodes.set(key,{id:key,label:raw,deg:0});
    graphNodes.get(key).deg++;
  }
  graphLinks.push({source:sKey,target:oKey,fact});
}

function showHover(link){
  if(!link||!link.fact){ if(!window._pinHover)hideHover(); return; }
  const f=link.fact;
  const tags=(f.tags||[]).map(t=>'<span class="hc-tag">'+esc(t)+'</span>').join('');
  const card=document.getElementById('hovercard');
  card.innerHTML=
    '<div class="hc-title">'+esc(f.title||(f.subject+' '+f.predicate+' '+f.object))+'</div>'
    +'<div class="hc-triple"><span class="s">'+esc(f.subject)+'</span><span>→</span><span class="p">'+esc(f.predicate)+'</span><span>→</span><span class="o">'+esc(f.object)+'</span></div>'
    +(f.description?'<div class="hc-desc">'+esc(f.description)+'</div>':'')
    +'<div class="hc-meta"><span>conf '+(f.confidence!=null?f.confidence:'-')+'%</span>'+(f.is_duplicate?'<span>[duplicate]</span>':'')+'</div>'
    +(tags?'<div class="hc-tags">'+tags+'</div>':'')
    +(f.evidence_span?'<div class="hc-ev">'+esc(f.evidence_span)+'</div>':'')
    +(f.source_file?'<div class="hc-src">📄 '+esc(f.source_file)+'</div>':'');
  card.classList.remove('hidden');
  positionHover();
}
function positionHover(){
  const card=document.getElementById('hovercard');
  if(card.classList.contains('hidden'))return;
  const main=document.querySelector('.main').getBoundingClientRect();
  let x=(window._mx||0)-main.left+16, y=(window._my||0)-main.top+16;
  if(x+340>main.width)x=(window._mx||0)-main.left-336;
  if(y+card.offsetHeight>main.height)y=main.height-card.offsetHeight-10;
  card.style.left=Math.max(4,x)+'px';card.style.top=Math.max(4,y)+'px';
}
function hideHover(){document.getElementById('hovercard').classList.add('hidden')}
function zoomFit(){if(Graph && graphLinks.length)Graph.zoomToFit(350,50)}

function toggleNodeLabels(){
  showNodeLabels=!showNodeLabels;
  document.getElementById('node-labels-toggle').classList.toggle('on',showNodeLabels);
  if(Graph)Graph.nodeCanvasObject(Graph.nodeCanvasObject()); // force re-render
}
function toggleEdgeLabels(){
  showEdgeLabels=!showEdgeLabels;
  document.getElementById('edge-labels-toggle').classList.toggle('on',showEdgeLabels);
  if(Graph){
    Graph.linkCanvasObjectMode(()=> showEdgeLabels ? 'after' : undefined);
    Graph.nodeCanvasObject(Graph.nodeCanvasObject()); // force re-render
  }
}

// Longest simple path following edge DIRECTION (subject -> object only).
// Graph can be cyclic, so we run a bounded DFS from each node and keep the best.
function computeLongestPath(){
  const adj=new Map();   // nodeId -> [{to, link}] (directed: source -> target)
  for(const l of graphLinks){
    if(l.fact && l.fact.is_duplicate) continue;
    const s=(typeof l.source==='object')?l.source.id:l.source;
    const t=(typeof l.target==='object')?l.target.id:l.target;
    if(s===t) continue;
    if(!adj.has(s))adj.set(s,[]);
    if(!adj.has(t))adj.set(t,[]);
    adj.get(s).push({to:t,link:l}); // directed traversal only
  }
  const nodes=[...adj.keys()];
  if(!nodes.length)return {nodes:[],links:[]};
  let best={nodes:[],links:[]};
  const CAP=2000; let steps=0;
  function dfs(cur, visited, pathNodes, pathLinks){
    if(steps++>CAP*nodes.length) return;
    if(pathNodes.length>best.nodes.length) best={nodes:[...pathNodes],links:[...pathLinks]};
    for(const {to,link} of (adj.get(cur)||[])){
      if(visited.has(to))continue;
      visited.add(to);pathNodes.push(to);pathLinks.push(link);
      dfs(to,visited,pathNodes,pathLinks);
      visited.delete(to);pathNodes.pop();pathLinks.pop();
    }
  }
  // start DFS from higher-degree nodes first (more likely on long paths), cap fanout
  nodes.sort((a,b)=>(adj.get(b).length)-(adj.get(a).length));
  const starts=nodes.slice(0, Math.min(nodes.length, 40));
  for(const s of starts){
    dfs(s,new Set([s]),[s],[]);
  }
  return best;
}

function toggleLongestPath(){
  pathOn=!pathOn;
  const tEl=document.getElementById('path-toggle');
  const bar=document.getElementById('path-bar');
  if(!pathOn){
    _pathNodes=new Set();_pathLinks=new Set();
    tEl.classList.remove('on');bar.classList.add('hidden');
    if(Graph)Graph.nodeColor(Graph.nodeColor());
    return;
  }
  const best=computeLongestPath();
  if(!best.nodes.length){pathOn=false;bar.classList.remove('hidden');bar.innerHTML='<span class="ptitle">longest path</span>(no path)';return;}
  _pathNodes=new Set(best.nodes);
  _pathLinks=new Set(best.links);
  tEl.classList.add('on');
  // Render the chain at top: node -> [predicate] -> node ...
  const labelOf=id=>{const n=graphNodes.get(id);return n?(n.label||n.id):id;};
  let html='<span class="ptitle">longest path · '+best.nodes.length+' nodes</span>';
  for(let i=0;i<best.nodes.length;i++){
    html+='<span class="pn">'+esc(labelOf(best.nodes[i]))+'</span>';
    if(i<best.links.length){
      const f=best.links[i].fact||{};
      html+='<span class="parrow">→</span><span class="pe">'+esc(f.predicate||'')+'</span><span class="parrow">→</span>';
    }
  }
  bar.innerHTML=html;bar.classList.remove('hidden');
  if(Graph)Graph.nodeColor(Graph.nodeColor()); // re-render highlight
}

// ---------- File list ----------
function renderFileList(files){
  const fl=document.getElementById('filelist');
  fl.innerHTML='';
  files.forEach((name,i)=>{
    fl.insertAdjacentHTML('beforeend',
      '<div class="fl-item" id="fl-'+i+'"><span class="fl-stat"><span class="fl-wait">○</span></span>'
      +'<span class="fl-name" title="'+esc(name)+'">'+esc(name)+'</span><span class="fl-count" id="flc-'+i+'"></span></div>');
  });
  document.getElementById('files-section').classList.remove('hidden');
  updateFilesProgress(0,files.length);
}
function setFileState(i,state,count){
  const item=document.getElementById('fl-'+i);if(!item)return;
  const stat=item.querySelector('.fl-stat');
  document.querySelectorAll('.fl-item').forEach(x=>x.classList.remove('active'));
  if(state==='running'){stat.innerHTML='<span class="fl-spin"></span>';item.classList.add('active');}
  else if(state==='done'){stat.innerHTML='<span class="fl-check">✓</span>';}
  else if(state==='skip'){stat.innerHTML='<span class="fl-skip">—</span>';}
  if(count!=null)document.getElementById('flc-'+i).textContent=count;
}
function updateFilesProgress(done,total){document.getElementById('files-progress').textContent=done+'/'+total}

// ---------- Job-based extraction ----------
// Each Extract creates a server-side JOB. The single-slot scheduler runs one job
// at a time; a new job preempts the running one (which goes to the paused pool and
// auto-backfills later). We subscribe to /api/jobs/{id}/events for live updates,
// and can reload any past job's jsonl into the graph.
let viewingJob=null;       // job_id currently rendered in the graph
let eventAbort=null;       // AbortController for the current event stream

function resetGraphView(){
  jsonlLines=[];graphNodes=new Map();graphLinks=[];
  _pathNodes=new Set();_pathLinks=new Set();pathOn=false;
  document.getElementById('path-toggle').classList.remove('on');
  document.getElementById('path-bar').classList.add('hidden');
  document.getElementById('graph-empty').classList.add('hidden');
  document.getElementById('graph-overlay').classList.remove('hidden');
  document.getElementById('results-section').classList.remove('hidden');
  document.getElementById('dl-btn').disabled=true;
  document.getElementById('s-edges').textContent='0';
  document.getElementById('s-nodes').textContent='0';
  document.getElementById('s-tps').textContent='0';
  initGraph();refreshGraph();
}
function ingestFact(rec){
  jsonlLines.push(JSON.stringify(rec));
  if(!rec.is_duplicate){ addFactEdge(rec); }
}

async function extract(){
  const k=parseInt(document.getElementById('k-input').value)||1;
  const prompt=document.getElementById('prompt-edit').value;
  const threshold=parseFloat(document.getElementById('dedup-slider').value);
  const dedupField=document.getElementById('dedup-field').value;
  const dedupModel=document.getElementById('dedup-model').value;
  const btn=document.getElementById('extract-btn');
  btn.disabled=true;btn.textContent='Submitting...';
  try{
    let job_id;
    if(currentTab==='zip'){
      if(!zipFileObj){fail('Choose a .zip file first');return;}
      const fd=new FormData();
      fd.append('file',zipFileObj);
      fd.append('k',k);fd.append('prompt',prompt);
      fd.append('dedup_threshold',threshold);fd.append('dedup_field',dedupField);fd.append('dedup_model',dedupModel);
      const r=await fetch(API_BASE+'/api/jobs-zip',{method:'POST',body:fd});
      job_id=(await r.json()).job_id;
    }else if(currentTab==='text' && document.getElementById('text-paste').value.length>20000){
      // Large paste -> send as a plain-text file upload (multipart), avoiding
      // JSON body size/encoding limits. Full text is preserved + chunked server-side.
      const txt=document.getElementById('text-paste').value;
      const fd=new FormData();
      fd.append('file',new File([txt],'pasted.txt',{type:'text/plain'}));
      fd.append('k',k);fd.append('prompt',prompt);
      fd.append('dedup_threshold',threshold);fd.append('dedup_field',dedupField);fd.append('dedup_model',dedupModel);
      const r=await fetch(API_BASE+'/api/jobs-text',{method:'POST',body:fd});
      job_id=(await r.json()).job_id;
    }else{
      const payload={k,prompt,dedup_threshold:threshold,dedup_field:dedupField,dedup_model:dedupModel};
      if(currentTab==='url')payload.url=document.getElementById('url').value;
      else payload.text=document.getElementById('text-paste').value;
      const r=await fetch(API_BASE+'/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      job_id=(await r.json()).job_id;
    }
    document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>Job '+job_id.slice(0,6)+' queued</div>';
    await viewJob(job_id);
    refreshJobs();
  }catch(e){fail(e.message);}
  btn.disabled=false;btn.textContent='Extract';
}
function fail(msg){
  document.getElementById('status-area').innerHTML='<div class="status-msg err">'+esc(msg)+'</div>';
  document.getElementById('extract-btn').disabled=false;
  document.getElementById('extract-btn').textContent='Extract';
}

// Subscribe to a job's live event stream and render into the graph.
async function viewJob(job_id){
  if(eventAbort){ eventAbort.abort(); eventAbort=null; }
  viewingJob=job_id;
  resetGraphView();
  document.getElementById('files-section').classList.add('hidden');
  const ac=new AbortController(); eventAbort=ac;
  let resp;
  try{ resp=await fetch(API_BASE+'/api/jobs/'+job_id+'/events',{signal:ac.signal}); }
  catch(e){ return; }
  const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
  let fileNames=[];
  while(true){
    let chunk; try{ chunk=await reader.read(); }catch(e){ break; }
    if(chunk.done)break;
    buf+=dec.decode(chunk.value,{stream:true});
    const lines=buf.split('\n'); buf=lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: '))continue;
      let d;try{d=JSON.parse(line.slice(6));}catch(e){continue;}
      switch(d.type){
        case 'replay':
          // full rebuild from persisted facts
          jsonlLines=[];graphNodes=new Map();graphLinks=[];
          for(const rec of d.facts) ingestFact(rec);
          refreshGraph();setTimeout(zoomFit,200);
          document.getElementById('dl-btn').disabled=jsonlLines.length===0;
          if(autoPathPending){autoPathPending=false;if(graphLinks.length){setTimeout(()=>{if(!pathOn)toggleLongestPath();},300);}}
          break;
        case 'job_status':
          renderJobStatusBadge(d.meta);break;
        case 'status':
          document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>'+esc(d.message)+'</div>';break;
        case 'fetched':
          document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>'+esc(d.title||'')+' ('+(d.chars||0).toLocaleString()+' chars)</div>';break;
        case 'filelist':
          fileNames=d.files;renderFileList(d.files);
          (d.done_files||[]).forEach(n=>{const i=fileNames.indexOf(n);if(i>=0)setFileState(i,'done');});
          updateFilesProgress((d.done_files||[]).length,fileNames.length);break;
        case 'file_start':
          setFileState(d.file_index,'running');break;
        case 'file_end':{
          setFileState(d.file_index,d.skipped?'skip':'done',d.file_facts);
          const done=document.querySelectorAll('.fl-check,.fl-skip').length;
          updateFilesProgress(done,fileNames.length||done);break;}
        case 'metrics':
          document.getElementById('s-tps').textContent=d.tps;break;
        case 'fact':
          ingestFact(d.fact);refreshGraph();
          document.getElementById('s-tps').textContent=d.tps||document.getElementById('s-tps').textContent;
          document.getElementById('dl-btn').disabled=false;break;
        case 'done':
          document.getElementById('status-area').innerHTML='<div class="status-msg ok">'+(d.unique_facts||0)+' unique facts'+(d.num_files!=null?' · '+d.num_files+' files':'')+' · '+(d.duplicate_facts||0)+' dup</div>';
          setTimeout(zoomFit,200);refreshJobs();break;
        case 'warn':
          document.getElementById('status-area').innerHTML='<div class="status-msg err">'+esc(d.message)+'</div>';break;
        case 'error':
          document.getElementById('status-area').innerHTML='<div class="status-msg err">'+esc(d.message)+'</div>';refreshJobs();break;
      }
    }
  }
}

function renderJobStatusBadge(m){
  if(!m)return;
  const map={queued:'queued',running:'running',paused:'paused (backfill)',held:'paused',done:'done',failed:'failed',pausing:'pausing'};
  document.getElementById('status-area').innerHTML='<div class="status-msg">'+esc(m.title||'')+' — '+(map[m.status]||m.status)+'</div>';
  prefillFromJob(m);
}

// Prefill the New Job form with a selected job's settings (params + dedup +
// prompt + source). Lets the user clone/re-run a past job's config.
function prefillFromJob(m){
  if(!m)return;
  if(m.k!=null){document.getElementById('k-input').value=m.k;}
  if(m.dedup_model!=null){const el=document.getElementById('dedup-model');el.value=m.dedup_model;el.dispatchEvent(new Event('change'));}
  if(m.dedup_field){document.getElementById('dedup-field').value=m.dedup_field;}
  if(m.dedup_threshold!=null){const s=document.getElementById('dedup-slider');s.value=m.dedup_threshold;document.getElementById('dedup-val').textContent=(+m.dedup_threshold).toFixed(2);}
  if(m.prompt){document.getElementById('prompt-edit').value=m.prompt;}
  // restore source into the matching tab
  if(m.source_kind==='url' && m.source_name){switchTab('url');document.getElementById('url').value=m.source_name;}
  else if(m.source_kind==='text'){switchTab('text');}
  else if(m.source_kind==='zip'){switchTab('zip');}
}

// ---------- Jobs panel ----------
async function refreshJobs(){
  let data;try{ data=await (await fetch(API_BASE+'/api/jobs')).json(); }catch(e){ return; }
  const jobs=(data.jobs||[]).slice();
  // Running jobs always pinned to the top of the list (stable order otherwise).
  jobs.sort((a,b)=>((b.status==='running')-(a.status==='running')));
  const wrap=document.getElementById('jobs-list');
  if(!jobs.length){ document.getElementById('jobs-section').classList.add('hidden'); return; }
  document.getElementById('jobs-section').classList.remove('hidden');
  const stIcon={running:'▶',queued:'…',paused:'⏸',held:'⏸',pausing:'⏸',done:'✓',failed:'✕'};
  wrap.innerHTML=jobs.map(j=>{
    const active=j.job_id===viewingJob?' active':'';
    const prog=j.num_files?(' · '+(j.files_done||0)+'/'+j.num_files+' files'):'';
    const canPause=(j.status==='running'||j.status==='queued');
    const canResume=(j.status==='paused'||j.status==='held'||j.status==='failed');
    // Only allow deleting jobs with NO extracted data (empty/failed). A job that
    // produced edges is kept; users can't delete a good result from the UI.
    const canDelete=((j.unique_facts||0)===0);
    let btns='';
    if(canDelete)btns+='<button class="jb-x" onclick="event.stopPropagation();delJob(\''+j.job_id+'\')" title="delete">✕</button>';
    if(canPause)btns='<button class="jb-x" onclick="event.stopPropagation();pauseJob(\''+j.job_id+'\')" title="pause">⏸</button>'+btns;
    if(canResume)btns='<button class="jb-x" onclick="event.stopPropagation();resumeJob(\''+j.job_id+'\')" title="resume">▶</button>'+btns;
    const stCls='jb-st'+(j.status==='running'?' running':'');
    return '<div class="jb'+active+'" onclick="viewJob(\''+j.job_id+'\');highlightJob(\''+j.job_id+'\')">'
      +'<span class="'+stCls+'" title="'+j.status+'">'+(stIcon[j.status]||'·')+'</span>'
      +'<span class="jb-name" title="'+esc(j.title||'')+'">'+esc(j.title||j.job_id)+'</span>'
      +'<span class="jb-meta">'+ (j.unique_facts||0) +' edges'+prog+'</span>'
      +'<span class="jb-btns">'+btns+'</span></div>';
  }).join('');
}
function highlightJob(id){viewingJob=id;refreshJobs();}
async function pauseJob(id){await fetch(API_BASE+'/api/jobs/'+id+'/pause',{method:'POST'});refreshJobs();}
async function resumeJob(id){await fetch(API_BASE+'/api/jobs/'+id+'/resume',{method:'POST'});refreshJobs();if(id===viewingJob)viewJob(id);}
async function delJob(id){if(!confirm('Delete this job?'))return;await fetch(API_BASE+'/api/jobs/'+id,{method:'DELETE'});if(id===viewingJob){viewingJob=null;}refreshJobs();}

function downloadJsonl(){
  const blob=new Blob([jsonlLines.join('\n')+'\n'],{type:'application/x-ndjson'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  const ts=new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
  a.download='kg-facts-'+ts+'.jsonl';a.click();URL.revokeObjectURL(a.href);
}

document.getElementById('url').addEventListener('keydown',e=>{if(e.key==='Enter')extract()});
window.addEventListener('resize',()=>{if(Graph){const m=document.querySelector('.main');Graph.width(m.clientWidth).height(m.clientHeight);zoomFit();}});

let autoPathPending=false;  // set during first-load auto-select to flip longest path on

// On first visit (no job being viewed yet), auto-select the jina-corpus job and
// turn longest path ON so the landing page shows a populated, highlighted graph.
async function initDefaultView(){
  try{
    const data=await (await fetch(API_BASE+'/api/jobs')).json();
    const jobs=data.jobs||[];
    if(!jobs.length)return;
    // Prefer the COMPLETED jina-corpus job with the most edges (avoid a partial/
    // running re-run that happens to share the same title).
    const score=j=>((j.status==='done'?1e9:0)+(j.unique_facts||0));
    const corpus=jobs.filter(j=>/jina-corpus/i.test(j.title||''))
                     .sort((a,b)=>score(b)-score(a));
    const pick=corpus[0]
      || jobs.filter(j=>(j.unique_facts||0)>0).sort((a,b)=>score(b)-score(a))[0]
      || jobs[0];
    if(!pick)return;
    autoPathPending=true;
    await viewJob(pick.job_id);
    highlightJob(pick.job_id);
  }catch(e){}
}

refreshJobs();
setInterval(refreshJobs,3000);
initDefaultView();
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
