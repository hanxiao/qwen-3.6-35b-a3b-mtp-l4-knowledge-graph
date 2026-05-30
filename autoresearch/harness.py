#!/usr/bin/env python3
"""
Eval harness for the KI-extractor autoresearch loop.

Role analogy to karpathy/autoresearch:
  - this file == prepare.py : fixed measurement rig, NOT edited by experiments.
  - a "config" (llama-server flags + sampling) == train.py : the thing iterated.
  - metric == decode tok/s (ground truth from llama-server timings).
  - hard constraint == quality guard (no fact loss vs baseline).

A single experiment:
  1. (re)launch llama-server container with the config's flags
  2. run a fixed-seed 3-round extraction on the cached article
  3. read ground-truth timings from the server
  4. dedup + score quality, compare against baseline
  5. emit one result dict (appended to experiments.jsonl by run.py)

Speed is measured from llama-server's own `timings` object, not by counting
SSE chunks (which is what the repo UI does and is only approximate).
"""
import os, sys, json, time, subprocess, hashlib, re
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL_PATH = "/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf"
MODELS_DIR = os.path.join(REPO, "models")
TEMPLATES_DIR = os.path.join(REPO, "templates")
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
CONTAINER = "llama-bench"
PORT = 8080
DOC_CACHE = os.path.join(HERE, "doc_cache.md")
ARTICLE_URL = "https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/"

# Fixed seeds -> reproducible 3-round runs (the repo uses random.randint per round).
# Three distinct seeds keep cross-round diversity (coverage) while staying comparable.
SEEDS = [101, 202, 303]

# --- baseline llama-server flags, copied verbatim from docker-compose.yml ---
# Stored as the exact CLI token list that follows the image name.
BASELINE_SERVER_ARGS = [
    "--model", MODEL_PATH,
    "--host", "0.0.0.0", "--port", "8080",
    "--ctx-size", "16384",
    "--parallel", "1",
    "--flash-attn", "1",
    "--no-mmap",
    "--threads", "8",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "2",
    "--n-predict", "8192",
    "--jinja",
    "--chat-template-file", "/templates/chat_template.jinja",
    "-fitt", "512",
    "--cache-reuse", "256",
]

BASELINE_SAMPLING = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "max_tokens": 8192,
}

# ----------------------------------------------------------------------------
# Prompt + schema: extracted live from the repo app.py (via AST, no import) so
# we never drift from the real pipeline and avoid pulling app.py's web deps.
# ----------------------------------------------------------------------------
import ast
def _extract_from_app(names):
    with open(os.path.join(REPO, "app.py")) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.literal_eval(node.value)
    return out
_consts = _extract_from_app({"DEFAULT_PROMPT", "FACT_SCHEMA"})
DEFAULT_PROMPT = _consts["DEFAULT_PROMPT"]
FACT_SCHEMA = _consts["FACT_SCHEMA"]

# ----------------------------------------------------------------------------
# Embedding model for dedup + coverage (jina-v5-nano on CPU, same as app.py).
# Loaded once per process and reused across many experiments.
# ----------------------------------------------------------------------------
_EMB = None
def emb_model():
    global _EMB
    if _EMB is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("jinaai/jina-embeddings-v5-text-nano",
                                   device="cpu", trust_remote_code=True)
    return _EMB

def _embed(texts):
    m = emb_model()
    return m.encode(texts, task="text-matching", normalize_embeddings=True)

def triple_text(f):
    return f"{f.get('subject','')} {f.get('predicate','')} {f.get('object','')}"

def info_text(f):  # richer text for semantic coverage
    return f"{f.get('title','')} {f.get('description','')}"

# ----------------------------------------------------------------------------
# Container lifecycle
# ----------------------------------------------------------------------------
def _sh(cmd, check=True, capture=False):
    return subprocess.run(cmd, shell=True, check=check,
                          capture_output=capture, text=True)

def stop_server():
    _sh(f"sudo docker rm -f {CONTAINER} 2>/dev/null || true", check=False)

def start_server(server_args, load_timeout=360):
    stop_server()
    args = " ".join(server_args)
    cmd = (f"sudo docker run -d --rm --name {CONTAINER} --gpus all "
           f"-p {PORT}:8080 -v {MODELS_DIR}:/models -v {TEMPLATES_DIR}:/templates "
           f"{IMAGE} {args}")
    _sh(cmd)
    t0 = time.time()
    while time.time() - t0 < load_timeout:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5) as r:
                if r.status == 200 and b"ok" in r.read().lower():
                    time.sleep(2)
                    return True
        except Exception:
            pass
        # surface a hard crash early
        st = _sh(f"sudo docker ps -q -f name={CONTAINER}", capture=True, check=False).stdout.strip()
        if not st:
            log = _sh(f"sudo docker logs {CONTAINER} 2>&1 | tail -20", capture=True, check=False).stdout
            raise RuntimeError(f"llama-server container died during load:\n{log}")
        time.sleep(3)
    raise RuntimeError(f"llama-server not healthy within {load_timeout}s")

def server_logs(n=30):
    return _sh(f"sudo docker logs {CONTAINER} 2>&1 | tail -{n}", capture=True, check=False).stdout

# ----------------------------------------------------------------------------
# Document + extraction
# ----------------------------------------------------------------------------
def load_doc():
    with open(DOC_CACHE) as f:
        return f.read()

def build_payload(doc_text, prompt, sampling, seed, no_schema=False):
    docid = hashlib.md5(doc_text[:500].encode()).hexdigest()[:8]
    full = f"{prompt}\n\nDocument:\n  docid: {docid}\n  url: {ARTICLE_URL}\n  text: {doc_text}"
    p = {
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": full}],
        "stream": False,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if not no_schema:  # default: constrain with the JSON schema grammar
        p["response_format"] = {"type": "json_schema",
                                "json_schema": {"name": "ki_facts", "strict": True, "schema": FACT_SCHEMA}}
    p.update(sampling)
    return p

def call_round(payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                                 data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    wall = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    timings = resp.get("timings", {})
    return content, timings, wall

def parse_facts(content):
    """Strict json_schema guarantees a valid object; fall back to regex if needed."""
    # reasoning models (e.g. LFM2.5) prefix a <think>...</think> block even in
    # nothink mode (empty tags); strip it so json.loads sees the object. No-op
    # for Qwen nothink (no tags).
    content = re.sub(r'^\s*<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
            return obj["facts"], True
    except json.JSONDecodeError:
        pass
    facts = []
    for m in re.finditer(r'\{\s*"title"\s*:', content):
        start = m.start(); depth = 0; i = start
        while i < len(content):
            if content[i] == '{': depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        f = json.loads(content[start:i+1])
                        if "title" in f and "subject" in f:
                            facts.append(f)
                    except json.JSONDecodeError:
                        pass
                    break
            i += 1
    return facts, False

# ----------------------------------------------------------------------------
# Quality scoring
# ----------------------------------------------------------------------------
def dedup_unique(rounds_facts, threshold=0.90):
    """Replicate app.py cross-round dedup on the 'triple' field. Returns unique facts."""
    flat = [f for rf in rounds_facts for f in rf]
    if not flat:
        return [], 0, 0
    embs = _embed([triple_text(f) for f in flat])
    kept_embs, unique = [], []
    dupes = 0
    for f, e in zip(flat, embs):
        is_dup = any(float(np.dot(e, ke)) >= threshold for ke in kept_embs)
        if is_dup:
            dupes += 1
        else:
            kept_embs.append(e); unique.append(f)
    return unique, len(flat), dupes

def groundedness(facts, doc_text):
    if not facts:
        return 1.0
    norm = re.sub(r"\s+", " ", doc_text).lower()
    ok = 0
    for f in facts:
        span = re.sub(r"\s+", " ", str(f.get("evidence_span", ""))).lower().strip()
        if span and span in norm:
            ok += 1
    return ok / len(facts)

def coverage(candidate_unique, baseline_unique, cov_thr=0.80):
    """Recall of baseline facts by the candidate set (semantic, title+desc, cosine)."""
    if not baseline_unique:
        return 1.0
    if not candidate_unique:
        return 0.0
    be = _embed([info_text(f) for f in baseline_unique])
    ce = _embed([info_text(f) for f in candidate_unique])
    rec = 0
    for b in be:
        if max(float(np.dot(b, c)) for c in ce) >= cov_thr:
            rec += 1
    return rec / len(baseline_unique)

# ----------------------------------------------------------------------------
# Run one config
# ----------------------------------------------------------------------------
def run_config(cfg, baseline=None, reuse_server=False):
    doc = load_doc()
    server_args = cfg["server_args"]
    sampling = cfg.get("sampling", BASELINE_SAMPLING)
    prompt = cfg.get("prompt") or DEFAULT_PROMPT

    parallel_rounds = cfg.get("parallel_rounds", False)

    if not reuse_server:
        start_server(server_args)

    rounds = [None] * len(SEEDS)
    timings_all = [None] * len(SEEDS)
    schema_flags = [True] * len(SEEDS)

    no_schema = cfg.get("no_schema", False)
    def do_seed(i, seed):
        content, timings, wall = call_round(build_payload(doc, prompt, sampling, seed, no_schema))
        facts, strict = parse_facts(content)
        rounds[i] = facts
        timings_all[i] = timings
        schema_flags[i] = strict

    phase_t0 = time.time()
    if parallel_rounds:
        # fire all rounds concurrently -> exploits GPU headroom; measures the
        # wall-clock of the whole concurrent generation phase (task throughput).
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(SEEDS)) as ex:
            list(ex.map(lambda a: do_seed(*a), list(enumerate(SEEDS))))
    else:
        for i, seed in enumerate(SEEDS):
            do_seed(i, seed)
    phase_wall = time.time() - phase_t0

    pred_n = sum(t.get("predicted_n", 0) for t in timings_all)
    pred_ms = sum(t.get("predicted_ms", 0.0) for t in timings_all)
    prompt_n = sum(t.get("prompt_n", 0) for t in timings_all)
    prompt_ms = sum(t.get("prompt_ms", 0.0) for t in timings_all)
    schema_ok = all(schema_flags)

    # per-stream decode rate (from server timings; sum/sum = single-stream rate)
    decode_tps = (pred_n / pred_ms * 1000.0) if pred_ms > 0 else 0.0
    prefill_tps = (prompt_n / prompt_ms * 1000.0) if prompt_ms > 0 else 0.0
    # task throughput = all generated tokens / wall-clock of the generation phase.
    # For sequential this ~= decode_tps; for parallel it captures the overlap win.
    task_tps = (pred_n / phase_wall) if phase_wall > 0 else 0.0
    wall_total = phase_wall

    unique, total, dupes = dedup_unique(rounds)
    all_flat = [f for rf in rounds for f in rf]
    ground = groundedness(unique, doc)
    cov = coverage(unique, baseline["unique_facts_list"], ) if baseline else None

    res = {
        "id": cfg["id"], "desc": cfg.get("desc", ""),
        "server_args": server_args, "sampling": sampling,
        "metrics": {
            "decode_tps": round(decode_tps, 2),
            "task_tps": round(task_tps, 2),
            "parallel_rounds": parallel_rounds,
            "prefill_tps": round(prefill_tps, 1),
            "predicted_tokens": pred_n,
            "wall_3rounds_s": round(wall_total, 1),
            "unique_facts": len(unique),
            "total_facts": total,
            "dupes": dupes,
            "groundedness": round(ground, 3),
            "schema_valid": schema_ok,
            "coverage_of_baseline": round(cov, 3) if cov is not None else None,
        },
        "unique_facts_list": unique,
    }
    return res

def quality_pass(res, baseline, cov_min=0.97):
    """Hard guard: no quality loss vs baseline.

    Coverage is the primary, trustworthy gate: cov==1.0 means every baseline
    fact is still semantically recalled by the candidate set (no information
    lost). Raw unique-fact count swings +-3 across configs purely from temp-0.7
    RNG-path differences, so it is only a catastrophic-collapse floor here, not
    the gate. Groundedness (verbatim evidence_span match) is brittle/noisy
    (observed 0.48-0.64 across equivalent configs) -> loose floor only.
    The final winner is re-confirmed with a multi-seed comparison separately.
    """
    m = res["metrics"]; b = baseline["metrics"]
    reasons = []
    if not m["schema_valid"]:
        reasons.append("schema invalid")
    if m["coverage_of_baseline"] is not None and m["coverage_of_baseline"] < cov_min:
        reasons.append(f"coverage {m['coverage_of_baseline']}<{cov_min}")
    if m["unique_facts"] < b["unique_facts"] - 2:
        reasons.append(f"fact collapse {m['unique_facts']}<{b['unique_facts']}-2")
    if m["groundedness"] < b["groundedness"] - 0.08:
        reasons.append(f"groundedness drop {m['groundedness']}<{b['groundedness']}")
    return (len(reasons) == 0), reasons

if __name__ == "__main__":
    # smoke: run baseline once
    cfg = {"id": "baseline", "desc": "repo defaults",
           "server_args": BASELINE_SERVER_ARGS, "sampling": BASELINE_SAMPLING}
    r = run_config(cfg)
    print(json.dumps(r["metrics"], indent=2))
    stop_server()
