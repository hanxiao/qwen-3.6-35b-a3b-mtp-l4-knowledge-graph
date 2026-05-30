#!/usr/bin/env python3
"""KI Extractor - Knowledge Indicator extraction via Qwen3.6 MTP + jina-v5-nano dedup."""

import json, time, hashlib, re, random, os, numpy as np
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import httpx
import uvicorn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only for embedding model

app = FastAPI()

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080")
JINA_KEY = os.environ.get("JINA_API_KEY", "")
CTX_SIZE = 16384
MAX_INPUT_CHARS = 100000

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
    """Returns (is_dup, max_sim, dup_of_list) where dup_of_list has {index, similarity} for all matches above threshold."""
    if not existing_embs:
        return False, 0.0, []
    sims = [float(np.dot(new_emb, e)) for e in existing_embs]
    max_sim = max(sims)
    dup_of = []
    if max_sim >= threshold:
        for i, s in enumerate(sims):
            if s >= threshold:
                dup_of.append({"index": existing_indices[i], "similarity": round(s, 4)})
        dup_of.sort(key=lambda x: -x["similarity"])
    return max_sim >= threshold, max_sim, dup_of


DEFAULT_PROMPT = """Return a JSON object with key "facts" containing 0-15 atomic facts.
Long, fact-dense documents (Wikipedia articles, news features, profiles,
academic-staff pages) typically warrant 8-15 facts. Short or generic
documents may warrant 0-3.
Each fact MUST be self-contained: title + description together fully answer
the implied W-question (who/what/when/where/how/which) without requiring the
source document. A future agent should be able to commit to an answer by
reading just title+description -- the description must include the answer
value, supporting evidence (date, location, named witness, exact quantity,
physical detail), and a short verbatim quote (<=30 words) when it adds
disambiguating signal. This bias toward density is intentional even at the
cost of slightly longer descriptions.
Each fact:
 {
 "title": "<one natural sentence <=140 chars stating the fact, ending with the answer value when possible (e.g. 'Townsend was last seen wearing a red shirt.')>",
 "description": "<2-3 sentences <=350 chars carrying the answer + evidence: entity, relation, value, date/location/source detail, and an inline verbatim quote when it disambiguates. Avoid restating the title verbatim.>",
 "subject": "<canonical entity name>",
 "predicate": "<precise snake_case relation, <=32 chars>",
 "object": "<the value of the fact, plain prose>",
 "evidence_span": "<verbatim 1-3 sentence quote, substring of the doc text above>",
 "confidence": <0..100 integer>,
 "tags": ["<entity/topic/year tags, lowercase, alphanumeric+hyphen>", ...]
 }
Coverage priorities -- extract a fact for EACH of the following whenever it's grounded in the doc text:
- Every named person mentioned + their role / position / title (no matter how briefly named --
 a one-line mention of "the secretary, Mary" still warrants its own fact).
- Every named organisation + its relation to the main entity.
- Every concrete date + the event that occurred on it (graduation 22 June 2003, trip 1 Nov 2022, etc.).
- Every named location + what happened there.
- Every distinctive descriptive detail: clothing colour, building material, exact age, weight,
 height, vehicle, distinguishing feature, last-seen description.
- Every cross-entity relationship: X collaborated with Y, X worked for Y, X spoke at Y's
 conference, X's child is Z, X co-edited a book with Y.
Anti-patterns -- do NOT do these:
- Don't only extract facts about the most famous / dominant entity in the doc. Secondary
 individuals named once still warrant their own fact.
- Don't fill the budget with generic claims (founded-year, location, leadership) at the
 expense of specific concrete details that sit deeper in the doc body.
- Don't skip a fact because it seems minor -- minor facts are often what disambiguate two
 similar entities at retrieval time.
Predicate guidance:
- Use a precise snake_case predicate (<=32 chars). Prefer reusing common terms when they fit:
 located_in, founded_in, founded_by, held_event, published_article, won_award, member_of,
 position_held, born_in, died_in, created_by, parent_of, succeeded_by, field_of_study,
 co_authored_with, organized_by, attended_by, physical_description, last_seen_wearing,
 clothing_worn, cross_link.
- Coin a new specific predicate when none of those fit. AVOID the catch-all affiliated_with.
Title and description constraints (CRITICAL -- items violating these are dropped):
- title and description MUST read as natural standalone fact statements.
- They MUST NOT contain the strings: "BrowseComp", "qid", "qid:",
 "use this fact", "anchor a criterion", "without re-reading".
- They MUST NOT mention the document, the dataset, or this task.
Fact constraints:
- Favor specificity (proper nouns, dates, numbers) over generic claims.
- Skip the doc entirely (return empty facts list) for navigation pages, login walls,
 error pages, very short or generic content.
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
        truncated = len(text) > MAX_INPUT_CHARS
        if truncated:
            text = text[:MAX_INPUT_CHARS]
        return text, title, truncated


async def stream_single_extraction(
    body_text: str, extraction_prompt: str, url: str, docid: str,
    round_num: int, seed: int, fact_offset: int,
    existing_embs: list, existing_indices: list, dedup_threshold: float, dedup_field: str = "triple",
    dedup_enabled: bool = True
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


@app.post("/api/extract")
async def extract(request: Request):
    body = await request.json()
    url = body.get("url", "").strip()
    text = body.get("text", "").strip()
    k_rounds = max(1, min(10, int(body.get("k", 1))))
    extraction_prompt = body.get("prompt", DEFAULT_PROMPT).strip()
    dedup_threshold = float(body.get("dedup_threshold", 0.90))
    dedup_field = body.get("dedup_field", "triple")
    dedup_enabled = bool(body.get("dedup_model", "v5-nano"))

    async def event_stream():
        global _active_extractions, _queue_counter, _done_counter
        _queue_counter += 1
        my_pos = _queue_counter
        _active_extractions += 1
        try:
            if url:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching via Jina Reader...'})}\n\n"
                fetched_text, title, truncated = await fetch_url(url)
                yield f"data: {json.dumps({'type': 'fetched', 'title': title, 'chars': len(fetched_text), 'truncated': truncated})}\n\n"
                doc_text = fetched_text
            elif text:
                truncated = len(text) > MAX_INPUT_CHARS
                doc_text = text[:MAX_INPUT_CHARS] if truncated else text
                title = "Pasted text"
                yield f"data: {json.dumps({'type': 'fetched', 'title': title, 'chars': len(doc_text), 'truncated': truncated})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No URL or text provided'})}\n\n"
                return

            docid = hashlib.md5(doc_text[:500].encode()).hexdigest()[:8]
            existing_embs: list = []
            existing_indices: list = []
            total_facts = 0
            total_dupes = 0
            total_tokens = 0
            total_start = time.time()
            all_facts_list = []

            for r in range(1, k_rounds + 1):
                seed = random.randint(1, 999999)
                async for event in stream_single_extraction(
                    doc_text, extraction_prompt, url, docid,
                    round_num=r, seed=seed,
                    fact_offset=total_facts,
                    existing_embs=existing_embs,
                    existing_indices=existing_indices,
                    dedup_threshold=dedup_threshold, dedup_field=dedup_field,
                    dedup_enabled=dedup_enabled,
                ):
                    yield event
                    if event.startswith("data: "):
                        try:
                            d = json.loads(event[6:])
                            if d.get("type") == "round_end":
                                total_facts += d["round_facts"]
                                total_dupes += d["round_dupes"]
                                total_tokens += d["tokens"]
                            elif d.get("type") == "fact":
                                all_facts_list.append({**d["fact"], "_is_duplicate": d["is_duplicate"], "_max_similarity": d["max_similarity"]})
                        except:
                            pass

            total_elapsed = time.time() - total_start
            overall_tps = total_tokens / total_elapsed if total_elapsed > 0 else 0
            unique_facts = total_facts - total_dupes
            unique_list = [f for f in all_facts_list if not f.get("_is_duplicate")]
            clean_list = [{k: v for k, v in f.items() if not k.startswith("_")} for f in unique_list]
            yield f"data: {json.dumps({'type': 'done', 'k_rounds': k_rounds, 'total_facts': total_facts, 'unique_facts': unique_facts, 'duplicate_facts': total_dupes, 'total_tokens': total_tokens, 'elapsed': round(total_elapsed, 1), 'tps': round(overall_tps, 1), 'raw_json': {'facts': clean_list}})}\n\n"

        except Exception as e:
            import traceback; traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            _active_extractions -= 1
            _done_counter += 1

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Track active extractions and queue
import asyncio
_active_extractions = 0
_queue_counter = 0  # total requests received
_done_counter = 0   # total requests completed

@app.get("/api/default-prompt")
async def get_default_prompt():
    return {"prompt": DEFAULT_PROMPT}

@app.get("/api/busy")
async def get_busy():
    queued = _queue_counter - _done_counter
    return {"busy": _active_extractions > 0, "active": _active_extractions, "queued": queued, "total": _queue_counter}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.on_event("startup")
async def startup():
    get_dedup_model()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KI Extractor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#fff;--bg2:#fafafa;--bg3:#f4f4f5;
  --border:#e4e4e7;--border2:#d4d4d8;
  --text:#18181b;--text2:#52525b;--text3:#a1a1aa;
  --black:#18181b;
  --mono:'SF Mono','JetBrains Mono','Fira Code','Consolas',monospace;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI','Helvetica Neue',sans-serif;
  --radius:8px;
}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;font-size:14px;line-height:1.5}

/* NAV */
nav{border-bottom:1px solid var(--border);padding:10px 24px;display:flex;align-items:center;gap:10px}
nav .logo{font-weight:700;font-size:14px;font-family:var(--mono);color:var(--black)}
nav .sep{color:var(--border2)}
nav .tag{font-size:11px;color:var(--text3);font-family:var(--mono)}

/* LAYOUT */
.layout{display:flex;min-height:calc(100vh - 42px)}
.sidebar{width:320px;min-width:320px;border-right:1px solid var(--border);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:14px}
.main{flex:1;padding:20px 24px;overflow-y:auto;background:var(--bg2)}

/* SECTIONS */
.section{margin-bottom:0}
.section-title{font-size:11px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
input[type="text"],textarea{width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:7px 10px;border-radius:6px;font-size:13px;font-family:var(--sans);transition:border-color .15s}
input:focus,textarea:focus{outline:none;border-color:var(--text3)}
textarea{font-family:var(--mono);font-size:11px;line-height:1.5;resize:vertical}
input[type="number"]{width:52px;text-align:center;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:5px 4px;border-radius:6px;font-size:13px;font-family:var(--mono)}
.hidden{display:none}

/* TABS */
.tabs{display:flex;gap:0;margin-bottom:8px}
.tab{padding:5px 12px;cursor:pointer;color:var(--text3);font-size:12px;font-weight:500;border-bottom:2px solid transparent;transition:all .15s}
.tab:hover{color:var(--text2)}
.tab.active{color:var(--text);border-bottom-color:var(--text)}

/* BUTTONS */
.btn{background:var(--black);color:#fff;border:none;padding:8px 0;border-radius:6px;font-size:13px;cursor:pointer;font-weight:500;font-family:var(--sans);width:100%;transition:opacity .15s}
.btn:hover{opacity:.85}
.btn:disabled{background:var(--bg3);color:var(--text3);cursor:not-allowed;opacity:1}
.btn-sm{background:var(--bg);border:1px solid var(--border);color:var(--text2);padding:4px 10px;font-size:11px;font-weight:500;border-radius:5px;cursor:pointer}
.btn-sm:hover{border-color:var(--text3)}

/* PARAMS */
.param-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.param-row label{font-size:12px;color:var(--text2);min-width:65px}
.param-hint{font-size:11px;color:var(--text3);font-family:var(--mono)}
.threshold-row{display:flex;align-items:center;gap:8px}
.threshold-val{font-size:12px;font-family:var(--mono);color:var(--text);font-weight:500;min-width:32px}
input[type="range"]{-webkit-appearance:none;width:100%;height:4px;background:var(--bg3);border-radius:2px;outline:none;border:none;padding:0;margin:0}
input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--black);cursor:pointer;border:2px solid #fff;box-shadow:0 0 0 1px var(--border2)}
.dedup-model{font-size:11px;font-family:var(--mono);color:var(--text3);padding:5px 8px;background:var(--bg3);border-radius:4px;border:none}
.dedup-select{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:5px 8px;border-radius:6px;font-size:12px;font-family:var(--sans)}

/* CONTEXT BAR */
.ctx-bar-inner{height:6px;background:var(--bg3);border-radius:3px;overflow:hidden;display:flex;margin-top:4px}
.ctx-seg{height:100%;transition:width .3s}
.ctx-seg.prompt{background:var(--text3)}
.ctx-seg.doc{background:var(--text2)}
.ctx-seg.output{background:var(--text)}
.ctx-meta{display:flex;justify-content:space-between;margin-top:3px}
.ctx-legend{display:flex;gap:8px}
.ctx-legend-item{display:flex;align-items:center;gap:3px;font-size:9px;color:var(--text3)}
.ctx-legend-dot{width:6px;height:6px;border-radius:1px}
.ctx-total{font-size:10px;color:var(--text3);font-family:var(--mono)}

/* STATUS */
.status-msg{padding:8px 12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text2);font-size:12px;margin-bottom:12px}
.status-msg.err{border-color:#fca5a5;color:#b91c1c;background:#fef2f2}
.status-msg.ok{border-color:#bbf7d0;color:#166534;background:#f0fdf4}
.spinner{display:inline-block;width:10px;height:10px;border:1.5px solid var(--border);border-top-color:var(--text2);border-radius:50%;animation:spin .7s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* STATS */
.stats-bar{display:flex;gap:0;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:16px}
.stat{flex:1;padding:10px 12px;text-align:center;border-right:1px solid var(--border)}
.stat:last-child{border-right:none}
.sv{font-size:18px;font-weight:700;color:var(--text);font-family:var(--mono)}
.sv.muted{color:var(--text3)}
.sl{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:1px}

/* FACT CARDS */
.facts{display:flex;flex-wrap:wrap;gap:10px}
.fc{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:14px;animation:fadeIn .25s ease-out;width:calc(33.333% - 7px);min-width:260px}
.fc.dup{opacity:.35;border-style:dashed}
.fc.dup .fc-dup-link{cursor:pointer;text-decoration:underline;text-decoration-style:dotted}
.fc.dup .fc-dup-link:hover{opacity:.7}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

/* MODAL */
.modal-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.4);z-index:200;display:flex;align-items:center;justify-content:center;animation:modalFadeIn .15s}
@keyframes modalFadeIn{from{opacity:0}to{opacity:1}}
.modal{background:var(--bg);border-radius:10px;width:90%;max-width:700px;max-height:80vh;overflow-y:auto;box-shadow:0 8px 30px rgba(0,0,0,.12);padding:20px}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.modal-title{font-size:14px;font-weight:600;color:var(--text)}
.modal-close{background:none;border:none;font-size:18px;cursor:pointer;color:var(--text3);padding:4px 8px}
.modal-close:hover{color:var(--text)}
.modal-card{border:1px solid var(--border);border-radius:var(--radius);padding:12px;margin-bottom:10px}
.modal-card.highlight{border-color:var(--text);border-width:2px;background:var(--bg2)}
.modal-card .fc-title{font-size:13px}
.modal-card .fc-desc{font-size:11px}
.modal-sim{font-size:10px;font-family:var(--mono);color:var(--text3);margin-top:4px}
.fc-header{display:flex;align-items:center;gap:6px;margin-bottom:5px}
.fc-num{font-size:10px;font-weight:600;color:var(--text2);font-family:var(--mono);background:var(--bg3);padding:1px 6px;border-radius:3px}
.fc-round{font-size:10px;color:var(--text3);font-family:var(--mono)}
.fc-dup-badge{font-size:9px;color:var(--text3);background:var(--bg3);padding:1px 5px;border-radius:3px;font-family:var(--mono)}
.fc-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;line-height:1.4}
.fc-desc{font-size:11.5px;color:var(--text2);margin-bottom:8px;line-height:1.5}
.fc-desc-toggle{font-size:10px;color:var(--text3);cursor:pointer;margin-bottom:6px;user-select:none}
.fc-desc-toggle:hover{color:var(--text2)}
.busy-banner{position:fixed;top:42px;left:0;right:0;background:#fefce8;border-bottom:1px solid #fde68a;color:#92400e;font-size:12px;padding:6px 24px;z-index:99;display:flex;align-items:center;gap:8px}
.busy-dot{width:8px;height:8px;border-radius:50%;background:#f59e0b;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.triple{display:flex;align-items:center;gap:4px;font-family:var(--mono);font-size:10px;color:var(--text3);margin-bottom:6px;flex-wrap:wrap}
.t-s{color:var(--text);font-weight:600}.t-p{color:var(--text2)}.t-o{color:var(--text);font-weight:600}
.conf-row{display:flex;align-items:center;gap:6px;margin-bottom:6px}
.conf-bar{flex:1;max-width:80px;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden}
.conf-fill{height:100%;border-radius:2px;background:var(--text3)}
.conf-lbl{font-size:10px;color:var(--text3);font-family:var(--mono)}
.tags{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:5px}
.tag{background:var(--bg3);color:var(--text2);padding:1px 6px;border-radius:3px;font-size:9px}
.fc-detail{margin-top:4px}
.fc-detail summary{font-size:10px;color:var(--text3);cursor:pointer}
.fc-detail summary:hover{color:var(--text2)}
.fc-detail .fc-desc{margin-top:4px}
.fc-evidence{margin-top:6px;padding:6px 10px;background:var(--bg2);border-left:2px solid var(--border);font-size:10px;color:var(--text2);font-style:italic;line-height:1.4}
.fc-raw{margin-top:6px;padding:8px;background:var(--bg2);border:1px solid var(--border);border-radius:4px;font-family:var(--mono);font-size:9px;color:var(--text2);overflow-x:auto;white-space:pre-wrap;max-height:200px;overflow-y:auto}

.raw-section{margin-top:20px}
.raw-section summary{font-size:12px;color:var(--text3);cursor:pointer;margin-bottom:4px}
.raw-json{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:12px;font-family:var(--mono);font-size:10px;color:var(--text);overflow-x:auto;white-space:pre-wrap;max-height:350px;overflow-y:auto}

.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:300px;color:var(--text3);width:100%}
.empty-text{font-size:13px}

@media(max-width:1200px){.fc{width:calc(50% - 5px)}}
@media(max-width:900px){
  .layout{flex-direction:column}
  .sidebar{width:100%;min-width:auto;border-right:none;border-bottom:1px solid var(--border)}
  .fc{width:100%}
}
</style>
</head>
<body>

<nav>
  <div class="logo">KI Extractor</div>
  <span class="sep">|</span>
  <span class="tag">Qwen3.6-35B-A3B-MTP &middot; NVIDIA L4 24GB</span>
</nav>

<div class="layout">
  <div class="sidebar">
    <div class="section">
      <div class="section-title">Source</div>
      <div class="tabs">
        <div class="tab active" data-tab="url">URL</div>
        <div class="tab" data-tab="text">Paste</div>
      </div>
      <div id="p-url"><input type="text" id="url" placeholder="https://..." value="https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/"></div>
      <div id="p-text" class="hidden"><textarea id="text-paste" rows="5" placeholder="Paste markdown or plain text..."></textarea></div>
    </div>

    <div class="section">
      <div class="section-title">Extraction Prompt</div>
      <textarea id="prompt-edit" rows="6"></textarea>
      <div style="margin-top:4px;text-align:right"><button class="btn-sm" onclick="resetPrompt()">Reset</button></div>
    </div>

    <div class="section">
      <div class="section-title">Parameters</div>
      <div class="param-row">
        <label>Rounds</label>
        <input type="number" id="k-input" value="3" min="1" max="10" onchange="updateEstimate()">
        <span class="param-hint" id="rounds-est">~15-45 facts</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Deduplication</div>
      <div class="param-row" style="margin:0;margin-bottom:6px">
        <label>Model</label>
        <select id="dedup-model" class="dedup-select">
          <option value="v5-nano" selected>jina-embeddings-v5-text-nano (CPU)</option>
          <option value="">None (disable dedup)</option>
        </select>
      </div>
      <div style="margin-top:6px">
        <div class="param-row" style="margin:0;margin-bottom:6px">
          <label>Field</label>
          <select id="dedup-field" class="dedup-select">
            <option value="triple" selected>Triple (S→P→O)</option>
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
    </div>

    <div class="section">
      <div class="section-title">Context</div>
      <div class="ctx-bar-inner" id="ctx-bar">
        <div class="ctx-seg prompt" id="ctx-prompt" style="width:0%"></div>
        <div class="ctx-seg doc" id="ctx-doc" style="width:0%"></div>
        <div class="ctx-seg output" id="ctx-output" style="width:0%"></div>
      </div>
      <div class="ctx-meta">
        <div class="ctx-legend">
          <span class="ctx-legend-item"><span class="ctx-legend-dot" style="background:var(--text3)"></span>Prompt</span>
          <span class="ctx-legend-item"><span class="ctx-legend-dot" style="background:var(--text2)"></span>Doc</span>
          <span class="ctx-legend-item"><span class="ctx-legend-dot" style="background:var(--text)"></span>Output</span>
        </div>
        <span class="ctx-total" id="ctx-total">0 / 16384</span>
      </div>
    </div>

    <button class="btn" id="extract-btn" onclick="extract()">Extract</button>
    <div id="busy-banner" class="busy-banner" style="display:none"><div class="busy-dot"></div><span id="busy-text">Server busy</span></div>
  </div>

  <div class="main">
    <div id="status-area"></div>
    <div id="stats-area" class="stats-bar hidden">
      <div class="stat"><div class="sv" id="s-unique">0</div><div class="sl">Unique</div></div>
      <div class="stat"><div class="sv muted" id="s-total">0</div><div class="sl">Total</div></div>
      <div class="stat"><div class="sv" id="s-elapsed">0s</div><div class="sl">Time</div></div>
      <div class="stat"><div class="sv" id="s-tps">0</div><div class="sl">tok/s</div></div>
      <div class="stat"><div class="sv" id="s-round">-</div><div class="sl">Round</div></div>
    </div>
    <div class="facts" id="facts-area">
      <div class="empty" id="empty-state">
        <div class="empty-text">Enter a URL and click Extract</div>
      </div>
    </div>
    <details class="raw-section hidden" id="raw-section">
      <summary>Raw JSON (unique facts only)</summary>
      <pre class="raw-json" id="raw-json"></pre>
    </details>
  </div>
</div>

<script>
let allFacts=[], currentTab='url', defaultPrompt='';
let factsData=[];  // {index, round, fact, isDup, maxSim, dupOf[]}

fetch('/api/default-prompt').then(r=>r.json()).then(d=>{
  defaultPrompt=d.prompt;
  document.getElementById('prompt-edit').value=d.prompt;
});

function toggleDedupControls(){
  const on=document.getElementById('dedup-model').value!=='';
  document.getElementById('dedup-field').disabled=!on;
  document.getElementById('dedup-slider').disabled=!on;
}
document.getElementById('dedup-model').addEventListener('change',toggleDedupControls);

function resetPrompt(){document.getElementById('prompt-edit').value=defaultPrompt}
function updateEstimate(){
  const k=parseInt(document.getElementById('k-input').value)||3;
  document.getElementById('rounds-est').textContent='~'+Math.min(15,k*5)+'-'+k*15+' facts';
}

document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
  const tab=t.dataset.tab;currentTab=tab;
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('p-url').classList.toggle('hidden',tab!=='url');
  document.getElementById('p-text').classList.toggle('hidden',tab!=='text');
}));

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}

function renderFact(idx,round,fact,isDup,maxSim,dupOf){
  const c=fact.confidence||0,
    tags=(fact.tags||[]).map(t=>'<span class="tag">'+esc(t)+'</span>').join(''),
    ev=fact.evidence_span?'<details class="ev"><summary>evidence</summary><blockquote>'+esc(fact.evidence_span)+'</blockquote></details>':'';
  let dupBadge='';
  if(isDup && dupOf && dupOf.length){
    const refs=dupOf.map(d=>'#'+d.index).join(', ');
    dupBadge='<span class="fc-dup-badge fc-dup-link" onclick="showDupModal('+idx+')">dup of '+refs+' ('+maxSim.toFixed(2)+')</span>';
  } else if(isDup){
    dupBadge='<span class="fc-dup-badge">dup '+maxSim.toFixed(2)+'</span>';
  }
  const dupCls=isDup?' dup':'';
  return '<div class="fc'+dupCls+'" data-fact-idx="'+idx+'"><div class="fc-header"><span class="fc-num">#'+idx+'</span><span class="fc-round">R'+round+'</span>'+dupBadge+'</div>'
    +'<div class="fc-title">'+esc(fact.title||'')+'</div>'
    +'<div class="triple"><span class="t-s">'+esc(fact.subject||'')+'</span><span style="color:var(--text3)">&rarr;</span><span class="t-p">'+esc(fact.predicate||'')+'</span><span style="color:var(--text3)">&rarr;</span><span class="t-o">'+esc(fact.object||'')+'</span></div>'
    +'<div class="conf-row"><div class="conf-bar"><div class="conf-fill" style="width:'+c+'%"></div></div><div class="conf-lbl">'+c+'%</div></div>'
    +'<div class="tags">'+tags+'</div>'
    +'<details class="fc-detail"><summary>details</summary>'
    +'<div class="fc-desc">'+esc(fact.description||'')+'</div>'
    +(fact.evidence_span?'<blockquote class="fc-evidence">'+esc(fact.evidence_span)+'</blockquote>':'')
    +'<pre class="fc-raw">'+esc(JSON.stringify(fact,null,2))+'</pre>'
    +'</details></div>';
}

function showDupModal(dupIdx){
  const entry=factsData.find(f=>f.index===dupIdx);
  if(!entry)return;
  const dupOf=entry.dupOf||[];
  const relatedIndices=dupOf.map(d=>d.index);
  // Build modal
  let cards='';
  // Show the dup fact first (highlighted as "this fact")
  cards+='<div class="modal-card"><div style="font-size:10px;color:var(--text3);margin-bottom:4px;font-family:var(--mono)">This fact (#'+dupIdx+', R'+entry.round+')</div>'
    +'<div class="fc-title">'+esc(entry.fact.title||'')+'</div>'
    +'<div class="fc-desc">'+esc(entry.fact.description||'')+'</div>'
    +'<div class="triple"><span class="t-s">'+esc(entry.fact.subject||'')+'</span> &rarr; <span class="t-p">'+esc(entry.fact.predicate||'')+'</span> &rarr; <span class="t-o">'+esc(entry.fact.object||'')+'</span></div></div>';
  // Show each matched fact
  for(const d of dupOf){
    const matched=factsData.find(f=>f.index===d.index);
    if(!matched)continue;
    cards+='<div class="modal-card highlight"><div style="font-size:10px;color:var(--text3);margin-bottom:4px;font-family:var(--mono)">Matched: #'+d.index+' (R'+matched.round+') &middot; similarity '+d.similarity.toFixed(4)+'</div>'
      +'<div class="fc-title">'+esc(matched.fact.title||'')+'</div>'
      +'<div class="fc-desc">'+esc(matched.fact.description||'')+'</div>'
      +'<div class="triple"><span class="t-s">'+esc(matched.fact.subject||'')+'</span> &rarr; <span class="t-p">'+esc(matched.fact.predicate||'')+'</span> &rarr; <span class="t-o">'+esc(matched.fact.object||'')+'</span></div></div>';
  }
  const overlay=document.createElement('div');
  overlay.className='modal-overlay';
  overlay.onclick=e=>{if(e.target===overlay)overlay.remove()};
  overlay.innerHTML='<div class="modal"><div class="modal-header"><div class="modal-title">Duplicate Analysis &middot; #'+dupIdx+'</div><button class="modal-close" onclick="this.closest(\'.modal-overlay\').remove()">&times;</button></div>'+cards+'</div>';
  document.body.appendChild(overlay);
}

function updateCtx(pTok,dTok,oTok){
  const t=pTok+dTok+oTok;
  document.getElementById('ctx-prompt').style.width=(pTok/16384*100)+'%';
  document.getElementById('ctx-doc').style.width=(dTok/16384*100)+'%';
  document.getElementById('ctx-output').style.width=(oTok/16384*100)+'%';
  document.getElementById('ctx-total').textContent=t+' / 16384';
}

let uniqueCount=0, totalCount=0;

async function extract(){
  const btn=document.getElementById('extract-btn');
  btn.disabled=true;btn.textContent='Extracting...';
  window._selfExtracting=true;
  document.getElementById('raw-section').classList.add('hidden');
  const es=document.getElementById('empty-state');if(es)es.remove();
  allFacts=[];factsData=[];uniqueCount=0;totalCount=0;
  document.getElementById('facts-area').innerHTML='';
  document.getElementById('status-area').innerHTML='';
  document.getElementById('stats-area').classList.remove('hidden');
  document.getElementById('s-unique').textContent='0';
  document.getElementById('s-total').textContent='0';
  document.getElementById('s-elapsed').textContent='0s';
  document.getElementById('s-tps').textContent='0';
  document.getElementById('s-round').textContent='-';
  updateCtx(0,0,0);

  const k=parseInt(document.getElementById('k-input').value)||3;
  const prompt=document.getElementById('prompt-edit').value;
  const threshold=parseFloat(document.getElementById('dedup-slider').value);
  const dedupField=document.getElementById('dedup-field').value;
  const dedupModel=document.getElementById('dedup-model').value;
  const payload={k, prompt, dedup_threshold: threshold, dedup_field: dedupField, dedup_model: dedupModel};
  if(currentTab==='url') payload.url=document.getElementById('url').value;
  else payload.text=document.getElementById('text-paste').value;

  let totalTokens=0;
  const t0=performance.now();

  try{
    const resp=await fetch('/api/extract',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const reader=resp.body.getReader();
    const dec=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        try{
          const d=JSON.parse(line.slice(6));
          switch(d.type){
            case 'status':
              document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>'+esc(d.message)+'</div>';break;
            case 'fetched':
              document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>'+esc(d.title||'')+' ('+d.chars.toLocaleString()+' chars)</div>';break;
            case 'round_start':
              document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>Round '+d.round+'/'+k+' &middot; seed '+d.seed+'</div>';
              document.getElementById('s-round').textContent=d.round+'/'+k;
              updateCtx(d.system_tokens_est||0,d.doc_tokens_est||0,0);
              break;
            case 'metrics':
              const el=(performance.now()-t0)/1000;
              document.getElementById('s-elapsed').textContent=el.toFixed(1)+'s';
              document.getElementById('s-tps').textContent=d.tps;
              updateCtx(d.system_tokens_est||0,d.doc_tokens_est||0,d.tokens);
              break;
            case 'fact':
              totalCount++;
              if(!d.is_duplicate) uniqueCount++;
              allFacts.push(d.fact);
              factsData.push({index:d.index, round:d.round, fact:d.fact, isDup:d.is_duplicate, maxSim:d.max_similarity, dupOf:d.dup_of||[]});
              // Insert: unique at top, dup at bottom
              if(d.is_duplicate){
                document.getElementById('facts-area').insertAdjacentHTML('beforeend',renderFact(d.index,d.round,d.fact,true,d.max_similarity,d.dup_of||[]));
              } else {
                // Insert before first dup card, or at top
                const firstDup=document.querySelector('.fc.dup');
                if(firstDup){
                  firstDup.insertAdjacentHTML('beforebegin',renderFact(d.index,d.round,d.fact,false,0,[]));
                } else {
                  document.getElementById('facts-area').insertAdjacentHTML('afterbegin',renderFact(d.index,d.round,d.fact,false,0,[]));
                }
              }
              document.getElementById('s-unique').textContent=uniqueCount;
              document.getElementById('s-total').textContent=totalCount;
              break;
            case 'round_end':
              totalTokens+=d.tokens;
              const nf=d.round_facts-d.round_dupes;
              document.getElementById('status-area').innerHTML='<div class="status-msg">Round '+d.round+' &middot; '+nf+' new, '+d.round_dupes+' dup &middot; '+d.tps+' tok/s '+(d.note||'')+'</div>';
              break;
            case 'done':
              const finalEl=(performance.now()-t0)/1000;
              document.getElementById('status-area').innerHTML='<div class="status-msg ok">'+d.unique_facts+' unique facts ('+d.duplicate_facts+' duplicates filtered) &middot; '+d.k_rounds+' rounds &middot; '+finalEl.toFixed(1)+'s</div>';
              document.getElementById('s-elapsed').textContent=finalEl.toFixed(1)+'s';
              document.getElementById('s-tps').textContent=d.tps;
              document.getElementById('s-unique').textContent=d.unique_facts;
              document.getElementById('s-total').textContent=d.total_facts;
              if(d.raw_json){document.getElementById('raw-json').textContent=JSON.stringify(d.raw_json,null,2)}
              document.getElementById('raw-section').classList.remove('hidden');
              break;
            case 'error':
              document.getElementById('status-area').innerHTML='<div class="status-msg err">'+esc(d.message)+'</div>';break;
          }
        }catch(e){}
      }
    }
  }catch(e){
    document.getElementById('status-area').innerHTML='<div class="status-msg err">'+esc(e.message)+'</div>';
  }
  btn.disabled=false;btn.textContent='Extract';
  window._selfExtracting=false;
}

document.getElementById('url').addEventListener('keydown',e=>{if(e.key==='Enter')extract()});
updateEstimate();
setInterval(()=>{fetch('/api/busy').then(r=>r.json()).then(d=>{
  const showBusy=d.busy && !window._selfExtracting;
  document.getElementById('busy-banner').style.display=showBusy?'flex':'none';
  if(showBusy){
    document.getElementById('busy-text').textContent='Server busy \u00b7 Queue: '+d.queued+' request'+(d.queued>1?'s':'');
  }
}).catch(()=>{})},3000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
