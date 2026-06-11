#!/usr/bin/env python3
"""KI Extractor - Knowledge Indicator extraction via Qwen3.6 MTP + jina-v5-nano dedup."""

import json, time, hashlib, re, random, os, io, zipfile, tempfile, shutil, numpy as np
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


@app.post("/api/extract-zip")
async def extract_zip(
    file: UploadFile = File(...),
    k: int = Form(1),
    prompt: str = Form(DEFAULT_PROMPT),
    dedup_threshold: float = Form(0.90),
    dedup_field: str = Form("triple"),
    dedup_model: str = Form(""),
):
    k_rounds = max(1, min(10, int(k)))
    extraction_prompt = (prompt or DEFAULT_PROMPT).strip()
    dedup_enabled = bool(dedup_model)
    raw = await file.read()

    async def event_stream():
        global _active_extractions, _queue_counter, _done_counter
        _queue_counter += 1
        _active_extractions += 1
        tmpdir = tempfile.mkdtemp(prefix="kizip_")
        zip_path = os.path.join(tmpdir, "upload.zip")
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with open(zip_path, "wb") as f:
                f.write(raw)
            try:
                files = list_zip_files(zip_path, extract_dir)
            except zipfile.BadZipFile:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Not a valid zip file'})}\n\n"
                return
            if not files:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No supported text files found in zip'})}\n\n"
                return

            file_list = [f["name"] for f in files]
            yield f"data: {json.dumps({'type': 'filelist', 'files': file_list})}\n\n"

            # dedup state shared across all files (cross-file dedup)
            existing_embs: list = []
            existing_indices: list = []
            total_facts = 0
            total_dupes = 0
            total_tokens = 0
            total_start = time.time()
            all_facts_list = []

            for fi, entry in enumerate(files):
                fname = entry["name"]
                yield f"data: {json.dumps({'type': 'file_start', 'file_index': fi, 'file': fname})}\n\n"
                doc_text = read_file_text(entry["path"], fname)
                truncated = len(doc_text) > MAX_INPUT_CHARS
                if truncated:
                    doc_text = doc_text[:MAX_INPUT_CHARS]
                if not doc_text.strip():
                    yield f"data: {json.dumps({'type': 'file_end', 'file_index': fi, 'file': fname, 'file_facts': 0, 'skipped': True})}\n\n"
                    continue
                docid = hashlib.md5(doc_text[:500].encode()).hexdigest()[:8]
                file_facts = 0
                for r in range(1, k_rounds + 1):
                    seed = random.randint(1, 999999)
                    async for event in stream_single_extraction(
                        doc_text, extraction_prompt, "", docid,
                        round_num=r, seed=seed,
                        fact_offset=total_facts,
                        existing_embs=existing_embs,
                        existing_indices=existing_indices,
                        dedup_threshold=dedup_threshold, dedup_field=dedup_field,
                        dedup_enabled=dedup_enabled, source_file=fname,
                    ):
                        yield event
                        if event.startswith("data: "):
                            try:
                                d = json.loads(event[6:])
                                if d.get("type") == "round_end":
                                    total_facts += d["round_facts"]
                                    total_dupes += d["round_dupes"]
                                    total_tokens += d["tokens"]
                                    file_facts += d["round_facts"]
                                elif d.get("type") == "fact":
                                    all_facts_list.append({**d["fact"], "_is_duplicate": d["is_duplicate"], "_max_similarity": d["max_similarity"]})
                            except Exception:
                                pass
                yield f"data: {json.dumps({'type': 'file_end', 'file_index': fi, 'file': fname, 'file_facts': file_facts})}\n\n"

            total_elapsed = time.time() - total_start
            overall_tps = total_tokens / total_elapsed if total_elapsed > 0 else 0
            unique_facts = total_facts - total_dupes
            # JSONL: one line per fact (all facts, dup flag included)
            jsonl_lines = []
            for f in all_facts_list:
                clean = {k2: v for k2, v in f.items() if not k2.startswith("_")}
                clean["is_duplicate"] = f.get("_is_duplicate", False)
                jsonl_lines.append(json.dumps(clean, ensure_ascii=False))
            jsonl_text = "\n".join(jsonl_lines)
            yield f"data: {json.dumps({'type': 'done', 'k_rounds': k_rounds, 'num_files': len(files), 'total_facts': total_facts, 'unique_facts': unique_facts, 'duplicate_facts': total_dupes, 'total_tokens': total_tokens, 'elapsed': round(total_elapsed, 1), 'tps': round(overall_tps, 1), 'jsonl': jsonl_text})}\n\n"
        except Exception as e:
            import traceback; traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            _active_extractions -= 1
            _done_counter += 1

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Track active extractions and queue
import asyncio
_active_extractions = 0
_queue_counter = 0  # total requests received
_done_counter = 0   # total requests completed

DEFAULT_ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jina-corpus.zip")
DEFAULT_ZIP_NAME = "jina-corpus.zip"

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

.layout{display:flex;height:calc(100vh - 42px)}
.sidebar{width:330px;min-width:330px;border-right:1px solid var(--border);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:14px;background:var(--bg2)}
.main{flex:1;position:relative;background:var(--bg);overflow:hidden}

.section-title{font-size:10px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
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
.graph-overlay{position:absolute;top:12px;left:12px;display:flex;gap:6px;align-items:center;z-index:10}
.graph-badge{background:var(--bg);border:1px solid var(--border);border-radius:0;padding:4px 9px;font-size:10px;font-family:var(--mono);color:var(--text);text-transform:uppercase;letter-spacing:.5px}

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
</style>
</head>
<body>

<nav>
  <div class="logo">KI Extractor</div>
  <span class="sep">|</span>
  <span class="tag">qwen3.6-35b-a3b-mtp / nvidia l4 24gb</span>
</nav>

<div class="layout">
  <div class="sidebar">
    <div class="section">
      <div class="section-title">Source</div>
      <div class="tabs">
        <div class="tab active" data-tab="url">URL</div>
        <div class="tab" data-tab="text">Paste</div>
        <div class="tab" data-tab="zip">Zip</div>
      </div>
      <div id="p-url"><input type="text" id="url" placeholder="https://..." value="https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/"></div>
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
      <div class="section-title">Extraction Prompt</div>
      <textarea id="prompt-edit" rows="5"></textarea>
      <div style="margin-top:4px;text-align:right"><button class="btn-sm" onclick="resetPrompt()">Reset</button></div>
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

    <div id="busy-banner" class="busy-banner" style="display:none"><div class="busy-dot"></div><span id="busy-text">Server busy</span></div>
  </div>

  <div class="main">
    <div id="graph"></div>
    <div class="graph-empty" id="graph-empty">Extract facts to build the relation graph</div>
    <div class="graph-overlay hidden" id="graph-overlay">
      <span class="graph-badge" id="graph-stats">0 edges</span>
      <span class="graph-badge" style="cursor:pointer" onclick="zoomFit()">fit</span>
    </div>
    <div class="hovercard hidden" id="hovercard"></div>
  </div>
</div>

<script>
let currentTab='url', defaultPrompt='', zipFileObj=null;
let jsonlLines=[];           // raw jsonl strings for download
let graphNodes=new Map();    // id -> node
let graphLinks=[];           // link objects, each carries .fact
let Graph=null;

fetch('/api/default-prompt').then(r=>r.json()).then(d=>{
  defaultPrompt=d.prompt;
  document.getElementById('prompt-edit').value=d.prompt;
});

// Preload the bundled default corpus zip so the Zip tab works out of the box.
let defaultZipLoaded=false;
fetch('/api/default-zip-info').then(r=>r.json()).then(d=>{
  if(!d.available)return;
  const zn=document.getElementById('zip-name');
  zn.textContent='loading '+d.name+' ('+d.files+' files, '+d.size_mb+' MB)...';
  fetch('/api/default-zip').then(r=>r.blob()).then(b=>{
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

document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
  const tab=t.dataset.tab;currentTab=tab;
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('p-url').classList.toggle('hidden',tab!=='url');
  document.getElementById('p-text').classList.toggle('hidden',tab!=='text');
  document.getElementById('p-zip').classList.toggle('hidden',tab!=='zip');
}));

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
    .nodeLabel(n=>n.id)
    .nodeVal(n=>Math.min(10,1.5+n.deg))
    .nodeColor(()=> '#1a1a1a')
    .nodeRelSize(4)
    .linkColor(()=> 'rgba(26,26,26,0.30)')
    .linkWidth(1)
    .linkDirectionalArrowLength(3.5)
    .linkDirectionalArrowColor(()=> 'rgba(26,26,26,0.45)')
    .linkDirectionalArrowRelPos(1)
    .linkHoverPrecision(8)
    .nodeCanvasObjectMode(()=> 'after')
    .nodeCanvasObject((n,ctx,scale)=>{
      const r=Math.min(10,1.5+n.deg)/scale;
      if(scale>1.0){
        const label=n.id.length>30?n.id.slice(0,29)+'…':n.id;
        ctx.font=`${Math.max(2.5,10/scale)}px 'SF Mono',monospace`;
        ctx.fillStyle='#1a1a1a';ctx.textAlign='center';ctx.textBaseline='top';
        ctx.fillText(label,n.x,n.y+r+1.5/scale);
      }
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
  _fitPending=setTimeout(()=>zoomFit(),250);
}

function addFactEdge(fact){
  const s=(fact.subject||'').trim()||'?';
  const o=(fact.object||'').trim()||'?';
  for(const id of [s,o]){
    if(!graphNodes.has(id)) graphNodes.set(id,{id,deg:0});
    graphNodes.get(id).deg++;
  }
  graphLinks.push({source:s,target:o,fact});
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

// ---------- Extraction ----------
let totalTps=0;
async function extract(){
  const btn=document.getElementById('extract-btn');
  btn.disabled=true;btn.textContent='Extracting...';
  window._selfExtracting=true;
  // reset
  jsonlLines=[];graphNodes=new Map();graphLinks=[];
  document.getElementById('graph-empty').classList.add('hidden');
  document.getElementById('graph-overlay').classList.remove('hidden');
  document.getElementById('results-section').classList.remove('hidden');
  document.getElementById('dl-btn').disabled=true;
  document.getElementById('s-edges').textContent='0';
  document.getElementById('s-nodes').textContent='0';
  document.getElementById('s-tps').textContent='0';
  document.getElementById('files-section').classList.add('hidden');
  document.getElementById('status-area').innerHTML='';
  initGraph();refreshGraph();

  const k=parseInt(document.getElementById('k-input').value)||1;
  const prompt=document.getElementById('prompt-edit').value;
  const threshold=parseFloat(document.getElementById('dedup-slider').value);
  const dedupField=document.getElementById('dedup-field').value;
  const dedupModel=document.getElementById('dedup-model').value;

  let resp;
  try{
    if(currentTab==='zip'){
      if(!zipFileObj){fail('Choose a .zip file first');return;}
      const fd=new FormData();
      fd.append('file',zipFileObj);
      fd.append('k',k);fd.append('prompt',prompt);
      fd.append('dedup_threshold',threshold);fd.append('dedup_field',dedupField);fd.append('dedup_model',dedupModel);
      resp=await fetch('/api/extract-zip',{method:'POST',body:fd});
    }else{
      const payload={k,prompt,dedup_threshold:threshold,dedup_field:dedupField,dedup_model:dedupModel};
      if(currentTab==='url')payload.url=document.getElementById('url').value;
      else payload.text=document.getElementById('text-paste').value;
      resp=await fetch('/api/extract',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    }
    await consume(resp,k);
  }catch(e){fail(e.message);}
  btn.disabled=false;btn.textContent='Extract';window._selfExtracting=false;
}
function fail(msg){
  document.getElementById('status-area').innerHTML='<div class="status-msg err">'+esc(msg)+'</div>';
  document.getElementById('extract-btn').disabled=false;
  document.getElementById('extract-btn').textContent='Extract';
  window._selfExtracting=false;
}

async function consume(resp,k){
  const reader=resp.body.getReader();const dec=new TextDecoder();let buf='';
  let totalFiles=0, doneFiles=0;
  while(true){
    const{done,value}=await reader.read();if(done)break;
    buf+=dec.decode(value,{stream:true});
    const lines=buf.split('\n');buf=lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: '))continue;
      let d;try{d=JSON.parse(line.slice(6));}catch(e){continue;}
      switch(d.type){
        case 'status':
          document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>'+esc(d.message)+'</div>';break;
        case 'fetched':
          document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>'+esc(d.title||'')+' ('+d.chars.toLocaleString()+' chars)</div>';break;
        case 'filelist':
          totalFiles=d.files.length;renderFileList(d.files);break;
        case 'file_start':
          setFileState(d.file_index,'running');break;
        case 'file_end':
          doneFiles++;setFileState(d.file_index,d.skipped?'skip':'done',d.file_facts);updateFilesProgress(doneFiles,totalFiles);break;
        case 'round_start':
          document.getElementById('status-area').innerHTML='<div class="status-msg"><span class="spinner"></span>Round '+d.round+'/'+k+' · seed '+d.seed+'</div>';break;
        case 'metrics':
          document.getElementById('s-tps').textContent=d.tps;break;
        case 'fact':
          // one jsonl line per fact (one edge per line)
          const rec={...d.fact};rec.is_duplicate=d.is_duplicate;
          jsonlLines.push(JSON.stringify(rec));
          // graph: only draw unique facts (skip duplicates)
          if(!d.is_duplicate){ addFactEdge(rec);refreshGraph(); }
          document.getElementById('s-tps').textContent=d.tps;
          break;
        case 'round_end':
          document.getElementById('s-tps').textContent=d.tps;break;
        case 'done':
          let msg=d.unique_facts+' unique facts';
          if(d.num_files!=null)msg+=' · '+d.num_files+' files';
          msg+=' · '+(d.duplicate_facts||0)+' dup · '+d.elapsed+'s';
          document.getElementById('status-area').innerHTML='<div class="status-msg ok">'+msg+'</div>';
          // prefer server jsonl when present (zip path)
          if(d.jsonl!=null && d.jsonl!=='') jsonlLines=d.jsonl.split('\n').filter(x=>x);
          document.getElementById('dl-btn').disabled=jsonlLines.length===0;
          setTimeout(zoomFit,300);
          break;
        case 'error':
          document.getElementById('status-area').innerHTML='<div class="status-msg err">'+esc(d.message)+'</div>';break;
      }
    }
  }
  document.getElementById('dl-btn').disabled=jsonlLines.length===0;
}

function downloadJsonl(){
  const blob=new Blob([jsonlLines.join('\n')+'\n'],{type:'application/x-ndjson'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  const ts=new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
  a.download='ki-facts-'+ts+'.jsonl';a.click();URL.revokeObjectURL(a.href);
}

document.getElementById('url').addEventListener('keydown',e=>{if(e.key==='Enter')extract()});
window.addEventListener('resize',()=>{if(Graph){const m=document.querySelector('.main');Graph.width(m.clientWidth).height(m.clientHeight);zoomFit();}});

setInterval(()=>{fetch('/api/busy').then(r=>r.json()).then(d=>{
  const showBusy=d.busy && !window._selfExtracting;
  document.getElementById('busy-banner').style.display=showBusy?'flex':'none';
  if(showBusy)document.getElementById('busy-text').textContent='Server busy · Queue: '+d.queued+' request'+(d.queued>1?'s':'');
}).catch(()=>{})},3000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
