# KI Extractor

Extract structured Knowledge Indicators (KIs) from any document using a self-hosted LLM. Each KI is an atomic fact triple (subject, predicate, object) with evidence span, confidence score, and tags.

![demo](assets/demo.gif)

## What it does

1. Feed a URL or paste text
2. LLM extracts 0-15 structured facts per round, multiple rounds with different seeds for coverage
3. Real-time semantic dedup via [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) removes duplicate facts across rounds
4. Stream results as they arrive with live stats (tok/s, unique/total facts, context usage)

Inspired by the [Knowledge Indicators concept from Elastic](https://www.elastic.co/search-labs/blog/pre-computed-context-llm-agent-costs) for pre-computing agent context.

## Stack

| Component | What | Port |
|-----------|------|------|
| **llama-server** | [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA, serves Qwen3.6-35B-A3B via OpenAI-compatible API | 8080 |
| **ki-extractor** | FastAPI app with extraction UI + jina-v5-nano dedup on CPU | 3000 |

## Hardware

Single NVIDIA L4 24GB GPU (e.g. GCP `g2-standard-8`). The model runs in Q4_K_XL quantization with MTP (Multi-Token Prediction) speculative decoding.

## Quick start

```bash
git clone https://github.com/hanxiao/ki-extractor.git
cd ki-extractor

# Set your Jina API key (free at https://jina.ai/api-key, used for URL-to-markdown)
cp .env.example .env
# edit .env and add your JINA_API_KEY

# On a fresh GCP L4 instance, run the one-shot setup:
bash scripts/setup.sh
```

This downloads the model (~22GB), pulls Docker images, and starts both services.

Once running, open `http://<your-ip>:3000`.

## Manual setup

If you already have Docker + NVIDIA Container Toolkit:

```bash
# Download model
mkdir -p models
huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
    Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --local-dir models
mv models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf

# Start
docker compose up -d --build
```

## Configuration

### llama-server flags (in `docker-compose.yml`)

| Flag | Value | Why |
|------|-------|-----|
| `--ctx-size` | 16384 | Balance between input capacity and VRAM |
| `-fitt` | 512 | Auto-fit threshold, prevents OOM with MTP |
| `--spec-type draft-mtp` | — | MTP speculative decoding (+16% speed) |
| `--spec-draft-n-max` | 2 | Draft 2 tokens per step (sweet spot for L4) |
| `--cache-reuse` | 256 | KV cache reuse across rounds (40x prefill speedup on same doc) |
| `--flash-attn` | 1 | Flash attention |
| `--no-mmap` | — | Required when auto-fit offloads tensors to CPU |
| `--n-predict` | 8192 | Max generation length |

### Extraction parameters (in UI)

| Parameter | Default | What |
|-----------|---------|------|
| Rounds | 3 | Extraction passes with different seeds |
| Dedup model | jina-v5-nano | Runs on CPU, 23 texts/sec |
| Dedup field | Triple (S+P+O) | Compare subject+predicate+object |
| Dedup threshold | 0.90 | Cosine similarity cutoff |

### Key findings from benchmarking

- **nothink mode** required: thinking wastes ctx on reasoning tokens
- **MTP n=2** is the sweet spot: +16% vs no-MTP, n=4 only +3% with worse acceptance
- **JSON schema constraint**: <3% overhead, guarantees valid output
- **Multi-fact extraction** (0-15 facts/round): 4.4x faster than single-fact
- **KV cache reuse**: 40x prefill speedup on same-document subsequent rounds
- **fitt 256 + MTP = always OOM** on L4 24GB

## Cost

| Mode | $/hr | $/month |
|------|------|---------|
| GCP L4 standard | ~$0.86 | ~$620 |
| GCP L4 spot | ~$0.26 | ~$190 |

## File structure

```
├── app.py                  # FastAPI app (extraction logic + UI)
├── Dockerfile              # Container for ki-extractor
├── docker-compose.yml      # Both services
├── .env.example            # Environment variables template
├── templates/
│   └── chat_template.jinja # Qwen3.6 chat template (nothink mode)
├── scripts/
│   └── setup.sh            # One-shot GCP L4 setup
├── models/                 # Model files (gitignored, ~22GB)
└── assets/
    └── screenshot.png
```

## License

MIT
