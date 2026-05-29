#!/usr/bin/env python3
"""
Candidate config generator. Emits batch JSON files consumed by run.py.

A config = {id, desc, server_args, sampling, prompt?}.
server_args is the exact CLI token list passed to llama-server.

Central hypothesis for the L4 24GB bottleneck:
  22GB weights + KV cache + compute buffers > 24GB, so -fitt offloads tensors
  to CPU and the offloaded MoE experts make decode CPU-bound. Anything that
  frees VRAM (KV quantization, smaller ctx) should reduce offload and raise
  decode tok/s. MTP draft tuning changes speed without changing the output
  distribution (speculative sampling is distribution-preserving), so it is a
  safe speed-only lever.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

CANON = {
    "--ctx-size": "16384", "--parallel": "1", "--flash-attn": "1", "--no-mmap": "",
    "--threads": "8", "--spec-type": "draft-mtp", "--spec-draft-n-max": "2",
    "--n-predict": "8192", "--jinja": "", "--chat-template-file": "/templates/chat_template.jinja",
    "-fitt": "512", "--cache-reuse": "256",
}

def args(model=H.MODEL_PATH, **ov):
    d = dict(CANON); d.update(ov)
    out = ["--model", model, "--host", "0.0.0.0", "--port", "8080"]
    for k, v in d.items():
        if v is None:
            continue
        out.append(k)
        if v != "":
            out.append(str(v))
    return out

SAMP = dict(H.BASELINE_SAMPLING)

# winning serving flags from round 1 (confirmed +2.9%): carry into round-2 probes
WIN = {"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.1"}

def cfg(id, desc, model=H.MODEL_PATH, **ov):
    return {"id": id, "desc": desc, "server_args": args(model=model, **ov), "sampling": SAMP}

# ----- Batch 1: single-variable probes -----
BATCH1 = [
    cfg("kv_q8",       "KV cache q8_0 k+v (frees VRAM, tiny numeric change)",
        **{"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"}),
    cfg("kv_q8k_f16v", "KV k=q8_0 v=f16 (flash-attn likes f16 v)",
        **{"--cache-type-k": "q8_0"}),
    cfg("kv_q4",       "KV cache q4_0 k+v (max VRAM saving, higher risk)",
        **{"--cache-type-k": "q4_0", "--cache-type-v": "q4_0"}),
    cfg("mtp_n3",      "MTP draft n-max 3", **{"--spec-draft-n-max": "3"}),
    cfg("mtp_n4",      "MTP draft n-max 4", **{"--spec-draft-n-max": "4"}),
    cfg("mtp_n3_min1", "MTP n-max 3 n-min 1", **{"--spec-draft-n-max": "3", "--spec-draft-n-min": "1"}),
    cfg("ubatch1024",  "batch 2048 ubatch 1024", **{"--batch-size": "2048", "--ubatch-size": "1024"}),
    cfg("ctx12k",      "ctx 12288 (smaller KV -> less offload)", **{"--ctx-size": "12288"}),
]

def pcfg(id, desc, parallel_rounds=True, **ov):
    c = cfg(id, desc, **ov)
    c["parallel_rounds"] = parallel_rounds
    return c

# ----- Batch 2: parallelism (the 3 rounds are independent -> run concurrently).
# GPU util is only ~56% on single-stream decode, so concurrent rounds should
# raise aggregate task throughput. --parallel N splits ctx across N slots, so
# ctx-size is raised to keep >=8k tokens/slot (doc ~4.5k prompt + ~2.7k output).
BATCH2 = [
    pcfg("par2",       "parallel 2 slots, ctx16384 (8192/slot)", **{"--parallel": "2"}),
    pcfg("par3",       "parallel 3 slots, ctx24576 (8192/slot)",
         **{"--parallel": "3", "--ctx-size": "24576"}),
    pcfg("par3_q8",    "parallel 3 + KV q8 (VRAM-safe), ctx24576",
         **{"--parallel": "3", "--ctx-size": "24576", "--cache-type-k": "q8_0", "--cache-type-v": "q8_0"}),
    pcfg("par3_mtp3",  "parallel 3 + MTP n-max 3 (stack batch1 win), ctx24576",
         **{"--parallel": "3", "--ctx-size": "24576", "--spec-draft-n-max": "3"}),
]

# ----- Batch 3: MTP/spec-decode acceptance sweep (the only positive lever).
# Build default --spec-draft-n-max is 3 (repo overrode to 2). --spec-draft-p-min
# (default 0.0) gates which positions to draft: higher = draft only confident
# positions -> fewer rollbacks but fewer drafts. Sweep the trade-off. These are
# distribution-preserving (speed-only); coverage guard catches any RNG drift.
BATCH3 = [
    cfg("mtp_n2_ctrl",   "MTP n-max 2 (repo's override, control)", **{"--spec-draft-n-max": "2"}),
    cfg("mtp_n3_rc",     "MTP n-max 3 (build default; reconfirm +2%)", **{"--spec-draft-n-max": "3"}),
    cfg("mtp_n3_pmin01", "n3 + p-min 0.1", **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.1"}),
    cfg("mtp_n3_pmin03", "n3 + p-min 0.3", **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.3"}),
    cfg("mtp_n3_pmin05", "n3 + p-min 0.5", **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.5"}),
    cfg("mtp_n4_pmin03", "n4 + p-min 0.3", **{"--spec-draft-n-max": "4", "--spec-draft-p-min": "0.3"}),
    cfg("mtp_n5_pmin05", "n5 + p-min 0.5", **{"--spec-draft-n-max": "5", "--spec-draft-p-min": "0.5"}),
    pcfg("par3_clean",   "parallel 3, ctx36864 (12288/slot, no truncation) - clean close-out",
         **{"--parallel": "3", "--ctx-size": "36864"}),
]

# ----- Batch 4: lower-bit quant of the SAME model (bandwidth lever).
# Decode is memory-bandwidth bound, so fewer bits/weight should scale decode
# ~linearly. Q4_K_XL ~4.5bpw -> Q3_K_XL ~3.5bpw (UD = unsloth dynamic, keeps
# sensitive layers higher precision for quality retention). All at the round-1
# winning serving flags. Quality guard = coverage_of_baseline vs the Q4 baseline.
Q4 = H.MODEL_PATH
Q3_K_XL = "/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
BATCH4 = [
    cfg("q4_win",     "Q4_K_XL @ winning flags (same-session speed control)", model=Q4, **WIN),
    cfg("q3kxl_win",  "UD-Q3_K_XL @ winning flags (~3.5bpw, bandwidth lever)", model=Q3_K_XL, **WIN),
]

# ----- Batch 5: push the bit-width frontier below Q3 to find the quality cliff.
# Q3_K_XL (~3.5bpw) already gave +34% at coverage 1.0. Each step down = more
# speed (bandwidth) but rising quality risk. UD quants protect sensitive layers.
IQ3_XXS = "/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"  # ~3.1bpw
Q2_K_XL = "/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"  # ~2.7bpw
BATCH5 = [
    cfg("iq3xxs_win", "UD-IQ3_XXS @ win (~3.1bpw)", model=IQ3_XXS, **WIN),
    cfg("q2kxl_win",  "UD-Q2_K_XL @ win (~2.7bpw, aggressive)", model=Q2_K_XL, **WIN),
]

# ----- Batch 6: re-tune MTP draft depth at the NEW Q3 base.
# Win flags (n3,pmin0.1) were tuned on Q4. The Q3 target has different numerics
# and cheaper bytes/token, so deeper drafting (n4/n5) might now pay off where it
# didn't at Q4. Control = current winner. All on Q3_K_XL.
BATCH6 = [
    cfg("q3_n3_p01", "Q3 + n3 p-min0.1 (current winner, control)", model=Q3_K_XL, **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.1"}),
    cfg("q3_n4_p01", "Q3 + n4 p-min0.1", model=Q3_K_XL, **{"--spec-draft-n-max": "4", "--spec-draft-p-min": "0.1"}),
    cfg("q3_n5_p01", "Q3 + n5 p-min0.1", model=Q3_K_XL, **{"--spec-draft-n-max": "5", "--spec-draft-p-min": "0.1"}),
    cfg("q3_n3_p0",  "Q3 + n3 p-min0 (no gate)", model=Q3_K_XL, **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.0"}),
]

# ----- Batch 7: model-free n-gram speculative decoding (vs MTP) at Q3.
# Output is highly repetitive (8 schema keys x 15 facts) AND evidence_span is a
# verbatim quote from the in-context doc -> n-gram lookup can draft whole spans
# from the prompt at near-100% acceptance, at zero draft cost. Speculative
# decoding is distribution-preserving (any draft source) -> quality-safe.
# n-max raised to 8 to allow long copied runs.
def ngcfg(id, desc, spec_type, **extra):
    ov = {"--spec-type": spec_type, "--spec-draft-n-max": "8"}
    ov.update(extra)
    return cfg(id, desc, model=Q3_K_XL, **ov)
BATCH7 = [
    cfg("q3_mtp_ctrl", "Q3 + MTP n3 p0.1 (control)", model=Q3_K_XL, **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.1"}),
    ngcfg("q3_ngram_simple", "Q3 + ngram-simple draft (n-max 8)", "ngram-simple"),
    ngcfg("q3_ngram_cache",  "Q3 + ngram-cache draft (n-max 8)", "ngram-cache"),
    ngcfg("q3_ngram_mapk",   "Q3 + ngram-map-k draft (n-max 8)", "ngram-map-k"),
]

# ----- Batch 8: refine the 3.1-3.5bpw frontier. Q3_K_XL (3.5) preserves KIs;
# IQ3_XXS (3.1, i-quant) lost ~32% of KIs. Probe the band between for a quant
# that is faster than Q3_K_XL yet still preserves KI count. Q3_K_M is k-quant
# (like the KI-preserving Q3_K_XL) but smaller; IQ3_S is a higher-bpw i-quant.
Q3_K_M = "/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"   # ~3.3bpw, k-quant
IQ3_S  = "/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"    # ~3.44bpw, i-quant
BATCH8 = [
    cfg("q3km_win",  "UD-Q3_K_M @ win (~3.3bpw k-quant)", model=Q3_K_M, **WIN),
    cfg("iq3s_win",  "UD-IQ3_S @ win (~3.44bpw i-quant)", model=IQ3_S, **WIN),
]

# ----- Batch 9: MXFP4_MOE - is the ~33% bandwidth efficiency format-dependent?
# MXFP4 is 4bpw (more bytes than Q3's 3.5) but has native fast dequant on Ada
# (L4). To beat Q3 (75.9) it must be >14% more kernel-efficient. Tests whether
# the efficiency ceiling, not just bits, can be moved. Quality ~Q4-level.
MXFP4 = "/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
BATCH9 = [
    cfg("mxfp4_win", "MXFP4_MOE @ win (~4bpw, Ada-native kernel)", model=MXFP4, **WIN),
]

# ----- Batch 10: JSON-schema grammar overhead at Q3. The schema grammar is
# enforced per-token on CPU; at Q3's higher token rate that CPU overhead is a
# larger fraction of step time, so dropping response_format may speed decode.
# Quality-safe IFF the model still emits valid JSON unprompted (coverage guard).
# This changes the sampling/serving config, NOT the extraction prompt.
def nsc(id, desc, **ov):
    c = cfg(id, desc, model=Q3_K_XL, **ov); c["no_schema"] = True; return c
BATCH10 = [
    cfg("q3_schema_ctrl", "Q3 + win + schema (control)", model=Q3_K_XL, **WIN),
    nsc("q3_noschema",    "Q3 + win, NO json-schema grammar", **WIN),
]

# ----- Batch 11: IS MTP EVEN HELPING? (literature-driven, lossless)
# thc1006's RTX-3090 benchmark + MoESD (2505.19645) / Utility-Driven SD
# (2506.20675): for 3B-active MoE, expert-saturation threshold ~94 >> draft K,
# so the verify pass loads the expert union with no amortization -> spec-decode
# is often NET-NEGATIVE even at 100% acceptance. I tuned MTP but never tested
# turning it OFF. --spec-type none = pure autoregressive (exact/lossless). If it
# beats MTP, the repo's whole premise is wrong on this hardware.
def nospec(id, desc, model):
    return cfg(id, desc, model=model,
               **{"--spec-type": "none", "--spec-draft-n-max": None})
BATCH11 = [
    cfg("q4_mtp_ab",  "Q4 + MTP win (A/B control)", model=Q4, **WIN),
    nospec("q4_none", "Q4 + NO spec decoding (pure autoregressive)", Q4),
    cfg("q3_mtp_ab",  "Q3 + MTP win (A/B control = current winner)", model=Q3_K_XL, **WIN),
    nospec("q3_none", "Q3 + NO spec decoding (pure autoregressive)", Q3_K_XL),
]

BATCHES = {"batch1": BATCH1, "batch2": BATCH2, "batch3": BATCH3,
           "batch4": BATCH4, "batch5": BATCH5, "batch6": BATCH6,
           "batch7": BATCH7, "batch8": BATCH8, "batch9": BATCH9,
           "batch10": BATCH10, "batch11": BATCH11}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "batch1"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{name}.json")
    with open(out, "w") as f:
        json.dump(BATCHES[name], f, indent=2)
    print(f"wrote {out} ({len(BATCHES[name])} configs)")
