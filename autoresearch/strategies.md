# Strategy ledger

Running log of what worked (GOOD) and what didn't (BAD), with the reasoning.
Updated after every batch. The leaderboard (`leaderboard.md`) holds the numbers;
this file holds the *why*.

## Baseline

- Config: repo defaults from `docker-compose.yml`.
- **decode 56.46 tok/s** | prefill 1163 tok/s | wall(3 rounds) 148.9s
- **21 unique facts** (45 total, 24 dupes) | groundedness 0.524 | schema valid
- predicted 8143 tokens over 3 rounds (fixed seeds 101/202/303).
- Hardware: L4 24GB, VRAM ~21.8/22.5 GB used at ctx 16384 f16 KV (95% full ->
  the auto-fit (`-fitt 512`) is near its offload edge; freeing VRAM is the lever).

## ROUND 2 (quant unlocked) — BIG WIN

User relaxed the constraint: different quantization of the same model is allowed
if quality holds. Since decode is memory-bandwidth bound, fewer bits/weight ->
proportionally faster decode. Confirmed:

- **UD-Q3_K_XL (~3.5bpw) @ winning flags: decode 75.39 t/s** vs Q4_K_XL@win 58.29
  vs baseline 56.19 = **+34% over baseline**, coverage 1.0, 23 unique facts,
  ground 0.565. Single-run; reconfirming with repeats + frontier probe (IQ3/Q2).
  Scaling matches theory: 4.5/3.5 bpw ~= 1.29x, observed 75.4/58.3 = 1.29x.
- Bandwidth-bound thesis (round 1) now turned into the main lever: drop bits.
- Grammar jump-forward: dead end in llama.cpp (only masks logits, no jump-ahead;
  that's SGLang/vLLM/outlines). The rigid JSON already boosts MTP acceptance.

**Quant speed/quality frontier (single-run @ winning flags):**

| quant | bpw | decode t/s | Δ base | cov | ground | verdict |
|---|---|---|---|---|---|---|
| Q4_K_XL | 4.5 | 58.3 | +4% | 1.0 | 0.60 | baseline quality |
| Q3_K_XL | 3.5 | 75.4 | +34% | 1.0 | 0.565 | KEEP |
| **IQ3_XXS** | 3.1 | **78.7** | **+40%** | 1.0 | 0.75 | **KEEP (best)** |
| Q2_K_XL | 2.7 | 85.4 | +52% | 1.0 | **0.263** | REJECT (cliff) |

**Cliff at Q2 (2.7bpw):** decode keeps rising but groundedness collapses to 0.26
-- the model fabricates/paraphrases evidence_span instead of quoting verbatim.
Key: coverage stayed 1.0 even for broken Q2, so coverage ALONE would have passed
it; groundedness is the guard that caught the degradation. IQ3_XXS (3.1bpw) is
the speed-optimal quality-preserving point. Confirming with 5 repeats + manual
fact spot-check vs Q4.

## WINNER (round 1, 5-repeat confirmed)

**`--spec-draft-n-max 3 --spec-draft-p-min 0.1`**: decode **57.84 +- 0.59 t/s
vs baseline 56.19 +- 0.29 = +2.9%**, task_tps 57.4 vs 55.8. coverage_of_baseline
= 1.0 on ALL 5 repeats (baseline's own self-coverage is only 0.952 -> zero
quality loss). Applied to docker-compose.yml. See REPORT.md.

This is the only quality-preserving lever that moved decode rate. Everything
else was inert or harmful (see BAD). Decode is memory-bandwidth bound at
~56-58 t/s single-stream on L4 -- that is the practical ceiling.

## GOOD strategies (kept)

- **`--spec-draft-n-max 3` + `--spec-draft-p-min 0.1`** is the decode peak.
  batch3 single-run sweep (decode t/s): n2 57.0 | n3 57.8 | n3+pmin0.1 **58.0** |
  n3+pmin0.3 57.9 | n3+pmin0.5 55.1 | n4 55.0 | n5 52.5. Coverage 1.0 throughout.
  NOTE: n-max 3 is this llama.cpp build's DEFAULT; the repo overrode it to 2,
  leaving ~+2% on the table. p-min 0.1 adds a hair more. Confirming with repeats.

## BAD strategies (rejected)

- **KV quant for speed (H1): FALSE.** kv_q8 56.4, kv_q4 56.22 == baseline 56.46.
  Root cause: KV is tiny here (~15.6 KB/token; ~256 MiB for ctx 16384). The L4
  VRAM (95% full) is consumed by the ~20GB model weights + compute buffers, NOT
  KV. So freeing KV frees nothing useful and decode stays ~56 t/s. The model is
  effectively fully on GPU; decode is GPU compute/bandwidth bound, not offload
  bound. This kills the original "reduce CPU offload" thesis.
- **Smaller ctx (H3): inert.** ctx12k 56.61 == baseline. Same reason as above.
- **Mixed-precision KV (kv_q8k_f16v): -57% (24.04 t/s).** k=q8_0 + v=f16
  disables the flash-attn fast path -> attention falls to a slow kernel. Never
  mix KV precisions. Quantize both k and v together or neither.
- **ubatch1024 (batch 2048 / ubatch 1024): server crash** ("remote end closed"),
  and irrelevant to single-stream decode anyway.
- **MTP n=4: groundedness/quality wobble, no speed gain** (56.61). n=3 is the
  better operating point on this build.
- **Parallelism / continuous batching (H5): FALSE, decisively.** Running the 3
  independent rounds concurrently via `--parallel N` REDUCES task throughput:
  par2 51.5, par3 49.8, par3_q8 49.8 -- all below the 54.95 sequential baseline.
  Per-stream decode collapses 56 -> 32 (2 streams) -> 18 (3 streams), i.e. near
  pure 1/N split with no aggregate gain. The ~56% GPU-util reading was
  misleading: bs=1 MoE decode already saturates L4 memory bandwidth, so adding
  streams just shares the same bandwidth. Bonus failure mode: ctx/N per-slot
  truncates long JSON -> schema-invalid. Memory bandwidth is the hard wall;
  decode ~56 t/s is near the practical single-stream limit for this model on L4.
  CONFIRMED airtight: par3_clean (ctx 36864, 12288/slot, zero truncation,
  coverage 1.0) still gives task_tps 49.0 / per-stream 17.8 -- so the regression
  is real batching contention, not a ctx artifact.
- **MTP n>=4 and p-min>=0.5: slower.** n4 55.0, n5 52.5, n3+pmin0.5 55.1. Too
  many draft tokens (low acceptance, wasted compute) or too-aggressive gating
  (too few drafts). n3 + p-min 0.0-0.1 is the operating point.

## Methodology learnings (cont.)

- **Decode-rate noise floor is ~+-1 t/s (~2%).** The n2 control measured 57.0
  while the identical-config baseline measured 56.5 -- a 0.5 t/s gap from pure
  run-to-run variance. Since the best candidate's gain (~+1.5 t/s) is the same
  order as the noise, single-run comparisons are untrustworthy. Final claims use
  confirm.py: 5 repeats per config, distinct seed-triple each, report mean+-std.

## Methodology learnings

- **Quality guard fix:** raw unique-fact count swings 20-26 across equivalent
  configs purely from temp-0.7 RNG-path differences (changing MTP n changes the
  sampling RNG path -> different but equally valid facts). `coverage_of_baseline`
  (semantic recall of every baseline fact) was 1.0 for ALL configs -> it is the
  trustworthy "no loss" gate. Switched guard to coverage-primary; unique-count is
  only a catastrophic-collapse floor. Groundedness (verbatim evidence_span match)
  is brittle (0.48-0.64 across equivalents) -> loose floor only.
- Decode rate is remarkably stable (56.2-57.6) -> the lever is NOT per-stream
  decode flags but **task-level parallelism** (aggregate throughput).

## Open hypotheses (to test)

- H5: `--parallel N` + concurrent rounds raises aggregate task_tps (batch2).
- H6: stack parallel + mtp_n3.
- H7: if parallel helps, push --parallel to 4-6 with continuous batching.
