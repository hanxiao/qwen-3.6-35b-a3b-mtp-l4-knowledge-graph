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

## ROUND 2 cont. — levers tested at the Q3 base (no further gain)

- **MTP depth re-tune at Q3 (batch6):** n3 still optimal (n3 75.3, n4 74.2,
  n5 70.1 + quality drop); p-min 0 == 0.1 (~75.5). Same optimum as Q4 -> the
  n3+p-min0.1 flags transfer across quants. No gain.
- **Model-free n-gram speculative decoding (batch7): DEAD END.** Hypothesis was
  that repetitive JSON + verbatim evidence_span (copyable from the in-context
  doc) would give n-gram drafting near-100% acceptance. Reality: all variants
  far SLOWER than MTP -- ngram-cache 50.0, ngram-simple 52.5, ngram-map-k 54.6
  vs MTP 75.4. The trained MTP head (1 draft pass -> 3 good tokens) beats
  model-free lookup (longer drafts, lower acceptance, + lookup overhead).
  Quality fine throughout (cov 1.0), just no speed. (--spec-type also offers
  draft-eagle3 / draft-simple, but those need external draft weights this model
  doesn't ship.)

**Quant frontier fully mapped (batch8): k-quants preserve quality, i-quants don't.**

| quant | bpw | type | decode | KI | ground | verdict |
|---|---|---|---|---|---|---|
| Q4_K_XL | 4.5 | k | 58 | 22.6 | .60 | base |
| Q3_K_M | 3.3 | k | 74.3 | 23 | .565 | OK but slower than XL |
| **Q3_K_XL** | 3.5 | k(UD) | **75.9** | 22.8 | .67 | **WINNER** |
| IQ3_S | 3.44 | i | 78.8 | 23 | **.30** | REJECT (grounding) |
| IQ3_XXS | 3.1 | i | 78.4 | **15** | .57 | REJECT (KI count) |
| Q2_K_XL | 2.7 | k | 85 | 21 | **.26** | REJECT (grounding) |

i-quants degrade quality in different ways (IQ3_XXS drops KIs, IQ3_S/IQ2 drop
grounding) -> avoid for this task. Q3_K_M is lower-bpw but slower than the
UD-tuned Q3_K_XL. **Q3_K_XL is the speed-optimal quality-preserving quant.**

- **MXFP4_MOE (batch9): 59.9 t/s** -- above Q4 (58) but far below Q3 (75.9). At
  4bpw it tracks bytes, so the ~33% bandwidth efficiency is NOT meaningfully
  format/kernel-improvable on L4. Confirms bits/weight is the sole lever.

**Converged optimum: Q3_K_XL + `--spec-draft-n-max 3 --spec-draft-p-min 0.1`
= 75.9 t/s, +34%, zero quality loss.** Decode is bandwidth-bound at a fixed
~33% efficiency; quant bit-width is the only quality-safe knob and ~3.5bpw
k-quant is the floor.

- **JSON-schema grammar overhead (batch10): null.** Q3 no-schema 75.7 vs schema
  75.4 (within noise). The grammar mask is ~free at decode time, and it
  guarantees valid JSON -> keep it.

## LITERATURE-DRIVEN CHECK: is MTP even helping? (batch11) — YES, it's load-bearing

Prompted to dig the literature (MoESD arXiv 2505.19645, Utility-Driven SD for MoE
arXiv 2506.20675, and thc1006's 19-config RTX-3090 benchmark) which all report
spec-decode is NET-NEGATIVE for 3B-active MoE on consumer Ampere (expert-
saturation T_thres~94 >> draft K). I had tuned MTP but never tested OFF. So I
A/B'd `--spec-type none` (pure autoregressive, exact/lossless):

| config | decode t/s | MTP effect |
|---|---|---|
| Q4 + MTP | 58.3 | +13% |
| Q4 no-spec | 51.7 | |
| **Q3 + MTP** | **75.5** | **+39%** |
| Q3 no-spec | 54.3 | |

Two findings, both contradicting/refining the prior framing:
1. **MTP HELPS on L4 — opposite of the 3090 result.** On the fast 3090, forward
   passes are cheap so the expert-union verify overhead dominates -> spec loses.
   On bandwidth-starved L4 (~300 GB/s), passes are expensive so MTP's
   pass-amortization wins big. Hardware-class-dependent (matches thc1006's
   retraction of the "hardware-independent" claim).
2. **The quant win is realized THROUGH MTP, not independently.** Pure
   autoregressive is ~52-54 t/s REGARDLESS of Q4 vs Q3 (per-token-overhead-bound,
   not weight-bandwidth-bound at bs=1). Only WITH MTP does lighter Q3 translate
   to speed (58->75). MTP x low-bit quant are SYNERGISTIC. Earlier "pure
   bandwidth, bits is the lever" was incomplete: it's bits-realized-via-MTP.

So MTP is essential and Q3+MTP stands, now mechanistically understood.

### Spec-decode internals measured (Q3 + MTP n3, p-min 0.1)

From llama-server `timings` + `draft-mtp` stats on one round:
- decode 76.2 t/s, predicted_n 2802 in ~799 verify passes -> **3.5 tokens/pass**.
- draft_n 2397, draft_n_accepted 2005 -> **token acceptance 83.6%** (draft-round
  level 735/799 = 92%).

The MoE penalty, quantified: pure autoregressive = 1 tok/pass @ 54 t/s; MTP =
3.5 tok/pass @ 76 t/s. So MTP delivers 3.5x the tokens/pass but only 1.4x the
throughput -> each verify pass costs ~2.5x a normal pass (it loads the expert
union of the 3 drafted positions). That ~2.5x expert-union cost is the ceiling,
and it lives in the 35B target, not the draft.

Implications for "would a better draft / jump-forward help?":
- **Better draft: ~+10% ceiling.** At n=3, lifting acceptance 84% -> ~100% raises
  tokens/pass 3.5 -> ~3.85 (verify cost unchanged) ≈ +10% best case. Needs an
  EAGLE-3-class trained head (unsupported in llama.cpp) and the Q3 hidden-state
  shift fights it. Low ROI.
- **Jump-forward: redundant.** The 84% accepted tokens already include the
  grammar-forced structure; the 16% rejected are high-entropy fact *values* that
  jump-forward also can't force. So its benefit is a subset of MTP's. Plus it's
  not in llama.cpp.
- The only lever that would actually move the ceiling is cutting the per-verify
  expert-union cost (expert prefetch/caching across drafted positions, à la
  MoE-SpeQ / Utility-Driven SD) — a target-model/kernel change, not a config.

## SEARCH CONVERGED (rounds 1+2, ~31 experiments)

Quality-safe decode-rate levers are exhausted within the llama.cpp+MTP stack:
- WIN: Q3_K_XL quant (+34%) x MTP n3+p-min0.1 (+2%). Combined **56.5 -> 75.9 t/s**.
- Inert: KV quant, ctx size, ubatch, schema grammar, MXFP4.
- Harmful: --parallel (-10%, bandwidth split), mixed-KV (-57%), MTP n>=4.
- Quality-breaking (rejected): IQ3_XXS (-KIs), IQ3_S/Q2 (-grounding), sub-3.5bpw.
- Dead ends: n-gram drafting (MTP wins), Q3_K_M (slower than UD-Q3_K_XL).
Decode is bandwidth-bound at a hardware ~33% efficiency (format-independent).
Beyond this needs a quality tradeoff or a different engine (FP8 tensor cores +
grammar jump-forward, e.g. SGLang/vLLM) -- out of scope for the llama.cpp+MTP repo.

## Engine-pivot frontier: investigated, REJECTED (would regress)

Scoped vLLM/SGLang + xgrammar jump-forward as the one remaining quality-safe
lever. Conclusion: it would LOSE to the current setup on L4.
- Jump-forward's "up to 5x" is under batched load (TPOT); single-stream gain is
  only the fraction of grammar-forced JSON tokens (~1.2-1.6x, lossless).
- Switching engines forfeits the trained MTP head (our biggest lever) and the
  best available vLLM quant is AWQ/GPTQ 4-bit (~4.5bpw) vs our Q3 (3.5bpw) ->
  more bandwidth.
- Reported MoE decode: Qwen3.5-35B-A3B ~51 t/s (NVIDIA Spark), Qwen3-Next-80B-A3B
  -AWQ ~40 t/s (RTX 6000, > L4). On an L4, vLLM base would likely be ~40-55 t/s;
  even with jump-forward ~55-65 -- still BELOW our 75.9.
So llama.cpp + Q3_K_XL + MTP is the practical optimum for this model on an L4.
TERMINAL: +34% at zero quality loss; no quality-safe lever left that wins.

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

## MODEL SWAP: LiquidAI LFM2.5-8B-A1B (1B active) — REJECTED for this task

Swapped the model (kept the OG prompt + JSON schema), Q4_K_M, on the same L4.
LFM2.5 is a reasoning model; tested both modes:

| mode | decode t/s | out tok/round | wall/round | facts | triple fields | verdict |
|---|---|---|---|---|---|---|
| nothink (--reasoning-budget 0) | 131 | ~1200 | ~10s | 22/round but | **S/P/O/evidence ALL EMPTY** | useless KIs (dedup->1, ground 0) |
| thinking (default) | ~125 | ~7600 (25k-char reasoning) | ~61s | ~10 | filled, plausible | proper but FEWER, no speed win |

Findings:
- At 1B active, LFM2.5 CANNOT do the structured extraction without reasoning:
  nothink yields title-only shells (subject/predicate/object/evidence_span all
  ""), so every "fact" collapses (dedup 69->1, groundedness 0).
- With thinking on it fills the triples, but the 25k-char reasoning costs ~7600
  tok/round -> ~61s/round (≈ Qwen's wall, erasing the per-token speed edge) and
  yields only ~10 facts vs Qwen's ~22.
- Qwen3.6-35B-A3B Q3 wins on quality-per-second: 22 grounded facts, ~120s/3
  rounds, vs LFM-thinking ~10 facts, ~180s/3 rounds.
Caveats: used Qwen sampling (temp0.7, presence_penalty1.5) not LFM-recommended;
OG prompt is complex for a 1B-active model. But empty-triples is a capability
gap, not a sampling artifact. Conclusion: LFM2.5-8B-A1B is not a viable swap for
this structured KI task. (harness parse_facts now strips <think> for reasoning
models.)

## LFM2.5-8B-A1B — CORRECTED (read the model card; first run used wrong params)

My first LFM run was unfair: I used Qwen's sampling (temp0.7, **presence_penalty 1.5**)
+ forced nothink. The model card recommends **temp 0.2, top_k 80, repetition_penalty
1.05** and it's a reasoning model. presence_penalty 1.5 penalized the repeated
schema-key tokens -> empty triples. Re-ran with recommended sampling (3 seeds):

| arm (rec. sampling) | decode t/s | unique facts | coverage | groundedness | schema |
|---|---|---|---|---|---|
| nothink | 130.5 | 36 | 0.857 | 0.25 | some invalid |
| thinking | 127.5 | 26 | **0.952** | 0.269 | some invalid |

Corrected verdict:
- The empty-triples collapse was the presence_penalty, NOT a capability gap. With
  correct params LFM produces good, semantically-correct triples (release_date,
  parameter_count, scores) and thinking-mode recalls **95% of the Qwen baseline KIs**.
- Real remaining weakness: it **paraphrases evidence_span** instead of quoting
  verbatim (groundedness ~0.26 vs Qwen ~0.52) -> fails a strict verbatim-evidence
  requirement though the evidence content is correct. Plus occasional schema-invalid
  rounds.
- Speed: very high token rate (~130 t/s), but thinking generates ~5600 tok/round
  (~45s) so end-to-end ≈ Qwen; nothink is genuinely faster but lower coverage (0.857).
- Take: LFM2.5-8B-A1B (8B/1.5B-active, ~5GB) is a viable on-device extractor IF you
  accept descriptive (non-verbatim) evidence. For strict verbatim-grounded KIs,
  Qwen3.6-35B-A3B Q3 + MTP remains better. Lesson: always read the model card for
  sampling/template before judging a swap.
