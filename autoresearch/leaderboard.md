# Leaderboard

Baseline task_tps: **54.95** tok/s  |  decode_tps: **56.72**  |  unique facts: **21**

Ranked by task throughput (tokens/sec for the 3-round task). decode_tps = per-stream rate.

| rank | id | task tok/s | vs base | decode tok/s | par | unique | cov | ground | verdict | desc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | q2kxl_win | 82.27 | +49.7% | 85.41 | - | 19 | 1.0 | 0.263 | REJECT | UD-Q2_K_XL @ win (~2.7bpw, aggressiv |
| 2 | iq3xxs_win | 76.44 | +39.1% | 78.72 | - | 24 | 1.0 | 0.75 | KEEP | UD-IQ3_XXS @ win (~3.1bpw) |
| 3 | iq3s_win | 76.01 | +38.3% | 78.76 | - | 23 | 1.0 | 0.304 | REJECT | UD-IQ3_S @ win (~3.44bpw i-quant) |
| 4 | q3_noschema | 75.53 | +37.5% | 75.72 | - | 23 | 1.0 | 0.565 | KEEP | Q3 + win, NO json-schema grammar |
| 5 | q3_n3_p0 | 73.24 | +33.3% | 75.5 | - | 23 | 1.0 | 0.565 | KEEP | Q3 + n3 p-min0 (no gate) |
| 6 | q3_mtp_ab | 73.18 | +33.2% | 75.48 | - | 23 | 1.0 | 0.565 | KEEP | Q3 + MTP win (A/B control = current  |
| 7 | q3_schema_ctrl | 73.12 | +33.1% | 75.36 | - | 23 | 1.0 | 0.565 | KEEP | Q3 + win + schema (control) |
| 8 | q3kxl_win | 73.07 | +33.0% | 75.39 | - | 23 | 1.0 | 0.565 | KEEP | UD-Q3_K_XL @ winning flags (~3.5bpw, |
| 9 | q3_mtp_ctrl | 73.07 | +33.0% | 75.37 | - | 23 | 1.0 | 0.565 | KEEP | Q3 + MTP n3 p0.1 (control) |
| 10 | q3_n3_p01 | 73.04 | +32.9% | 75.33 | - | 23 | 1.0 | 0.565 | KEEP | Q3 + n3 p-min0.1 (current winner, co |
| 11 | q3km_win | 71.97 | +31.0% | 74.33 | - | 23 | 1.0 | 0.565 | KEEP | UD-Q3_K_M @ win (~3.3bpw k-quant) |
| 12 | q3_n4_p01 | 71.88 | +30.8% | 74.16 | - | 20 | 1.0 | 0.7 | KEEP | Q3 + n4 p-min0.1 |
| 13 | q3_n5_p01 | 67.92 | +23.6% | 70.13 | - | 21 | 1.0 | 0.143 | REJECT | Q3 + n5 p-min0.1 |
| 14 | mxfp4_win | 58.22 | +6.0% | 59.91 | - | 22 | 1.0 | 0.455 | KEEP | MXFP4_MOE @ win (~4bpw, Ada-native k |
| 15 | mtp_n3 | 57.58 | +4.8% | 57.58 | - | 20 | 1.0 | 0.6 | REJECT | MTP draft n-max 3 |
| 16 | mtp_n3_min1 | 57.56 | +4.7% | 57.56 | - | 20 | 1.0 | 0.6 | REJECT | MTP n-max 3 n-min 1 |
| 17 | ctx12k | 56.61 | +3.0% | 56.61 | - | 21 | 1.0 | 0.524 | KEEP | ctx 12288 (smaller KV -> less offloa |
| 18 | mtp_n4 | 56.61 | +3.0% | 56.61 | - | 23 | 1.0 | 0.478 | REJECT | MTP draft n-max 4 |
| 19 | kv_q8 | 56.4 | +2.6% | 56.4 | - | 22 | 1.0 | 0.636 | QUALITY_OK_NO_SPEEDUP | KV cache q8_0 k+v (frees VRAM, tiny  |
| 20 | q4_mtp_ab | 56.37 | +2.6% | 58.28 | - | 20 | 1.0 | 0.6 | KEEP | Q4 + MTP win (A/B control) |
| 21 | q4_win | 56.36 | +2.6% | 58.29 | - | 20 | 1.0 | 0.6 | KEEP | Q4_K_XL @ winning flags (same-sessio |
| 22 | kv_q4 | 56.22 | +2.3% | 56.22 | - | 22 | 1.0 | 0.545 | QUALITY_OK_NO_SPEEDUP | KV cache q4_0 k+v (max VRAM saving,  |
| 23 | mtp_n3_pmin01 | 56.06 | +2.0% | 58.01 | - | 20 | 1.0 | 0.6 | KEEP | n3 + p-min 0.1 |
| 24 | mtp_n3_pmin03 | 55.93 | +1.8% | 57.86 | - | 22 | 1.0 | 0.545 | KEEP | n3 + p-min 0.3 |
| 25 | mtp_n3_rc | 55.9 | +1.7% | 57.83 | - | 20 | 1.0 | 0.6 | KEEP | MTP n-max 3 (build default; reconfir |
| 26 | mtp_n2_ctrl | 55.21 | +0.5% | 57.02 | - | 21 | 1.0 | 0.524 | QUALITY_OK_NO_SPEEDUP | MTP n-max 2 (repo's override, contro |
| 27 | baseline | 54.95 | +0.0% | 56.72 | - | 21 | 1.0 | 0.524 | baseline | repo defaults (docker-compose.yml) |
| 28 | q3_ngram_mapk | 53.55 | +-2.5% | 54.62 | - | 24 | 1.0 | 0.583 | QUALITY_OK_NO_SPEEDUP | Q3 + ngram-map-k draft (n-max 8) |
| 29 | mtp_n4_pmin03 | 53.54 | +-2.6% | 54.99 | - | 27 | 1.0 | 0.63 | QUALITY_OK_NO_SPEEDUP | n4 + p-min 0.3 |
| 30 | mtp_n3_pmin05 | 53.45 | +-2.7% | 55.07 | - | 21 | 1.0 | 0.714 | QUALITY_OK_NO_SPEEDUP | n3 + p-min 0.5 |
| 31 | q3_none | 53.17 | +-3.2% | 54.28 | - | 23 | 1.0 | 0.652 | QUALITY_OK_NO_SPEEDUP | Q3 + NO spec decoding (pure autoregr |
| 32 | par2 | 51.51 | +-6.3% | 32.3 | Y | 20 | 1.0 | 0.7 | REJECT | parallel 2 slots, ctx16384 (8192/slo |
| 33 | q3_ngram_simple | 51.5 | +-6.3% | 52.46 | - | 21 | 1.0 | 0.714 | QUALITY_OK_NO_SPEEDUP | Q3 + ngram-simple draft (n-max 8) |
| 34 | mtp_n5_pmin05 | 50.85 | +-7.5% | 52.5 | - | 24 | 1.0 | 0.417 | REJECT | n5 + p-min 0.5 |
| 35 | q4_none | 50.64 | +-7.8% | 51.74 | - | 24 | 1.0 | 0.292 | REJECT | Q4 + NO spec decoding (pure autoregr |
| 36 | par3_mtp3 | 50.51 | +-8.1% | 18.5 | Y | 22 | 1.0 | 0.5 | REJECT | parallel 3 + MTP n-max 3 (stack batc |
| 37 | par3_q8 | 49.82 | +-9.3% | 17.97 | Y | 20 | 0.952 | 0.45 | REJECT | parallel 3 + KV q8 (VRAM-safe), ctx2 |
| 38 | par3 | 49.82 | +-9.3% | 18.19 | Y | 19 | 1.0 | 0.474 | REJECT | parallel 3 slots, ctx24576 (8192/slo |
| 39 | q3_ngram_cache | 49.11 | +-10.6% | 49.99 | - | 20 | 1.0 | 0.6 | QUALITY_OK_NO_SPEEDUP | Q3 + ngram-cache draft (n-max 8) |
| 40 | par3_clean | 49.03 | +-10.8% | 17.79 | Y | 21 | 1.0 | 0.333 | REJECT | parallel 3, ctx36864 (12288/slot, no |
| 41 | kv_q8k_f16v | 24.04 | +-56.3% | 24.04 | - | 26 | 1.0 | 0.5 | REJECT | KV k=q8_0 v=f16 (flash-attn likes f1 |
