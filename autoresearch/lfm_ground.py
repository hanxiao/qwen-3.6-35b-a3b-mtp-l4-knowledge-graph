#!/usr/bin/env python3
"""
Can an explicit verbatim-evidence instruction lift LFM2.5's groundedness?
LFM extracts good triples but paraphrases evidence_span (groundedness ~0.26).
The OG prompt asks for verbatim; LFM ignores it. Here we strengthen that one
instruction (emphatic 'copy character-for-character, never paraphrase, omit if
no exact quote') and re-measure. Everything else identical. LFM-recommended
sampling. Compares thinking vs nothink against the Qwen baseline.
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

SAMP = {"temperature": 0.2, "top_k": 80, "repeat_penalty": 1.05, "max_tokens": 8192}
SEEDS = [101, 202, 303]
BASE = "/models/LFM2.5-8B-A1B-Q4_K_M.gguf"
COMMON = ["--model", BASE, "--host", "0.0.0.0", "--port", "8080", "--ctx-size", "16384",
          "--flash-attn", "1", "--threads", "8", "--n-predict", "8192", "--jinja",
          "--n-gpu-layers", "999", "--cache-reuse", "256"]
ARMS = {"think": COMMON, "nothink": COMMON + ["--reasoning-budget", "0"]}

def strengthen(p):
    a = p.replace(
        "- evidence_span must be a verbatim substring of the doc text supplied above.",
        "- evidence_span MUST be copied CHARACTER-FOR-CHARACTER from the document text "
        "above: an exact substring with identical wording, numbers and punctuation. Do "
        "NOT paraphrase, summarize, rewrite, translate, or fix anything. If you cannot "
        "find an exact supporting quote in the document, OMIT the fact entirely.")
    b = a.replace(
        '"evidence_span": "<verbatim 1-3 sentence quote, substring of the doc text above>"',
        '"evidence_span": "<1-3 sentences COPIED VERBATIM as an exact substring of the '
        'document above -- never paraphrased or reworded>"')
    return b

PROMPT = strengthen(H.DEFAULT_PROMPT)
assert PROMPT != H.DEFAULT_PROMPT, "strengthen() matched nothing -- prompt text drifted"

def grounded_frac(facts, doc):
    norm = re.sub(r"\s+", " ", doc).lower()
    if not facts: return 0.0
    return sum(1 for f in facts if (lambda e: e and e in norm)(re.sub(r"\s+"," ",str(f.get("evidence_span",""))).lower().strip()))/len(facts)

if __name__ == "__main__":
    base = json.load(open(os.path.join(os.path.dirname(__file__), "baseline.json")))
    doc = H.load_doc()
    out = {}
    for name, args in ARMS.items():
        H.start_server(args, load_timeout=240)
        rounds = []
        for s in SEEDS:
            c, t, w = H.call_round(H.build_payload(doc, PROMPT, SAMP, s))
            f, strict = H.parse_facts(c); rounds.append(f)
        uniq, total, dupes = H.dedup_unique(rounds)
        g = round(H.groundedness(uniq, doc), 3)
        cov = round(H.coverage(uniq, base["unique_facts_list"]), 3)
        out[name] = {"unique": len(uniq), "ground": g, "cov": cov}
        print(f"=== {name} (strengthened verbatim prompt) ===")
        print(f"  unique={len(uniq)} groundedness={g} coverage={cov}", flush=True)
        # show whether evidence is now verbatim
        for x in uniq[:4]:
            ev = re.sub(r"\s+"," ",str(x.get("evidence_span",""))).lower().strip()
            vb = ev in re.sub(r"\s+"," ",doc).lower() if ev else "EMPTY"
            print(f"    verbatim={vb} ev={str(x.get('evidence_span',''))[:70]!r}")
    H.stop_server()
    print("\nSUMMARY vs OG-prompt LFM (think g0.269/cov0.952, nothink g0.25/cov0.857) & Qwen (g~0.56):")
    print(json.dumps(out, indent=2))
