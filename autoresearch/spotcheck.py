#!/usr/bin/env python3
"""
Manual quality spot-check: dump the unique facts a quant config extracts next to
the Q4 baseline facts, so a human can eyeball factual correctness (coverage +
groundedness are proxies; this is the eyeball check for aggressive quants).

Usage: python spotcheck.py <config_id>   # e.g. iq3xxs_win
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, candidates as C

cid = sys.argv[1] if len(sys.argv) > 1 else "iq3xxs_win"
cfgs = {c["id"]: c for b in C.BATCHES.values() for c in b}
cfg = cfgs[cid]
baseline = json.load(open(os.path.join(os.path.dirname(__file__), "baseline.json")))

res = H.run_config(cfg, baseline=baseline)  # manages its own server
H.stop_server()

def show(title, facts):
    print(f"\n===== {title} ({len(facts)} unique facts) =====")
    for i, f in enumerate(facts, 1):
        print(f"[{i}] {f.get('title','')}")
        print(f"     S/P/O: {f.get('subject','')} -> {f.get('predicate','')} -> {f.get('object','')}")
        ev = str(f.get('evidence_span', ''))[:140]
        print(f"     evidence: {ev}")

show(f"Q4 BASELINE", baseline["unique_facts_list"])
show(f"{cid}", res["unique_facts_list"])
print("\nmetrics:", json.dumps(res["metrics"], indent=2))
