#!/usr/bin/env python3
"""
Quality A/B: does a JSON-schema-driven guidance beat the natural-language prompt?

Fairness: the GRAMMAR constraint (response_format) is held identical across arms
(the generic FACT_SCHEMA -> all arms emit valid JSON). The only thing that varies
is HOW the extraction guidance is delivered to the model:
  A prose      : the repo's full natural-language DEFAULT_PROMPT (control)
  B schema_only: minimal instruction + a RICH JSON schema (per-field descriptions
                 + a coverage-strategy array description) pasted into the prompt
  C hybrid     : a concise strategy prose + the same RICH schema in the prompt

Quality is judgment, so we report objective proxies + dump facts for human eval:
  - unique_facts (dedup), schema_valid, groundedness (verbatim evidence_span)
  - specificity (object carries a number/year/proper-noun -> concrete vs generic)
  - predicate_diversity (unique predicates / facts)
  - cross-coverage between arms (do they extract the same information?)
Server = Q3_K_XL + winning MTP flags, fixed seeds, reused across arms.
"""
import os, sys, json, re, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H, candidates as C

SEEDS = [101, 202, 303]

# Rich schema: per-field guidance distilled from DEFAULT_PROMPT + a coverage
# strategy on the array. Pasted into the prompt (NOT used as the grammar).
RICH_SCHEMA = {
  "type": "object",
  "description": "Atomic knowledge-indicator (KI) facts extracted from the document.",
  "properties": {
    "facts": {
      "type": "array",
      "description": ("0-15 atomic facts. Long fact-dense docs warrant 8-15; short/"
        "generic docs 0-3. COVERAGE: extract a fact for EACH named person + their "
        "role, each named organisation + its relation, each concrete date + the "
        "event, each named location + what happened there, each distinctive detail "
        "(clothing/material/age/quantity), and each cross-entity relationship. "
        "Include secondary entities named only once. Favor specific (proper nouns, "
        "dates, numbers) over generic claims. Skip nav/login/error pages."),
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string", "description": "One natural sentence <=140 chars stating the fact, ending with the answer value when possible."},
          "description": {"type": "string", "description": "2-3 sentences <=350 chars carrying the answer + evidence (entity, relation, value, date/location/source) and a short inline verbatim quote when it disambiguates. Do not restate the title."},
          "subject": {"type": "string", "description": "Canonical entity name."},
          "predicate": {"type": "string", "description": "Precise snake_case relation <=32 chars (e.g. located_in, founded_by, position_held, born_in, published_article). Avoid the catch-all affiliated_with."},
          "object": {"type": "string", "description": "The value of the fact, plain prose."},
          "evidence_span": {"type": "string", "description": "Verbatim 1-3 sentence quote that is a substring of the document text."},
          "confidence": {"type": "integer", "description": "0-100 integer confidence."},
          "tags": {"type": "array", "items": {"type": "string"}, "description": "entity/topic/year tags, lowercase, alphanumeric+hyphen."}
        },
        "required": ["title","description","subject","predicate","object","evidence_span","confidence","tags"]
      }
    }
  },
  "required": ["facts"]
}

MINIMAL_PROMPT = ("You extract atomic knowledge-indicator facts from a document into a "
  "JSON object. Follow the JSON schema below exactly, including the guidance in each "
  "field's \"description\" and the array's coverage strategy. Output ONLY the JSON object.\n\n"
  "JSON schema:\n" + json.dumps(RICH_SCHEMA, indent=2))

CONCISE_STRATEGY = ("Extract atomic knowledge-indicator facts from the document into a JSON "
  "object. Bias toward dense, specific coverage: a fact for every named person+role, every "
  "organisation, every date+event, every location, every distinctive detail, and every "
  "cross-entity relationship -- including secondary entities named only once. Favor proper "
  "nouns/dates/numbers over generic claims; don't just cover the dominant entity; skip nav/"
  "login/error pages. Output ONLY a JSON object per the schema below.\n\n"
  "JSON schema:\n" + json.dumps(RICH_SCHEMA, indent=2))

ARMS = {
  "A_prose":       H.DEFAULT_PROMPT,
  "B_schema_only": MINIMAL_PROMPT,
  "C_hybrid":      CONCISE_STRATEGY,
}

PROPER = re.compile(r"\b\d|\b(19|20|21)\d{2}\b|\b[A-Z][a-zA-Z]+")
def specificity(facts):
    if not facts: return 0.0
    return sum(1 for f in facts if PROPER.search(str(f.get("object","")))) / len(facts)

def pred_diversity(facts):
    if not facts: return 0.0
    return len(set(f.get("predicate","") for f in facts)) / len(facts)

def run_arm(prompt):
    rounds, sv = [], True
    pred_n=pred_ms=prompt_n=prompt_ms=0.0; wall=0.0
    for s in SEEDS:
        p = H.build_payload(H.load_doc(), prompt, H.BASELINE_SAMPLING, s)  # grammar = generic FACT_SCHEMA
        content, t, w = H.call_round(p)
        facts, strict = H.parse_facts(content)
        if not strict: sv = False
        rounds.append(facts)
        pred_n += t.get("predicted_n",0); pred_ms += t.get("predicted_ms",0.0)
        prompt_n += t.get("prompt_n",0); prompt_ms += t.get("prompt_ms",0.0); wall += w
    uniq, total, dupes = H.dedup_unique(rounds)
    doc = H.load_doc()
    return {"unique": uniq, "n_unique": len(uniq), "total": total,
            "ground": round(H.groundedness(uniq, doc),3),
            "specificity": round(specificity(uniq),3),
            "pred_div": round(pred_diversity(uniq),3),
            "schema_valid": sv,
            "decode_tps": round(pred_n/pred_ms*1000,2) if pred_ms else 0,
            "predicted_tokens": int(pred_n),
            "prompt_tokens": int(prompt_n/len(SEEDS)),
            "prefill_tps": round(prompt_n/prompt_ms*1000,1) if prompt_ms else 0,
            "wall_3rounds_s": round(wall,1)}

if __name__ == "__main__":
    cfg = C.cfg("pe","pe", model=C.Q3_K_XL, **C.WIN)
    H.start_server(cfg["server_args"])
    res = {}
    for name, prompt in ARMS.items():
        print(f"=== running {name} ===", flush=True)
        res[name] = run_arm(prompt)
        m = res[name]
        print(f"  n_unique={m['n_unique']} ground={m['ground']} spec={m['specificity']} "
              f"pred_div={m['pred_div']} valid={m['schema_valid']} || "
              f"decode_tps={m['decode_tps']} prompt_tok={m['prompt_tokens']} "
              f"out_tok={m['predicted_tokens']} wall_3r={m['wall_3rounds_s']}s", flush=True)
    H.stop_server()
    # cross-coverage (does arm X recall arm Y's facts?)
    print("\n=== cross-coverage (row recalled-by col, info_text @0.80) ===")
    names = list(res)
    print("        " + "  ".join(f"{n:>13}" for n in names))
    for a in names:
        row = []
        for b in names:
            row.append(f"{H.coverage(res[a]['unique'], res[b]['unique']):.2f}")
        print(f"{a:>13} " + "  ".join(f"{v:>13}" for v in row))
    # dump
    for name in names:
        print(f"\n##### {name} unique facts (n={res[name]['n_unique']}) #####")
        for i,f in enumerate(res[name]["unique"],1):
            print(f"[{i}] {f.get('title','')}")
            print(f"     {f.get('subject','')} -> {f.get('predicate','')} -> {f.get('object','')}")
    out = {k:{kk:vv for kk,vv in v.items() if kk!='unique'} for k,v in res.items()}
    json.dump(out, open(os.path.join(os.path.dirname(__file__),"prompt_eval.json"),"w"), indent=2)
    print("\nwrote prompt_eval.json")
