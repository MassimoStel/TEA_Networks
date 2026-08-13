"""
Validate the Italian TEA extractor on a REAL text (public-domain narrative,
Collodi, *Le avventure di Pinocchio*, ch. 2), comparing Agent/Event/Target
extraction WITHOUT and WITH the pro-drop resolver (prodrop_coref).

Sentences are processed in text order so the resolver can use discourse
context (subject continuity across sentences).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from tea_italian import get_it_nlp, extract_main_triple_it
from prodrop_coref import ProDropResolver, extract_main_triple_resolved


def _norm(x):
    s = str(x).strip().lower()
    return "__none__" if s in {"", "nan", "none"} else s


def _prf(pred, gold):
    p = [_norm(x) for x in pred]; g = [_norm(x) for x in gold]
    both = [(a != "__none__" and b != "__none__") for a, b in zip(p, g)]
    tp = sum(a == b and c for a, b, c in zip(p, g, both))
    n_pred = sum(a != "__none__" for a in p); n_gold = sum(b != "__none__" for b in g)
    P = tp / n_pred if n_pred else 0.0
    R = tp / n_gold if n_gold else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"TP": tp, "Pred": n_pred, "Gold": n_gold,
            "P": round(P, 3), "R": round(R, 3), "F1": round(F, 3)}


def run(gold_csv):
    nlp = get_it_nlp()
    df = pd.read_csv(gold_csv).fillna("")

    # --- baseline: each sentence independently, no resolver ----------------
    base = [extract_main_triple_it(nlp(str(r["sentence"]))) for _, r in df.iterrows()]

    # --- with resolver: sentences in order, shared discourse state ---------
    # Reset discourse state at each new document (``doc`` column, if present).
    res = []
    resolver = ProDropResolver()
    prev_doc = None
    for _, r in df.iterrows():
        cur_doc = r["doc"] if "doc" in df.columns else None
        if cur_doc != prev_doc:
            resolver = ProDropResolver()
            prev_doc = cur_doc
        res.append(extract_main_triple_resolved(nlp(str(r["sentence"])), resolver))

    ga, ge, gt = df["agent"], df["event"], df["target"]
    docs = ", ".join(sorted(df["doc"].unique())) if "doc" in df.columns else os.path.basename(gold_csv)
    print("=== REAL TEXT [%s] (%d sentences) ===\n" % (docs, len(df)))
    print("Agent  — baseline :", _prf([b[0] for b in base], ga))
    print("Agent  — resolver :", _prf([b[0] for b in res],  ga))
    print("Event  — resolver :", _prf([b[1] for b in res],  ge))
    print("Target — resolver :", _prf([b[2] for b in res],  gt))

    # --- pro-drop focus ----------------------------------------------------
    pd_mask = df["prodrop"].astype(int) == 1
    idx = [i for i, m in enumerate(pd_mask) if m]
    base_hit = sum(_norm(base[i][0]) == _norm(ga[i]) for i in idx)
    res_hit = sum(_norm(res[i][0]) == _norm(ga[i]) for i in idx)
    print(f"\n--- PRO-DROP subset ({len(idx)} sentences) ---")
    print(f"Agent recovered  baseline: {base_hit}/{len(idx)}")
    print(f"Agent recovered  resolver: {res_hit}/{len(idx)}")
    print("\nper-sentence (pro-drop):")
    for i in idx:
        mark = "OK " if _norm(res[i][0]) == _norm(ga[i]) else "MISS"
        print(f"  [{mark}] gold={ga[i]!r:12} base={base[i][0]!r:12} res={res[i][0]!r:12} | {df['sentence'][i][:42]}")


if __name__ == "__main__":
    gold = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "gold_realtext_it.csv")
    run(gold)
