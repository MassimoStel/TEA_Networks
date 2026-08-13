"""
Validate the Italian TEA extractor (tea_italian.py) against a gold standard,
reusing the exact Precision/Recall/F1 definition of teanets.svo_validation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from tea_italian import get_it_nlp, extract_main_triple_it


def _norm(x):
    s = str(x).strip().lower()
    return "__none__" if s in {"", "nan", "none"} else s


def _prf(pred, gold):
    p = pred.apply(_norm); g = gold.apply(_norm)
    both = (p != "__none__") & (g != "__none__")
    tp = ((p == g) & both).sum()
    n_pred = (p != "__none__").sum(); n_gold = (g != "__none__").sum()
    P = tp / n_pred if n_pred else 0.0
    R = tp / n_gold if n_gold else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"TP": int(tp), "Pred": int(n_pred), "Gold": int(n_gold),
            "Precision": round(P, 3), "Recall": round(R, 3), "F1": round(F, 3)}


def run(gold_csv, show_diff=True):
    nlp = get_it_nlp()
    df = pd.read_csv(gold_csv).fillna("")
    rows = []
    for _, r in df.iterrows():
        a, e, t, pa, isp = extract_main_triple_it(nlp(str(r["sentence"])))
        rows.append({
            "sent": r["sentence"],
            "pa": a, "pe": e, "pt": t, "ppa": pa, "pisp": isp,
            "ga": r["agent"], "ge": r["event"], "gt": r["target"],
            "gpa": int(r["passive_approx"]), "gisp": int(r["is_passive"]),
        })
    ev = pd.DataFrame(rows)

    metrics = pd.DataFrame({
        "Agent":  _prf(ev["pa"], ev["ga"]),
        "Event":  _prf(ev["pe"], ev["ge"]),
        "Target": _prf(ev["pt"], ev["gt"]),
    }).T

    pass_ok = (ev["ppa"] == ev["gpa"]).sum()
    voice_ok = (ev["pisp"] == ev["gisp"]).sum()
    n = len(ev)

    print("=== Agent / Event / Target extraction (main clause) ===")
    print(metrics.to_string())
    print()
    print(f"Passive_approx accuracy : {pass_ok}/{n} = {pass_ok/n:.3f}")
    print(f"is_passive     accuracy : {voice_ok}/{n} = {voice_ok/n:.3f}")

    if show_diff:
        print("\n=== Mismatches ===")
        for _, r in ev.iterrows():
            diffs = []
            if _norm(r["pa"]) != _norm(r["ga"]): diffs.append(f"A:{r['pa']!r}!={r['ga']!r}")
            if _norm(r["pe"]) != _norm(r["ge"]): diffs.append(f"E:{r['pe']!r}!={r['ge']!r}")
            if _norm(r["pt"]) != _norm(r["gt"]): diffs.append(f"T:{r['pt']!r}!={r['gt']!r}")
            if r["ppa"] != r["gpa"]: diffs.append(f"PA:{r['ppa']}!={r['gpa']}")
            if diffs:
                print(f"[{r['sent'][:45]:45}] " + " | ".join(diffs))
    return metrics, ev


if __name__ == "__main__":
    gold = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "gold_standard_svo_it.csv")
    run(gold)
