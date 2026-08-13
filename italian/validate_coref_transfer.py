"""
Cross-lingual transfer of the trained CorefUD (Spanish-AnCora) zero-subject
mention-ranker to Italian, compared against the recency heuristic (the core of
the morphological resolver) on the SAME candidate pools.

Italian has no CorefUD gold, so we build antecedent-selection instances from
the hand-annotated Italian real-text gold (3 genres) and score both models with
`coref_model.evaluate`, which reuses the trained model and the recency baseline.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import joblib
from tea_italian import get_it_nlp, _SUBJ_ACTIVE, _SUBJ_PASS
from coref_model import evaluate, FEATURE_NAMES


def _mentions_in(doc, sent_idx):
    """Contentful mention heads (NOUN/PROPN) in a sentence, with UD features."""
    out = []
    for t in doc:
        if t.pos_ in ("NOUN", "PROPN"):
            num = t.morph.get("Number")
            out.append({
                "number": num[0] if num else None,
                "person": None,
                "upos": t.pos_,
                "is_subject": t.dep_ in (_SUBJ_ACTIVE | _SUBJ_PASS),
                "lemma": t.lemma_.lower(),
                "sent_idx": sent_idx,
            })
    return out


def _has_overt_subject(doc):
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return True
    return any(c.dep_ in (_SUBJ_ACTIVE | _SUBJ_PASS) for c in root.children)


def _match(gold_agent, lemma):
    g = str(gold_agent).lower()
    return lemma in g or g in lemma


def build_italian_instances(gold_csv, window=12):
    nlp = get_it_nlp()
    df = pd.read_csv(gold_csv).fillna("")
    instances = []
    pool = []          # running list of candidate mentions (most recent last)
    prev_doc = None
    sent_idx = 0
    for _, r in df.iterrows():
        cur_doc = r["doc"] if "doc" in df.columns else None
        if cur_doc != prev_doc:
            pool = []; sent_idx = 0; prev_doc = cur_doc
        doc = nlp(str(r["sentence"]))
        root = next((t for t in doc if t.dep_ == "ROOT"), None)

        is_prodrop = int(r.get("prodrop", 0)) == 1
        vp = root.morph.get("Person") if root is not None else None
        vp = vp[0] if vp else None
        vn = root.morph.get("Number") if root is not None else None
        vn = vn[0] if vn else None

        # a genuine 3rd-person zero subject (dropped, no overt nsubj)
        if is_prodrop and vp in ("3", None) and not _has_overt_subject(doc):
            cands = list(reversed(pool))[:window]
            for c in cands:
                c = c  # already dicts
            # recompute sent_dist relative to current sentence
            cands = [dict(c, sent_dist=sent_idx - c["sent_idx"]) for c in cands]
            gold_idx = next((k for k, c in enumerate(cands)
                             if _match(r["agent"], c["lemma"])), None)
            instances.append({"verb_person": vp, "verb_number": vn,
                              "candidates": cands, "gold_idx": gold_idx,
                              "agent": r["agent"], "sentence": r["sentence"]})

        pool.extend(_mentions_in(doc, sent_idx))
        sent_idx += 1
    return instances


if __name__ == "__main__":
    clf = joblib.load(os.path.join(os.path.dirname(__file__), "coref_es_model.joblib"))
    files = [os.path.join(os.path.dirname(__file__), f) for f in
             ("gold_realtext_it.csv", "gold_wiki_it.csv", "gold_firstperson_it.csv")]
    all_ins = []
    for f in files:
        all_ins += build_italian_instances(f)
    resolvable = [i for i in all_ins if i["gold_idx"] is not None]
    print("Italian 3rd-person zero-subject cases:", len(all_ins),
          " with antecedent in candidate pool:", len(resolvable))
    res = evaluate(clf, all_ins)
    print("\n=== Transfer to Italian (trained ES model vs recency) ===")
    print(res)
    # per-case detail
    from coref_model import _rank_pick, _recency_pick
    print("\nper-case:")
    for ins in resolvable:
        mp = _rank_pick(clf, ins); rp = _recency_pick(ins)
        pick = ins["candidates"][mp]["lemma"]; rpick = ins["candidates"][rp]["lemma"]
        print(f"  gold={ins['agent']!r:22} trained={pick!r:14} recency={rpick!r:14} | {ins['sentence'][:38]}")
