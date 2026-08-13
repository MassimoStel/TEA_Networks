"""
coref_model.py
==============

A *trained* coreference model for pro-drop zero-subject antecedent selection,
built on CorefUD data, to be compared against the morphological resolver
(`prodrop_coref.ProDropResolver`).

Why Spanish/Catalan?  Italian is NOT in the public CorefUD (1.2 ships 15
languages, Italian absent). We therefore train on the typologically closest
available CorefUD datasets — Spanish-AnCora and Catalan-AnCora (Romance,
pro-drop, with gold *zero* subject mentions annotated as empty nodes) — using
only **language-universal UD features** (agreement, grammatical role, recency,
POS), so the model transfers to Italian via spaCy's UD annotations.

Task: given a clause with a dropped (zero) subject, rank the candidate
antecedents (recent contentful mentions) and pick the coreferent one. This is
exactly the decision the morphological resolver makes with a recency +
agreement heuristic; here we let a logistic-regression mention-ranker learn the
feature weights from gold data.

Pipeline:
    build_instances(conllu)  -> list of anaphor dicts (candidates + gold idx)
    featurize(anaphor, cand) -> feature vector (universal)
    train / evaluate on Spanish; recency baseline for comparison.
"""
import numpy as np


# --------------------------------------------------------------------------
# Universal feature extraction (works for udapi nodes AND for the lightweight
# candidate dicts we build from spaCy for Italian transfer).
# --------------------------------------------------------------------------
def featurize(verb_person, verb_number, cand, rank, sent_dist):
    """
    verb_person/number : morphology of the anaphor's governing verb (str/None)
    cand   : dict with keys number, person, upos, is_subject (bool)
    rank   : recency rank among candidates (0 = most recent)
    sent_dist : #sentences between candidate and anaphor
    """
    num_agree = 1.0 if (verb_number and cand["number"] == verb_number) else 0.0
    num_known = 1.0 if verb_number else 0.0
    cand_plur = 1.0 if cand["number"] == "Plur" else 0.0
    verb_plur = 1.0 if verb_number == "Plur" else 0.0
    return [
        1.0,                                   # bias
        num_agree,
        num_known,
        1.0 if cand["upos"] == "PROPN" else 0.0,
        1.0 if cand["upos"] == "NOUN" else 0.0,
        1.0 if cand["upos"] == "PRON" else 0.0,
        1.0 if cand["is_subject"] else 0.0,    # subject salience
        cand_plur,
        verb_plur,
        1.0 / (1.0 + rank),                    # recency (closer = larger)
        float(rank),
        1.0 / (1.0 + sent_dist),
        float(sent_dist),
        1.0 if rank == 0 else 0.0,             # is nearest
    ]


FEATURE_NAMES = ["bias", "num_agree", "num_known", "cand_PROPN", "cand_NOUN",
                 "cand_PRON", "cand_subject", "cand_plur", "verb_plur",
                 "recency", "rank", "sent_recency", "sent_dist", "is_nearest"]


# --------------------------------------------------------------------------
# Build antecedent-selection instances from a CorefUD .conllu file
# --------------------------------------------------------------------------
def _enh_subject_parent(node):
    """Return the governing verb of an empty node via enhanced deps, if it is
    a (c)subject; else None."""
    for d in (getattr(node, "deps", None) or []):
        dr = d.get("deprel", "") if isinstance(d, dict) else getattr(d, "deprel", "")
        pa = d.get("parent") if isinstance(d, dict) else getattr(d, "parent", None)
        if dr and dr.split(":")[0] in ("nsubj", "csubj"):
            return pa
    return None


def build_instances(conllu_path, window=12, third_person_only=True, max_docs=None):
    """
    Yield antecedent-selection instances for zero (empty-node) subjects.

    Each instance: dict(verb_person, verb_number, candidates=[cand,...],
                        gold_idx=int or None).
    A candidate is a contentful (non-empty) mention head occurring before the
    anaphor, within *window* recent mentions.
    """
    import udapi
    doc = udapi.Document(conllu_path)

    # global mention list with entity id and linear position
    mentions = []  # (pos, sent_idx, head_node, entity_id, is_empty)
    ent_of = {}
    for ent in doc.coref_entities:
        for m in ent.mentions:
            h = m.head
            if h is None:
                continue
            is_empty = not isinstance(h.ord, int)
            tree = h.root
            sent_idx = getattr(tree, "_sent_idx", None)
            mentions.append([h, ent.eid, is_empty, m])
    # assign sentence order
    order = {}
    for i, tree in enumerate(doc.trees):
        order[id(tree)] = i
    def pos_of(h):
        return (order.get(id(h.root), 0), float(h.ord))
    mentions.sort(key=lambda x: pos_of(x[0]))

    instances = []
    for idx, (h, eid, is_empty, m) in enumerate(mentions):
        if not is_empty:
            continue
        verb = _enh_subject_parent(h)
        if verb is None:
            continue
        vp = verb.feats.get("Person") or None
        vn = verb.feats.get("Number") or None
        if third_person_only and vp not in ("3", None):
            continue
        # candidates: contentful mentions before this anaphor, within window
        cands = []
        j = idx - 1
        seen = 0
        anaphor_sent = pos_of(h)[0]
        while j >= 0 and seen < window:
            ch, ceid, cempty, cm = mentions[j]
            if not cempty:
                csent = pos_of(ch)[0]
                cands.append({
                    "number": ch.feats.get("Number") or None,
                    "person": ch.feats.get("Person") or None,
                    "upos": ch.upos,
                    "is_subject": (ch.udeprel in ("nsubj", "csubj")),
                    "entity": ceid,
                    "sent_dist": anaphor_sent - csent,
                    "form": ch.form,
                })
                seen += 1
            j -= 1
        if not cands:
            continue
        gold_idx = next((k for k, c in enumerate(cands) if c["entity"] == eid), None)
        instances.append({"verb_person": vp, "verb_number": vn,
                          "candidates": cands, "gold_idx": gold_idx})
    return instances


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
def make_xy(instances):
    X, y = [], []
    for ins in instances:
        gi = ins["gold_idx"]
        if gi is None:
            continue
        for rank, c in enumerate(ins["candidates"]):
            X.append(featurize(ins["verb_person"], ins["verb_number"], c,
                               rank, c["sent_dist"]))
            y.append(1 if rank == gi else 0)
    return np.array(X), np.array(y)


def train_model(instances):
    from sklearn.linear_model import LogisticRegression
    X, y = make_xy(instances)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def _rank_pick(clf, ins):
    scores = [clf.decision_function([featurize(ins["verb_person"],
              ins["verb_number"], c, rank, c["sent_dist"])])[0]
              for rank, c in enumerate(ins["candidates"])]
    return int(np.argmax(scores))


def _recency_pick(ins):
    """Baseline mirroring the morphological resolver: nearest candidate that
    agrees in Number (strict for plural), else nearest."""
    vn = ins["verb_number"]
    for rank, c in enumerate(ins["candidates"]):
        if c["upos"] not in ("PROPN", "NOUN"):
            continue
        if vn == "Plur":
            if c["number"] == "Plur":
                return rank
        else:
            if c["number"] in (None, "Sing") or vn is None:
                return rank
    return 0


def evaluate(clf, instances):
    resolvable = [ins for ins in instances if ins["gold_idx"] is not None]
    n = len(resolvable)
    model_hit = sum(_rank_pick(clf, ins) == ins["gold_idx"] for ins in resolvable)
    rec_hit = sum(_recency_pick(ins) == ins["gold_idx"] for ins in resolvable)
    return {"n_resolvable": n,
            "trained_acc": round(model_hit / n, 3) if n else 0.0,
            "recency_acc": round(rec_hit / n, 3) if n else 0.0}
