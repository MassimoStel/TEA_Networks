"""
train_coref.py
==============

Reproducibly build the trained CorefUD coreference model used to compare
against the morphological pro-drop resolver.

Italian is absent from public CorefUD, so we train on Spanish-AnCora (Romance,
pro-drop, gold zero-subject mentions) and rely on language-universal UD
features for transfer. Downloads CorefUD-1.2-public (~88 MB) once, extracts the
Spanish files, trains a logistic-regression mention-ranker, evaluates on dev
(vs a recency baseline), and saves ``coref_es_model.joblib``.

Usage:
    python italian/train_coref.py
"""
import os, sys, zipfile, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")
import joblib
from coref_model import build_instances, train_model, evaluate, FEATURE_NAMES

# CorefUD 1.2 public edition, LINDAT bitstream (stable content URL)
COREFUD_URL = ("https://lindat.mff.cuni.cz/repository/server/api/core/"
               "bitstreams/7f737cb3-d8f4-4760-bb79-048768609ca0/content")
HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, ".corefud_cache")


def _ensure_data():
    os.makedirs(CACHE, exist_ok=True)
    train = os.path.join(CACHE, "es_ancora-corefud-train.conllu")
    dev = os.path.join(CACHE, "es_ancora-corefud-dev.conllu")
    if os.path.exists(train) and os.path.exists(dev):
        return train, dev
    zpath = os.path.join(CACHE, "CorefUD-1.2.zip")
    if not os.path.exists(zpath):
        print("Downloading CorefUD-1.2-public (~88 MB) ...")
        urllib.request.urlretrieve(COREFUD_URL, zpath)
    print("Extracting Spanish-AnCora ...")
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if "Spanish-AnCora" in n and n.endswith(".conllu"):
                open(os.path.join(CACHE, os.path.basename(n)), "wb").write(z.read(n))
    return train, dev


def main():
    train_path, dev_path = _ensure_data()
    print("Building instances (zero-subject, 3rd person) ...")
    tr = build_instances(train_path, third_person_only=True)
    dev = build_instances(dev_path, third_person_only=True)
    print(f"  train={len(tr)}  dev={len(dev)}")
    clf = train_model(tr)
    res = evaluate(clf, dev)
    print("\n=== Spanish-AnCora dev: trained model vs recency heuristic ===")
    print(res)
    print("\nLearned feature weights:")
    for name, val in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {name:14} {val:+.2f}")
    out = os.path.join(HERE, "coref_es_model.joblib")
    joblib.dump(clf, out)
    print("\nSaved", out)


if __name__ == "__main__":
    main()
