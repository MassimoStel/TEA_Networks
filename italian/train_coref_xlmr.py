"""
Train the XLM-R + LightGBM coreference model on CorefUD-Italian
and save artifacts as coref_it_xlmr.joblib.
"""
import os, sys, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "corefud_builder"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

HERE = os.path.join(os.path.dirname(__file__), "corefud_builder")

from train_xlmr_coref_v2 import (extract_passage_data, encode_words_xlmr,
                                  build_features_v2)
from coref_model import featurize

PINOCCHIO_PREFIXES = ("narr_pino", "narr_ciliegia", "narr_fata", "narr_grillo",
                      "narr_lucignolo", "narr_gatto_volpe", "narrativo")


def is_pinocchio(name):
    return any(name.startswith(p) for p in PINOCCHIO_PREFIXES)


def main():
    print("Loading XLM-R model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    print("Loading Spanish model...")
    es_model = joblib.load(os.path.join(os.path.dirname(HERE), "coref_es_model.joblib"))

    folders = ["corpora", "annotation_batch", "annotation_batch2", "annotation_batch3",
               "annotation_batch4", "annotation_batch7", "annotation_batch13",
               "annotation_batch14", "annotation_batch15"]

    all_hand, all_emb, all_y = [], [], []
    n_passages = 0
    n_instances = 0

    t1 = time.time()
    for folder in folders:
        fpath = os.path.join(HERE, folder)
        if not os.path.isdir(fpath):
            continue
        for cu in sorted(glob.glob(os.path.join(fpath, "*.conllu"))):
            name = os.path.basename(cu)[:-7]
            if is_pinocchio(name):
                continue
            words, instances = extract_passage_data(cu, third_person_only=True)
            resolvable = [i for i in instances if i["gold_idx"] is not None]
            if not resolvable:
                continue
            word_embs = encode_words_xlmr(words, tokenizer, model)
            rows = build_features_v2(resolvable, word_embs, es_model=es_model)

            row_idx = 0
            for ins in resolvable:
                n_cands = len(ins["candidates"])
                cos_toks = [rows[row_idx + r]["cos_tok"] for r in range(n_cands)]
                max_cos = max(cos_toks) if cos_toks else 0.0
                cos_sorted_ranks = np.argsort(cos_toks)[::-1]

                for rank, c in enumerate(ins["candidates"]):
                    r = rows[row_idx]
                    hand, extra, cos_tok = r["hand"], r["extra"], r["cos_tok"]

                    word_dist = abs(ins["verb_word_idx"] - c["word_idx"])
                    is_propn = 1.0 if c["upos"] == "PROPN" else 0.0
                    is_noun = 1.0 if c["upos"] == "NOUN" else 0.0
                    is_pron = 1.0 if c["upos"] == "PRON" else 0.0
                    is_full_np = 1.0 if c["upos"] in ("PROPN", "NOUN") else 0.0
                    salience = is_propn * 1.0 + is_noun * 0.5
                    log_word_dist = np.log1p(word_dist)
                    norm_rank = rank / max(n_cands - 1, 1)

                    cos_rank_idx = int(np.where(cos_sorted_ranks == rank)[0][0])
                    cos_rank_norm = cos_rank_idx / max(n_cands - 1, 1)
                    is_best_cos = 1.0 if cos_rank_idx == 0 else 0.0
                    cos_margin = cos_tok - max_cos

                    enhanced = [
                        float(word_dist), log_word_dist, salience, is_full_np,
                        norm_rank, float(n_cands),
                        cos_tok * is_propn, cos_tok * is_noun, cos_tok * is_pron,
                        cos_rank_norm, is_best_cos, cos_margin,
                    ]

                    full_hand = np.concatenate([hand, extra, [cos_tok], enhanced])
                    all_hand.append(full_hand.astype(np.float32))
                    all_emb.append(np.concatenate([r["elem_tok"], r["elem_wide"]]))
                    all_y.append(r["is_gold"])
                    row_idx += 1

            n_passages += 1
            n_instances += len(resolvable)

    elapsed = time.time() - t1
    print(f"\n{n_passages} passages, {n_instances} instances in {elapsed:.0f}s")

    Xh = np.array(all_hand)
    Xe = np.array(all_emb)
    y = np.array(all_y)

    PCA_DIM = 12
    ALPHA = 0.01

    scaler = StandardScaler().fit(Xe)
    nc = min(PCA_DIM, Xe.shape[0], Xe.shape[1])
    pca = PCA(n_components=nc).fit(scaler.transform(Xe))
    X = np.hstack([Xh, pca.transform(scaler.transform(Xe))])

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale = n_neg / max(n_pos, 1)

    clf = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale, verbosity=-1,
        reg_alpha=ALPHA, subsample=0.8, colsample_bytree=0.8,
    )
    clf.fit(X, y)

    out_path = os.path.join(os.path.dirname(HERE), "coref_it_xlmr.joblib")
    bundle = {
        "lgb_model": clf,
        "pca": pca,
        "scaler": scaler,
        "config": {
            "pca_dim": PCA_DIM,
            "alpha": ALPHA,
            "n_hand_features": Xh.shape[1],
            "n_emb_features": Xe.shape[1],
        },
    }
    joblib.dump(bundle, out_path)
    print(f"\nSaved model to {out_path}")
    print(f"  Hand features: {Xh.shape[1]}")
    print(f"  Embedding dim: {Xe.shape[1]} -> PCA {nc}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Training samples: {len(y)} ({int(n_pos)} pos, {int(n_neg)} neg)")


if __name__ == "__main__":
    main()
