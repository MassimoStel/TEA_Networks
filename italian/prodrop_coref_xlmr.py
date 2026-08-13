"""
XLM-R + LightGBM zero-subject (pro-drop) resolver for Italian TEA extraction.
Trained on CorefUD-Italian with leave-one-passage-out cross-validation.
"""
import os
import numpy as np
import joblib

_HERE = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_HERE, "coref_it_xlmr.joblib")
_ES_MODEL_PATH = os.path.join(_HERE, "coref_es_model.joblib")

_CLITICS = {"lo", "la", "li", "le", "gli", "mi", "ti", "ci", "vi", "si", "ne",
            "l'", "Lo", "La", "Li", "Le", "Gli", "Mi", "Ti", "Ci", "Vi", "Si"}


def _load_xlmr():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    model.eval()
    return tokenizer, model


def _encode_words(words, tokenizer, model):
    if not words:
        return np.zeros((0, 768), dtype=np.float32)
    import torch
    encoding = tokenizer(words, is_split_into_words=True,
                         return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoding)
    hidden = output.last_hidden_state[0].numpy()
    word_ids_list = encoding.word_ids()
    n_words = max(w for w in word_ids_list if w is not None) + 1
    word_embs = np.zeros((n_words, hidden.shape[1]), dtype=np.float32)
    counts = np.zeros(n_words, dtype=np.float32)
    for si, wi in enumerate(word_ids_list):
        if wi is not None:
            word_embs[wi] += hidden[si]
            counts[wi] += 1
    counts = np.maximum(counts, 1)
    word_embs /= counts[:, None]
    return word_embs


def _wide_context_emb(word_embs, idx, k=3):
    n = len(word_embs)
    start = max(0, idx - k)
    end = min(n, idx + k + 1)
    return word_embs[start:end].mean(axis=0)


def _featurize_hand(verb_person, verb_number, cand, rank, sent_dist):
    from coref_model import featurize
    return featurize(verb_person, verb_number, cand, rank, sent_dist)


class ProDropResolverXLMR:
    """XLM-R + LightGBM pro-drop resolver for Italian.

    Usage::

        resolver = ProDropResolverXLMR()
        resolver.encode_document(spacy_doc)
        # then pass resolver to extract_tea_it(..., resolver=resolver)
    """

    def __init__(self, model_path=None, lazy=True):
        self._model_path = model_path or _MODEL_PATH
        self._bundle = None
        self._es_model = None
        self._xlmr_tok = None
        self._xlmr_model = None
        self._word_embs = None
        self._words = None
        self._token_to_word_idx = {}
        self.candidates = []
        self._sent_boundaries = []
        if not lazy:
            self._load_all()

    def _load_bundle(self):
        if self._bundle is None:
            self._bundle = joblib.load(self._model_path)
        return self._bundle

    def _load_es(self):
        if self._es_model is None and os.path.exists(_ES_MODEL_PATH):
            self._es_model = joblib.load(_ES_MODEL_PATH)
        return self._es_model

    def _load_xlmr(self):
        if self._xlmr_tok is None:
            self._xlmr_tok, self._xlmr_model = _load_xlmr()
        return self._xlmr_tok, self._xlmr_model

    def _load_all(self):
        self._load_bundle()
        self._load_es()
        self._load_xlmr()

    def encode_document(self, doc):
        """Pre-compute XLM-R embeddings for the full spaCy document."""
        tok, mdl = self._load_xlmr()
        self._words = [t.text for t in doc]
        self._word_embs = _encode_words(self._words, tok, mdl)
        self._token_to_word_idx = {t.i: t.i for t in doc}
        self._sent_boundaries = []
        for sent in doc.sents:
            self._sent_boundaries.append((sent.start, sent.end))
        self.candidates = []

    def _sent_of(self, token_idx):
        for si, (s, e) in enumerate(self._sent_boundaries):
            if s <= token_idx < e:
                return si
        return 0

    def register_mention(self, tok):
        """Register a spaCy token as a candidate antecedent mention."""
        if tok.pos_ not in ("NOUN", "PROPN", "PRON"):
            return
        num = tok.morph.get("Number")
        number = num[0] if num else None
        per = tok.morph.get("Person")
        person = per[0] if per else None
        self.candidates.append({
            "form": tok.text,
            "lemma": tok.lemma_.lower(),
            "upos": tok.pos_,
            "number": number,
            "person": person,
            "is_subject": tok.dep_ in ("nsubj", "csubj", "nsubj:pass", "csubj:pass"),
            "word_idx": tok.i,
            "sent_idx": self._sent_of(tok.i),
            "is_clitic": tok.text in _CLITICS or (tok.pos_ == "PRON" and len(tok.text) <= 3),
            "is_relative": (tok.dep_ == "nsubj" and tok.pos_ == "PRON"
                            and tok.text.lower() in ("che", "cui", "quale", "quali")),
        })

    def _person_number(self, head):
        """Extract Person/Number from the verb or its auxiliary."""
        per = head.morph.get("Person")
        person = per[0] if per else None
        num = head.morph.get("Number")
        number = num[0] if num else None
        if person is None or number is None:
            for c in head.children:
                if c.dep_ in ("aux", "aux:pass", "cop"):
                    if person is None:
                        p = c.morph.get("Person")
                        if p:
                            person = p[0]
                    if number is None:
                        n = c.morph.get("Number")
                        if n:
                            number = n[0]
        return person, number

    def resolve(self, head, window=12):
        """Score candidate antecedents and return the best lemma, or None."""
        person, number = self._person_number(head)

        if person == "1":
            return "noi" if number == "Plur" else "io"
        if person == "2":
            return "voi" if number == "Plur" else "tu"

        if not self.candidates or self._word_embs is None:
            return None

        bundle = self._load_bundle()
        clf = bundle["lgb_model"]
        pca = bundle["pca"]
        scaler = bundle["scaler"]
        es_model = self._load_es()

        verb_idx = head.i
        verb_sent = self._sent_of(verb_idx)
        verb_emb = self._word_embs[verb_idx] if verb_idx < len(self._word_embs) else np.zeros(768)
        verb_wide = _wide_context_emb(self._word_embs, verb_idx, k=3)

        recent = list(reversed(self.candidates))[:window]
        if not recent:
            return None

        es_scores = []
        if es_model is not None:
            for rank, c in enumerate(recent):
                sd = verb_sent - c["sent_idx"]
                feat = _featurize_hand(person, number, c, rank, sd)
                es_scores.append(es_model.decision_function([feat])[0])
        else:
            es_scores = [0.0] * len(recent)
        es_scores = np.array(es_scores)
        es_max = es_scores.max() if len(es_scores) > 0 else 0.0

        cos_toks = []
        for c in recent:
            cidx = c["word_idx"]
            cand_emb = self._word_embs[cidx] if cidx < len(self._word_embs) else np.zeros(768)
            cos = float(np.dot(verb_emb, cand_emb) /
                        (np.linalg.norm(verb_emb) * np.linalg.norm(cand_emb) + 1e-9))
            cos_toks.append(cos)
        max_cos = max(cos_toks) if cos_toks else 0.0
        cos_sorted_ranks = np.argsort(cos_toks)[::-1]

        rows_hand = []
        rows_emb = []
        for rank, c in enumerate(recent):
            sd = verb_sent - c["sent_idx"]
            hand = _featurize_hand(person, number, c, rank, sd)
            cidx = c["word_idx"]
            cand_emb = self._word_embs[cidx] if cidx < len(self._word_embs) else np.zeros(768)
            cand_wide = _wide_context_emb(self._word_embs, cidx, k=2)

            cos_tok = cos_toks[rank]
            elem_tok = verb_emb * cand_emb
            cos_wide = float(np.dot(verb_wide, cand_wide) /
                             (np.linalg.norm(verb_wide) * np.linalg.norm(cand_wide) + 1e-9))
            elem_wide = verb_wide * cand_wide

            num_mismatch = 0.0
            cn = c.get("number")
            if number and cn and number != cn:
                num_mismatch = 1.0

            extra = [
                1.0 if c["is_clitic"] else 0.0,
                num_mismatch,
                1.0,
                0.0,
                1.0 if c["is_relative"] else 0.0,
                es_scores[rank],
                1.0 if es_scores[rank] == es_max else 0.0,
                cos_wide,
            ]

            word_dist = abs(verb_idx - c["word_idx"])
            is_propn = 1.0 if c["upos"] == "PROPN" else 0.0
            is_noun = 1.0 if c["upos"] == "NOUN" else 0.0
            is_pron = 1.0 if c["upos"] == "PRON" else 0.0
            is_full_np = 1.0 if c["upos"] in ("PROPN", "NOUN") else 0.0
            salience = is_propn * 1.0 + is_noun * 0.5
            log_word_dist = np.log1p(word_dist)
            n_cands = len(recent)
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

            full_hand = np.concatenate([np.array(hand, dtype=np.float32),
                                        np.array(extra, dtype=np.float32),
                                        [cos_tok],
                                        np.array(enhanced, dtype=np.float32)])
            rows_hand.append(full_hand)
            rows_emb.append(np.concatenate([elem_tok, elem_wide]).astype(np.float32))

        Xh = np.array(rows_hand)
        Xe = np.array(rows_emb)
        Xe_pca = pca.transform(scaler.transform(Xe))
        X = np.hstack([Xh, Xe_pca])

        scores = clf.predict_proba(X)[:, 1]

        valid = []
        for rank, c in enumerate(recent):
            if person == "3" and c.get("person") in ("1", "2"):
                continue
            cn = c.get("number")
            if number and cn and number != cn and c["upos"] != "PROPN":
                continue
            valid.append(rank)
        if not valid:
            valid = list(range(len(recent)))

        best = max(valid, key=lambda r: scores[r])
        chosen = recent[best]

        if chosen["upos"] in ("PROPN", "NOUN"):
            return chosen["lemma"]
        return chosen["form"].lower()


def update_history_xlmr(resolver, head, agent_str):
    """Register the subject of a clause as a future antecedent.

    This is the XLM-R resolver's version of prodrop_coref.update_history.
    It registers all relevant tokens from the clause for the mention window.
    """
    from tea_italian import _SUBJ_ACTIVE, _SUBJ_PASS
    subj_tok = next((c for c in head.children
                     if c.dep_ in _SUBJ_ACTIVE | _SUBJ_PASS), None)
    if subj_tok is not None:
        resolver.register_mention(subj_tok)
        for conj in subj_tok.conjuncts:
            resolver.register_mention(conj)
    for c in head.children:
        if c.dep_ in ("obj", "iobj", "obl") and c.pos_ in ("NOUN", "PROPN", "PRON"):
            resolver.register_mention(c)
