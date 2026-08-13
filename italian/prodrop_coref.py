"""
prodrop_coref.py
================

Zero-subject (pro-drop) resolver for Italian TEA extraction.

Motivation
----------
Italian is a pro-drop language: the subject pronoun is routinely omitted
because person/number are encoded on the verb ("Inseguì il topo" = "(He/she)
chased the mouse"). There is then no ``nsubj`` token to fill the Agent role,
and — as quantified in VALIDATION_REPORT_IT.md — Agent-node coverage collapses.

A neural Italian coreference resolver that predicts *zero mentions* (the empty
subjects of pro-drop, annotated in CorefUD) is **not available as an
installable package**: maverick-coref ships only English models
(OntoNotes/LitBank/PreCo), stanza/fastcoref coref are English-only, and the
CorefUD multilingual systems (CorPipe, …) are research code, not on PyPI.

This module therefore implements a *linguistically grounded* pro-drop resolver
that recovers the dropped subject from:

  1. **Verb morphology** — Person & Number features (from the head or its
     auxiliary/copula). 1st/2nd person are deictic → io/tu/noi/voi.
  2. **Subject continuity** — for 3rd person, the antecedent is the most recent
     overt nominal subject (PROPN/NOUN) that agrees in Number. This is the
     dominant discourse pattern of Italian narrative (topic/subject
     continuity), so recency-based resolution is a strong baseline.

It is applied on a running document context: subjects (overt or resolved) are
registered as they are read, so continuity chains propagate across sentences.
"""


class ProDropResolver:
    def __init__(self):
        # antecedent history: list of dicts {lemma, pos, number}
        self.history = []

    # -- morphology ---------------------------------------------------------
    @staticmethod
    def _feat(head, feat):
        vals = head.morph.get(feat)
        if vals:
            return vals[0]
        for c in head.children:
            if c.dep_ in {"aux", "aux:pass", "cop"}:
                v = c.morph.get(feat)
                if v:
                    return v[0]
        return None

    def person_number(self, head):
        return self._feat(head, "Person"), self._feat(head, "Number")

    # -- history ------------------------------------------------------------
    def register(self, lemma, pos, number):
        if lemma:
            self.history.append({"lemma": lemma.lower(), "pos": pos, "number": number})

    # -- resolution ---------------------------------------------------------
    def resolve(self, head, window=5):
        """
        Return the inferred subject lemma for a clause *head* that has no overt
        subject, or None if it cannot be recovered.

        3rd person uses subject continuity: the most recent overt nominal
        (PROPN/NOUN) antecedent within a recency *window* that agrees in Number.
        A plural head accepts only a plural antecedent (proper nouns carry no
        Number feature and must not fill a plural drop) — otherwise it falls
        back to "loro".
        """
        person, number = self.person_number(head)
        if person == "1":
            return "noi" if number == "Plur" else "io"
        if person == "2":
            return "voi" if number == "Plur" else "tu"
        # 3rd person (or unmarked): scan the recency window
        for cand in list(reversed(self.history))[:window]:
            if cand["pos"] not in {"PROPN", "NOUN"}:
                continue
            if number == "Plur":
                if cand["number"] == "Plur":
                    return cand["lemma"]
            else:  # singular or unmarked head
                if cand["number"] in (None, "Sing") or number is None:
                    return cand["lemma"]
        if number == "Plur":
            return "loro"
        return None


def _root(doc):
    return next((t for t in doc if t.dep_ == "ROOT"), None)


def _overt_subject_token(head):
    from tea_italian import _SUBJ_ACTIVE, _SUBJ_PASS
    return next((c for c in head.children
                 if c.dep_ in _SUBJ_ACTIVE | _SUBJ_PASS), None)


_DEICTIC = {"io", "tu", "noi", "voi", "loro", "lui", "lei"}


def update_history(resolver, head, agent_str):
    """
    Register the subject of a clause *head* as a future antecedent, using the
    overt grammatical subject token when present, otherwise the (resolved)
    ``agent_str``. Deictic pronouns are stored as PRON so they do not act as
    nominal antecedents for later 3rd-person continuity.
    """
    from tea_italian import compound_phrase
    subj_tok = _overt_subject_token(head)
    if subj_tok is not None:
        phrase = compound_phrase(subj_tok)
        pos = subj_tok.pos_
        num = subj_tok.morph.get("Number")
        number = num[0] if num else None
    elif agent_str:
        phrase = agent_str
        _, number = resolver.person_number(head)
        pos = "PRON" if agent_str in _DEICTIC else (
            "PROPN" if agent_str[:1].isupper() else "NOUN")
    else:
        return
    # dates/numbers are never discourse subjects: do not let a temporal that
    # the parser mislabelled as a subject poison subject continuity.
    if any(ch.isdigit() for ch in phrase):
        return
    resolver.register(phrase, pos, number)


def extract_main_triple_resolved(doc, resolver):
    """
    Like tea_italian.extract_main_triple_it, but when the main clause has no
    overt subject (pro-drop), fill the Agent from *resolver* and update the
    resolver's antecedent history. Call sentences in text order.
    """
    from tea_italian import extract_main_triple_it, _passive_info_it

    a, e, t, pa, isp = extract_main_triple_it(doc)
    head = _root(doc)
    if head is None:
        return (a, e, t, pa, isp)

    pinfo = _passive_info_it(head)
    if a is None:
        # only resolve genuine active pro-drop (not passive-with-agent)
        if not (pinfo["is_passive"] and pinfo["has_agent"]):
            r = resolver.resolve(head)
            if r is not None:
                a = r

    # topic = grammatical subject (patient for passive-with-agent)
    topic = t if (pinfo["is_passive"] and pinfo["has_agent"]) else a
    update_history(resolver, head, topic)
    return (a, e, t, pa, isp)
