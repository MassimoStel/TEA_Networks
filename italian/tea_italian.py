"""
tea_italian.py
==============

Italian adaptation of the TEA_Networks SVO / Agent-Event-Target extraction
pipeline (https://github.com/MassimoStel/TEA_Networks).

The upstream library is English-only: the core `teanets.svo_extraction`
module is written against spaCy's English (ClearNLP) dependency scheme
(`dobj`, `pobj`, `prep`, `agent`, `attr`, `acomp`, `oprd`, `dative`,
`npadvmod`) and English-specific tags (`VBN`, `VB`), it computes node
valence with the English VADER lexicon, and it resolves synonymy with the
English WordNet.

This module re-implements the same *semantics* (who did what to whom, with
the TEA passive-voice policy: semantic agent -> Agent role, patient ->
Target role; patient-in-Agent approximation when the agent is implicit) on
top of the Italian model `it_core_news_lg`, which uses the Universal
Dependencies (UD) scheme:

    active object            obj
    prepositional/oblique    obl        (+ case marker child)
    passive subject/patient  nsubj:pass
    passive auxiliary         aux:pass   (essere / venire)
    passive by-agent         obl:agent  (case "da")
    copula                   cop
    impersonal/passivante si expl:impers

Language-specific resources:
    * node valence  -> teanets.valence.italian lexicon (VADER is EN-only)
    * synonymy      -> Italian WordNet via Open Multilingual WordNet
                       (nltk wn.synsets(..., lang='ita'))

Known limitations (documented, validated in validate_it.py):
    * Coreference resolution (fastcoref / stanza) is English-only -> off.
    * Pro-drop: Italian omits subject pronouns; when the subject is not
      lexicalised there is no `nsubj` to extract (agent -> None). This is a
      genuine property of the language, quantified in the validation.
    * it_core_news_lg has no transformer variant; short agentless passives
      (e.g. "La finestra fu rotta") are sometimes mis-parsed as copular.
"""

from functools import lru_cache
from itertools import combinations

import spacy
import pandas as pd

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_NLP = None
_MODEL_NAME = "it_core_news_lg"


def get_it_nlp():
    """Return a process-wide singleton it_core_news_lg pipeline."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(_MODEL_NAME)
    return _NLP


def itnlp(text):
    return get_it_nlp()(text)


# ---------------------------------------------------------------------------
# UD label sets (Italian)
# ---------------------------------------------------------------------------
_SUBJ_ACTIVE = {"nsubj", "csubj"}
_SUBJ_PASS = {"nsubj:pass", "csubj:pass"}
_OBJ_DIRECT = {"obj"}          # iobj handled as fallback (clitic datives)
_AUX_PASS = {"aux:pass"}
_AGENT = {"obl:agent"}
# clitic "si" impersonale / passivante
_IMPERS = {"expl:impers", "expl:pass"}
# negations (ADV) whose lemma we keep in the verb phrase
_NEG_LEMMAS = {"non"}

# Surface forms / lemmas of 'essere' and 'venire' (the periphrastic-passive
# auxiliaries). it_core_news_lg frequently mislemmatises these (e.g. "fui",
# "fu") and mis-tags them as plain ``aux``/``cop`` instead of ``aux:pass``,
# so we also match the raw surface forms.
_ESSERE_VENIRE = {
    "essere", "venire",
    "sono", "sei", "è", "siamo", "siete",
    "fui", "fosti", "fu", "fummo", "foste", "furono",
    "ero", "eri", "era", "eravamo", "eravate", "erano",
    "sarò", "sarai", "sarà", "saremo", "sarete", "saranno",
    "sarei", "saresti", "sarebbe", "fosse", "fossi", "fossero", "stato", "stata",
    "vengo", "vieni", "viene", "veniamo", "vengono",
    "venni", "venisti", "venne", "venimmo", "vennero",
    "verrò", "verrà", "verranno", "veniva", "venivano", "venga",
}

# Unaccusative / motion / copular verbs that take 'essere' in the ACTIVE
# perfect ("sono andato" = I went, NOT passive). Their past participle with an
# 'essere' aux must NOT be read as a passive.
_UNACCUSATIVE = {
    "andare", "venire", "arrivare", "partire", "tornare", "ritornare",
    "entrare", "uscire", "salire", "scendere", "cadere", "nascere", "morire",
    "restare", "rimanere", "diventare", "divenire", "stare", "essere",
    "riuscire", "sembrare", "apparire", "scomparire", "sparire", "crescere",
    "succedere", "accadere", "avvenire", "esistere", "durare", "giungere",
    "fuggire", "scappare", "piacere", "dispiacere",
}
_REFLEXIVE_CLITICS = {"mi", "ti", "ci", "vi"}
# Concessive / temporal prepositions introducing circumstantial adjuncts, not
# verb arguments; excluded from the oblique-object fallback.
_ADJUNCT_PREPS = {"nonostante", "malgrado", "durante", "dopo", "prima",
                  "mediante", "tramite", "secondo", "oltre", "salvo",
                  "eccetto", "tranne", "verso", "circa"}


# ---------------------------------------------------------------------------
# Valence  (Italian lexicon; VADER is English-only)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _valence_sets():
    from teanets.valence.italian import _positive, _negative, _ambivalent
    pos = {w.lower() for w in _positive}
    neg = {w.lower() for w in _negative}
    amb = {w.lower() for w in _ambivalent}
    return pos, neg, amb


def compute_valence_it(text):
    """
    Lexicon-based valence for an Italian phrase, mirroring the ternary
    positive/negative/neutral output of `teanets.nlp_utils.compute_valence`.

    A word matches if its surface form or lemma is in the positive/negative
    lexicon; a preceding negation ("non ...") flips the polarity. Score is the
    signed count of polarity words; |score| ties resolve to 'neutral'.
    """
    pos, neg, _amb = _valence_sets()
    doc = get_it_nlp()(text)
    score = 0
    negate = False
    for tok in doc:
        low = tok.text.lower()
        lem = tok.lemma_.lower()
        if lem in _NEG_LEMMAS:
            negate = True
            continue
        s = 0
        if low in pos or lem in pos:
            s = 1
        elif low in neg or lem in neg:
            s = -1
        if s and negate:
            s = -s
        score += s
        if s:
            negate = False
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Synonymy  (Italian WordNet via Open Multilingual WordNet)
# ---------------------------------------------------------------------------
def _ensure_wordnet():
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find("corpora/" + pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=8192)
def _ita_lemma_names(word):
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn
    names = set()
    for syn in wn.synsets(word.replace(" ", "_"), pos="n", lang="ita"):
        names.update(syn.lemma_names("ita"))
    if not names:  # fall back to per-token union for multiword phrases
        for w in word.split():
            for syn in wn.synsets(w, pos="n", lang="ita"):
                names.update(syn.lemma_names("ita"))
    return names


@lru_cache(maxsize=8192)
def _has_noun_synset_it(word):
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn
    return bool(wn.synsets(word, pos="n", lang="ita"))


@lru_cache(maxsize=8192)
def are_synonymous_it(word1, word2):
    """True if two Italian noun words/phrases share a WordNet synset lemma."""
    for phrase in (word1, word2):
        if sum(_has_noun_synset_it(w) for w in phrase.split()) >= 2:
            return False
    return bool(_ita_lemma_names(word1) & _ita_lemma_names(word2))


# ---------------------------------------------------------------------------
# Phrase construction
# ---------------------------------------------------------------------------
_MOD_DEPS = {"compound", "amod", "nmod", "flat", "flat:name"}


def _lemma(tok):
    return tok.lemma_.lower()


def compound_phrase(token):
    """
    Return the token lemma together with its compound / adjectival modifiers
    (UD: compound, amod, nmod, flat), ordered by position, lowercased.
    Mirrors `get_compound_parts` but with UD labels and Italian word order.
    """
    parts = [token]
    for child in token.children:
        # Skip prepositional-phrase modifiers (nmod with a case marker):
        # "caffè per il mio amico" -> keep "caffè", drop "amico".
        if child.dep_ == "nmod" and any(gc.dep_ == "case" for gc in child.children):
            continue
        if child.dep_ in _MOD_DEPS and child.pos_ not in {"SCONJ", "CCONJ", "PART", "DET"}:
            parts.append(child)
            for gc in child.children:
                if gc.dep_ in _MOD_DEPS and not any(g2.dep_ == "case" for g2 in gc.children):
                    parts.append(gc)
    parts = sorted(set(parts), key=lambda t: t.i)
    return " ".join(_lemma(t) for t in parts)


def _oblique_phrase(obl_token):
    """A prepositional/oblique object rendered as '<case-lemma> <noun>'."""
    case = [c for c in obl_token.children if c.dep_ == "case"]
    prep = _lemma(case[0]) if case else ""
    # normalise fused preposition+article lemmas ("da il" -> "da", "a il" -> "a")
    prep = prep.split()[0] if prep else ""
    noun = compound_phrase(obl_token)
    return (prep + " " + noun).strip()


def _collect_conj(token):
    """token plus its coordinated conjuncts (same role)."""
    out = [token]
    out.extend(token.conjuncts)
    seen, uniq = set(), []
    for t in out:
        if t.i not in seen:
            seen.add(t.i)
            uniq.append(t)
    return uniq


# ---------------------------------------------------------------------------
# Passive detection (Italian)
# ---------------------------------------------------------------------------
def _is_essere_venire(tok):
    return tok.lemma_.lower() in _ESSERE_VENIRE or tok.text.lower() in _ESSERE_VENIRE


def _find_patient_it(head):
    """Passive patient: nsubj:pass if present, else a plain nsubj."""
    p = next((c for c in head.children if c.dep_ in _SUBJ_PASS), None)
    if p is None:
        p = next((c for c in head.children if c.dep_ == "nsubj"), None)
    return p


def _passive_info_it(head):
    """
    Describe the passive status of a predicate *head*.

    Signals (Italian):
      1. canonical periphrastic passive: aux:pass / nsubj:pass / obl:agent.
      2. under-tagged periphrastic passive: a past-participle VERB with an
         'essere'/'venire' aux or cop child (it_core_news_lg often tags the
         passive auxiliary as plain ``aux``/``cop``). Excluded for
         unaccusative/reflexive verbs, whose 'essere' perfect is ACTIVE.
      3. explicit by-agent: obl:agent, or an ``obl`` introduced by "da".
      4. coordinated passive: the by-agent may sit on a conj participle.
      5. impersonal / si-passivante: expl:impers on an active verb.
    """
    children = list(head.children)
    deps = {c.dep_ for c in children}

    # --- agent (obl:agent, or "da"+obl on a passive verb, or on a conj) ------
    agent = next((c for c in children if c.dep_ in _AGENT), None)
    if agent is None:
        for c in children:
            if c.dep_ == "conj" and c.pos_ == "VERB":
                agent = next((gc for gc in c.children if gc.dep_ in _AGENT), None)
                if agent is not None:
                    break

    patient_pass = _find_patient_it(head)

    # --- is this a passive construction at all? ------------------------------
    canonical = bool(deps & _AUX_PASS) or agent is not None
    is_part_past = (head.pos_ == "VERB"
                    and "Part" in head.morph.get("VerbForm")
                    and "Past" in head.morph.get("Tense"))
    essere_aux = any(c.dep_ in {"aux", "aux:pass", "cop"} and _is_essere_venire(c)
                     for c in children)
    has_reflexive = any(c.dep_ in {"expl", "expl:pass"} and c.lemma_.lower() in _REFLEXIVE_CLITICS
                        for c in children)
    undertagged = (is_part_past and essere_aux
                   and head.lemma_.lower() not in _UNACCUSATIVE
                   and not has_reflexive)

    # a "da"-obl on an otherwise-passive verb is the agent
    if agent is None and (canonical or undertagged):
        for c in children:
            if c.dep_ == "obl":
                case = next((gc for gc in c.children if gc.dep_ == "case"), None)
                if case is not None and case.lemma_.lower().split()[0] == "da":
                    agent = c
                    break

    impers = bool(not (canonical or undertagged) and (deps & _IMPERS))

    is_passive = canonical or undertagged or impers
    return {
        "is_passive": is_passive,
        "has_agent": agent is not None,
        "agent": agent,
        "patient_pass": patient_pass,
        "impersonal": impers,
    }


# ---------------------------------------------------------------------------
# Core role extraction for one predicate head
# ---------------------------------------------------------------------------
def _is_copular(head):
    return any(c.dep_ == "cop" for c in head.children)


def _event_phrase(head):
    """
    Build the Event string: negation + (aux for passive) + main predicate,
    plus tightly-bound adverbs. Copular heads emit the copula lemma ('essere').
    """
    # Nominal/adjectival copular head ("è interessante") -> event is the
    # copula ('essere'). A VERB head keeps its own lemma even when the passive
    # auxiliary is mis-tagged as ``cop`` ("fu mangiata" -> 'mangiare').
    if head.pos_ != "VERB" and _is_copular(head):
        cop = next(c for c in head.children if c.dep_ == "cop")
        event_core = _lemma(cop)
    else:
        event_core = _lemma(head)

    negs = [_lemma(c) for c in head.children
            if c.dep_ == "advmod" and _lemma(c) in _NEG_LEMMAS]
    return " ".join(negs + [event_core])


def _subjects_of(head, pinfo):
    """Agent-role fillers for *head* (list of phrase strings)."""
    subs = []
    if pinfo["is_passive"]:
        if pinfo["has_agent"]:
            for t in _collect_conj(pinfo["agent"]):
                subs.append(compound_phrase(t))
        elif pinfo["patient_pass"] is not None:
            # agentless passive: patient stays in Agent (approximation)
            for t in _collect_conj(pinfo["patient_pass"]):
                subs.append(compound_phrase(t))
        return subs

    # active
    nsubj = next((c for c in head.children if c.dep_ in _SUBJ_ACTIVE), None)
    if nsubj is not None:
        for t in _collect_conj(nsubj):
            subs.append(compound_phrase(t))
    else:
        # inherit subject from head verb if this is a conjunct / clausal dep
        if head.dep_ in {"conj", "advcl", "acl", "acl:relcl", "xcomp", "ccomp"}:
            gov = head.head
            gsub = next((c for c in gov.children if c.dep_ in _SUBJ_ACTIVE | _SUBJ_PASS), None)
            if gsub is not None:
                for t in _collect_conj(gsub):
                    subs.append(compound_phrase(t))
    return subs


def _objects_of(head, pinfo):
    """Target-role fillers for *head* (list of phrase strings)."""
    objs = []
    if pinfo["is_passive"]:
        # Passive WITH agent -> patient is the Target.
        # Passive WITHOUT agent -> patient is in Agent (approx), no Target.
        if pinfo["has_agent"] and pinfo["patient_pass"] is not None:
            for t in _collect_conj(pinfo["patient_pass"]):
                objs.append(compound_phrase(t))
        return objs

    if _is_copular(head):
        # predicate nominal / adjective is the Target
        if head.pos_ in {"NOUN", "PROPN", "ADJ", "PRON", "NUM"}:
            objs.append(compound_phrase(head))
        return objs

    # direct object
    for c in head.children:
        if c.dep_ in _OBJ_DIRECT:
            for t in _collect_conj(c):
                objs.append(compound_phrase(t))
    # predicative nominal xcomp: "diventai una vittima" -> vittima.
    # (bare-ADJ xcomp such as "abbaia forte" is adverbial, not an argument.)
    for c in head.children:
        if c.dep_ == "xcomp" and c.pos_ in {"NOUN", "PROPN"}:
            objs.append(compound_phrase(c))
    # xcomp verb objects: "vuole imparare il francese"
    for c in head.children:
        if c.dep_ == "xcomp" and c.pos_ == "VERB":
            objs.extend(_objects_of(c, _passive_info_it(c)))
    # clitic dative (iobj) fallback when no direct object
    if not objs:
        for c in head.children:
            if c.dep_ == "iobj":
                for t in _collect_conj(c):
                    objs.append(compound_phrase(t))
    # oblique / prepositional objects (obl with a simple-preposition case
    # marker) fallback. Skip clearly-adjunct prepositions (concessive/temporal)
    # that introduce circumstantial, non-argument phrases.
    if not objs:
        for c in head.children:
            if c.dep_ == "obl":
                case = next((gc for gc in c.children if gc.dep_ == "case"), None)
                if case is None:
                    continue
                if case.lemma_.lower().split()[0] in _ADJUNCT_PREPS:
                    continue
                for t in _collect_conj(c):
                    objs.append(_oblique_phrase(t))
    return objs


# ---------------------------------------------------------------------------
# Clause heads
# ---------------------------------------------------------------------------
def _clause_heads(doc):
    """Yield predicate heads: main/subordinate verbs and copular predicates."""
    heads = []
    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ not in {"aux", "aux:pass", "cop", "amod"}:
            heads.append(tok)
        elif tok.pos_ in {"NOUN", "PROPN", "ADJ", "PRON", "NUM"} and \
                any(c.dep_ == "cop" for c in tok.children):
            heads.append(tok)
    return heads


def extract_main_triple_it(doc):
    """
    Return a single (agent, event, target, passive_approx, is_passive) tuple
    for the ROOT predicate of *doc*. Used for gold-standard validation
    (one row per sentence), analogous to `extract_svo_dep` upstream.
    """
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return (None, None, None, 0, 0)
    # a copular ROOT adjective/noun is itself the head
    head = root
    pinfo = _passive_info_it(head)
    subs = _subjects_of(head, pinfo)
    objs = _objects_of(head, pinfo)
    event = _event_phrase(head)
    passive_approx = int(pinfo["is_passive"] and not pinfo["has_agent"])
    is_passive = int(pinfo["is_passive"])
    agent = subs[0] if subs else None
    target = objs[0] if objs else None
    return (agent, event, target, passive_approx, is_passive)


# ---------------------------------------------------------------------------
# Full TEA hypergraph (same DataFrame schema as teanets.extract_svos)
# ---------------------------------------------------------------------------
def extract_tea_it(doc, semantic_relations=True, resolve_prodrop=False,
                   resolver=None, use_xlmr=False):
    """
    Build the TEA edge DataFrame (Agent/Event/Target hypergraph) for *doc*,
    matching the column schema of `teanets.svo_extraction.extract_svos`.

    Parameters
    ----------
    resolve_prodrop : bool
        If True, recover dropped subjects (pro-drop) via coreference
        resolution. Clause heads are processed in text order so subject
        continuity propagates across clauses/sentences.
    resolver : ProDropResolver or ProDropResolverXLMR, optional
        An explicit resolver to carry discourse state across multiple calls
        (e.g. document processed sentence-by-sentence). If None and
        ``resolve_prodrop`` is True, a fresh resolver is created for this doc.
    use_xlmr : bool
        If True (and no explicit *resolver* is given), use the XLM-R +
        LightGBM trained resolver instead of the morphological baseline.
        Requires ``coref_it_xlmr.joblib``.
    """
    if resolve_prodrop and resolver is None:
        if use_xlmr:
            from prodrop_coref_xlmr import ProDropResolverXLMR
            resolver = ProDropResolverXLMR()
            resolver.encode_document(doc)
        else:
            from prodrop_coref import ProDropResolver
            resolver = ProDropResolver()

    triples = []  # (subs, event, objs, passive_approx, is_passive)
    # sort heads by position so the resolver sees antecedents left-to-right
    for head in sorted(_clause_heads(doc), key=lambda t: t.i):
        pinfo = _passive_info_it(head)
        subs = _subjects_of(head, pinfo)
        objs = _objects_of(head, pinfo)
        event = _event_phrase(head)
        if resolver is not None:
            if not subs and not (pinfo["is_passive"] and pinfo["has_agent"]):
                r = resolver.resolve(head)
                if r is not None:
                    subs = [r]
            if pinfo["is_passive"] and pinfo["has_agent"]:
                topic = objs[0] if objs else None
            else:
                topic = subs[0] if subs else None
            if hasattr(resolver, 'register_mention'):
                from prodrop_coref_xlmr import update_history_xlmr
                update_history_xlmr(resolver, head, topic)
            else:
                from prodrop_coref import update_history
                update_history(resolver, head, topic)
        passive_approx = pinfo["is_passive"] and not pinfo["has_agent"]
        triples.append((subs, event, objs, passive_approx, pinfo["is_passive"]))

    rows = []
    svo_id = 0
    for subs, event, objs, papprox, is_pass in triples:
        hyper = str([subs, [event], objs])
        for s in subs:
            rows.append(dict(**{"Node 1": s, "TEA": "Agent", "Node 2": event,
                                "TEA2": "Event", "Hypergraph": hyper,
                                "Semantic-Syntactic": 0, "svo_id": svo_id,
                                "passive_approx": int(papprox),
                                "is_passive": int(is_pass)}))
        for o in objs:
            rows.append(dict(**{"Node 1": event, "TEA": "Event", "Node 2": o,
                                "TEA2": "Target", "Hypergraph": hyper,
                                "Semantic-Syntactic": 0, "svo_id": svo_id,
                                "passive_approx": 0, "is_passive": int(is_pass)}))
        for a, b in combinations(subs, 2):
            rows.append(dict(**{"Node 1": a, "TEA": "Agent", "Node 2": b,
                                "TEA2": "Agent", "Hypergraph": hyper,
                                "Semantic-Syntactic": 0, "svo_id": svo_id,
                                "passive_approx": int(papprox),
                                "is_passive": int(is_pass)}))
        for a, b in combinations(objs, 2):
            rows.append(dict(**{"Node 1": a, "TEA": "Target", "Node 2": b,
                                "TEA2": "Target", "Hypergraph": hyper,
                                "Semantic-Syntactic": 0, "svo_id": svo_id,
                                "passive_approx": 0, "is_passive": int(is_pass)}))
        svo_id += 1

    if semantic_relations:
        subjects = list({s for subs, _, _, _, _ in triples for s in subs})
        objects = list({o for _, _, objs, _, _ in triples for o in objs})
        for a, b in combinations(subjects, 2):
            if are_synonymous_it(a, b):
                rows.append({"Node 1": a, "TEA": "Agent", "Node 2": b, "TEA2": "Agent",
                             "Hypergraph": "N/A", "Semantic-Syntactic": 1, "svo_id": "N/A",
                             "passive_approx": 0, "is_passive": 0})
        for a, b in combinations(objects, 2):
            if are_synonymous_it(a, b):
                rows.append({"Node 1": a, "TEA": "Target", "Node 2": b, "TEA2": "Target",
                             "Hypergraph": "N/A", "Semantic-Syntactic": 1, "svo_id": "N/A",
                             "passive_approx": 0, "is_passive": 0})

    return pd.DataFrame(rows, columns=[
        "Node 1", "TEA", "Node 2", "TEA2", "Hypergraph",
        "Semantic-Syntactic", "svo_id", "passive_approx", "is_passive"],
        dtype=object)


def extract_tea_from_text_it(text, semantic_relations=True,
                            resolve_prodrop=False, use_xlmr=False):
    """Convenience: text -> TEA DataFrame. Set ``resolve_prodrop=True`` to
    recover dropped subjects via coreference resolution. Set ``use_xlmr=True``
    to use the trained XLM-R + LightGBM resolver."""
    return extract_tea_it(get_it_nlp()(text),
                          semantic_relations=semantic_relations,
                          resolve_prodrop=resolve_prodrop,
                          use_xlmr=use_xlmr)
