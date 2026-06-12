from functools import lru_cache

from teanets.textloader import text_preparation
from teanets.resources import _VAGUE_ADVMODS, _VAGUE_AUX
from .nlp_utils import spacynlp, compute_valence, ensure_wordnet_downloaded
from itertools import combinations
from nltk.corpus import wordnet as wn
import pandas as pd

# Lemmas of modal and copulative auxiliaries that do NOT signal passive even when they appear as dep=aux in front of a VBN.

_NONPASSIVE_AUX_LEMMAS = {
    "be", "have", "do", "will", "shall", "would", "should",
    "could", "might", "may", "must", "can",
}


def extract_svos_from_text(text, coref_solver="fastcoref", semantic_relations=True):
    """
    Extract Subject-Verb-Object (SVO) triples from a given text.

    Parameters
    ----------
    text : str
        The input text.
    coref_solver : str or None
        Coreference solver to use ('fastcoref', 'stanza', or None to skip).
    semantic_relations : bool
        If True (default), also add WordNet-synonymy edges between nodes of
        the same role. Set to False to speed up large batch jobs.
    """
    prepared_text = text_preparation(text, coref_solver=coref_solver)
    doc = spacynlp(prepared_text)
    svos = extract_svos(doc, semantic_relations=semantic_relations)
    return svos


################################################################################################
## Extract svos code.
################################################################################################


@lru_cache(maxsize=8192)
def get_synsets(phrase):
    """
    Get noun synsets for the phrase as a whole.
    """
    ensure_wordnet_downloaded()
    # Attempt to get synsets for the whole phrase as a noun
    synsets = wn.synsets(phrase.replace(" ", "_"), pos="n")
    if synsets:
        return synsets
    else:
        # Split the phrase into words and get noun synsets for each noun word
        words = phrase.split()
        synsets = []
        for word in words:
            # Get noun synsets for each word
            synsets.extend(wn.synsets(word, pos="n"))
        return synsets


@lru_cache(maxsize=8192)
def _has_noun_synsets(word):
    ensure_wordnet_downloaded()
    return bool(wn.synsets(word, pos="n"))


@lru_cache(maxsize=8192)
def are_synonymous(word1, word2):
    """
    Check if two words or phrases are synonyms considering only nouns.
    Enforces the rule: if a word contains two nouns, return False as synonyms.
    """
    # Check if either word contains two or more nouns
    for phrase in [word1, word2]:
        words = phrase.split()
        noun_count = 0
        for word in words:
            # Check if the word has noun synsets
            if _has_noun_synsets(word):
                noun_count += 1
        if noun_count >= 2:
            return False

    synsets1 = get_synsets(word1)
    synsets2 = get_synsets(word2)
    for syn1 in synsets1:
        for syn2 in synsets2:
            # Check if synsets are the same
            if syn1 == syn2:
                return True
            # Check if they share any lemma names (nouns only)
            if set(syn1.lemma_names()).intersection(set(syn2.lemma_names())):
                return True
    return False


def _is_verb_candidate(token):
    """
    Return True when *token* should be treated as a verb candidate for SVO
    extraction. This is the single source of truth for the candidate filter,
    shared by ``extract_svos()`` and ``is_sentence_passive()``.
    """
    return (
        (token.pos_ in ("VERB", "AUX"))
        and (
            token.dep_
            not in ("aux", "auxpass", "amod", "npadvmod", "prep", "xcomp", "csubj")
        )
        and not (
            token.dep_ == "conj"
            and token.head.dep_
            in ("aux", "auxpass", "amod", "prep", "xcomp", "csubj")
        )
    )


def is_sentence_passive(doc):
    """
    Return True if at least one verb candidate in the parsed *doc* is flagged
    as passive by ``_passive_info()``. Uses the same verb-candidate filter as
    ``extract_svos()``, so sentence-level passive detection is consistent with
    the extraction pipeline.
    """
    return any(
        _passive_info(token)["is_passive"] for token in doc if _is_verb_candidate(token)
    )


def extract_svos(doc, semantic_relations=True):
    """
    Extract Subject-Verb-Object (SVO) triples from a parsed document.
    Returns a pandas DataFrame with specified columns.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The parsed document.
    semantic_relations : bool, optional
        If True (default), also add WordNet-synonymy edges between nodes of
        the same role (``Semantic-Syntactic`` = 1). Set to False to skip this
        O(n^2) step, e.g. in large batch jobs.
    """
    # Collect SVO triples
    svo_triples = []

    for possible_verb in doc:
        if _is_verb_candidate(possible_verb):

            subjects = get_verb_subjects(possible_verb)
            objects = get_verb_objects(possible_verb)
            verb_phrase = get_verb_phrase(possible_verb)
            # passive_approx=True when verb is passive but no explicit agent
            # was found; in this case the "subject" is the patient/victim
            # placed in Agent as a best-available approximation.
            pinfo = _passive_info(possible_verb)
            passive_approx = pinfo["is_passive"] and not pinfo["has_agent"]
            is_passive = pinfo["is_passive"]

            svo_triples.append([subjects, verb_phrase, objects, passive_approx, is_passive])

    # Initialize list to hold DataFrame rows
    data_rows = []

    # Process SVO triples to create DataFrame rows
    svo_id = 0
    for svo in svo_triples:
        subjects, verbs, objects, passive_approx, is_passive = svo
        # The hypergraph description contains only the three node groups;
        # the voice flags are already exposed as dedicated columns.
        hypergraph = str([subjects, verbs, objects])

        # For syntactic relations (Semantic-Syntactic = 0)
        # Create Subject-Verb relations
        for subj, _ in subjects:
            for verb in verbs:
                # Subject - Verb relation
                data_rows.append(
                    {
                        "Node 1": subj,
                        "TEA": "Agent",
                        "Node 2": verb,
                        "TEA2": "Event",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                        # 1 = victim-in-Agent approximation (passive, no explicit perpetrator)
                        # 0 = genuine agent or active construction
                        "passive_approx": int(passive_approx),
                        "is_passive": int(is_passive),
                    }
                )

        # Create Verb-Object relations
        for verb in verbs:
            for obj, _ in objects:
                # Verb - Object relation
                data_rows.append(
                    {
                        "Node 1": verb,
                        "TEA": "Event",
                        "Node 2": obj,
                        "TEA2": "Target",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                        "passive_approx": 0,
                        "is_passive": int(is_passive),
                    }
                )

        # Create Subject-Subject relations when multiple subjects are present
        if len(subjects) > 1:
            for (subj1, _), (subj2, _) in combinations(subjects, 2):
                data_rows.append(
                    {
                        "Node 1": subj1,
                        "TEA": "Agent",
                        "Node 2": subj2,
                        "TEA2": "Agent",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                        "passive_approx": int(passive_approx),
                        "is_passive": int(is_passive),
                    }
                )

        # Create Object-Object relations when multiple objects are present
        if len(objects) > 1:
            for (obj1, _), (obj2, _) in combinations(objects, 2):
                data_rows.append(
                    {
                        "Node 1": obj1,
                        "TEA": "Target",
                        "Node 2": obj2,
                        "TEA2": "Target",
                        "Hypergraph": hypergraph,
                        "Semantic-Syntactic": 0,
                        "svo_id": svo_id,
                        "passive_approx": 0,
                        "is_passive": int(is_passive),
                    }
                )
        svo_id += 1

    # Now, process for semantic relations (Semantic-Syntactic = 1)
    if semantic_relations:
        # Collect all subjects and objects across all SVOs
        all_subjects = [subj for svo in svo_triples for subj, _ in svo[0]]
        all_objects = [obj for svo in svo_triples for obj, _ in svo[2]]

        # Remove duplicates
        subject_nodes = list(set(all_subjects))
        object_nodes = list(set(all_objects))

        # For each pair of subjects, check if they are synonyms
        subject_pairs = combinations(subject_nodes, 2)
        for subj1, subj2 in subject_pairs:
            if are_synonymous(subj1, subj2):
                # Add semantic relation
                data_rows.append(
                    {
                        "Node 1": subj1,
                        "TEA": "Agent",
                        "Node 2": subj2,
                        "TEA2": "Agent",
                        "Hypergraph": "N/A",
                        "Semantic-Syntactic": 1,
                        "svo_id": "N/A",
                    }
                )

        # Similarly for objects
        object_pairs = combinations(object_nodes, 2)
        for obj1, obj2 in object_pairs:
            if are_synonymous(obj1, obj2):
                data_rows.append(
                    {
                        "Node 1": obj1,
                        "TEA": "Target",
                        "Node 2": obj2,
                        "TEA2": "Target",
                        "Hypergraph": "N/A",
                        "Semantic-Syntactic": 1,
                        "svo_id": "N/A",
                    }
                )

    # Semantic rows (Semantic-Syntactic=1) have no passive_approx concept
    for row in data_rows:
        row.setdefault("passive_approx", 0)
        row.setdefault("is_passive", 0)

    # Create DataFrame
    df = pd.DataFrame(
        data_rows,
        columns=[
            "Node 1",
            "TEA",
            "Node 2",
            "TEA2",
            "Hypergraph",
            "Semantic-Syntactic",
            "svo_id",
            "passive_approx",
            "is_passive",
        ],
    )

    return df


def get_verb_phrase(verb):
    """
    Retrieve the verb and its modifiers (negations, adverbs) as a single string.
    Also includes auxiliary verbs and connected verbs (xcomp, ccomp, conj).
    """
    parts = []

    # Get auxiliaries and negations before the verb
    aux = [
        child.lemma_
        for child in verb.lefts
        if child.dep_ in {"aux", "auxpass", "neg"} and child.lemma_ not in _VAGUE_AUX
    ]
    parts.extend(aux)

    ## Get adverbial modifiers before the verb
    # adverbs_before = [child.lemma_ for child in verb.lefts if child.dep_ in {'advmod', 'amod','npadvmod'} and \
    #                   child.pos_ not in {'SCONJ', 'CCONJ', 'PART', 'DET'} and child.lemma_ not in _VAGUE_ADVMODS]
    # parts.extend(adverbs_before)
    # Get adverbial before after the verb
    for child in verb.lefts:
        if (
            child.dep_ in {"advmod", "amod", "npadvmod"}
            and child.pos_ not in {"SCONJ", "CCONJ", "PART", "DET"}
            and child.lemma_ not in _VAGUE_ADVMODS
        ):
            # Include the noun and its modifiers
            noun_phrase, prep_phrases = get_compound_parts(child)
            parts.append(noun_phrase)

    # Add the main verb
    parts.append(verb.lemma_)

    ## In get_verb_phrase(verb), after processing adverbs_after
    # for child in verb.children:
    #    if child.dep_ == 'acomp':
    #        # Include the 'acomp' adjective in the verb phrase
    #        parts.append(child.lemma_)
    #        # Include any modifiers of the 'acomp' adjective
    #        for grandchild in child.children:
    #            if grandchild.dep_ in {'advmod', 'amod', 'npadvmod'} and\
    #                grandchild.pos_ not in {'SCONJ', 'CCONJ', 'PART', 'DET'} and\
    #                    grandchild.lemma_ not in VAGUE_ADVMODS:
    #                parts.append(grandchild.lemma_)

    # Get adverbial modifiers after the verb
    # adverbs_after = [child.lemma_ for child in verb.rights if child.dep_ in {'advmod', 'amod', 'npadvmod'} and\
    #                  child.pos_ not in {'SCONJ', 'CCONJ', 'PART', 'DET'} and child.lemma_ not in VAGUE_ADVMODS]
    # parts.extend(adverbs_after)

    # Get adverbial modifiers after the verb
    for child in verb.rights:
        if (
            child.dep_ in {"advmod", "amod", "npadvmod"}
            and child.pos_ not in {"SCONJ", "CCONJ", "PART", "DET"}
            and child.lemma_ not in _VAGUE_ADVMODS
        ):
            # Include the noun and its modifiers
            noun_phrase, prep_phrases = get_compound_parts(child)
            parts.append(noun_phrase)

    # Get auxiliaries and negations after the verb
    aux = [
        child.lemma_
        for child in verb.rights
        if child.dep_ in {"aux", "auxpass", "neg"} and child.lemma_ not in _VAGUE_AUX
    ]
    parts.extend(aux)

    for child in verb.children:
        if child.dep_ in {"xcomp"}:
            # Include the full xcomp verb phrase (auxiliaries, negations and
            # modifiers of the xcomp verb included), so that e.g.
            # "did not seem to care" is captured as a unit.
            child_verb_phrase = get_verb_phrase(child)[0]
            if child_verb_phrase:
                parts.append(child_verb_phrase)

    return [" ".join(parts)]


def _passive_info(verb):
    """
    Returns a dict describing the passive status of *verb*.

    Four independent signals are checked:

    1. CANONICAL passive  – spaCy uses ``auxpass``/``nsubjpass`` directly on
       the verb (e.g. "I was raped", "I got raped").
    2. FEEL-passive       – verb is VBN and has a child with ``dep='aux'``
       whose lemma is NOT a modal/copula (e.g. "I felt abused",
       "She seemed targeted").  spaCy mis-tags the aux as active here.
    3. AGENT-phrase       – verb is VBN and has a child with ``dep='agent'``
       (by-phrase), regardless of auxiliaries (e.g. "She seemed targeted by
       everyone", "He appeared beaten by the crowd").
    4. COORDINATED conj   – verb is VBN with ``dep='conj'`` and its head verb
       is itself passive (e.g. "raped" in "I was held down and raped").
       The conjunct inherits the passive status (and agent, if any) from head.

    Returns
    -------
    dict with keys:
        is_passive  : bool
        has_agent   : bool  – True only when an explicit by-phrase exists
        agent_pobj  : Token | None  – the pobj of the agent prep, if found
        feel_aux    : Token | None  – the non-standard aux token, if any
    """
    _nsubjpass_labels = {"nsubjpass", "nsubj:pass"}
    _auxpass_labels   = {"auxpass",   "aux:pass"}
    children_deps = {child.dep_ for child in verb.children}

    # Signal 1 – canonical
    canonical = bool(
        (children_deps & _nsubjpass_labels) or
        (children_deps & _auxpass_labels)
    )

    # Signal 2 – feel-passive (VBN + non-modal aux)
    feel_aux = None
    if verb.tag_ == "VBN":
        for child in verb.children:
            if child.dep_ == "aux" and child.lemma_ not in _NONPASSIVE_AUX_LEMMAS:
                feel_aux = child
                break

    # Signal 3 – explicit agent (by-phrase)
    agent_pobj = None
    for child in verb.children:
        if child.dep_ == "agent":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    agent_pobj = gc
                    break
            if agent_pobj:
                break

    # Coordinated passive: agent may attach to a conj verb
    if agent_pobj is None:
        for child in verb.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                for gc in child.children:
                    if gc.dep_ == "agent":
                        for ggc in gc.children:
                            if ggc.dep_ == "pobj":
                                agent_pobj = ggc
                                break
                    if agent_pobj:
                        break
            if agent_pobj:
                break

    has_agent = agent_pobj is not None

    # Signal 4 – coordinated VBN conjunct inherits passive from head verb.
    # "I was held down and raped": raped=conj(held), VBN, no aux/nsubjpass
    # of its own, but held IS passive → raped inherits passive status.
    conj_passive = False
    if not (canonical or feel_aux or has_agent):
        if verb.tag_ == "VBN" and verb.dep_ == "conj" and verb.head.pos_ == "VERB":
            head_info = _passive_info(verb.head)
            conj_passive = head_info["is_passive"]
            # If the head has an agent, propagate it to this conjunct
            if head_info["has_agent"]:
                has_agent = True
                agent_pobj = head_info["agent_pobj"]

    is_passive = canonical or (feel_aux is not None) or has_agent or conj_passive

    return {
        "is_passive": is_passive,
        "has_agent":  has_agent,
        "agent_pobj": agent_pobj,
        "feel_aux":   feel_aux,
    }


def _find_patient(verb, info):
    """
    Return a list of patient tokens for a passive *verb*, searching in order:
    1. nsubjpass direct children (canonical passives).
    2. nsubj direct children (feel-passives where spaCy uses nsubj).
    3. nsubj/nsubjpass children of the head verb (raised passives: oprd/xcomp,
       or coordinated passives: conj).
    4. nsubj children of feel_aux (inconsistent spaCy parse).
    """
    _nsubjpass_labels = {"nsubjpass", "nsubj:pass"}

    patients = [child for child in verb.children if child.dep_ in _nsubjpass_labels]
    if not patients:
        patients = [child for child in verb.children if child.dep_ == "nsubj"]
    if not patients and verb.dep_ in {"oprd", "xcomp", "advcl", "conj"}:
        patients = [
            child for child in verb.head.children
            if child.dep_ in _nsubjpass_labels | {"nsubj"}
        ]
    # Also try feel_aux's children (inconsistent spaCy parse with advmod)
    if not patients and info.get("feel_aux") is not None:
        patients = [
            child for child in info["feel_aux"].children
            if child.dep_ in _nsubjpass_labels | {"nsubj"}
        ]
    return patients


def get_verb_subjects(verb):
    """
    Retrieve the subjects of a given verb, including through coordination and inheritance.
    Coordination refers to when multiple subjects are linked by a conjunction (e.g., "and" or "or") and share the same verb.
    Inheritance occurs when a subject is implied from a previous clause or sentence and applies to a following verb without being explicitly repeated.

    For compound nouns, each component is included separately.

    PASSIVE VOICE HANDLING:
      Uses ``_passive_info()`` to detect three types of passive constructions:
        1. Canonical passive (auxpass / nsubjpass) — e.g. "I was raped by X".
        2. Feel-passive (non-modal aux + VBN) — e.g. "I felt abused by X".
        3. Agent-phrase passive (VBN + by-phrase) — e.g. "She seemed targeted by Y".

      - Passive WITH explicit agent ("by Y"):
          Y (pobj of the agent prep) is the semantic agent → goes to Agent.
          X (patient) → goes to Target via ``get_verb_objects()``.
      - Passive WITHOUT explicit agent ("I was raped", "I felt abused"):
          X (patient) stays in Agent as best-available approximation;
          flagged with ``passive_approx=1`` in the output DataFrame.
      - Coordinated passive ("X was written and reviewed by Y"):
          The agent may be a child of a conj verb, not of ROOT.
          ``_passive_info()`` searches conj children as well.

    Args:
        verb: The verb token.

    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main subject and its prepositional phrases.

    """
    subjects = []

    # Handle acl and relcl dependencies
    if verb.dep_ in {"acl", "relcl"}:
        # If the verb has its own subject, use it
        nsubjs = [child for child in verb.children if child.dep_ == "nsubj"]
        if nsubjs:
            for subj in nsubjs:
                # Relative pronouns: refer back to the modified noun.
                # NOTE: "target" was previously listed here by mistake (residual
                # of an old role-name refactor). Removed: it is a regular noun.
                if subj.lower_ in {"which", "that", "whom", "whose"}:
                    # Use the modified noun as the subject
                    modified_noun = verb.head
                    main_part, prep_parts = get_compound_parts(
                        modified_noun, lemmatize=True
                    )
                    subjects.append((main_part, prep_parts))
                else:
                    subjects.extend(extract_subjects(subj))
        else:
            # Use the modified noun as the subject
            modified_noun = verb.head
            main_part, prep_parts = get_compound_parts(modified_noun, lemmatize=True)
            subjects.append((main_part, prep_parts))
        return subjects

    # ── PASSIVE VOICE HANDLING ────────────────────────────────────────────────
    # Uses _passive_info() which detects canonical, feel-, and agent-phrase
    # passive constructions uniformly.
    pinfo = _passive_info(verb)

    if pinfo["is_passive"]:
        if pinfo["has_agent"]:
            # Passive WITH explicit agent → agent goes to Agent role
            subjects.extend(extract_subjects(pinfo["agent_pobj"]))
        else:
            # Passive WITHOUT explicit agent → patient stays in Agent as
            # best-available approximation (same policy as canonical passive).
            # Flag is recorded via passive_approx column in extract_svos().
            for patient in _find_patient(verb, pinfo):
                subjects.extend(extract_subjects(patient))

        # Always return here: prevents false inheritance of an active ancestor's
        # nsubj when the current verb is passive (e.g. "I went out and got raped").
        return subjects
    # ── END PASSIVE VOICE HANDLING ────────────────────────────────────────────

    # Standard active voice: use nsubj
    direct_subjects = [
        child for child in verb.children if child.dep_ in {"nsubj"}
    ]

    for subj in direct_subjects:
        subjects.extend(extract_subjects(subj))

    # Handle clausal subjects (csubj)
    csubj_subjects = [child for child in verb.children if child.dep_ == "csubj"]
    for csubj in csubj_subjects:
        clause = extract_clause(csubj)
        subjects.append((clause, []))  # No preps in clause

    # Inherit subjects from ancestor verbs if none found
    if not subjects:
        for ancestor in verb.ancestors:
            ancestor_subjects = [
                child
                for child in ancestor.children
                if child.dep_ in {"nsubj", "nsubjpass", "nsubj:pass", "csubj"}
            ]
            for anc_subj in ancestor_subjects:
                subjects.extend(extract_subjects(anc_subj))
                if subjects:
                    break
            if subjects:
                break

    # Inherit subjects from head verb if verb is a conjunct
    if not subjects and verb.dep_ == "conj" and verb.head.pos_ == "VERB":
        subjects = get_verb_subjects(verb.head)

    # Handle 'conj' nouns connected to the verb
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ in {"NOUN", "PROPN", "PRON"}:
            subjects.extend(extract_subjects(child))

    # Handle imperative sentences (assuming 'you' as subject)
    if not subjects and verb.head == verb:
        # Check if the verb is in imperative mood
        if verb.morph.get("Mood") == ["Imp"] or verb.tag_ == "VB":
            subjects.append(("you", []))

    # Remove duplicates while preserving order
    seen = set()
    unique_subjects = []
    for subj in subjects:
        key = (subj[0], tuple(subj[1]))
        if key not in seen:
            seen.add(key)
            unique_subjects.append(subj)

    return unique_subjects


def extract_subjects(subject_token):
    """
    Extract the subject and its compound components.
    Args:
        subject_token (spacy.tokens.Token): The subject token.
    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main subject and its prepositional phrases.
    """
    subjects = []

    if subject_token.pos_ in {"NOUN", "PROPN", "PRON"}:
        # Extract compounds and the main noun
        main_part, prep_parts = get_compound_parts(subject_token, lemmatize=True)
        subjects.append((main_part, prep_parts))

    # Additionally, handle conjunct subjects (e.g., "Alice and Bob")
    conjunct_subjects = [
        conj
        for conj in subject_token.conjuncts
        if conj.pos_ in {"NOUN", "PROPN", "PRON"}
    ]
    for conj in conjunct_subjects:
        main_part, prep_parts = get_compound_parts(conj, lemmatize=True)
        subjects.append((main_part, prep_parts))

    return subjects


def get_compound_parts(token, lemmatize=True):
    """
    Retrieve the main token and its compound modifiers as a single string.
    Also includes adverbial and adjectival modifiers of adjectives.

    Args:
        token (spacy.tokens.Token): The token to extract compounds from.
        lemmatize (bool, optional): Whether to lemmatize the extracted parts.

    Returns:
        Tuple[str, List[str]]: A tuple containing the compound phrase and a list of prepositional phrases.
    """
    parts = []
    prep_parts = []

    # Extract determiners (excluding 'the', 'a', 'an')
    dets = [
        child
        for child in token.children
        if (child.dep_ == "det") and child.lemma_ not in {"the", "a", "an"}
    ]
    if lemmatize:
        parts.extend([child.lemma_ for child in dets])
    else:
        parts.extend([child.text for child in dets])

    # Extract modifiers
    # Adjusted condition in get_compound_parts
    modifiers = []
    for child in token.children:
        if (
            child.dep_ in {"compound", "amod", "nmod", "poss", "case"}
            or (
                child.dep_ in {"advmod", "npadvmod"}
                and token.pos_ in {"VERB", "AUX", "ADJ", "NOUN", "PROPN"}
            )
        ) and child.pos_ not in {"SCONJ", "CCONJ", "PART"}:
            modifiers.append(child)
            # Include modifiers of modifiers, including advmod for adjectives and nouns
            modifiers.extend(
                [
                    gc
                    for gc in child.children
                    if gc.dep_ in {"compound", "amod", "nmod", "poss", "case", "advmod"}
                ]
            )

    # Sort modifiers based on their position in the text
    modifiers = sorted(set(modifiers), key=lambda x: x.i)

    # Add all modifier text
    for modifier in modifiers:
        if lemmatize:
            parts.append(modifier.lemma_.lower())
        else:
            parts.append(modifier.text)

    # Add the main token's text
    if lemmatize:
        parts.append(token.lemma_.lower())
    else:
        parts.append(token.text)

    # Handle prepositional phrases separately
    preps = [child for child in token.children if child.dep_ == "prep"]
    for prep in preps:
        # Process the preposition and its conjuncts
        prep_conjuncts = [prep] + list(prep.conjuncts)
        for p in prep_conjuncts:
            pobj_list = [child for child in p.children if child.dep_ == "pobj"]
            for pobj in pobj_list:
                # Process the pobj and its conjuncts
                pobj_conjuncts = [pobj] + list(pobj.conjuncts)
                for pobj_item in pobj_conjuncts:
                    # For each pobj_conjunct, create a separate prep_phrase
                    prep_phrase = [p.lemma_]  # Start with the preposition
                    pobj_main, pobj_preps = get_compound_parts(
                        pobj_item, lemmatize=lemmatize
                    )
                    prep_phrase.append(pobj_main)
                    # Handle nested prepositional phrases
                    for nested_prep in pobj_preps:
                        prep_phrase.append(nested_prep)
                    # Join and append to prep_parts
                    prep_parts.append(" ".join(prep_phrase))

    main_part = " ".join(parts)

    return main_part, prep_parts


def get_conjuncts(token, visited=None):
    """
    Recursively retrieve the token and its conjuncts, avoiding infinite loops.

    Args:
        token (spacy.tokens.Token): The token to find conjuncts for.
        visited (Set[spacy.tokens.Token], optional): Set of already visited tokens.

    Returns:
        List[Token]: A list of tokens including the original token and its conjuncts.
    """
    if visited is None:
        visited = set()
    conjuncts = [token]
    visited.add(token)
    for conj in token.conjuncts:
        if conj not in visited:
            conjuncts.extend(get_conjuncts(conj, visited))
    return conjuncts


def extract_clause(token):
    """
    Extract the full clause connected to a token.
    """
    clause_tokens = list(token.subtree)
    clause_tokens = sorted(clause_tokens, key=lambda x: x.i)
    lemmas = [t.lemma_ for t in clause_tokens]
    clause = " ".join(lemmas)
    return clause


def get_verb_objects(verb):
    """
    Retrieve the objects of a given verb, considering passive voice and the presence of agents.
    This function handles various cases to accurately extract objects, including:
    - Direct objects
    - Indirect objects
    - Objects in prepositional phrases
    - Passive constructions with or without agents
    - Clausal complements
    - Objects from xcomp dependencies
    - Inheritance from conjoined verbs

    Args:
        verb (spacy.tokens.Token): The verb token.

    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main object and its prepositional phrases.
    """
    objects = []

    # Helper function to process and extend objects list, including conjuncts
    def process_objects(object_tokens):
        for obj in object_tokens:
            objects.extend(extract_objects(obj))

    # ── PASSIVE VOICE: patient as Target ─────────────────────────────────────
    # For all passive constructions (canonical, feel-, agent-phrase) that have
    # an explicit agent, we look up the patient and add it as Target.
    pinfo = _passive_info(verb)
    if pinfo["is_passive"] and pinfo["has_agent"]:
        patients = _find_patient(verb, pinfo)
        process_objects(patients)
        return objects
    # ── END PASSIVE VOICE ─────────────────────────────────────────────────────

    # Standard active voice: proceed with object extraction
    # (Passive with agent already returned above.)

    # 1. Direct Objects (e.g., "eat an apple")
    direct_objects = [
        child
        for child in verb.children
        if child.dep_ in {"dobj", "attr", "oprd", "acomp"}
    ]
    process_objects(direct_objects)

    # 2. Indirect Objects (e.g., "give me the book")
    indirect_objects = [
        child for child in verb.children if child.dep_ in {"iobj", "dative"}
    ]
    process_objects(indirect_objects)

    # 3. Objects in Prepositional Phrases (e.g., "look at the sky")
    prep_phrases = [child for child in verb.children if child.dep_ == "prep"]
    for prep in prep_phrases:
        # Handles Conjuncts in Prepositional Phrases
        preps = get_conjuncts(prep)
        for p in preps:
            pobj = [child for child in p.children if child.dep_ == "pobj"]
            process_objects(pobj)

    # 4. If no direct objects, include prepositional phrases as objects
    if not objects:
        for prep in prep_phrases:
            prep_text = prep.lemma_
            pobj = [child for child in prep.children if child.dep_ == "pobj"]
            for obj in pobj:
                main_part, prep_parts = get_compound_parts(obj)
                full_object = prep_text + " " + main_part
                objects.append((full_object, prep_parts))

    # 5. Objects in Noun Phrase Adverbial Modifiers (e.g., "He arrived yesterday")
    npadvmod_objects = [
        child
        for child in verb.children
        if (child.dep_ == "npadvmod") and (child.pos_ in {"PROPN", "PRON", "NOUN"})
    ]
    process_objects(npadvmod_objects)

    # 6. Handle clausal complements (ccomp)
    for child in verb.children:
        if child.dep_ == "ccomp":
            # Include the objects of the ccomp verb
            ccomp_objects = get_verb_objects(child)
            objects.extend(ccomp_objects)

    # 7. Handle xcomp dependencies: include objects from xcomp verb
    xcomp_children = [child for child in verb.children if child.dep_ == "xcomp"]
    for xcomp_child in xcomp_children:
        xcomp_objects = get_verb_objects(xcomp_child)
        if xcomp_objects:
            objects.extend(xcomp_objects)
        else:
            # If no objects found, try to get direct objects from the xcomp verb
            xcomp_dobjs = [
                child
                for child in xcomp_child.children
                if child.dep_ in {"dobj", "attr", "oprd"}
            ]
            process_objects(xcomp_dobjs)

    # 8. Inherit objects from conjoined verbs if none found
    if not objects:
        for child in verb.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                conj_objects = get_verb_objects(child)
                if conj_objects:
                    objects.extend(conj_objects)
                    break  # Avoid duplicates by stopping after the first found

    return objects


def extract_objects(object_token):
    """
    Extract the object and its compound components, including prepositional phrases.

    Args:
        object_token (spacy.tokens.Token): The object token.
    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing the main object and its prepositional phrases.
    """
    objects = []

    # Check if the token is a noun, proper noun, or pronoun
    if (object_token.pos_ in {"NOUN", "PROPN", "PRON"}) or (
        object_token.dep_ in ("acomp")
    ):
        # Extract compounds and the main noun
        main_part, prep_parts = get_compound_parts(object_token)
        # If the object is a 'pobj', include the preposition
        if object_token.dep_ == "pobj":
            preposition = object_token.head.lemma_
            main_part = preposition + " " + main_part
        objects.append((main_part, prep_parts))

    # Additionally, handle conjunct objects (e.g., "apples and oranges")

    conjunct_objects = [
        conj
        for conj in object_token.conjuncts
        if conj.pos_ in {"NOUN", "PROPN", "PRON", "ADJ"}
    ]
    for conj in conjunct_objects:
        # Extract compounds and the main noun for the conjunct
        main_part, prep_parts = get_compound_parts(conj)
        # If the conjunct object is a 'pobj', include the preposition
        if conj.dep_ == "pobj":
            preposition = conj.head.lemma_
            main_part = preposition + " " + main_part
        objects.append((main_part, prep_parts))

    return objects
