import re
from .nlp_utils import get_stanza_nlp, get_spacy_nlp
from teanets.resources import _COREFERENCE_NOUNS


# Module-level singletons for the fastcoref model, keyed by device.
# Loading the model is expensive (downloads weights, allocates GPU memory),
# so we keep one instance per process and device. Both the interactive
# ``extract_svos_from_text`` path and ``teanets.batch_extract`` share this
# loader.
_FASTCOREF_MODELS = {}

# Possessive pronouns need special treatment during coreference replacement:
# substituting "my" with a plain mention ("John") would produce ungrammatical
# text ("John brother"); the genitive form ("John's brother") is used instead.
_POSSESSIVE_PRONOUNS = {
    "my", "your", "his", "her", "its", "our", "their",
}


def load_fastcoref_model(device="cpu"):
    """
    Load (once per process and device) and return the fastcoref model.

    The model is patched to force the ``eager`` attention implementation,
    which avoids incompatibilities between fastcoref and recent versions of
    ``transformers``. NOTE: the patch temporarily monkeypatches
    ``AutoModel.from_config`` and is therefore not thread-safe; load the
    model from the main thread before spawning workers.
    """
    global _FASTCOREF_MODELS
    if device in _FASTCOREF_MODELS:
        return _FASTCOREF_MODELS[device]

    import functools
    import logging

    from fastcoref import FCoref as OriginalFCoref
    from transformers import AutoModel

    # Suppress transformers/fastcoref logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("fastcoref").setLevel(logging.ERROR)

    class PatchedFCoref(OriginalFCoref):
        def __init__(self, *args, **kwargs):
            original_from_config = AutoModel.from_config

            def patched_from_config(config, *a, **kw):
                kw["attn_implementation"] = "eager"
                return original_from_config(config, *a, **kw)

            try:
                AutoModel.from_config = functools.partial(
                    patched_from_config, attn_implementation="eager"
                )
                super().__init__(*args, **kwargs)
            finally:
                AutoModel.from_config = original_from_config

    _FASTCOREF_MODELS[device] = PatchedFCoref(nlp=get_spacy_nlp(), device=device)
    return _FASTCOREF_MODELS[device]


def text_preparation(text, clean=True, coref_solver="fastcoref"):
    """
    Prepares the text for further processing by cleaning it and resolving coreferences.

    Parameters:
    text (str): The input text to prepare.
    coref_solver (str): The coreference solver to use. Currently supports 'stanza' and 'fastcoref'. If None, coreferences are not resolved.

    Returns:
    str: The prepared text with coreferences resolved.
    """

    # Clean the text
    cleaned_text = clean_text(text)

    if coref_solver is None:
        return cleaned_text

    # Solve coreferences
    resolved_text = solve_coreferences(cleaned_text, coref_solver=coref_solver)

    return resolved_text


def clean_text(text):
    """
    Cleans a given text by removing multiple spaces and content inside parentheses/brackets.

    Parameters:
    text (str): The input string to clean.
    
    Returns:
    str: The cleaned text.
    """
    # Remove anything inside () or [] in a single pass for better performance
    cleaned_text = re.sub(r"\(.*?\)|\[.*?\]", "", text)

    # Replace 2 or more whitespace characters with a single space
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

    return cleaned_text.strip()


def solve_coreferences(text, coref_solver="fastcoref"):
    """
    Resolves coreferences in a given text using either the Stanza library or fastcoref.

    Args:
        text (str): The input text to resolve coreferences in.
        coref_solver (str): The coreference solver to use. Currently supports 'stanza' and 'fastcoref'.
    Return:
        str: The text with coreferences resolved.
    """

    if coref_solver not in {"stanza", "fastcoref"}:
        raise ValueError(
            "Only stanza and fastcoref coreference solvers are supported at the moment."
        )

    if coref_solver == "stanza":
        # Load the Stanza pipeline
        stanzanlp = get_stanza_nlp()
        # Process the text
        doc = stanzanlp(text)
        output_text = stanza_solve_coreferences(doc)

    if coref_solver == "fastcoref":
        output_text = fastcoref_solve_coreferences(text)

    return output_text


def _apply_replacements(text, replacements):
    """
    Apply a list of ``{"start", "end", "replacement"}`` span replacements
    to *text*. Replacements are applied from the end of the string backwards
    (so earlier offsets stay valid) and overlapping spans are skipped.
    """
    replacements = sorted(replacements, key=lambda x: x["start"], reverse=True)
    last_applied_start = len(text) + 1
    resolved = text
    for repl in replacements:
        if repl["end"] > last_applied_start:
            # Overlaps a replacement already applied: skip to avoid
            # corrupting character offsets.
            continue
        resolved = (
            resolved[: repl["start"]] + repl["replacement"] + resolved[repl["end"] :]
        )
        last_applied_start = repl["start"]
    return resolved


def resolve_coref_prediction(pred_result, original_text):
    """
    Apply coreference replacements from a single fastcoref prediction result
    to *original_text* and return the resolved text.

    Mentions containing words in ``_COREFERENCE_NOUNS`` are replaced with the
    earliest mention of the cluster that contains none of those words. When
    the mention being replaced is a bare possessive pronoun (e.g. "my"), the
    replacement is turned into a genitive ("John" -> "John's") to keep the
    resolved text grammatical.
    """
    clusters_positions = pred_result.get_clusters(as_strings=False)
    clusters_strings = pred_result.get_clusters()

    replacements = []
    for cluster_idx, cluster in enumerate(clusters_positions):
        mentions_positions = cluster  # List of (start_char, end_char)
        mentions_texts = clusters_strings[cluster_idx]

        mentions = []
        for pos, text_mention in zip(mentions_positions, mentions_texts):
            start, end = pos
            mentions.append({"start": start, "end": end, "text": text_mention})

        # Identify mentions containing words in _COREFERENCE_NOUNS
        mentions_with_coref_nouns = [
            m
            for m in mentions
            if any(
                word in _COREFERENCE_NOUNS
                for word in re.findall(r"\b\w+\b", m["text"].lower())
            )
        ]

        if mentions_with_coref_nouns:
            # Find a replacement mention that does not contain any word in
            # _COREFERENCE_NOUNS
            replacement_mentions = [
                m
                for m in mentions
                if not any(
                    word in _COREFERENCE_NOUNS
                    for word in re.findall(r"\b\w+\b", m["text"].lower())
                )
            ]
            if not replacement_mentions:
                # If no replacement mention is available, skip this cluster
                continue
            # Prefer the earliest mention in the text
            replacement_mentions.sort(key=lambda m: (m["start"], -len(m["text"])))
            replacement_text = replacement_mentions[0]["text"]
            for mention in mentions_with_coref_nouns:
                replacement = replacement_text
                if mention["text"].strip().lower() in _POSSESSIVE_PRONOUNS:
                    replacement = replacement_text + "'s"
                replacements.append(
                    {
                        "start": mention["start"],
                        "end": mention["end"],
                        "replacement": replacement,
                    }
                )

    return _apply_replacements(original_text, replacements)


def fastcoref_solve_coreferences(text_to_resolve):
    """
    Replaces coreferent mentions with their representative texts.

    The fastcoref model is loaded **once** per process (see
    ``load_fastcoref_model``) so repeated calls are cheap. For large-batch
    processing prefer ``teanets.batch_extract.batch_coref_resolve()`` which
    performs a single forward pass on a list of texts and supports GPU.

    Args:
        text_to_resolve (str): The input text.

    Returns:
        str: The text with coreferences resolved.
    """
    model = load_fastcoref_model(device="cpu")

    # fastcoref expects a list of texts; depending on the version, predict()
    # may return a single CorefResult or a list of them.
    preds = model.predict(texts=[text_to_resolve])
    pred = preds[0] if isinstance(preds, list) else preds

    return resolve_coref_prediction(pred, text_to_resolve)


def stanza_solve_coreferences(doc):
    """
    Replaces coreferent mentions with their representative texts in the text reconstructed from the doc.

    Args:
        doc (stanza.Document): The processed document with coreference information.

    Returns:
        str: The text with coreferences resolved.
    """
    # Use doc.text to get the original text
    original_text = doc.text

    # List to hold the replacements (start_char, end_char, representative_text)
    replacements = []

    # Dictionary to keep track of active mentions per coreference chain
    active_mentions = {}

    # Iterate over sentences and words to build mentions
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.lower() not in _COREFERENCE_NOUNS:
                continue
            word_start = word.start_char
            word_end = word.end_char

            if word.coref_chains:
                # Get the coref_attachment with the lowest chain_idx
                min_coref_attachment = min(
                    word.coref_chains, key=lambda x: x.chain.index
                )
                chain_idx = min_coref_attachment.chain.index
                rep_text = min_coref_attachment.chain.representative_text

                if len(rep_text) > 15:
                    continue

                if word.text in rep_text:
                    continue

                # Start of a mention
                if min_coref_attachment.is_start:
                    active_mentions[chain_idx] = {
                        "start_char": word_start,
                        "end_char": word_end,
                        "rep_text": rep_text,
                    }
                # Continuation of a mention
                elif chain_idx in active_mentions:
                    active_mentions[chain_idx]["end_char"] = word_end

                # End of a mention
                if min_coref_attachment.is_end:
                    if chain_idx in active_mentions:
                        mention = active_mentions[chain_idx]
                        # Record the mention span and its representative text
                        replacements.append(
                            (
                                mention["start_char"],
                                mention["end_char"],
                                mention["rep_text"],
                            )
                        )
                        # Remove the mention from active mentions
                        del active_mentions[chain_idx]
            else:
                # Word is not part of any coreference chain; nothing to do
                pass

    # Possessive pronouns are replaced with the genitive form of the
    # representative text ("my" -> "John's") to keep the text grammatical.
    replacement_dicts = []
    for start_char, end_char, rep_text in replacements:
        span_text = original_text[start_char:end_char].strip().lower()
        if span_text in _POSSESSIVE_PRONOUNS:
            rep_text = rep_text + "'s"
        replacement_dicts.append(
            {"start": start_char, "end": end_char, "replacement": rep_text}
        )

    return _apply_replacements(original_text, replacement_dicts)
