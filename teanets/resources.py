
def _valences(language="english"):
    """
    Return the (positive, negative, ambivalent) word lists for *language*.

    NOTE: these lexicons are currently not used by the pipeline (node valence
    is computed with VADER, see ``nlp_utils.compute_valence``). They are kept
    available for custom analyses.
    """
    if language == "english":
        from teanets.valence.english import _positive, _negative, _ambivalent
    elif language == "italian":
        from teanets.valence.italian import _positive, _negative, _ambivalent
    else:
        raise ValueError(
            f"Unsupported language: {language!r}. "
            "Supported languages are 'english' and 'italian'."
        )

    return _positive, _negative, _ambivalent


_VAGUE_ADVMODS = {
    "only",
    "respectively",
    "actually",
    "basically",
    "generally",
    "normally",
    "usually",
    "approximately",
    "roughly",
    "virtually",
    "exactly",
    "precisely",
    "literally",
    "effectively",
    "essentially",
    "more",
}

_VAGUE_AUX = {"be", "have", "to", "shall", "would", "should", "is"}

_VAGUE_ADJ = {
    "certain",
    "significant",
    "some",
    "various",
    "several",
    "numerous",
    "other",
    "different",
    "specific",
    "typical",
    "important",
    "general",
    "usual",
    "frequent",
    "considerable",
    "sufficient",
    "major",
    "relevant",
    "adequate",
    "appropriate",
    "minor",
    "potential",
    "many",
    "few",
    "particular",
    "additional",
    "existing",
    "regular",
    "normal",
    "ordinary",
    "standard",
    "common",
    "average",
    "basic",
    "random",
    "miscellaneous",
    "diverse",
    "assorted",
    "respective",
    "relative",
    "suitable",
    "proper",
    "given",
    "multiple",
    "sundry",
    "arbitrary",
    "undefined",
    "unspecified",
    "approximate",
    "estimated",
    "nominal",
    "generic",
    "universal",
    "conventional",
    "customary",
    "moderate",
    "reasonable",
    "notable",
    "substantial",
    "possible",
    "probable",
    "likely",
    "unlikely",
    "primary",
    "secondary",
    "tertiary",
    "minimal",
    "maximal",
    "optimal",
    "various",
    "countless",
    "innumerable",
    "limited",
    "extensive",
    "abundant",
    "scarce",
    "ample",
    "sparse",
    "much",
    "current",
}

_COREFERENCE_NOUNS = {
    "he",
    "she",
    "they",
    "it",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "theirs",
    "this",
    "object",
    "person",
    "thing",
    "city",
    "item",
    "place",
    "country",
    "animal",
    "plant",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "its",
}
