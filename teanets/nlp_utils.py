import spacy
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_nlp_spacy = None
_nlp_stanza = None
_vader_analyzer = None


def _get_vader_analyzer():
    """Return a process-wide singleton SentimentIntensityAnalyzer.
    Avoids re-instantiating the analyzer (which loads its lexicon)
    on every call to ``compute_valence``.
    """
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def spacynlp(text):
    """
    Process the text using the spaCy NLP pipeline.
    """
    nlp = get_spacy_nlp()

    return nlp(text)


_wordnet_checked = False

def ensure_wordnet_downloaded():
    """Ensure WordNet exists, silently, without repeated noisy downloads."""
    global _wordnet_checked
    if _wordnet_checked:
        return

    import nltk
    from nltk.data import find

    for resource in ("corpora/wordnet", "corpora/wordnet.zip"):
        try:
            find(resource)
            _wordnet_checked = True
            return
        except LookupError:
            pass

    nltk.download("wordnet", quiet=True)
    _wordnet_checked = True


def get_spacy_nlp():
    global _nlp_spacy
    if _nlp_spacy is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_spacy = spacy.load("en_core_web_trf")
    return _nlp_spacy


def get_stanza_nlp():
    import stanza

    global _nlp_stanza
    if _nlp_stanza is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_stanza = stanza.Pipeline(
                lang="en", processors="tokenize,coref", verbose=False
            )
    return _nlp_stanza


def compute_valence(text):
    """
    Compute the valence of a given text based on positive and negative words using Vader Sentiment Analysis.
    Args:
        text (str): The text to analyze.

    Returns:
        str: 'positive', 'negative', or 'neutral'
    """
    analyzer = _get_vader_analyzer()

    vs = analyzer.polarity_scores(text)

    if vs["compound"] > 0.05:
        return "positive"
    elif vs["compound"] < -0.05:
        return "negative"
    else:
        return "neutral"
