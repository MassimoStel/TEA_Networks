from .svo_extraction import *
from .svo_extraction import is_sentence_passive
from .nlp_utils import get_spacy_nlp, get_stanza_nlp
from .svo_validation import validate_svo, validate_passive, extract_svo_dep
from .teaplot import plot_svo_graph
from .analytics import (
    tea_degree_centrality_overview,
    tea_weighted_degree_centrality,
    drop_approximated_svos,
)
from . import batch_extract

__version__ = "0.3.1"
