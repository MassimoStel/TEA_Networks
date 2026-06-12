<p align="center">
  <img src="./logofis.png" alt="FIS logo" height="180">
  <img src="./PENSO_Logo.png" alt="PENSO logo" height="240">
</p>

<p align="center">
  FIS2 Project PENSO - Funded by MUR - CogNosco Lab, University of Trento
</p>

<h1 align="center">
  Target-Event-Agent Networks in NLP for understanding "who" "did" "what" in narratives
</h1>

<p align="center">
  🤖 LLMs · 🧮 NLP · 📊 Library 🕸️ Complex Networks 📚 Text Analysis 
</p>


A Python library that implements a novel framework to measure bias through its actors, actions and consequences in English.


## Description
Target-Event-Agent cognitive networks are a novel framework to measure bias through its actors, actions and consequences. These semantic/syntactic multilayer networks are composed of 3 layers:
- **Agent**: Subjects/actors
- **Event**: Verbs/actions
- **Target**: Objects/consequences


<p align="center">
  <img src="tea logo.png" alt="TEA" width="200"/>
</p>

Part-of-speech tagging is determined by an AI reading each sentence in a text (spaCy). Inter-layer connections are established with a rule-based approach applied on the syntactic parsing of an AI (spaCy). Intra-layer connections are semantic and established only if two words are synonyms (e.g. father and dad), highlighted in green. Like in textual forma mentis networks (Stella, PeerJ Comp. Sci, 2020), individual concepts are labelled as “positive” (cyan), “negative” (red) and “neutral” (gray) according to Vader Sentiment Analysis. Inter-layer paths indicate “target event agent” - i.e. which actions and which consequences were portrayed by specific agents in texts. Whereas tools such as EmoAtlas can give general results about the overall context of biased perceptions, Target-Event-Agent networks can complement TFMNs by providing a focus on actors, actions and consequences.

This framework only works for the English language.


<p align="center">
  <img src="tea_overview.jpg"/>
</p>


## Methodological Revisions

### 1. Passive Voice Handling
The TEA library correctly distinguishes between the semantic agent and the grammatical subject in passive constructions. 
- **Passive WITH Agent** (*"The physician and the doctor were praised by the hospital."*): The agent (*hospital*) is correctly identified as the **Agent**, while the physician and doctor (*physician*, *doctor*) are mapped to the **Target**.
- **Passive WITHOUT Agent** (*"The window was broken"*): If no agent is present, the patient remains in the **Agent** position as the best available approximation.

| Sentence | Agent | Target |
|:---------|:------------|:--------------|
| *The physician and the doctor were praised by the hospital.* | hospital | physician, doctor |
| *Mark and Rose go to the park.* | Mark, Rose | park |
| *The hacker destroyed the sensitive database.* | hacker | sensitive database |

### 2. SVO Extraction Validation
To ensure accuracy, we validated the extraction logic against a **Gold Standard** of 122 manually annotated sentences (`data/gold_standard_svo.csv`) covering active/passive voices, imperatives, and complex clauses, plus the 1,150-sentence PassivePy crowd-sourced dataset for passive voice detection (`data/passive_gold.csv`).
- **Metric-based Evaluation**: We report per-role Accuracy for each component (Agent, Event, Target) and per-class accuracy for passive voice detection (see `Docs & Guides/Validation.ipynb`).
- **Benchmark Driven**: The implementation is continuously tested against manually annotated data to avoid regressions.

---
### Features

- **SVO Extraction**: Extracts subjects (who), verbs(did), and objects(what) from sentences.
- **Coreference Resolution**: Handles pronouns and other references using either `stanza` or `fastcoref`.
- **Valence Analysis**: Determines the sentiment (positive, negative, neutral) of words using Vader.
- **Graph Visualization**: Visualizes SVO relationships using NetworkX and Matplotlib.
- **Hypergraph extraction**: Handles the exporting of the target-event-agent relationships as a pandas dataframe.
- **Semantic enrichment**: Synonyms are included in the SVO extraction and in the visualization.



---
## Installation

Currently, the package can be installed through this Github repository. Note that this requires [Git](https://git-scm.com/) to be installed.

```bash
pip install git+https://github.com/MassimoStel/TEA_Networks.git
python -m spacy download en_core_web_trf
```

### Prerequisites

- Python >=3.9 and **<=3.12** (Python 3.13/3.14 are not supported yet: the spaCy 3.8 stack — `spacy-transformers`, `spacy-alignments`, `en_core_web_trf` — has no prebuilt wheels for them, so installation would require building from source with a Rust compiler)
- pip package manager

### Local development setup (cloned repository)

If you cloned this repository (e.g. to run the notebooks in `Docs & Guides/` or to modify the library), installing `requirements.txt` is not enough: that only installs the **dependencies**. You also need to install the `teanets` package itself, in editable mode:

```bash
# with uv (https://docs.astral.sh/uv/)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e . --no-deps

# or with plain pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Without the `pip install -e .` step, `import teanets` raises `ModuleNotFoundError` whenever Python is not launched from the repository root (which is exactly what happens inside the `Docs & Guides/` notebooks).

**Jupyter users**: make sure the notebook kernel uses the virtual environment's Python (`.venv/bin/python`), not the system one. In VS Code: "Select Kernel" → Python Environments → `.venv`. With classic Jupyter you can register the kernel with:

```bash
python -m ipykernel install --user --name teanets --display-name "Python 3.12 (teanets)"
```

---
## Usage and guides

For detailed documentation and examples, please refer to the folder 'Docs & Guides' or check out the Google Colab demo.


```python
import teanets as tea

text = """The physician and the doctor were praised by the hospital."""

svo = tea.extract_svos_from_text(text)
display(svo)
```

| Node 1 | TEA | Node 2 | TEA2 | Hypergraph | Semantic-Syntactic | svo_id | passive_approx | is_passive |
|:---|:---|:---|:---|:---|---:|---:|---:|---:|
| hospital | Agent | praise | Event | [[('hospital', [])], ['praise'], [('physician'... | 0 | 0 | 0 | 1 |
| praise | Event | physician | Target | [[('hospital', [])], ['praise'], [('physician'... | 0 | 0 | 0 | 1 |
| praise | Event | doctor | Target | [[('hospital', [])], ['praise'], [('physician'... | 0 | 0 | 0 | 1 |
| physician | Target | doctor | Target | [[('hospital', [])], ['praise'], [('physician'... | 0 | 0 | 0 | 1 |
| doctor | Target | physician | Target | N/A | 1 | N/A | 0 | 0 |

```python
tea.plot_svo_graph(svo)
```
<p align="center">
  <img src="tea_quickstart_passive.jpg" width="700"/>
</p>

### Multi-sentence example

A richer example showing multiple agents and verbs across sentences:

```python
text = """The system detected an error. Finally, the loaf was sold. The mayor praised the volunteers and thanked the community."""

svo = tea.extract_svos_from_text(text)
display(svo)
```

| Node 1 | TEA | Node 2 | TEA2 | Hypergraph | Semantic-Syntactic | svo_id | passive_approx | is_passive |
|:---|:---|:---|:---|:---|---:|---:|---:|---:|
| system | Agent | detect | Event | [[('system', [])], ['detect'], [('error', [])]... | 0 | 0 | 0 | 0 |
| detect | Event | error | Target | [[('system', [])], ['detect'], [('error', [])]... | 0 | 0 | 0 | 0 |
| loaf | Agent | finally sell | Event | [[('loaf', [])], ['finally sell'], []] | 0 | 1 | 1 | 1 |
| mayor | Agent | praise | Event | [[('mayor', [])], ['praise'], [('volunteer', [... | 0 | 2 | 0 | 0 |
| praise | Event | volunteer | Target | [[('mayor', [])], ['praise'], [('volunteer', [... | 0 | 2 | 0 | 0 |
| mayor | Agent | thank | Event | [[('mayor', [])], ['thank'], [('community', []... | 0 | 3 | 0 | 0 |
| thank | Event | community | Target | [[('mayor', [])], ['thank'], [('community', []... | 0 | 3 | 0 | 0 |

> The ``passive_approx`` column flags rows where the *Agent* slot does not contain
> a true semantic agent but rather a patient placed there as best-available
> approximation (passive without by-phrase, e.g. *"the report was reviewed"*).
> Set to ``1`` only for those Agent–Event and Agent–Agent edges; always ``0``
> elsewhere. Older outputs without this column are still fully supported by the
> analytics and plotting helpers (treated as ``0``).

```python
tea.plot_svo_graph(svo)
```
<p align="center">
  <img src="tea_multisentence.jpg?v=2" width="700"/>
</p>

### Generating TEA Figures

A step-by-step tutorial for generating TEA network figures: [Generating TEA Figures](https://github.com/MassimoStel/TEA_Networks/blob/main/Docs%20%26%20Guides/Generating_TEA_Figures.ipynb)

### Starting Guide

You can access the Starting Guide here: [Starting Guide](https://github.com/MassimoStel/TEA_Networks/blob/main/Docs%20%26%20Guides/Starting%20Guide.ipynb)

The starting guide features a more complete description of the package and a usage guide.


### Google Colab Demo

You can access the Google Colab demo here: [Target-Event-Agent Demo](https://colab.research.google.com/drive/1fs73YQkPu55C_Cwt2lYhuvfc0aBXjnQo?usp=sharing).


---
## Citation

Franchini, S., Carrillo, A., De Duro, E. S., Improta, R., Ardebili, A. A., & Stella, M. (2026). TEA Nets combine AI and cognitive network science to model actors, actions, and consequences in text [Preprint]. arXiv. https://github.com/MassimoStel/TEA_Networks

## References:
- *Stella, M. (2020). Text-mining forma mentis networks reconstruct public perception of the STEM gender gap in social media. PeerJ Computer Science, 6, e295.*
- *Hutto, C., & Gilbert, E. (2014, May). Vader: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings of the international AAAI conference on web and social media (Vol. 8, No. 1, pp. 216-225).*

## Created within CogNosco Lab - Check our Research: https://cognosco.dipsco.unitn.it/

<p align="center">
  <img src="./NewLogoCognosco.png" alt="FIS logo" height="260">
  <img src="./PENSO_Logo.png" alt="PENSO logo" height="260">
</p>

## Acknowledgements

This work is part of the PENSO project, supported by the Ministero dell'Università e della Ricerca (MUR) according to Decreto N. 23178 of 10 dicembre 2024 — Bando FIS 2. The authors acknowledge support from CALCOLO, funded by Fondazione VRT, for support with the computational infrastructure simulating LLMs.


## License

This project is licensed under the BSD 3-Clause License.
