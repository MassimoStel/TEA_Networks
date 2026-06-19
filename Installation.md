# TEA Networks — Installation

## Prerequisites

- **Python 3.9–3.12** — Python 3.13 and above are not yet supported. The spaCy 3.8 stack (`spacy-transformers`, `spacy-alignments`, `en_core_web_trf`) has no prebuilt wheels for newer versions, so installation would require building from source with a Rust compiler.
- **pip** package manager
- **Git** — currently the package is installed through this GitHub repository, so Git must be installed.

## Local development setup (cloned repository)

### Option A — Regular install (just use the library)

**Step 1.** Create and activate a virtual environment:

```bash
python -m venv teanets-env
teanets-env\Scripts\activate        # Windows
# source teanets-env/bin/activate   # Mac/Linux
```

**Step 2.** Install the package and the spaCy model:

```bash
pip install git+https://github.com/MassimoStel/TEA_Networks.git
python -m spacy download en_core_web_trf
```

**Step 3.** Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name teanets --display-name "Python (teanets)"
```

> Then, when you open a notebook, **select the `Python (teanets)` kernel** so it runs inside this environment — in Jupyter: *Kernel → Change kernel → Python (teanets)*; in VS Code: *Select Kernel → Python Environments → teanets-env*.

### Option B — Local development setup (cloned repository)

Use this if you want to run the notebooks in `Docs & Guides/` or modify the library source code.

**Step 1.** Clone the repository and enter the folder:

```bash
git clone https://github.com/MassimoStel/TEA_Networks.git
cd TEA_Networks
```

**Step 2.** Create and activate a virtual environment:

```bash
python -m venv teanets-env
teanets-env\Scripts\activate        # Windows
# source teanets-env/bin/activate   # Mac/Linux
```

**Step 3.** Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

> **Why `pip install -e . --no-deps`?** This installs `teanets` in editable mode directly from the local source code, so any changes you make to the library are immediately reflected without reinstalling. Without this step, `import teanets` raises `ModuleNotFoundError` when running notebooks outside the repository root. The `--no-deps` flag skips reinstalling dependencies already covered by `requirements.txt`.

**Step 4.** Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name teanets --display-name "Python (teanets)"
```

> Then, when you open a notebook, **select the `Python (teanets)` kernel** so it runs inside this environment — in Jupyter: *Kernel → Change kernel → Python (teanets)*; in VS Code: *Select Kernel → Python Environments → teanets-env*.
