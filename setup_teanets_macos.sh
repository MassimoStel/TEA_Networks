#!/usr/bin/env bash
# ============================================================
# TEA Networks / teanets macOS self-installing setup
# Mirrors setup_teanets_linux.sh, following Installation.md
# Method A.
#
# What it does:
#   1. Checks for Python 3.9-3.12 and Git; suggests Homebrew packages if missing.
#   2. Creates/updates a Python virtual environment: $HOME/teanets/teanets-env
#   3. Installs teanets from GitHub, downloads en_core_web_trf, and
#      registers a Jupyter kernel.
#
# macOS has no CUDA support, so the installation is always CPU-only
# and there is no --cpu switch.
# ============================================================

set -u

INSTALL_DIR="$HOME/teanets"
ENV_DIR="$INSTALL_DIR/teanets-env"
KERNEL_NAME="teanets"
KERNEL_DISPLAY="Python (teanets)"
REPO_URL="https://github.com/MassimoStel/TEA_Networks.git"

echo "============================================================"
echo "TEA Networks macOS setup: venv + CPU install + Jupyter kernel"
echo "============================================================"
echo

fatal() {
    echo
    echo "ERROR: $1"
    echo "Setup failed. Scroll up to see the first error."
    exit 1
}

# ------------------------------------------------------------
# Prerequisite checks
# ------------------------------------------------------------
PYTHON_CMD=""
for v in 3.12 3.11 3.10 3.9; do
    if command -v "python$v" >/dev/null 2>&1; then
        PYTHON_CMD="python$v"
        break
    fi
done
if [ -z "$PYTHON_CMD" ] && command -v python3 >/dev/null 2>&1; then
    if python3 -c 'import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)' >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    fi
fi
if [ -z "$PYTHON_CMD" ]; then
    echo "Python 3.9-3.12 was not found."
    echo "Install it with Homebrew (https://brew.sh):"
    echo "  brew install python@3.12"
    fatal "Python 3.9-3.12 is required."
fi
echo "Found supported Python:"
echo "  $("$PYTHON_CMD" --version 2>&1) via $PYTHON_CMD"

if command -v git >/dev/null 2>&1; then
    echo "Found Git."
else
    echo "Git was not found."
    echo "Install it with the Xcode Command Line Tools:"
    echo "  xcode-select --install"
    echo "or with Homebrew:"
    echo "  brew install git"
    fatal "Git is required for pip install from GitHub."
fi

mkdir -p "$INSTALL_DIR" || fatal "Could not create $INSTALL_DIR."

echo
echo "Using installation folder:"
echo "  $INSTALL_DIR"
echo "Virtual environment:"
echo "  $ENV_DIR"
echo

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "Creating virtual environment with: $PYTHON_CMD"
    "$PYTHON_CMD" -m venv "$ENV_DIR" || fatal "Virtual environment creation failed."
else
    echo "Reusing existing virtual environment."
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate" || fatal "Could not activate the virtual environment."

if ! python -c 'import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)' >/dev/null 2>&1; then
    echo "The existing virtual environment does not use Python 3.9-3.12."
    echo "Delete this folder and run the installer again:"
    echo "  $ENV_DIR"
    fatal "Unsupported Python version in the virtual environment."
fi

echo "Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel || fatal "A pip command failed."

# ------------------------------------------------------------
# teanets installation
# ------------------------------------------------------------
echo
echo "Installing teanets from GitHub..."
python -m pip install --no-cache-dir "git+$REPO_URL" || fatal "A pip command failed."

echo
echo "Downloading the transformer English spaCy model: en_core_web_trf"
python -m spacy download en_core_web_trf || fatal "A pip command failed."

echo
echo "Installing Jupyter kernel support..."
python -m pip install --upgrade ipykernel jupyterlab notebook || fatal "A pip command failed."

echo
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY" || fatal "Kernel registration failed."

echo
echo "Verifying teanets import and spaCy model..."
python -c "import teanets as tea; print('teanets import OK')" || fatal "teanets import failed."
python -c "import spacy; spacy.load('en_core_web_trf'); print('spaCy en_core_web_trf load OK')" || fatal "spaCy model load failed."

echo
echo "Checking package consistency..."
if ! python -m pip check; then
    echo "WARNING: pip check reported one or more dependency notices. The installer will still finish."
fi

echo
echo "============================================================"
echo "Setup complete."
echo
echo "Installation folder:"
echo "  $INSTALL_DIR"
echo
echo "To activate this environment later:"
echo "  source \"$ENV_DIR/bin/activate\""
echo
echo "In Jupyter or VS Code, choose kernel:"
echo "  $KERNEL_DISPLAY"
echo "============================================================"
