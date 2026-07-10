#!/usr/bin/env bash
# ============================================================
# TEA Networks / teanets Linux self-installing setup
# Mirrors setup_teanets_windows.bat, following Installation.md
# Method A and cuda-installation.md.
#
# What it does:
#   1. Checks for Python 3.9-3.12 and Git; suggests distro packages if missing.
#   2. Creates/updates a Python virtual environment: $HOME/teanets/teanets-env
#   3. Checks for CUDA-capable NVIDIA hardware (nvidia-smi or lspci).
#   4. If CUDA is available, installs cupy-cuda12x first and verifies it.
#      (On Linux the cupy-cuda12x wheels bundle the CUDA runtime, so only
#      the NVIDIA driver is required — no CUDA Toolkit installation.)
#   5. Installs teanets from GitHub, downloads en_core_web_trf, and
#      registers a Jupyter kernel.
#
# Optional switch:
#   ./setup_teanets_linux.sh --cpu     Force CPU-only installation and skip CUDA.
# ============================================================

set -u

INSTALL_DIR="$HOME/teanets"
ENV_DIR="$INSTALL_DIR/teanets-env"
KERNEL_NAME="teanets"
KERNEL_DISPLAY="Python (teanets)"
REPO_URL="https://github.com/MassimoStel/TEA_Networks.git"
FORCE_CPU=0

case "${1:-}" in
    --cpu|-cpu|/cpu) FORCE_CPU=1 ;;
esac

echo "============================================================"
echo "TEA Networks Linux setup: venv + CUDA-aware install + Jupyter kernel"
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
    echo "Install it with your package manager, for example on Ubuntu/Debian:"
    echo "  sudo apt-get update && sudo apt-get install python3.11 python3.11-venv"
    fatal "Python 3.9-3.12 is required."
fi
echo "Found supported Python:"
echo "  $("$PYTHON_CMD" --version 2>&1) via $PYTHON_CMD"

if command -v git >/dev/null 2>&1; then
    echo "Found Git."
else
    echo "Git was not found."
    echo "Install it with your package manager, for example on Ubuntu/Debian:"
    echo "  sudo apt-get update && sudo apt-get install git"
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
    if ! "$PYTHON_CMD" -m venv "$ENV_DIR"; then
        echo "Could not create the virtual environment."
        echo "On Ubuntu/Debian the venv module may be missing; install it with:"
        echo "  sudo apt-get install ${PYTHON_CMD}-venv"
        fatal "Virtual environment creation failed."
    fi
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
# CUDA detection and setup
# ------------------------------------------------------------
CUDA_CAPABLE=0
GPU_NAME=""

if [ "$FORCE_CPU" = "1" ]; then
    echo
    echo "CPU-only mode requested. Skipping CUDA detection and CuPy installation."
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
        if [ -n "$GPU_NAME" ]; then
            CUDA_CAPABLE=1
            NVIDIA_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
        fi
    fi
    if [ "$CUDA_CAPABLE" = "0" ] && command -v lspci >/dev/null 2>&1; then
        GPU_NAME="$(lspci 2>/dev/null | grep -i 'vga\|3d' | grep -i nvidia | head -n 1 | sed 's/.*: //')"
        if [ -n "$GPU_NAME" ]; then
            echo
            echo "An NVIDIA GPU was detected, but nvidia-smi is not available."
            echo "GPU: $GPU_NAME"
            echo "Install the NVIDIA driver first if you want GPU acceleration."
            echo "Proceeding with CPU-only installation."
        fi
    fi
fi

if [ "$CUDA_CAPABLE" = "1" ]; then
    echo
    echo "CUDA-capable NVIDIA hardware detected."
    echo "GPU: $GPU_NAME"
    [ -n "${NVIDIA_DRIVER_VERSION:-}" ] && echo "NVIDIA driver: $NVIDIA_DRIVER_VERSION"
    echo
    echo "Installing CuPy for CUDA 12.x in the active teanets environment..."
    # Pinned to the same version as requirements.txt: newer CuPy releases
    # (14.x) fail CUDA detection on some driver setups where 13.0.0 works.
    CUPY_OK=1
    python -m pip install --no-cache-dir cupy-cuda12x==13.0.0 || CUPY_OK=0
    if [ "$CUPY_OK" = "1" ]; then
        echo "Verifying CuPy can see the CUDA runtime..."
        python -c "import cupy as cp; print('CuPy runtime version:', cp.cuda.runtime.runtimeGetVersion()); print('CuPy GPU test:', cp.array([1,2,3]))" || CUPY_OK=0
    fi
    if [ "$CUPY_OK" = "0" ]; then
        echo
        echo "WARNING: The CUDA installation/verification branch failed."
        echo "You can still install teanets CPU-only now, or quit and fix CUDA manually."
        printf "Continue CPU-only installation? [C=continue, Q=quit] "
        read -r answer
        case "$answer" in
            [Qq]*) fatal "Aborted by user after CUDA failure." ;;
        esac
        CUDA_CAPABLE=0
    fi
else
    if [ "$FORCE_CPU" != "1" ]; then
        echo
        echo "No CUDA-capable NVIDIA setup was verified. Proceeding with CPU-only Method A."
    fi
fi

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
