#!/bin/bash
set -e

echo "=============================================="
echo "Initializing Neural Guitar Environment"
echo "=============================================="

# 0. Define Python Version (Using 3.9 for PyTorch compatibility)
PYTHON_EXEC="/usr/local/bin/python3.9"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "[ERROR] Python 3.9 not found at $PYTHON_EXEC"
    echo "Please check your python installation."
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON_EXEC --version)"

# 1. Create Virtual Environment
# Remove old venv if it exists and was using the wrong python version
if [ -d "venv" ]; then
    # Simple check: if current venv python isn't 3.9, nuke it.
    VENV_VER=$(./venv/bin/python --version 2>&1 || true)
    if [[ "$VENV_VER" != *"3.9"* ]]; then
        echo "[INFO] Removing incompatible venv ($VENV_VER)..."
        rm -rf venv
    fi
fi

if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment 'venv'..."
    $PYTHON_EXEC -m venv venv
else
    echo "[INFO] 'venv' (Python 3.9) already exists."
fi

# 2. Activate and Install
echo "[INFO] Installing dependencies from requirements.txt..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "=============================================="
echo "Setup Complete!"
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo "=============================================="
