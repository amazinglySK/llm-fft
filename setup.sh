#!/usr/bin/env bash
set -euo pipefail

GDRIVE_LINK="https://drive.google.com/file/d/1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE/view?usp=sharing"

VENV_DIR="llmfft"
TAR_NAME="nonmonash_datasets.tar.gz"
DATASET_DIR="lag-llama/datasets"

# choose python
if command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN=python3.10
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Error: No suitable python interpreter found (need python3)."
  exit 1
fi

echo "Using interpreter: $($PYTHON_BIN --version)"

echo "👉 Creating virtual environment in $VENV_DIR ..."
$PYTHON_BIN -m venv "$VENV_DIR"

echo "👉 Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "👉 Upgrading pip..."
python -m pip install --upgrade pip

echo "👉 Installing gdown..."
python -m pip install --upgrade gdown

# install requirements
if [ -f "lag-llama/requirements.txt" ]; then
  echo "👉 Installing requirements from lag-llama/requirements.txt ..."
  python -m pip install -r lag-llama/requirements.txt
else
  echo "⚠️ requirements.txt not found, skipping"
fi

# make datasets folder
mkdir -p "$DATASET_DIR"

echo "👉 Downloading dataset from Google Drive..."
if [[ "$GDRIVE_LINK" == http* ]]; then
  python -m gdown "$GDRIVE_LINK" -O "$TAR_NAME"
else
  python -m gdown --id "$GDRIVE_LINK" -O "$TAR_NAME"
fi

echo "👉 Extracting $TAR_NAME into $DATASET_DIR ..."
tar -xvzf "$TAR_NAME" -C "$DATASET_DIR"

echo "🧹 Removing $TAR_NAME ..."
rm -f "$TAR_NAME"

echo "✅ Setup complete!"
echo "To activate environment later: source $VENV_DIR/bin/activate"
