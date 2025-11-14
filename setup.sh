#!/usr/bin/env bash
set -euo pipefail

GDRIVE_LINK="https://drive.google.com/file/d/1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE/view?usp=sharing"

CONDA_ENV_NAME="llmfft"
TAR_NAME="nonmonash_datasets.tar.gz"
DATASET_DIR="lag-llama/datasets"

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Please install conda first."
  exit 1
fi

echo "Using conda: $(conda --version)"

echo "👉 Creating conda environment '$CONDA_ENV_NAME' with Python 3.10.8..."
conda create -n "$CONDA_ENV_NAME" python=3.10.8 -y

echo "👉 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

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
  python -m gdown "$GDRIVE_LINK" -O "$TAR_NAME" --fuzzy
else
  python -m gdown --id "$GDRIVE_LINK" -O "$TAR_NAME" --fuzzy
fi

echo "👉 Extracting $TAR_NAME into $DATASET_DIR ..."
tar -xvzf "$TAR_NAME" -C "$DATASET_DIR"

echo "🧹 Removing $TAR_NAME ..."
rm -f "$TAR_NAME"

echo "✅ Setup complete!"
echo "To activate environment later: conda activate $CONDA_ENV_NAME"
