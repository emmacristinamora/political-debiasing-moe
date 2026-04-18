#!/bin/bash
#SBATCH --job-name=extract_activations
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_activations_%j.out
#SBATCH --error=logs/extract_activations_%j.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmamora2003@gmail.com

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-no_slurm}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"
echo "[info] partition=${SLURM_JOB_PARTITION:-unknown}"

# repo root — update this to the cluster path for political-debiasing-moe
REPO_ROOT="/home/3210604/projects/political-debiasing-moe"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p data/steering-vectors/raw_pairs
mkdir -p data/steering-vectors/validated_pairs
mkdir -p data/steering-vectors/activations
mkdir -p data/steering-vectors/vectors
mkdir -p data/steering-vectors/reports

# env — lt-proj is a conda env, not a venv
source "$REPO_ROOT/.venv/bin/activate"


# cache
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# sanity checks
if [ ! -f "src/03_extract_activations.py" ]; then
  echo "[error] src/03_extract_activations.py not found — aborting"
  exit 1
fi

if [ ! -f "config/config.yaml" ]; then
  echo "[error] config/config.yaml not found — aborting"
  exit 1
fi

if [ ! -s "data/steering-vectors/validated_pairs/economic_pairs_validated.jsonl" ]; then
  echo "[error] economic_pairs_validated.jsonl is missing or empty — aborting"
  exit 1
fi

if [ ! -s "data/steering-vectors/validated_pairs/social_pairs_validated.jsonl" ]; then
  echo "[error] social_pairs_validated.jsonl is missing or empty — aborting"
  exit 1
fi

# === ECONOMIC AXIS ===
echo ""
echo "[info] ===== Running economic axis ====="
python -u src/03_extract_activations.py --axis economic
echo "[info] ===== Economic axis done ====="

# === SOCIAL AXIS ===
echo ""
echo "[info] ===== Running social axis ====="
python -u src/03_extract_activations.py --axis social
echo "[info] ===== Social axis done ====="

echo ""
echo "[info] end=$(date)"
