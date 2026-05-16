#!/bin/bash
#SBATCH --job-name=compass_center
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/compass_center_%j.out
#SBATCH --error=logs/compass_center_%j.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmamora2003@gmail.com

set -euo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-no_slurm}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"
echo "[info] partition=${SLURM_JOB_PARTITION:-unknown}"

REPO_ROOT="/home/3210604/projects/political-debiasing-moe"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p data/compass_center

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"

# ── inputs / outputs ───────────────────────────────────────────────────────────

DATASET="data/neutral_prompts.jsonl"
OUTPUT="data/compass_center/center.json"

# ── sanity checks ──────────────────────────────────────────────────────────────

if [ ! -f "$DATASET" ]; then
  echo "[error] dataset not found: ${DATASET} — update the path and resubmit"
  exit 1
fi

if [ ! -f "src/18_compass_center.py" ]; then
  echo "[error] src/18_compass_center.py not found — aborting"
  exit 1
fi

# ── run ────────────────────────────────────────────────────────────────────────

echo "[info] projecting neutral prompts onto the compass"

python -u src/18_compass_center.py \
  --dataset    "$DATASET" \
  --output     "$OUTPUT"  \
  --device     cuda

echo ""
echo "[info] end=$(date)"
echo "[info] center written to: ${OUTPUT}"
