#!/bin/bash
#SBATCH --job-name=hein_project
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/hein_project_%j.out
#SBATCH --error=logs/hein_project_%j.err
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
mkdir -p data/external/hein_dwnominate

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"

# ── inputs / outputs ───────────────────────────────────────────────────────────

DATASET="data/external/hein_dwnominate/legislator_dataset.jsonl"
PROJECTIONS="data/external/hein_dwnominate/compass_projections.jsonl"
HEIN_DIR="data/external/hein-daily"

# ── sanity checks ──────────────────────────────────────────────────────────────

for required in src/14_hein_build_dataset.py src/15_hein_project_compass.py; do
  if [ ! -f "$required" ]; then
    echo "[error] ${required} not found — aborting"
    exit 1
  fi
done

# ── stage 14: build the labeled dataset (CPU, only if not already staged) ──────
# The per-legislator dataset is small (~21 MB) and is normally built once and
# rsynced over. If it is missing, rebuild it here — that needs the raw
# hein-daily corpus and the Voteview CSV to be present.

if [ -s "$DATASET" ]; then
  echo "[info] stage 14 — legislator dataset already present, skipping build"
elif [ -d "$HEIN_DIR" ]; then
  echo "[info] stage 14 — building legislator dataset from raw hein-daily"
  python -u src/14_hein_build_dataset.py
else
  echo "[error] ${DATASET} missing and ${HEIN_DIR} absent — rsync one of them first"
  exit 1
fi

# ── stage 15: project legislator corpora onto the compass (GPU) ────────────────
# Loads Mistral-7B-v0.1, splits each corpus into token windows, mean-pools the
# steering layers and projects onto the economic and social directions.

echo "[info] stage 15 — projecting legislator corpora onto the compass"

python -u src/15_hein_project_compass.py \
  --dataset "$DATASET" \
  --output  "$PROJECTIONS" \
  --device  cuda

echo ""
echo "[info] end=$(date)"
echo "[info] projections: ${PROJECTIONS}"
echo "[info] next: run src/16_hein_dwnominate_analysis.py (CPU)"
