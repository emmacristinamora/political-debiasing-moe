#!/bin/bash
#SBATCH --job-name=train_experts
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_experts_%j.out
#SBATCH --error=logs/train_experts_%j.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmamora2003@gmail.com

# Do NOT use set -e here: each quadrant runs independently and a single failure
# should not abort the whole job.  Failures are collected and reported at the end.
set -uo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-no_slurm}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"
echo "[info] partition=${SLURM_JOB_PARTITION:-unknown}"

REPO_ROOT="/home/3210604/projects/political-debiasing-moe"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p data/experts/training-outputs

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# ── sanity checks ─────────────────────────────────────────────────────────────

if [ ! -f "src/07_train_experts.py" ]; then
  echo "[error] src/07_train_experts.py not found — aborting"
  exit 1
fi

if [ ! -f "config/config.yaml" ]; then
  echo "[error] config/config.yaml not found — aborting"
  exit 1
fi

TRAIN_VALIDATE_DIR="data/experts/train-validate"

for quadrant in right_auth left_auth left_lib right_lib; do
  for split in train val_indist val_source val_topic; do
    f="${TRAIN_VALIDATE_DIR}/${quadrant}/${split}.jsonl"
    if [ ! -s "$f" ]; then
      echo "[error] ${f} is missing or empty — run 06_validate_experts_datasets.py first"
      exit 1
    fi
  done
done

echo "[info] all pre-flight checks passed"
echo "[info] training 4 quadrants × 3 seeds = 12 runs sequentially"

# ── quadrant runner ────────────────────────────────────────────────────────────
# Runs all seeds for one quadrant, logs elapsed time, records any failure.
# Usage: run_quadrant <quadrant>

FAILED_QUADRANTS=""

run_quadrant() {
  local quadrant="$1"

  echo ""
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  quadrant=${quadrant}"
  echo "============================================================"

  local start_ts
  start_ts=$(date +%s)

  if python -u src/07_train_experts.py --quadrant "${quadrant}"; then
    local end_ts elapsed
    end_ts=$(date +%s)
    elapsed=$(( end_ts - start_ts ))
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE   quadrant=${quadrant}  elapsed=${elapsed}s"
    echo "============================================================"
  else
    local exit_code=$?
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED quadrant=${quadrant}  exit=${exit_code}"
    echo "============================================================"
    FAILED_QUADRANTS="${FAILED_QUADRANTS} ${quadrant}"
  fi
}

# ── quadrants ─────────────────────────────────────────────────────────────────

run_quadrant right_auth

run_quadrant left_auth

run_quadrant left_lib

run_quadrant right_lib

# ── final report ──────────────────────────────────────────────────────────────

echo ""
echo "[info] end=$(date)"

if [ -n "${FAILED_QUADRANTS}" ]; then
  echo "[error] the following quadrants failed:${FAILED_QUADRANTS}"
  echo "[error] check the log above for their tracebacks"
  exit 1
else
  echo "[info] all 4 quadrants completed successfully"
fi
