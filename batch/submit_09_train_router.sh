#!/bin/bash
#SBATCH --job-name=train_router
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/train_router_%j.out
#SBATCH --error=logs/train_router_%j.err
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
mkdir -p data/router/checkpoints
mkdir -p data/router/reports

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"

# ── sanity checks ─────────────────────────────────────────────────────────────

if [ ! -f "src/router_training/train_pipeline.py" ]; then
  echo "[error] src/router_training/train_pipeline.py not found — aborting"
  exit 1
fi

if [ ! -f "src/router_training/trainer.py" ]; then
  echo "[error] src/router_training/trainer.py not found — aborting"
  exit 1
fi

if [ ! -f "config/config.yaml" ]; then
  echo "[error] config/config.yaml not found — aborting"
  exit 1
fi

if [ ! -f "data/router/hidden.pt" ]; then
  echo "[error] data/router/hidden.pt not found — run features.py first"
  exit 1
fi

# splitter writes to data/router/splits/{train,val,test}/records.jsonl
for split in train val test; do
  if [ ! -s "data/router/splits/${split}/records.jsonl" ]; then
    echo "[error] data/router/splits/${split}/records.jsonl missing or empty — run splitter.py first"
    exit 1
  fi
done

echo "[info] all pre-flight checks passed"
echo "[info] start=$(date)"

# ── train calibrated router head ───────────────────────────────────────────────
# nn.Linear(4096 → 4): ~16K params trained on ~27K examples for 20 epochs.
# Expected runtime: under 5 minutes on H200.

python -u src/router_training/train_pipeline.py \
  --config              config/config.yaml \
  --train-records-path  data/router/splits/train/records.jsonl \
  --val-records-path    data/router/splits/val/records.jsonl \
  --test-records-path   data/router/splits/test/records.jsonl \
  --hidden-path         data/router/hidden.pt \
  --output-path         data/router/checkpoints/calibrated_router.pt \
  --trainer-report-path data/router/reports/train_report.json \
  --pipeline-report-path data/router/reports/pipeline_train_report.json \
  --device              cuda

echo ""
echo "[info] end=$(date)"
echo "[info] checkpoint saved to data/router/checkpoints/calibrated_router.pt"
echo "[info] reports in data/router/reports/"
