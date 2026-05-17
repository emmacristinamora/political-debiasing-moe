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
# neutral_prompts.jsonl must follow the R5 schema:
#   {id, category, subtype, topic, text}  where category ∈ {apolitical, generic_task}

DATASET="data/neutral_prompts.jsonl"
PAIRS_DIR="data/steering-vectors/validated_pairs"
OUTPUT="data/compass_center/center.json"
REPORT="data/compass_center/validation_report.json"

# ── sanity checks ──────────────────────────────────────────────────────────────

for required in "$DATASET" \
                "${PAIRS_DIR}/economic_pairs_validated.jsonl" \
                "${PAIRS_DIR}/social_pairs_validated.jsonl" \
                "src/18_compass_center.py"; do
  if [ ! -f "$required" ]; then
    echo "[error] required file not found: ${required}"
    exit 1
  fi
done

# quick schema check — must have the R5 fields
python -c "
import json, sys
row = json.loads(open('${DATASET}').readline())
missing = {'id','category','subtype','topic','text'} - set(row.keys())
if missing:
    print('[error] neutral_prompts.jsonl missing fields:', sorted(missing)); sys.exit(1)
print('[info] schema OK:', list(row.keys()))
"

# ── run ────────────────────────────────────────────────────────────────────────

echo "[info] projecting neutral prompts and running R6 validation checks"

python -u src/18_compass_center.py \
  --dataset    "$DATASET"   \
  --pairs-dir  "$PAIRS_DIR" \
  --output     "$OUTPUT"    \
  --report     "$REPORT"    \
  --device     cuda

echo ""
echo "[info] end=$(date)"
echo "[info] center:  ${OUTPUT}"
echo "[info] report:  ${REPORT}"
