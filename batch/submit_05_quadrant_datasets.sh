#!/bin/bash
#SBATCH --job-name=build_quadrant_datasets
#SBATCH --partition=long_gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_quadrant_datasets_%j.out
#SBATCH --error=logs/build_quadrant_datasets_%j.err
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmamora2003@gmail.com

# Do NOT use set -e here: each source runs independently and a single failure
# should not abort the whole job.  Failures are collected and reported at the end.
set -uo pipefail

echo "[info] job_id=${SLURM_JOB_ID:-no_slurm}"
echo "[info] node=$(hostname)"
echo "[info] start=$(date)"
echo "[info] partition=${SLURM_JOB_PARTITION:-unknown}"

REPO_ROOT="/home/3210604/projects/political-debiasing-moe"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p data/experts/quadrant_pools

source "$REPO_ROOT/.venv/bin/activate"

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python=$(which python)"
echo "[info] gpu info:"
nvidia-smi || true

# ── sanity checks ────────────────────────────────────────────────────────────

if [ ! -f "src/05_quadrant_datasets.py" ]; then
  echo "[error] src/05_quadrant_datasets.py not found — aborting"
  exit 1
fi

if [ ! -f "config/config.yaml" ]; then
  echo "[error] config/config.yaml not found — aborting"
  exit 1
fi

if [ ! -f "data/steering-vectors/vectors/economic_vectors.pt" ]; then
  echo "[error] economic_vectors.pt not found — run 04_build_steering_vectors.py first"
  exit 1
fi

if [ ! -f "data/steering-vectors/vectors/social_vectors.pt" ]; then
  echo "[error] social_vectors.pt not found — run 04_build_steering_vectors.py first"
  exit 1
fi

NORMALIZED_DIR="data/experts/raw/normalized"

for source_file in \
  allsides.jsonl \
  reddit_liberal.jsonl \
  reddit_conservative.jsonl \
  uk_house_of_commons.jsonl \
  ec_press_releases.jsonl \
  uk_gov_press_releases.jsonl \
  ire_gov_press_releases.jsonl
do
  if [ ! -s "${NORMALIZED_DIR}/${source_file}" ]; then
    echo "[error] ${NORMALIZED_DIR}/${source_file} is missing or empty — run normalize_corpora.py first"
    exit 1
  fi
done

echo "[info] all pre-flight checks passed"

# ── source runner ─────────────────────────────────────────────────────────────
# Runs one source, logs elapsed time, and records any failure.
# Usage: run_source <source_key> [extra python args...]

FAILED_SOURCES=""

run_source() {
  local source="$1"
  shift
  local extra_args=("$@")

  echo ""
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  source=${source}"
  echo "============================================================"

  local start_ts
  start_ts=$(date +%s)

  if python -u src/05_quadrant_datasets.py \
      --source "${source}" \
      --config-yaml-path config/config.yaml \
      "${extra_args[@]}"; then

    local end_ts elapsed
    end_ts=$(date +%s)
    elapsed=$(( end_ts - start_ts ))
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE   source=${source}  elapsed=${elapsed}s"
    echo "============================================================"
  else
    local exit_code=$?
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED source=${source}  exit=${exit_code}"
    echo "============================================================"
    FAILED_SOURCES="${FAILED_SOURCES} ${source}"
  fi
}

# ── sources ───────────────────────────────────────────────────────────────────
# allsides texts are short excerpts (~75 words); lower min-chunk-tokens so they
# are not all silently discarded by the chunker.

run_source allsides \
  --min-chunk-tokens 30

run_source ec_press

run_source ire_press

run_source uk_press

# HoC: 1.6M speeches — medium stratified sample (decade × party).
# 150 000 speeches ≈ 75 000 chunks after the 30-word floor, ~1-2h on A100.
run_source hoc \
  --hoc-sample-n 150000 \
  --min-chunk-tokens 30

run_source reddit_liberal

# reddit_conservative is the largest source (~690K chunks).
# Expected runtime: ~9-10h on A100 at batch_size=8.
run_source reddit_conservative

# ── final report ──────────────────────────────────────────────────────────────

echo ""
echo "[info] end=$(date)"

if [ -n "${FAILED_SOURCES}" ]; then
  echo "[error] the following sources failed:${FAILED_SOURCES}"
  echo "[error] check the log above for their tracebacks"
  exit 1
else
  echo "[info] all sources completed successfully"
fi
