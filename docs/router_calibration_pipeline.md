# Router Calibration Pipeline

## Overview

The router-calibration pipeline learns a small **correction head** that sits
on top of a heuristic prior over four political quadrants
(`left_lib`, `left_auth`, `right_lib`, `right_auth`) and produces the
calibrated routing distribution used by the MoCE runtime.

The supervision is **synthetic**: for every prompt we generate a fan of
candidate quadrant policies, score each candidate with our existing bias /
quality / refusal / vagueness / KL signals, and aggregate the scores into a
soft target distribution. The trained head's job is to learn the residual
between the heuristic prior and that aggregated target — nothing more,
nothing less.

The pipeline is **strictly offline**: every stage writes its output as
JSONL or `.pt` artifacts, and every downstream stage reads only those files
plus `config/config.yaml`. No stage requires a live MoCE engine.


## Pipeline Steps

### Step 1 — Configuration loader

Loads and validates the `router_calibration` block of `config/config.yaml`
into typed dataclasses (`RouterCalibrationConfig`, `RouterPaths`,
`InputTransformerConfig`, `TrainingConfig`, …). Every later stage gets its
runtime values from this loader, so the YAML is the single source of truth.

- **Inputs:** `config/config.yaml`
- **Outputs:** in-memory `RouterCalibrationConfig`
- **Script:** `src/router_training/config.py`


### Step 2 — Prompt-set builder

Pulls prompts from the existing expert datasets (Method 1+2, Method 3, and
the per-expert validation dirs). Deduplicates, optionally caps the corpus
size, and writes a flat JSONL of prompts plus a small report.

- **Inputs:** prompt source files configured under
  `paths.prompt_sources.*`
- **Outputs:** `data/router/prompts.jsonl`
- **Script:** `src/router_training/prompt_set.py`


### Step 3 — Feature builder (input transformer)

Runs every prompt through the base model with the configured
`input_transformer` (vector method, layer selection, pooling, optional
centering) to produce two artifacts: a per-row feature record (with
quadrant scores and `bias_magnitude`) and the contiguous hidden-vector
tensor that the trainer indexes by row. This is the only stage that needs
the base model loaded.

- **Inputs:** `prompts.jsonl`, steering vectors, `paths.model.*`
- **Outputs:** `data/router/features.jsonl`, `data/router/hidden.pt`
- **Script:** `src/router_training/features.py`


### Step 4 — Candidate-policy generator

For every feature row, builds a small fan of candidate routing policies:
the heuristic prior itself, uniform, sharpened, softened, opposite-heavy,
adjacent-heavy variants, and Dirichlet samples around the prior.
Post-processed with `apply_min_probability` and de-duplicated. This stage
is torch-free.

- **Inputs:** `features.jsonl`
- **Outputs:** in-memory list of candidate policies per example
- **Script:** `src/router_training/utils.py`
  (function `generate_candidate_policies`)


### Step 5 — Forced-policy MoCE runner

For every (prompt, candidate-policy) pair, runs the MoCE engine with the
router output **forced** to that candidate distribution and captures the
generated text plus internal trace fields. Used solely to materialise what
each candidate would have produced — never used at training time.

- **Inputs:** `features.jsonl`, candidate policies, expert checkpoints
- **Outputs:** `data/router/candidate_traces.jsonl`
- **Script:** `src/router_training/forced_policy_runner.py`


### Step 6 — Candidate scorer

Scores every traced (prompt, candidate, output) tuple with the configured
combination of bias-radius, quality, refusal, vagueness, and KL-to-prior
signals (with optional LLM-as-judge). Output is the per-candidate
`final_candidate_score` plus its full per-signal breakdown. Torch-required
for projection but kept independent of the trainer.

- **Inputs:** `candidate_traces.jsonl`, scoring weights from config
- **Outputs:** `data/router/candidate_scores.jsonl`
- **Script:** `src/router_training/scorer.py`


### Step 7 — Target-policy builder

Aggregates the per-candidate scores into a single supervision distribution:
a softmax over scores at `score_temperature` defines mixture weights, and
the convex combination of candidate policies is then floored by
`min_probability` and renormalised. Validates every input row before
mixing. Torch-free.

- **Inputs:** `features.jsonl`, `candidate_scores.jsonl`
- **Outputs:** `data/router/records.jsonl`,
  `data/router/target_report.json`
- **Script:** `src/router_training/targets.py`


### Step 8 — Dataset validator

Strict end-to-end check of `records.jsonl` against `hidden.pt`: required
fields, canonical quadrant keys, strictly-positive target distributions
that sum to 1, valid hidden refs (`hidden.pt:<row_index>`), and
non-NaN/non-inf hidden vectors of the configured dimension. Importable
without torch — falls back to structural-only checks when torch is missing.

- **Inputs:** `records.jsonl`, `hidden.pt`
- **Outputs:** raises on the first failure; logs success otherwise
- **Script:** `src/router_training/validator.py`


### Step 9 — Dataset splitter

Stratified, deterministic train/val/test split. Stratification keys are
derived from `metadata.source` / `metadata.axis` (or `none` for a single
stratum). Within each stratum, records are sorted by `example_id` before
shuffling so input row order does not affect the result. Largest-remainder
allocation guarantees every bucket gets at least one record once n ≥ 3.

- **Inputs:** `records.jsonl`
- **Outputs:** `data/router/{train,val,test}/records.jsonl`,
  `data/router/split_report.json`
- **Script:** `src/router_training/splitter.py`


### Step 10 — Training pipeline wrapper

Reproducible CLI wrapper around the standalone trainer. Resolves every
input/output path, validates the train/val/test records and `hidden.pt`
against the schema, builds the deterministic `argv` for
`src/router_training/trainer.py`, optionally executes it via `subprocess`,
and writes a pipeline report containing the command, validation status,
captured stdout/stderr tails, and hyperparameters. Fails fast on missing
artifacts unless `--skip-validation` is set.

- **Inputs:** validated `records.jsonl`, `hidden.pt`, hparams from config
- **Outputs:** `data/router/checkpoints/calibrated_router.pt`,
  `data/router/reports/train_report.json`,
  `data/router/reports/pipeline_train_report.json`
- **Script:** `src/router_training/train_pipeline.py`
  (wraps `src/router_training/trainer.py`)


### Step 11 — Checkpoint evaluator

Offline evaluator that compares **heuristic prior**, **calibrated policy
from the checkpoint**, and **target policy from the records** on val and
test splits. Reports per-split mean / median KL improvements
(`KL(target‖heuristic) − KL(target‖calibrated)`), entropies, L1 distances,
and top-1 accuracies. Loads the trained head independently — no MoCE
runtime, no decoder, no experts. Accepts both the new
`calibration_input_dim` field and the legacy `router_hidden_dim` alias in
the checkpoint payload.

- **Inputs:** trained checkpoint, `hidden.pt`, val/test `records.jsonl`
- **Outputs:** `data/router/reports/router_checkpoint_eval.json`
- **Script:** `src/router_training/evaluator.py`


## Data Flow

```
prompts        →  features          →  candidates       →  forced outputs
(Step 2)         (Step 3)              (Step 4)            (Step 5)

    └→ candidate scores  →  target records  →  splits  →  trained checkpoint
       (Step 6)              (Step 7)            (Step 9)    (Step 10)

                                                          └→ evaluation report
                                                             (Step 11)
```


## How to Run

### Minimal (smoke)

A torch-free synthetic run that exercises every wiring stage in under a
second. Use it to verify the pipeline glue before any real artifacts exist:

```bash
python src/router_training/smoke.py
```

The runner writes `data/router/smoke_pipeline_report.json` containing the
synthetic prompts, features, candidate scores, records, split counts, and
the deterministic training `argv` it would have invoked.


### Full (after MoCE ready)

Once the MoCE runtime is wired and the expert checkpoints are in place,
run the pipeline end-to-end:

```bash
# 1. build prompts
python src/router_training/prompt_set.py --config config/config.yaml

# 2. build features + hidden.pt (requires the base model)
python src/router_training/features.py --config config/config.yaml

# 3. run forced-policy MoCE per (prompt, candidate)
python src/router_training/forced_policy_runner.py --config config/config.yaml

# 4. score candidate traces
python src/router_training/scorer.py --config config/config.yaml

# 5. build target policies and write records.jsonl
python src/router_training/targets.py --config config/config.yaml

# 6. validate the records + hidden.pt
python src/router_training/validator.py \
    --config config/config.yaml \
    --records-path data/router/records.jsonl \
    --hidden-path  data/router/hidden.pt

# 7. split into train/val/test
python src/router_training/splitter.py --config config/config.yaml

# 8. train the calibrated-router head
python src/router_training/train_pipeline.py --config config/config.yaml

# 9. evaluate the trained checkpoint
python src/router_training/evaluator.py --config config/config.yaml
```

For a fast end-to-end debug pass, append `--max-examples 32` to step 8.


### Training with all available prompt sources

Step 2 (`prompt_set.py`) merges three corpora into `data/router/prompts.jsonl`:

| Source                | Path key (under `paths.prompt_sources.*`) | `prompt_set` switch              |
|-----------------------|-------------------------------------------|----------------------------------|
| Method 1+2 statements | `method12_path`                           | `include_method12: true`         |
| Method 3 questions    | `method3_path`                            | `include_method3: true`          |
| Expert-validation set | `expert_validation_dir`                   | `include_expert_validation: true` (per-quadrant `val_indist` / `val_source` / `val_topic` JSONL files under that dir) |

To train with the **maximum-size** corpus — Method 1+2 + Method 3 + every
expert-validation split (`val_indist`, `val_source`, `val_topic`) — set the
`router_calibration.prompt_set` block in `config/config.yaml` to:

```yaml
router_calibration:
  prompt_set:
    include_method12: true
    include_method3: true
    include_expert_validation: true
    expert_validation_splits: ["val_indist", "val_source", "val_topic"]
    max_prompts: null            # no cap
    seed: 42
```

Source order is fixed and matters for deduplication: when the same
stripped `prompt_text` appears in multiple sources, the **first**
occurrence wins (Method 1+2 → Method 3 → expert_val_indist →
expert_val_source → expert_val_topic) and the rest are dropped silently.

Confirm the source paths exist before running Step 2:

```bash
ls data/experts/test_experts/methode_1+2_data.jsonl \
   data/experts/test_experts/methode_3_data.jsonl \
   data/experts/train-validate/{left_lib,left_auth,right_lib,right_auth}/{val_indist,val_source,val_topic}.jsonl
```

Then run the full pipeline as documented above. The prompt-set builder
logs a one-line summary on completion:

```
built N prompts — method12=A method3=B expert_val=C duplicates_removed=D cap=none
```

`A`, `B`, `C` count the rows produced per source **before** dedup; `D`
counts cross-source duplicates dropped; `N = A + B + C − D` (or
`min(A + B + C − D, max_prompts)` when capped). Use `max_prompts: <int>`
to cap the corpus deterministically — the cap is applied **after** dedup,
in source order, so smaller caps shed expert-val prompts before Method 3
and Method 1+2.

To run only a subset, flip individual `include_*` switches off or restrict
`expert_validation_splits` to a smaller list (e.g. `["val_indist"]`). The
config validator rejects an empty `expert_validation_splits` list,
unknown split names, and duplicate entries.


## Expected Artifacts

```
data/router/
  prompts.jsonl                    # Step 2
  features.jsonl                   # Step 3
  hidden.pt                        # Step 3
  candidate_traces.jsonl           # Step 5
  candidate_scores.jsonl           # Step 6
  records.jsonl                    # Step 7
  target_report.json               # Step 7
  split_report.json                # Step 9
  smoke_pipeline_report.json       # smoke runner

  train/records.jsonl              # Step 9
  val/records.jsonl                # Step 9
  test/records.jsonl               # Step 9

  checkpoints/
    calibrated_router.pt           # Step 10

  reports/
    train_report.json              # Step 10 (trainer-internal)
    pipeline_train_report.json     # Step 10 (wrapper)
    router_checkpoint_eval.json    # Step 11
```


## Common Failure Modes

- **Missing `hidden.pt`** — the trainer wrapper raises
  `FileNotFoundError: hidden tensor file not found` and refuses to run.
  Either rerun Step 3 or pass `--skip-validation` if you intentionally
  want to dry-run without the artifact.
- **Hidden-dim mismatch between checkpoint and config** — the evaluator
  raises `checkpoint hidden dim X does not match expected Y`. The
  `calibration_input_dim` in the YAML must match what the trainer wrote
  into the checkpoint.
- **Malformed `target_policy`** — the validator raises with a `[example_id]`
  prefix and a precise message. Causes are usually keys missing,
  non-canonical ordering, non-finite values, or a sum that drifts more
  than 1 e-6 from 1. Fix the upstream stage that produced the row; do
  not loosen the tolerance.
- **`torch` missing** — the trainer, smoke runner, validator, and
  evaluator's pure helpers all stay importable, but the trainer and the
  evaluator's checkpoint loader cannot run without torch. The evaluator's
  test module skips cleanly via a `load_tests` hook; install torch to
  exercise the suite.
- **MoCE runtime not implemented** — Steps 5 and 6 will be the
  bottlenecks. Until they are ready, use the smoke runner to validate
  every other stage and stub `candidate_scores.jsonl` from synthetic
  scores to dry-run Steps 7 → 11.


## Design Decisions

- **Canonical quadrant order is fixed and centralised**
  (`router_training.config.CANONICAL_QUADRANT_ORDER`). Every script
  imports it; the trainer duplicates it locally only because the runtime
  module (`src/09_moce_components.py`) starts with a digit, but the order
  matches by construction.

- **Strict validation, no silent fallback.** Every stage raises
  `ValueError` or `FileNotFoundError` on the first malformed input. We
  never quietly drop bad rows — the dataset must be repaired upstream.

- **Synthetic supervision via forced policies.** The router never needs a
  hand-labelled "true" routing distribution. Instead, we define what
  *good* output looks like (low bias radius, high quality, low refusal /
  vagueness, controlled KL to prior) and let a softmax over scores tell
  us which candidate distributions to imitate.

- **Heuristic prior is parameter-free and reproducible.** The trainer
  recomputes `softmax(-β · q / T)` from `quadrant_scores` at every step,
  matching the runtime router exactly. No prior-state baked into the
  checkpoint, so the head can be retrained against new β / T without
  regenerating the dataset.

- **Trainer stays standalone.** `src/router_training/trainer.py` is the
  one piece of the pipeline that genuinely needs `torch`, and it stays
  pure: no config loading, no orchestration, no path resolution. All of
  that lives in the wrapper (Step 10), so the trainer can be invoked
  directly with explicit `argv` from any orchestration system.

- **Deterministic everywhere it matters.** Splitter sorts by
  `example_id` before shuffling and uses a seeded RNG; candidate
  generation uses an injected `random.Random`; the trainer derives its
  shuffling generator from `--seed`; the smoke runner is byte-for-byte
  reproducible across two invocations with the same seed.
