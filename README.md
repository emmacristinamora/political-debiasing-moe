# political-debiasing-moe

Mixture-of-Calibrated-Experts (MoCE) for debiasing political text generation.
A base causal LM (Mistral-7B) is paired with four LoRA experts — one per
quadrant of the political compass (left-libertarian, left-authoritarian,
right-libertarian, right-authoritarian) — and a router that mixes their
hidden states at decode time. The router emits a counterbalancing prior
that pushes the mixture *away* from the prompt's measured bias, and an
editor refines the mixture in compass space.

## What's implemented

End-to-end pipeline, GPU-ready:

- **Steering-vector pipeline** (`src/01_*` → `src/04_*`) — contrastive
  pairs on the economic and social axes, validation, activation extraction
  from Mistral-7B, and steering-vector construction (mean-difference and
  logistic-regression methods; five layers: 8, 12, 16, 20, 24).
- **Steering-vector validation** (`src/13_*`, `src/17_*`) — Tier 1
  geometry checks (ROC-AUC, permutation tests, k-fold CV) and Tier 3
  robustness checks (leave-one-group-out generalization, cross-template
  invariance via Pearson correlation).
- **Compass center calibration** ([src/18_compass_center.py](src/18_compass_center.py)) —
  projects a set of politically neutral prompts at layer 20 to determine
  Mistral-7B's baseline "neutral" position; validates against four R6
  acceptance criteria (subcategory agreement, midpoint, outlier, bootstrap
  stability). Required before stage 05.
- **Expert datasets and training** (`src/05_*` → `src/08_*`) — quadrant
  pools from 9 corpora, topic labeling (9 topics via cosine similarity
  to keyword prototypes), dataset validation with cell-capping and
  document-level train/val splits, LoRA training of the four quadrant
  experts, and a three-method behavioral test.
- **MoCE architecture** ([src/09_moce_components.py](src/09_moce_components.py)) —
  one self-contained module with all components:
  - `InputTransformer` projects prompts into compass space at layer 20.
  - `Router` emits the heuristic counterbalancing prior π₀ and the
    optional calibrated policy π = softmax(log π₀ + δ(h)).
  - `ExpertManager` runs all four LoRA experts in dense mode.
  - `Editor` recursively re-mixes the experts' hidden states under
    alignment-driven corrections.
  - `MoCEEngine` owns end-to-end orchestration including token-by-token
    decode under the editor's converged mixture (see `MoCEEngine.run`).
- **End-to-end runner** ([src/10_run_moce.py](src/10_run_moce.py)) —
  CLI that loads the base model and adapters, builds `MoCEEngine`, and
  decodes one or many prompts. Heuristic routing is the default; calibrated
  routing is opt-in. See "Running the engine" below.
- **Calibrated-router training pipeline** ([src/router_training/](src/router_training/)) —
  prompt-set construction, feature/hidden-state extraction, forced-policy
  candidate-trace collection, candidate scoring, dataset splitting,
  trainer, evaluator, and a top-level pipeline driver. Outputs a checkpoint
  consumable by `--router-checkpoint` on the runner.
- **LLM-as-judge evaluation** ([src/12_judge_evaluation.py](src/12_judge_evaluation.py)) —
  stance classification (6-point scale → compass coordinates via polarity
  key) and blind pairwise comparison using Llama-3.1-8B-Instruct.
- **Hein congressional validation** (`src/14_*` → `src/16_*`) — external
  validity check: projects US congressional speeches (sessions 97–114)
  onto the political compass and compares economic scores to DW-NOMINATE
  dimension 1.

Partially implemented:

- `src/11_moce_evaluation.py` — subcommand structure and dataclasses in
  place; individual metric CLIs (bias-radius, refusal/vagueness, quality,
  robustness) not yet wired up.

## Repo layout

```
config/config.yaml          single source of truth for every stage
data/experts/normalize_corpora.py   corpus normalization to canonical format
src/01_build_pairs.py       build 180 contrastive prompt pairs
src/02_validate_pairs.py    validate pairs before activation extraction
src/03_extract_activations.py  extract Mistral-7B activations at layers 8,12,16,20,24
src/04_build_steering_vectors.py  mean-diff + logistic-regression steering vectors
src/05_quadrant_datasets.py score + chunk corpora, assign quadrant + topic
src/06_validate_experts_datasets.py  train/val splits with cell-capping
src/07_train_experts.py     LoRA training of the four quadrant experts
src/08_test_experts.py      three-method behavioral evaluation of experts
src/09_moce_components.py   MoCE architecture (InputTransformer, Router,
                            ExpertManager, Editor, MoCEEngine)
src/10_run_moce.py          end-to-end inference runner
src/11_moce_evaluation.py   evaluation stub (subcommand structure in place)
src/12_judge_evaluation.py  LLM-as-judge stance + pairwise evaluation
src/13_steering_vector_geometry.py  Tier 1 vector geometry checks
src/17_steering_vector_robustness.py  Tier 3 leave-one-out + template invariance
src/18_compass_center.py    compute compass center from neutral prompts
src/14_hein_build_dataset.py  }
src/15_hein_project_compass.py  } external validity via Hein congressional + DW-NOMINATE
src/16_hein_dwnominate_analysis.py  }
src/router_training/        calibrated-router training pipeline + CLI
batch/                      SLURM submission scripts for cluster runs
notebooks/                  exploratory analysis (steering vectors, experts, evaluation)
tests/                      pytest suite (engine + router_training)
docs/                       training analysis + calibrated-router pipeline docs
context.md                  full technical context for the paper write-up
```

## Running the engine

`src/10_run_moce.py` is the single entrypoint for end-to-end MoCE
inference. It loads Mistral-7B + the four LoRA experts, builds a
`MoCEEngine`, and decodes prompts through
`transform → route → run experts → edit → decode`.

This is GPU-class work (~14 GB base weights in bf16, plus four LoRA
adapters and four parallel forward passes per generated token). Use a
single H100/A100, or submit it via the existing `batch/` SLURM pattern.

### Synopsis

```text
python src/10_run_moce.py
    --config <yaml>
    (--prompt <text> | --prompts-file <jsonl>)
    [--output-path <jsonl>]
    [--device cuda|cpu]
    [--calibrated --router-checkpoint <ckpt> [--calibration-input-dim N]]
```

### CLI reference

| Flag | Type | Required | Description |
| --- | --- | --- | --- |
| `--config` | path | yes | Path to `config.yaml`; must contain a `moce_inference:` block. |
| `--prompt` | string | one of these two | Single prompt string. Mutually exclusive with `--prompts-file`. |
| `--prompts-file` | path | one of these two | JSONL with one prompt per row: `{"prompt_text": "...", "id": "...optional..."}`. |
| `--output-path` | path | no | When provided, appends one JSON row per prompt with the full result. |
| `--device` | string | no | Overrides `moce_inference.model.device` (e.g. `cuda`, `cuda:0`, `cpu`). |
| `--calibrated` | flag | no | Switch the router to calibrated mode. Requires `--router-checkpoint`. |
| `--router-checkpoint` | path | iff `--calibrated` | Calibrated-router checkpoint produced by `src/router_training/`. |
| `--calibration-input-dim` | int | no | Override `moce_inference.input_transformer.calibration_input_dim`; must match the checkpoint. `--router-hidden-dim` is a deprecated alias. |
| `-h`, `--help` | flag | no | Print argparse help and exit. |

Contract checks fail loudly:

- exactly one of `--prompt` / `--prompts-file` must be given,
- `--calibrated` without `--router-checkpoint` is rejected,
- `--router-checkpoint` without `--calibrated` is rejected,
- missing `moce_inference` block or sub-key in `config.yaml` is rejected.

### `--prompts-file` JSONL schema

One JSON object per line:

```json
{"id": "q1", "prompt_text": "What role should government play in reducing economic inequality?"}
{"id": "q2", "prompt_text": "How should a society balance individual freedom and public safety?"}
```

`id` is optional; if present it is echoed in stdout and `--output-path` rows.

### `--output-path` JSONL schema

When `--output-path` is provided, one row per prompt is appended:

```json
{
  "id": "q1",
  "prompt_text": "...",
  "final_text": "...",
  "router_mode": "heuristic",
  "bias_magnitude": 0.34,
  "economic_score": -0.12,
  "social_score": 0.27,
  "quadrant_scores":   {"left_lib": 0.41, "left_auth": 0.18, "right_lib": 0.27, "right_auth": 0.14},
  "heuristic_prior":   {"left_lib": 0.12, "left_auth": 0.31, "right_lib": 0.22, "right_auth": 0.35},
  "calibrated_policy": {"left_lib": 0.12, "left_auth": 0.31, "right_lib": 0.22, "right_auth": 0.35},
  "final_alpha":       {"left_lib": 0.10, "left_auth": 0.33, "right_lib": 0.21, "right_auth": 0.36},
  "final_alignment":   {"left_lib": 0.08, "left_auth": 0.29, "right_lib": 0.18, "right_auth": 0.45},
  "num_edit_steps": 1,
  "stopped_early": false
}
```

In heuristic mode `calibrated_policy` mirrors `heuristic_prior` exactly.

### Example invocations

Single prompt, default heuristic routing:

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompt "What role should government play in reducing economic inequality?"
```

Batch with persisted rows:

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompts-file data/prompts/eval.jsonl \
    --output-path data/runs/heuristic_eval.jsonl
```

Calibrated routing (after running the router-training pipeline):

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompts-file data/prompts/eval.jsonl \
    --output-path data/runs/calibrated_eval.jsonl \
    --calibrated \
    --router-checkpoint data/router/checkpoints/best.pt
```

CPU smoke check (will load Mistral-7B; expect minutes per token):

```bash
python src/10_run_moce.py \
    --config config/config.yaml \
    --prompt "..." \
    --device cpu
```

### `config.yaml` — `moce_inference` block

Self-contained block consumed only by `10_run_moce.py`:

```yaml
moce_inference:
  model:
    base_model: mistralai/Mistral-7B-v0.1
    dtype: bfloat16              # bfloat16 | float16 | float32
    device: cuda                 # overridable via --device

  steering_vectors:
    economic_vector_path: data/steering-vectors/vectors/economic_vectors.pt
    social_vector_path:   data/steering-vectors/vectors/social_vectors.pt

  input_transformer:
    vector_method: logistic_regression
    use_final_aggregated_vectors: true
    selected_layers: [8, 12, 16, 20, 24]
    pooling_method: mean
    use_centering: false
    neutral_reference_path: null
    calibration_input_dim: 4096  # overridable via --calibration-input-dim

  expert_checkpoints:
    left_lib_checkpoint:   data/experts/final-experts/left_lib
    left_auth_checkpoint:  data/experts/final-experts/left_auth
    right_lib_checkpoint:  data/experts/final-experts/right_lib
    right_auth_checkpoint: data/experts/final-experts/right_auth

  router:
    beta: 1.0                              # scales -beta * q_i in the prior
    temperature: 1.0                       # softmax temperature
    fallback_to_uniform_if_centered: true
    center_threshold: 0.05                 # bias_magnitude threshold for fallback

  editor:
    max_edit_steps: 1
    correction_beta: 1.0
    initialization_mode: router_policy     # router_policy | uniform
    use_recursive_editing: true
    initialize_from_router: true
    convergence_threshold: 1.0e-3

  generation:
    max_new_tokens: 256
    temperature: 0.7
    do_sample: false
    top_p: 1.0
```

Each adapter directory under `expert_checkpoints` must contain an
`adapter_config.json` plus PEFT weights. The four checkpoint paths are
mapped to the canonical quadrant order
`left_lib, left_auth, right_lib, right_auth`.

### Stdout output

For each prompt the runner prints a multi-line block:

```
========================================================================
id: q1
prompt:        What role should government play in reducing economic inequality?
final_text:    …generated answer…
prior:         left_lib=0.120, left_auth=0.310, right_lib=0.220, right_auth=0.350
policy:        left_lib=0.120, left_auth=0.310, right_lib=0.220, right_auth=0.350
final_alpha:   left_lib=0.105, left_auth=0.333, right_lib=0.210, right_auth=0.352
edit_steps:    1  stopped_early=False
```

## Other pipelines (brief pointers)

The numbered scripts under `src/` form an offline pipeline that produces
the artifacts consumed by `10_run_moce.py`:

- **Corpus normalization** (`data/experts/normalize_corpora.py`): converts
  raw corpora (CSV, JSON, RDS) to the canonical `Document` JSONL schema.
- **Steering vectors** (`01` → `04`): 180 contrastive pairs → activations at
  5 layers → mean-difference and logistic-regression vectors. Produces
  `data/steering-vectors/vectors/{economic,social}_vectors.pt`.
- **Steering vector validation** (`13`, `17`): geometry and robustness checks.
  Run after `04` to verify the vectors before using them downstream.
- **Compass center** (`18`): projects neutral prompts at layer 20, validates
  against 4 acceptance criteria. Produces `data/compass_center/center.json`.
  Must run before `05`.
- **Quadrant datasets** (`05` → `06`): scores and chunks all corpora, assigns
  quadrant and topic labels, then builds balanced train/val splits.
- **Expert training** (`07`): LoRA-trains the four quadrant experts (r=8,
  q_proj+v_proj, bfloat16). Outputs land under `data/experts/`.
- **Expert testing** (`08`): three-method behavioral evaluation
  (Representativeness, Inverse Steerability, Consistency).
- **Calibrated router** ([src/router_training/](src/router_training/)): 11-step
  pipeline: prompt-set → features → forced-policy traces → scored candidates →
  target policies → dataset splits → trainer → evaluator. See
  [docs/router_calibration_pipeline.md](docs/router_calibration_pipeline.md).
  The checkpoint is the file you pass to `--router-checkpoint`.
- **Judge evaluation** (`12`): stance classification and pairwise comparison
  using Llama-3.1-8B-Instruct.
- **Hein validation** (`14` → `16`): external DW-NOMINATE correlation check.

SLURM submission scripts for the cluster jobs are in [batch/](batch/).

## Tests

```bash
pytest tests/
```

The suite covers `MoCEEngine` integration, every individual MoCE
component, and the router-training pipeline. Tests do not require GPUs or
the Mistral-7B weights.

## Configuration

Every stage reads from a single [config/config.yaml](config/config.yaml).
Each stage has its own top-level block (`build_pairs`, `extract_activations`,
`quadrant_dataset`, `train_experts`, `router_calibration`, `moce_inference`, …)
so changes to one stage's hyperparameters do not affect the others.
