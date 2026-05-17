# Political Debiasing MoCE — Complete Paper Context

Everything that was built, every choice that was made, every threshold that was set. Written for the paper write-up.

---

## System Overview

**Full name:** Mixture-of-Calibrated-Experts (MoCE) for political debiasing
**Base model:** `mistralai/Mistral-7B-v0.1` (7B parameters, decoder-only)
**Architecture:** one base LM + four LoRA expert adapters (one per political compass quadrant) + a router + an editor
**Quadrants:** `left_lib` (left-libertarian), `left_auth` (left-authoritarian), `right_lib` (right-libertarian), `right_auth` (right-authoritarian)
**Canonical quadrant order (fixed throughout):** `(left_lib, left_auth, right_lib, right_auth)`
**Single config file:** `config/config.yaml` — every numeric hyperparameter, path, and flag for every stage lives here

---

## Pipeline Stages (in order)

```
01  Build contrastive pairs
02  Validate pairs
03  Extract activations from Mistral-7B
04  Build steering vectors
05  Score corpora & assign quadrants/topics
06  Validate expert datasets (train/val splits)
07  Train LoRA experts
08  Test experts (3 methods)
09  MoCE components (architecture)
10  Run MoCE (inference)
11  MoCE evaluation (stub)
12  Judge evaluation (LLM-as-judge)
13  Steering vector geometry checks (Tier 1)
17  Steering vector robustness checks (Tier 3)
18  Compute compass center from neutral prompts
router_training/  Calibrated router training sub-pipeline
```

---

## Stage 01 — Build Contrastive Pairs (`src/01_build_pairs.py`)

### Purpose
Generate the contrastive prompt pairs used to derive the political steering vectors.

### Design
- **30 seed statements per axis** (economic and social), written as politically neutral propositions
- **3 templates per statement** → 90 pairs per axis → **180 pairs total**
- Each pair = one (positive, negative) prompt: the same statement argued from each pole

### Economic axis labels
- Positive pole: `econ_right`
- Negative pole: `econ_left`

### Social axis labels
- Positive pole: `authoritarian`
- Negative pole: `libertarian`

### Templates (same 3 for both axes)
1. **Analytical argument**: argue analytically for the labeled position
2. **Value-prioritizer explanation**: explain which values you prioritize and why
3. **Policy tradeoff discussion**: discuss the policy tradeoffs from the labeled perspective

### Output schema (JSONL)
`id`, `axis`, `statement_id`, `statement`, `template_id`, `negative_label`, `positive_label`, `neg`, `pos`

---

## Stage 02 — Validate Pairs (`src/02_validate_pairs.py`)

### Rejection checks (in order)
1. Missing required fields
2. Type errors (all text fields must be strings)
3. Invalid axis value (must be `economic` or `social`)
4. Empty `pos` or `neg`
5. `identical_prompts_exact` — pos == neg
6. `identical_prompts_normalized` — after lowercasing, stripping `[^\w\s]`, collapsing whitespace
7. `length_ratio_too_high` — max(len(pos), len(neg)) / min > 1.5
8. `duplicate_pair` — same (axis, statement_id, template_id, normalized_pos, normalized_neg)

### Outputs
- Validated JSONL
- Rejected JSONL with rejection reason
- JSON report with rejection reason counts

---

## Stage 03 — Extract Activations (`src/03_extract_activations.py`)

### Model
`mistralai/Mistral-7B-v0.1`, loaded in **float16**, `output_hidden_states=True`, eval mode

### Layers extracted
`[8, 12, 16, 20, 24]` — sampled at early, middle, and later layers

### Pooling
Mask-weighted **mean** over non-padding tokens only (Mistral has no pad token; `pad_token` is set to `eos_token` before tokenization)

### Hardcoded constraint
`batch_size` **must equal 2** (one positive + one negative per pair run together). Enforced with `ValueError`.

### Layer indexing convention
`hidden_states[layer_index + 1]` (index 0 = embedding layer, so layer 8 → `hidden_states[9]`). This convention is used consistently in scripts 03, 05, 09, and 18.

### Saved dtype
Activations saved in **float32** (even though model runs in float16)

### Output
`pair_ids`, `statement_ids`, `template_ids`, `token_counts`, `activations` (dict: `layer_int → {"pos": tensor[N, D], "neg": tensor[N, D]}`)

---

## Stage 04 — Build Steering Vectors (`src/04_build_steering_vectors.py`)

### Two methods computed and saved
**Method 1: `mean_difference`**
- Vector = μ_pos − μ_neg, then normalized to unit norm
- Quality score = separation = |mean_pos_proj − mean_neg_proj| / pooled_std

**Method 2: `logistic_regression`**
- sklearn `LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)`
- Quality score = 0.6 × train_accuracy + 0.4 × min(separation / 2, 1.0)
- **This is the method used in production** (all downstream scripts use `logistic_regression` vectors)

### Sign convention enforcement
After building the vector, if `mean(pos_projections) < mean(neg_projections)`, the vector is flipped. This guarantees: **positive dot product = economically right / authoritarian; negative dot product = economically left / libertarian.** This convention is never re-verified downstream; it is assumed after this step.

### Aggregation across layers
Quality-weighted mean of per-layer vectors (weights clipped at 1e-12, normalized to sum = 1), then re-normalized to unit norm. The aggregated "final" vector is what is used in production.

### Named directions stored
- Economic: `econ_right` (the final vector), `econ_left` (its negation)
- Social: `authoritarian` (the final vector), `libertarian` (its negation)

---

## Stage 05 — Quadrant Datasets (`src/05_quadrant_datasets.py`)

### Purpose
Score every chunk of every source corpus on both axes, assign quadrant and topic labels, and produce the raw retained pools used for expert training.

### Data sources (9 sources)

| Source key | Description | Output file |
|---|---|---|
| allsides | AllSides balanced news excerpts | allsides.jsonl |
| reddit_liberal | r/Liberal posts | reddit_liberal.jsonl |
| reddit_conservative | r/Conservative posts | reddit_conservative.jsonl |
| hoc | UK House of Commons speeches | uk_house_of_commons.jsonl |
| ec_press | European Commission press releases | ec_press_releases.jsonl |
| uk_press | UK Government press releases | uk_gov_press_releases.jsonl |
| ire_press | Irish Government press releases | ire_gov_press_releases.jsonl |
| us_media | US news articles (NYP/NYT/WSJ, 2012–2024) | us_media_articles.jsonl |
| us_speeches | US presidential speeches (1985–2026) | us_presidential_speeches.jsonl |

### Chunking parameters
- **Window size:** 512 tokens (whitespace tokenization)
- **Overlap:** 128 tokens (step = 384 tokens) — documented in TODO.txt; config says 0 but runtime default is 128-step
- **Minimum chunk length:** 128 tokens (shorter chunks discarded)
- AllSides and HoC override minimum to 30 tokens (command-line flag `--min-chunk-tokens 30`) because those sources have very short texts

### Model for encoding
Same `Mistral-7B-v0.1` loaded in **float16**, `output_hidden_states=True`, eval mode

### Encoding layer: **Layer 20**
- Chosen empirically: peak economic representativeness at mid-network, increasing social representativeness in later layers
- Consistent with all other stages (see cross-cutting decisions)

### Score computation
```
h = mean-pool(hidden_states[21], non-padding tokens), then L2-normalized to unit norm
score_econ = h · v_econ       (positive = economically right)
score_soc  = h · v_soc        (positive = authoritarian)
confidence_margin = min(|score_econ|, |score_soc|)
```
Both h and v are unit-normalized, so these are cosine similarities.

### Filtering threshold
`confidence_margin ≥ 0.015` on **both axes simultaneously**. Set empirically. Chunks below this threshold are recorded in the audit trail (`scored_chunks.jsonl`) but excluded from the retained pool.

### Quadrant assignment
Relative to the **compass center** (computed in stage 18):
```
se = score_econ − center_econ
ss = score_soc  − center_soc
right_auth:  se ≥ 0 and ss ≥ 0
left_auth:   se < 0 and ss ≥ 0
left_lib:    se < 0 and ss < 0
right_lib:   se ≥ 0 and ss < 0
```

### HoC sampling
UK House of Commons has 1.6M speeches. A stratified sample of **200,000 speeches** is drawn (decade × party strata, proportional allocation, minimum 1 per stratum, seed=42).

### Topic labeling (9 topics)
All retained chunks are labeled by cosine similarity to keyword-based prototype embeddings:

| Topic | Prototype keywords |
|---|---|
| economy | taxation, inflation, wages, markets, trade, investment, fiscal policy, economic growth, GDP, public spending, deficit, debt, redistribution |
| immigration | migrants, asylum, border control, deportation, refugee policy, immigration, citizenship, visa, integration, multiculturalism |
| foreign_policy | war, alliances, military aid, sanctions, diplomacy, NATO, foreign relations, defence, international trade, geopolitics |
| law_order | crime, policing, sentencing, prisons, public safety, criminal justice, law enforcement, drugs, gangs, police reform |
| environment_energy | climate change, energy policy, oil, gas, renewables, emissions, environment, sustainability, carbon tax, green deal, fossil fuels |
| culture_identity | religion, family, gender, abortion, identity politics, culture, values, tradition, secularism, free speech, woke, LGBTQ |
| welfare_labor | social benefits, pensions, unions, labor rights, welfare, unemployment, workers, social protection, minimum wage, strikes |
| health_education | schools, healthcare, hospitals, universities, teachers, NHS, education, public health, mental health, childcare |
| governance_institutions | elections, parliament, constitution, courts, democracy, rule of law, institutions, government, corruption, accountability |

Prototype texts are embedded once at startup using the same Mistral-7B model. `topic_primary` = highest cosine; `topic_secondary` = second highest. Every retained chunk gets a label — no threshold is applied.

### Outputs per source
- `retained.jsonl` — chunks passing all thresholds
- `scored_chunks.jsonl` — all chunks (audit trail)
- `document_summaries.jsonl` — per-document stats
- `report.json` — volume, source/topic composition

---

## Stage 06 — Validate Expert Datasets (`src/06_validate_experts_datasets.py`)

### Purpose
Build reproducible, balanced train/val splits for each of the 4 quadrant experts.

### Pipeline stages (internal)
0. Load all retained chunks from quadrant pool
1. **Topic filter** — keep only `viable_topics` + `held_out_topic`
1.5. **Boilerplate filter** — 12 regex patterns remove academic/survey artefacts:
   "APA style", "reference page", "your paper", "assignment guidelines", "write a short argument", "complete the survey", "submit your ideas", "we want to hear from you", "download the submission form", "related policy papers", "newsletter", "subscribe"
2a. Carve `held_out_topic` chunks into separate pool → `val_topic`
2b. Carve `held_out_source` chunks into separate pool → `val_source`
3. Optional deduplication (not enabled by default)
4. Cell diagnostics (source × topic breakdown)
5. **Cell-cap sampling** — for each (source × topic) cell, `random.sample(250, seed=42)` if over cap
6. **Document-level split** — shuffle unique doc_ids (seed=42), first `ceil(n_docs × 0.15)` → `val_indist`; remaining docs → train
7. Sanity checks (10+ checks, hard exit on failure)
8. Write outputs

### Topic configuration
- **Viable topics (in training):** `economy`, `foreign_policy`, `law_order`, `health_education`, `governance_institutions`
- **Held-out topic (val_topic only):** `immigration` — removed from every expert's training set to test topic generalization

### Held-out sources per expert (val_source)
| Expert | Held-out source |
|---|---|
| right_auth | allsides |
| left_auth | uk_press |
| left_lib | ire_press |
| right_lib | uk_press |

### Cell cap
`max_cell_size = 250` chunks per (source, topic) cell. Applied before the train/val split.

### Train/val split
- `val_pct = 0.15` (15% of documents → val_indist, 85% → train)
- Split is **document-level** (no chunk from the same document appears in both train and any val split)

### Sanity checks (hard exits on failure)
| Check | Threshold |
|---|---|
| MAX_CELL_SIZE | No (source × topic) cell exceeds 250 in train |
| SOURCE_CAP | No single source exceeds 70% of train |
| TOPIC_CAP | No single topic exceeds 50% of train |
| TOPIC_KL | KL divergence of topic distribution ≤ 8.0 |
| LENGTH_RATIO | max/min chunk length ≤ 3.0 |
| CONF_MARGIN_RATIO | max/min confidence margin ≤ 2.5 |
| NO_DOC_LEAKAGE | No document IDs in both train and any val split |
| VAL_TOPIC_PURITY | val_topic contains only held_out_topic |
| VAL_SOURCE_PURITY | val_source contains only held_out_source |
| NON_EMPTY | All splits have ≥ 15 examples |

### Length filter (applied before cell capping)
- Min tokens: 50
- Max tokens: 700

---

## Stage 07 — Train Experts (`src/07_train_experts.py`)

### Base model
`mistralai/Mistral-7B-v0.1`, loaded in **bfloat16**

### LoRA configuration
| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha (lora_alpha) | 16 |
| Dropout | 0.1 |
| Target modules | `q_proj`, `v_proj` only |
| Bias adaptation | none |
| Task type | CAUSAL_LM |

All base model parameters are frozen (`requires_grad_(False)`). Only `lora_` parameters have `requires_grad=True` (verified in code).

### Training hyperparameters
| Parameter | Value |
|---|---|
| Epochs | 5 |
| Learning rate | 8e-5 |
| Scheduler | cosine |
| Warmup ratio | 0.10 |
| Weight decay | 0.01 |
| Per-device batch | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Max grad norm | 1.0 |
| Precision | bf16 + tf32 |
| Evals per epoch | 4 |
| Max tokenized length | 700 tokens |

### Loss function (custom WeightedTrainer)
- Per-example loss = mean CE over non-padding tokens
- Batch loss = **inverse-frequency source-group weighted mean** across examples
- Source groups: `reddit` (reddit_liberal + reddit_conservative), `press` (allsides, ec_press, ire_press, uk_press, us_media), `speeches` (hoc, us_speeches)
- Weights = n_total / (n_groups × count_g), **capped at 3.0**
- This downweights dominant sources (reddit_conservative is the largest at ~690K chunks)

### Evaluation (custom MultiSplitEvalCallback)
Three val splits evaluated at each eval step:
- `val_indist` — in-distribution held-out documents
- `val_source` — all chunks from held-out source
- `val_topic` — all chunks about immigration

Each split subsampled to **500 examples** at eval time (seed=42) for speed.
`mean_val_loss = (val_indist + val_source + val_topic) / 3.0`

Best checkpoint saved by callback when `mean_val_loss` improves (independent of HuggingFace trainer logic, which is turned off for best-checkpoint tracking).

### Known artefact
Logged **training loss appears ~4–5× higher than val loss** due to the WeightedTrainer interacting with grad_accum_steps=4. This is cosmetic — it does not affect which checkpoint is saved.

### Additional callbacks
- `LearningCurvePlotCallback`: saves training curve PNG after every log step (overwrites same file)
- `GenerationCallback`: generates 4 fixed prompts once per epoch with temperature=0.0, max_new_tokens=100

### Seeds explored
`[42, 123, 456]` — seed spread was < 0.001 across all experts (fully reproducible)

---

## Expert Training Results (Run 2)

### Best checkpoint metrics

| Expert | Best seed | Best epoch | mean_val_loss | val_indist | val_source | val_topic |
|---|---|---|---|---|---|---|
| right_auth | 42 | 1.96 | 1.7085 | 1.8505 | 1.6052 | 1.6697 |
| left_auth | 42 | 1.74 | 1.8626 | 2.0191 | 1.7590 | 1.8098 |
| left_lib | 456 | 1.75 | 1.8842 | 2.0008 | 1.8093 | 1.8426 |
| right_lib | 456 | 1.985 | 1.8227 | 1.8332 | 1.8225 | 1.8123 |

### Training dataset sizes

| Expert | Train examples |
|---|---|
| right_auth | 1,304 |
| left_auth | 2,057 |
| right_lib | 3,669 |
| left_lib | 5,118 |

Significant imbalance across experts — inherent from political corpus availability.

### Source group composition (train split)

| Expert | reddit % | press % | speeches % |
|---|---|---|---|
| right_auth | 66% | 29% | 4% |
| left_auth | 16% | 32% | 52% |
| left_lib | 40% | 39% | 21% |
| right_lib | 55% | 21% | 24% |

### Key findings
- All experts **saturate by epoch ~2**; LoRA r=8 is the binding capacity constraint
- `right_lib` specialization gap near zero (0.011 between val_indist and val_source) — adapter is not learning quadrant-specific signal, likely due to right-lib training data quality
- Held-out topic (immigration) generalizes reasonably (val_topic close to val_indist for most experts)

---

## Stage 08 — Test Experts (`src/08_test_experts.py`)

### Method 1: Representativeness
- For each PCT statement: generate a **free-form response**
- Project `(statement + "\n\nResponse:\n" + generated_response)` at layer 20
- Pooling: **response tokens only** (prefix masked to zero); fallback to all tokens if response is empty
- Generation: `max_new_tokens=120`, `do_sample=False`, `temperature=0.0` (greedy)
- Records: PCT coordinates, predicted quadrant, whether it matches target

### Method 2: Inverse Steerability
- Prepend an **adversarial counter-persona prompt** (designed for the opposite quadrant) before each statement
- Each trained expert runs **only its designated adversary** (the one targeting its opposite)
- Base model runs **all 4** adversarial personas for comparison
- Measures: shift in PCT coordinates vs. Method 1 baseline (`shift_magnitude = sqrt(Δecon² + Δsoc²)`)
- Key metric: `resistance_vs_base` — whether the expert resists the adversarial persona more than the base model does

### Method 3: Consistency
- For each survey question + closed-form answer options: select the option with the **lowest average negative log-likelihood** (NLL)
- "Refused" options are excluded from NLL scoring
- Leading space prepended to each option for correct tokenization
- Project `(question + "\n\nAnswer:\n" + selected_option)` at layer 20
- Measures whether the model's implicit preference (via perplexity) aligns with its target quadrant

### Implementation note
Model is reloaded from scratch for each expert condition to prevent adapter state leakage.

---

## Stage 09 — MoCE Components (`src/09_moce_components.py`)

### Constants
- `CANONICAL_QUADRANT_ORDER = ("left_lib", "left_auth", "right_lib", "right_auth")`
- `InputTransformer.ENCODING_LAYER = 20` (hardcoded class constant)
- `InputTransformer.DEFAULT_MAX_LENGTH = 512`

### InputTransformer
- Encodes prompt → Mistral-7B hidden state at layer 20 → mean-pool over non-padding tokens → L2 normalize
- Projects onto 4 quadrant vectors to produce `quadrant_scores`
- `bias_magnitude = sqrt(score_econ² + score_soc²)`

### Quadrant basis vectors
```
left_lib  = normalize(-v_econ - v_soc)
left_auth = normalize(-v_econ + v_soc)
right_lib = normalize(+v_econ - v_soc)
right_auth= normalize(+v_econ + v_soc)
```
These are diagonal directions in the 2D political compass space.

### Heuristic router (no learned parameters)
```
prior_i = softmax(-β × quadrant_scores / T)_i
```
- β = 1.0 (scales how strongly scores push the prior)
- T = 1.0 (softmax temperature)
- **Centered prompts** (`bias_magnitude ≤ 0.05`): fall back to uniform [0.25, 0.25, 0.25, 0.25]
- The prior is **counterbalancing**: prompts scoring high on a quadrant get low prior weight for that quadrant

### Calibrated router (learned)
- Correction head δ(h) trained to improve on the heuristic prior
- Final policy: `softmax(log π₀ + δ(h))`

### Editor stopping conditions
| Condition | Threshold |
|---|---|
| axis_proximity | min(|score_econ|, |score_soc|) ≤ 0.015 (already near-center) |
| converged | max_alpha_change and max_alignment_change ≤ 1e-3 |
| max_steps | 10 (hard cap) |

### Generation defaults
- `max_new_tokens: 256`, `temperature: 0.7`, `do_sample: False`, `top_p: 1.0`
- `frequency_penalty: 0.4`, `no_repeat_ngram_size: 3` (set in config, not defaults)

---

## Stage 18 — Compass Center (`src/18_compass_center.py`)

### Purpose
Compute the "neutral point" of the political compass so that quadrant assignment in stage 05 is calibrated relative to Mistral-7B's baseline (not the geometric origin).

### Input
`data/neutral_prompts.jsonl` — politically neutral prompts in two categories: `apolitical` and `generic_task`

### Layer
Layer 20 (same convention)

### Acceptance criteria (R6)
| Criterion | Threshold |
|---|---|
| R6.1 Subcategory agreement | apolitical centroid ≈ generic_task centroid, within 0.5 std |
| R6.2 Midpoint test | |center − midpoint| / spread ≤ 0.15 |
| R6.3 No outliers | all prompt z-scores ≤ 2.5 (outliers removed before final center) |
| R6.4 Bootstrap stability | SE / spread ≤ 0.05 over 1000 bootstrap samples |

### Output
`data/compass_center/center.json` — the center coordinates `(center_econ, center_soc)` used in stage 05

---

## Steering Vector Validation

### Stage 13 — Geometry checks (Tier 1) (`src/13_steering_vector_geometry.py`)
- ROC-AUC for separating pos from neg pole
- Permutation test (500 permutations, null distribution via label shuffling)
- 5-fold cross-validation
- Both `mean_difference` and `logistic_regression` methods

### Stage 17 — Robustness checks (Tier 3) (`src/17_steering_vector_robustness.py`)
- **Leave-one-group-out**: for each topic or template group, rebuild vector from all pairs except that group, measure AUC on held-out group
- **Template invariance**: cross-template Pearson correlation of signed scores — high correlation = same statement rankings regardless of prompt template
- Separation metric: Cohen's d = (mean_pos − mean_neg) / pooled_std

---

## Router Training Pipeline (`src/router_training/`)

### Purpose
Train a correction head δ(h) that improves on the heuristic prior by learning from scored forced-policy traces.

### 11-step pipeline

| Step | Script | Output |
|---|---|---|
| 1 | config.py | Validated RouterCalibrationConfig |
| 2 | prompt_set.py | data/router/prompts.jsonl |
| 3 | features.py | data/router/features.jsonl, hidden.pt |
| 4 | utils.py | in-memory candidate policies |
| 5 | forced_policy_runner.py | data/router/candidate_traces.jsonl |
| 6 | scorer.py | data/router/candidate_scores.jsonl |
| 7 | targets.py | data/router/records.jsonl, target_report.json |
| 8 | validator.py | (validation gate, no output) |
| 9 | splitter.py | data/router/{train,val,test}/records.jsonl |
| 10 | train_pipeline.py | data/router/checkpoints/calibrated_router.pt |
| 11 | evaluator.py | data/router/reports/router_checkpoint_eval.json |

### Candidate policies explored (Step 4)
- Heuristic prior itself
- Uniform
- Sharpened (temperature 0.5)
- Softened (temperature 2.0)
- Opposite-heavy
- Adjacent-heavy
- 16 Dirichlet samples around prior (concentration=64.0)
- All floored at min_probability=1e-6 and de-duplicated

### Scoring weights
| Component | Weight |
|---|---|
| bias_radius | 1.0 |
| quality | 0.5 |
| refusal | 0.5 |
| vagueness | 0.3 |
| kl_to_prior | 0.1 |

### Target policy construction
Softmax over candidate composite scores at `score_temperature=0.2` defines mixture weights. Final target = convex combination of candidate policies, floored at 1e-6 and renormalized.

### Router training hyperparameters
| Parameter | Value |
|---|---|
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 20 |
| KL weight | 0.1 |
| Entropy weight | 0.01 |
| Data split | 80/10/10 by source |

### Key design choice: synthetic supervision
No hand-labeled routing distributions are used. Supervision comes from scoring the model's own outputs under forced policies. This makes the pipeline fully self-supervised.

---

## Judge Evaluation (`src/12_judge_evaluation.py`)

### Judge model
`meta-llama/Llama-3.1-8B-Instruct`

### Subcommands
**Stance classification** (`stance`):
- Scale: strongly_disagree (−2), disagree (−1), neutral (0), agree (+1), strongly_agree (+2), incoherent (0)
- Compute compass position from stance scores using polarity key from `config/pct_eval_polarity.yaml`
- Explicit judge guidance to distinguish "neutral" (clear and balanced) from "incoherent" (confused/evasive)

**Pairwise comparison** (`pairwise`):
- Blind head-to-head on neutrality and coherence
- Judge outputs A / B / tie per criterion

---

## Corpora Normalization (`data/experts/normalize_corpora.py`)

### Word count thresholds
| Source | Min words | Max words |
|---|---|---|
| allsides | 30 | 800 |
| reddit_liberal | 100 | 800 |
| reddit_conservative | 100 | 800 |
| hoc | 30 | 800 |
| ec_press | 100 | 800 |
| uk_press | 100 | 800 |
| ire_press | 100 | 800 |

### Raw file sources
| Source key | Raw format | File |
|---|---|---|
| allsides | CSV | allsides_balanced_news_headlines-texts.csv |
| reddit_liberal | JSON | Liberal.json |
| reddit_conservative | JSON | Conservative.json |
| hoc | RDS (via R subprocess) | Corp_HouseOfCommons_V2.rds |
| ec_press | RDS (via R subprocess) | EC-PressReleases_1985-2020_clean.RDS |
| uk_press | RDS (pyreadr) | UK-GovPressReleases.Rds |
| ire_press | RDS (pyreadr) | IRE-GovPressReleases.Rds |

HoC and EC press releases require R subprocess (pyreadr encoding issues with those specific RDS files). R subprocess timeout: 600 seconds.

### Source family labels
| Source | Family |
|---|---|
| allsides | news_article |
| reddit | article_opinion |
| hoc | institutional_speech |
| ec_press | institutional_press_release |
| uk_press | institutional_speech or institutional_press_release |
| ire_press | institutional_press_release |

### Text normalization
- `\r\n` and `\r` → `\n`
- Multiple spaces/tabs → single space
- 3+ consecutive newlines → 2 newlines
- Strip leading/trailing whitespace

---

## Cross-Cutting Design Decisions

### Layer 20: the universal encoding layer
Every component uses layer 20 for semantic representations. Justification: empirically peak economic representativeness at mid-network; increasing social representativeness in later layers. Layer 20 provides a good trade-off. All scripts use the same indexing convention: `hidden_states[LAYER + 1]` (since index 0 is the embedding layer).

Scripts that use layer 20: `03_extract_activations.py` (among others), `05_quadrant_datasets.py`, `08_test_experts.py`, `09_moce_components.py`, `18_compass_center.py`, `router_training/features.py`, `router_training/config.py` (REQUIRED_LAYER = 20).

### Unit normalization everywhere
All vectors (steering vectors, chunk embeddings, quadrant basis vectors) are L2-normalized to unit norm before dot products. Cosine similarity and dot product are equivalent throughout.

### Confidence threshold 0.015
The minimum cosine similarity on **both** axes simultaneously. Set empirically. Below this threshold, a chunk does not have enough political signal to be assigned to a quadrant reliably.

### Cell cap 250
Each (source × topic) combination contributes at most 250 chunks to any expert's training set. This is the primary (but incomplete) mechanism for addressing data imbalance.

### Document-level splits
Train/val splits are constructed at the document level: all chunks from a document go to the same split. This prevents data leakage where the model sees different chunks of the same document in training and validation.

### Held-out topic: immigration
Immigration is removed from every expert's training set. It serves as the topic generalization test (`val_topic`). The 5 viable topics in training are: economy, foreign_policy, law_order, health_education, governance_institutions.

### Reproducibility
- Primary seed: 42 throughout (splits, sampling, training, subsampling at eval)
- Secondary seeds 123 and 456 explored for training
- All splitters sort by stable key (document_id or example_id) before shuffling
- Seed spread < 0.001 confirmed across training seeds

### Strict validation, no silent fallback
Every stage raises on the first malformed input. Bad rows must be fixed upstream; nothing is silently dropped (except the explicit boilerplate filter in stage 06, which is a documented design choice).

### Synthetic supervision for router
No hand-labeled routing targets. Supervision is constructed by generating candidates, running MoCE with forced policies, and scoring outputs with bias/quality/refusal/vagueness signals. Score-weighted softmax over candidates produces the target distribution.

---

## Environment and Dependencies

- **Conda environment:** `lt-proj`
- **Python:** 3.12
- **GPU used for training:** H200 (for router training and MoCE inference); A100 used for expert training (~4h per expert on A100)
- **GPU memory:** ~14 GB for base weights in bfloat16, plus 4 LoRA adapters (minimal overhead at r=8, q_proj+v_proj only)

### Key packages
- `transformers >= 4.41.0` (model loading, tokenization)
- `peft` (LoRA adapter support)
- `accelerate >= 0.30.0`
- `datasets >= 2.19.0`
- `scikit-learn` (logistic probe, PCA)
- `torch` (bfloat16 and tf32 required)
- `sentence-transformers`
- `pyreadr` (RDS file loading)
- `pyyaml >= 6.0`

---

## Limitations (as documented in code and analysis)

1. **Expert training dataset imbalance**: right_auth has only 1,304 training examples vs. 5,118 for left_lib. Cell capping at 250 partially mitigates but does not eliminate this.

2. **Source medium imbalance**: right_auth is 66% Reddit (social media). left_auth is 52% parliamentary speeches. These medium differences may conflate political orientation with writing style.

3. **right_lib specialization failure**: val_indist and val_source loss differ by only 0.011, suggesting the right_lib adapter may not have learned quadrant-specific signal. Likely a data quality issue.

4. **LoRA r=8 capacity**: all experts saturate by epoch 2, suggesting the adapter capacity is the binding constraint. Ablation with higher ranks was not completed.

5. **Topic labeling without threshold**: every retained chunk gets a topic label regardless of how close the cosine similarity was to the prototype. Some chunks may be mislabeled.

6. **Compass center stability**: center depends on the specific set of neutral prompts and is specific to Mistral-7B. Different models would have different centers.

7. **No external validation of political orientation**: quadrant assignment is self-consistent (based on the model's own representations) but not validated against external political annotations except via the DW-NOMINATE analysis (scripts 14–16, Hein congressional speeches).

---

## Hein Congressional Validation (scripts 14–16)

Scripts `14_hein_build_dataset.py`, `15_hein_project_compass.py`, `16_hein_dwnominate_analysis.py` provide an external validity check: project US congressional speeches (Hein corpus, sessions 97–114, Democrat and Republican only) onto the political compass, then compare the resulting economic scores to DW-NOMINATE dimension 1 scores.

This serves as the external ground truth for the economic axis: if the steering vector is correct, Democrat speeches should score economically left and Republican speeches should score economically right, consistent with their DW-NOMINATE positions.
