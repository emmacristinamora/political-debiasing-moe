# MoCE — Technical Paper Context

Organized by component. All numeric values, thresholds, and design choices extracted directly from code and config. Written for the paper.

**Base model:** `mistralai/Mistral-7B-v0.1` (7B parameters, decoder-only, no instruction tuning)
**Single config:** `config/config.yaml` — every hyperparameter, path, and flag for every stage

---

## 1. Steering Vectors

**Scripts:** `01–04`, `13`, `17`
**Purpose:** Build two axis-aligned unit vectors in Mistral-7B's representation space — one for the economic axis (left ↔ right) and one for the social axis (libertarian ↔ authoritarian). These vectors are the backbone of every downstream projection in the system.

---

### 1.1 Contrastive Pair Construction (`src/01_build_pairs.py`, `src/02_validate_pairs.py`)

Steering vectors are derived from **180 contrastive prompt pairs** (90 per axis). Each pair consists of two prompts arguing the same political proposition from opposing poles.

**Generation:**
- 30 seed statements per axis, written as politically neutral propositions
- 3 paraphrase templates applied to each statement → 90 pairs per axis
- Templates: (1) analytical argument, (2) value-prioritization explanation, (3) policy-tradeoff discussion

**Labels:**
- Economic: positive = `econ_right`, negative = `econ_left`
- Social: positive = `authoritarian`, negative = `libertarian`

**Output schema (JSONL):** `id`, `axis`, `statement_id`, `statement`, `template_id`, `negative_label`, `positive_label`, `neg`, `pos`

**Validation checks (script 02, in order):**
1. Missing required fields
2. Type errors (all text fields must be strings)
3. Invalid axis (`economic` or `social` only)
4. Empty `pos` or `neg`
5. `identical_prompts_exact` — pos == neg
6. `identical_prompts_normalized` — after lowercasing, stripping `[^\w\s]`, collapsing whitespace
7. `length_ratio_too_high` — max(len(pos), len(neg)) / min > 1.5
8. `duplicate_pair` — same (axis, statement_id, template_id, normalized_pos, normalized_neg)

---

### 1.2 Activation Extraction (`src/03_extract_activations.py`)

**Model:** `mistralai/Mistral-7B-v0.1`, float16, `output_hidden_states=True`, eval mode
**Layers extracted:** `[8, 12, 16, 20, 24]`
**Pooling:** mask-weighted mean over non-padding tokens (Mistral has no pad token; `pad_token` set to `eos_token` before tokenization)
**Batch size:** hard-constrained to 2 — one positive + one negative per pair, enforced with `ValueError`
**Layer indexing:** `hidden_states[layer_index + 1]` (index 0 = embedding layer); this convention is used identically in scripts 05, 09, 18, 20
**Saved dtype:** float32 (model runs in float16; upcast on save)

**Output artifact:** `{axis}_activations.pt`
- Keys: `pair_ids`, `statement_ids`, `template_ids`, `token_counts`, `activations`
- `activations`: `layer_int → {"pos": tensor[N, D], "neg": tensor[N, D]}`

---

### 1.3 Vector Construction (`src/04_build_steering_vectors.py`)

Two methods are computed and saved for every layer. **Only `logistic_regression` is used in production.**

**Method 1 — `mean_difference`:**
- Vector = μ_pos − μ_neg, L2-normalized
- Quality score = |mean_pos_proj − mean_neg_proj| / pooled_std (Cohen's d style separation)

**Method 2 — `logistic_regression` (production):**
- `sklearn.LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)`
- Coefficient vector extracted, L2-normalized
- Quality score = 0.6 × train_accuracy + 0.4 × min(separation / 2, 1.0)

**Sign convention (enforced once, never re-checked downstream):**
After building the vector, if `mean(pos_projections) < mean(neg_projections)` the vector is flipped. After this step: positive dot product = economically right / authoritarian; negative dot product = economically left / libertarian.

**Cross-layer aggregation:**
- Quality-weighted mean of per-layer unit vectors
- Weights clipped at 1e-12, normalized to sum = 1
- Final vector re-normalized to unit norm
- The aggregated "final" vector is what all downstream code uses

**Named directions in artifact:**
- Economic: `econ_right` (final vector), `econ_left` (its negation)
- Social: `authoritarian` (final vector), `libertarian` (its negation)

---

### 1.4 Validation

#### Tier 1 — Geometry (`src/13_steering_vector_geometry.py`)
- ROC-AUC for separating pos from neg projections
- Permutation test: 500 label shuffles, null distribution of |Spearman r|
- 5-fold cross-validation accuracy
- Both methods evaluated

#### Tier 3 — Robustness (`src/17_steering_vector_robustness.py`)
- **Leave-one-group-out:** for each topic/template group, rebuild vector from all other pairs, test AUC on held-out group → measures generalization beyond training statements
- **Template invariance:** cross-template Pearson correlation of signed scores (pos_proj − neg_proj) per statement; high correlation = paraphrase-invariant rankings
- Separation metric: Cohen's d = (mean_pos − mean_neg) / pooled_std

---

## 2. Experts

**Scripts:** `normalize_corpora.py`, `05–08`, `19`
**Purpose:** Fine-tune four LoRA adapters on Mistral-7B, each specialized for one quadrant of the political compass.

---

### 2.1 Corpus Normalization (`data/experts/normalize_corpora.py`)

Converts raw corpora (CSV, JSON, RDS) to a canonical `Document` JSONL schema before any scoring.

**Canonical Document fields:** `document_id`, `text`, `source_name`, `source_family`, `language`, `raw_dataset`, `title`, `date`, `speaker_or_author`, `twitter_flag`, `metadata`

**Sources and formats:**

| Source key | Format | Raw file | Min words |
|---|---|---|---|
| allsides | CSV | allsides_balanced_news_headlines-texts.csv | 30 |
| reddit_liberal | JSON | Liberal.json | 100 |
| reddit_conservative | JSON | Conservative.json | 100 |
| hoc | RDS (R subprocess) | Corp_HouseOfCommons_V2.rds | 30 |
| ec_press | RDS (R subprocess) | EC-PressReleases_1985-2020_clean.RDS | 100 |
| uk_press | RDS (pyreadr) | UK-GovPressReleases.Rds | 100 |
| ire_press | RDS (pyreadr) | IRE-GovPressReleases.Rds | 100 |

HoC and EC press releases require an R subprocess (pyreadr encoding failures with those specific RDS files). R subprocess timeout: 600 s. Global max words: 800 for all sources.

**Text normalization:** `\r\n`/`\r` → `\n`; multiple spaces/tabs → single space; 3+ consecutive newlines → 2 newlines; strip leading/trailing whitespace.

**Source family labels:** `news_article` (allsides), `article_opinion` (reddit), `institutional_speech` (hoc, uk_press if speech=True), `institutional_press_release` (ec_press, uk_press if speech=False, ire_press)

---

### 2.2 Corpus Scoring and Quadrant Assignment (`src/05_quadrant_datasets.py`)

Scores every chunk of every source on both political axes and assigns quadrant + topic labels.

**Model:** `Mistral-7B-v0.1`, float16, `output_hidden_states=True`, eval mode — same model as activation extraction

**Chunking:**
- Whitespace-token sliding window: size 512, overlap 128 (step = 384)
- Minimum chunk length: 128 tokens (global); overridden to 30 for allsides and hoc via `--min-chunk-tokens`
- For us_media: `--min-chunk-tokens 64`

**Encoding layer: Layer 20**
- Chosen empirically: peak economic separability at mid-network; social separability increases in later layers; layer 20 provides the best combined trade-off
- Layer indexing: `hidden_states[21]` (same convention as extraction)

**Score computation:**
```
h = mean-pool(hidden_states[21], non-padding tokens)
h = h / ||h||₂                          # L2-normalize
score_econ = h · v_econ                 # cosine similarity (both unit-normed)
score_soc  = h · v_soc
confidence_margin = min(|score_econ|, |score_soc|)
```

**Retention threshold:** `confidence_margin ≥ 0.015` on both axes simultaneously. Set empirically. Chunks below this threshold are written to `scored_chunks.jsonl` as an audit trail but excluded from the retained pool.

**Quadrant assignment** (relative to compass center `(c_e, c_s)`):
```
se = score_econ − c_e
ss = score_soc  − c_s
right_auth:  se ≥ 0, ss ≥ 0
left_auth:   se < 0, ss ≥ 0
left_lib:    se < 0, ss < 0
right_lib:   se ≥ 0, ss < 0
```
In the current experiments the compass center was set to `(0, 0)` (see §2.8 Limitations).

**HoC stratified sampling:** 200,000 speeches drawn from 1.6M total; strata = decade × party, proportional allocation with minimum 1 per stratum, seed=42.

**Nine corpora across three medium categories:**

| Medium | Sources |
|---|---|
| Social media | reddit_liberal, reddit_conservative |
| Press/media | allsides, ec_press, uk_press, ire_press, us_media |
| Speeches | hoc, us_speeches |

The three-medium split guards against the model learning medium style rather than political orientation.

**Topic labeling (9 topics, all retained chunks):**
Prototype texts are embedded once at startup using the same Mistral-7B. `topic_primary` = nearest prototype (cosine); `topic_secondary` = second nearest. No threshold — every retained chunk is assigned a topic.

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

**Outputs per source:**
- `retained.jsonl` — chunks passing all thresholds
- `scored_chunks.jsonl` — all chunks (audit trail, no GPU required for downstream re-use)
- `document_summaries.jsonl` — per-document stats (mean/std scores, dominant quadrant)
- `report.json` — volume, source/topic composition, top-5 highest-confidence examples

---

### 2.3 Dataset Construction (`src/06_validate_experts_datasets.py`)

Builds reproducible train/val splits for each of the 4 experts with three held-out generalization axes.

**Internal pipeline (in order):**
1. Load all retained chunks from quadrant pool
2. Topic filter: keep only `viable_topics ∪ {held_out_topic}`
3. Boilerplate filter: 12 regex patterns strip academic/survey artefacts ("APA style", "reference page", "your paper", "assignment guidelines", "write a short argument", "complete the survey", "submit your ideas", "we want to hear from you", "download the submission form", "related policy papers", "newsletter", "subscribe")
4. Carve `held_out_topic` chunks → `val_topic` pool
5. Carve `held_out_source` chunks → `val_source` pool
6. Cell diagnostics (source × topic breakdown)
7. Cell-cap: for each (source × topic) cell, `random.sample(250, seed=42)` if over cap
8. Document-level split: shuffle unique doc_ids (seed=42), first `ceil(n_docs × 0.15)` doc_ids → `val_indist`, rest → train
9. Sanity checks (hard `sys.exit(1)` on failure)
10. Write outputs

**Topic configuration:**
- Viable (in training): `economy`, `foreign_policy`, `law_order`, `health_education`, `governance_institutions`
- Held-out topic: `immigration` — excluded from every expert's training set; used as topic generalization test

**Held-out source per expert:**
| Expert | Held-out source (val_source) |
|---|---|
| right_auth | allsides |
| left_auth | uk_press |
| left_lib | ire_press |
| right_lib | uk_press |

**Cell cap:** 250 chunks per (source × topic) cell, applied before the train/val split. Primary (but incomplete) mechanism for addressing corpus imbalance.

**Split ratios:** 85% train / 15% val_indist (document-level; no chunk from the same document appears in both splits).

**Length filter (before cell capping):** min 50 tokens, max 700 tokens.

**Sanity checks:**
| Check | Threshold |
|---|---|
| MAX_CELL_SIZE | No (source × topic) cell > 250 in train |
| SOURCE_CAP | No single source > 70% of train |
| TOPIC_CAP | No single topic > 50% of train |
| TOPIC_KL | KL divergence of topic distribution ≤ 8.0 |
| LENGTH_RATIO | max/min chunk length ≤ 3.0 |
| CONF_MARGIN_RATIO | max/min confidence margin ≤ 2.5 |
| NO_DOC_LEAKAGE | No doc_id in both train and any val split |
| VAL_TOPIC_PURITY | val_topic contains only held_out_topic |
| VAL_SOURCE_PURITY | val_source contains only held_out_source |
| NON_EMPTY | All splits ≥ 15 examples |

**Final dataset sizes:**
| Expert | Train | Val_indist | Val_source | Val_topic |
|---|---|---|---|---|
| right_auth | 1,304 | — | — | — |
| left_auth | 2,057 | — | — | — |
| right_lib | 3,669 | — | — | — |
| left_lib | 5,118 | — | — | — |

**Source group composition (train):**
| Expert | reddit % | press % | speeches % |
|---|---|---|---|
| right_auth | 66% | 29% | 4% |
| left_auth | 16% | 32% | 52% |
| left_lib | 40% | 39% | 21% |
| right_lib | 55% | 21% | 24% |

---

### 2.4 Expert Training (`src/07_train_experts.py`)

**Base model:** `mistralai/Mistral-7B-v0.1`, bfloat16. All base weights frozen.

**LoRA configuration:**
| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Target modules | `q_proj`, `v_proj` |
| Bias adaptation | none |
| Task type | CAUSAL_LM |

**Training hyperparameters:**
| Parameter | Value |
|---|---|
| Epochs | 5 |
| Learning rate | 8e-5 |
| LR scheduler | cosine |
| Warmup ratio | 0.10 |
| Weight decay | 0.01 |
| Per-device batch | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Max grad norm | 1.0 |
| Precision | bf16 + tf32 |
| Evals per epoch | 4 |
| Max tokenized length | 700 tokens |
| Hardware | H200 GPU (~4h per expert) |

**Custom loss (WeightedTrainer):**
- Per-example loss = mean CE over non-padding tokens
- Batch loss = inverse-frequency source-group weighted mean
- Groups: `reddit` (reddit_liberal + reddit_conservative), `press` (allsides, ec_press, ire_press, uk_press, us_media), `speeches` (hoc, us_speeches)
- Weight formula: `n_total / (n_groups × count_g)`, capped at 3.0
- Motivation: downweight reddit_conservative (~690K chunks, the dominant source)

**Known artefact:** logged training loss appears ~4–5× higher than val loss due to WeightedTrainer × grad_accum_steps=4 interaction. Cosmetic only — does not affect checkpoint selection.

**Evaluation (MultiSplitEvalCallback):**
- Three val splits evaluated at each eval step
- Each subsampled to 500 examples at eval time (seed=42) for speed
- `mean_val_loss = (val_indist + val_source + val_topic) / 3.0`
- Best checkpoint saved by callback (independent of HuggingFace trainer logic, which is disabled for checkpoint tracking)

**Seeds explored:** [42, 123, 456]. Seed spread < 0.001 across all experts — training is fully reproducible.

**Best checkpoint results (Run 2):**
| Expert | Best seed | Best epoch | mean_val_loss | val_indist | val_source | val_topic |
|---|---|---|---|---|---|---|
| right_auth | 42 | 1.96 | 1.7085 | 1.8505 | 1.6052 | 1.6697 |
| left_auth | 42 | 1.74 | 1.8626 | 2.0191 | 1.7590 | 1.8098 |
| left_lib | 456 | 1.75 | 1.8842 | 2.0008 | 1.8093 | 1.8426 |
| right_lib | 456 | 1.985 | 1.8227 | 1.8332 | 1.8225 | 1.8123 |

All experts saturate by epoch ~2. LoRA r=8 is the binding capacity constraint.

---

### 2.5 Expert Evaluation (`src/08_test_experts.py`)

Three behavioral methods, all using layer 20 projections. Model is fully reloaded between expert conditions to prevent adapter state leakage.

**Method 1 — Representativeness:**
- For each PCT statement: generate a free-form response (greedy decoding, max_new_tokens=120)
- Project `(statement + "\n\nResponse:\n" + generated_response)` at layer 20
- Pooling: response tokens only (prefix positions masked to zero); falls back to all tokens if response is empty
- Records: PCT coordinates, predicted quadrant, quadrant-match rate

**Method 2 — Inverse Steerability:**
- Prepend an adversarial counter-persona (targeting the opposite quadrant) before each statement
- Each trained expert runs only its designated adversary; base model runs all 4 for comparison
- Measures: `shift_magnitude = sqrt(Δecon² + Δsoc²)` vs. Method 1 baseline
- Key metric: `resistance_vs_base` — expert resists adversarial framing more than the base model does

**Method 3 — Consistency:**
- Closed-form survey questions with multiple answer options
- Select option with lowest average NLL (leading space prepended for correct tokenization; "Refused" options excluded)
- Project `(question + "\n\nAnswer:\n" + selected_option)` at layer 20
- Measures whether implicit preference (via perplexity) aligns with target quadrant

---

### 2.6 Repartition (`src/19_repartition_chunks.py`)

Re-applies a calibrated compass center to existing scored chunks **without re-running the model**. Reads `scored_chunks.jsonl` (which stores raw projections), recomputes only the center-relative fields (`quadrant`, `threshold_pass`, `score_abs_econ`, `score_abs_soc`, `confidence_margin`), and rewrites `retained.jsonl` and `report.json` per quadrant.

Used when the compass center changes post-hoc. No GPU required.

---

### 2.7 Dataset Limitations

1. **Expert size imbalance:** right_auth has 1,304 training examples vs. 5,118 for left_lib. Cell capping at 250 partially mitigates but does not eliminate the imbalance.

2. **Medium imbalance per expert:** right_auth is 66% Reddit; left_auth is 52% parliamentary speeches. These differences may conflate political orientation with writing style or register.

3. **right_lib specialization failure:** val_indist and val_source loss differ by only 0.011, suggesting the right_lib adapter learned little quadrant-specific signal. Likely a data quality issue (insufficient right-libertarian corpora).

4. **Topic labeling without threshold:** every retained chunk is assigned a topic regardless of the similarity margin to the prototype. Some chunks may be mislabeled.

5. **Compass center calibration (script 18):** script 18 computes a neutral reference point by projecting apolitical prompts at layer 20 and validating the centroid against four acceptance criteria (subcategory agreement, midpoint distance, outlier z-scores, bootstrap stability). The calibrated center is consumed by script 19 (repartition) to shift all quadrant boundaries. **This experiment was not included in the main results** because we are not confident in the validity of the neutral prompt set and the resulting center estimate. The compass center was set to `(0, 0)` for all reported results. This will need to be repeated with a better-validated neutral set before including the calibration in the paper.

---

## 3. MoCE Architecture

**Scripts:** `09_moce_components.py`, `10_run_moce.py`

---

### 3.1 Components (`src/09_moce_components.py`)

**Constants:**
- `CANONICAL_QUADRANT_ORDER = ("left_lib", "left_auth", "right_lib", "right_auth")` — fixed everywhere
- `InputTransformer.ENCODING_LAYER = 20`
- `InputTransformer.DEFAULT_MAX_LENGTH = 512`

**InputTransformer:**
- Encodes a prompt via Mistral-7B at layer 20, mean-pools over non-padding tokens, L2-normalizes
- Projects onto 4 quadrant basis vectors to produce `quadrant_scores`
- `bias_magnitude = sqrt(score_econ² + score_soc²)` — scalar measure of the prompt's distance from the compass center

**Quadrant basis vectors** (diagonal directions in compass space):
```
left_lib   = normalize(−v_econ − v_soc)
left_auth  = normalize(−v_econ + v_soc)
right_lib  = normalize(+v_econ − v_soc)
right_auth = normalize(+v_econ + v_soc)
```

**Heuristic router (no learned parameters):**
```
π₀ = softmax(−β × quadrant_scores / T)
```
- β = 1.0, T = 1.0 (both from config)
- The prior is counterbalancing: a prompt scoring high on `right_auth` receives high prior weight on `left_lib`
- Centered prompts (`bias_magnitude ≤ 0.05`): fall back to uniform [0.25, 0.25, 0.25, 0.25]

**Calibrated router (learned):**
```
π = softmax(log π₀ + δ(h))
```
Correction head δ(h) is a small learned network trained via synthetic supervision (see §5).

**ExpertManager:** loads all four LoRA adapters and runs them in dense mode (all four forward passes per step).

**Editor stopping conditions:**
| Condition | Threshold |
|---|---|
| axis_proximity | min(|score_econ|, |score_soc|) ≤ 0.015 — prompt already near-center |
| converged | max_alpha_change ≤ 1e-3 and max_alignment_change ≤ 1e-3 |
| max_steps | 10 (hard cap) |

**Generation defaults (config):**
- `max_new_tokens: 256`, `temperature: 0.7`, `do_sample: False`, `top_p: 1.0`
- `frequency_penalty: 0.4`, `no_repeat_ngram_size: 3`

---

### 3.2 Inference Runner (`src/10_run_moce.py`)

CLI entry point. Execution order: `transform → route → edit → decode`.

**Mutual exclusions (enforced):**
- Exactly one of `--prompt` / `--prompts-file`
- `--calibrated` requires `--router-checkpoint` and vice versa

**Output fields per prompt (JSONL):** `id`, `prompt_text`, `final_text`, `router_mode`, `bias_magnitude`, `economic_score`, `social_score`, `quadrant_scores`, `heuristic_prior`, `calibrated_policy`, `final_alpha`, `final_alignment`, `num_edit_steps`, `stopped_early`, `stop_reason`, `editor_trace`

**Editor trace per step:** `step_index`, `alpha_before`, `delta`, `alpha_after`, `alignment_before`, `alignment_after`, `max_alpha_change`, `max_alignment_change`, `economic_score`, `social_score`, `bias_magnitude`

**GPU memory:** ~14 GB for base weights in bfloat16, plus four LoRA adapters (minimal overhead at r=8, q_proj+v_proj only). Requires a single A100 (40 GB) or H200.

---

## 4. Evaluation

**Scripts:** `11_moce_evaluation.py`, `12_judge_evaluation.py`, `20_compass_comparison.py`, `21_plot_compass_comparison.py`

---

### 4.1 Automatic Evaluation (`src/11_moce_evaluation.py`)

Subcommand structure and dataclasses are in place. Individual metric CLIs (bias radius, refusal/vagueness rate, quality, robustness) are not yet fully implemented.

Planned metrics:
- `bias_radius` — distance from compass center in the output projections
- `refusal` / `vagueness` — heuristic string matching
- `quality` — word count, sentence count, long-word density
- `robustness` — consistency across paraphrase variants of the same prompt

---

### 4.2 LLM-as-Judge (`src/12_judge_evaluation.py`)

**Judge model:** `meta-llama/Llama-3.1-8B-Instruct`

**Stance classification (`stance` subcommand):**
- 6-point scale: strongly_disagree (−2), disagree (−1), neutral (0), agree (+1), strongly_agree (+2), incoherent (0)
- Compass coordinates derived from stance scores using the polarity key in `config/pct_eval_polarity.yaml`
- Judge prompt explicitly distinguishes "neutral" (clear and balanced) from "incoherent" (confused/evasive)

**Pairwise comparison (`pairwise` subcommand):**
- Blind head-to-head on neutrality and coherence
- Judge outputs A / B / tie per criterion

---

### 4.3 Multi-Model Compass Comparison (`src/20_compass_comparison.py`, `src/21_plot_compass_comparison.py`)

**Purpose:** Compare the political compass position of MoCE outputs against five baseline models on a shared set of evaluation prompts.

**Models evaluated:**
| Model | Role |
|---|---|
| mistralai/Mistral-7B-v0.1 | Base reference |
| run_moce | MoCE debiasing architecture |
| Qwen/Qwen2.5-7B-Instruct | Qwen 2.5 7B |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1 7B |
| meta-llama/Llama-3.1-8B-Instruct | Llama 3.1 8B |
| google/gemma-2-9b-it | Gemma 2 9B |

**Evaluation protocol (script 20):**
- N_RESPONSES = 10 per prompt (temperature=0.8, max_new_tokens=300)
- Each response projected at layer 20 using `mean_difference` vectors (Mistral-7B as fixed projector)
- Per-prompt centroid = mean of 10 (econ, soc) pairs
- Global centroid = mean of per-prompt centroids
- Models are loaded and unloaded sequentially (no two large models in memory simultaneously)
- Response cache is written to `data/evaluation/compass_comparison/responses/` and checked before regenerating (script is resumable)

**Vector method used for projection:** `mean_difference` (not logistic_regression — this is a deliberate choice for the comparison script; the rest of the system uses logistic_regression)

**Plot (script 21):**
- Paper-quality scatter plot with quadrant background tints (standard political compass color convention)
- ±1σ ellipse drawn for MoCE only (spread across prompts)
- Separate annotation position (`label_xy`) and leader-arrow toggle per model
- Output: `docs/fig_compass_comparison.png`, 300 dpi

---

## 5. Other

### 5.1 Calibrated Router Training (`src/router_training/`)

Trains a correction head δ(h) to improve on the heuristic prior via synthetic supervision.

**11-step pipeline:**
| Step | Script | Output |
|---|---|---|
| 1 | config.py | Validated RouterCalibrationConfig |
| 2 | prompt_set.py | data/router/prompts.jsonl |
| 3 | features.py | data/router/features.jsonl, hidden.pt |
| 4 | utils.py | candidate policies (in-memory) |
| 5 | forced_policy_runner.py | data/router/candidate_traces.jsonl |
| 6 | scorer.py | data/router/candidate_scores.jsonl |
| 7 | targets.py | data/router/records.jsonl, target_report.json |
| 8 | validator.py | validation gate (no output artifact) |
| 9 | splitter.py | data/router/{train,val,test}/records.jsonl |
| 10 | train_pipeline.py | data/router/checkpoints/calibrated_router.pt |
| 11 | evaluator.py | data/router/reports/router_checkpoint_eval.json |

**Candidate policies explored (Step 4):** heuristic prior, uniform, sharpened (T=0.5), softened (T=2.0), opposite-heavy, adjacent-heavy, 16 Dirichlet samples (concentration=64.0). All floored at `min_probability=1e-6` and de-duplicated.

**Composite score weights:**
| Component | Weight |
|---|---|
| bias_radius | 1.0 |
| quality | 0.5 |
| refusal | 0.5 |
| vagueness | 0.3 |
| kl_to_prior | 0.1 |

**Target policy construction:** softmax over candidate scores at `score_temperature=0.2` → mixture weights → convex combination of candidate policies, floored at 1e-6.

**Router training hyperparameters:**
| Parameter | Value |
|---|---|
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 20 |
| KL weight | 0.1 |
| Entropy weight | 0.01 |
| Data split | 80/10/10 by source |

**Key design choice:** no hand-labeled routing distributions — supervision is fully synthetic.

**REQUIRED_LAYER = 20** enforced in `router_training/config.py` (raises if missing from `selected_layers`).

---

### 5.2 Hein Congressional Validation (`src/14` → `src/16`)

External validity check for the economic steering vector.

**Data:** Hein congressional speeches, sessions 97–114 (1981–2017), Democrat and Republican only (`VALID_PARTIES = {"D", "R"}`). Procedural speeches filtered out via substring match against a `procedural.txt` list.

**Pipeline:**
1. `14_hein_build_dataset.py` — normalize congressional speeches to the canonical Document schema
2. `15_hein_project_compass.py` — project each speech at layer 20 onto both axes
3. `16_hein_dwnominate_analysis.py` — correlate per-legislator economic_coord means with DW-NOMINATE dimension 1

**Headline test:** Spearman correlation of `economic_coord` vs. `nominate_dim1`. A strong positive correlation means Democrat speeches score economically left and Republican speeches score economically right, consistent with DW-NOMINATE ideology estimates. **Split-half reliability** of the per-legislator economic coordinate provides an upper bound on how high any external correlation can be, given within-legislator variance.

**Additional checks:**
- Full coord × dimension correlation matrix (discriminant validity)
- Party separability AUC: economic_coord as classifier of Democrat vs. Republican

**Permutation test:** 10,000 shuffles, two-sided, on Spearman r.

---

### 5.3 Cross-Cutting Design Decisions

**Layer 20 universally.** Economic axis peaks at mid-network; social axis grows in later layers; layer 20 is the empirically validated compromise. All scripts use `hidden_states[LAYER + 1]` (index 0 = embedding layer) — this convention is enforced by `REQUIRED_LAYER = 20` in router_training/config.py and is assumed (not checked) in all other scripts.

**Unit normalization throughout.** All vectors — steering vectors, chunk embeddings, quadrant basis vectors — are L2-normalized before any dot product. Cosine similarity and dot product are interchangeable throughout the codebase.

**Sign convention fixed in script 04.** Positive dot product = economically right / authoritarian. Never re-verified downstream.

**Document-level train/val splits.** All chunks from a document are assigned to the same split. Prevents the model seeing different excerpts of the same source document in both train and validation.

**Canonical quadrant order fixed.** `("left_lib", "left_auth", "right_lib", "right_auth")` defined once in `09_moce_components.py` as `CANONICAL_QUADRANT_ORDER` and replicated in `router_training/trainer.py` (which cannot import the file due to its digit prefix but matches by construction).

**Strict validation, no silent fallback.** Every stage raises on the first malformed input. The only intentional silent drop is the boilerplate filter in script 06.

**Reproducibility.** Primary seed: 42 throughout (splits, sampling, eval subsampling, training). Secondary seeds 123 and 456 for training. Seed spread < 0.001 confirmed. All splitters sort by a stable key (document_id or example_id) before shuffling.
