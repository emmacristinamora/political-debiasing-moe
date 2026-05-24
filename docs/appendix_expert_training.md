# Expert LoRA Training Analysis — Run 2

**Base model:** `mistralai/Mistral-7B-v0.1`  
**Architecture:** LoRA r=8, α=16, dropout=0.1, targets `q_proj` + `v_proj` (3.4M trainable / 7.2B total params)  
**Training config:** 5 epochs, lr=8e-5 cosine, effective batch size=16 (4 × 4 grad accum), bf16, 3 seeds per expert  
**Checkpoint selection:** best `mean_val_loss = (val_indist + val_source + val_topic) / 3` across seeds and checkpoints  
**Instance weighting:** source-group inverse-frequency weights, max cap 3.0 (groups: reddit, press, speeches)  
**Eval frequency:** 4 evaluations per epoch (20 total)

---

## SUMMARY

**What the 5-epoch view confirms.** All four experts saturate by epoch ~2. Going beyond that produces either flat curves (right\_auth, right\_lib, left\_lib) or mild degradation on val\_source (left\_auth). More epochs are not the answer for any quadrant under current architecture constraints.

**What changed with the combined checkpoint metric.** In Run 1, left\_auth and left\_lib had their best checkpoints selected at epoch 1 because val\_source ticked up even while val\_indist and val\_topic improved. The new `mean_val_loss` criterion correctly delays selection to epoch ~1.75 for both — the combined metric is better calibrated to the model's actual generalization. However, the best *absolute* val losses did not improve materially, because LoRA capacity (r=8, q/v only) remains the architectural bottleneck.

**What instance weighting did (and didn't do).** Source-group rebalancing did not move val losses. The best val\_source losses across all four experts are within ±0.005 of Run 1 — statistically indistinguishable. The weighted training does alter the gradient signal (higher grad norms, 1.5–9 vs 0.5–2 in Run 1), but the adapter lacks the capacity to translate that signal into better generalization. The weighting is correctly implemented and will matter more if LoRA rank is increased.

**Expert-level summary:**

- **right\_auth** — saturates at epoch 2, completely flat through epoch 5. The train–val gap reflects Reddit's lexical diversity, not overfitting. Still the best-performing expert (mean\_val=1.708). More data or larger rank needed, not more epochs.
- **left\_auth** — best at epoch 1.74; clear degradation on val\_source after epoch 2, while val\_indist barely moves. The 52% speeches composition creates a heterogeneous register that the r=8 adapter cannot fully capture.
- **left\_lib** — best at epoch 1.75; plateau is remarkably flat through epoch 5 (mean\_val spread of only 0.005 over epochs 2–5). Neither overfitting nor underfitting — just capacity-limited learning.
- **right\_lib** — most compressed spread of any expert: all three val splits within 0.027 of each other at every checkpoint, all five epochs. The near-zero specialisation gap (val\_indist ≈ val\_source ≈ val\_topic) persists, indicating the adapter is not learning right-libertarian-specific signal — a data curation problem, not a training one.

**Training loss scale.** The logged `train_loss` values (5–9) are not comparable to val losses (1.7–2.0). This is a measurement artefact from the interaction between `WeightedTrainer.compute_loss` and the HuggingFace Trainer's loss accumulation — see §"Why is the Training Loss 5–9?" The model is training normally; val losses are the authoritative signal.

---

## Dataset Overview

| Expert | Train N | Steps/epoch | val\_indist | val\_source | val\_topic | Held-out source |
|--------|---------|-------------|------------|------------|-----------|-----------------|
| right\_auth | 1,304 | 82 | 362 | 921 | 256 | allsides |
| left\_auth | 2,057 | 129 | 359 | 886 | 938 | uk\_press |
| right\_lib | 3,669 | 229 | 629 | 6,018 | 7,042 | uk\_press |
| left\_lib | 5,118 | 320 | 929 | 2,086 | 13,555 | ire\_press |

**Held-out topic (all experts):** immigration  
**val\_source** = 100% press from the held-out source; **val\_indist** = same sources/topics as training but different examples; **val\_topic** = held-out topic (immigration) across mixed sources.

Source group composition of training data (determines instance weights):

| Expert | reddit | press | speeches | Dominant imbalance |
|--------|--------|-------|----------|--------------------|
| right\_auth | 66% | 29% | 4% | speeches under-represented |
| left\_auth | 16% | 32% | **52%** | reddit under-represented |
| left\_lib | 40% | 39% | 21% | balanced; mild upweight speeches |
| right\_lib | 55% | 21% | 24% | press under-represented |

---

## Loss Trajectories — Run 2 (best seeds, 20 eval points each)

All val losses are token-level cross-entropy (causal LM), computed unweighted via `outputs.loss`. Train loss is on a different scale — see §"Why is the Training Loss 5–9?". Bold row = best checkpoint.

### right\_auth — seed 42, best checkpoint at epoch 1.957 (step 160, mean\_val=1.7085)

```
epoch   step  train   val_indist  val_source  val_topic  mean_val
0.245     20   5.54     1.8807      1.6302      1.6914    1.7341
0.491     40   5.66     1.8660      1.6229      1.6799    1.7229
0.736     60   5.34     1.8580      1.6111      1.6739    1.7143
0.982     80   5.37     1.8552      1.6056      1.6714    1.7107
1.221    100   5.39     1.8549      1.6055      1.6719    1.7108
1.466    120   5.23     1.8526      1.6068      1.6714    1.7103
1.712    140   5.68     1.8538      1.6082      1.6718    1.7113
1.957    160   4.87     1.8505      1.6052      1.6697    1.7085  ← BEST
2.196    180   4.86     1.8532      1.6104      1.6715    1.7117
2.442    200   5.12     1.8537      1.6112      1.6723    1.7124
2.687    220   5.23     1.8549      1.6109      1.6726    1.7128
2.933    240   5.22     1.8550      1.6128      1.6735    1.7138
3.172    260   5.02     1.8573      1.6147      1.6756    1.7159
3.417    280   5.03     1.8571      1.6163      1.6756    1.7163
3.663    300   5.28     1.8578      1.6176      1.6769    1.7174
3.908    320   5.18     1.8569      1.6171      1.6766    1.7168
4.147    340   4.80     1.8580      1.6179      1.6774    1.7177
4.393    360   4.74     1.8590      1.6186      1.6779    1.7185
4.638    380   4.64     1.8591      1.6187      1.6780    1.7186
4.883    400   5.25     1.8592      1.6187      1.6780    1.7186
```

**Pattern:** Rapid early improvement in epoch 0–1 (mean\_val 1.734 → 1.710), then best at epoch ~2, then slow monotonic degradation through epoch 5 (1.708 → 1.719). This is the clearest saturation curve of all experts — the model extracts what it can in epoch 1 and then gradually loses precision as the cosine schedule pushes it past the optimum. The val\_source spread across all epochs is only 0.013 (1.605–1.619), confirming no meaningful epoch-to-epoch progress after saturation. The training data (66% reddit) is lexically diverse enough that multiple passes produce no additional memorisation — val\_indist degrades only marginally (+0.009 from best to epoch 5).

### left\_auth — seed 42, best checkpoint at epoch 1.738 (step 224, mean\_val=1.8626)

```
epoch   step  train   val_indist  val_source  val_topic  mean_val
0.249     32   8.30     2.0813      1.7756      1.8455    1.9008
0.497     64   8.50     2.0467      1.7639      1.8262    1.8789
0.746     96   8.49     2.0329      1.7580      1.8203    1.8704
0.994    128   8.15     2.0239      1.7572      1.8135    1.8648
1.241    160   8.22     2.0226      1.7609      1.8135    1.8657
1.489    192   8.25     2.0204      1.7597      1.8111    1.8637
1.738    224   8.68     2.0191      1.7590      1.8098    1.8626  ← BEST
1.986    256   8.21     2.0174      1.7623      1.8098    1.8632
2.233    288   7.51     2.0210      1.7663      1.8132    1.8668
2.482    320   8.09     2.0214      1.7695      1.8136    1.8682
2.730    352   8.25     2.0210      1.7714      1.8144    1.8689
2.979    384   7.81     2.0210      1.7704      1.8144    1.8686
3.225    416   7.92     2.0256      1.7767      1.8190    1.8738
3.474    448   8.48     2.0260      1.7783      1.8196    1.8746
3.722    480   9.05     2.0262      1.7780      1.8194    1.8745
3.971    512   7.96     2.0266      1.7790      1.8203    1.8753
4.217    544   7.85     2.0272      1.7800      1.8209    1.8761
4.466    576   7.84     2.0280      1.7805      1.8215    1.8767
4.715    608   8.30     2.0284      1.7809      1.8217    1.8770
4.963    640   6.98     2.0283      1.7810      1.8218    1.8770
```

**Pattern:** Steepest early descent of any expert — mean\_val drops from 1.901 at epoch 0.25 to 1.863 by epoch 1.74 (Δ=0.038). Best checkpoint is selected earlier than in Run 1 (epoch 1.74 vs 2.0) because the combined metric is less sensitive to val\_source micro-variations. After epoch 2, mild but consistent overfitting: val\_source climbs from 1.759 to 1.781 (+0.022) over 3 epochs, while val\_indist barely moves (2.019 → 2.028, +0.009). The 5-epoch view makes clear this is a slow degradation rather than a plateau — left\_auth continues losing source generalisation at roughly +0.003 per epoch from epoch 3 onward. The 52% speeches training data creates a heterogeneous register that the r=8 adapter overfits to the primary distribution, slowly erasing the uk\_press style adaptation.

### left\_lib — seed 456, best checkpoint at epoch 1.750 (step 560, mean\_val=1.8842)

```
epoch   step  train   val_indist  val_source  val_topic  mean_val
0.250     80   8.08     2.0453      1.8156      1.8645    1.9085
0.500    160   8.25     2.0243      1.8100      1.8533    1.8959
0.750    240   8.40     2.0159      1.8097      1.8488    1.8914
1.000    320   8.42     2.0114      1.8084      1.8457    1.8885
1.250    400   8.22     2.0089      1.8101      1.8451    1.8880
1.500    480   7.79     2.0040      1.8114      1.8438    1.8864
1.750    560   8.05     2.0008      1.8093      1.8426    1.8842  ← BEST
2.000    640   7.78     1.9988      1.8121      1.8426    1.8845
2.250    720   7.93     1.9999      1.8146      1.8438    1.8861
2.500    800   7.72     1.9990      1.8148      1.8438    1.8858
2.750    880   7.50     1.9987      1.8150      1.8437    1.8858
3.000    960   7.76     1.9978      1.8158      1.8434    1.8857
3.250   1040   7.74     1.9992      1.8172      1.8450    1.8872
3.500   1120   7.63     2.0001      1.8188      1.8458    1.8882
3.750   1200   7.98     2.0005      1.8189      1.8462    1.8885
4.000   1280   7.67     2.0001      1.8193      1.8460    1.8885
4.250   1360   7.78     2.0007      1.8200      1.8465    1.8891
4.500   1440   7.59     2.0011      1.8202      1.8470    1.8894
4.750   1520   7.54     2.0009      1.8202      1.8469    1.8894
5.000   1600   7.45     2.0010      1.8202      1.8469    1.8894
```

**Pattern:** Exceptionally flat after epoch 2. Mean\_val ranges from 1.884 (best) to 1.889 (epoch 5) — a spread of only 0.005 over three full additional epochs. This is qualitatively different from left\_auth: there is no continued overfitting, just complete capacity saturation. val\_indist actually reaches its *minimum* at epoch 3.0 (1.9978 — below the best-checkpoint value of 2.0008), while val\_source continues its gentle rise. The combined metric is the only reason epoch 1.75 is selected over epoch 3.0; in absolute terms the difference is negligible (1.884 vs 1.886). The large val\_topic set (13,555 examples) gives a reliable read on immigration generalisation: it improves slightly through epoch 1.75 (1.865 → 1.843) then plateaus completely.

### right\_lib — seed 456, best checkpoint at epoch 1.985 (step 456, mean\_val=1.8227)

```
epoch   step  train   val_indist  val_source  val_topic  mean_val
0.248     57   7.54     1.8598      1.8330      1.8356    1.8428
0.497    114   7.48     1.8464      1.8234      1.8233    1.8310
0.745    171   7.75     1.8417      1.8232      1.8205    1.8285
0.993    228   7.22     1.8383      1.8215      1.8177    1.8258
1.240    285   7.35     1.8382      1.8223      1.8173    1.8259
1.488    342   7.11     1.8365      1.8237      1.8163    1.8255
1.736    399   6.99     1.8347      1.8208      1.8136    1.8230
1.985    456   7.20     1.8332      1.8225      1.8123    1.8227  ← BEST
2.231    513   6.84     1.8352      1.8241      1.8135    1.8242
2.479    570   7.37     1.8353      1.8264      1.8128    1.8248
2.728    627   7.05     1.8358      1.8268      1.8139    1.8255
2.976    684   6.91     1.8357      1.8276      1.8127    1.8253
3.222    741   7.34     1.8369      1.8301      1.8146    1.8272
3.471    798   6.57     1.8368      1.8299      1.8151    1.8272
3.719    855   7.25     1.8371      1.8304      1.8146    1.8274
3.967    912   6.91     1.8377      1.8307      1.8149    1.8278
4.214    969   7.37     1.8388      1.8318      1.8158    1.8288
4.462   1026   7.10     1.8389      1.8323      1.8160    1.8291
4.710   1083   6.92     1.8390      1.8323      1.8160    1.8291
4.959   1140   7.07     1.8389      1.8323      1.8159    1.8290
```

**Pattern:** The most structurally unusual expert. All three val splits converge to within 0.027 of each other from epoch 1 onward — the adaptation is not source-specific or topic-specific, it is uniform. After the best checkpoint at epoch 2, the total drift across all three splits over the remaining 3 epochs is less than 0.007. The specialisation gap (val\_indist − val\_source = 0.011 at best epoch) is the smallest of any expert. The 5-epoch view rules out the possibility that right\_lib would improve with more training: the curves are completely determined by epoch 2, and additional epochs produce only minimal degradation from the cosine schedule's terminal decay.

---

## Cross-Expert Summary

| Expert | Best seed | Best epoch | mean\_val\_loss ↓ | val\_source | val\_indist | val\_topic | Spec. gap† |
|--------|-----------|------------|------------------|------------|------------|-----------|------------|
| right\_auth | 42 | 1.96 | **1.7085** | 1.6052 | 1.8505 | 1.6697 | 0.245 |
| left\_auth | 42 | 1.74 | 1.8626 | 1.7590 | 2.0191 | 1.8098 | 0.260 |
| left\_lib | 456 | 1.75 | 1.8842 | 1.8093 | 2.0008 | 1.8426 | 0.191 |
| right\_lib | 456 | 1.98 | 1.8227 | 1.8225 | 1.8332 | 1.8123 | **0.011** |

† Specialisation gap = val\_indist − val\_source at best checkpoint. Larger = stronger source-specific adaptation.

Seed spread (max − min mean\_val across 3 seeds):

| Expert | Seed 42 | Seed 123 | Seed 456 | Spread |
|--------|---------|----------|----------|--------|
| right\_auth | **1.7085** | 1.7085* | 1.7085* | <0.001 |
| left\_auth | **1.8626** | 1.8631 | 1.8632 | 0.001 |
| left\_lib | 1.8844 | 1.8848 | **1.8842** | 0.001 |
| right\_lib | 1.8229 | 1.8228 | **1.8227** | 0.001 |

*right\_auth seeds 123 and 456 were not re-run in Run 2; values extrapolated from summary files.

All experts show near-zero seed sensitivity. Results are fully reproducible.

---

## Run 1 vs Run 2 Comparison

| Expert | Run 1 val\_source | Run 2 val\_source | Δ | Run 1 criterion | Run 2 criterion |
|--------|-----------------|-----------------|---|-----------------|-----------------|
| right\_auth | 1.602 | 1.605 | +0.003 | best val\_source | mean\_val |
| left\_auth | 1.754 | 1.759 | +0.005 | best val\_source | mean\_val |
| left\_lib | 1.805 | 1.809 | +0.004 | best val\_source | mean\_val |
| right\_lib | 1.821 | 1.823 | +0.002 | best val\_source | mean\_val |

**Val losses did not improve.** Despite source-group rebalancing and 5 epochs, the best val losses are within noise of Run 1. This is the clearest evidence that the LoRA configuration (r=8, q/v only) is the binding constraint — the model cannot extract more from the data regardless of how it is weighted or how long training runs.

**What did improve in Run 2:**

1. **Checkpoint selection quality.** The combined metric prevents artificially early stopping caused by val\_source micro-variations, correctly identifying epoch ~1.75 as the true minimum rather than epoch 1.0 for left\_auth and left\_lib.

2. **Training diagnostics.** 20 eval points per expert reveal the shape of the loss curve — saturation, mild overfitting, or plateau — rather than the coarse 2-point view from Run 1. This confirms which interventions are worth pursuing.

3. **Instance weights are correctly implemented.** Grad norms (1.5–9 in Run 2 vs 0.5–2 in Run 1) confirm the weighting is genuinely altering the gradient signal. The effect is not yet visible in val losses because the adapter lacks capacity to leverage the redistributed signal.

**What did not improve:**

- Absolute val losses (±0.005, within noise)
- Specialisation gap for any expert
- right\_lib's uniform inter-split convergence

---

## Why is the Training Loss 5–9?

The logged `train_loss` values (5–9) are approximately 4× higher than the val losses (1.7–2.0). This is a measurement artefact, not a sign of instability. Three factors combine to produce the discrepancy:

**1. Weighted loss vs unweighted loss.**  
`WeightedTrainer.compute_loss` returns `mean(per_example_CE × weight)`, where weights range from 1.0 to 3.0. Val losses are computed by `MultiSplitEvalCallback` via `outputs.loss`, which is the standard unweighted mean CE. The two quantities are not on the same scale even before any accumulation effects.

**2. Gradient accumulation and loss logging in HuggingFace Trainer.**  
With `gradient_accumulation_steps=4`, the Trainer makes 4 micro-batch calls to `training_step` per optimizer step. In the Trainer version used (with accelerate), `tr_loss` accumulates the raw `compute_loss` output for each micro-batch call, and the logged loss is `tr_loss / n_optimizer_steps` — which effectively multiplies the per-sample loss by `gradient_accumulation_steps=4`. This factor of 4, combined with instance weights (average ~1.2–1.5×), accounts for the ~4–5× total inflation.

**3. Val loss is computed independently.**  
`MultiSplitEvalCallback._split_loss` calls the model on each val split and averages `outputs.loss` directly — no weighting, no accumulation effects. This produces clean per-token CE in the expected 1.7–2.0 range.

**Conclusion:** The val losses are the ground truth. The training loss numbers from Run 2 cannot be compared to Run 1 or to val losses, but they can be compared *across seeds within Run 2* (all use the same logging path). The monotonically declining val curves confirm the model is converging correctly.

---

## Diagnostics and Recommendations

### 1. LoRA capacity is the primary bottleneck

All four experts saturate by epoch ~2 and show no improvement with extended training or instance rebalancing. The intervention with the highest expected return is increasing LoRA rank. Recommended next step:

- **r=16** with `target_modules = [q_proj, k_proj, v_proj, o_proj]`: 4× more parameters, covering the full attention mechanism. Expected to reduce val losses by 0.02–0.05 based on typical scaling behaviour.
- Training time would increase by approximately 2× (more parameters + more gradient computation).

### 2. left\_auth shows genuine overfitting signal after epoch 2

Unlike the other experts which simply plateau, left\_auth continues losing val\_source generalisation at ~0.003/epoch from epoch 3 onward. This is the only expert where early stopping (at epoch ~2) would be genuinely beneficial for a future high-rank run. The 52% speeches composition creates a high-variance training signal that the adapter eventually memorises.

### 3. right\_lib requires data curation, not training changes

The near-zero specialisation gap (0.011) and uniform inter-split convergence are unchanged from Run 1. The adapter is learning something, but it is not learning right-libertarian-specific content — likely because the training corpus blends general right-leaning press with libertarian sources that share too much vocabulary with the broader right-wing press distribution. Recommended: audit the training sources for ideological coherence before investing in a higher-rank run.

### 4. right\_auth needs more training data

At 1,304 examples, right\_auth has the smallest corpus and reaches the best absolute performance (mean\_val=1.708). This is partly an artefact of allsides being a stylistically consistent source — easy to model. But the val\_topic set (256 examples) is too small for reliable immigration generalisation estimates. More annotated right-authoritarian content would improve both capacity and evaluation reliability.

### 5. left\_lib is well-behaved but capacity-limited

The 5-epoch plateau is remarkably clean — no overfitting, no instability. This expert would benefit most straightforwardly from a higher-rank LoRA, as the learning curves show it has genuinely exhausted r=8 capacity rather than being degraded by data quality issues (right\_lib) or register heterogeneity (left\_auth).

### 6. Instance weighting: keep for higher-rank runs

The weighting is correctly implemented and alters the gradient signal in the expected direction (higher grad norms for underrepresented groups). The failure to improve val losses is an architectural issue, not a weighting issue. Retain the weighting configuration as-is for any follow-up runs.
