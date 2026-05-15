# Expert LoRA Training Analysis

**Base model:** `mistralai/Mistral-7B-v0.1`  
**Architecture:** LoRA r=8, α=16, dropout=0.1, targets `q_proj` + `v_proj` (3.4M trainable / 7.2B total params)  
**Training config:** 3 epochs, lr=8e-5 cosine, effective batch size=16 (4 × 4 grad accum), bf16, 3 seeds per expert  
**Checkpoint selection:** best `val_source_loss` across seeds and epochs

---

## SUMMARY
Q1 (right_auth) — "overfitting" is close but slightly imprecise. It's more accurately saturated: val loss doesn't climb dramatically, it just plateaus completely at epoch 1 and then ticks up marginally at epoch 3. The root cause is that the model has extracted everything it can from 1,304 examples given r=8 on q/v only. More epochs won't help; more data or more LoRA capacity would.

Q2/Q3 (left_auth, left_lib) — yes, the selection criterion is cutting training short artificially. The model is still learning (val_indist and val_topic both improve at epoch 2) but val_source nudges up by ~0.005–0.007, which is enough to trigger early best-checkpoint selection. The underlying expert is undertrained, not overfitted. The fix is either a combined checkpoint metric or simply ignoring val_source degradation below some threshold.

Q4 (right_lib) — yes. The training itself is clean and stable, but the data doesn't seem to carry a strong enough right-libertarian signal to produce a distinctive specialist. The expert ends up learning something closer to generic right-leaning prose. That's a data curation problem upstream of training, not a training problem.

So the short version: Q1 needs more data, Q2/Q3 need a better stopping criterion, Q4 needs better data quality/curation.

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

Source group composition of training data:

| Expert | reddit | press | speeches |
|--------|--------|-------|----------|
| right\_auth | 66% | 29% | 4% |
| left\_auth | 16% | 32% | **52%** |
| left\_lib | 40% | 39% | 21% |
| right\_lib | 55% | 21% | 24% |

---

## Loss Trajectories (best seeds)

All losses are token-level cross-entropy (causal LM). Evaluations run once per epoch.

### right\_auth — best seed: 42, best epoch: 2

```
epoch 1:  train=1.555  val_indist=1.854  val_source=1.602  val_topic=1.669
epoch 2:  train=1.580  val_indist=1.854  val_source=1.602  val_topic=1.669
─ seed_456 ran to epoch 3 ─
epoch 3:  train=1.576  val_indist=1.857↑  val_source=1.605↑  val_topic=1.672↑
```

**Best val\_source across all experts (1.602).** Despite the smallest training set (1,304 examples), right\_auth achieves the lowest perplexity on its held-out source (allsides). The train–val\_indist gap is the largest of any expert (1.56 vs 1.85, Δ=0.30), but val curves are flat epoch 1→2 (1.8541 → 1.8541 to four decimal places): the model has **saturated by epoch 1**. The only overfitting evidence comes from seed\_456 at epoch 3, where all val splits tick up by ~0.003.

The large train–val gap likely reflects the 66% reddit training composition: high lexical diversity prevents tight in-distribution memorisation, yet the model has clearly internalised right-authoritarian press style given the low val\_source.

### left\_auth — best seed: 456, best epoch: 1

```
epoch 1:  train=2.002  val_indist=2.019  val_source=1.754  val_topic=1.808
epoch 2:  train=2.058  val_indist=2.014↓  val_source=1.761↑  val_topic=1.805↓
```

The most unusual pattern: train loss (2.00) ≈ val\_indist (2.02), meaning the LoRA barely fits the training data better than the validation set. With 52% speeches + 16% reddit, left\_auth has the most heterogeneous register of any expert; the adapter capacity (r=8, q/v only) is the bottleneck.

Best epoch is 1 because val\_source (uk\_press) degrades at epoch 2 (+0.007) even though val\_indist (−0.005) and val\_topic (−0.003) both improve. This is a **checkpoint selection artefact**: the model continues learning but uk\_press-specific style is slowly diluted by the second pass over heterogeneous training data.

val\_source (1.754) is notably lower than val\_indist (2.019) — a Δ=0.265 gap. uk\_press text is more formal and stylistically predictable than the speeches+reddit training corpus, so the model achieves lower perplexity there despite never having seen it.

### left\_lib — best seed: 123, best epoch: 1

```
epoch 1:  train=1.977  val_indist=2.009  val_source=1.805  val_topic=1.842
epoch 2:  train=1.963  val_indist=1.999↓  val_source=1.810↑  val_topic=1.839↓
```

Same divergence pattern as left\_auth: val\_source worsens at epoch 2 (+0.005) while val\_indist (−0.010) and val\_topic (−0.003) both improve. Best checkpoint is epoch 1 solely because of the val\_source selection criterion.

The best-provisioned expert by data volume (5,118 train, 320 steps/epoch), yet produces the second-worst val\_source (1.805) — reflecting a genuinely harder language modelling target. The left-lib register blends reddit, press, and speeches in a more ambiguous political voice than right\_auth.

val\_topic is enormous (13,555 examples — 2.6× the training set), meaning immigration-topic content is exceptionally rich for this quadrant. val\_topic (1.842) > val\_source (1.805) indicates immigration is harder for this expert than the ire\_press held-out source, consistent with it being ideologically contested ground.

### right\_lib — best seed: 456, best epoch: 1

```
epoch 1:  train=1.860  val_indist=1.834  val_source=1.821  val_topic=1.809
epoch 2:  train=1.837  val_indist=1.829↓  val_source=1.822↑  val_topic=1.801↓
```

**Most distinctive pattern: all four losses cluster within ~0.05 of each other** (1.80–1.86). Train loss is actually *higher* than val\_indist (1.860 vs 1.834) — the training data is not easier than the validation data, and there is effectively zero in-distribution advantage.

This inter-split uniformity is a double-edged result: it signals excellent generalisation, but also that the expert has learned very little that is exclusive to right-lib sources. The specialisation gap (val\_indist − val\_source = 0.013) is the smallest of any expert, compared to 0.25–0.30 for others.

val\_topic (1.809) is the lowest of the three val splits — immigration is *easier* for this expert than in-distribution or held-out source content, the opposite of the left\_lib pattern.

---

## Cross-Expert Summary

| Expert | Best val\_source ↓ | Specialisation gap† | Seed spread | Best epoch | Train–val\_indist‡ |
|--------|-------------------|---------------------|-------------|------------|--------------------|
| right\_auth | **1.602** | 0.25 | ±0.002 | 2 | +0.30 (memorisation) |
| left\_auth | 1.754 | 0.27 | ±0.001 | 1 | ≈0 (under-capacity) |
| left\_lib | 1.805 | 0.20 | ±0.001 | 1 | −0.03 (healthy) |
| right\_lib | 1.821 | **0.01** | <0.001 | 1 | ≈0 inverted |

† val\_indist − val\_source at best epoch. Larger = stronger source-specific specialisation.  
‡ train\_loss − val\_indist\_loss at best epoch. Positive = training data harder; negative = normal overfitting signal.

All experts show excellent seed stability (spread ≤ 0.002), confirming that results are robust and not seed-sensitive.

---

## Diagnostics and Recommendations

### 1. Checkpoint selection metric is suppressing left-side training

For both left\_auth and left\_lib, val\_indist improves at epoch 2 while val\_source worsens. Because the policy selects on val\_source alone, the saved checkpoint is underfitted on in-distribution data. Consider using a combined criterion — e.g. `0.5 × val_source + 0.5 × val_indist` — or saving the epoch-2 checkpoint for these two experts.

### 2. right\_auth is architecturally bottlenecked, not data-bottlenecked

The 0.30 train–val\_indist gap cannot be closed with more epochs. The model has saturated: epoch 3 (seed\_456) shows val degradation across all splits. A larger LoRA rank (r=16) or additional target modules (`k_proj`, `o_proj`) would provide more capacity to fit the highly diverse, Reddit-heavy training distribution without hurting val performance.

### 3. right\_lib shows weak quadrant specialisation

The near-zero specialisation gap (0.013) means the adapter has learned very little that is exclusive to right-lib sources, compared to the 0.20–0.27 gaps seen in the other experts. This could manifest as poor routing signal in the MoE or weak stylistic distinctiveness in generation. It is worth inspecting whether the training corpus has sufficient right-libertarian ideological signal, or whether it is blending into generic right-leaning news prose.

### 4. Potential for longer training (left\_auth, left\_lib)

Neither left\_auth nor left\_lib shows any overfitting signal. For both, val\_indist and val\_topic continue improving at epoch 2. With a better checkpoint criterion (see point 1), 2–3 additional epochs at a reduced learning rate could meaningfully improve in-distribution and topic generalisation without degrading source generalisation.

### 5. right\_auth data adequacy

At 1,304 training examples, right\_auth has the smallest corpus. More data (not more epochs) is the right intervention for this expert. Its val\_topic set is also the smallest (256 examples), offering limited insight into topic generalisation.

### 6. val\_topic size imbalance

val\_topic sizes range from 256 (right\_auth) to 13,555 (left\_lib). This imbalance means that topic generalisation evaluations are far less reliable for right\_auth than for the other experts. Downstream MoE evaluation on immigration-topic data should account for this.
