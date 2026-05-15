# src/07_train_experts.py

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless cluster nodes
import matplotlib.pyplot as plt
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# === CONSTANTS ===

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
CONFIG_PATH      = PROJECT_ROOT / "config" / "config.yaml"
VALID_QUADRANTS        = ["right_auth", "left_auth", "left_lib", "right_lib"]
TRAIN_SPLIT            = "train"
VAL_INDIST_SPLIT       = "val_indist"
VAL_SOURCE_SPLIT       = "val_source"
VAL_TOPIC_SPLIT        = "val_topic"
REQUIRED_CHUNK_FIELDS  = {"text", "source", "topic_label", "document_id", "chunk_id"}

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATACLASSES ===

@dataclass
class LoraParams:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: str
    task_type: str


@dataclass
class TrainParams:
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    bf16: bool
    tf32: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int


@dataclass
class EvalParams:
    evals_per_epoch: int
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool


@dataclass
class WeightingParams:
    enabled: bool
    max_weight_cap: float
    source_groups: dict[str, list[str]]  # group_name -> [source_name, ...]


@dataclass
class GenerationParams:
    enabled: bool
    max_new_tokens: int
    temperature: float
    do_sample: bool
    fixed_prompts: list[str]


@dataclass
class ExpertConfig:
    base_model: str
    precision: str
    lora: LoraParams
    training: TrainParams
    evaluation: EvalParams
    weighting: WeightingParams
    generation: GenerationParams
    logging_steps: int
    report_to: str
    seeds: list[int]
    train_validate_dir: Path
    output_dir: Path


# === CONFIG LOADING ===

def load_config(path: Path) -> ExpertConfig:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if "train_experts" not in raw:
        raise ValueError(f"config missing 'train_experts' block: {path}")
    cfg = raw["train_experts"]

    required = {"base_model", "precision", "lora", "training", "evaluation",
                "weighting", "generation", "logging", "seeds", "paths"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"train_experts config missing keys: {missing}")

    lora_raw = cfg["lora"]
    if not isinstance(lora_raw.get("r"), int) or lora_raw["r"] <= 0:
        raise ValueError("train_experts.lora.r must be a positive integer")

    lr = cfg["training"].get("learning_rate")
    if not (0 < lr < 1):
        raise ValueError(f"train_experts.training.learning_rate={lr} must be in (0, 1)")

    seeds = cfg.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds):
        raise ValueError("train_experts.seeds must be a non-empty list of ints")

    paths_raw = cfg["paths"]
    train_validate_dir = PROJECT_ROOT / paths_raw["train_validate_dir"]
    if not train_validate_dir.exists():
        raise ValueError(f"train_validate_dir does not exist: {train_validate_dir}")
    output_dir = PROJECT_ROOT / paths_raw["output_dir"]

    train_raw   = cfg["training"]
    eval_raw    = cfg["evaluation"]
    weight_raw  = cfg["weighting"]
    gen_raw     = cfg["generation"]
    log_raw     = cfg["logging"]

    # build source -> group reverse map for fast lookup
    source_groups: dict[str, list[str]] = weight_raw.get("source_groups", {})

    return ExpertConfig(
        base_model=cfg["base_model"],
        precision=cfg["precision"],
        lora=LoraParams(
            r=lora_raw["r"],
            lora_alpha=lora_raw["lora_alpha"],
            lora_dropout=lora_raw["lora_dropout"],
            target_modules=lora_raw["target_modules"],
            bias=lora_raw["bias"],
            task_type=lora_raw["task_type"],
        ),
        training=TrainParams(
            num_train_epochs=train_raw["num_train_epochs"],
            per_device_train_batch_size=train_raw["per_device_train_batch_size"],
            gradient_accumulation_steps=train_raw["gradient_accumulation_steps"],
            learning_rate=train_raw["learning_rate"],
            lr_scheduler_type=train_raw["lr_scheduler_type"],
            warmup_ratio=train_raw["warmup_ratio"],
            weight_decay=train_raw["weight_decay"],
            bf16=train_raw["bf16"],
            tf32=train_raw["tf32"],
            gradient_checkpointing=train_raw["gradient_checkpointing"],
            dataloader_num_workers=train_raw["dataloader_num_workers"],
        ),
        evaluation=EvalParams(
            evals_per_epoch=eval_raw["evals_per_epoch"],
            save_total_limit=eval_raw["save_total_limit"],
            metric_for_best_model=eval_raw["metric_for_best_model"],
            greater_is_better=eval_raw["greater_is_better"],
        ),
        weighting=WeightingParams(
            enabled=weight_raw["enabled"],
            max_weight_cap=float(weight_raw["max_weight_cap"]),
            source_groups=source_groups,
        ),
        generation=GenerationParams(
            enabled=gen_raw["enabled"],
            max_new_tokens=gen_raw["max_new_tokens"],
            temperature=gen_raw["temperature"],
            do_sample=gen_raw["do_sample"],
            fixed_prompts=gen_raw["fixed_prompts"],
        ),
        logging_steps=log_raw["logging_steps"],
        report_to=log_raw["report_to"],
        seeds=seeds,
        train_validate_dir=train_validate_dir,
        output_dir=output_dir,
    )


# === DATA LOADING AND TOKENIZATION ===

def load_split(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{lineno}") from exc
            missing = REQUIRED_CHUNK_FIELDS - set(record.keys())
            if missing:
                raise ValueError(f"{path}:{lineno} missing required fields: {sorted(missing)}")
            records.append(record)
    log.info("loaded %d records from %s", len(records), path.name)
    return records


def compute_source_group_weights(
    chunks: list[dict],
    source_groups: dict[str, list[str]],
    max_cap: float = 3.0,
) -> list[float]:
    """
    Inverse-frequency weights by source group, capped and normalised to mean=1.

    source_groups maps group_name -> list[source_name]. Sources not present in
    the mapping are assigned to an implicit "other" group.
    """
    # build reverse map: source_name -> group_name
    src_to_group: dict[str, str] = {}
    for group, sources in source_groups.items():
        for src in sources:
            src_to_group[src] = group

    groups = [src_to_group.get(c["source"], "other") for c in chunks]
    counts = Counter(groups)
    n_groups = len(counts)
    n_total  = len(groups)

    # raw inverse-frequency: w_g = total / (n_groups * count_g)
    raw = {g: n_total / (n_groups * c) for g, c in counts.items()}
    capped = {g: min(w, max_cap) for g, w in raw.items()}

    # log the resulting distribution
    for g, w in sorted(capped.items()):
        log.info("  source_group %-12s  n=%5d  raw_w=%.3f  capped_w=%.3f",
                 g, counts[g], raw[g], w)

    return [capped[g] for g in groups]


def tokenize_chunks(
    chunks: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 700,
    weights: list[float] | None = None,
) -> Dataset:
    """
    Tokenize chunks for causal LM. Optionally stores a per-example weight column
    used by WeightedTrainer to apply source-group loss reweighting.
    """
    token_lengths: list[int] = []
    n_truncated = 0
    rows: list[dict] = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text.strip():
            raise ValueError(f"chunk {i} has missing or empty 'text' field")

        enc = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        seq_len = len(enc["input_ids"])
        if seq_len == max_length:
            n_truncated += 1
        token_lengths.append(seq_len)

        row = dict(enc)
        if weights is not None:
            row["weight"] = weights[i]
        rows.append(row)

    mean_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    log.info(
        "tokenized %d chunks — min=%d  mean=%.1f  max=%d tokens — %d truncated",
        len(rows), min(token_lengths), mean_len, max(token_lengths), n_truncated,
    )
    return Dataset.from_list(rows)


# === WEIGHTED TRAINING UTILITIES ===

class WeightedDataCollator(DataCollatorForLanguageModeling):
    """
    Wraps DataCollatorForLanguageModeling to extract per-example weights before
    padding (the tokenizer's pad() call does not know how to handle scalar floats)
    and re-attach them to the batch as a float32 tensor.
    """

    def __call__(self, features: list[dict]) -> dict:
        weights = [float(f.pop("weight", 1.0)) for f in features]
        batch = super().__call__(features)
        batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedTrainer(Trainer):
    """
    Trainer subclass that applies per-example loss weights.

    Computes the mean token-level cross-entropy per example, multiplies by
    the example's weight, then averages over the batch. Falls back to the
    standard Trainer.compute_loss if no weight column is present (e.g. during
    evaluation or dry-run).
    """

    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        weights = inputs.pop("weight", None)
        labels  = inputs.get("labels")

        outputs = model(**inputs)

        if weights is None or labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[..., 1:].contiguous()        # (B, T-1)

        loss_fct    = CrossEntropyLoss(reduction="none")
        token_loss  = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())                        # (B, T-1); -100 positions → 0

        mask = (shift_labels != -100).float()              # (B, T-1)
        per_example = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B,)
        loss = (per_example * weights.to(per_example.device)).mean()

        return (loss, outputs) if return_outputs else loss


# === LORA SETUP ===

_PRECISION_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def build_lora_model(
    base_model_name: str,
    lora_params: LoraParams,
    precision: str,
) -> tuple[PeftModel, AutoTokenizer]:
    if precision not in _PRECISION_MAP:
        raise ValueError(f"unsupported precision '{precision}'; choose from {list(_PRECISION_MAP)}")

    log.info("loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("loading base model: %s  precision=%s", base_model_name, precision)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=_PRECISION_MAP[precision],
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.requires_grad_(False)

    lora_config = LoraConfig(
        r=lora_params.r,
        lora_alpha=lora_params.lora_alpha,
        lora_dropout=lora_params.lora_dropout,
        target_modules=lora_params.target_modules,
        bias=lora_params.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            raise AssertionError(f"non-LoRA parameter has requires_grad=True: {name}")

    return model, tokenizer


def _input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


# === CALLBACKS ===

class MultiSplitEvalCallback(TrainerCallback):
    """
    Evaluates on val_indist (full), val_source (subsampled), and val_topic (subsampled)
    at every eval step. Logs all three losses plus their mean to train_log.jsonl.
    Saves the adapter weights directly whenever the mean val loss improves, so the
    best checkpoint is always available regardless of trainer save_total_limit.
    """

    _SUBSAMPLE_SIZE  = 500
    _SUBSAMPLE_SEED  = 42
    _EVAL_BATCH_SIZE = 8

    def __init__(
        self,
        val_indist_dataset: Dataset,
        val_source_dataset: Dataset,
        val_topic_dataset: Dataset,
        tokenizer: AutoTokenizer,
        log_path: Path,
        best_ckpt_path: Path,
        precision: str,
    ) -> None:
        self.val_indist_dataset = val_indist_dataset
        self.tokenizer          = tokenizer
        self.log_path           = log_path
        self.best_ckpt_path     = best_ckpt_path
        self.eval_dtype         = _PRECISION_MAP[precision]
        self.best_mean_val_loss = float("inf")
        self.eval_logs: list[dict] = []
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        rng = random.Random(self._SUBSAMPLE_SEED)
        self.val_source_sub = val_source_dataset.select(
            rng.sample(range(len(val_source_dataset)),
                       min(self._SUBSAMPLE_SIZE, len(val_source_dataset)))
        )
        self.val_topic_sub = val_topic_dataset.select(
            rng.sample(range(len(val_topic_dataset)),
                       min(self._SUBSAMPLE_SIZE, len(val_topic_dataset)))
        )
        log.info(
            "val_source subsample=%d/%d  val_topic subsample=%d/%d",
            len(self.val_source_sub), len(val_source_dataset),
            len(self.val_topic_sub),  len(val_topic_dataset),
        )

    def _split_loss(self, model: Any, dataset: Dataset) -> float:
        total_loss   = 0.0
        total_tokens = 0
        for i in range(0, len(dataset), self._EVAL_BATCH_SIZE):
            batch_slice = dataset[i : i + self._EVAL_BATCH_SIZE]
            n = len(next(iter(batch_slice.values())))
            batch_list = [
                {k: (v[j].tolist() if hasattr(v[j], "tolist") else v[j])
                 for k, v in batch_slice.items()}
                for j in range(n)
            ]
            batch  = self.data_collator(batch_list)
            device = _input_device(model)
            batch  = {k: v.to(device) for k, v in batch.items()}
            use_cuda = device.type == "cuda"
            with torch.no_grad(), torch.autocast(device.type, dtype=self.eval_dtype, enabled=use_cuda):
                outputs = model(**batch)
            n_tok = (batch["labels"] != -100).sum().item()
            total_loss   += outputs.loss.item() * n_tok
            total_tokens += n_tok
        return total_loss / total_tokens if total_tokens > 0 else float("nan")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        model.eval()
        val_indist = self._split_loss(model, self.val_indist_dataset)
        val_source = self._split_loss(model, self.val_source_sub)
        val_topic  = self._split_loss(model, self.val_topic_sub)
        model.train()

        mean_val = (val_indist + val_source + val_topic) / 3.0

        log.info(
            "step %d (epoch %.2f) — val_indist=%.4f  val_source=%.4f  val_topic=%.4f  mean=%.4f",
            state.global_step, state.epoch, val_indist, val_source, val_topic, mean_val,
        )

        if metrics is not None:
            metrics["eval_val_indist_loss"] = val_indist
            metrics["eval_val_source_loss"] = val_source
            metrics["eval_val_topic_loss"]  = val_topic
            metrics["eval_mean_val_loss"]   = mean_val

        train_entries = [e for e in state.log_history if "loss" in e and "eval_loss" not in e]
        last_train = train_entries[-1] if train_entries else {}

        self.eval_logs.append({
            "epoch":           state.epoch,
            "global_step":     state.global_step,
            "train_loss":      last_train.get("loss", float("nan")),
            "learning_rate":   last_train.get("learning_rate", float("nan")),
            "grad_norm":       last_train.get("grad_norm", float("nan")),
            "val_indist_loss": val_indist,
            "val_source_loss": val_source,
            "val_topic_loss":  val_topic,
            "mean_val_loss":   mean_val,
        })

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as fh:
            for record in self.eval_logs:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # save adapter directly when we see a new best — independent of trainer checkpoints
        if mean_val < self.best_mean_val_loss:
            self.best_mean_val_loss = mean_val
            self.best_ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(self.best_ckpt_path))
            self.tokenizer.save_pretrained(str(self.best_ckpt_path))
            log.info("new best (mean_val=%.4f) saved to %s", mean_val, self.best_ckpt_path.name)


class LearningCurvePlotCallback(TrainerCallback):
    """
    Maintains running loss histories and re-renders a single PNG after every
    log step (train loss) and every eval (val losses). Overwrites the same file
    each time — disk cost is one PNG (~150 KB) for the entire training run.
    """

    def __init__(self, save_path: Path, quadrant: str, seed: int) -> None:
        self.save_path    = save_path
        self.title        = f"{quadrant} — seed {seed}"
        self.train_steps:  list[int]   = []
        self.train_losses: list[float] = []
        self.eval_steps:   list[int]   = []
        self.val_indist:   list[float] = []
        self.val_source:   list[float] = []
        self.val_topic:    list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(logs["loss"])
            self._render()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_val_source_loss" in metrics:
            self.eval_steps.append(state.global_step)
            self.val_indist.append(metrics.get("eval_val_indist_loss", float("nan")))
            self.val_source.append(metrics.get("eval_val_source_loss", float("nan")))
            self.val_topic.append(metrics.get("eval_val_topic_loss",  float("nan")))
            self._render()

    def _render(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))

        if self.train_steps:
            ax.plot(self.train_steps, self.train_losses,
                    color="#888", lw=0.9, alpha=0.6, label="train loss")

        if self.eval_steps:
            ax.plot(self.eval_steps, self.val_indist,
                    "o--", color="#e07b39", lw=1.6, ms=5, label="val_indist")
            ax.plot(self.eval_steps, self.val_source,
                    "s-",  color="#2a6ebb", lw=1.6, ms=5, label="val_source")
            ax.plot(self.eval_steps, self.val_topic,
                    "^:",  color="#3aa86a", lw=1.6, ms=5, label="val_topic")

        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy loss")
        ax.set_title(self.title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)


class GenerationCallback(TrainerCallback):
    """Generates text from fixed prompts once per epoch."""

    def __init__(
        self,
        prompts: list[str],
        generation_params: GenerationParams,
        tokenizer: AutoTokenizer,
        quadrant: str,
        log_path: Path,
    ) -> None:
        self.prompts           = prompts
        self.generation_params = generation_params
        self.tokenizer         = tokenizer
        self.quadrant          = quadrant
        self.log_path          = log_path
        self._last_epoch       = -1

    def on_evaluate(self, args, state, control, **kwargs):
        # only run once per epoch (on_evaluate fires at eval_steps, not epoch_end)
        epoch = math.floor(state.epoch)
        if epoch == self._last_epoch or not self.generation_params.enabled:
            return
        self._last_epoch = epoch

        model = kwargs.get("model")
        if model is None:
            return

        model.eval()
        entries: list[dict] = []
        for prompt in self.prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            inputs = {k: v.to(_input_device(model)) for k, v in inputs.items()}
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.generation_params.max_new_tokens,
                "do_sample": self.generation_params.do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if self.generation_params.do_sample:
                gen_kwargs["temperature"] = self.generation_params.temperature
            else:
                gen_kwargs["temperature"] = None
                gen_kwargs["top_p"]       = None
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            prompt_len     = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True
            )
            entries.append({
                "epoch": epoch,
                "global_step": state.global_step,
                "quadrant": self.quadrant,
                "prompt": prompt,
                "generated_text": generated_text,
            })
        model.train()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        log.info("generation logged for epoch %d — %d prompts", epoch, len(self.prompts))


# === TRAINING ===

def run_training(
    quadrant: str,
    seed: int,
    config: ExpertConfig,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    t_start = time.time()
    set_seed(seed)

    for stale_ckpt in output_dir.glob("checkpoint-*"):
        shutil.rmtree(stale_ckpt)
        log.info("removed stale checkpoint %s", stale_ckpt.name)
    for stale in [output_dir / "train_log.jsonl", output_dir / "generation_log.jsonl"]:
        if stale.exists():
            stale.unlink()

    model, tokenizer = build_lora_model(config.base_model, config.lora, config.precision)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())

    data_dir     = config.train_validate_dir / quadrant
    train_chunks  = load_split(data_dir / "train.jsonl")
    indist_chunks = load_split(data_dir / "val_indist.jsonl")
    source_chunks = load_split(data_dir / "val_source.jsonl")
    topic_chunks  = load_split(data_dir / "val_topic.jsonl")

    # compute source-group weights for training examples only
    train_weights: list[float] | None = None
    if config.weighting.enabled:
        log.info("computing source-group weights for %s train set:", quadrant)
        train_weights = compute_source_group_weights(
            train_chunks,
            config.weighting.source_groups,
            max_cap=config.weighting.max_weight_cap,
        )

    log.info("tokenizing splits for quadrant=%s", quadrant)
    train_dataset      = tokenize_chunks(train_chunks,  tokenizer, weights=train_weights)
    val_indist_dataset = tokenize_chunks(indist_chunks, tokenizer)
    val_source_dataset = tokenize_chunks(source_chunks, tokenizer)
    val_topic_dataset  = tokenize_chunks(topic_chunks,  tokenizer)

    log.info(
        "dataset sizes — train=%d  val_indist=%d  val_source=%d  val_topic=%d",
        len(train_dataset), len(val_indist_dataset),
        len(val_source_dataset), len(val_topic_dataset),
    )

    # compute eval_steps so we get exactly evals_per_epoch checkpoints per epoch
    effective_batch    = (config.training.per_device_train_batch_size *
                          config.training.gradient_accumulation_steps)
    steps_per_epoch    = math.ceil(len(train_dataset) / effective_batch)
    eval_steps         = max(1, steps_per_epoch // config.evaluation.evals_per_epoch)
    log.info("steps_per_epoch=%d  eval_steps=%d", steps_per_epoch, eval_steps)

    data_collator = (
        WeightedDataCollator(tokenizer=tokenizer, mlm=False)
        if config.weighting.enabled
        else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    best_ckpt_path = output_dir / "best"
    multi_split_cb = MultiSplitEvalCallback(
        val_indist_dataset=val_indist_dataset,
        val_source_dataset=val_source_dataset,
        val_topic_dataset=val_topic_dataset,
        tokenizer=tokenizer,
        log_path=output_dir / "train_log.jsonl",
        best_ckpt_path=best_ckpt_path,
        precision=config.precision,
    )
    plot_cb = LearningCurvePlotCallback(
        save_path=output_dir / "learning_curve.png",
        quadrant=quadrant,
        seed=seed,
    )
    generation_cb = GenerationCallback(
        prompts=config.generation.fixed_prompts,
        generation_params=config.generation,
        tokenizer=tokenizer,
        quadrant=quadrant,
        log_path=output_dir / "generation_log.jsonl",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        dataloader_num_workers=config.training.dataloader_num_workers,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=config.evaluation.save_total_limit,
        load_best_model_at_end=False,
        metric_for_best_model=config.evaluation.metric_for_best_model,
        greater_is_better=config.evaluation.greater_is_better,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        seed=seed,
        run_name=f"{quadrant}_r{config.lora.r}_lr{config.training.learning_rate}_seed{seed}",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    TrainerClass = WeightedTrainer if config.weighting.enabled else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_indist_dataset,
        data_collator=data_collator,
        callbacks=[multi_split_cb, plot_cb, generation_cb],
    )

    if dry_run:
        log.info("=== dry run — skipping trainer.train() ===")
        log.info(
            "train=%d  val_indist=%d  val_source_sub=%d  val_topic_sub=%d  eval_steps=%d",
            len(train_dataset), len(val_indist_dataset),
            len(multi_split_cb.val_source_sub), len(multi_split_cb.val_topic_sub),
            eval_steps,
        )
        assert len(multi_split_cb.val_source_sub) > 0
        assert len(multi_split_cb.val_topic_sub)  > 0

        example_batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
        batch = data_collator(example_batch)
        batch = {k: v.to(_input_device(model)) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(**{k: v for k, v in batch.items() if k != "weight"})
        loss_val = outputs.loss.item()
        assert not math.isnan(loss_val) and loss_val > 0, f"dry-run forward pass loss={loss_val}"
        log.info("forward pass ok — loss=%.4f  weighted_training=%s", loss_val, config.weighting.enabled)
        return {"quadrant": quadrant, "seed": seed, "dry_run": True, "forward_pass_loss": loss_val}

    log.info("starting training: quadrant=%s  seed=%d", quadrant, seed)
    trainer.train()

    # best adapter was saved incrementally by MultiSplitEvalCallback — just record metadata
    best_record = (
        min(multi_split_cb.eval_logs, key=lambda r: r["mean_val_loss"])
        if multi_split_cb.eval_logs else {}
    )

    train_entries    = [e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    final_train_loss = train_entries[-1]["loss"] if train_entries else float("nan")
    total_time       = time.time() - t_start

    summary = {
        "quadrant":            quadrant,
        "seed":                seed,
        "base_model":          config.base_model,
        "trainable_params":    trainable_params,
        "total_params":        total_params,
        "best_step":           best_record.get("global_step", -1),
        "best_epoch":          best_record.get("epoch", -1),
        "best_mean_val_loss":  best_record.get("mean_val_loss", float("nan")),
        "best_val_source_loss":best_record.get("val_source_loss", float("nan")),
        "best_val_indist_loss":best_record.get("val_indist_loss", float("nan")),
        "best_val_topic_loss": best_record.get("val_topic_loss", float("nan")),
        "final_train_loss":    final_train_loss,
        "total_steps":         trainer.state.global_step,
        "total_time_seconds":  round(total_time, 1),
        "best_checkpoint":     str(best_ckpt_path),
        "best_metric":         "mean_val_loss",
        "weighted_training":   config.weighting.enabled,
        "config": {
            "lora":     asdict(config.lora),
            "training": asdict(config.training),
        },
    }
    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("training summary written to %s", summary_path)
    return summary


# === MAIN ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="train one LoRA expert adapter for a given political quadrant"
    )
    p.add_argument("--quadrant", required=True, choices=VALID_QUADRANTS)
    p.add_argument("--config",   type=Path, default=CONFIG_PATH)
    p.add_argument("--seed",     type=int, default=None,
                   help="single seed — overrides config.seeds")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    t_total = time.time()
    args    = parse_args()
    config  = load_config(args.config)
    log.info("config loaded from %s", args.config)

    output_root = args.output_dir if args.output_dir is not None else config.output_dir
    quadrant    = args.quadrant

    data_dir = config.train_validate_dir / quadrant
    for split_name in [TRAIN_SPLIT, VAL_INDIST_SPLIT, VAL_SOURCE_SPLIT, VAL_TOPIC_SPLIT]:
        split_path = data_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split file: {split_path}")
        with split_path.open(encoding="utf-8") as fh:
            count = sum(1 for line in fh if line.strip())
        log.info("  %-14s  %6d chunks", split_name, count)

    seeds = [args.seed] if args.seed is not None else config.seeds
    log.info("quadrant=%s  seeds=%s", quadrant, seeds)

    summaries: list[dict] = []
    for seed in seeds:
        run_dir = output_root / quadrant / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info("starting quadrant=%s  seed=%d → %s", quadrant, seed, run_dir)

        summary = run_training(
            quadrant=quadrant,
            seed=seed,
            config=config,
            output_dir=run_dir,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        log.info(
            "finished seed=%d — best_mean_val=%.4f  time=%.0fs",
            seed,
            summary.get("best_mean_val_loss", float("nan")),
            summary.get("total_time_seconds", 0.0),
        )

    if len(seeds) > 1 and not args.dry_run:
        best = min(summaries, key=lambda s: s.get("best_mean_val_loss", float("inf")))
        log.info(
            "best seed for %s: seed=%d  mean_val_loss=%.4f",
            quadrant, best["seed"], best["best_mean_val_loss"],
        )
        best_seed_summary = {
            "quadrant":            quadrant,
            "seeds_run":           seeds,
            "best_seed":           best["seed"],
            "best_mean_val_loss":  best["best_mean_val_loss"],
            "best_val_source_loss":best["best_val_source_loss"],
            "all_summaries":       summaries,
        }
        summary_path = output_root / quadrant / "best_seed_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(best_seed_summary, fh, indent=2, ensure_ascii=False)
        log.info("best seed summary written to %s", summary_path)

    log.info("total elapsed: %.0fs", time.time() - t_total)


if __name__ == "__main__":
    main()
