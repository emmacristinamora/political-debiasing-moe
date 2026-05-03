# src/07_train_experts.py


# === IMPORTS ===

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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
    eval_strategy: str
    save_strategy: str
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    early_stopping_patience: int


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
    generation: GenerationParams
    logging_steps: int
    report_to: str
    seeds: list[int]
    train_validate_dir: Path
    output_dir: Path


# === CONFIG LOADING ===

def load_config(path: Path) -> ExpertConfig:
    """
    Load and validate the train_experts block from config.yaml.

    Args:
        path: path to config.yaml

    Returns:
        Validated ExpertConfig dataclass.

    Logic:
        Reads YAML, extracts train_experts block, validates all required keys
        and field constraints, then constructs and returns ExpertConfig.
    """
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if "train_experts" not in raw:
        raise ValueError(f"config missing 'train_experts' block: {path}")
    cfg = raw["train_experts"]

    required = {"base_model", "precision", "lora", "training", "evaluation",
                "generation", "logging", "seeds", "paths"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"train_experts config missing keys: {missing}")

    if not isinstance(cfg["base_model"], str) or not cfg["base_model"].strip():
        raise ValueError("train_experts.base_model must be a non-empty string")

    lora_raw = cfg["lora"]
    if not isinstance(lora_raw.get("r"), int) or lora_raw["r"] <= 0:
        raise ValueError("train_experts.lora.r must be a positive integer")
    if not isinstance(lora_raw.get("target_modules"), list) or not lora_raw["target_modules"]:
        raise ValueError("train_experts.lora.target_modules must be a non-empty list")

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

    train_raw = cfg["training"]
    eval_raw  = cfg["evaluation"]
    gen_raw   = cfg["generation"]
    log_raw   = cfg["logging"]

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
            eval_strategy=eval_raw["eval_strategy"],
            save_strategy=eval_raw["save_strategy"],
            save_total_limit=eval_raw["save_total_limit"],
            load_best_model_at_end=eval_raw["load_best_model_at_end"],
            metric_for_best_model=eval_raw["metric_for_best_model"],
            greater_is_better=eval_raw["greater_is_better"],
            early_stopping_patience=eval_raw["early_stopping_patience"],
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


def tokenize_chunks(
    chunks: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 700,
) -> Dataset:
    """
    Tokenize a list of chunk dicts for causal LM training.

    Args:
        chunks:     list of dicts, each with a non-empty 'text' field
        tokenizer:  HuggingFace tokenizer
        max_length: truncation cap in tokens

    Returns:
        HuggingFace Dataset with input_ids, attention_mask, and labels columns.

    Logic:
        Tokenizes each chunk's text with truncation. Sets labels equal to
        input_ids so the model learns the full text distribution (no prompt
        masking). Logs token length statistics and truncation count.
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
        enc["labels"] = enc["input_ids"].copy()
        rows.append(enc)

    mean_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    log.info(
        "tokenized %d chunks — min=%d  mean=%.1f  max=%d tokens — %d truncated",
        len(rows), min(token_lengths), mean_len, max(token_lengths), n_truncated,
    )
    return Dataset.from_list(rows)


def build_data_collator(tokenizer: AutoTokenizer) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


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
    """
    Load base model and wrap with LoRA adapter.

    Args:
        base_model_name: HuggingFace model id
        lora_params:     LoRA hyperparameters
        precision:       one of bfloat16 / float16 / float32

    Returns:
        (PeftModel, AutoTokenizer) with only LoRA parameters trainable.

    Logic:
        Loads tokenizer, aliases pad→eos for Mistral (no pad token in vocab),
        loads base model frozen at the requested dtype, wraps with LoraConfig,
        then asserts that no non-LoRA parameter has requires_grad=True.
    """
    if precision not in _PRECISION_MAP:
        raise ValueError(f"unsupported precision '{precision}'; choose from {list(_PRECISION_MAP)}")

    log.info("loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        # mistral has no pad token; alias to eos without resizing embeddings
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("loading base model: %s  precision=%s", base_model_name, precision)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=_PRECISION_MAP[precision],
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


# === CALLBACKS ===

def _input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

class MultiSplitEvalCallback(TrainerCallback):
    """
    Evaluates model on three val splits after each epoch and handles early stopping.

    Args:
        val_indist_dataset:      full in-distribution val set
        val_source_dataset:      held-out source val set (subsampled at init)
        val_topic_dataset:       held-out topic val set (subsampled at init)
        tokenizer:               tokenizer for the data collator
        log_path:                path for train_log.jsonl (one line per epoch)
        early_stopping_patience: stop if val_source_loss doesn't improve for N epochs
        early_stopping_threshold: minimum improvement to count as progress

    Logic:
        Hooks on_evaluate (not on_epoch_end) so metrics are injected into the
        Trainer's metrics dict before any downstream callback reads them. Early
        stopping is implemented here directly because EarlyStoppingCallback cannot
        read metrics injected by a callback. val_source and val_topic losses are
        computed on a fixed 500-chunk subsample seeded once at __init__ time so
        losses are comparable across epochs.
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
        early_stopping_patience: int,
        precision: str,
        early_stopping_threshold: float = 0.001,
    ) -> None:
        self.val_indist_dataset       = val_indist_dataset
        self.tokenizer                = tokenizer
        self.log_path                 = log_path
        self.early_stopping_patience  = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.eval_dtype               = _PRECISION_MAP[precision]
        self.best_val_source_loss     = float("inf")
        self.bad_epochs               = 0
        self.epoch_logs: list[dict]   = []
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        rng = random.Random(self._SUBSAMPLE_SEED)
        src_idx = rng.sample(
            range(len(val_source_dataset)),
            min(self._SUBSAMPLE_SIZE, len(val_source_dataset)),
        )
        top_idx = rng.sample(
            range(len(val_topic_dataset)),
            min(self._SUBSAMPLE_SIZE, len(val_topic_dataset)),
        )
        self.val_source_sub = val_source_dataset.select(src_idx)
        self.val_topic_sub  = val_topic_dataset.select(top_idx)
        log.info(
            "val_source subsample=%d (of %d)  val_topic subsample=%d (of %d)",
            len(self.val_source_sub), len(val_source_dataset),
            len(self.val_topic_sub), len(val_topic_dataset),
        )

    def _compute_split_loss(self, model: Any, dataset: Dataset) -> float:
        total_loss   = 0.0
        total_tokens = 0

        for i in range(0, len(dataset), self._EVAL_BATCH_SIZE):
            batch_slice = dataset[i : i + self._EVAL_BATCH_SIZE]
            n_items = len(next(iter(batch_slice.values())))
            # dataset slices may return tensors or plain lists; normalise to list
            batch_list = [
                {k: (v[j].tolist() if hasattr(v[j], "tolist") else v[j])
                 for k, v in batch_slice.items()}
                for j in range(n_items)
            ]
            batch = self.data_collator(batch_list)
            device = _input_device(model)
            batch  = {k: v.to(device) for k, v in batch.items()}

            use_cuda = device.type == "cuda"
            with torch.no_grad(), torch.autocast(device.type, dtype=self.eval_dtype, enabled=use_cuda):
                outputs = model(**batch)

            n_tokens = (batch["labels"] != -100).sum().item()
            total_loss   += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        return total_loss / total_tokens if total_tokens > 0 else float("nan")

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict | None = None,
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if model is None:
            return

        model.eval()
        val_indist_loss = self._compute_split_loss(model, self.val_indist_dataset)
        val_source_loss = self._compute_split_loss(model, self.val_source_sub)
        val_topic_loss  = self._compute_split_loss(model, self.val_topic_sub)
        model.train()

        log.info(
            "epoch %.0f — val_indist=%.4f  val_source=%.4f  val_topic=%.4f",
            state.epoch, val_indist_loss, val_source_loss, val_topic_loss,
        )

        if metrics is not None:
            metrics["eval_val_indist_loss"] = val_indist_loss
            metrics["eval_val_source_loss"] = val_source_loss
            metrics["eval_val_topic_loss"]  = val_topic_loss

        train_entries = [e for e in state.log_history if "loss" in e and "eval_loss" not in e]
        last_train = train_entries[-1] if train_entries else {}

        self.epoch_logs.append({
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_loss": last_train.get("loss", float("nan")),
            "learning_rate": last_train.get("learning_rate", float("nan")),
            "grad_norm": last_train.get("grad_norm", float("nan")),
            "val_indist_loss": val_indist_loss,
            "val_source_loss": val_source_loss,
            "val_topic_loss": val_topic_loss,
        })

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as fh:
            for record in self.epoch_logs:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # early stopping — EarlyStoppingCallback can't read custom-injected metrics
        if val_source_loss < self.best_val_source_loss - self.early_stopping_threshold:
            self.best_val_source_loss = val_source_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.early_stopping_patience:
            log.info(
                "early stopping after epoch %.0f — no improvement over %d epochs",
                state.epoch, self.early_stopping_patience,
            )
            control.should_training_stop = True


class GenerationCallback(TrainerCallback):
    """Generates text from fixed prompts at the end of each training epoch."""

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
        self.log_path = log_path

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        if not self.generation_params.enabled:
            return

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
                gen_kwargs["top_p"] = None

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            prompt_len     = inputs["input_ids"].shape[1]
            generated_ids  = output_ids[0, prompt_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            entries.append({
                "epoch": state.epoch,
                "global_step": state.global_step,
                "quadrant": self.quadrant,
                "prompt": prompt,
                "generated_text": generated_text,
            })

        model.train()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.info("generation logged for epoch %.0f — %d prompts", state.epoch, len(self.prompts))


# === GENERATION ===


# === EVALUATION ===


# === TRAINING ===

def run_training(
    quadrant: str,
    seed: int,
    config: ExpertConfig,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Run one complete training run for one quadrant at one seed.

    Args:
        quadrant:   one of the four quadrant names
        seed:       random seed for this run
        config:     full ExpertConfig
        output_dir: output directory for this run (quadrant/seed already included)
        dry_run:    if True, build all objects and run one forward pass but skip training

    Returns:
        Summary dict with quadrant, seed, best_val_source_loss, final_train_loss,
        total_steps, total_time_seconds, and config snapshot.

    Logic:
        Loads model and tokenizer, tokenizes all four splits, initialises callbacks,
        builds TrainingArguments and Trainer, then trains. After training, identifies
        the best checkpoint by val_source_loss from callback logs and copies that
        checkpoint directory to output_dir/best/.
    """
    t_start = time.time()
    set_seed(seed)

    # remove stale artifacts so re-runs start clean
    for stale_ckpt in output_dir.glob("checkpoint-*"):
        shutil.rmtree(stale_ckpt)
        log.info("removed stale checkpoint %s", stale_ckpt.name)
    for stale_log in [output_dir / "train_log.jsonl", output_dir / "generation_log.jsonl"]:
        if stale_log.exists():
            stale_log.unlink()
            log.info("removed stale %s", stale_log.name)

    model, tokenizer = build_lora_model(config.base_model, config.lora, config.precision)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())

    data_dir = config.train_validate_dir / quadrant
    train_chunks  = load_split(data_dir / "train.jsonl")
    indist_chunks = load_split(data_dir / "val_indist.jsonl")
    source_chunks = load_split(data_dir / "val_source.jsonl")
    topic_chunks  = load_split(data_dir / "val_topic.jsonl")

    log.info("tokenizing splits for quadrant=%s", quadrant)
    train_dataset      = tokenize_chunks(train_chunks,  tokenizer)
    val_indist_dataset = tokenize_chunks(indist_chunks, tokenizer)
    val_source_dataset = tokenize_chunks(source_chunks, tokenizer)
    val_topic_dataset  = tokenize_chunks(topic_chunks,  tokenizer)

    log.info(
        "dataset sizes — train=%d  val_indist=%d  val_source=%d  val_topic=%d",
        len(train_dataset), len(val_indist_dataset),
        len(val_source_dataset), len(val_topic_dataset),
    )

    data_collator = build_data_collator(tokenizer)

    multi_split_callback = MultiSplitEvalCallback(
        val_indist_dataset=val_indist_dataset,
        val_source_dataset=val_source_dataset,
        val_topic_dataset=val_topic_dataset,
        tokenizer=tokenizer,
        log_path=output_dir / "train_log.jsonl",
        early_stopping_patience=config.evaluation.early_stopping_patience,
        precision=config.precision,
    )
    generation_callback = GenerationCallback(
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
        eval_strategy=config.evaluation.eval_strategy,
        save_strategy=config.evaluation.save_strategy,
        save_total_limit=config.evaluation.save_total_limit,
        load_best_model_at_end=False,   # best checkpoint loaded manually below
        metric_for_best_model=config.evaluation.metric_for_best_model,
        greater_is_better=config.evaluation.greater_is_better,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        seed=seed,
        run_name=f"{quadrant}_r{config.lora.r}_lr{config.training.learning_rate}_seed{seed}",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_indist_dataset,
        data_collator=data_collator,
        callbacks=[
            multi_split_callback,   # must precede any metric-dependent callbacks
            generation_callback,
        ],
    )

    if dry_run:
        log.info("=== dry run — skipping trainer.train() ===")
        log.info(
            "train=%d  val_indist=%d  val_source_sub=%d  val_topic_sub=%d",
            len(train_dataset), len(val_indist_dataset),
            len(multi_split_callback.val_source_sub),
            len(multi_split_callback.val_topic_sub),
        )
        assert len(multi_split_callback.val_source_sub) > 0, "val_source subsample is empty"
        assert len(multi_split_callback.val_topic_sub)  > 0, "val_topic subsample is empty"
        log.info("callback subsamples verified ok")

        batch = data_collator([train_dataset[i] for i in range(min(2, len(train_dataset)))])
        batch = {k: v.to(_input_device(model)) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss_val = outputs.loss.item()
        assert not math.isnan(loss_val), "dry-run forward pass returned NaN loss"
        assert loss_val > 0, "dry-run forward pass returned zero loss"
        log.info("forward pass ok — loss=%.4f", loss_val)
        return {"quadrant": quadrant, "seed": seed, "dry_run": True, "forward_pass_loss": loss_val}

    log.info("starting training: quadrant=%s  seed=%d", quadrant, seed)
    trainer.train()

    # copy best checkpoint by val_source_loss; match by nearest global_step in case
    # step numbers at on_evaluate time don't align perfectly with saved checkpoint names
    best_dir  = output_dir / "best"
    ckpt_dirs = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    if multi_split_callback.epoch_logs and ckpt_dirs:
        best_epoch_record = min(
            multi_split_callback.epoch_logs, key=lambda r: r["val_source_loss"]
        )
        target_step = best_epoch_record["global_step"]
        best_ckpt = min(ckpt_dirs, key=lambda p: abs(int(p.name.split("-")[-1]) - target_step))
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(best_ckpt, best_dir)
        tokenizer.save_pretrained(str(best_dir))
        log.info(
            "copied %s → best/ (epoch %.0f  val_source_loss=%.4f)",
            best_ckpt.name, best_epoch_record["epoch"], best_epoch_record["val_source_loss"],
        )
    else:
        best_epoch_record = multi_split_callback.epoch_logs[-1] if multi_split_callback.epoch_logs else {}
        if not ckpt_dirs:
            log.warning("no checkpoint dirs found — saving current model weights")
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))

    log.info("best adapter saved to %s", best_dir)

    total_time = time.time() - t_start

    train_entries    = [e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    final_train_loss = train_entries[-1]["loss"] if train_entries else float("nan")

    summary = {
        "quadrant": quadrant,
        "seed": seed,
        "base_model": config.base_model,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "best_epoch": best_epoch_record.get("epoch", -1),
        "best_val_source_loss": best_epoch_record.get("val_source_loss", float("nan")),
        "best_val_indist_loss": best_epoch_record.get("val_indist_loss", float("nan")),
        "final_train_loss": final_train_loss,
        "total_steps": trainer.state.global_step,
        "total_time_seconds": round(total_time, 1),
        "best_checkpoint": str(best_dir),
        "best_metric": "val_source_loss",
        "config": {
            "lora": asdict(config.lora),
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
    p.add_argument(
        "--quadrant",
        required=True,
        choices=VALID_QUADRANTS,
        help="quadrant to train: right_auth, left_auth, left_lib, right_lib",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="path to config.yaml (default: project root config/config.yaml)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="single seed to run — overrides config.seeds",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build all objects and run one forward pass — do not train",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override config output_dir",
    )
    return p.parse_args()


def main() -> None:
    t_total = time.time()
    args = parse_args()

    config = load_config(args.config)
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
            "finished seed=%d — best_val_source_loss=%.4f  time=%.0fs",
            seed,
            summary.get("best_val_source_loss", float("nan")),
            summary.get("total_time_seconds", 0.0),
        )

    if len(seeds) > 1 and not args.dry_run:
        best = min(summaries, key=lambda s: s.get("best_val_source_loss", float("inf")))
        log.info(
            "best seed for %s: seed=%d  val_source_loss=%.4f",
            quadrant, best["seed"], best["best_val_source_loss"],
        )
        best_seed_summary = {
            "quadrant": quadrant,
            "seeds_run": seeds,
            "best_seed": best["seed"],
            "best_val_source_loss": best["best_val_source_loss"],
            "all_summaries": summaries,
        }
        summary_path = output_root / quadrant / "best_seed_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(best_seed_summary, fh, indent=2, ensure_ascii=False)
        log.info("best seed summary written to %s", summary_path)

    log.info("total elapsed: %.0fs", time.time() - t_total)


if __name__ == "__main__":
    main()
