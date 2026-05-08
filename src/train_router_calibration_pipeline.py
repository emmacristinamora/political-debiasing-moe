# src/train_router_calibration_pipeline.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any


# router_calibration_config and validate_router_dataset are torch-free.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_calibration_config import (  # noqa: E402
    load_router_calibration_config,
)
from validate_router_dataset import (  # noqa: E402
    load_hidden_tensor_safe,
    load_records_jsonl,
    validate_router_dataset,
)


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINER_SCRIPT_RELATIVE = Path("src/train_calibrated_router.py")

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
RECORDS_FILENAME: str = "records.jsonl"

DEFAULT_CHECKPOINT_FILENAME: str = "calibrated_router.pt"
DEFAULT_TRAINER_REPORT_FILENAME: str = "train_report.json"
DEFAULT_PIPELINE_REPORT_FILENAME: str = "pipeline_train_report.json"

# fallback locations (relative to PROJECT_ROOT) used only when the config does
# not provide checkpoints_dir / reports_dir / output_dir. exact strings match
# the values documented in the wrapper spec.
FALLBACK_DATASET_DIR: Path = Path("data/router")
FALLBACK_CHECKPOINT_PATH: Path = Path("data/router/checkpoints") / DEFAULT_CHECKPOINT_FILENAME
FALLBACK_TRAINER_REPORT_PATH: Path = Path("data/router/reports") / DEFAULT_TRAINER_REPORT_FILENAME
FALLBACK_PIPELINE_REPORT_PATH: Path = (
    Path("data/router/reports") / DEFAULT_PIPELINE_REPORT_FILENAME
)
FALLBACK_HIDDEN_PATH: Path = Path("data/router") / "hidden.pt"

TAIL_DEFAULT_CHARS: int = 4000


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === ARGUMENT PARSING ===

def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for the router-calibration training pipeline wrapper.
    Returns:
        argparse.Namespace: parsed args.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible wrapper around src/train_calibrated_router.py: "
            "validate inputs, build a deterministic command, optionally "
            "execute training, and write an auditable pipeline report."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-records-path", type=Path, default=None)
    parser.add_argument("--val-records-path",   type=Path, default=None)
    parser.add_argument("--test-records-path",  type=Path, default=None)
    parser.add_argument("--hidden-path",        type=Path, default=None)
    parser.add_argument("--output-path",        type=Path, default=None)
    parser.add_argument("--trainer-report-path",  type=Path, default=None)
    parser.add_argument("--pipeline-report-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


# === PATH AND HPARAM RESOLUTION ===

def resolve_paths(cfg: Any, args: argparse.Namespace) -> dict[str, Path]:
    """
    Resolve every input/output path the pipeline needs.
    Logic:
        Train/val/test default to <output_dir>/<split>/records.jsonl, mirroring
        the layout produced by src/split_router_dataset.py. CLI overrides win
        over config values; missing config fields fall back to the
        FALLBACK_* constants. All returned paths are pathlib.Path.
    """
    paths_cfg = getattr(cfg, "paths", None)

    output_dir = getattr(paths_cfg, "output_dir", None) if paths_cfg is not None else None
    base_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / FALLBACK_DATASET_DIR

    train_records_path = (
        args.train_records_path if args.train_records_path is not None
        else base_dir / "train" / RECORDS_FILENAME
    )
    val_records_path = (
        args.val_records_path if args.val_records_path is not None
        else base_dir / "val" / RECORDS_FILENAME
    )
    test_records_path = (
        args.test_records_path if args.test_records_path is not None
        else base_dir / "test" / RECORDS_FILENAME
    )

    hidden_path_cfg = getattr(paths_cfg, "hidden_path", None) if paths_cfg is not None else None
    if args.hidden_path is not None:
        hidden_path = args.hidden_path
    elif hidden_path_cfg is not None:
        hidden_path = Path(hidden_path_cfg)
    else:
        hidden_path = PROJECT_ROOT / FALLBACK_HIDDEN_PATH

    checkpoints_dir = (
        getattr(paths_cfg, "checkpoints_dir", None) if paths_cfg is not None else None
    )
    reports_dir = getattr(paths_cfg, "reports_dir", None) if paths_cfg is not None else None

    if args.output_path is not None:
        router_checkpoint = args.output_path
    elif checkpoints_dir is not None:
        router_checkpoint = Path(checkpoints_dir) / DEFAULT_CHECKPOINT_FILENAME
    else:
        router_checkpoint = PROJECT_ROOT / FALLBACK_CHECKPOINT_PATH

    if args.trainer_report_path is not None:
        trainer_report = args.trainer_report_path
    elif reports_dir is not None:
        trainer_report = Path(reports_dir) / DEFAULT_TRAINER_REPORT_FILENAME
    else:
        trainer_report = PROJECT_ROOT / FALLBACK_TRAINER_REPORT_PATH

    if args.pipeline_report_path is not None:
        pipeline_report = args.pipeline_report_path
    elif reports_dir is not None:
        pipeline_report = Path(reports_dir) / DEFAULT_PIPELINE_REPORT_FILENAME
    else:
        pipeline_report = PROJECT_ROOT / FALLBACK_PIPELINE_REPORT_PATH

    return {
        "train_records_path": Path(train_records_path),
        "val_records_path":   Path(val_records_path),
        "test_records_path":  Path(test_records_path),
        "hidden_path":        Path(hidden_path),
        "router_checkpoint":  Path(router_checkpoint),
        "trainer_report":     Path(trainer_report),
        "pipeline_report":    Path(pipeline_report),
    }


def resolve_training_hparams(cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    """
    Pull hyperparameters from cfg.training, with calibration_input_dim sourced
    from cfg.training if present, otherwise cfg.input_transformer. CLI --device
    overrides cfg.training.device. save_every_epoch is optional and defaults
    to False if missing from the config.
    """
    training = getattr(cfg, "training", None)
    if training is None:
        raise ValueError("config is missing the 'training' section")

    calibration_input_dim = getattr(training, "calibration_input_dim", None)
    if calibration_input_dim is None:
        input_transformer = getattr(cfg, "input_transformer", None)
        calibration_input_dim = getattr(input_transformer, "calibration_input_dim", None)
    if calibration_input_dim is None:
        raise ValueError(
            "config missing calibration_input_dim under 'training' or "
            "'input_transformer'"
        )

    device = args.device if args.device is not None else getattr(training, "device", "cpu")
    save_every_epoch = bool(getattr(training, "save_every_epoch", False))

    return {
        "calibration_input_dim": int(calibration_input_dim),
        "beta":             float(training.beta),
        "temperature":      float(training.temperature),
        "learning_rate":    float(training.learning_rate),
        "weight_decay":     float(training.weight_decay),
        "batch_size":       int(training.batch_size),
        "epochs":           int(training.epochs),
        "kl_weight":        float(training.kl_weight),
        "entropy_weight":   float(training.entropy_weight),
        "seed":             int(training.seed),
        "device":           str(device),
        "save_every_epoch": save_every_epoch,
    }


# === COMMAND CONSTRUCTION ===

def build_training_command(
    paths: dict[str, Path],
    hparams: dict[str, Any],
    max_examples: int | None = None,
) -> list[str]:
    """
    Build a deterministic argv list for src/train_calibrated_router.py.
    Logic:
        Mirrors the trainer's CLI exactly. Optional flags (--save-every-epoch,
        --max-examples) are appended only when set. All paths are stringified;
        no shell quoting is applied because subprocess is invoked with
        shell=False.
    """
    cmd: list[str] = [
        sys.executable,
        str(TRAINER_SCRIPT_RELATIVE),
        "--records-path", str(paths["train_records_path"]),
        "--hidden-path",  str(paths["hidden_path"]),
        "--output-path",  str(paths["router_checkpoint"]),
        "--report-path",  str(paths["trainer_report"]),
        "--calibration-input-dim", str(hparams["calibration_input_dim"]),
        "--beta",          str(hparams["beta"]),
        "--temperature",   str(hparams["temperature"]),
        "--learning-rate", str(hparams["learning_rate"]),
        "--weight-decay",  str(hparams["weight_decay"]),
        "--batch-size",    str(hparams["batch_size"]),
        "--epochs",        str(hparams["epochs"]),
        "--kl-weight",     str(hparams["kl_weight"]),
        "--entropy-weight", str(hparams["entropy_weight"]),
        "--seed",          str(hparams["seed"]),
        "--device",        str(hparams["device"]),
    ]
    if hparams.get("save_every_epoch"):
        cmd.append("--save-every-epoch")
    if max_examples is not None:
        cmd.extend(["--max-examples", str(max_examples)])
    return cmd


# === VALIDATION ===

def validate_training_inputs(
    paths: dict[str, Path],
    hparams: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate train/val/test records against the shared hidden tensor.
    Logic:
        hidden.pt and the train records file are required (raise on miss).
        Val/test files are optional; a missing one becomes a warning entry.
        Each present file is validated with the Step-8 record-level checks
        plus the cross-tensor row-index/hidden-dim/filename checks.
    """
    hidden_path: Path = paths["hidden_path"]
    if not hidden_path.is_file():
        raise FileNotFoundError(f"hidden tensor file not found: {hidden_path}")
    hidden_tensor = load_hidden_tensor_safe(hidden_path)

    train_path: Path = paths["train_records_path"]
    if not train_path.is_file():
        raise FileNotFoundError(f"train records file not found: {train_path}")
    train_records = load_records_jsonl(train_path)
    validate_router_dataset(
        train_records,
        hidden_tensor,
        expected_hidden_dim=hparams["calibration_input_dim"],
        hidden_filename=hidden_path.name,
    )

    result: dict[str, Any] = {
        "train": {"ok": True, "num_records": len(train_records)},
        "val":   {"ok": None, "num_records": None, "warning": None},
        "test":  {"ok": None, "num_records": None, "warning": None},
        "hidden_filename": hidden_path.name,
    }

    for split_name, split_path in (
        ("val",  paths["val_records_path"]),
        ("test", paths["test_records_path"]),
    ):
        if not split_path.is_file():
            result[split_name]["ok"] = False
            result[split_name]["warning"] = (
                f"{split_name} records file not found: {split_path}"
            )
            continue
        split_records = load_records_jsonl(split_path)
        validate_router_dataset(
            split_records,
            hidden_tensor,
            expected_hidden_dim=hparams["calibration_input_dim"],
            hidden_filename=hidden_path.name,
        )
        result[split_name]["ok"] = True
        result[split_name]["num_records"] = len(split_records)

    return result


def _empty_validation(paths: dict[str, Path]) -> dict[str, Any]:
    """Validation skeleton produced when --skip-validation is in effect."""
    return {
        "train": {"ok": None, "num_records": None},
        "val":   {"ok": None, "num_records": None, "warning": None},
        "test":  {"ok": None, "num_records": None, "warning": None},
        "hidden_filename": paths["hidden_path"].name,
    }


# === REPORTING HELPERS ===

def tail_text(text: str | None, max_chars: int = TAIL_DEFAULT_CHARS) -> str | None:
    """
    Return the last max_chars characters of text, preserving None.
    """
    if text is None:
        return None
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def write_pipeline_report(report: dict[str, Any], path: Path) -> None:
    """
    Persist the pipeline report as JSON, creating parent dirs as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)


def _build_report(
    *,
    config_path: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
    hparams: dict[str, Any],
    validation: dict[str, Any],
    command: list[str],
    execution: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble the pipeline-report dict per the spec schema."""
    return {
        "config_path": str(config_path),
        "dry_run": bool(args.dry_run),
        "skip_validation": bool(args.skip_validation),
        "input_paths": {
            "train_records_path": str(paths["train_records_path"]),
            "val_records_path":   str(paths["val_records_path"]),
            "test_records_path":  str(paths["test_records_path"]),
            "hidden_path":        str(paths["hidden_path"]),
        },
        "output_paths": {
            "router_checkpoint": str(paths["router_checkpoint"]),
            "trainer_report":    str(paths["trainer_report"]),
            "pipeline_report":   str(paths["pipeline_report"]),
        },
        "hyperparameters": dict(hparams),
        "validation": validation,
        "command": list(command),
        "execution": dict(execution),
        "warnings": list(warnings),
    }


# === RUNNER ===

def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """
    Top-level orchestration: load config, resolve paths and hyperparameters,
    validate dataset artifacts, build the trainer command, optionally execute
    it, and write the pipeline report. Returns the report dict.
    Logic:
        --dry-run skips subprocess execution entirely. --skip-validation
        bypasses every check (including hidden.pt presence) but the report
        records both flags. A non-zero trainer return code raises RuntimeError
        after the report has been written so the run remains auditable.
    """
    cfg = load_router_calibration_config(args.config)

    paths = resolve_paths(cfg, args)
    hparams = resolve_training_hparams(cfg, args)
    command = build_training_command(paths, hparams, max_examples=args.max_examples)

    warnings: list[str] = []
    execution: dict[str, Any] = {
        "executed": False,
        "returncode": None,
        "stdout_tail": None,
        "stderr_tail": None,
    }

    if args.skip_validation:
        validation = _empty_validation(paths)
        warnings.append("validation skipped via --skip-validation")
    else:
        validation = validate_training_inputs(paths, hparams)
        for split_name in ("val", "test"):
            warning = validation.get(split_name, {}).get("warning")
            if warning:
                warnings.append(warning)

    paths["router_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    paths["trainer_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["pipeline_report"].parent.mkdir(parents=True, exist_ok=True)

    error: Exception | None = None
    if not args.dry_run:
        log.info("executing trainer: %s", " ".join(command))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        execution["executed"] = True
        execution["returncode"] = int(completed.returncode)
        execution["stdout_tail"] = tail_text(completed.stdout)
        execution["stderr_tail"] = tail_text(completed.stderr)
        if completed.returncode != 0:
            error = RuntimeError(
                "train_calibrated_router.py failed with exit code "
                f"{completed.returncode}; see {paths['pipeline_report']} for details"
            )

    report = _build_report(
        config_path=args.config,
        args=args,
        paths=paths,
        hparams=hparams,
        validation=validation,
        command=command,
        execution=execution,
        warnings=warnings,
    )
    write_pipeline_report(report, paths["pipeline_report"])

    if error is not None:
        raise error

    return report


# === MAIN ===

def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
