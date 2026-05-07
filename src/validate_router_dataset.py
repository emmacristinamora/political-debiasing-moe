# src/validate_router_dataset.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any


# router_calibration_config carries CANONICAL_QUADRANT_ORDER and is torch-free.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_calibration_config import (  # noqa: E402
    CANONICAL_QUADRANT_ORDER,
    load_router_calibration_config,
)


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# tolerance for "target_policy sums to 1" check; must equal the value used
# by src/train_calibrated_router.py so the validator passes the same set of
# records the trainer would accept (and rejects the same ones).
DISTRIBUTION_SUM_TOLERANCE: float = 1e-6

REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "example_id",
    "prompt_text",
    "quadrant_scores",
    "bias_magnitude",
    "target_policy",
    "hidden_representation_ref",
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === LAZY TORCH ===

_torch_load_attempted: bool = False
_torch_module: Any | None = None


def _try_import_torch() -> Any | None:
    """
    Return the torch module if importable, else None. Cached so we attempt
    the import at most once per process. The validator stays usable in
    environments without torch — it just skips tensor-internals checks.
    """
    global _torch_load_attempted, _torch_module
    if _torch_load_attempted:
        return _torch_module
    try:
        _torch_module = importlib.import_module("torch")
    except ImportError:
        _torch_module = None
    finally:
        _torch_load_attempted = True
    return _torch_module


# === FIELD VALIDATORS ===

def validate_quadrant_dict(d: Any, name: str, example_id: str) -> None:
    """
    Validate a finite numeric dict over CANONICAL_QUADRANT_ORDER (no positivity
    or sum constraints). Mirrors src/train_calibrated_router.validate_score_dict
    so the validator accepts/rejects the same inputs the trainer does.
    """
    if not isinstance(d, dict):
        raise ValueError(
            f"[{example_id}] {name} must be a dict, got {type(d).__name__}"
        )
    expected = set(CANONICAL_QUADRANT_ORDER)
    actual = set(d.keys())
    missing = expected - actual
    if missing:
        raise ValueError(
            f"[{example_id}] {name} missing required keys: {sorted(missing)}; "
            f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
        )
    unexpected = actual - expected
    if unexpected:
        raise ValueError(
            f"[{example_id}] {name} has unexpected keys: {sorted(unexpected)}; "
            f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
        )
    for key in CANONICAL_QUADRANT_ORDER:
        value = d[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"[{example_id}] {name}[{key!r}] must be int or float, "
                f"got {type(value).__name__}"
            )
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"[{example_id}] {name}[{key!r}] is not finite")


def validate_probability_dict(d: Any, name: str, example_id: str) -> None:
    """
    Validate a strictly-positive distribution over CANONICAL_QUADRANT_ORDER
    summing to 1 within DISTRIBUTION_SUM_TOLERANCE. Mirrors the trainer's
    rule of the same name.
    """
    validate_quadrant_dict(d, name, example_id)
    for key in CANONICAL_QUADRANT_ORDER:
        v = d[key]
        if v <= 0:
            raise ValueError(
                f"[{example_id}] {name}[{key!r}] must be strictly positive; got {v}"
            )
    total = sum(float(d[key]) for key in CANONICAL_QUADRANT_ORDER)
    if abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
        raise ValueError(
            f"[{example_id}] {name} must sum to 1 within "
            f"{DISTRIBUTION_SUM_TOLERANCE}; got sum={total}"
        )


def parse_hidden_ref(ref: Any) -> tuple[str, int]:
    """
    Parse 'filename:row_index' into (filename, row_index). Filename is the
    full pre-colon portion (callers compare it to the expected basename).
    Row index is a non-negative int. Raises ValueError on any malformation.
    """
    if not isinstance(ref, str):
        raise ValueError(
            f"hidden_representation_ref must be a string; got {type(ref).__name__}"
        )
    if ":" not in ref:
        raise ValueError(
            f"hidden_representation_ref must be '<filename>:<row_index>'; got {ref!r}"
        )
    filename, _, index_part = ref.rpartition(":")
    if not filename:
        raise ValueError(
            f"hidden_representation_ref filename portion is empty; got {ref!r}"
        )
    try:
        row_index = int(index_part)
    except ValueError as exc:
        raise ValueError(
            f"hidden_representation_ref row index {index_part!r} is not a "
            f"valid int; ref={ref!r}"
        ) from exc
    if row_index < 0:
        raise ValueError(
            f"hidden_representation_ref row index must be >= 0; got {row_index}"
        )
    return filename, row_index


# === RECORD-LEVEL VALIDATION ===

def _validate_single_record(record: Any, where: str) -> str:
    """
    Validate one JSONL row's required fields, returning the example_id for
    duplicate-id tracking by the caller.
    """
    if not isinstance(record, dict):
        raise ValueError(f"{where}: record must be a dict, got {type(record).__name__}")

    eid_raw = record.get("example_id", "<missing example_id>")
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            raise ValueError(
                f"[{eid_raw}] record missing required field {field!r} ({where})"
            )

    eid = record["example_id"]
    if not isinstance(eid, str):
        raise ValueError(
            f"{where}: example_id must be str, got {type(eid).__name__}"
        )
    if not eid.strip():
        raise ValueError(f"{where}: example_id must be a non-empty string")

    prompt = record["prompt_text"]
    if not isinstance(prompt, str):
        raise ValueError(f"[{eid}] prompt_text must be str, got {type(prompt).__name__}")
    if not prompt.strip():
        raise ValueError(f"[{eid}] prompt_text must be a non-empty string")

    bm = record["bias_magnitude"]
    if isinstance(bm, bool) or not isinstance(bm, (int, float)):
        raise ValueError(
            f"[{eid}] bias_magnitude must be int or float, got {type(bm).__name__}"
        )
    if math.isnan(bm) or math.isinf(bm):
        raise ValueError(f"[{eid}] bias_magnitude is not finite")

    validate_quadrant_dict(record["quadrant_scores"], "quadrant_scores", eid)
    validate_probability_dict(record["target_policy"], "target_policy", eid)

    href = record["hidden_representation_ref"]
    if not isinstance(href, str) or not href:
        raise ValueError(
            f"[{eid}] hidden_representation_ref must be a non-empty string"
        )

    if "metadata" in record and not isinstance(record["metadata"], dict):
        raise ValueError(f"[{eid}] metadata must be a dict if present")

    return eid


# === HIDDEN-TENSOR VALIDATION ===

def _safe_shape(tensor: Any) -> tuple[int, ...] | None:
    """Return tuple(tensor.shape) if available, else None."""
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except (TypeError, ValueError):
        return None


def validate_hidden_tensor(tensor: Any, expected_hidden_dim: int | None) -> None:
    """
    Strict tensor-internals validation. Requires torch; if torch is missing,
    callers should skip this and call only the structural checks. Mirrors the
    trainer's expectations: 2D float-like tensor with no NaN/inf, hidden_dim
    matching expected_hidden_dim if provided.
    """
    torch_module = _try_import_torch()
    if torch_module is None:
        raise ValueError(
            "validate_hidden_tensor requires torch to be installed; "
            "install torch or pass hidden_tensor=None to the top-level validator"
        )
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError(
            f"hidden_tensor must be torch.Tensor, got {type(tensor).__name__}"
        )
    if tensor.dim() != 2:
        raise ValueError(
            f"hidden_tensor must be 2D [N, hidden_dim]; got shape {tuple(tensor.shape)}"
        )
    if torch_module.is_floating_point(tensor):
        if not torch_module.isfinite(tensor).all().item():
            raise ValueError("hidden_tensor contains NaN or inf entries")
    if expected_hidden_dim is not None and tensor.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"hidden_tensor hidden_dim {tensor.shape[1]} does not match "
            f"expected {expected_hidden_dim}"
        )


def load_hidden_tensor_safe(path: Path) -> Any | None:
    """
    Load hidden.pt with torch if available, returning the tensor. If torch
    is missing, log a warning and return None — the caller continues with
    structural-only validation.
    """
    if not path.is_file():
        raise FileNotFoundError(f"hidden tensor file not found: {path}")
    torch_module = _try_import_torch()
    if torch_module is None:
        log.warning(
            "torch not installed; skipping tensor-internals validation for %s", path,
        )
        return None
    return torch_module.load(path, map_location="cpu", weights_only=False)


# === IO ===

def load_records_jsonl(path: Path) -> list[dict]:
    """Read records.jsonl. Raises on missing/empty/malformed input."""
    if not path.is_file():
        raise FileNotFoundError(f"records file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"records file is empty: {path}")
    out: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}") from exc
        out.append(record)
    return out


# === MAIN VALIDATOR ===

def validate_router_dataset(
    records: list[dict],
    hidden_tensor: Any | None,
    *,
    expected_hidden_dim: int | None = None,
    hidden_filename: str | None = None,
) -> None:
    """
    Strict end-to-end validation of (records, hidden_tensor) against the
    trainer's expectations. Returns None on success; raises ValueError with
    a precise message on the first failure.

    Args:
        records:             list of dicts loaded from records.jsonl.
        hidden_tensor:       loaded hidden tensor, or None to skip tensor-side
                             checks (structural ref checks still run).
        expected_hidden_dim: optional cross-check against tensor.shape[1].
        hidden_filename:     optional basename to compare against the pre-colon
                             portion of every hidden_representation_ref.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    seen_ids: set[str] = set()
    tensor_n_rows: int | None = None
    if hidden_tensor is not None:
        shape = _safe_shape(hidden_tensor)
        if shape is None or len(shape) < 1:
            raise ValueError(
                f"hidden_tensor must expose a .shape attribute with at least 1 dim; "
                f"got {type(hidden_tensor).__name__}"
            )
        tensor_n_rows = shape[0]

    for i, record in enumerate(records):
        eid = _validate_single_record(record, where=f"records[{i}]")
        if eid in seen_ids:
            raise ValueError(f"duplicate example_id: {eid!r}")
        seen_ids.add(eid)

        ref_filename, row_index = parse_hidden_ref(record["hidden_representation_ref"])

        if hidden_filename is not None and ref_filename != hidden_filename:
            raise ValueError(
                f"[{eid}] hidden_representation_ref filename {ref_filename!r} does "
                f"not match expected {hidden_filename!r}"
            )

        if tensor_n_rows is not None and row_index >= tensor_n_rows:
            raise ValueError(
                f"[{eid}] hidden row index {row_index} out of range (max="
                f"{tensor_n_rows - 1})"
            )

    # tensor-side checks
    if hidden_tensor is not None:
        torch_module = _try_import_torch()
        if torch_module is not None and isinstance(hidden_tensor, torch_module.Tensor):
            validate_hidden_tensor(hidden_tensor, expected_hidden_dim)
        elif expected_hidden_dim is not None:
            # torch unavailable or duck-typed tensor — still cross-check
            # hidden_dim via .shape since callers may pass mock objects in tests
            shape = _safe_shape(hidden_tensor)
            if shape is None or len(shape) < 2:
                raise ValueError(
                    "hidden_tensor must expose a 2D .shape; "
                    f"got shape={shape!r}"
                )
            if shape[1] != expected_hidden_dim:
                raise ValueError(
                    f"hidden_tensor hidden_dim {shape[1]} does not match "
                    f"expected {expected_hidden_dim}"
                )

        # cross-consistency: records must fit inside the tensor; unused rows allowed
        if tensor_n_rows is not None and len(records) > tensor_n_rows:
            raise ValueError(
                f"records count {len(records)} exceeds hidden_tensor row count "
                f"{tensor_n_rows}"
            )


# === CLI ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "validate router-calibration records.jsonl + hidden.pt against the "
            "schema expected by src/train_calibrated_router.py"
        ),
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="path to config.yaml — only consulted for calibration_input_dim",
    )
    p.add_argument("--records-path", type=Path, required=True)
    p.add_argument("--hidden-path",  type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    expected_hidden_dim: int | None = None
    if args.config is not None:
        cfg = load_router_calibration_config(args.config)
        expected_hidden_dim = cfg.input_transformer.calibration_input_dim

    log.info("loading records from %s", args.records_path)
    records = load_records_jsonl(args.records_path)
    log.info("loaded %d records", len(records))

    hidden_tensor: Any | None = None
    hidden_filename: str | None = None
    if args.hidden_path is not None:
        log.info("loading hidden tensor from %s", args.hidden_path)
        hidden_tensor = load_hidden_tensor_safe(args.hidden_path)
        hidden_filename = args.hidden_path.name

    validate_router_dataset(
        records,
        hidden_tensor,
        expected_hidden_dim=expected_hidden_dim,
        hidden_filename=hidden_filename,
    )

    torch_module = _try_import_torch()
    torch_status = "available" if torch_module is not None else "missing"
    if hidden_tensor is not None:
        shape = _safe_shape(hidden_tensor)
        hidden_dim_str = str(shape[1]) if shape is not None and len(shape) >= 2 else "unknown"
    else:
        hidden_dim_str = "n/a"
    log.info(
        "Validated %d records (hidden_dim=%s, torch=%s)",
        len(records), hidden_dim_str, torch_status,
    )


if __name__ == "__main__":
    main()
