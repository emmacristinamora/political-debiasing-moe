# src/build_router_prompt_set.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


# router_calibration_config lives next to this script in src/. add the parent
# directory to sys.path so it can be imported by name regardless of cwd.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from router_calibration_config import (  # noqa: E402
    ALLOWED_EXPERT_VALIDATION_SPLITS,
    CANONICAL_QUADRANT_ORDER,
    PromptSetConfig,
    RouterCalibrationConfig,
    load_router_calibration_config,
)


# === CONSTANTS ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ID_PREFIX = "router_prompt_"
EXAMPLE_ID_PAD_WIDTH = 6

SOURCE_METHOD12 = "method12"
SOURCE_METHOD3 = "method3"
# split names are already val_indist / val_source / val_topic, so the
# "expert_" prefix concatenates to expert_val_indist / _source / _topic.
EXPERT_VAL_SOURCE_PREFIX = "expert_"

REFUSED_OPTION_TEXT = "refused"
OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

REQUIRED_METHOD12_FIELDS: tuple[str, ...] = ("id", "text", "axis")
REQUIRED_METHOD3_FIELDS:  tuple[str, ...] = ("key", "question", "options")

# default expert-validation root, used only when the config does not provide
# paths.prompt_sources.expert_validation_dir.
DEFAULT_EXPERT_VAL_ROOT = PROJECT_ROOT / "data" / "experts" / "train-validate"

METHOD12_PROMPT_TEMPLATE = (
    "You are responding to a political attitude statement.\n\n"
    "Statement:\n{text}\n\n"
    "Write a concise 2-3 sentence response explaining your position."
)
METHOD3_PROMPT_HEADER = (
    "Answer the following survey question by choosing exactly one option.\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
    "Answer:"
)
EXPERT_VAL_PROMPT_TEMPLATE = (
    "Read the following political text excerpt and write a concise, balanced "
    "2-3 sentence response that addresses its main claim without adopting a "
    "partisan stance.\n\n"
    "Excerpt:\n{text}\n\n"
    "Response:"
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# === DATA LOADING ===

def _load_json_or_jsonl(path: Path) -> Any:
    """
    Load a file that may be a single JSON object/array or a JSONL sequence.
    Mirrors the helper in src/08_test_experts.py so this script stays
    importable without pulling in the torch-dependent test_experts module.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty input file: {path}")
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    records: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}") from exc
    return records


def _validate_required_fields(
    record: dict,
    required: tuple[str, ...],
    where: str,
) -> None:
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"{where} missing required fields: {missing}")


def load_method12_records(path: Path) -> list[dict]:
    """
    Load and validate method 1/2 statements from a JSON or JSONL file.

    Returns:
        list of statement dicts, each guaranteed to have id / text / axis.

    Logic:
        Accepts either a list of statements, or a top-level dict with a
        'statements' key. Validates required fields per record and preserves
        original ordering.
    """
    if not path.is_file():
        raise FileNotFoundError(f"method12 input not found: {path}")

    data = _load_json_or_jsonl(path)
    if isinstance(data, list):
        statements = data
    elif isinstance(data, dict):
        if "statements" not in data:
            raise ValueError(f"method12 file missing 'statements' key: {path}")
        statements = data["statements"]
    else:
        raise ValueError(f"unexpected method12 format in {path}")

    if not isinstance(statements, list):
        raise ValueError(f"method12 statements must be a list in {path}")

    for i, record in enumerate(statements):
        if not isinstance(record, dict):
            raise ValueError(f"method12 record {i} is not an object in {path}")
        _validate_required_fields(record, REQUIRED_METHOD12_FIELDS, f"{path.name}:statement[{i}]")

    log.info("loaded %d method12 statements from %s", len(statements), path.name)
    return statements


def load_method3_records(path: Path) -> list[dict]:
    """
    Load and validate method 3 multiple-choice questions from a JSON or JSONL
    file. Each record must have key / question / options.
    """
    if not path.is_file():
        raise FileNotFoundError(f"method3 input not found: {path}")

    data = _load_json_or_jsonl(path)
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        if "questions" not in data:
            raise ValueError(f"method3 file missing 'questions' key: {path}")
        questions = data["questions"]
    else:
        raise ValueError(f"unexpected method3 format in {path}")

    if not isinstance(questions, list):
        raise ValueError(f"method3 questions must be a list in {path}")

    for i, record in enumerate(questions):
        if not isinstance(record, dict):
            raise ValueError(f"method3 record {i} is not an object in {path}")
        _validate_required_fields(record, REQUIRED_METHOD3_FIELDS, f"{path.name}:question[{i}]")
        if not isinstance(record["options"], list) or not record["options"]:
            raise ValueError(
                f"{path.name}:question[{i}] options must be a non-empty list"
            )

    log.info("loaded %d method3 questions from %s", len(questions), path.name)
    return questions


def _iter_jsonl_with_lineno(path: Path):
    """
    Yield (lineno, record) for each non-blank JSONL line. Raises ValueError
    naming the file:line on malformed JSON; raises FileNotFoundError on a
    missing path.
    """
    if not path.is_file():
        raise FileNotFoundError(f"jsonl file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{lineno}") from exc


# === PROMPT CONSTRUCTION ===

def _format_method12_prompt(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("method12 statement has empty 'text' field")
    return METHOD12_PROMPT_TEMPLATE.format(text=text.strip())


def _filter_refused_options(options: list[Any]) -> list[str]:
    out: list[str] = []
    for opt in options:
        if not isinstance(opt, str):
            raise ValueError(f"method3 option is not a string: {opt!r}")
        if opt.strip().lower() == REFUSED_OPTION_TEXT:
            continue
        out.append(opt.strip())
    return out


def _format_method3_prompt(question: str, options: list[Any]) -> tuple[str, int]:
    """
    Render a method-3 prompt and return it together with the count of
    non-refused options actually included. Caller stores the count in metadata.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("method3 question has empty 'question' field")
    kept = _filter_refused_options(options)
    if not kept:
        raise ValueError("method3 question has no valid options after refused filtering")
    if len(kept) > len(OPTION_LABELS):
        raise ValueError(f"method3 question has too many options ({len(kept)} > 26)")
    labelled = "\n".join(f"{OPTION_LABELS[i]}. {opt}" for i, opt in enumerate(kept))
    prompt = METHOD3_PROMPT_HEADER.format(question=question.strip(), options=labelled)
    return prompt, len(kept)


def _format_expert_validation_prompt(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expert validation chunk has empty 'text' field")
    return EXPERT_VAL_PROMPT_TEMPLATE.format(text=text.strip())


def _build_method12_prompts(records: list[dict], source_file: str) -> list[dict]:
    out: list[dict] = []
    for record in records:
        prompt_text = _format_method12_prompt(record["text"])
        out.append({
            "prompt_text": prompt_text,
            "source": SOURCE_METHOD12,
            "metadata": {
                "original_id": record["id"],
                "axis": record.get("axis"),
                "source_file": source_file,
            },
        })
    return out


def _build_method3_prompts(records: list[dict], source_file: str) -> list[dict]:
    """
    Render every method-3 question into a prompt row. Per spec, metadata for
    method-3 prompts includes original_id (=question key), question_key,
    num_options (after refused-filtering), axis, and source_file.
    """
    out: list[dict] = []
    for record in records:
        prompt_text, num_options = _format_method3_prompt(
            record["question"], record["options"],
        )
        # spec: axis = record["axis"] if present, else axis_relevance, else None
        axis = record.get("axis")
        if axis is None:
            axis = record.get("axis_relevance")
        question_key = record["key"]
        out.append({
            "prompt_text": prompt_text,
            "source": SOURCE_METHOD3,
            "metadata": {
                "original_id": question_key,
                "axis": axis,
                "source_file": source_file,
                "question_key": question_key,
                "num_options": num_options,
            },
        })
    return out


def _build_expert_validation_prompts_for_file(
    *,
    file_path: Path,
    quadrant: str,
    split: str,
) -> list[dict]:
    """
    Build prompt rows from one quadrant's <split>.jsonl. Spec rules:
    - missing/empty 'text' raises ValueError naming file and line.
    - malformed JSON raises ValueError naming file and line.
    - source = 'expert_val_<split>'; metadata carries quadrant, split, and the
      original chunk_id / document_id / topic_label / source fields.
    """
    out: list[dict] = []
    source_name = f"{EXPERT_VAL_SOURCE_PREFIX}{split}"
    source_file = file_path.name
    for lineno, record in _iter_jsonl_with_lineno(file_path):
        if not isinstance(record, dict):
            raise ValueError(
                f"{file_path}:{lineno} expert validation row must be a JSON object"
            )
        text_raw = record.get("text")
        if not isinstance(text_raw, str) or not text_raw.strip():
            raise ValueError(
                f"{file_path}:{lineno} expert validation row missing or empty 'text'"
            )
        prompt_text = _format_expert_validation_prompt(text_raw)

        chunk_id = record.get("chunk_id")
        document_id = record.get("document_id")
        topic_label = record.get("topic_label")
        original_source = record.get("source")

        if isinstance(chunk_id, str) and chunk_id.strip():
            original_id = chunk_id
        elif isinstance(document_id, str) and document_id.strip():
            original_id = document_id
        else:
            # deterministic per (quadrant, split, lineno) — globally unique
            # within one builder run because (quadrant, split, lineno) is unique.
            original_id = f"{quadrant}/{split}#{lineno}"

        out.append({
            "prompt_text": prompt_text,
            "source": source_name,
            "metadata": {
                "original_id":     original_id,
                "axis":            None,
                "source_file":     source_file,
                "quadrant":        quadrant,
                "split":           split,
                "document_id":     document_id if isinstance(document_id, str) else None,
                "chunk_id":        chunk_id if isinstance(chunk_id, str) else None,
                "topic_label":     topic_label if isinstance(topic_label, str) else None,
                "original_source": original_source if isinstance(original_source, str) else None,
            },
        })
    return out


def _load_expert_validation_prompts(
    root: Path,
    splits: list[str],
) -> tuple[list[dict], dict[str, int]]:
    """
    Walk <root>/<quadrant>/<split>.jsonl in canonical quadrant order and
    deterministic split order, building prompt rows for every present file.
    Missing per-quadrant-split files are logged as warnings; only the root
    must exist (raised before this function is called).

    Returns:
        (prompts, per_source_counts) where per_source_counts is keyed by
        the prompt-row 'source' string ('expert_val_<split>').
    """
    prompts: list[dict] = []
    per_source: dict[str, int] = {f"{EXPERT_VAL_SOURCE_PREFIX}{s}": 0 for s in splits}

    for split in splits:
        if split not in ALLOWED_EXPERT_VALIDATION_SPLITS:
            raise ValueError(
                f"unknown expert validation split: {split!r}; expected one of "
                f"{list(ALLOWED_EXPERT_VALIDATION_SPLITS)}"
            )
        for quadrant in CANONICAL_QUADRANT_ORDER:
            file_path = root / quadrant / f"{split}.jsonl"
            if not file_path.is_file():
                log.warning(
                    "expert validation file missing: %s — skipping", file_path,
                )
                continue
            split_rows = _build_expert_validation_prompts_for_file(
                file_path=file_path, quadrant=quadrant, split=split,
            )
            prompts.extend(split_rows)
            per_source[f"{EXPERT_VAL_SOURCE_PREFIX}{split}"] += len(split_rows)
            log.info(
                "expert val %s/%s: %d prompts loaded from %s",
                quadrant, split, len(split_rows), file_path.name,
            )
    return prompts, per_source


def _dedupe_prompts(prompts: list[dict]) -> tuple[list[dict], int]:
    """
    Drop later rows that share a prompt_text (after .strip()) with an earlier
    row. Returns the deduped list plus the count of removed duplicates.
    """
    seen: set[str] = set()
    out: list[dict] = []
    duplicates = 0
    for record in prompts:
        text = record["prompt_text"].strip()
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        out.append(record)
    return out, duplicates


def _assign_example_ids(prompts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for i, record in enumerate(prompts, start=1):
        example_id = f"{EXAMPLE_ID_PREFIX}{i:0{EXAMPLE_ID_PAD_WIDTH}d}"
        if example_id in seen:
            raise ValueError(f"duplicate example_id generated: {example_id}")
        seen.add(example_id)
        if not record["prompt_text"]:
            raise ValueError(f"empty prompt_text at position {i}")
        out.append({
            "example_id": example_id,
            "prompt_text": record["prompt_text"],
            "source": record["source"],
            "metadata": record["metadata"],
        })
    return out


def build_prompt_records(
    method12_records: list[dict] | None,
    method3_records: list[dict] | None,
    method12_source_file: str | None,
    method3_source_file: str | None,
    max_prompts: int | None,
    expert_validation_prompts: list[dict] | None = None,
) -> list[dict]:
    """
    Assemble the full prompt set from optional method12 + method3 inputs and
    pre-built expert-validation rows.

    Args:
        method12_records:           method 1/2 statements list, or None to skip.
        method3_records:            method 3 questions list, or None to skip.
        method12_source_file:       filename embedded in metadata for method12 prompts.
        method3_source_file:        filename embedded in metadata for method3 prompts.
        max_prompts:                hard cap applied AFTER merge + dedup; None to keep all.
        expert_validation_prompts:  pre-formatted expert-validation rows
                                    (already carry their own 'source' tag), or None.

    Returns:
        list of fully populated prompt records, each matching the schema:
        {example_id, prompt_text, source, metadata}.

    Logic:
        Source order: method12 → method3 → expert_val_indist → expert_val_source
        → expert_val_topic. Deduplicates on stripped prompt_text keeping the
        first occurrence, then truncates to max_prompts (no shuffling), then
        assigns sequential zero-padded example_ids and validates uniqueness.
    """
    merged: list[dict] = []
    counts: dict[str, int] = {SOURCE_METHOD12: 0, SOURCE_METHOD3: 0}

    if method12_records is not None:
        if method12_source_file is None:
            raise ValueError("method12_source_file required when method12_records provided")
        m12 = _build_method12_prompts(method12_records, method12_source_file)
        merged.extend(m12)
        counts[SOURCE_METHOD12] = len(m12)

    if method3_records is not None:
        if method3_source_file is None:
            raise ValueError("method3_source_file required when method3_records provided")
        m3 = _build_method3_prompts(method3_records, method3_source_file)
        merged.extend(m3)
        counts[SOURCE_METHOD3] = len(m3)

    expert_val_count = 0
    if expert_validation_prompts:
        merged.extend(expert_validation_prompts)
        expert_val_count = len(expert_validation_prompts)
        for row in expert_validation_prompts:
            counts[row["source"]] = counts.get(row["source"], 0) + 1

    if not merged:
        raise ValueError(
            "no prompts generated — check prompt_set toggles and input files"
        )

    if max_prompts is not None and max_prompts <= 0:
        raise ValueError(f"max_prompts must be positive, got {max_prompts}")

    deduped, duplicates_removed = _dedupe_prompts(merged)

    if max_prompts is not None:
        deduped = deduped[:max_prompts]

    finalised = _assign_example_ids(deduped)

    log.info(
        "built %d prompts — method12=%d method3=%d expert_val=%d "
        "duplicates_removed=%d cap=%s",
        len(finalised),
        counts[SOURCE_METHOD12],
        counts[SOURCE_METHOD3],
        expert_val_count,
        duplicates_removed,
        max_prompts if max_prompts is not None else "none",
    )
    return finalised


# === IO ===

def write_prompts_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records to a JSONL file, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("wrote %d prompts to %s", len(records), output_path)


# === ORCHESTRATION ===

def _resolve_method12_inputs(cfg: RouterCalibrationConfig) -> tuple[list[dict], str]:
    src_path = cfg.paths.prompt_sources.method12_path
    records = load_method12_records(src_path)
    return records, src_path.name


def _resolve_method3_inputs(cfg: RouterCalibrationConfig) -> tuple[list[dict], str]:
    src_path = cfg.paths.prompt_sources.method3_path
    records = load_method3_records(src_path)
    return records, src_path.name


def _resolve_expert_validation_root(cfg: RouterCalibrationConfig) -> Path:
    """
    Pick the expert-validation root from config or fall back to the canonical
    location under data/experts/train-validate. Raise FileNotFoundError if the
    root does not exist when expert validation is enabled.
    """
    cfg_root: Path | None = getattr(
        cfg.paths.prompt_sources, "expert_validation_dir", None,
    )
    root = Path(cfg_root) if cfg_root is not None else DEFAULT_EXPERT_VAL_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"expert validation root not found: {root} (include_expert_validation=true)"
        )
    return root


def run_build(cfg: RouterCalibrationConfig) -> Path:
    """
    Top-level driver: pulls toggles + paths from config, loads enabled
    sources (method12, method3, expert validation), builds prompt records
    with dedup + cap, writes JSONL, and returns the output path.
    """
    prompt_set: PromptSetConfig = cfg.prompt_set

    enabled: list[str] = []
    if prompt_set.include_method12:
        enabled.append("method12")
    if prompt_set.include_method3:
        enabled.append("method3")
    if prompt_set.include_expert_validation:
        enabled.append("expert_validation")
    log.info("enabled prompt sources: %s", enabled if enabled else "<none>")

    method12_records: list[dict] | None = None
    method12_source_file: str | None = None
    if prompt_set.include_method12:
        log.info("loading method12 from %s", cfg.paths.prompt_sources.method12_path)
        method12_records, method12_source_file = _resolve_method12_inputs(cfg)

    method3_records: list[dict] | None = None
    method3_source_file: str | None = None
    if prompt_set.include_method3:
        log.info("loading method3 from %s", cfg.paths.prompt_sources.method3_path)
        method3_records, method3_source_file = _resolve_method3_inputs(cfg)

    expert_val_prompts: list[dict] | None = None
    if prompt_set.include_expert_validation:
        root = _resolve_expert_validation_root(cfg)
        log.info(
            "loading expert validation prompts from %s (splits=%s)",
            root, list(prompt_set.expert_validation_splits),
        )
        expert_val_prompts, _ = _load_expert_validation_prompts(
            root, list(prompt_set.expert_validation_splits),
        )

    if (
        method12_records is None
        and method3_records is None
        and not expert_val_prompts
    ):
        raise ValueError(
            "prompt_set has no sources enabled — set include_method12, "
            "include_method3, and/or include_expert_validation to true in "
            "router_calibration.prompt_set"
        )

    records = build_prompt_records(
        method12_records=method12_records,
        method3_records=method3_records,
        method12_source_file=method12_source_file,
        method3_source_file=method3_source_file,
        max_prompts=prompt_set.max_prompts,
        expert_validation_prompts=expert_val_prompts,
    )

    output_path = cfg.paths.prompts_path
    write_prompts_jsonl(records, output_path)
    return output_path


# === MAIN ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="build the router calibration prompt set (data/router/prompts.jsonl)",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to config.yaml containing the router_calibration block",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_router_calibration_config(args.config)
    log.info("config loaded from %s", args.config)
    run_build(cfg)


if __name__ == "__main__":
    main()
