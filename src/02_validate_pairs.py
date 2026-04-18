# src/02_validate_pairs.py


# === IMPORTS ===

from __future__ import annotations
import json
import re
import yaml
from collections import Counter
from pathlib import Path
from typing import Any


# === CONFIG ===

REQUIRED_FIELDS = [
    "id",
    "axis",
    "statement_id",
    "statement",
    "template_id",
    "pos",
    "neg",
]

VALID_AXES = {"economic", "social"}


# === HELPERS ===

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file into memory.
    Args:
        path: Input JSONL path.
    Returns:
        list[dict[str, Any]]: Parsed rows.
    Logic:
        Reads one JSON object per line and raises loudly if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """
    Write records to JSONL.
    Args:
        records: Records to write.
        path: Output JSONL path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(payload: dict[str, Any], path: Path) -> None:
    """
    Save a JSON summary file.
    Args:
        payload: JSON-serializable dictionary.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate and identity checks.
    Args:
        text: Raw text.
    Returns:
        str: Lowercased, whitespace-normalized text with punctuation stripped.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def word_count(text: str) -> int:
    """
    Count words in a string.
    Args:
        text: Input text.
    Returns:
        int: Number of whitespace-separated tokens.
    """
    return len(text.split())


def validate_required_fields(record: dict[str, Any]) -> list[str]:
    """
    Check whether a record contains all required fields and valid types.
    Args:
        record: Input pair record.
    Returns:
        list[str]: Reject reasons, empty if valid.
    """
    reasons: list[str] = []

    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            reasons.append(f"missing_field:{field_name}")

    if reasons:
        return reasons

    for field_name in ["id", "axis", "statement_id", "statement", "template_id", "pos", "neg"]:
        if not isinstance(record[field_name], str):
            reasons.append(f"wrong_type:{field_name}")

    if record.get("axis") not in VALID_AXES:
        reasons.append("invalid_axis")

    return reasons


def validate_prompt_content(record: dict[str, Any], max_length_ratio: float) -> tuple[list[str], dict[str, Any]]:
    """
    Run minimal integrity checks on the pair prompts.
    Args:
        record: Input pair record.
        max_length_ratio: Maximum allowed word-count ratio.
    Returns:
        tuple[list[str], dict[str, Any]]:
            - reject reasons
            - validation stats
    """
    reasons: list[str] = []

    pos_text = record["pos"].strip()
    neg_text = record["neg"].strip()

    pos_words = word_count(pos_text)
    neg_words = word_count(neg_text)

    normalized_pos = normalize_text(pos_text)
    normalized_neg = normalize_text(neg_text)

    if not pos_text:
        reasons.append("empty_pos")

    if not neg_text:
        reasons.append("empty_neg")

    if pos_text == neg_text:
        reasons.append("identical_prompts_exact")

    if normalized_pos == normalized_neg:
        reasons.append("identical_prompts_normalized")

    min_words = min(pos_words, neg_words)
    max_words = max(pos_words, neg_words)

    if min_words == 0:
        length_ratio = None
    else:
        length_ratio = max_words / min_words
        if length_ratio > max_length_ratio:
            reasons.append("length_ratio_too_high")

    stats = {
        "pos_word_count": pos_words,
        "neg_word_count": neg_words,
        "length_ratio": length_ratio,
    }

    return reasons, stats


def build_duplicate_key(record: dict[str, Any]) -> str:
    """
    Create a normalized duplicate key for a pair.
    Args:
        record: Input pair record.
    Returns:
        str: Stable normalized key.
    """
    return "||".join(
        [
            record["axis"],
            record["statement_id"],
            record["template_id"],
            normalize_text(record["pos"]),
            normalize_text(record["neg"]),
        ]
    )


# === MAIN ===

def validate_axis(
    input_path: Path,
    validated_output: Path,
    rejected_output: Path,
    report_output: Path,
    max_length_ratio: float,
) -> None:
    records = load_jsonl(input_path)

    validated_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    reject_counter: Counter[str] = Counter()
    seen_duplicate_keys: set[str] = set()

    for record in records:
        reasons = validate_required_fields(record)

        stats: dict[str, Any] = {}
        if not reasons:
            prompt_reasons, stats = validate_prompt_content(
                record=record,
                max_length_ratio=max_length_ratio,
            )
            reasons.extend(prompt_reasons)

        if not reasons:
            duplicate_key = build_duplicate_key(record)
            if duplicate_key in seen_duplicate_keys:
                reasons.append("duplicate_pair")
            else:
                seen_duplicate_keys.add(duplicate_key)

        enriched_record = dict(record)
        enriched_record["validation"] = {
            "passed": len(reasons) == 0,
            "reject_reasons": reasons,
            **stats,
        }

        if reasons:
            rejected_records.append(enriched_record)
            reject_counter.update(reasons)
        else:
            validated_records.append(enriched_record)

    summary = {
        "input_file": str(input_path),
        "total_records": len(records),
        "validated_records": len(validated_records),
        "rejected_records": len(rejected_records),
        "max_length_ratio": max_length_ratio,
        "reject_reason_counts": dict(reject_counter),
    }

    write_jsonl(validated_records, validated_output)
    write_jsonl(rejected_records, rejected_output)
    save_json(summary, report_output)

    print(f"[{input_path.stem}] Loaded {len(records)} | Validated {len(validated_records)} | Rejected {len(rejected_records)}")
    print(f"  Report → {report_output}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config" / "config.yaml")

    paths = config["paths"]
    max_length_ratio = config.get("validate_pairs", {}).get("max_length_ratio", 1.5)

    validated_dir = project_root / paths["validated_pairs_dir"]
    reports_dir = project_root / paths["reports_dir"]

    axes = {
        "economic": project_root / paths["economic_pairs_file"],
        "social": project_root / paths["social_pairs_file"],
    }

    for axis_name, input_path in axes.items():
        validate_axis(
            input_path=input_path,
            validated_output=validated_dir / f"{axis_name}_pairs_validated.jsonl",
            rejected_output=validated_dir / f"{axis_name}_pairs_rejected.jsonl",
            report_output=reports_dir / f"{axis_name}_validation_report.json",
            max_length_ratio=max_length_ratio,
        )


if __name__ == "__main__":
    main()