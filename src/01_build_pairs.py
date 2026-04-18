# src/01_build_pairs.py — Build raw contrastive prompt pairs for the economic and social Political Compass axes from seed statements and templates


# === IMPORTS ===

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import yaml


# === CONFIG & UTILS ===

def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_values(values: list[str]) -> str:
    return ", ".join(values)


def build_pairs_for_axis(
    axis_name: str,
    axis_cfg: dict[str, Any],
    templates: list[dict[str, str]],
    left_stance: str,
    right_stance: str,
    ) -> list[dict[str, Any]]:
    """
    Build contrastive pairs for a single axis.
    Logic: For each seed statement and each template, create a pair of prompts contrasting the left and right stances, with values filled in from the config.
    Args: 
        - axis_name: "economic" or "social"
        - axis_cfg: config dict for the axis, containing seed statements and values
        - templates: list of prompt templates to apply to each statement
        - left_stance: key for the left stance in the config (e.g. "econ_left")
        - right_stance: key for the right stance in the config (e.g. "econ_right")
    Returns:
        - List of dicts, each containing the pair ID, axis, statement info, template info, labels, and the left and right prompts
    """
    pairs = []

    statements = axis_cfg["seed_statements"]
    values = axis_cfg["values"]

    left_values = format_values(values[left_stance])
    right_values = format_values(values[right_stance])

    for statement in statements:
        statement_id = statement["id"]
        statement_text = statement["text"]

        for template in templates:
            template_id = template["id"]
            template_text = template["text"]

            left_prompt = template_text.format(
                statement=statement_text,
                stance=left_stance,
                values=left_values,
            )

            right_prompt = template_text.format(
                statement=statement_text,
                stance=right_stance,
                values=right_values,
            )

            pair = {
                "id": f"{statement_id}_{template_id}",
                "axis": axis_name,
                "statement_id": statement_id,
                "statement": statement_text,
                "template_id": template_id,
                "negative_label": left_stance,
                "positive_label": right_stance,
                "neg": left_prompt,
                "pos": right_prompt,
            }
            pairs.append(pair)

    return pairs


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# === MAIN ===

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    paths = config["paths"]
    build_pairs_cfg = config["build_pairs"]

    economic_cfg = build_pairs_cfg["economic"]
    social_cfg = build_pairs_cfg["social"]
    templates = build_pairs_cfg["templates"]

    economic_pairs = build_pairs_for_axis(
        axis_name="economic",
        axis_cfg=economic_cfg,
        templates=templates,
        left_stance="econ_left",
        right_stance="econ_right",
    )

    social_pairs = build_pairs_for_axis(
        axis_name="social",
        axis_cfg=social_cfg,
        templates=templates,
        left_stance="libertarian",
        right_stance="authoritarian",
    )

    write_jsonl(economic_pairs, project_root / paths["economic_pairs_file"])
    write_jsonl(social_pairs, project_root / paths["social_pairs_file"])

    print(f"Saved {len(economic_pairs)} economic pairs")
    print(f"Saved {len(social_pairs)} social pairs")


if __name__ == "__main__":
    main()