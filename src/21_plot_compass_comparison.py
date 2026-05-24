# src/21_plot_compass_comparison.py

# Paper-ready political compass figure from script-20 results.
#
# Reads one entry per model from the global_centroid + per_prompt fields
# produced by script 20, computes per-prompt standard deviations, and
# draws a publication-quality scatter plot.
#
#   input:  data/evaluation/compass_comparison/results.json
#   output: docs/fig_compass_comparison.png


# === IMPORTS ===

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# === CONFIG ===

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS = PROJECT_ROOT / "data" / "evaluation" / "compass_comparison" / "results.json"
DEFAULT_OUTPUT  = PROJECT_ROOT / "docs" / "fig_compass_comparison.png"

# Display configuration per model key.
# label_xy: annotation text position in data coordinates.
# arrow: whether to draw a leader line from text to marker.
MODEL_CFG: dict[str, dict[str, Any]] = {
    "mistralai/Mistral-7B-v0.1": {
        "label":    "Mistral 7B (base)",
        "color":    "#666666",
        "marker":   "o",
        "ms":       90,
        "lw":       1.2,
        "zorder":   3,
        "label_xy": (-0.36, +0.195),
        "arrow":    False,
    },
    "run_moce": {
        "label":    "MoCE architecture",
        "color":    "#cc2200",
        "marker":   "*",
        "ms":       320,
        "lw":       1.5,
        "zorder":   5,
        "label_xy": (-0.30, -0.105),
        "arrow":    True,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "label":    "Qwen 2.5 7B",
        "color":    "#2255cc",
        "marker":   "s",
        "ms":       85,
        "lw":       1.2,
        "zorder":   3,
        "label_xy": (-0.72, -0.27),
        "arrow":    True,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "label":    "DeepSeek-R1 7B",
        "color":    "#228833",
        "marker":   "^",
        "ms":       90,
        "lw":       1.2,
        "zorder":   3,
        "label_xy": (-0.36, +0.085),
        "arrow":    False,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "label":    "Llama 3.1 8B",
        "color":    "#aa44cc",
        "marker":   "D",
        "ms":       75,
        "lw":       1.2,
        "zorder":   3,
        "label_xy": (-0.62, +0.04),
        "arrow":    True,
    },
    "google/gemma-2-9b-it": {
        "label":    "Gemma 2 9B",
        "color":    "#dd8800",
        "marker":   "P",
        "ms":       95,
        "lw":       1.2,
        "zorder":   3,
        "label_xy": (-0.30, -0.22),
        "arrow":    True,
    },
}

MOCE_KEY = "run_moce"
AXIS_PAD = 0.12


# === HELPERS: IO ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script 21 — paper-ready political compass comparison figure."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi",     type=int,  default=300)
    return parser.parse_args()


def load_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# === HELPERS: STATISTICS ===

def extract_model_stats(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """
    Return per-model centroid and per-prompt standard deviation.

    Uses the global_centroid for the plotted point and computes std
    directly from the per_prompt centroid list.
    """
    out: dict[str, dict[str, float]] = {}
    for model, v in data["models"].items():
        gc   = v["global_centroid"]
        econ = [p["centroid"]["economic"] for p in v["per_prompt"]]
        soc  = [p["centroid"]["social"]   for p in v["per_prompt"]]
        n    = len(econ)
        me   = sum(econ) / n
        ms   = sum(soc)  / n
        out[model] = {
            "econ":  gc["economic"],
            "soc":   gc["social"],
            "std_e": math.sqrt(sum((x - me) ** 2 for x in econ) / n),
            "std_s": math.sqrt(sum((x - ms) ** 2 for x in soc)  / n),
            "n":     n,
        }
    return out


def compute_axis_limits(
    stats: dict[str, dict[str, float]],
    pad: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute (xlim, ylim) that contain all centroids ± 1σ and always
    include a margin around the origin so the compass centre is visible.
    """
    all_e  = [v["econ"]  for v in stats.values()]
    all_s  = [v["soc"]   for v in stats.values()]
    all_se = [v["std_e"] for v in stats.values()]
    all_ss = [v["std_s"] for v in stats.values()]

    xlim = (
        min(e - se for e, se in zip(all_e, all_se)) - pad,
        max(max(e + se for e, se in zip(all_e, all_se)) + pad, 0.18),
    )
    ylim = (
        min(min(s - ss for s, ss in zip(all_s, all_ss)) - pad, -0.18),
        max(max(s + ss for s, ss in zip(all_s, all_ss)) + pad,  0.18),
    )
    return xlim, ylim


# === HELPERS: PLOTTING ===

def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  3,
        "ytick.major.size":  3,
    })


def draw_quadrant_backgrounds(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    x0, x1 = xlim
    y0, y1 = ylim
    ax.add_patch(mpatches.Rectangle((0,  0),   x1,  y1,  facecolor="#fce8e8", edgecolor="none", zorder=0))
    ax.add_patch(mpatches.Rectangle((x0, 0),  -x0,  y1,  facecolor="#e8eef8", edgecolor="none", zorder=0))
    ax.add_patch(mpatches.Rectangle((x0, y0), -x0, -y0,  facecolor="#e8f4e8", edgecolor="none", zorder=0))
    ax.add_patch(mpatches.Rectangle((0,  y0),  x1, -y0,  facecolor="#fef8e0", edgecolor="none", zorder=0))

    ax.axhline(0, color="#888", lw=0.9, zorder=1)
    ax.axvline(0, color="#888", lw=0.9, zorder=1)

    pad = 0.06
    kw  = dict(fontsize=8.5, color="#aaa", ha="center", va="center", style="italic")
    if x1 > 0 and y1 > 0:
        ax.text(x1 - pad, y1 - pad, "Right-Auth", **kw)
    if x0 < 0 and y1 > 0:
        ax.text(x0 + pad, y1 - pad, "Left-Auth",  **kw)
    if x0 < 0 and y0 < 0:
        ax.text(x0 + pad, y0 + pad, "Left-Lib",   **kw)
    if x1 > 0 and y0 < 0:
        ax.text(x1 - pad, y0 + pad, "Right-Lib",  **kw)


def draw_moce_ellipse(
    ax: plt.Axes,
    stats: dict[str, dict[str, float]],
) -> None:
    """Draw a ±1σ spread ellipse for the MoCE model only."""
    v   = stats[MOCE_KEY]
    cfg = MODEL_CFG[MOCE_KEY]
    shared = dict(
        xy=(v["econ"], v["soc"]),
        width=2 * v["std_e"],
        height=2 * v["std_s"],
    )
    ax.add_patch(Ellipse(**shared, facecolor=cfg["color"], edgecolor="none",
                         alpha=0.12, linewidth=0, zorder=2))
    ax.add_patch(Ellipse(**shared, facecolor="none", edgecolor=cfg["color"],
                         alpha=0.45, linewidth=0.9, linestyle="--", zorder=2))


def draw_centroids(ax: plt.Axes, stats: dict[str, dict[str, float]]) -> None:
    for model, v in stats.items():
        cfg = MODEL_CFG[model]
        ax.scatter(
            v["econ"], v["soc"],
            s=cfg["ms"], c=cfg["color"], marker=cfg["marker"],
            edgecolors="white", linewidths=cfg["lw"],
            zorder=cfg["zorder"],
        )


def draw_labels(ax: plt.Axes, stats: dict[str, dict[str, float]]) -> None:
    arrow_props = dict(arrowstyle="-", linewidth=0.7, color="#999999")
    for model, v in stats.items():
        cfg     = MODEL_CFG[model]
        is_moce = model == MOCE_KEY
        lx, ly  = cfg["label_xy"]
        ax.annotate(
            cfg["label"],
            xy=(v["econ"], v["soc"]),
            xytext=(lx, ly),
            xycoords="data",
            textcoords="data",
            fontsize=9.5 if is_moce else 8.5,
            fontweight="bold" if is_moce else "normal",
            color=cfg["color"],
            ha="left",
            va="center",
            zorder=6,
            arrowprops=arrow_props if cfg["arrow"] else None,
        )


def build_legend_handles() -> list[mlines.Line2D]:
    handles = []
    for _, cfg in MODEL_CFG.items():
        h = mlines.Line2D(
            [], [],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle="None",
            markersize=11 if cfg["marker"] == "*" else 7,
            label=cfg["label"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        handles.append(h)
    return handles


# === MAIN ===

def main() -> None:
    args  = parse_args()
    data  = load_results(args.results)
    stats = extract_model_stats(data)
    xlim, ylim = compute_axis_limits(stats, pad=AXIS_PAD)

    apply_paper_style()
    _, ax = plt.subplots(figsize=(6.5, 5.5))

    draw_quadrant_backgrounds(ax, xlim, ylim)
    draw_moce_ellipse(ax, stats)
    draw_centroids(ax, stats)
    draw_labels(ax, stats)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Economic axis  (← Left · Right →)", fontsize=10, labelpad=6)
    ax.set_ylabel("Social axis  (← Libertarian · Authoritarian →)", fontsize=10, labelpad=6)
    ax.set_title(
        "Political compass positions of evaluated models\n"
        r"(global centroid across 52 prompts × 10 responses; MoCE ellipse = ±1σ)",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax.legend(
        handles=build_legend_handles(),
        fontsize=8,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handletextpad=0.5,
        labelspacing=0.4,
    )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
