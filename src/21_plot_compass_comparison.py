#!/usr/bin/env python3
"""
Script 21 — Paper-ready political compass figure from script-20 results.

Reads:  data/evaluation/compass_comparison/results.json
Writes: docs/fig_compass_comparison.pdf
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data/evaluation/compass_comparison/results.json"
OUT     = ROOT / "docs/fig_compass_comparison.png"

# ── paper style ───────────────────────────────────────────────────────────────
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

# ── model display config ───────────────────────────────────────────────────────
# label_xy: absolute data-coords for the annotation text
# arrow: whether to draw a leader line from text → marker
MODEL_CFG = {
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

# ── load data ─────────────────────────────────────────────────────────────────
data = json.load(open(RESULTS))

models_data = {}
for model, v in data["models"].items():
    gc   = v["global_centroid"]
    econ = [p["centroid"]["economic"] for p in v["per_prompt"]]
    soc  = [p["centroid"]["social"]   for p in v["per_prompt"]]
    n    = len(econ)
    me   = sum(econ) / n
    ms_  = sum(soc)  / n
    std_e = math.sqrt(sum((x - me) ** 2 for x in econ) / n)
    std_s = math.sqrt(sum((x - ms_) ** 2 for x in soc)  / n)
    models_data[model] = {
        "econ":  gc["economic"],
        "soc":   gc["social"],
        "std_e": std_e,
        "std_s": std_s,
        "n":     n,
    }

# ── axis limits ───────────────────────────────────────────────────────────────
# pad around the centroid cluster so ellipses fit
all_e = [v["econ"] for v in models_data.values()]
all_s = [v["soc"]  for v in models_data.values()]
all_se = [v["std_e"] for v in models_data.values()]
all_ss = [v["std_s"] for v in models_data.values()]

PAD = 0.12
# always include (0, 0) so the compass centre is visible
xlim = (min(e - se for e, se in zip(all_e, all_se)) - PAD,
        max(max(e + se for e, se in zip(all_e, all_se)) + PAD, 0.18))
ylim = (min(min(s - ss for s, ss in zip(all_s, all_ss)) - PAD, -0.18),
        max(max(s + ss for s, ss in zip(all_s, all_ss)) + PAD, 0.18))

# ── draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.5))

# quadrant background tints (standard political compass convention)
x0, x1 = xlim; y0, y1 = ylim
ax.add_patch(mpatches.Rectangle((0,  0),  x1, y1,  facecolor="#fce8e8", edgecolor="none", zorder=0))  # right_auth
ax.add_patch(mpatches.Rectangle((x0, 0), -x0, y1,  facecolor="#e8eef8", edgecolor="none", zorder=0))  # left_auth
ax.add_patch(mpatches.Rectangle((x0, y0),-x0, -y0, facecolor="#e8f4e8", edgecolor="none", zorder=0))  # left_lib
ax.add_patch(mpatches.Rectangle((0,  y0), x1, -y0, facecolor="#fef8e0", edgecolor="none", zorder=0))  # right_lib

# axis dividers
ax.axhline(0, color="#888", lw=0.9, zorder=1)
ax.axvline(0, color="#888", lw=0.9, zorder=1)

# quadrant corner labels (only draw the corners that are within the plot range)
kw_q = dict(fontsize=8.5, color="#aaa", ha="center", va="center", style="italic")
_lbl_pad = 0.06
if x1 > 0 and y1 > 0:
    ax.text(x1 - _lbl_pad, y1 - _lbl_pad, "Right-Auth", **kw_q)
if x0 < 0 and y1 > 0:
    ax.text(x0 + _lbl_pad, y1 - _lbl_pad, "Left-Auth",  **kw_q)
if x0 < 0 and y0 < 0:
    ax.text(x0 + _lbl_pad, y0 + _lbl_pad, "Left-Lib",   **kw_q)
if x1 > 0 and y0 < 0:
    ax.text(x1 - _lbl_pad, y0 + _lbl_pad, "Right-Lib",  **kw_q)

# ±1σ ellipse for MoCE only
moce_key = "run_moce"
v_moce   = models_data[moce_key]
cfg_moce = MODEL_CFG[moce_key]
ax.add_patch(Ellipse(
    xy=(v_moce["econ"], v_moce["soc"]),
    width=2 * v_moce["std_e"], height=2 * v_moce["std_s"],
    facecolor=cfg_moce["color"], edgecolor="none",
    alpha=0.12, linewidth=0, zorder=2,
))
ax.add_patch(Ellipse(
    xy=(v_moce["econ"], v_moce["soc"]),
    width=2 * v_moce["std_e"], height=2 * v_moce["std_s"],
    facecolor="none", edgecolor=cfg_moce["color"],
    alpha=0.45, linewidth=0.9, linestyle="--", zorder=2,
))

# model centroids
for model, v in models_data.items():
    cfg = MODEL_CFG[model]
    ax.scatter(
        v["econ"], v["soc"],
        s=cfg["ms"], c=cfg["color"], marker=cfg["marker"],
        edgecolors="white", linewidths=cfg["lw"],
        zorder=cfg["zorder"],
    )

# labels with leader arrows where needed
ARROW_PROPS = dict(
    arrowstyle="-",
    linewidth=0.7,
    color="#999999",
)
for model, v in models_data.items():
    cfg    = MODEL_CFG[model]
    is_moce = (model == "run_moce")
    lx, ly  = cfg["label_xy"]
    ax.annotate(
        cfg["label"],
        xy=(v["econ"], v["soc"]),
        xytext=(lx, ly),
        xycoords="data",
        textcoords="data",
        fontsize=8.5 if not is_moce else 9.5,
        fontweight="bold" if is_moce else "normal",
        color=cfg["color"],
        ha="left",
        va="center",
        zorder=6,
        arrowprops=ARROW_PROPS if cfg["arrow"] else None,
    )

# ── axes decoration ───────────────────────────────────────────────────────────
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("Economic axis  (← Left · Right →)", fontsize=10, labelpad=6)
ax.set_ylabel("Social axis  (← Libertarian · Authoritarian →)", fontsize=10, labelpad=6)
ax.set_title(
    "Political compass positions of evaluated models\n"
    r"(global centroid across 52 prompts × 10 responses; MoCE ellipse = ±1σ)",
    fontsize=10.5, fontweight="bold", pad=10,
)

# legend — markers only
legend_handles = []
for model, cfg in MODEL_CFG.items():
    h = mlines.Line2D(
        [], [],
        color=cfg["color"],
        marker=cfg["marker"],
        linestyle="None",
        markersize=7 if cfg["marker"] != "*" else 11,
        label=cfg["label"].replace("\n", " "),
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    legend_handles.append(h)

ax.legend(
    handles=legend_handles,
    fontsize=8, loc="lower right",
    frameon=True, framealpha=0.92, edgecolor="#cccccc",
    handletextpad=0.5, labelspacing=0.4,
)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.show()
