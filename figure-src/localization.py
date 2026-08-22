"""Generate the method-labeled CatchBench localization figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
METRICS = ("Top-1", "Top-3", "MRR")
METHODS = (
    ("random", (0.119, 0.346, 0.324), "#c7c7c7"),
    ("position prior", (0.159, 0.516, 0.407), "#b9d9cc"),
    ("auditable (blast share)", (0.159, 0.516, 0.407), "#e8a23a"),
    ("exec-rank (sup.)", (0.211, 0.614, 0.454), "#3c5488"),
)


fig, ax = plt.subplots(figsize=(7.4, 4.2))
x = np.arange(len(METRICS))
width = 0.19

for offset, (label, values, color) in enumerate(METHODS):
    bars = ax.bar(
        x + (offset - 1.5) * width,
        values,
        width,
        label=label,
        color=color,
        edgecolor="#20242a",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)

ax.set_title(
    "Supervised exec-rank leads structural baselines on Top-3",
    loc="left",
    fontsize=14,
    fontweight="bold",
    pad=18,
)
ax.text(
    0,
    1.02,
    "Who&When, CatchBench; auditable blast share ties the position prior",
    transform=ax.transAxes,
    fontsize=9.5,
    color="#666666",
)
ax.set_ylabel("score (higher is better)")
ax.set_xticks(x, METRICS)
ax.set_ylim(0, 0.72)
ax.grid(axis="y", color="#dddddd", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, ncols=2, loc="upper left")
fig.tight_layout()

for directory in (ROOT / "assets", ROOT / "docs" / "assets"):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / "grade-localization.png", dpi=200, facecolor="white")

print("wrote grade-localization.png to assets/ and docs/assets/")
