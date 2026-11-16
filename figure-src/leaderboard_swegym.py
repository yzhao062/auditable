"""Generate the detect/swegym leaderboard bar chart for the README.

Reads the CatchBench reference ROC-AUC values from
``src/auditable/bench/golden/detect__swegym.json`` and resolves each method's
type and control role from the ``auditable`` method registry.

Run with the project interpreter:
    python figure-src/leaderboard_swegym.py
Writes assets/leaderboard-swegym.png (PNG only, dpi 240).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import auditable.methods as methods_mod

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "src" / "auditable" / "bench" / "golden" / "detect__swegym.json"
OUT_PATH = ROOT / "assets" / "leaderboard-swegym.png"

# ---- palette ---------------------------------------------------------------
INK = "#1f2733"          # primary text
MUTE = "#4b5563"         # secondary text / subtitle / reference label
REF_LINE = "#64748b"     # random = 0.5 dashed reference line
GRID = "#e5e7eb"         # light x-axis gridlines
SPINE = "#cbd5e1"        # subtle axis spines

TYPE_ORDER = (
    "Graph detector",
    "Supervised",
    "PyOD detector",
    "Baseline",
)
TYPE_COLORS = {
    "Graph detector": "#6d28d9",
    "Supervised": "#0f766e",
    "PyOD detector": "#d97706",
    "Baseline": "#94a3b8",
}

FONT = "Arial"
MONO_FONT = "DejaVu Sans Mono"
DPI = 240


def _tint_white(hex_color: str, factor: float = 0.58) -> tuple[float, float, float]:
    """Blend a hex color toward white for control bars."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (
        r + (1.0 - r) * factor,
        g + (1.0 - g) * factor,
        b + (1.0 - b) * factor,
    )


def load_leaderboard_rows() -> list[dict]:
    """Load golden detect/swegym rows and attach method type & control flag."""
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    entries = []
    for method_id, row in golden["rows"].items():
        card = methods_mod.get_method(method_id).card
        method_type = methods_mod._method_type(card)
        is_control = getattr(card, "benchmark_role", None) == "control"
        entries.append({
            "method_id": method_id,
            "roc_auc": float(row["roc_auc"]),
            "type": method_type,
            "is_control": is_control,
        })

    entries.sort(key=lambda item: (-item["roc_auc"], item["method_id"]))
    return entries


def build_chart(entries: list[dict]) -> plt.Figure:
    """Render the horizontal leaderboard bar chart."""
    plt.rcParams["font.family"] = FONT
    plt.rcParams["hatch.linewidth"] = 0.9

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = list(range(len(entries)))
    scores = [e["roc_auc"] for e in entries]
    labels = [
        f"{e['method_id']} (control)" if e["is_control"] else e["method_id"]
        for e in entries
    ]

    bar_height = 0.68
    for y, entry in zip(y_pos, entries):
        base_color = TYPE_COLORS[entry["type"]]
        if entry["is_control"]:
            ax.barh(
                y,
                entry["roc_auc"],
                height=bar_height,
                color=_tint_white(base_color, factor=0.58),
                edgecolor=base_color,
                linewidth=1.1,
                hatch="///",
                zorder=3,
            )
        else:
            ax.barh(
                y,
                entry["roc_auc"],
                height=bar_height,
                color=base_color,
                edgecolor=base_color,
                linewidth=0.6,
                zorder=3,
            )

        ax.text(
            entry["roc_auc"] + 0.007,
            y,
            f"{entry['roc_auc']:.3f}",
            va="center",
            ha="left",
            fontsize=10.0,
            color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
            zorder=5,
        )

    # Thin dashed vertical reference line at random = 0.5
    ax.axvline(
        0.5,
        color=REF_LINE,
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=2,
    )
    ax.text(
        0.506,
        len(entries) - 1,
        "random = 0.5",
        va="center",
        ha="left",
        fontsize=10.0,
        color=MUTE,
        zorder=4,
    )

    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5, fontfamily=MONO_FONT, color=INK)
    ax.tick_params(axis="y", length=0, pad=7)

    ax.set_xlim(0.25, 0.91)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.tick_params(axis="x", labelsize=10.5, colors=MUTE, length=4, color=SPINE)
    ax.set_xlabel("ROC-AUC", fontsize=11.0, color=INK, labelpad=7)

    ax.set_ylim(len(entries) - 0.35, -0.75)

    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)

    # Single legend for method types, placed cleanly above the plot area
    legend_handles = [
        Patch(facecolor=TYPE_COLORS[t], edgecolor=TYPE_COLORS[t], label=t)
        for t in TYPE_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=4,
        frameon=False,
        fontsize=10.2,
        handlelength=1.25,
        handleheight=0.9,
        columnspacing=1.5,
        borderaxespad=0.0,
    )

    # Small subtitle line at top left (no big title)
    ax.text(
        0.0,
        1.095,
        "detect/swegym · did the run fail? · 376 SWE-Gym runs, 188 failed",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        color=MUTE,
    )

    fig.tight_layout()
    return fig


def main() -> None:
    entries = load_leaderboard_rows()
    for rank, entry in enumerate(entries, start=1):
        suffix = " (control)" if entry["is_control"] else ""
        print(f"{rank:2d}  {entry['type']:15s}  {entry['method_id']:22s}  {entry['roc_auc']:.3f}{suffix}")

    fig = build_chart(entries)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_PATH,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.14,
    )
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
