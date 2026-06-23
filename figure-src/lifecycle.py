"""Generate the auditable Lifecycle figure.

One typed two-layer decision graph sits at the center as a horizontal row of
step nodes. Teal arrows carry the observed execution order; curved arcs above
the row carry the recovered dependency edges, colored by source (observed,
declared, inferred). Three lifecycle pillars below the graph (PRE, LIVE,
POST) are the attach points that read the one graph.

Run with the project interpreter:
    python figure-src/lifecycle.py
Writes assets/lifecycle.png and docs/assets/lifecycle.png (PNG only, dpi 240).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- palette ---------------------------------------------------------------
INK = "#1f2733"          # primary text
MUTE = "#6b7280"         # secondary text / annotations
FAINT = "#9aa3af"        # faintest annotations

TEAL = "#0d9488"         # execution edges (teal)
OBSERVED = "#e8703a"     # dependency: observed (solid)
DECLARED = "#e8a23a"     # dependency: declared (dashed)
INFERRED = "#94a3b8"     # dependency: inferred (dotted)

# soft pastel node fills, left to right along the run
NODE_FILLS = ["#d4ecec", "#d4ecec", "#dcefe2", "#e7dcf5", "#e7dcf5"]
NODE_RING = "#3f4a59"

# three lifecycle pillars
PRE_EDGE, PRE_FILL, PRE_INK = "#4f46e5", "#eef0ff", "#312e9e"      # indigo
RT_EDGE, RT_FILL, RT_INK = "#0d9488", "#e3f6f3", "#0b5f57"         # teal
POST_EDGE, POST_FILL, POST_INK = "#e8542e", "#fdece6", "#9e3216"   # coral

FONT = "Arial"

fig, ax = plt.subplots(figsize=(12.6, 7.6))
ax.set_xlim(0, 126)
ax.set_ylim(0, 76)
ax.axis("off")


def text(cx, cy, s, color=INK, size=10, weight="normal", ha="center",
         va="center", style="normal", spacing=1.4):
    return ax.text(cx, cy, s, color=color, fontsize=size, fontweight=weight,
                   ha=ha, va=va, family=FONT, fontstyle=style,
                   linespacing=spacing)


def exec_arrow(x1, x2, y, color=TEAL, lw=3.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=17,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, capstyle="round"))


def arc(x1, x2, base_y, height, color, style, lw=2.1):
    """A smooth dependency arc bowed upward from node x1 to node x2."""
    midx = (x1 + x2) / 2.0
    top = base_y + height
    verts = [(x1, base_y), (x1 + (midx - x1) * 0.45, top),
             (midx, top), (x2 - (x2 - midx) * 0.45, top), (x2, base_y)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
    dash = {"solid": "solid", "dashed": (0, (5, 3)), "dotted": (0, (1, 2.6))}[style]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none",
                           edgecolor=color, lw=lw, linestyle=dash,
                           joinstyle="round", capstyle="round"))


def pillar(cx, top_y, w, h, edge, fill, ink, title, body, attach_x, attach_y):
    """A lifecycle attach card sitting below the graph, joined to it by a stem."""
    x = cx - w / 2.0
    ax.add_patch(FancyBboxPatch(
        (x, top_y - h), w, h,
        boxstyle="round,pad=0,rounding_size=2.2",
        linewidth=2.0, edgecolor=edge, facecolor=fill))
    # accent bar on the left edge of the card
    ax.add_patch(FancyBboxPatch(
        (x + 1.2, top_y - h + 1.6), 1.5, h - 3.2,
        boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=0, edgecolor="none", facecolor=edge))
    text(x + 5.4, top_y - 3.9, title, color=ink, size=13.0, weight="bold",
         ha="left", va="center")
    text(x + 5.4, top_y - h + 3.0, body, color=ink, size=9.2, ha="left",
         va="bottom", spacing=1.55)
    # dotted stem from just above the card up to its attach point on the graph
    ax.add_patch(FancyArrowPatch(
        (cx, top_y + 0.6), (attach_x, attach_y),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.8,
        color=edge, linestyle=(0, (1.2, 1.7)),
        shrinkA=0, shrinkB=2))


# ===========================================================================
# Title
# ===========================================================================
text(6.0, 71.5, "One Decision Graph, Read at Three Points in a Run",
     color=INK, size=19.0, weight="bold", ha="left")
text(6.0, 67.3,
     "Recover the run as a typed two-layer graph, then attach to it before, during, and after.",
     color=MUTE, size=10.5, ha="left")

# ===========================================================================
# Legend (top right)
# ===========================================================================
lx, ly = 92.0, 71.6
leg_dy = 3.05
ax.add_patch(FancyArrowPatch(
    (lx, ly), (lx + 5.0, ly), arrowstyle="-|>", mutation_scale=13,
    color=TEAL, lw=2.8, shrinkA=0, shrinkB=0, capstyle="round"))
text(lx + 6.6, ly, "execution edge  (observed order)", color=INK, size=8.8,
     ha="left")
for i, (col, sty, lab) in enumerate([
        (OBSERVED, "solid", "dependency: observed"),
        (DECLARED, (0, (5, 3)), "dependency: declared"),
        (INFERRED, (0, (1, 2.6)), "dependency: inferred")], start=1):
    yy = ly - leg_dy * i
    ax.plot([lx, lx + 5.0], [yy, yy], color=col, lw=2.4, linestyle=sty,
            solid_capstyle="round", dash_capstyle="round")
    text(lx + 6.6, yy, lab, color=INK, size=8.8, ha="left")

# ===========================================================================
# Central graph: a horizontal row of step nodes (the hero)
# ===========================================================================
node_y = 41.0
node_r = 2.9
xs = [24, 42, 60, 78, 96]
roles = ["plan", "retrieve", "act", "act", "check"]

# left-side two-layer annotation
text(4.0, 50.5, "dependency layer", color=OBSERVED, size=10.5, weight="bold",
     ha="left", style="italic")
text(4.0, 47.7, "recovered, keyed by source", color=MUTE, size=8.5, ha="left")
text(4.0, 36.4, "execution layer", color=TEAL, size=10.5, weight="bold",
     ha="left", style="italic")
text(4.0, 33.6, "read from the trace", color=MUTE, size=8.5, ha="left")

# graph label, centered above the dependency arc fan
text(60, 61.6, "The Typed Two-Layer Decision Graph", color=INK, size=13.0,
     weight="bold")
text(60, 58.4, "execution edges over dependency edges, one model for the run",
     color=MUTE, size=9.0)

# execution arrows between consecutive nodes (drawn first, behind nodes)
for x1, x2 in zip(xs[:-1], xs[1:]):
    exec_arrow(x1 + node_r, x2 - node_r, node_y)

# dependency arcs above the row, keyed by source
arc_base = node_y + node_r - 0.2
arc(xs[0], xs[1], arc_base, 4.2, OBSERVED, "solid")
arc(xs[1], xs[2], arc_base, 4.2, OBSERVED, "solid")
arc(xs[2], xs[3], arc_base, 4.2, OBSERVED, "solid")
arc(xs[3], xs[4], arc_base, 4.2, OBSERVED, "solid")
arc(xs[0], xs[2], arc_base, 7.6, DECLARED, "dashed")
arc(xs[2], xs[4], arc_base, 7.6, DECLARED, "dashed")
arc(xs[1], xs[4], arc_base, 11.0, INFERRED, "dotted")

# nodes on top
for x, fill, role in zip(xs, NODE_FILLS, roles):
    ax.add_patch(Circle((x, node_y), node_r, facecolor=fill,
                        edgecolor=NODE_RING, linewidth=1.7, zorder=5))
    text(x, node_y - node_r - 2.4, role, color=INK, size=9.2, va="top")

# arXiv tag, parked past the last node at node-row height (clear of the stems)
ax.add_patch(FancyBboxPatch(
    (105.4, 39.5), 14.4, 3.1, boxstyle="round,pad=0,rounding_size=1.55",
    linewidth=1.1, edgecolor=FAINT, facecolor="#f5f6f8"))
text(112.6, 41.05, "arXiv:2606.22741", color=MUTE, size=8.4, weight="bold")
text(112.6, 36.9, "GRADE (Zhao 2026)", color=FAINT, size=7.6)

# ===========================================================================
# Three lifecycle pillars (attach points) below the graph
# ===========================================================================
pillar_top = 19.6
ph = 13.6
pw = 35.0
stem_top = node_y - node_r - 4.4

pillar(24, pillar_top, pw, ph, PRE_EDGE, PRE_FILL, PRE_INK,
       "PRE",
       "Lint a declared plan\nbefore deploy.",
       attach_x=xs[0], attach_y=stem_top)

pillar(63, pillar_top, pw, ph, RT_EDGE, RT_FILL, RT_INK,
       "LIVE",
       "Replay and recover\na live decision.",
       attach_x=xs[2], attach_y=stem_top)

pillar(102, pillar_top, pw, ph, POST_EDGE, POST_FILL, POST_INK,
       "POST",
       "Rank a finished run,\nname the keystone.",
       attach_x=xs[4], attach_y=stem_top)

# ===========================================================================
# Save (PNG only, both targets)
# ===========================================================================
fig.subplots_adjust(left=0.012, right=0.988, top=0.99, bottom=0.01)
for d in (os.path.join(HERE, "..", "assets"),
          os.path.join(HERE, "..", "docs", "assets")):
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "lifecycle.png"), dpi=240,
                bbox_inches="tight", facecolor="white", pad_inches=0.14)
print("wrote lifecycle.png to assets/ and docs/assets/")
