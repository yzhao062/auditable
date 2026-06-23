"""Chart generation for the human-facing audit-report PDF.

This module renders the five audit-report charts to PNG bytes. The charts are for
the PDF (the human-facing artifact), never for the agent-facing Markdown: an agent
consumes structured text, not raster images, so ``AuditReport.to_markdown`` embeds
no figure from here.

Each function takes one or more of the STABLE upstream report objects (the PRE
:class:`~auditable.graph.pre.PreReport`, the POST
:class:`~auditable.analysis.AnalysisReport`, and the LIVE
:class:`~auditable.chain.Verdict` sequence) plus a few derived primitives, and
returns the PNG bytes for one chart. The functions are small and pure: they read
the passed objects, draw, and return bytes. They hold no state and write no files.

The heavy dependencies (matplotlib and networkx) are part of the optional
``report`` extra and are imported lazily inside :func:`_pyplot` and
:func:`_networkx`. Importing this module with the extra absent still succeeds;
only calling a figure function triggers the guarded import, which raises a clear
``pip install auditable[report]`` error when the extra is missing. This mirrors how
:mod:`auditable.graph` guards the ``graph`` extra.

matplotlib runs headless: :func:`_pyplot` calls ``matplotlib.use("Agg")`` before it
imports ``matplotlib.pyplot``, so no figure function ever reaches for an interactive
display, and the same code path works in CI on Windows and Linux. Each figure is
rendered to an in-memory PNG via :func:`_fig_to_png`, which closes the figure with
``plt.close`` to bound memory, so no temporary files are produced.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "REPORT_EXTRA_HINT",
    "blast_share_figure",
    "coverage_gauge_figure",
    "decision_graph_figure",
    "recovery_figure",
    "findings_by_severity_figure",
]

REPORT_EXTRA_HINT = (
    "the audit-report charts require the 'report' extra: pip install auditable[report]"
)

# A small, consistent palette so the five charts read as one report. The four
# FixAction outcomes keep a stable colour each (allow green, human-review amber,
# rollback orange, block red), so the same outcome looks the same across charts.
_COLOR_PRIMARY = "#2b6cb0"  # blast-share bars, dependency edges
_COLOR_KEYSTONE = "#c53030"  # the highlighted keystone bar / node marker
_COLOR_WITHHELD = "#a0aec0"  # a withheld (no_score) bar, grey so it never reads as zero
_COLOR_OBSERVED = "#2f855a"
_COLOR_DECLARED = "#dd6b20"
_COLOR_INFERRED = "#a0aec0"
_FIXACTION_COLORS = {
    "allow": "#2f855a",
    "human_review": "#d69e2e",
    "rollback": "#dd6b20",
    "block": "#c53030",
}
_SOURCE_COLORS = {
    "PRE": "#3182ce",
    "LIVE": "#dd6b20",
    "POST": "#805ad5",
}


def _pyplot():
    """Return ``matplotlib.pyplot`` after forcing the headless Agg backend.

    ``matplotlib.use("Agg")`` is called BEFORE ``matplotlib.pyplot`` is imported, so
    the import never selects an interactive backend and the figure functions run in
    headless CI. The import is lazy and guarded: with the ``report`` extra absent it
    raises a clear ``pip install auditable[report]`` error, mirroring the
    ``auditable.graph`` extra guard.
    """
    try:
        import matplotlib
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(REPORT_EXTRA_HINT) from exc
    matplotlib.use("Agg")  # headless; must precede the pyplot import
    import matplotlib.pyplot as plt

    return plt


def _networkx():
    """Return ``networkx`` lazily, with the same guarded ``report`` extra error."""
    try:
        import networkx as nx
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(REPORT_EXTRA_HINT) from exc
    return nx


def _fig_to_png(fig, *, dpi: int = 150) -> bytes:
    """Render a matplotlib figure to PNG bytes, then close it to bound memory.

    The figure is saved to an in-memory buffer (no temporary file), and
    ``plt.close(fig)`` releases it so a report that draws five figures does not leak
    five live figures. Returns the raw PNG bytes, ready to hand to ``FPDF.image``.
    """
    import io

    plt = _pyplot()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _no_data_figure(message: str) -> bytes:
    """A small placeholder PNG that states a pillar is absent.

    A missing pillar is drawn as an explicit "no data" panel rather than an empty or
    fabricated-zero chart, mirroring the upstream no-score honesty boundary (a
    withheld value is not a zero value).
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 1.6))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
        color="#4a5568",
        wrap=True,
    )
    return _fig_to_png(fig)


# --- chart 1: blast-share ranking --------------------------------------------


def blast_share_figure(post: Optional[Any]) -> bytes:
    """Horizontal bar of the POST blast-share ranking, keystone highlighted.

    Reads :class:`~auditable.analysis.AnalysisReport`: each ``ranked``
    :class:`~auditable.analysis.DecisionRisk` becomes one bar of its normalized
    transitive blast share (highest first), and the blast-radius keystone bar is
    drawn in the keystone colour and annotated with its "N of M other steps rest on
    it" count. In a no-score state the scores are ``None``; those bars are drawn grey
    at zero length and labelled "withheld", so a withheld score never reads as a real
    zero. A ``None`` report renders the explicit no-POST-data panel.
    """
    if post is None:
        return _no_data_figure("No POST data: run analyze_run to rank blast share.")

    plt = _pyplot()
    ranked = list(getattr(post, "ranked", []) or [])
    n_steps = int(getattr(post, "n_steps", 0) or 0)
    keystone = getattr(post, "keystone", None)
    keystone_idx = getattr(keystone, "idx", None) if keystone is not None else None

    if not ranked:
        return _no_data_figure("No POST data: the run has no ranked steps.")

    # Draw highest blast share at the top: reverse so barh's bottom-up order reads
    # top-down by rank.
    rows = list(reversed(ranked))
    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    withheld_flags: List[bool] = []
    for d in rows:
        idx = getattr(d, "idx", "?")
        label = getattr(d, "label", "") or getattr(d, "kind", "step")
        score = getattr(d, "score", None)
        labels.append(f"step {idx}: {label}")
        withheld = score is None
        withheld_flags.append(withheld)
        values.append(0.0 if withheld else float(score))
        if withheld:
            colors.append(_COLOR_WITHHELD)
        elif idx == keystone_idx:
            colors.append(_COLOR_KEYSTONE)
        else:
            colors.append(_COLOR_PRIMARY)

    height = max(2.2, 0.45 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, height))
    y = range(len(rows))
    ax.barh(list(y), values, color=colors, edgecolor="#2d3748", linewidth=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("normalized transitive blast share")
    ax.set_title("Blast-Share Ranking (POST)", fontsize=12, fontweight="bold")

    any_withheld = any(withheld_flags)
    if any_withheld:
        # A withheld bar is explicitly labelled at the axis origin.
        for i, withheld in enumerate(withheld_flags):
            if withheld:
                ax.text(0.005, i, "withheld", va="center", ha="left", fontsize=7, color="#4a5568")
        ax.set_xlim(0, 1.0)
    else:
        top = max(values) if values else 1.0
        ax.set_xlim(0, max(0.05, top * 1.25))

    # Annotate the keystone bar with its downstream count, if it is scored.
    if keystone is not None and getattr(keystone, "score", None) is not None and n_steps > 1:
        k_score = float(keystone.score)
        n_down = round(k_score * (n_steps - 1))
        for i, d in enumerate(rows):
            if getattr(d, "idx", None) == keystone_idx and not withheld_flags[i]:
                ax.text(
                    values[i],
                    i,
                    f"  keystone: {n_down} of {n_steps - 1} rest on it",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color=_COLOR_KEYSTONE,
                    fontweight="bold",
                )
                break

    ax.invert_yaxis()  # keep the highest-rank row visually at the top after reversal
    return _fig_to_png(fig)


# --- chart 2: preflight / coverage gauge -------------------------------------


def _coverage_view(source: Any) -> Optional[Dict[str, Any]]:
    """Normalize a PRE ``PreflightCoverage`` or POST ``EdgeCoverage`` to one dict.

    Returns a dict with the grade-mix counts, ``rho``, ``observed_fraction``,
    ``n_dep_edges``, and an optional ``no_score_reason`` (present on the PRE preflight
    view, absent on the POST edge coverage). Returns ``None`` when the source carries
    no recognizable coverage shape.
    """
    if source is None:
        return None
    n_dep = getattr(source, "n_dep_edges", None)
    rho = getattr(source, "rho", None)
    observed_fraction = getattr(source, "observed_fraction", None)
    if n_dep is None or rho is None or observed_fraction is None:
        return None

    # PRE PreflightCoverage carries flat observed/declared/inferred ints; POST
    # EdgeCoverage carries a by_grade Grade -> int mapping.
    if hasattr(source, "observed") and hasattr(source, "declared"):
        observed = int(getattr(source, "observed", 0) or 0)
        declared = int(getattr(source, "declared", 0) or 0)
        inferred = int(getattr(source, "inferred", 0) or 0)
    else:
        by_grade = getattr(source, "by_grade", {}) or {}
        observed = declared = inferred = 0
        for grade, count in by_grade.items():
            name = getattr(grade, "value", str(grade))
            if name == "observed":
                observed = int(count)
            elif name == "declared":
                declared = int(count)
            elif name == "inferred":
                inferred = int(count)

    return {
        "n_dep_edges": int(n_dep),
        "rho": float(rho),
        "observed_fraction": float(observed_fraction),
        "observed": observed,
        "declared": declared,
        "inferred": inferred,
        "no_score_reason": getattr(source, "no_score_reason", None),
    }


def coverage_gauge_figure(
    coverage: Optional[Any],
    *,
    touch: Optional[Any] = None,
) -> bytes:
    """Segmented coverage bar (observed / declared / inferred) with rho annotated.

    ``coverage`` is a PRE :class:`~auditable.graph.pre.PreflightCoverage` or a POST
    :class:`~auditable.graph.session.EdgeCoverage`; both reduce to the same grade-mix
    plus ``rho`` and ``observed_fraction`` via :func:`_coverage_view`. The top bar
    stacks the dependency-edge grade mix, annotated with ``rho``,
    ``observed_fraction``, and (when present) the exact ``no_score_reason``, so a
    withheld runtime score is explained rather than implied. When a PRE
    :class:`~auditable.graph.pre.ResourceTouchCompleteness` is passed as ``touch``, a
    second sub-bar shows reads / writes / edges identified versus missing a resource
    identity. A ``None`` coverage renders the explicit no-coverage panel.
    """
    view = _coverage_view(coverage)
    if view is None:
        return _no_data_figure("No coverage data: no PRE or POST coverage to gauge.")

    plt = _pyplot()
    have_touch = touch is not None and getattr(touch, "n_reads", None) is not None
    n_panels = 2 if have_touch else 1
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(7.5, 2.4 if n_panels == 1 else 3.6),
        squeeze=False,
    )
    ax = axes[0][0]

    # Top bar: the dependency-edge grade mix as a single stacked horizontal bar.
    segments: List[Tuple[str, int, str]] = [
        ("observed", view["observed"], _COLOR_OBSERVED),
        ("declared", view["declared"], _COLOR_DECLARED),
        ("inferred", view["inferred"], _COLOR_INFERRED),
    ]
    total = sum(c for _, c, _ in segments)
    left = 0.0
    if total > 0:
        for name, count, color in segments:
            if count <= 0:
                continue
            ax.barh(0, count, left=left, color=color, edgecolor="#2d3748", linewidth=0.5)
            ax.text(
                left + count / 2.0,
                0,
                f"{name}\n{count}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
            left += count
        ax.set_xlim(0, total)
    else:
        ax.text(0.5, 0, "no dependency edges", ha="center", va="center", fontsize=9, color="#4a5568")
        ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("dependency edges by grade")
    subtitle = (
        f"rho={view['rho']:.3f}   observed_fraction={view['observed_fraction']:.0%}"
        f"   ({view['n_dep_edges']} edge(s))"
    )
    ax.set_title("Preflight Coverage Gauge", fontsize=12, fontweight="bold")
    ax.annotate(
        subtitle,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=9,
        color="#2d3748",
    )
    reason = view.get("no_score_reason")
    if reason:
        ax.annotate(
            f"runtime no-score: {reason}",
            xy=(0.0, 0.0),
            xycoords="axes fraction",
            xytext=(0, -34),
            textcoords="offset points",
            fontsize=7.5,
            color="#4a5568",
            wrap=True,
        )

    # Optional sub-bar: resource-touch completeness (identified versus missing).
    if have_touch:
        ax2 = axes[1][0]
        categories = ["reads", "writes", "edges"]
        identified = [
            int(getattr(touch, "reads_with_id", 0) or 0),
            int(getattr(touch, "writes_with_id", 0) or 0),
            int(getattr(touch, "edges_with_id", 0) or 0),
        ]
        totals = [
            int(getattr(touch, "n_reads", 0) or 0),
            int(getattr(touch, "n_writes", 0) or 0),
            int(getattr(touch, "n_edges", 0) or 0),
        ]
        missing = [max(0, t - i) for t, i in zip(totals, identified)]
        ypos = range(len(categories))
        ax2.barh(list(ypos), identified, color=_COLOR_OBSERVED, label="identified", edgecolor="#2d3748", linewidth=0.5)
        ax2.barh(list(ypos), missing, left=identified, color=_COLOR_WITHHELD, label="missing id", edgecolor="#2d3748", linewidth=0.5)
        ax2.set_yticks(list(ypos))
        ax2.set_yticklabels(categories, fontsize=9)
        ax2.set_xlabel("resource-touch completeness (count)")
        ax2.legend(fontsize=7, loc="lower right")
        top = max(totals) if any(totals) else 1
        ax2.set_xlim(0, max(1, top))

    return _fig_to_png(fig)


# --- chart 3: two-layer decision graph ---------------------------------------


def _step_networkx(graph: Any):
    """Return a NetworkX step graph from either a SessionGraph or a ready graph.

    Accepts a :class:`~auditable.graph.session.SessionGraph` (calls
    ``to_networkx``), an already-projected NetworkX graph (used as is), or ``None``.
    Raises ``TypeError`` for anything else so a caller error fails loudly instead of
    drawing an empty graph.
    """
    if graph is None:
        return None
    if hasattr(graph, "to_networkx"):
        return graph.to_networkx()
    # Duck-type a NetworkX graph: it exposes nodes(data=...) and edges(data=...).
    if hasattr(graph, "nodes") and hasattr(graph, "edges"):
        return graph
    raise TypeError(
        "decision_graph_figure expects a SessionGraph or a NetworkX graph, "
        f"got {type(graph).__name__}"
    )


def decision_graph_figure(
    graph: Optional[Any],
    *,
    structural_keystone_idx: Optional[int] = None,
    blast_keystone_idx: Optional[int] = None,
    seed: int = 7,
) -> bytes:
    """Two-layer node-link drawing: dependency edges and control-flow edges.

    ``graph`` is a :class:`~auditable.graph.session.SessionGraph` (or its
    already-projected NetworkX graph). The drawing keeps the two edge classes
    visually distinct: ``depends_on`` dependency edges are solid blue arrows
    (dependent -> dependency), and ``handoff_to`` control-flow edges are dashed grey
    arrows (predecessor -> successor). The PRE structural keystone and the POST
    blast-radius keystone are marked with distinct node markers, kept as two named
    concepts. The layout is a seeded spring layout, so the rendered graph is
    deterministic across runs. A ``None`` graph renders the explicit no-graph panel.
    """
    G = _step_networkx(graph)
    if G is None:
        return _no_data_figure("No graph data: no session graph to draw.")

    nx = _networkx()
    plt = _pyplot()

    # Step nodes only (drop the agent and resource scaffold nodes).
    step_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") in ("decision", "tool_call")]
    if not step_nodes:
        return _no_data_figure("No graph data: the session graph has no step nodes.")

    dep_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("etype") == "depends_on" and u in step_nodes and v in step_nodes
    ]
    handoff_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("etype") == "handoff_to" and u in step_nodes and v in step_nodes
    ]

    # A deterministic layout over the step subgraph. A layered position map keyed on
    # idx gives a stable left-to-right reading order; spring_layout (seeded) refines
    # it without losing determinism.
    sub = nx.DiGraph()
    sub.add_nodes_from(step_nodes)
    sub.add_edges_from(handoff_edges)
    sub.add_edges_from(dep_edges)
    idx_of = {n: G.nodes[n].get("idx", 0) for n in step_nodes}
    init_pos = {n: (float(idx_of[n]), 0.0) for n in step_nodes}
    try:
        pos = nx.spring_layout(sub, pos=init_pos, seed=seed, k=1.2)
    except Exception:  # pragma: no cover - tiny graphs always lay out
        pos = init_pos

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Node colours / markers: structural keystone (PRE) and blast keystone (POST).
    s_node = f"step::{structural_keystone_idx}" if structural_keystone_idx is not None else None
    b_node = f"step::{blast_keystone_idx}" if blast_keystone_idx is not None else None

    base_nodes = [n for n in step_nodes if n != s_node and n != b_node]
    nx.draw_networkx_nodes(
        sub, pos, nodelist=base_nodes, node_color=_COLOR_PRIMARY,
        node_size=520, ax=ax, edgecolors="#2d3748",
    )
    if s_node in step_nodes:
        nx.draw_networkx_nodes(
            sub, pos, nodelist=[s_node], node_color="#dd6b20", node_shape="s",
            node_size=720, ax=ax, edgecolors="#2d3748",
            label="PRE structural keystone",
        )
    if b_node in step_nodes and b_node != s_node:
        nx.draw_networkx_nodes(
            sub, pos, nodelist=[b_node], node_color=_COLOR_KEYSTONE, node_shape="D",
            node_size=720, ax=ax, edgecolors="#2d3748",
            label="POST blast keystone",
        )

    # Two edge styles: dependency solid blue, control-flow dashed grey.
    if dep_edges:
        nx.draw_networkx_edges(
            sub, pos, edgelist=dep_edges, edge_color=_COLOR_PRIMARY, style="solid",
            arrows=True, arrowsize=14, width=1.6, ax=ax, connectionstyle="arc3,rad=0.08",
        )
    if handoff_edges:
        nx.draw_networkx_edges(
            sub, pos, edgelist=handoff_edges, edge_color="#718096", style="dashed",
            arrows=True, arrowsize=12, width=1.2, ax=ax, connectionstyle="arc3,rad=-0.08",
        )

    labels = {n: str(idx_of[n]) for n in step_nodes}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=9, font_color="white", ax=ax)

    # A compact manual legend for the two edge classes plus any keystone markers.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=_COLOR_PRIMARY, lw=1.8, label="depends_on (dependency)"),
        Line2D([0], [0], color="#718096", lw=1.2, linestyle="--", label="handoff_to (control flow)"),
    ]
    if s_node in step_nodes:
        legend_handles.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#dd6b20",
                   markersize=10, label="PRE structural keystone")
        )
    if b_node in step_nodes and b_node != s_node:
        legend_handles.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor=_COLOR_KEYSTONE,
                   markersize=10, label="POST blast keystone")
        )
    ax.legend(handles=legend_handles, fontsize=7, loc="upper center", ncol=2, frameon=True)
    ax.set_title("Two-Layer Decision Graph", fontsize=12, fontweight="bold")
    ax.axis("off")
    return _fig_to_png(fig)


# --- chart 4: snapshot-vs-live drift / recovery ------------------------------


def _verdict_counts(verdicts: Sequence[Any]) -> Dict[str, int]:
    """Count a Verdict sequence by FixAction value (allow/block/human_review/rollback)."""
    counts = {"allow": 0, "block": 0, "human_review": 0, "rollback": 0}
    for v in verdicts or ():
        action = getattr(v, "action", None)
        name = getattr(action, "value", None) or str(action)
        if name in counts:
            counts[name] += 1
    return counts


def recovery_figure(verdicts: Optional[Sequence[Any]]) -> bytes:
    """Bar of LIVE Verdict outcomes grouped by FixAction.

    ``verdicts`` is a sequence of :class:`~auditable.chain.Verdict`. The chart counts
    the four :class:`~auditable.chain.FixAction` outcomes (allow, human-review,
    rollback, block) as a bar, with rollback and block emphasized, so a reviewer sees
    where replay caught stale-state drift and routed a fix. An empty or ``None``
    sequence renders the explicit no-LIVE-data panel.
    """
    if not verdicts:
        return _no_data_figure("No LIVE data: no replay verdicts to chart.")

    plt = _pyplot()
    counts = _verdict_counts(verdicts)
    order = ["allow", "human_review", "rollback", "block"]
    labels = ["allow", "human review", "rollback", "block"]
    values = [counts[k] for k in order]
    colors = [_FIXACTION_COLORS[k] for k in order]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3748", linewidth=0.6)
    # Emphasize the drift-caught outcomes (rollback, block) with a heavier edge.
    for k, bar in zip(order, bars):
        if k in ("rollback", "block"):
            bar.set_linewidth(1.8)
            bar.set_edgecolor("#1a202c")
    ax.set_ylabel("verdict count")
    ax.set_title("Snapshot-vs-Live Drift and Recovery (LIVE)", fontsize=12, fontweight="bold")
    top = max(values) if values else 1
    ax.set_ylim(0, max(1, top) * 1.2)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    n_recovered = counts["rollback"] + counts["block"] + counts["human_review"]
    ax.annotate(
        f"{n_recovered} of {sum(values)} verdict(s) routed a fix (non-allow)",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=8.5,
        color="#2d3748",
    )
    return _fig_to_png(fig)


# --- chart 5: findings by severity and source --------------------------------


def _finding_field(row: Any, name: str, default: str) -> str:
    """Read ``name`` off a finding row, whether it is an attribute object or a mapping.

    The unified findings table may reach the chart as a row object (a dataclass with
    ``.source`` / ``.severity``) or as a mapping (a dict with the same keys). Reading
    both keeps the chart composable with whatever row shape the aggregator passes.
    """
    if isinstance(row, dict):
        value = row.get(name)
    else:
        value = getattr(row, name, None)
    return value if isinstance(value, str) and value else default


def findings_by_severity_figure(findings: Optional[Sequence[Any]]) -> bytes:
    """Grouped bar of finding counts split by source (PRE / LIVE / POST) and severity.

    ``findings`` is a sequence of rows, each carrying a ``source`` (one of ``PRE``,
    ``LIVE``, ``POST``) and a ``severity`` (e.g. ``warning``, ``block``), as
    either an attribute object or a mapping. Bars are grouped by source and coloured
    by severity, the at-a-glance triage view that complements the detailed findings
    table in the report. An empty or ``None`` sequence renders the explicit
    no-findings panel.
    """
    if not findings:
        return _no_data_figure("No findings: no PRE lints or non-allow verdicts to summarize.")

    plt = _pyplot()
    sources = ["PRE", "LIVE", "POST"]
    # Collect the severities present, in a stable, severity-desc-leaning order.
    severity_rank = {"block": 0, "rollback": 1, "human_review": 2, "warning": 3, "info": 4}
    present_sev: Dict[str, int] = {}
    counts: Dict[str, Dict[str, int]] = {s: {} for s in sources}
    other_sources: Dict[str, Dict[str, int]] = {}
    for f in findings:
        src = _finding_field(f, "source", "POST")
        sev = _finding_field(f, "severity", "warning")
        present_sev.setdefault(sev, severity_rank.get(sev, 5))
        bucket = counts.get(src)
        if bucket is None:
            bucket = other_sources.setdefault(src, {})
        bucket[sev] = bucket.get(sev, 0) + 1

    # Fold any unexpected source label into the chart so nothing is silently dropped.
    for src, bucket in other_sources.items():
        sources.append(src)
        counts[src] = bucket

    severities = sorted(present_sev, key=lambda s: (present_sev[s], s))
    severity_colors = {
        "block": "#c53030",
        "rollback": "#dd6b20",
        "human_review": "#d69e2e",
        "warning": "#3182ce",
        "info": "#a0aec0",
    }

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    n_sev = max(1, len(severities))
    group_width = 0.8
    bar_width = group_width / n_sev
    x_base = range(len(sources))
    any_bar = False
    for j, sev in enumerate(severities):
        offsets = [i - group_width / 2.0 + bar_width * (j + 0.5) for i in x_base]
        heights = [counts.get(s, {}).get(sev, 0) for s in sources]
        if any(heights):
            any_bar = True
        ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=sev,
            color=severity_colors.get(sev, "#718096"),
            edgecolor="#2d3748",
            linewidth=0.5,
        )

    ax.set_xticks(list(x_base))
    ax.set_xticklabels(sources, fontsize=9)
    ax.set_ylabel("finding count")
    ax.set_title("Findings by Source and Severity", fontsize=12, fontweight="bold")
    if any_bar:
        ax.legend(fontsize=7, title="severity", title_fontsize=7)
    # Integer y ticks: counts are whole numbers.
    max_h = max(
        (counts.get(s, {}).get(sev, 0) for s in sources for sev in severities),
        default=0,
    )
    ax.set_ylim(0, max(1, max_h) * 1.2)
    return _fig_to_png(fig)
