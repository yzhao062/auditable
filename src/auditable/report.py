"""Markdown rendering for the analysis reports: one dependency-free renderer.

``analyze_plan`` (PRE) and ``analyze_run`` (POST) each return a typed report whose
``summary()`` is terse indented plaintext. This module adds a second, additive
surface: a clean Markdown form of the SAME typed fields, suitable for pasting into
a pull request, an issue, or a design doc. It computes nothing new. The PRE report
already carries the execution-topology keystone, the four lint findings, the
preflight coverage views, and the notes; the POST report already carries the
blast-radius keystone, the ranked decisions, the grounding, and the notes. The
renderer only formats what is there.

Two reach paths produce the same string:

1. ``report.to_markdown()`` on each report object (a thin method that delegates
   here), paralleling the existing ``report.summary()``.
2. ``render_report(report)``, a top-level dispatcher that picks the right renderer
   by type. This is the single import for a caller who does not want the method.

Both PRE and POST render five labeled parts: the lifecycle stage (the banner plus
a meta line), what is risky on the graph, the keystone (PRE: an execution-topology
chokepoint; POST: a dependency-DAG blast-radius keystone, two distinct concepts
the wording keeps apart), the per-finding detail, and a short "what to do" line.

Dependency-free by design: standard library only, no NetworkX, no templating
engine, no table library. The ``PreReport`` / ``AnalysisReport`` imports are done
lazily inside the functions, because ``report.py`` sits beside ``analysis.py`` and
would otherwise import both ``auditable.graph.pre`` and ``auditable.analysis`` at
module load and risk an import cycle.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

__all__ = ["render_report", "pre_to_markdown", "post_to_markdown"]


# --- module-private string helpers -------------------------------------------
#
# Each returns a single Markdown line (or a small block). They hold the house
# style in one place: Title Case headings, GitHub pipe tables, and a withheld
# score that renders as "n/a" rather than a misleading 0.


def _h(level: int, text: str) -> str:
    """A heading line: ``level`` hashes, a space, then ``text`` (kept Title Case)."""
    return f"{'#' * max(1, level)} {text}"


def _bullet(text: str) -> str:
    """A single bullet line."""
    return f"- {text}"


def _kv(label: str, value: Any) -> str:
    """A ``- label: value`` line for a labeled scalar."""
    return f"- {label}: {value}"


def _fmt_score(x: Optional[float]) -> str:
    """Render a blast-share score: ``n/a`` when withheld, else three decimals.

    A withheld (``None``) score must never read as ``0.000``: a zero score means
    "nothing rests on this step", which is a real and different claim.
    """
    return "n/a" if x is None else f"{x:.3f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """A minimal GitHub pipe table as a list of lines.

    With no rows, return a single ``(none)`` line instead of an empty table, so a
    section with nothing to show stays readable.
    """
    if not rows:
        return ["(none)"]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def _cell(value: Any) -> str:
    """Make a value safe inside a pipe-table cell: ``-`` for empty, escape ``|``."""
    if value is None or value == "":
        return "-"
    return str(value).replace("|", "\\|")


# --- PRE: declared-plan analysis ---------------------------------------------


def pre_to_markdown(report: Any, *, level: int = 1) -> str:
    """Render a :class:`~auditable.graph.pre.PreReport` as Markdown.

    Mirrors the section order of ``PreReport.summary`` so the two stay
    recognizable, and keeps the PRE keystone labeled as an execution-topology
    chokepoint (distinct from the POST blast-radius keystone). No new computation:
    every value is read off ``report``.
    """
    lines: List[str] = []

    # 1. Lifecycle stage: the banner plus a one-line meta line.
    lines.append(_h(level, "Auditable PRE Report: Declared-Plan Analysis"))
    lines.append("")
    lines.append(_bullet("stage: PRE (design-time, before any step runs)"))
    lines.append(_kv("adapter", report.adapter))
    lines.append(_kv("steps", report.n_steps))

    # 2. The keystone: an execution-topology chokepoint, NOT the POST keystone.
    lines.append("")
    lines.append(_h(level + 1, "Keystone (Execution-Topology Chokepoint)"))
    lines.append("")
    if report.keystone_idx is not None:
        others = max(report.n_steps - 1, 0)
        lines.append(
            _bullet(
                f"step {report.keystone_idx}: {report.keystone_followers} of {others} "
                "other steps transitively follow it in control flow"
            )
        )
        lines.append(
            _bullet(
                "note: a structural design lint, not the POST blast-radius keystone"
            )
        )
    else:
        lines.append(_bullet("(none): no step has control-flow followers"))

    # 3. What is risky on the graph: the lint findings as a table, plus the
    #    honest State-B withhold boundary.
    lines.append("")
    lines.append(_h(level + 1, "What Is Risky on the Graph"))
    lines.append("")
    findings = list(report.findings)
    if findings:
        lines.append(_kv("lint findings", len(findings)))
        lines.append("")
        rows = [
            [
                _cell(f.severity),
                _cell(f.lint),
                _cell(f.node_idx),
                _cell(f.resource_id),
                _cell(f.detail),
            ]
            for f in findings
        ]
        lines.extend(
            _table(["severity", "lint", "step", "resource", "detail"], rows)
        )
    else:
        lines.append(_bullet("no lint findings"))
    lines.append("")
    lines.append(
        _bullet(
            "State B (dependency-state) blast-share risk: WITHHELD. "
            f"{report.state_b_withheld_reason}"
        )
    )

    # 4. Coverage readiness: descriptive, labeled NOT a risk score. Optional.
    cov_lines = _pre_coverage_block(report, level)
    if cov_lines:
        lines.append("")
        lines.extend(cov_lines)

    # 5. The per-finding "what to do" line, derived purely from existing fields.
    lines.append("")
    lines.append(_h(level + 1, "Recommended Action"))
    lines.append("")
    if findings:
        keystone_in_findings = report.keystone_idx is not None and any(
            f.node_idx == report.keystone_idx for f in findings
        )
        tail = (
            ", starting at the keystone step (it appears among them)"
            if keystone_in_findings
            else ""
        )
        lines.append(
            _bullet(
                f"address the {len(findings)} flagged lint(s) before deploy{tail}"
            )
        )
    else:
        lines.append(
            _bullet(
                "no design-time lints fired; the runtime scorer still withholds "
                "State B until live coverage arrives (see the no-score reason)"
            )
        )
    lines.append(
        _bullet(
            "reminder: PRE never emits a State-B number; scoring it is the runtime "
            "and POST job"
        )
    )

    # 6. Notes.
    notes = list(report.notes)
    if notes:
        lines.append("")
        lines.append(_h(level + 1, "Notes"))
        lines.append("")
        lines.extend(_bullet(n) for n in notes)

    return "\n".join(lines)


def _pre_coverage_block(report: Any, level: int) -> List[str]:
    """The optional Coverage Readiness block: preflight, touch gaps, barriers.

    Descriptive structure over the declared graph, labeled explicitly as NOT a
    risk score. Returns an empty list when the report carries none of the three
    preflight views.
    """
    pc = report.preflight_coverage
    rtc = report.resource_touch_completeness
    bi = report.barrier_inventory
    if pc is None and rtc is None and bi is None:
        return []

    lines: List[str] = [_h(level + 1, "Coverage Readiness"), ""]
    lines.append(_bullet("descriptive coverage-readiness, not a risk score"))

    if pc is not None:
        lines.append(
            _kv(
                "grade mix",
                f"observed={pc.observed}, declared={pc.declared}, "
                f"inferred={pc.inferred} (of {pc.n_dep_edges} dependency edge(s))",
            )
        )
        lines.append(
            _kv(
                "saturation",
                f"observed_fraction={pc.observed_fraction:.3f}, rho={pc.rho:.3f}",
            )
        )
        lines.append(_kv("runtime no-score reason", pc.no_score_reason))

    if rtc is not None:
        lines.append(
            _kv("reads identified", f"{rtc.reads_with_id}/{rtc.n_reads}")
        )
        lines.append(
            _kv("writes identified", f"{rtc.writes_with_id}/{rtc.n_writes}")
        )
        lines.append(
            _kv(
                "dependency edges identified",
                f"{rtc.edges_with_id}/{rtc.n_edges} "
                f"({rtc.edges_missing_structured_resource} missing the structured "
                "resource the runtime touch contract fills)",
            )
        )
        for g in rtc.gaps:
            where = (
                f"step {g.node_idx}"
                if g.src_idx is None
                else f"edge {g.node_idx}->{g.src_idx}"
            )
            lines.append(_bullet(f"gap [{g.kind}] {where}: {g.detail}"))

    if bi is not None:
        if bi.by_resource:
            lines.append(_kv("barrier inventory", "declared revalidation re-reads"))
            for r, idxs in bi.by_resource.items():
                steps = ", ".join(f"step {i}" for i in idxs)
                lines.append(_bullet(f"'{r}': {steps}"))
        else:
            lines.append(
                _kv("barrier inventory", "(none): no declared revalidation barriers")
            )

    return lines


# --- POST: structural-risk analysis ------------------------------------------


def post_to_markdown(report: Any, *, level: int = 1) -> str:
    """Render an :class:`~auditable.analysis.AnalysisReport` as Markdown.

    Mirrors the section order of ``AnalysisReport.summary`` so the two stay
    recognizable, and keeps the POST keystone labeled as a dependency-DAG
    blast-radius keystone (distinct from the PRE execution-topology chokepoint).
    No new computation: every value is read off ``report``.
    """
    lines: List[str] = []

    # 1. Lifecycle stage: the banner plus a meta line with coverage.
    cov = report.coverage
    grade_mix = _grade_mix(cov)
    lines.append(_h(level, "Auditable POST Report: Structural Risk Analysis"))
    lines.append("")
    lines.append(_bullet("stage: POST (runtime, after the run completes)"))
    lines.append(_kv("adapter", report.adapter))
    lines.append(_kv("completeness", report.completeness.value))
    lines.append(_kv("state", report.state))
    lines.append(_kv("steps", report.n_steps))
    lines.append(
        _kv(
            "coverage",
            f"{cov.n_dep_edges} dependency edge(s), rho={cov.rho:.3f}, "
            f"observed={cov.observed_fraction:.0%}, grades: {grade_mix}",
        )
    )

    # 2. The keystone: a dependency-DAG blast-radius keystone, NOT the PRE one.
    lines.append("")
    lines.append(_h(level + 1, "Keystone (Blast-Radius)"))
    lines.append("")
    k = report.keystone
    if k is not None and k.score is not None:
        n_down = round(k.score * (report.n_steps - 1))
        others = max(report.n_steps - 1, 0)
        lines.append(
            _bullet(
                f"step {k.idx} [{k.label}]: structural risk {_fmt_score(k.score)} "
                f"({n_down} of {others} other steps transitively rest on it)"
            )
        )
        lines.append(_bullet(f"grounding: {_grounding_line(k.grounding)}"))
    else:
        lines.append(
            _bullet("(none): structural risk withheld; see the state above")
        )

    # 3. What is risky on the graph: ranked decisions as a table, plus the
    #    optional per-step model-basis grounding.
    lines.append("")
    lines.append(_h(level + 1, "What Is Risky on the Graph"))
    lines.append("")
    ranked = list(report.ranked)
    rows = [
        [_fmt_score(d.score), _cell(d.idx), _cell(d.kind), _cell(d.label)]
        for d in ranked
    ]
    lines.append("ranked decisions (structural blast share):")
    lines.append("")
    lines.extend(_table(["score", "step", "kind", "label"], rows))

    grounding = report.grounding
    if grounding:
        lines.append("")
        lines.append(_h(level + 2, "Model-Basis Grounding"))
        lines.append("")
        for idx in sorted(grounding):
            lines.append(_bullet(f"step {idx}: {_grounding_line(grounding[idx])}"))

    # 4 and 5 combined: the "what to do" line, derived from the existing state.
    lines.append("")
    lines.append(_h(level + 1, "Recommended Action"))
    lines.append("")
    lines.append(_bullet(_post_action(report)))

    # 6. Notes.
    notes = list(report.notes)
    if notes:
        lines.append("")
        lines.append(_h(level + 1, "Notes"))
        lines.append("")
        lines.extend(_bullet(n) for n in notes)

    return "\n".join(lines)


def _grade_mix(cov: Any) -> str:
    """The observed/declared/inferred grade mix as a compact string, or ``none``."""
    parts = []
    for grade, count in cov.by_grade.items():
        if count:
            name = getattr(grade, "value", grade)
            parts.append(f"{name}={count}")
    return ", ".join(parts) if parts else "none"


def _post_action(report: Any) -> str:
    """The one-line POST action, picked from the report state (no new analysis)."""
    # Local import: the state constants live under the graph package; keeping the
    # import lazy preserves report.py's dependency-light load.
    from .graph.risk import (
        STATE_LOW_COVERAGE,
        STATE_SCORED,
        STATE_SINGLE_DECISION,
    )

    state = report.state
    if state == STATE_SCORED:
        return (
            "triage the keystone step first (it carries the highest blast share); "
            "the score is a ranking signal, not a calibrated probability"
        )
    if state == STATE_LOW_COVERAGE:
        return (
            "gather more observed dependency coverage before relying on a "
            "structural score; the current layer is too sparse or declared-only to "
            "score"
        )
    if state == STATE_SINGLE_DECISION:
        return (
            "fall back to per-decision data and grounding; a single-decision run "
            "has no cross-decision structure to rank"
        )
    return f"review the per-step rows; the run is in state {state}"


def _grounding_line(g: Any) -> str:
    """One-line grounding render, matching ``analysis._grounding_line`` wording.

    Re-implemented here rather than imported so the Markdown surface stays
    independent of the plaintext renderer. A step with no checkable basis reads as
    ``n/a`` rather than a false zero.
    """
    if g is None:
        return "n/a (this step states no checkable model basis)"
    if g.score is None:
        return f"{g.state} (no numeric score)"
    extra = ""
    if g.unmatched:
        extra = f"; unsupported: {', '.join(g.unmatched[:4])}"
    return f"{g.score:.3f} supported ({g.state}){extra}"


# --- the top-level dispatcher ------------------------------------------------


def render_report(report: Any, *, level: int = 1) -> str:
    """Render a PRE or POST report to Markdown, dispatched by type.

    ``PreReport`` renders through :func:`pre_to_markdown`; ``AnalysisReport``
    through :func:`post_to_markdown`. Any other type raises ``TypeError``. This is
    the single import a caller reaches for when they do not want the
    ``report.to_markdown()`` method.

    The ``PreReport`` / ``AnalysisReport`` imports are function-local: ``report.py``
    sits beside ``analysis.py``, and importing both report modules at load time
    would risk an import cycle.
    """
    from .analysis import AnalysisReport
    from .graph.pre import PreReport

    if isinstance(report, PreReport):
        return pre_to_markdown(report, level=level)
    if isinstance(report, AnalysisReport):
        return post_to_markdown(report, level=level)
    raise TypeError(
        "render_report expects a PreReport (PRE) or an AnalysisReport (POST), "
        f"got {type(report).__name__}"
    )
