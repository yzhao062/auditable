"""The public offline-analysis entry: ``analyze_run`` and ``AnalysisReport``.

This is the small stable surface for v0.3 post-hoc analysis (the PyOD-style
public API). One call maps a source through an adapter into the typed session
graph, scores it structurally, grounds the model basis where one is stated, and
returns an :class:`AnalysisReport` the user reads:

    report = analyze_run(messages, adapter=tau_bench_prior_db_reads_v1)
    print(report)            # ranked decisions, the keystone, coverage + rho, grounding

The flow is ``adapter.to_steps(source)`` -> :meth:`SessionGraph.from_steps` ->
:func:`structural_risk`, plus :func:`ground_record` / :func:`ground_basis` on each
step that carries a stated basis. The heavy types (``SessionGraph``, ``Step``,
``RiskResult``) stay internal under ``auditable.graph.*``; this module is the thin
public wrapper, so the internals can change without touching the public call.

The report shape is fixed now so the v0.3b live and own-record sources are purely
additive: ``completeness`` already distinguishes a complete run from a live prefix,
``grounding`` is keyed per step so a live record lights up the same field, and the
honesty notes already separate observed-source-but-modeled corpus edges from
declared own-record edges. Nothing here promises calibration: the structural score
is a ranking / triage signal, and grounding is deterministic consistency evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .graph.grounding import GroundingResult, ground_basis, ground_record
from .graph.risk import (
    STATE_LOW_COVERAGE,
    STATE_SCORED,
    STATE_SINGLE_DECISION,
    RiskResult,
    structural_risk,
)
from .graph.session import (
    EdgeCoverage,
    Grade,
    GraphCompleteness,
    SessionGraph,
    Step,
)

__all__ = ["AnalysisReport", "DecisionRisk", "analyze_run"]


@dataclass
class DecisionRisk:
    """One step's row in the ranked structural-risk report.

    - ``idx`` / ``kind`` / ``agent``: the step's identity in the session graph
      (``kind`` is ``"decision"`` or ``"tool_call"``).
    - ``score``: the normalized transitive blast share (how much of the rest of the
      run, transitively, rests on this step). ``None`` in a no-score state, so a
      withheld score never reads as zero risk.
    - ``label``: a short human label (the tool name, model id, or action type when
      present, else the kind).
    - ``node_attrs``: the step's typed node attributes (``tool`` / ``model_id`` /
      ``decision_basis`` / ``action_type`` and so on), passed through for inspection.
    - ``grounding``: the model-basis grounding for this step, when it states a basis;
      ``None`` when the step carries no checkable basis (a corpus tool step does not).
    """

    idx: int
    kind: str
    agent: str
    score: Optional[float]
    label: str
    node_attrs: Dict[str, Any] = field(default_factory=dict)
    grounding: Optional[GroundingResult] = None


@dataclass
class AnalysisReport:
    """The result of :func:`analyze_run`: the run's structural risk plus grounding.

    Fields the user reads:

    - ``state``: ``scored`` / ``no_score:single_decision`` / ``no_score:low_coverage``
      (the same honest gate ``structural_risk`` applies; a no-score state still
      reports coverage and the descriptive structure).
    - ``ranked``: every step as a :class:`DecisionRisk`, highest structural risk
      first; in a no-score state the scores are ``None`` and the order is by index.
    - ``keystone``: the worst-blast step (what most of the run rests on), or ``None``
      when the run is not scored.
    - ``per_session``: the keystone's blast share (the run-level risk), or ``None``.
    - ``coverage``: dependency-edge coverage with the saturation ratio ``rho`` and the
      per-grade breakdown (observed / declared / inferred).
    - ``grounding``: per step index, the model-basis grounding where a basis is
      stated. Empty for a corpus tool trace (no step states a model basis); it lights
      up on auditable's own records, which carry ``decision_basis`` and read context.
    - ``completeness``: ``complete`` (offline) now; ``prefix`` for the v0.3b live path,
      with no field change.
    - ``adapter``: the ingestion adapter id (``<name>_<version>``), so the report names
      the source that produced it.
    - ``features``: the raw layered structural features (descriptive, present in every
      state).
    - ``notes``: plain-language honesty notes (modeled corpus edges, low coverage,
      the no-calibration statement).
    """

    state: str
    adapter: str
    completeness: GraphCompleteness
    coverage: EdgeCoverage
    n_steps: int
    ranked: List[DecisionRisk]
    keystone: Optional[DecisionRisk]
    per_session: Optional[float]
    grounding: Dict[int, GroundingResult]
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.state == STATE_SCORED

    def summary(self) -> str:
        """Render the human-readable report the examples print."""
        cov = self.coverage
        by_grade = ", ".join(
            f"{g.value}={cov.by_grade.get(g, 0)}" for g in Grade if cov.by_grade.get(g, 0)
        ) or "none"
        lines = [
            "auditable :: structural risk analysis",
            f"  adapter:      {self.adapter}",
            f"  completeness: {self.completeness.value}",
            f"  state:        {self.state}",
            (
                f"  steps:        {self.n_steps}   "
                f"(dependency edges: {cov.n_dep_edges}, rho={cov.rho:.3f}, "
                f"observed={cov.observed_fraction:.0%}; grades: {by_grade})"
            ),
        ]
        if self.keystone is not None and self.keystone.score is not None:
            k = self.keystone
            n_down = round(k.score * (self.n_steps - 1))
            lines += [
                "",
                f"  keystone decision: step {k.idx}  [{k.label}]",
                (
                    f"    structural risk: {k.score:.3f}  "
                    f"({n_down} of {self.n_steps - 1} other steps transitively rest on it)"
                ),
                f"    grounding:       {_grounding_line(k.grounding)}",
            ]
        else:
            lines += ["", "  keystone decision: (none -- structural risk withheld; see state)"]

        lines += ["", "  ranked decisions (structural blast share):"]
        for d in self.ranked:
            score = "  n/a" if d.score is None else f"{d.score:.3f}"
            lines.append(f"    {score}  step {d.idx:<3d} {d.label}")

        if self.grounding:
            lines += ["", "  model-basis grounding (where a basis is stated):"]
            for idx in sorted(self.grounding):
                lines.append(f"    step {idx}: {_grounding_line(self.grounding[idx])}")

        if self.notes:
            lines += ["", "  notes:"]
            lines += [f"    - {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:  # so print(report) renders the summary
        return self.summary()


def _grounding_line(g: Optional[GroundingResult]) -> str:
    """One-line grounding render for the summary."""
    if g is None:
        return "n/a (this step states no checkable model basis)"
    if g.score is None:
        return f"{g.state} (no numeric score)"
    extra = ""
    if g.unmatched:
        extra = f"; unsupported: {', '.join(g.unmatched[:4])}"
    return f"{g.score:.3f} supported ({g.state}){extra}"


def _label(step: Step) -> str:
    """A short human label for a step: the tool, action, or model when named."""
    na = step.node_attrs or {}
    tool = na.get("tool")
    if tool:
        return f"{step.kind} {tool}"
    action_type = na.get("action_type")
    if action_type:
        return f"{step.kind} {action_type}"
    model_id = na.get("model_id")
    if model_id:
        return f"{step.kind} ({model_id})"
    return step.kind


def _record_index(source: Any) -> Dict[str, Any]:
    """Map ``record_id -> record`` for a record-like source (own records / loaded log).

    Only a concrete ``Sequence`` is indexed, never a one-shot iterator (the adapter
    already consumed ``source``); a record-like item duck-types ``.model``, ``.data``,
    and a non-empty ``.record_id`` string. Corpus message dicts have none of these, so
    the index is empty and grounding falls back to the per-step basis path.
    """
    out: Dict[str, Any] = {}
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        return out
    for item in source:
        if not (hasattr(item, "model") and hasattr(item, "data")):
            continue
        rid = getattr(item, "record_id", None)
        if isinstance(rid, str) and rid:
            out[rid] = item
    return out


def _ground_step(step: Step, record_index: Dict[str, Any]) -> Optional[GroundingResult]:
    """Ground one step's stated basis, or ``None`` when it carries no basis.

    Two paths, both deterministic and offline. When the step carries a sealed
    ``record_id`` that resolves to a source record (auditable's own records, the
    offline log, a live record in v0.3b), ground straight from that record, so the
    full read context (``data.inputs`` / ``data.retrieved`` / ``data.snapshot.state``)
    is used. Otherwise fall back to the basis and any context carried on the step's
    ``node_attrs``. A step with no checkable basis (a corpus tool step) returns
    ``None`` rather than a false zero.
    """
    rid = step.record_id
    if rid and rid in record_index:
        return ground_record(record_index[rid])
    na = step.node_attrs or {}
    basis = na.get("decision_basis") or na.get("basis")
    if not basis:
        return None
    return ground_basis(
        basis,
        inputs=na.get("inputs"),
        retrieved=na.get("retrieved"),
        state=na.get("snapshot_state") or na.get("state"),
    )


def _notes(
    state: str, steps: Sequence[Step], coverage: EdgeCoverage, grounding: Dict[int, GroundingResult]
) -> List[str]:
    """Plain-language honesty notes for the report."""
    notes: List[str] = []
    n_modeled = sum(
        1
        for s in steps
        for e in s.deps
        if e.grade is Grade.OBSERVED and isinstance(e.evidence, dict) and e.evidence.get("modeled")
    )
    if n_modeled:
        notes.append(
            f"all {n_modeled} observed dependency edge(s) are MODELED: a conservative "
            "prior-read upper bound over the observed reads, not a causal label."
        )
    n_declared = coverage.by_grade.get(Grade.DECLARED, 0)
    if n_declared:
        notes.append(
            f"{n_declared} dependency edge(s) are DECLARED, not observed (sparse own-record "
            "edges pending the runtime resource-touch contract); marked low-coverage."
        )
    if state == STATE_LOW_COVERAGE:
        notes.append(
            "structural risk is withheld: the dependency layer is too sparse, declared, or "
            "saturated to score (it would only restate run size). This is the honest no-score "
            "gate, not a paper-validated structural signal."
        )
    elif state == STATE_SINGLE_DECISION:
        notes.append(
            "structural risk is withheld: a single-decision run has no cross-decision "
            "structure; it degrades to the per-decision data and grounding signals."
        )
    elif state == STATE_SCORED:
        notes.append(
            "structural signal adopts the graph-view paper's construction (blast structure "
            "over the dependency DAG); it is a ranking / triage signal, not calibrated."
        )
    if not grounding:
        notes.append(
            "grounding is empty: no step states a checkable model basis (a corpus tool trace "
            "does not). It lights up on records that carry a basis, such as auditable's own runs."
        )
    return notes


def analyze_run(source: Any, *, adapter: Any, ground: bool = True) -> AnalysisReport:
    """Analyze one agent run offline: structural risk plus model-basis grounding.

    ``source`` is whatever the ``adapter`` consumes (a public-corpus trajectory, a
    chain of auditable's own ``DecisionRecord``s, or, in v0.3b, a live stream).
    ``adapter`` is any :class:`~auditable.graph.adapters.protocol.Adapter` (it
    exposes ``to_steps`` plus a ``name`` / ``version``). The call maps the source to
    typed steps, builds the :class:`SessionGraph`, scores it with
    :func:`structural_risk`, grounds each step that states a basis, and returns an
    :class:`AnalysisReport`.

    Set ``ground=False`` to skip the (cheap, deterministic) grounding pass. Scoring
    requires the ``graph`` extra (NetworkX); without it the underlying projection
    raises a clear ``ImportError``.
    """
    steps: List[Step] = list(adapter.to_steps(source))
    graph = SessionGraph.from_steps(steps)
    risk: RiskResult = structural_risk(graph)

    grounding: Dict[int, GroundingResult] = {}
    if ground:
        record_index = _record_index(source)
        for s in steps:
            g = _ground_step(s, record_index)
            if g is not None:
                grounding[s.idx] = g

    scored = risk.state == STATE_SCORED
    ranked = [
        DecisionRisk(
            idx=s.idx,
            kind=s.kind,
            agent=s.agent,
            score=risk.per_decision.get(s.idx) if scored else None,
            label=_label(s),
            node_attrs=dict(s.node_attrs or {}),
            grounding=grounding.get(s.idx),
        )
        for s in steps
    ]
    # highest structural risk first; a withheld (None) score sorts last, then by index
    ranked.sort(key=lambda d: (d.score is None, -(d.score or 0.0), d.idx))
    keystone = ranked[0] if (scored and ranked and ranked[0].score is not None) else None

    adapter_id = (
        getattr(adapter, "id", None)
        or "_".join(p for p in (getattr(adapter, "name", ""), getattr(adapter, "version", "")) if p)
        or type(adapter).__name__
    )

    return AnalysisReport(
        state=risk.state,
        adapter=adapter_id,
        completeness=risk.completeness,
        coverage=risk.coverage,
        n_steps=len(steps),
        ranked=ranked,
        keystone=keystone,
        per_session=risk.per_session,
        grounding=grounding,
        features=risk.features,
        notes=_notes(risk.state, steps, risk.coverage, grounding),
    )
