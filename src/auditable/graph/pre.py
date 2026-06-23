"""PRE: design-time lints over a DECLARED agent plan (before any run).

The PRE entry mirrors :func:`auditable.analysis.analyze_run`'s adapter ->
SessionGraph flow, but computes only the parts that are honest before a single
step executes. A DECLARED plan (a LangGraph compiled graph, a CrewAI task DAG, or
an AutoGen topology, lowered through :class:`DeclaredPlanAdapter` into the neutral
plan dict) carries control flow and declared data reads / writes, but no observed
values. So PRE does two things and withholds a third:

1. **Execution-topology keystone.** The structural chokepoint of the declared
   plan: the node the most other nodes transitively FOLLOW in control flow, via
   :func:`execution_reach` over the ``handoff_to`` projection. This is a
   STRUCTURAL design lint (a chokepoint), NOT the POST blast-radius keystone from
   :mod:`auditable.graph.risk` (a blast-share triage signal over the dependency
   DAG). The two are distinct named concepts and must not be conflated.

2. **Four reachability lints** over the projected declared graph. All four are
   pure, read-only NetworkX queries: no mutation of the graph, no side effects, no
   runtime / value execution. The "would it flip" and drift-confirmation halves
   are runtime work, explicitly out of scope and noted in each finding's detail.

3. **State-B (dependency-state) blast-share risk is WITHHELD.** A declared
   dependency layer is declared-only (``observed_fraction=0``), so a multi-step
   plan makes :func:`structural_risk` return ``no_score:low_coverage`` (a 0- or
   1-step plan is gated earlier as ``no_score:single_decision``). Either way no
   number is emitted: PRE asserts the boundary as "the verdict is some
   ``no_score:*`` state" and surfaces ``state_b_risk=None`` with a reason string;
   it never presents a dependency-state risk number. Only a SCORED verdict on a
   declared graph violates the boundary, and then :func:`analyze_plan` raises
   rather than emit the number.

Alongside the withheld State-B number, PRE attaches a **Preflight Coverage
Report**: a descriptive, calibrated coverage-readiness view, NOT a risk score. It
reuses the existing :meth:`SessionGraph.coverage` model and the declared
resource-touch metadata to tell the user what the runtime scorer will need before
it can score (``preflight_coverage``), which declared touches still lack a resource
identity (``resource_touch_completeness``), and where declared revalidation
barriers exist per resource (``barrier_inventory``). This strengthens PRE without
selling a false score and leaves the State-B withhold boundary above unchanged.

PRE applies only where a declared graph exists. A free-form ReAct agent with no
declared plan degrades to the flat rule floor and is out of scope here.

This module lives under ``auditable.graph.*`` and adds no top-level public export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapters.declared_plan import declared_plan_v1
from .risk import (
    STATE_LOW_COVERAGE,
    STATE_SCORED,
    STATE_SINGLE_DECISION,
    structural_risk,
)
from .session import EdgeCoverage, Grade, SessionGraph, Step


def execution_projection(G) -> "Any":
    """The ``handoff_to`` projection as a simple ``DiGraph`` (predecessor -> successor).

    Step nodes plus the execution (control-flow) edges, dropping the dependency and
    ``emits`` layers. ``handoff_to`` points predecessor -> successor, so a node's
    transitive control-flow FOLLOWERS are its descendants here (the basis for
    :func:`auditable.graph.execution_reach`).
    """
    try:
        import networkx as nx
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "auditable.graph requires the 'graph' extra: pip install auditable[graph]"
        ) from exc
    proj = nx.DiGraph()
    proj.add_nodes_from(n for n, d in G.nodes(data=True) if d["ntype"] in ("decision", "tool_call"))
    proj.add_edges_from(
        (u, v) for u, v, d in G.edges(data=True) if d["etype"] == "handoff_to"
    )
    return proj


@dataclass
class LintFinding:
    """One PRE lint hit: a structural design issue read off the declared graph.

    - ``lint``: the lint name (e.g. ``'write_with_no_prior_read'``).
    - ``node_idx``: the offending step idx.
    - ``resource_id``: the resource the finding is about, or ``None`` when not
      resource-specific.
    - ``detail``: a one-line human reason. For the annotation-only halves it states
      that the runtime / value confirmation is out of scope at PRE.
    - ``severity``: ``'warning'`` by default; PRE findings are structural design
      warnings, not validated failure predictions.
    """

    lint: str
    node_idx: int
    resource_id: Optional[str]
    detail: str
    severity: str = "warning"


def _dependency_dag(G):
    """The ``depends_on`` projection (step -> what it relied on). Lazy to avoid a cycle."""
    from auditable.graph import dependency_dag

    return dependency_dag(G)


def _backward_slice_idxs(dep, node_idx: int) -> List[int]:
    """The step idxs in ``node_idx``'s backward slice = ``{node_idx}`` U its dependencies.

    ``depends_on`` points dependent -> dependency, so the dependencies a node
    transitively rests on are its DESCENDANTS in the dependency DAG.
    """
    import networkx as nx

    node = f"step::{node_idx}"
    if node not in dep:
        return [node_idx]
    slice_nodes = {node} | nx.descendants(dep, node)
    return [int(n.split("::", 1)[1]) for n in slice_nodes]


def _reads_by_idx(G) -> Dict[int, List[str]]:
    return {
        d["idx"]: list(d.get("reads") or [])
        for _, d in G.nodes(data=True)
        if d["ntype"] in ("decision", "tool_call")
    }


def write_with_no_prior_read(G) -> List[LintFinding]:
    """Fire when a node writes a resource never read in its backward slice.

    Primitive: ``nx.descendants`` over ``dependency_dag(G)`` from the write node (the
    backward slice = what the action transitively rests on, because ``depends_on``
    points dependent -> dependency), cross-referenced against the ``reads`` resource
    sets in ``node_attrs`` of the slice nodes plus the writer itself. For each step W
    whose ``writes`` is non-empty, FIRE one finding per written resource R that is
    NOT in the union of reads over ``{W}`` U slice.
    """
    dep = _dependency_dag(G)
    reads_by_idx = _reads_by_idx(G)
    findings: List[LintFinding] = []
    for node_idx, d in sorted(G.nodes(data=True), key=lambda kv: kv[1].get("idx", -1)):
        if d.get("ntype") not in ("decision", "tool_call"):
            continue
        writes = list(d.get("writes") or [])
        if not writes:
            continue
        idx = d["idx"]
        slice_idxs = _backward_slice_idxs(dep, idx)
        read_union = set()
        for j in slice_idxs:
            read_union.update(reads_by_idx.get(j, ()))
        for r in writes:
            if r not in read_union:
                findings.append(
                    LintFinding(
                        lint="write_with_no_prior_read",
                        node_idx=idx,
                        resource_id=r,
                        detail=(
                            f"node {idx} writes resource '{r}' that no node in its "
                            "backward slice (nor itself) declares reading first"
                        ),
                    )
                )
    return findings


def flippable_dependency_annotations(G) -> List[LintFinding]:
    """Annotate unpinned, non-revalidated volatile dependencies feeding a decision.

    Primitive: ``nx.descendants`` over ``dependency_dag(G)`` from each decision node
    (its backward slice / dependency set), intersected with the per-edge evidence
    flags on the DECLARED ``depends_on`` edges. For each decision D, over the DECLARED
    depends_on edges on D's backward slice that carry ``evidence['volatile']``, FIRE
    one annotation per such edge / resource that is neither ``evidence['pinned']`` nor
    ``evidence['revalidates']``. This is an ANNOTATION, not a value-flip proof:
    severity stays ``'warning'`` and the detail says the would-flip half needs runtime
    values.
    """
    import networkx as nx

    dep = _dependency_dag(G)
    findings: List[LintFinding] = []
    for u, v, d in sorted(
        G.edges(data=True), key=lambda e: (G.nodes[e[0]].get("idx", -1), G.nodes[e[1]].get("idx", -1))
    ):
        if d.get("etype") != "depends_on" or d.get("grade") is not Grade.DECLARED:
            continue
        ev = d.get("evidence") or {}
        if not ev.get("volatile") or ev.get("pinned") or ev.get("revalidates"):
            continue
        reader_idx = int(u.split("::", 1)[1])
        # the flippable dependency must feed a decision: either the reader is a
        # decision, or some decision's backward slice contains this reader.
        feeds_decision = False
        for n, nd in G.nodes(data=True):
            if nd.get("ntype") != "decision":
                continue
            d_idx = nd["idx"]
            node = f"step::{d_idx}"
            if d_idx == reader_idx or (node in dep and f"step::{reader_idx}" in nx.descendants(dep, node)):
                feeds_decision = True
                break
        if not feeds_decision:
            continue
        findings.append(
            LintFinding(
                lint="flippable_dependency_annotation",
                node_idx=reader_idx,
                resource_id=ev.get("resource_id"),
                detail=(
                    f"node {reader_idx} reads volatile resource '{ev.get('resource_id')}' "
                    "unpinned and un-revalidated feeding a decision; would-flip needs "
                    "runtime values (annotation, not a value-flip proof)"
                ),
            )
        )
    return findings


def scope_vs_snapshot(G) -> List[LintFinding]:
    """Fire when granted tool scope strictly exceeds the snapshot the node read.

    Primitive: set comparison of the declared scope (``node_attrs['scope']``, the
    granted resource ids) versus the read-resource set actually pulled into the
    node's snapshot, computed as the union of ``reads`` over ``{N}`` U
    ``nx.descendants(dependency_dag(G), 'step::N')``. For each node N whose ``scope``
    is present, FIRE when ``set(scope)`` is a STRICT superset of the read set (the
    grant exceeds the snapshot it validated; it can act on state it never read). The
    reported ``resource_id`` per finding is one of ``scope - read_set``.
    """
    dep = _dependency_dag(G)
    reads_by_idx = _reads_by_idx(G)
    findings: List[LintFinding] = []
    for _n, d in sorted(G.nodes(data=True), key=lambda kv: kv[1].get("idx", -1)):
        if d.get("ntype") not in ("decision", "tool_call"):
            continue
        scope = d.get("scope")
        if scope is None:
            continue  # no scope claim -> skipped by this lint
        idx = d["idx"]
        read_set = set()
        for j in _backward_slice_idxs(dep, idx):
            read_set.update(reads_by_idx.get(j, ()))
        scope_set = set(scope)
        if scope_set > read_set:  # strict superset
            extra = scope_set - read_set
            for r in sorted(extra):
                findings.append(
                    LintFinding(
                        lint="scope_vs_snapshot",
                        node_idx=idx,
                        resource_id=r,
                        detail=(
                            f"node {idx} is granted scope over '{r}' but never read it "
                            "into its snapshot; granted scope exceeds the validated state"
                        ),
                    )
                )
    return findings


def missing_revalidation_barrier(G) -> List[LintFinding]:
    """Fire when a volatile read reaches an action with no re-read between them.

    Two-projection query. First, ``nx.descendants`` over ``dependency_dag(G)`` locates
    a volatile read upstream of a consequential action (the backward slice). Then
    ``nx.descendants`` over :func:`execution_projection` (``handoff_to``) checks the
    control path from the volatile-read node to the action contains NO intervening
    barrier (a node re-reading that resource, i.e. with the resource in its
    ``node_attrs['barriers']`` set).

    For each consequential action A (``writes`` non-empty, or a decision with a
    volatile dependency), for each volatile read node V in A's backward slice whose
    resource R is volatile, FIRE when there is NO node B with R in its barrier set on
    a control path strictly between V and A in handoff order. No finding when such a
    barrier B exists.
    """
    import networkx as nx

    dep = _dependency_dag(G)
    execG = execution_projection(G)
    barriers_by_idx = {
        d["idx"]: set(d.get("barriers") or [])
        for _, d in G.nodes(data=True)
        if d["ntype"] in ("decision", "tool_call")
    }
    volatile_reads_by_idx = {
        d["idx"]: set(d.get("volatile_reads") or [])
        for _, d in G.nodes(data=True)
        if d["ntype"] in ("decision", "tool_call")
    }
    node_kind = {d["idx"]: d["ntype"] for _, d in G.nodes(data=True) if d["ntype"] in ("decision", "tool_call")}
    has_writes = {
        d["idx"]: bool(d.get("writes"))
        for _, d in G.nodes(data=True)
        if d["ntype"] in ("decision", "tool_call")
    }

    findings: List[LintFinding] = []
    for a_idx in sorted(node_kind):
        slice_idxs = _backward_slice_idxs(dep, a_idx)
        # volatile reads upstream of (or at) A, keyed by the reading node V and resource
        # R. A read that is itself a revalidation barrier of R clears the staleness on
        # its control path (schema semantics), so it is not an exposed stale read.
        volatile_hits = [
            (v_idx, r)
            for v_idx in slice_idxs
            for r in volatile_reads_by_idx.get(v_idx, ())
            if r not in barriers_by_idx.get(v_idx, ())
        ]
        # A is consequential if it writes, or it is a decision with a volatile dependency
        is_action = has_writes.get(a_idx, False) or (
            node_kind.get(a_idx) == "decision" and bool(volatile_hits)
        )
        if not is_action or not volatile_hits:
            continue
        a_node = f"step::{a_idx}"
        followers_of_a = nx.descendants(execG, a_node) if a_node in execG else set()
        for v_idx, r in volatile_hits:
            # the action node itself re-reading (revalidating) R at time-of-use clears the
            # prior volatile read: the volatile state was re-validated before the action.
            if r in barriers_by_idx.get(a_idx, ()):
                continue
            v_node = f"step::{v_idx}"
            followers_of_v = nx.descendants(execG, v_node) if v_node in execG else set()
            # a barrier B strictly between V and A in handoff order: B follows V and A follows B
            barrier_found = False
            for b_idx, b_resources in barriers_by_idx.items():
                if b_idx in (v_idx, a_idx) or r not in b_resources:
                    continue
                b_node = f"step::{b_idx}"
                if b_node in followers_of_v and a_node in (
                    nx.descendants(execG, b_node) if b_node in execG else set()
                ):
                    barrier_found = True
                    break
            if not barrier_found:
                findings.append(
                    LintFinding(
                        lint="missing_revalidation_barrier",
                        node_idx=a_idx,
                        resource_id=r,
                        detail=(
                            f"volatile resource '{r}' read at node {v_idx} reaches "
                            f"action node {a_idx} with no intervening re-read on the "
                            "control path; drift confirmation needs runtime values"
                        ),
                    )
                )
    return findings


_LINTS = (
    write_with_no_prior_read,
    flippable_dependency_annotations,
    scope_vs_snapshot,
    missing_revalidation_barrier,
)

_WITHHELD_REASON = (
    "dependency-state (State B) blast-share risk is WITHHELD at PRE: the declared "
    "dependency layer is declared-only (observed_fraction=0), so structural_risk "
    "returns no_score:low_coverage. State-B risk is the runtime/POST job."
)


# --- the Preflight Coverage Report (descriptive, NOT a risk number) ----------
#
# Three coverage-readiness views over the DECLARED graph, the calibrated signal
# the design note ("Preflight Coverage Report") asks for in place of a guessed
# pre-deploy risk score. Each is descriptive structure: it reuses the existing
# ``coverage()`` model and the declared resource-touch metadata to tell the user
# what the runtime scorer will need before it can score, never a number that the
# declared-only evidence cannot support. They hand LIVE / POST a clear
# contract (the edges and resources that must be observed live), and they leave
# the State-B withhold boundary exactly as is.


@dataclass
class PreflightCoverage:
    """The existing ``coverage()`` model surfaced over the DECLARED graph.

    Descriptive, NOT a risk number. Reads :meth:`SessionGraph.coverage` plus the
    exact ``no_score:*`` state :func:`structural_risk` would apply at runtime, so
    the user can see the grade mix and why the runtime scorer will withhold a
    State-B number rather than guessing one here.

    - ``n_steps``: plan node count (the size-normalized risk denominator basis).
    - ``n_dep_edges``: total dependency edges (every one DECLARED at PRE).
    - ``observed`` / ``declared`` / ``inferred``: the grade-mix counts from
      ``coverage().by_grade`` (the same three :class:`Grade` buckets, flattened to
      plain ints for legibility).
    - ``observed_fraction`` / ``rho``: the observed share and the saturation ratio
      from ``coverage()``; at PRE ``observed_fraction`` is ``0.0`` on any non-empty
      declared layer.
    - ``no_score_reason``: the exact ``no_score:*`` state ``structural_risk``
      applies -- ``no_score:low_coverage`` for a multi-step declared plan,
      ``no_score:single_decision`` for a 0- or 1-step plan. This is the reason the
      State-B score is withheld, surfaced descriptively (it is never a number).
    - ``would_score``: always ``False`` at PRE; present so the contract reads
      explicitly as "the runtime scorer cannot score this declared layer yet".
    """

    n_steps: int
    n_dep_edges: int
    observed: int
    declared: int
    inferred: int
    observed_fraction: float
    rho: float
    no_score_reason: str
    would_score: bool = False

    @classmethod
    def from_coverage(cls, coverage: EdgeCoverage, n_steps: int, no_score_reason: str) -> "PreflightCoverage":
        by = coverage.by_grade
        return cls(
            n_steps=n_steps,
            n_dep_edges=coverage.n_dep_edges,
            observed=by.get(Grade.OBSERVED, 0),
            declared=by.get(Grade.DECLARED, 0),
            inferred=by.get(Grade.INFERRED, 0),
            observed_fraction=coverage.observed_fraction,
            rho=coverage.rho,
            no_score_reason=no_score_reason,
        )


@dataclass
class ResourceGap:
    """One declared touch (a read, write, or dependency edge) lacking a resource id.

    - ``kind``: ``'read'`` / ``'write'`` / ``'edge'``.
    - ``node_idx``: the owning step idx (the dependent step for an ``'edge'``).
    - ``src_idx``: only for ``'edge'`` -- the producer step the edge points at.
    - ``detail``: a one-line reason naming the missing identity.
    """

    kind: str
    node_idx: int
    src_idx: Optional[int] = None
    detail: str = ""


@dataclass
class ResourceTouchCompleteness:
    """Which declared touches carry a resource identity, and which do not.

    The runtime touch contract (v0.3b own-record) matches a later read to an
    earlier write of the same ``{namespace, resource_id, key}`` and fills
    :class:`~auditable.graph.session.ResourceRef` on the observed edge. At PRE no
    edge is observed, so this view reports, descriptively, which writes, reads, and
    declared dependency edges are still missing an identity the runtime contract
    will need:

    - a read / write is complete when its ``node_attrs`` id string is non-empty;
    - a declared dependency edge is complete when it carries a resource id, either
      the structured ``DependencyEdge.resource`` (``ResourceRef``) or the
      ``evidence['resource_id']`` string the declared adapter records.

    Counts plus the per-touch gap list are exposed so a caller can see both the
    headline (``writes_with_id`` of ``n_writes``) and the exact offending touch.
    ``edges_missing_structured_resource`` separately counts edges that carry an
    ``evidence['resource_id']`` but a ``None`` structured ``resource`` -- the
    declared-corpus norm, and exactly the seam the runtime contract fills.
    """

    n_reads: int
    n_writes: int
    n_edges: int
    reads_with_id: int
    writes_with_id: int
    edges_with_id: int
    edges_missing_structured_resource: int
    gaps: List[ResourceGap] = field(default_factory=list)


@dataclass
class BarrierInventory:
    """The declared re-read / re-validation nodes, grouped per resource (structure only).

    A barrier is a node that re-reads (revalidates) a resource: the declared
    adapter records it in ``node_attrs['barriers']`` (a read flagged
    ``revalidates``). This view lists, per resource id, the step idxs that declare a
    revalidation barrier for it, and the flat set of resources that have at least
    one barrier. It is reported as STRUCTURE: a resource that appears as a volatile
    read but is absent from ``by_resource`` has no declared barrier, which a
    consuming view can surface without claiming any drift occurred (drift
    confirmation is runtime work, out of scope at PRE).

    - ``by_resource``: resource id -> sorted list of barrier step idxs.
    - ``barrier_nodes``: sorted list of every step idx that declares any barrier.
    - ``resources_with_barrier``: sorted list of resource ids that have a barrier.
    """

    by_resource: Dict[str, List[int]]
    barrier_nodes: List[int]
    resources_with_barrier: List[str]


def resource_touch_completeness(G) -> ResourceTouchCompleteness:
    """Report which declared reads, writes, and dependency edges carry a resource id.

    Pure read-only scan of the projected graph: it reads ``node_attrs['reads']`` /
    ``['writes']`` on each step node and the ``evidence`` / ``resource`` on each
    ``depends_on`` edge. A read or write counts as identified when its id string is
    non-empty; a dependency edge counts as identified when it carries either a
    structured :class:`ResourceRef` (``resource``) or an ``evidence['resource_id']``
    string. Edges that carry only the ``evidence`` string but a ``None`` structured
    ``resource`` are counted separately, since that ``ResourceRef`` is exactly what
    the runtime touch contract fills.
    """
    n_reads = n_writes = 0
    reads_with_id = writes_with_id = 0
    gaps: List[ResourceGap] = []
    for _n, d in sorted(G.nodes(data=True), key=lambda kv: kv[1].get("idx", -1)):
        if d.get("ntype") not in ("decision", "tool_call"):
            continue
        idx = d["idx"]
        for r in d.get("reads") or []:
            n_reads += 1
            if isinstance(r, str) and r:
                reads_with_id += 1
            else:
                gaps.append(
                    ResourceGap(
                        kind="read",
                        node_idx=idx,
                        detail=f"node {idx} declares a read with no resource identity",
                    )
                )
        for w in d.get("writes") or []:
            n_writes += 1
            if isinstance(w, str) and w:
                writes_with_id += 1
            else:
                gaps.append(
                    ResourceGap(
                        kind="write",
                        node_idx=idx,
                        detail=f"node {idx} declares a write with no resource identity",
                    )
                )

    n_edges = edges_with_id = edges_missing_structured = 0
    for u, v, d in sorted(
        G.edges(data=True),
        key=lambda e: (G.nodes[e[0]].get("idx", -1), G.nodes[e[1]].get("idx", -1)),
    ):
        if d.get("etype") != "depends_on":
            continue
        n_edges += 1
        node_idx = int(u.split("::", 1)[1])
        src_idx = int(v.split("::", 1)[1])
        ev = d.get("evidence") or {}
        ev_rid = ev.get("resource_id")
        structured = d.get("resource")  # a ResourceRef, or None on the declared corpus
        has_ev_id = isinstance(ev_rid, str) and bool(ev_rid)
        has_id = structured is not None or has_ev_id
        if has_id:
            edges_with_id += 1
        else:
            gaps.append(
                ResourceGap(
                    kind="edge",
                    node_idx=node_idx,
                    src_idx=src_idx,
                    detail=(
                        f"dependency edge {node_idx}->{src_idx} carries no resource "
                        "identity (neither a structured resource nor evidence resource_id)"
                    ),
                )
            )
        if structured is None:
            edges_missing_structured += 1
            if has_ev_id:
                gaps.append(
                    ResourceGap(
                        kind="edge",
                        node_idx=node_idx,
                        src_idx=src_idx,
                        detail=(
                            f"dependency edge {node_idx}->{src_idx} has evidence "
                            f"resource_id '{ev_rid}' but no structured resource (the "
                            "ResourceRef the runtime touch contract fills)"
                        ),
                    )
                )

    return ResourceTouchCompleteness(
        n_reads=n_reads,
        n_writes=n_writes,
        n_edges=n_edges,
        reads_with_id=reads_with_id,
        writes_with_id=writes_with_id,
        edges_with_id=edges_with_id,
        edges_missing_structured_resource=edges_missing_structured,
        gaps=gaps,
    )


def barrier_inventory(G) -> BarrierInventory:
    """Inventory the declared revalidation barriers per resource (structure only).

    Pure read-only scan of ``node_attrs['barriers']`` (the reads flagged
    ``revalidates`` by the declared adapter). Returns, per resource id, the sorted
    step idxs that declare a barrier for it, plus the flat barrier-node and
    barrier-resource sets. No drift is claimed: this is the declared structure of
    where revalidation re-reads exist, which a consuming view can compare against
    the volatile-read set to spot a missing barrier without runtime values.
    """
    by_resource: Dict[str, List[int]] = {}
    barrier_nodes: set = set()
    for _n, d in G.nodes(data=True):
        if d.get("ntype") not in ("decision", "tool_call"):
            continue
        idx = d["idx"]
        barriers = d.get("barriers") or []
        if barriers:
            barrier_nodes.add(idx)
        for r in barriers:
            by_resource.setdefault(r, []).append(idx)
    by_resource = {r: sorted(set(idxs)) for r, idxs in sorted(by_resource.items())}
    return BarrierInventory(
        by_resource=by_resource,
        barrier_nodes=sorted(barrier_nodes),
        resources_with_barrier=sorted(by_resource),
    )


def _no_score_reason(state: str) -> str:
    """Map the ``structural_risk`` no-score state to the descriptive reason string.

    ``analyze_plan`` has already asserted the state is some ``no_score:*`` value (it
    raises on ``STATE_SCORED``), so only the two no-score states reach here.
    """
    if state == STATE_SINGLE_DECISION:
        return (
            f"{STATE_SINGLE_DECISION}: a 0- or 1-step plan has no cross-decision "
            "structure to score (Gate 1 fires before the low-coverage gate)"
        )
    if state == STATE_LOW_COVERAGE:
        return (
            f"{STATE_LOW_COVERAGE}: the declared dependency layer is declared-only "
            "(observed_fraction=0), so the runtime scorer withholds a State-B number"
        )
    # defensive: any other no_score:* state surfaces verbatim
    return state


@dataclass
class PreReport:
    """The result of :func:`analyze_plan`: the PRE-honest view of a DECLARED plan.

    - ``adapter``: the ingestion adapter id (``declared_plan_v1``).
    - ``n_steps``: number of plan nodes.
    - ``keystone_idx`` / ``keystone_followers``: the execution-topology keystone (the
      argmax of :func:`execution_reach`) and its transitive control-flow followers.
      This is the STRUCTURAL chokepoint of the declared plan, NOT the POST blast-radius
      keystone from :mod:`auditable.graph.risk`.
    - ``execution_reach_by_idx``: every step idx -> its transitive control-flow
      followers.
    - ``findings``: the four lints' :class:`LintFinding`s.
    - ``state_b_risk`` / ``state_b_withheld`` / ``state_b_withheld_reason``: the
      dependency-state blast-share risk is ALWAYS withheld at PRE (declared-only); the
      number is never computed. ``state_b_risk`` stays ``None`` and ``state_b_withheld``
      stays ``True``.
    - ``preflight_coverage`` / ``resource_touch_completeness`` /
      ``barrier_inventory``: the Preflight Coverage Report -- a descriptive,
      calibrated coverage-readiness view (the grade mix, the exact no-score reason
      the runtime scorer will apply, which declared touches lack a resource
      identity, and the declared revalidation barriers per resource). It is NOT a
      risk number and does not touch the State-B withhold boundary.
    - ``notes``: plain-language notes, including that the keystone is a structural
      chokepoint (a design lint), not a failure predictor.
    """

    adapter: str
    n_steps: int
    keystone_idx: Optional[int]
    keystone_followers: int
    execution_reach_by_idx: Dict[int, int]
    findings: List[LintFinding]
    state_b_risk: None = None
    state_b_withheld: bool = True
    state_b_withheld_reason: str = _WITHHELD_REASON
    preflight_coverage: Optional[PreflightCoverage] = None
    resource_touch_completeness: Optional[ResourceTouchCompleteness] = None
    barrier_inventory: Optional[BarrierInventory] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Render a human-readable PRE report (mirrors ``AnalysisReport.summary`` style)."""
        lines = [
            "auditable :: PRE declared-plan analysis",
            f"  adapter:      {self.adapter}",
            f"  steps:        {self.n_steps}",
        ]
        if self.keystone_idx is not None:
            lines += [
                "",
                (
                    f"  execution keystone (structural chokepoint): step {self.keystone_idx} "
                    f"({self.keystone_followers} of {max(self.n_steps - 1, 0)} other steps "
                    "transitively follow it in control flow)"
                ),
                "    note: a STRUCTURAL design lint, not the POST blast-radius keystone",
            ]
        else:
            lines += ["", "  execution keystone: (none -- no control-flow followers)"]

        lines += ["", f"  lint findings: {len(self.findings)}"]
        for f in self.findings:
            rid = f"'{f.resource_id}'" if f.resource_id is not None else "-"
            lines.append(f"    [{f.severity}] {f.lint} @ step {f.node_idx} ({rid}): {f.detail}")

        lines += [
            "",
            "  State B (dependency-state) blast-share risk: WITHHELD",
            f"    reason: {self.state_b_withheld_reason}",
        ]

        pc = self.preflight_coverage
        if pc is not None:
            lines += [
                "",
                "  preflight coverage (descriptive, NOT a risk score):",
                (
                    f"    grade mix: observed={pc.observed} declared={pc.declared} "
                    f"inferred={pc.inferred} (of {pc.n_dep_edges} dependency edge(s))"
                ),
                f"    observed_fraction={pc.observed_fraction:.3f}  rho={pc.rho:.3f}",
                f"    runtime no-score reason: {pc.no_score_reason}",
            ]

        rtc = self.resource_touch_completeness
        if rtc is not None:
            lines += [
                "",
                "  resource-touch completeness (which declared touches carry a resource id):",
                f"    reads:  {rtc.reads_with_id}/{rtc.n_reads} identified",
                f"    writes: {rtc.writes_with_id}/{rtc.n_writes} identified",
                (
                    f"    dependency edges: {rtc.edges_with_id}/{rtc.n_edges} identified; "
                    f"{rtc.edges_missing_structured_resource} missing the structured "
                    "ResourceRef the runtime touch contract fills"
                ),
            ]
            for g in rtc.gaps:
                where = f"step {g.node_idx}" if g.src_idx is None else f"edge {g.node_idx}->{g.src_idx}"
                lines.append(f"      gap [{g.kind}] {where}: {g.detail}")

        bi = self.barrier_inventory
        if bi is not None:
            lines += ["", "  barrier inventory (declared revalidation re-reads, structure only):"]
            if bi.by_resource:
                for r, idxs in bi.by_resource.items():
                    nodes = ", ".join(f"step {i}" for i in idxs)
                    lines.append(f"    '{r}': {nodes}")
            else:
                lines.append("    (none -- no declared revalidation barriers)")

        if self.notes:
            lines += ["", "  notes:"]
            lines += [f"    - {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:  # so print(report) renders the summary
        return self.summary()

    def to_markdown(self, *, level: int = 1) -> str:
        """Render this PRE report as Markdown (the additive copy-pasteable form).

        Thin delegate to :func:`auditable.report.pre_to_markdown`; the plaintext
        ``summary`` / ``__str__`` are unchanged. Imported lazily to avoid an import
        cycle (``report.py`` imports this module).
        """
        from auditable.report import pre_to_markdown

        return pre_to_markdown(self, level=level)


def analyze_plan(plan: Any, *, adapter: Any = declared_plan_v1) -> PreReport:
    """Run PRE over a DECLARED plan: execution-topology keystone + reachability lints.

    Mirrors :func:`auditable.analysis.analyze_run`'s adapter -> SessionGraph flow, but
    computes only the PRE-honest parts. The plan is lowered through ``adapter`` (default
    :data:`declared_plan_v1`) into typed steps, projected to NetworkX once, and the four
    pure lints run over it. The execution-topology keystone is the argmax of
    :func:`execution_reach` (the structural chokepoint of the declared plan).

    State-B (dependency-state) blast-share risk is explicitly WITHHELD: the declared
    dependency layer is declared-only, so :func:`structural_risk` returns a
    ``no_score:*`` verdict (``no_score:low_coverage`` for a multi-step plan,
    ``no_score:single_decision`` for a 0- or 1-step plan). This function asserts that
    boundary -- the verdict must be some no_score state, never SCORED; if a scored
    verdict ever came back here, it raises rather than emit a number the evidence does
    not support. Scoring requires the ``graph`` extra (NetworkX); without it the
    projection raises a clear ``ImportError``.
    """
    from auditable.graph import execution_reach

    steps: List[Step] = list(adapter.to_steps(plan))
    graph = SessionGraph.from_steps(steps)

    # Boundary: the declared dependency layer must yield NO State-B score. The honest
    # invariant is "no number is emitted", i.e. any no_score:* state. A declared-only
    # dependency layer normally reads as no_score:low_coverage (observed_fraction=0),
    # but a small plan (0 or 1 step) is gated earlier as no_score:single_decision; both
    # legitimately withhold the score. Only a SCORED verdict violates the boundary, and
    # then we fail loudly rather than present a number PRE cannot honestly support.
    risk = structural_risk(graph)
    if risk.state == STATE_SCORED:
        raise AssertionError(
            "PRE invariant violated: structural_risk on a DECLARED plan returned "
            f"{risk.state!r}, expected a no_score:* state. A declared-only dependency "
            "layer must not yield a State-B blast-share score."
        )

    G = graph.to_networkx()
    execution_reach_by_idx = {s.idx: execution_reach(G, s.idx) for s in steps}
    if execution_reach_by_idx and max(execution_reach_by_idx.values()) > 0:
        keystone_idx = max(execution_reach_by_idx, key=lambda i: (execution_reach_by_idx[i], -i))
        keystone_followers = execution_reach_by_idx[keystone_idx]
    else:
        keystone_idx = None
        keystone_followers = 0

    findings: List[LintFinding] = []
    for lint in _LINTS:
        findings.extend(lint(G))

    adapter_id = (
        getattr(adapter, "id", None)
        or "_".join(p for p in (getattr(adapter, "name", ""), getattr(adapter, "version", "")) if p)
        or type(adapter).__name__
    )

    # The Preflight Coverage Report: descriptive coverage-readiness, NOT a risk
    # number. ``risk.coverage`` is the same EdgeCoverage structural_risk just read
    # off the graph, and ``risk.state`` is the no_score:* verdict asserted above, so
    # the no-score reason here is exactly the state the runtime scorer applied.
    preflight = PreflightCoverage.from_coverage(
        risk.coverage, len(steps), _no_score_reason(risk.state)
    )
    touch = resource_touch_completeness(G)
    barriers = barrier_inventory(G)

    notes = [
        "the execution keystone is the STRUCTURAL chokepoint of the declared plan "
        "(transitive control-flow followers via execution_reach); it is a design "
        "lint, not the POST blast-radius keystone and not a failure predictor.",
        "all four lints are pure, read-only NetworkX queries over the declared graph; "
        "the would-flip and drift-confirmation halves are runtime work, out of scope "
        "at PRE.",
        "the preflight coverage report is descriptive coverage-readiness (grade mix, "
        "the runtime no-score reason, resource-touch gaps, barrier structure), NOT a "
        "risk number; it leaves the State-B withhold boundary unchanged.",
        _WITHHELD_REASON,
    ]

    return PreReport(
        adapter=adapter_id,
        n_steps=len(steps),
        keystone_idx=keystone_idx,
        keystone_followers=keystone_followers,
        execution_reach_by_idx=execution_reach_by_idx,
        findings=findings,
        preflight_coverage=preflight,
        resource_touch_completeness=touch,
        barrier_inventory=barriers,
        notes=notes,
    )
