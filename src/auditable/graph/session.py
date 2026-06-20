"""The typed SessionGraph: the v0.3 stable foundation.

A typed wrapper over the :func:`auditable.graph.build_graph` kernel. The kernel
takes one graph-level dependency mode and stores only ``ntype`` / ``agent`` /
``idx`` on nodes; this layer carries what it does not, so the offline v0.3 path
and the v0.3b additions share one schema:

- a per-edge attachment ``grade`` (observed / declared / inferred), so the three
  coexist in one graph rather than one mode for the whole graph;
- per-edge ``evidence`` and an optional ``resource`` identity, the seam the v0.3b
  own-record resource-touch contract fills (corpus adapters leave ``resource``
  ``None`` and record modeled evidence);
- node attributes (``model_id``, ``decision_basis``), the model facet kept under
  the homogeneous-model assumption;
- explicit ``completeness`` (``complete`` offline now, ``prefix`` for the v0.3b
  live path) and dependency-edge ``coverage`` with the saturation ratio ``rho``.

v0.3 implements the offline path against these seams; live scoring and own-record
observed edges are additive, not breaking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import comb
from typing import Any, Dict, List, Optional, Sequence


class Grade(str, Enum):
    """How a dependency edge is known (the attachment model). ``inferred`` is the
    weakest (a full-history assumption); ``observed`` is a logged read/write match."""

    OBSERVED = "observed"
    DECLARED = "declared"
    INFERRED = "inferred"


@dataclass(frozen=True)
class ResourceRef:
    """A canonical resource-touch identity for an observed dependency edge.

    Filled by the v0.3b own-record touch contract (a later read of
    ``{namespace, resource_id, key}`` matched to an earlier write of the same
    resource). ``None`` on corpus-modeled edges this round.
    """

    namespace: str
    resource_id: str
    key: str = ""


@dataclass
class DependencyEdge:
    """One dependency edge: the owning step relied on prior step ``src_idx``.

    ``grade`` is per edge, not one mode for the whole graph. ``resource`` and
    ``evidence`` are the seams for the v0.3b own-record touch contract; corpus
    adapters leave ``resource`` ``None`` and record modeled evidence. The default
    is the weakest grade so nothing is silently presented as observed.
    """

    src_idx: int
    grade: Grade = Grade.INFERRED
    resource: Optional[ResourceRef] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """One decision or tool step.

    ``node_attrs`` carries the model facet (``model_id``, ``decision_basis``) and
    any future node attributes; the homogeneous-model assumption means these are
    present but not yet branched on.

    The last three fields are the load-bearing real-time seams (offline populates
    them trivially, so later live integration is a pure addition):

    - ``exec_preds``: explicit execution predecessors, so a concurrent / branched
      topology is representable instead of the linear handoff chain. ``None`` means
      the offline default (the single previous step, if any); ``[]`` means a root
      with no execution predecessor (distinct from ``None`` for parallel roots);
      ``[i, j]`` is a merge of two branches.
    - ``correlation_id``: a pre-digest live event id, so a framework adapter can
      assemble data / model / harness fragments that arrive out of order. It is
      distinct from the record digest, which is assigned only after the complete
      record is signed and is therefore too late for live assembly.
    - ``record_id``: the final signed record digest, carried when available (set
      offline from the loaded record; set live once the record is sealed).
    """

    idx: int
    agent: str
    kind: str  # "decision" | "tool_call"
    deps: List[DependencyEdge] = field(default_factory=list)
    node_attrs: Dict[str, Any] = field(default_factory=dict)
    exec_preds: Optional[List[int]] = None
    correlation_id: Optional[str] = None
    record_id: Optional[str] = None


class GraphCompleteness(str, Enum):
    """Whether the graph is the whole run or a live prefix."""

    COMPLETE = "complete"  # offline: the whole run
    PREFIX = "prefix"  # live, v0.3b: the session so far


@dataclass
class EdgeCoverage:
    """Dependency-edge coverage: how much is observed versus inferred, and ``rho``.

    ``rho`` is the saturation ratio ``|E_dep| / C(n_steps, 2)``; near 1 means the
    dependency layer has collapsed toward the full-history degenerate regime, where
    its structural features add nothing over run size.
    """

    n_dep_edges: int
    by_grade: Dict[Grade, int]
    rho: float
    observed_fraction: float


def _validate_step(step: "Step") -> None:
    """Reject a malformed step early so the projection and the score stay consistent.

    ``idx`` and every dependency / execution predecessor id must be a plain ``int``
    (``bool`` excluded, since ``True == 1`` would alias another step's ``step::<idx>``
    projection node). ``idx`` uniqueness is enforced by the graph, not here. ``kind``
    must be ``"decision"`` or ``"tool_call"``: the graph kernel scores only those two
    ``ntype`` values (``layered_features`` and ``dependency_dag`` count a node as a step
    only when its ``ntype`` is one of them), so an unrecognized kind would be counted by
    the size-normalized risk denominator yet dropped by the feature layer, a silent
    mismeasurement. A later kernel that scores any ``idx``-bearing node can relax this.
    """
    if not isinstance(step.idx, int) or isinstance(step.idx, bool):
        raise ValueError(f"Step.idx must be a plain int, got {step.idx!r}")
    if step.kind not in ("decision", "tool_call"):
        raise ValueError(
            f"Step.kind must be 'decision' or 'tool_call', got {step.kind!r}; the "
            "graph kernel scores only these two node types"
        )
    for e in step.deps:
        if not isinstance(e.src_idx, int) or isinstance(e.src_idx, bool):
            raise ValueError(
                f"DependencyEdge.src_idx must be a plain int, got {e.src_idx!r} "
                f"on step {step.idx!r}"
            )
    if step.exec_preds is not None:
        for p in step.exec_preds:
            if not isinstance(p, int) or isinstance(p, bool):
                raise ValueError(
                    f"Step.exec_preds entries must be plain ints, got {p!r} "
                    f"on step {step.idx!r}"
                )


class SessionGraph:
    """A typed session graph over decision steps.

    Holds the typed steps and their graded dependency edges; the NetworkX
    projection and scoring (added next) build on the existing ``build_graph``
    kernel. ``from_steps`` is the offline (batch) entry; ``add_step`` is reserved
    for the v0.3b live path so live scoring is additive.
    """

    def __init__(
        self,
        steps: Sequence[Step],
        *,
        completeness: GraphCompleteness = GraphCompleteness.COMPLETE,
    ) -> None:
        self.steps: List[Step] = list(steps)
        self.completeness = completeness
        seen: set = set()
        for s in self.steps:
            _validate_step(s)
            if s.idx in seen:
                raise ValueError(
                    f"duplicate Step.idx {s.idx!r}: step ids must be unique so the "
                    "projection and the size-normalized risk agree"
                )
            seen.add(s.idx)

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[Step],
        *,
        completeness: GraphCompleteness = GraphCompleteness.COMPLETE,
    ) -> "SessionGraph":
        return cls(steps, completeness=completeness)

    def add_step(self, step: Step) -> None:
        """Append a step. Reserved for the v0.3b live path; the offline path uses
        ``from_steps``. Present now so live scoring lands without an API change."""
        _validate_step(step)
        if any(s.idx == step.idx for s in self.steps):
            raise ValueError(
                f"duplicate Step.idx {step.idx!r}: step ids must be unique"
            )
        self.steps.append(step)

    def coverage(self) -> EdgeCoverage:
        """Aggregate dependency-edge coverage and the saturation ratio ``rho``."""
        by_grade: Dict[Grade, int] = {g: 0 for g in Grade}
        n_dep = 0
        for s in self.steps:
            for e in s.deps:
                by_grade[e.grade] += 1
                n_dep += 1
        n_steps = len(self.steps)
        denom = comb(n_steps, 2) if n_steps >= 2 else 0
        rho = (n_dep / denom) if denom else 0.0
        observed_fraction = (by_grade[Grade.OBSERVED] / n_dep) if n_dep else 0.0
        return EdgeCoverage(
            n_dep_edges=n_dep,
            by_grade=by_grade,
            rho=rho,
            observed_fraction=observed_fraction,
        )

    def to_networkx(self):
        """Project to a typed NetworkX ``MultiDiGraph`` that preserves what the
        low-level :func:`auditable.graph.build_graph` kernel drops.

        The kernel keeps only ``ntype`` / ``agent`` / ``idx`` on step nodes, stores
        no per-edge attributes on ``depends_on`` edges, and reads execution as a
        single linear handoff chain. This projection carries all three typed seams
        through:

        - **node attributes** (``model_id``, ``decision_basis``, and any other
          ``node_attrs``) on each step node, with the core typing kept authoritative
          so a stray attribute cannot clobber ``ntype`` / ``agent`` / ``idx``;
        - **per-edge ``grade`` / ``evidence`` / ``resource``** on each ``depends_on``
          edge, so observed / declared / inferred coexist in one graph;
        - **execution edges from ``exec_preds``**: ``None`` is the single linear
          previous step, ``[]`` is a root with no execution predecessor (distinct
          from ``None`` for a parallel root), and ``[i, j]`` is a merge of two
          branches.

        Node and edge conventions match the kernel (agent nodes emit typed step
        nodes named ``step::<idx>``; ``depends_on`` points dependent -> dependency),
        so :func:`characterize`, :func:`dependency_dag`, :func:`downstream_reach`,
        and :func:`layered_features` read this graph unchanged. Only dependency and
        execution edges to already-seen steps are wired, keeping the projection
        causal and the ``depends_on`` DAG acyclic.
        """
        try:
            import networkx as nx
        except Exception as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "auditable.graph requires the 'graph' extra: pip install auditable[graph]"
            ) from exc

        G = nx.MultiDiGraph()
        seen: set = set()
        prev_idx: Optional[int] = None  # the immediately prior step (the linear default)
        for s in self.steps:
            agent_node = f"agent::{s.agent}"
            if not G.has_node(agent_node):
                G.add_node(agent_node, ntype="agent", name=s.agent)
            step_node = f"step::{s.idx}"
            # node_attrs first, then the core typing, so node_attrs cannot clobber it
            node_data = {**dict(s.node_attrs), "ntype": s.kind, "agent": s.agent, "idx": s.idx}
            G.add_node(step_node, **node_data)
            G.add_edge(agent_node, step_node, etype="emits")

            # execution layer (harness): DAG-ready via exec_preds
            if s.exec_preds is None:
                preds = [prev_idx] if prev_idx is not None else []
            else:
                preds = list(s.exec_preds)
            for p in preds:
                if p in seen:
                    G.add_edge(f"step::{p}", step_node, etype="handoff_to")

            # dependency layer (data): typed and graded per edge
            for e in s.deps:
                if e.src_idx in seen:
                    G.add_edge(
                        step_node,
                        f"step::{e.src_idx}",
                        etype="depends_on",
                        grade=e.grade,
                        evidence=dict(e.evidence),
                        resource=e.resource,
                    )

            seen.add(s.idx)
            prev_idx = s.idx
        return G
