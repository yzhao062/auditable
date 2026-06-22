"""Declared-plan adapter: a framework-agnostic DECLARED agent plan to typed Steps.

This is the PRE-side ingestion seam. A DECLARED plan is the neutral target a
LangGraph compiled graph, a CrewAI task DAG, or an AutoGen topology would later
be lowered into; this adapter is explicitly NOT a parser for any of those. It
consumes a plain plan dict (with an optional thin dataclass mirror) and lowers it
onto the SAME typed :class:`~auditable.graph.session.Step` / ``DependencyEdge``
model the POST corpus adapters use, so PRE adds no new graph type, only a new
adapter plus read-only queries (see :mod:`auditable.graph.pre`).

The plan shape (``{"nodes": [<node>, ...]}``) maps 1:1 onto the typed model:

- ``idx`` (plain int, unique) -> ``Step.idx``; ``agent`` -> ``Step.agent``;
  ``kind`` (``"decision"`` | ``"tool_call"``) -> ``Step.kind``. The kind is
  validated up front with a clear error, not deferred to ``_validate_step``,
  because the kernel scores only those two node types.
- ``control_preds`` -> ``Step.exec_preds`` verbatim: omitted / ``None`` is the
  offline linear default (the single immediately-prior node), ``[]`` is an
  explicit root with no control predecessor, ``[i, j]`` is a merge of two
  control branches.
- ``reads`` that name a PRIOR producer node become a ``DependencyEdge`` graded
  :data:`Grade.DECLARED` (never observed / inferred), carrying the resource id and
  the ``volatile`` / ``pinned`` / ``revalidates`` flags under ``evidence``;
  ``resource`` stays ``None`` by the declared-corpus convention (matching
  ``own_record`` / ``tau_bench``), so the lints read ``evidence``, exactly as
  ``analyze_run._notes`` already inspects ``e.evidence``. A read with no
  ``producer`` named is a "free read": its resource id is retained in
  ``node_attrs`` so the lints still see it, but it wires no ``depends_on`` edge.
- ``writes`` mark the node as a mutation of a resource (a consequential action)
  for the lints; a write creates no edge by itself.
- ``scope`` is the granted tool-scope resource set, used only by scope-vs-snapshot.

Every per-node read / write / scope / barrier / volatile-read resource set is
recorded in ``node_attrs`` so the four PRE lints are pure node + edge queries
with no re-parse of the source plan.

Future framework lowering (a real LangGraph / CrewAI / AutoGen front-end) targets
this neutral plan dict; that work is out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..session import DependencyEdge, Grade, Step
from .protocol import _BaseAdapter

__all__ = ["DeclaredPlanNode", "DeclaredPlan", "DeclaredPlanAdapter", "declared_plan_v1"]


@dataclass
class DeclaredPlanNode:
    """A thin dataclass mirror of one plan node (the plain dict is authoritative).

    ``to_steps`` consumes plain dicts; this mirror is a convenience for callers that
    prefer typed construction. ``as_dict`` renders the dict shape the adapter reads,
    so ``DeclaredPlanAdapter().to_steps([n.as_dict() for n in nodes])`` is equivalent
    to building the dicts by hand.
    """

    idx: int
    agent: str
    kind: str
    control_preds: Optional[List[int]] = None
    reads: List[Any] = field(default_factory=list)
    writes: List[Any] = field(default_factory=list)
    scope: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"idx": self.idx, "agent": self.agent, "kind": self.kind}
        if self.control_preds is not None:
            out["control_preds"] = list(self.control_preds)
        if self.reads:
            out["reads"] = list(self.reads)
        if self.writes:
            out["writes"] = list(self.writes)
        if self.scope is not None:
            out["scope"] = list(self.scope)
        return out


@dataclass
class DeclaredPlan:
    """A thin dataclass mirror of a whole plan (the plain dict is authoritative)."""

    nodes: List[DeclaredPlanNode] = field(default_factory=list)
    plan_id: str = ""
    framework: str = ""

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"nodes": [n.as_dict() for n in self.nodes]}
        if self.plan_id:
            out["plan_id"] = self.plan_id
        if self.framework:
            out["framework"] = self.framework
        return out


_VALID_KINDS = ("decision", "tool_call")


def _coerce_ref(ref: Any, node_idx: int) -> Dict[str, Any]:
    """Normalize a <resource-ref> (bare string or dict) into a canonical dict.

    A bare string is the resource id; a dict must carry a string ``id`` and may
    carry ``producer`` / ``volatile`` / ``revalidates`` / ``pinned``. The shape is
    validated here so a malformed ref raises a clear error up front rather than
    failing later in graph construction.
    """
    if isinstance(ref, str):
        if not ref:
            raise ValueError(f"node {node_idx!r}: a resource-ref string must be non-empty")
        return {"id": ref, "producer": None, "volatile": False, "revalidates": False, "pinned": False}
    if not isinstance(ref, dict):
        raise ValueError(
            f"node {node_idx!r}: each resource-ref must be a string id or a dict, got {ref!r}"
        )
    rid = ref.get("id")
    if not isinstance(rid, str) or not rid:
        raise ValueError(f"node {node_idx!r}: a resource-ref dict requires a non-empty string 'id'")
    producer = ref.get("producer")
    if producer is not None and (not isinstance(producer, int) or isinstance(producer, bool)):
        raise ValueError(
            f"node {node_idx!r}: resource-ref 'producer' must be a plain int idx, got {producer!r}"
        )
    return {
        "id": rid,
        "producer": producer,
        "volatile": bool(ref.get("volatile", False)),
        "revalidates": bool(ref.get("revalidates", False)),
        "pinned": bool(ref.get("pinned", False)),
    }


class DeclaredPlanAdapter(_BaseAdapter):
    """Lower a framework-agnostic DECLARED plan dict into typed steps.

    ``name='declared_plan'`` / ``version='v1'`` (``id='declared_plan_v1'``), callable
    via the inherited ``__call__``. :meth:`to_steps` validates the plan shape and each
    node, then emits one :class:`Step` per node with DECLARED-graded dependency edges
    only. It is the seam a real LangGraph / CrewAI / AutoGen front-end would target;
    it does not parse any of those frameworks itself.
    """

    name = "declared_plan"
    version = "v1"

    def to_steps(self, plan: Any) -> List[Step]:
        """Validate the declared-plan schema and lower it to typed steps.

        The top-level shape must be ``{"nodes": [...]}`` (an empty / ``None`` plan
        yields ``[]``). Each node's ``idx`` must be a plain, unique int, ``kind`` must
        be ``"decision"`` or ``"tool_call"``, and every resource-ref must be a valid
        string or dict. Reads naming a prior producer become DECLARED dependency
        edges carrying the resource id and flags in ``evidence``; reads / writes /
        scope / barriers / volatile-reads are recorded in ``node_attrs`` so the PRE
        lints are pure node + edge queries.
        """
        if not plan:
            return []
        if isinstance(plan, DeclaredPlan):
            plan = plan.as_dict()
        if not isinstance(plan, dict) or "nodes" not in plan:
            raise ValueError("a declared plan must be a dict with a 'nodes' list")
        nodes = plan.get("nodes") or []
        if not isinstance(nodes, (list, tuple)):
            raise ValueError("'nodes' must be a list of node dicts")

        seen: set = set()
        steps: List[Step] = []
        for node in nodes:
            if isinstance(node, DeclaredPlanNode):
                node = node.as_dict()
            if not isinstance(node, dict):
                raise ValueError(f"each plan node must be a dict, got {node!r}")
            idx = node.get("idx")
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError(f"node 'idx' must be a plain int (no bool / float / str), got {idx!r}")
            if idx in seen:
                raise ValueError(f"duplicate node 'idx' {idx!r}: plan idx values must be unique")
            seen.add(idx)
            agent = node.get("agent")
            if not isinstance(agent, str) or not agent:
                raise ValueError(f"node {idx!r}: 'agent' must be a non-empty string")
            kind = node.get("kind")
            if kind not in _VALID_KINDS:
                raise ValueError(
                    f"node {idx!r}: 'kind' must be one of {_VALID_KINDS}, got {kind!r}; "
                    "the graph kernel scores only these two node types"
                )

            control_preds = node.get("control_preds")
            if control_preds is not None and not isinstance(control_preds, (list, tuple)):
                raise ValueError(f"node {idx!r}: 'control_preds' must be a list of ints or omitted")
            if control_preds is not None:
                for p in control_preds:
                    if not isinstance(p, int) or isinstance(p, bool):
                        raise ValueError(
                            f"node {idx!r}: 'control_preds' entries must be plain int idxs, got {p!r}"
                        )
                    if p == idx or p not in seen:
                        raise ValueError(
                            f"node {idx!r}: 'control_preds' references {p!r}, which is not a prior "
                            "declared node (list nodes in topological order; use [] for a root)"
                        )
            exec_preds = list(control_preds) if control_preds is not None else None

            reads = [_coerce_ref(r, idx) for r in (node.get("reads") or ())]
            writes = [_coerce_ref(w, idx) for w in (node.get("writes") or ())]
            scope = node.get("scope")
            if scope is not None and not isinstance(scope, (list, tuple)):
                raise ValueError(f"node {idx!r}: 'scope' must be a list of resource ids or omitted")

            deps = self._declared_deps(reads, seen, idx)
            node_attrs = {
                "reads": [r["id"] for r in reads],
                "writes": [w["id"] for w in writes],
                "scope": list(scope) if scope is not None else None,
                "volatile_reads": [r["id"] for r in reads if r["volatile"]],
                "barriers": [r["id"] for r in reads if r["revalidates"]],
            }
            steps.append(
                Step(
                    idx=idx,
                    agent=agent,
                    kind=kind,
                    deps=deps,
                    node_attrs=node_attrs,
                    exec_preds=exec_preds,
                )
            )
        return steps

    def _declared_deps(
        self, reads: List[Dict[str, Any]], declared_idxs: set, node_idx: int
    ) -> List[DependencyEdge]:
        """One DECLARED dependency edge per read that names an already-declared producer.

        A read whose ``producer`` is a prior (already-declared, lower-idx) node wires a
        ``depends_on`` edge graded :data:`Grade.DECLARED`, with the resource id and the
        ``volatile`` / ``pinned`` / ``revalidates`` flags under ``evidence`` and
        ``resource`` left ``None`` (the declared-corpus convention). A read with no
        ``producer`` named is a free read: it carries no edge here (its id is still kept
        in ``node_attrs`` by the caller). An explicitly named ``producer`` must be a
        prior declared node, else this raises; list nodes in topological / issue order.
        """
        edges: List[DependencyEdge] = []
        for r in reads:
            producer = r["producer"]
            if producer is None:
                continue  # free read: no producer named -> no edge (id kept in node_attrs)
            if producer == node_idx or producer not in declared_idxs:
                raise ValueError(
                    f"node {node_idx!r}: read of {r['id']!r} names 'producer' {producer!r}, which "
                    "is not a prior declared node; omit 'producer' for an external free read"
                )
            edges.append(
                DependencyEdge(
                    src_idx=producer,
                    grade=Grade.DECLARED,
                    resource=None,
                    evidence={
                        "declared": True,
                        "resource_id": r["id"],
                        "volatile": r["volatile"],
                        "pinned": r["pinned"],
                        "revalidates": r["revalidates"],
                        "relation": "declared_read",
                        "adapter": self.id,
                    },
                )
            )
        return edges


# The versioned singleton, callable as declared_plan_v1(plan).
declared_plan_v1 = DeclaredPlanAdapter()
