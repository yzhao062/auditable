"""Heterogeneous decision-graph construction and characterization (the kernel).

The signed records are the source of truth; this projects a normalized agent
trace into a NetworkX ``MultiDiGraph`` for analysis and characterization. It is
torch-free (NetworkX only); the heavy graph-OD stack (PyG, PyGOD) is optional and
lives benchmark-side.

Two edge classes, matching the attachment model: execution edges (``emits``,
``handoff_to``) are observed from the trace (State A); dependency edges
(``depends_on``) are inferred or declared (State B) and are never read off the
trace. The graph is a first-class, queryable part of the package; ``audit()``
stays the ergonomic capture entry and the user is never asked to build the graph
by hand.
"""
from __future__ import annotations

from typing import Sequence

try:
    import networkx as nx

    _HAS_NX = True
except Exception:  # pragma: no cover - exercised only where the extra is absent
    _HAS_NX = False

NODE_TYPES = ("agent", "decision", "tool_call", "dependency_resource")
EXECUTION_EDGES = ("emits", "handoff_to")
DEPENDENCY_EDGES = ("depends_on", "reads", "writes")


def build_graph(steps: Sequence[dict], *, dependency: str = "full_context", shared_resource: bool = True):
    """Build the typed decision graph from a normalized trace.

    ``steps`` are ordered dicts with keys ``idx`` (a unique int), ``agent`` (str),
    and ``kind`` (``"decision"`` or ``"tool_call"``). Execution edges come from the
    trace. Dependency edges follow ``dependency``: ``"full_context"`` (each step
    depends on every prior step, the full-history assumption these multi-agent
    systems actually use), ``"chain"`` (each step depends only on its immediate
    predecessor), or ``"explicit"`` (each step depends on exactly the prior steps
    named in its own ``deps`` list of indices). ``"explicit"`` is the observed-
    dependency case: when the trace exposes which state a step actually read
    (tool I/O, file reads), the adapter records it in ``deps`` instead of inferring
    it. ``shared_resource`` adds one ``dependency_resource`` node that every step
    reads and writes (the shared blackboard); pass ``False`` when dependencies are
    modeled explicitly and the coarse blackboard would double-count.
    """
    if not _HAS_NX:
        raise ImportError("auditable.graph requires the 'graph' extra: pip install auditable[graph]")
    idxs = [s["idx"] for s in steps]
    if any(type(i) is not int for i in idxs):
        raise ValueError("each step 'idx' must be a plain int (no float, bool, or str ids)")
    if len(set(idxs)) != len(idxs):
        raise ValueError("step 'idx' values must be unique")
    G = nx.MultiDiGraph()
    prior: list = []
    prior_ids: set = set()
    for s in steps:
        agent_node = f"agent::{s['agent']}"
        if not G.has_node(agent_node):
            G.add_node(agent_node, ntype="agent", name=s["agent"])
        step_node = f"step::{s['idx']}"
        G.add_node(step_node, ntype=s["kind"], agent=s["agent"], idx=s["idx"])
        G.add_edge(agent_node, step_node, etype="emits")
        if prior:
            G.add_edge(prior[-1], step_node, etype="handoff_to")
        if dependency == "full_context":
            for p in prior:
                G.add_edge(step_node, p, etype="depends_on")
        elif dependency == "chain" and prior:
            G.add_edge(step_node, prior[-1], etype="depends_on")
        elif dependency == "explicit":
            for d in dict.fromkeys(s.get("deps", ())):  # de-duped, order-preserving
                if d in prior_ids:  # only prior steps: drop self, forward, and unknown deps
                    G.add_edge(step_node, f"step::{d}", etype="depends_on")
        prior.append(step_node)
        prior_ids.add(s["idx"])
    if shared_resource and prior:
        res = "resource::shared_state"
        G.add_node(res, ntype="dependency_resource", name="shared_state")
        for sn in prior:
            G.add_edge(sn, res, etype="reads")
            G.add_edge(sn, res, etype="writes")
    return G


def _edges_of(G, etypes):
    return [(u, v) for u, v, d in G.edges(data=True) if d["etype"] in etypes]


def dependency_dag(G):
    """The ``depends_on`` projection as a simple DiGraph (step -> what it relied on)."""
    dep = nx.DiGraph()
    dep.add_nodes_from(n for n, d in G.nodes(data=True) if d["ntype"] in ("decision", "tool_call"))
    dep.add_edges_from(_edges_of(G, ("depends_on",)))
    return dep


def characterize(G) -> dict:
    """Structural properties of one decision graph (the per-graph measurement)."""
    if not _HAS_NX:
        raise ImportError("auditable.graph requires the 'graph' extra: pip install auditable[graph]")
    ntypes = [d["ntype"] for _, d in G.nodes(data=True)]
    type_counts = {t: ntypes.count(t) for t in NODE_TYPES if ntypes.count(t)}
    dep = dependency_dag(G)
    dep_depth = nx.dag_longest_path_length(dep) if dep.number_of_nodes() and nx.is_directed_acyclic_graph(dep) else 0
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_agents": type_counts.get("agent", 0),
        "n_steps": type_counts.get("decision", 0) + type_counts.get("tool_call", 0),
        "n_tool_calls": type_counts.get("tool_call", 0),
        "type_counts": type_counts,
        "n_exec_edges": len(_edges_of(G, EXECUTION_EDGES)),
        "n_dep_edges": len(_edges_of(G, ("depends_on",))),
        "dep_depth": dep_depth,
    }


def downstream_reach(G, step_idx: int) -> int:
    """How many steps (transitively) depend on ``step_idx`` -- its blast radius."""
    if not _HAS_NX:
        raise ImportError("auditable.graph requires the 'graph' extra: pip install auditable[graph]")
    dep = dependency_dag(G)
    node = f"step::{step_idx}"
    if node not in dep:
        return 0
    return len(nx.ancestors(dep, node))  # nodes with a depends_on path TO this one


# Layered structural features (flat / exec / dep), used by the failure-detection
# study to show each layer adds signal beyond run size. Imported last so the
# module-level names above are bound before features.py resolves them.
from .features import layered_features, feature_vector  # noqa: E402,F401
