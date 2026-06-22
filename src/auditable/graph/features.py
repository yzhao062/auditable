"""Layered structural features for an agent decision graph.

Three nested feature groups, matching the two-layer view of the representation:

  - ``flat``: size and counts only (no structure).
  - ``exec``: + execution-graph topology read from the agent / handoff structure
    (who acts, how concentrated, how control moves and returns). Deliberately uses
    agent-level structure, not the step chain, because the step handoff chain is
    just the run order and is collinear with ``n_steps``.
  - ``dep``: + dependency-layer structure from the ``depends_on`` DAG (how deep the
    chains run, the largest audit surface, the largest blast hub).

The groups are nested so callers can inspect strict feature prefixes
(``flat``, ``flat + exec``, ``flat + exec + dep``) without changing the
downstream schema.

Honest caveat about the dependency group: it carries independent signal only where
the dependency edges are OBSERVED and sparse (tool I/O, file reads). Where they are
INFERRED under a full-context assumption (every step depends on every prior step),
``dep_depth`` and ``n_dep_edges`` are functions of ``n_steps`` and add nothing the
flat group does not already have. The dependency group is therefore meaningful only
on the observed-and-sparse case; the structural scorer withholds a score (low
coverage) when the dependency layer is mostly inferred or saturated.
"""
from __future__ import annotations

from collections import Counter


def _gini(values) -> float:
    """Gini concentration of a list of non-negative counts; 0 for empty/uniform/singleton."""
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    n = len(vals)
    diffs = sum(abs(a - b) for a in vals for b in vals)
    return diffs / (2.0 * n * total)


def layered_features(G) -> dict:
    """Return named structural features grouped into ``flat`` / ``exec`` / ``dep``.

    The graph is a typed ``MultiDiGraph`` from :func:`auditable.graph.build_graph`:
    agent nodes emit step nodes, step nodes form a handoff chain, and ``depends_on``
    edges (observed or inferred) form the dependency layer.
    """
    try:
        import networkx as nx
    except Exception:  # pragma: no cover - exercised only where the extra is absent
        raise ImportError("auditable.graph requires the 'graph' extra: pip install auditable[graph]")
    from auditable.graph import dependency_dag  # lazy import avoids any package cycle

    step_nodes = [(d["idx"], d["agent"]) for _, d in G.nodes(data=True)
                  if d.get("ntype") in ("decision", "tool_call")]
    step_nodes.sort(key=lambda t: t[0])
    agent_seq = [a for _, a in step_nodes]
    n_steps = len(agent_seq)
    n_tool = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "tool_call")
    n_agents = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "agent")

    flat = {
        "n_steps": n_steps,
        "n_tool_calls": n_tool,
        "n_decisions": n_steps - n_tool,
        "n_agents": n_agents,
    }

    # execution topology: agent activity concentration and how control moves / returns
    counts = Counter(agent_seq)
    changes = sum(1 for i in range(1, n_steps) if agent_seq[i] != agent_seq[i - 1])
    transitions, returns, seen = set(), 0, set()
    for i, a in enumerate(agent_seq):
        if i and a != agent_seq[i - 1]:
            transitions.add((agent_seq[i - 1], a))
            if a in seen:
                returns += 1
        seen.add(a)
    exec_ = {
        "max_agent_outdeg": max(counts.values()) if counts else 0,
        "agent_gini": _gini(list(counts.values())),
        "n_agent_transitions": len(transitions),
        "agent_recurrence": (returns / changes) if changes else 0.0,
    }

    # dependency layer: shape of the depends_on DAG (step -> what it relied on)
    dep = dependency_dag(G)
    if dep.number_of_nodes():
        indeg = [d for _, d in dep.in_degree()]
        outdeg = [d for _, d in dep.out_degree()]
        dep_depth = nx.dag_longest_path_length(dep) if nx.is_directed_acyclic_graph(dep) else 0
    else:
        indeg, outdeg, dep_depth = [0], [0], 0
    # depends_on points dependent -> dependency: a step's out-degree is its audit
    # surface (what it rests on), a step's in-degree is its blast (who rests on it).
    dep_ = {
        "n_dep_edges": dep.number_of_edges(),
        "dep_depth": dep_depth,
        "max_blast_indeg": max(indeg),
        "max_audit_outdeg": max(outdeg),
    }

    return {"flat": flat, "exec": exec_, "dep": dep_}


def feature_vector(G, *, layer: str = "full"):
    """Flatten :func:`layered_features` into an ordered (names, values) pair.

    ``layer`` selects the nested group: ``"flat"``, ``"exec"`` (flat+exec), or
    ``"full"`` (flat+exec+dep). Names are returned so callers can keep a stable
    column order across runs.
    """
    groups = layered_features(G)
    order = {"flat": ["flat"], "exec": ["flat", "exec"], "full": ["flat", "exec", "dep"]}
    if layer not in order:
        raise ValueError(f"layer must be one of {sorted(order)}, got {layer!r}")
    names, values = [], []
    for g in order[layer]:
        for k, v in groups[g].items():
            names.append(k)
            values.append(float(v))
    return names, values
