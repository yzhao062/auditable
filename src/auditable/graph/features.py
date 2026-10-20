"""Layered structural features for an agent decision graph.

Three nested feature groups, matching the two-layer view of the representation:

  - ``flat``: size and counts only (no structure).
  - ``exec``: + execution-graph topology read from the agent / handoff structure
    (who acts, how concentrated, how control moves and returns). Deliberately uses
    agent-level structure, not the step chain, because the step handoff chain is
    just the run order and is collinear with ``n_steps``.
  - ``dep``: + dependency-layer structure from the ``depends_on`` DAG (how deep the
    chains run, the largest audit surface, the largest blast hub).

Two size-normalized groups sit beside them: ``dep_dens`` (repeated-resource edges
per step) and ``dep_shape`` (relative chain length and hub shares). They divide by
``n_steps``, which reports per-step rates and shares; it does not by itself remove a
feature's relationship to run size.

The groups are nested, so adjacent ablations compare strict feature prefixes and
each contrast is a paired difference between two fits of one learner.

What the contrast does not show: a detector that beats ``flat`` is not evidence that
the typed structure carries signal beyond run size. ``flat`` feeds a linear fit on
four size counts, so a nonlinear function of run size inside the dependency block
is scored as dependency signal. GRADE retired that reading on 2026-09-08; on
SWE-Gym a cubic-spline fit on the same four counts reaches 0.832 against the
dependency model's 0.804. Beating ``flat`` is a measurement of one model pair on one
corpus.

Caveat about the dependency group: it carries independent signal only where the
dependency edges are OBSERVED and sparse (tool I/O, file reads). Where they are
INFERRED under a full-context assumption (every step depends on every prior step),
``dep_depth`` and ``n_dep_edges`` are functions of ``n_steps`` and add nothing the
flat group does not already have. The structural scorer withholds a score (low
coverage) when the dependency layer is mostly inferred or saturated.

Ported from GRADE (https://github.com/yzhao062/grade, MIT License),
``src/grade/features.py`` at commit 3839a57; the numeric code is unchanged, and the
docstring follows GRADE's later revision.
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
    """Return named structural features grouped into ``flat`` / ``exec`` / ``dep`` /
    ``dep_dens`` / ``dep_shape``.

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

    # Size-normalized dependency block: the raw counts above are collinear with
    # n_steps, so they re-encode run size. Dividing by n_steps reports per-step rates
    # and shares instead of absolute volume and depth; it does not separate shape
    # from size by construction.
    ns = max(n_steps, 1)
    # dep_dens is how often steps re-touch state; dep_shape is how deep the chains
    # run and how concentrated the hubs are.
    dep_dens = {
        "dep_edges_per_step": dep.number_of_edges() / ns,
    }
    dep_shape = {
        "rel_dep_depth": dep_depth / ns,
        "max_blast_share": max(indeg) / ns,
        "max_audit_share": max(outdeg) / ns,
    }

    return {"flat": flat, "exec": exec_, "dep": dep_, "dep_dens": dep_dens, "dep_shape": dep_shape}


def feature_vector(G, *, layer: str = "full"):
    """Flatten :func:`layered_features` into an ordered (names, values) pair.

    ``layer`` selects the column set. Nested size groups: ``"flat"`` (size and
    counts), ``"exec"`` (flat + execution topology), ``"full"`` (flat + exec + raw
    dependency counts). Size-normalized variants: ``"flatdens"`` (flat + revisit
    density), ``"flatdep"`` (flat + size-normalized dependency density and shape),
    and ``"depnorm"`` (size-normalized dependency only). Names are returned so
    callers can keep a stable column order across runs.
    """
    groups = layered_features(G)
    order = {
        "flat": ["flat"],
        "exec": ["flat", "exec"],
        "full": ["flat", "exec", "dep"],
        "flatdens": ["flat", "dep_dens"],
        "flatdep": ["flat", "dep_dens", "dep_shape"],
        "depnorm": ["dep_dens", "dep_shape"],
    }
    if layer not in order:
        raise ValueError(f"layer must be one of {sorted(order)}, got {layer!r}")
    names, values = [], []
    for g in order[layer]:
        for k, v in groups[g].items():
            names.append(k)
            values.append(float(v))
    return names, values
