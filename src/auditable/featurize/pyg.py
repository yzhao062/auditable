"""Typed graphs as (node features, edge_index) pairs for graph anomaly detectors.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.2, ``graph_ad.nx_to_graph``. Every node of the typed graph becomes a row, agent and
resource nodes included, in the graph's node insertion order.
"""
from __future__ import annotations

from typing import Any

NODE_FEATURES = ("in_degree", "out_degree", "is_decision", "is_tool_call", "is_other")


def to_pyg(graph: Any):
    """``(X, edge_index)`` for a typed NetworkX graph: in and out degree (total degree and
    0 for an undirected graph) and a one-hot over decision / tool-call / other node types,
    with edges as a (2, m) int64 array. A :class:`~auditable.graph.session.SessionGraph`
    is projected with ``to_networkx()`` first."""
    import numpy as np

    if hasattr(graph, "to_networkx") and hasattr(graph, "coverage"):
        graph = graph.to_networkx()
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    directed = graph.is_directed()
    feats = []
    for node in nodes:
        ntype = graph.nodes[node].get("ntype", "")
        in_deg = float(graph.in_degree(node)) if directed else float(graph.degree(node))
        out_deg = float(graph.out_degree(node)) if directed else 0.0
        feats.append([
            in_deg,
            out_deg,
            1.0 if ntype == "decision" else 0.0,
            1.0 if ntype == "tool_call" else 0.0,
            1.0 if ntype not in ("decision", "tool_call") else 0.0,
        ])
    src, dst = [], []
    for u, v in graph.edges():
        src.append(index[u])
        dst.append(index[v])
    edges = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    return np.array(feats, dtype=float), edges
