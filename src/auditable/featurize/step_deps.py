"""Step-only dependency graphs and the per-step series online and Gold-style methods read.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.2: ``gold._step_graph``, ``gold._run_maxspan``, ``gold._run_edges``,
``live._span_series``, ``live._count_series`` and ``live._online_z``. All of them read
only the trace's own ``deps`` lists, and every per-step value at position ``i`` uses
steps ``0..i`` only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

STEP_GRAPH_FEATURES = ("position", "is_tool", "dep_count", "max_span")


def _steps(trace: Any) -> List[Dict[str, Any]]:
    from auditable.featurize.trace import normalize_trace

    return normalize_trace(trace).steps


def step_graph(trace: Any):
    """``(X, edge_index)``: per-step features ``STEP_GRAPH_FEATURES`` as an (n, 4) float
    array, and the dependency edges as a (2, m) int64 array pointing current -> prior (the
    direction of ``depends_on``). Dependencies on steps outside the trace or not earlier in
    it are ignored."""
    import numpy as np

    steps = _steps(trace)
    idx_of = {s["idx"]: i for i, s in enumerate(steps)}
    n = len(steps)
    feats, src, dst = [], [], []
    for i, s in enumerate(steps):
        dep_pos = [idx_of[d] for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i]
        feats.append([
            i / max(1, n - 1),
            1.0 if s.get("kind") == "tool_call" else 0.0,
            float(len(dep_pos)),
            float(i - min(dep_pos)) if dep_pos else 0.0,
        ])
        for dp in dep_pos:
            src.append(i)
            dst.append(dp)
    edges = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    return np.array(feats, dtype=float), edges


def run_max_span(trace: Any) -> float:
    """The largest dependency span in the run (how far back any step reaches)."""
    X, _ = step_graph(trace)
    return float(X[:, 3].max()) if len(X) else 0.0


def run_edge_count(trace: Any) -> float:
    """The number of in-trace dependency edges."""
    _, e = step_graph(trace)
    return float(e.shape[1])


def span_series(trace: Any):
    """Per-step maximum dependency span: position ``i`` minus the earliest in-trace step
    it depends on, 0 for a step with no earlier dependency."""
    import numpy as np

    steps = _steps(trace)
    idx_of = {s["idx"]: i for i, s in enumerate(steps)}
    out = np.zeros(len(steps))
    for i, s in enumerate(steps):
        dep_pos = [idx_of[d] for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i]
        out[i] = float(i - min(dep_pos)) if dep_pos else 0.0
    return out


def count_series(trace: Any):
    """Per-step count of in-trace dependencies on earlier steps."""
    import numpy as np

    steps = _steps(trace)
    idx_of = {s["idx"]: i for i, s in enumerate(steps)}
    out = np.zeros(len(steps))
    for i, s in enumerate(steps):
        out[i] = float(sum(1 for d in s.get("deps", ()) if d in idx_of and idx_of[d] < i))
    return out


def causal_z(series: Sequence[float]):
    """Causal z-score: each value against the values strictly before it. Positions with
    fewer than two earlier values, or with a constant history, score 0."""
    import numpy as np

    series = np.asarray(series, dtype=float)
    z = np.zeros(len(series))
    for t in range(len(series)):
        prior = series[:t]
        if len(prior) >= 2 and prior.std() > 1e-9:
            z[t] = (series[t] - prior.mean()) / prior.std()
    return z
