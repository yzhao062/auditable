"""Feature extraction shared by the auditing methods.

Every function reads its trace through :func:`normalize_trace`, so a
``SessionGraph``, GRADE-style step dicts, a GRADE run dict, or a typed NetworkX graph all
work. Run-level structural layers come from :mod:`auditable.graph.features`
(``feature_vector`` with ``layer`` in flat, exec, full, flatdens, flatdep, depnorm).
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from auditable.featurize.pyg import NODE_FEATURES, to_pyg
from auditable.featurize.step_deps import (
    STEP_GRAPH_FEATURES,
    causal_z,
    count_series,
    run_edge_count,
    run_max_span,
    span_series,
    step_graph,
)
from auditable.featurize.spec import AgentConfig, as_agent_config, derive_spec_features
from auditable.featurize.trace import TraceView, full_context_edges, normalize_trace
from auditable.featurize.who_when import STEP_FEATURES, agent_handoff_degrees, who_when_step_matrix

GRADE_LAYERS = ("flat", "exec", "full", "flatdens", "flatdep", "depnorm")


def grade_layer(trace: Any, layer: str = "full", *, dependency: str = "explicit",
                shared_resource: bool = True) -> Tuple[List[str], List[float]]:
    """``(names, values)`` of one GRADE structural layer for one trace. ``dependency`` and
    ``shared_resource`` apply only when the graph is built from the trace's steps; a typed
    graph passed in is read as it is."""
    from auditable.graph import feature_vector

    G = normalize_trace(trace).typed_graph(dependency=dependency, shared_resource=shared_resource)
    return feature_vector(G, layer=layer)


def grade_layer_matrix(traces: Sequence[Any], layer: str = "full", *, dependency: str = "explicit",
                       shared_resource: bool = True):
    """An (n_traces, n_features) float array of one GRADE layer, rows in input order."""
    import numpy as np

    return np.array([grade_layer(t, layer, dependency=dependency, shared_resource=shared_resource)[1]
                     for t in traces], dtype=float)


__all__ = [
    "AgentConfig",
    "GRADE_LAYERS",
    "NODE_FEATURES",
    "STEP_FEATURES",
    "STEP_GRAPH_FEATURES",
    "TraceView",
    "agent_handoff_degrees",
    "as_agent_config",
    "causal_z",
    "count_series",
    "derive_spec_features",
    "full_context_edges",
    "grade_layer",
    "grade_layer_matrix",
    "normalize_trace",
    "run_edge_count",
    "run_max_span",
    "span_series",
    "step_graph",
    "to_pyg",
    "who_when_step_matrix",
]
