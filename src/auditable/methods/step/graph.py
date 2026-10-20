"""Graph anomaly detection as step localization (transductive).

A DOMINANT graph autoencoder (Ding et al., 2019, via PyGOD) is fit over a batch of
per-trace step graphs; a step the model reconstructs poorly is the suspected fault. Two
step representations are registered separately because they are different methods:

- ``step.pygod_dominant_ww8``: the eight-column Who&When step matrix with full-context
  dependency edges (CatchBench ``PyGODLocalization``);
- ``step.pygod_dominant_sg4``: the four-column step graph (position, is_tool, dependency
  count, maximum span) with the trace's own dependency edges (CatchBench ``GoldPyGOD``).
"""
from __future__ import annotations

from typing import Any, List, Sequence

from auditable.base import BaseStepLocalizer, Citation, MethodCard
from auditable.methods import register

_DOMINANT = Citation(
    key="DOMINANT (Ding et al., 2019)",
    title="Deep Anomaly Detection on Attributed Networks",
    url="https://doi.org/10.1137/1.9781611975673.67",
)
_PYGOD = Citation(key="PyGOD (Liu et al., 2024)", title="PyGOD: A Python Library for Graph Outlier Detection",
                  arxiv="2204.12095")


class _PyGODStepLocalizer(BaseStepLocalizer):
    """Shared core: build per-trace graphs, then one chunked transductive fit."""

    def __init__(self, hid_dim: int = 32, epoch: int = 40, max_chunk_nodes: int = 3000,
                 seed: int = 0) -> None:
        self.hid_dim = hid_dim
        self.epoch = epoch
        self.max_chunk_nodes = max_chunk_nodes
        self.seed = seed

    def _graph(self, trace: Any):
        raise NotImplementedError

    def fit_step_scores(self, traces: Sequence[Any], y: Any = None) -> List[Any]:
        from auditable.methods._pygod import chunked_node_scores

        graphs = [self._graph(t) for t in traces]
        return chunked_node_scores("DOMINANT", graphs, hid_dim=self.hid_dim, epoch=self.epoch,
                                   max_chunk_nodes=self.max_chunk_nodes, seed=self.seed)


@register
class PyGODStepWW8(_PyGODStepLocalizer):
    """DOMINANT over the Who&When step matrix and full-context edges."""

    card = MethodCard(
        method_id="step.pygod_dominant_ww8",
        name="DOMINANT on the Who&When step matrix",
        family="step",
        fit_scope="transductive",
        supervised=False,
        output_kind="score",
        fidelity="wrapper",
        requires=frozenset({"agent", "kind", "exec_graph"}),
        cite=(_DOMINANT, _PYGOD),
        extras=("graph-ad",),
        board_aliases=("pygod (graph AD)",),
        benchmark_role="baseline",
        summary="Per-step reconstruction error of a DOMINANT autoencoder fit over the batch.",
    )

    def _graph(self, trace: Any):
        from auditable.featurize import full_context_edges, normalize_trace, who_when_step_matrix

        view = normalize_trace(trace)
        return who_when_step_matrix(view), full_context_edges(len(view.steps))


@register
class PyGODStepSG4(_PyGODStepLocalizer):
    """DOMINANT over the four-column step graph and the trace's own dependency edges."""

    card = MethodCard(
        method_id="step.pygod_dominant_sg4",
        name="DOMINANT on the step dependency graph",
        family="step",
        fit_scope="transductive",
        supervised=False,
        output_kind="score",
        fidelity="wrapper",
        requires=frozenset({"deps", "kind"}),
        cite=(_DOMINANT, _PYGOD),
        extras=("graph-ad",),
        board_aliases=("pygod (graph AD)",),
        benchmark_role="baseline",
        summary="Per-step reconstruction error over position, tool flag, dependency count and span.",
    )

    def _graph(self, trace: Any):
        from auditable.featurize import step_graph

        return step_graph(trace)
