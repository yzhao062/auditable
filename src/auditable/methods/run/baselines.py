"""Run-level reference scorers that read nothing from the run.

``run.random`` assigns each finished run a uniform random score. It is the floor of the
detection board: its ROC-AUC averages about 0.5 whatever the labels are. Ported from
CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.2
(``detection.RandomDetection``, ``live.RandomLive``).
"""
from __future__ import annotations

from typing import Any, Optional

from auditable.base import BaseRunDetector, Citation, MethodCard
from auditable.methods import register

_CATCHBENCH = Citation(key="CatchBench (2026)", title="CatchBench: When Can an Agent Failure Be Caught?",
                       arxiv="2608.22808")


@register
class RandomRun(BaseRunDetector):
    """Uniform random run scores, a floor. Pass a shared ``numpy.random.RandomState`` as
    ``random_state`` to draw several batches from one stream, in call order. An integer
    seeds a private stream on first use, and later calls continue it; setting a new seed
    restarts the stream from that seed. The scores depend only on the number of runs in
    ``X``."""

    card = MethodCard(
        method_id="run.random",
        name="Random run scores",
        family="run",
        fit_scope="stateless",
        supervised=False,
        output_kind="score",
        fidelity="original",
        requires=frozenset(),
        cite=(_CATCHBENCH,),
        extras=("tabular-ad",),
        board_aliases=("random",),
        benchmark_role="control",
        summary="One uniform random score per run; ROC-AUC near 0.5 by construction.",
    )

    def __init__(self, random_state: Optional[Any] = 0) -> None:
        self.random_state = random_state

    def _rng(self):
        import numpy as np

        if isinstance(self.random_state, np.random.RandomState):
            return self.random_state
        if not hasattr(self, "_own_rng") or self._own_seed != self.random_state:
            self._own_rng = np.random.RandomState(self.random_state)
            self._own_seed = self.random_state
        return self._own_rng

    def fit(self, X: Any = (), y: Any = None) -> "RandomRun":
        return self

    def decision_function(self, X: Any):
        return self._rng().rand(len(X))

    def fit_score(self, X: Any, y: Any = None):
        self.decision_scores_ = self.decision_function(X)
        return self.decision_scores_
