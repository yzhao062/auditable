"""Online (LIVE) rules over a run's own dependency structure.

Each method scores every arriving step from the steps up to it, with no labels and no
other runs. The series are ported from CatchBench (https://github.com/yzhao062/catchbench,
MIT License) at release 0.1.2 (``live._span_series``, ``live._count_series``,
``live._online_z``, ``live._mean_span``).
"""
from __future__ import annotations

from typing import Any, Optional

from auditable.base import BaseOnlineMonitor, Citation, MethodCard
from auditable.methods import register

_CATCHBENCH = Citation(key="CatchBench (2026)", title="CatchBench: When Can an Agent Failure Be Caught?",
                       arxiv="2608.22808")


def _card(method_id: str, name: str, alias: str, role: str, summary: str,
          fidelity: str = "original", requires=frozenset({"deps"})) -> MethodCard:
    return MethodCard(method_id=method_id, name=name, family="online", fit_scope="stateless",
                      supervised=False, output_kind="score", fidelity=fidelity, requires=requires,
                      cite=(_CATCHBENCH,), extras=("tabular-ad",), board_aliases=(alias,),
                      benchmark_role=role, summary=summary)


@register
class RawSpan(BaseOnlineMonitor):
    """The maximum dependency span of each step: how far back its earliest dependency
    reaches. A stale read that skips recent state lengthens it."""

    card = _card("online.raw_span", "Raw dependency span", "raw-span", "baseline",
                 "Per-step distance to the earliest step it depends on.")

    def step_alarm_scores(self, trace: Any):
        from auditable.featurize import span_series

        return span_series(trace)


@register
class DependencySpan(BaseOnlineMonitor):
    """The running mean of the per-step dependency span, so the score after ``k`` steps
    is the mean span over the first ``k`` steps."""

    card = _card("online.dep_span", "Mean dependency span (running)", "dep-span (online)", "baseline",
                 "Causal running mean of the per-step dependency span.")

    def step_alarm_scores(self, trace: Any):
        import numpy as np

        from auditable.featurize import span_series

        s = span_series(trace)
        if not len(s):
            return s
        return np.array([float(s[:t + 1].mean()) for t in range(len(s))])


@register
class SpanZScore(BaseOnlineMonitor):
    """The dependency span of each step as a z-score against the steps before it."""

    card = _card("online.span_z", "Causal z-score of dependency span", "auditable (span z-score)",
                 "baseline", "Each step's span against the mean and spread of the spans before it.")

    def step_alarm_scores(self, trace: Any):
        from auditable.featurize import causal_z, span_series

        return causal_z(span_series(trace))


@register
class DependencyCountZScore(BaseOnlineMonitor):
    """The number of dependencies of each step as a z-score against the steps before it.
    On CatchBench it is the negative control for the span signal, because a stale-state
    redirect moves an edge without changing any step's dependency count."""

    card = _card("online.dep_count_z", "Causal z-score of dependency count", "dep-count (control)",
                 "control", "Each step's dependency count against the counts before it.")

    def step_alarm_scores(self, trace: Any):
        from auditable.featurize import causal_z, count_series

        return causal_z(count_series(trace))


@register
class RandomOnline(BaseOnlineMonitor):
    """Uniform random alarm scores, a floor. Pass a shared ``numpy.random.RandomState`` as
    ``random_state`` to draw several traces from one stream, in call order."""

    card = _card("online.random", "Random alarm scores", "random", "control",
                 "Uniform random per-step scores.", fidelity="original", requires=frozenset())

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

    def step_alarm_scores(self, trace: Any):
        from auditable.featurize import normalize_trace

        return self._rng().rand(len(normalize_trace(trace).steps))
