"""Structural step rules: which step is the decisive mistake, from the run's own structure.

Each method scores every step of one run, with no labels and no other runs; higher means
more likely the decisive mistake. They are the leak-ablation baselines and mechanism
controls of CatchBench's localization boards:

- ``step.random``: uniform random scores, a floor;
- ``step.position_early_rule``: the early-fault prior, one minus the normalized position;
- ``step.blast``: the number of steps that transitively depend on each step, on a graph in
  which every step depends on all earlier ones (the Who&When assumption);
- ``step.degree``: the number of dependency edges touching each step, in or out;
- ``step.has_dep`` and ``step.max_span``: the dependency count and the dependency span,
  two controls keyed to the injected-fault construction;
- ``step.dep_anomaly``: the larger within-run z-score of the dependency count and span.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.2 (``post._auditable_blast``, ``gold._degrees``, ``gold._zmax``,
``gold.gold_localization_methods``, ``gold.GoldRandom``). The step features come from
:func:`auditable.featurize.step_graph`.
"""
from __future__ import annotations

from typing import Any, Optional

from auditable.base import BaseStepLocalizer, Citation, MethodCard
from auditable.methods import register

_CATCHBENCH = Citation(key="CatchBench (2026)", title="CatchBench: When Can an Agent Failure Be Caught?",
                       arxiv="2608.22808")


def _card(method_id: str, name: str, alias: str, role: str, summary: str,
          requires=frozenset({"deps"})) -> MethodCard:
    return MethodCard(method_id=method_id, name=name, family="step", fit_scope="stateless",
                      supervised=False, output_kind="score", fidelity="original", requires=requires,
                      cite=(_CATCHBENCH,), extras=("tabular-ad",), board_aliases=(alias,),
                      benchmark_role=role, summary=summary)


def _columns(trace: Any):
    """``(X, edges)`` of the step graph, with ``X`` always two-dimensional (an empty trace
    gives shape (0, 4))."""
    from auditable.featurize import step_graph

    X, edges = step_graph(trace)
    return X.reshape(-1, 4), edges


@register
class RandomStep(BaseStepLocalizer):
    """Uniform random step scores, a floor. Pass a shared ``numpy.random.RandomState`` as
    ``random_state`` to draw several traces from one stream, in call order: consecutive
    draws of ``n`` and ``m`` scores equal one draw of ``n + m``."""

    card = _card("step.random", "Random step scores", "random", "control",
                 "Uniform random per-step scores.", requires=frozenset())

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

    def step_scores(self, trace: Any):
        from auditable.featurize import normalize_trace

        return self._rng().rand(len(normalize_trace(trace).steps))


@register
class EarlyPositionRule(BaseStepLocalizer):
    """The early-fault prior: one minus the normalized position, so the first step scores 1
    and the last scores 0."""

    card = _card("step.position_early_rule", "Early-position prior", "position", "baseline",
                 "One minus the step's normalized position in the run.", requires=frozenset())

    def step_scores(self, trace: Any):
        X, _ = _columns(trace)
        return 1.0 - X[:, 0]


@register
class BlastShare(BaseStepLocalizer):
    """The blast radius of each step: how many steps transitively depend on it, as a raw
    count. With ``dependency="full_context"`` (CatchBench's reading of Who&When) every step
    depends on all earlier ones, so the score is monotone in position and coincides with
    the position prior. With ``dependency="explicit"`` the graph uses the dependencies the
    trace records."""

    card = _card("step.blast", "Blast radius (downstream reach)", "auditable (blast)", "baseline",
                 "Number of steps that transitively depend on each step.",
                 requires=frozenset({"agent", "kind"}))

    def __init__(self, dependency: str = "full_context") -> None:
        self.dependency = dependency

    def step_scores(self, trace: Any):
        import numpy as np

        from auditable.featurize import normalize_trace
        from auditable.graph import downstream_reach
        from auditable.graph.session import DependencyEdge, SessionGraph, Step

        if self.dependency not in ("full_context", "explicit"):
            raise ValueError("dependency must be 'full_context' or 'explicit'")
        steps = normalize_trace(trace).steps
        ids = [s["idx"] for s in steps]
        graph_steps = [
            Step(
                idx=s["idx"],
                kind=s["kind"],
                agent=s["agent"],
                deps=[DependencyEdge(src_idx=j) for j in
                      (ids[:pos] if self.dependency == "full_context" else s.get("deps", ()))],
            )
            for pos, s in enumerate(steps)
        ]
        graph = SessionGraph.from_steps(graph_steps).to_networkx()
        return np.array([downstream_reach(graph, s["idx"]) for s in steps], dtype=float)


@register
class StepDegree(BaseStepLocalizer):
    """The number of dependency edges touching each step, counting a step's own
    dependencies and the later steps that depend on it."""

    card = _card("step.degree", "Dependency degree", "degree", "baseline",
                 "Dependency edges into and out of each step.")

    def step_scores(self, trace: Any):
        import numpy as np

        X, edges = _columns(trace)
        deg = np.zeros(len(X))
        for s, d in edges.T:
            deg[int(s)] += 1
            deg[int(d)] += 1
        return deg


@register
class HasDependency(BaseStepLocalizer):
    """One for a step with at least one earlier dependency, zero otherwise. On CatchBench
    Gold it is the eligibility control: the injector only targets steps that have
    dependencies, so this scores the selection and not the fault."""

    card = _card("step.has_dep", "Has a dependency", "has-dep (control)", "control",
                 "One when the step depends on an earlier step.")

    def step_scores(self, trace: Any):
        X, _ = _columns(trace)
        return (X[:, 2] > 0).astype(float)


@register
class MaxDependencySpan(BaseStepLocalizer):
    """The dependency span of each step: how far back its earliest dependency reaches. On
    CatchBench Gold it is keyed to the stale-state fault, which lengthens a span."""

    card = _card("step.max_span", "Maximum dependency span", "max-span (control)", "control",
                 "Distance from each step to the earliest step it depends on.")

    def step_scores(self, trace: Any):
        X, _ = _columns(trace)
        return X[:, 3].copy()


def _zmax(*cols):
    """The largest absolute within-run z-score across ``cols`` at each position; a constant
    column contributes 0."""
    import numpy as np

    out = np.zeros(len(cols[0]))
    for col in cols:
        sd = col.std()
        z = np.abs((col - col.mean()) / sd) if sd > 1e-9 else np.zeros_like(col)
        out = np.maximum(out, z)
    return out


@register
class DependencyAnomaly(BaseStepLocalizer):
    """How unusual each step's dependency count and span are within its own run: the larger
    of the two absolute z-scores. On CatchBench Gold it is close to the raw span, because the
    span carries most of the signal the injector leaves."""

    card = _card("step.dep_anomaly", "Dependency anomaly (within-run z-score)", "auditable (dep-anomaly)",
                 "baseline", "Larger absolute within-run z-score of dependency count and span.")

    def step_scores(self, trace: Any):
        import numpy as np

        X, _ = _columns(trace)
        if not len(X):
            return np.zeros(0)
        return _zmax(X[:, 2], X[:, 3])
