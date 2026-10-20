"""Supervised step localizers on the Who&When step matrix.

A pointwise ranker: a standardized, class-balanced logistic regression (liblinear) scores
each step's probability of being the decisive mistake from per-step features, and a trace's
steps are ranked by that score. The two registered methods differ only in the columns they
read, and they are nested, as in GRADE's localization study:

- ``step.position_supervised`` reads the normalized step position alone, the early-fault
  prior (faults cluster early in Who&When);
- ``step.exec_rank`` reads all eight columns of the step matrix: position, tool flag,
  agent activity share and rank, agents seen so far, and agent handoff degrees.

``fit`` takes traces and, per trace, the ``idx`` of its decisive mistake step; a step is a
positive when its ``idx`` equals that value. Both methods build the full eight-column
matrix and slice the columns they use, so the position model sees the same array layout as
GRADE's. Ported from GRADE (https://github.com/yzhao062/grade, MIT License) at commit
3839a57 (``experiment/agent_failure_localization.py``: ``_LAYER_COLS``, ``_oof_scores``), as
CatchBench 0.1.2 runs it (``post._SeedModel``).
"""
from __future__ import annotations

from typing import Any, List, Sequence

from auditable.base import BaseStepLocalizer, Citation, MethodCard
from auditable.methods import register

_GRADE = Citation(key="GRADE (Zhao, 2026)",
                  title="GRADE: Graph Representation of LLM Agent Dependency and Execution",
                  arxiv="2606.22741")
_WHO_WHEN = Citation(key="Who&When (Zhang et al., 2025)",
                     title="Which Agent Causes Task Failures and When? On Automated Failure Attribution "
                           "of LLM Multi-Agent Systems",
                     arxiv="2505.00212")


class _PointwiseStepLocalizer(BaseStepLocalizer):
    """Shared core: pool the step matrices of a batch, fit or predict once on the pool."""

    _ncols: int = 8

    def _pool(self, traces: Sequence[Any]):
        """The pooled (n_steps, 8) step matrix, runs in order and steps within a run in
        order, with the step ``idx`` values of each run."""
        import numpy as np

        from auditable.featurize import STEP_FEATURES, normalize_trace, who_when_step_matrix

        mats, idxs = [], []
        for trace in traces:
            view = normalize_trace(trace)
            mats.append(who_when_step_matrix(view).reshape(-1, len(STEP_FEATURES)))
            idxs.append([s["idx"] for s in view.steps])
        X = np.vstack(mats) if mats else np.zeros((0, len(STEP_FEATURES)))
        return X, idxs

    def fit(self, traces: Sequence[Any] = (), y: Any = None) -> "_PointwiseStepLocalizer":
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        traces = list(traces)
        if y is None:
            raise ValueError(f"{type(self).__name__} is supervised; pass y, the decisive step idx of each trace")
        y = list(y)
        if len(y) != len(traces):
            raise ValueError(f"y has {len(y)} entries for {len(traces)} traces; pass one decisive step idx per trace")
        X, idxs = self._pool(traces)
        labels = []
        for r, (idx, mistake) in enumerate(zip(idxs, y)):
            if mistake not in idx:
                raise ValueError(f"y[{r}] = {mistake!r} is not the idx of any step of trace {r}")
            labels.extend(1 if i == mistake else 0 for i in idx)
        self.pipeline_ = make_pipeline(
            StandardScaler(), LogisticRegression(solver="liblinear", class_weight="balanced")
        ).fit(X[:, :self._ncols], np.array(labels))
        return self

    def step_scores_batch(self, traces: Sequence[Any]) -> List[Any]:
        import numpy as np

        if not hasattr(self, "pipeline_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(traces, y) first")
        X, idxs = self._pool(list(traces))
        # One prediction over the pooled batch, then a split by run.
        proba = self.pipeline_.predict_proba(X[:, :self._ncols])[:, 1] if len(X) else np.zeros(0)
        out, start = [], 0
        for idx in idxs:
            out.append(proba[start:start + len(idx)])
            start += len(idx)
        return out


_REQUIRES = frozenset({"agent", "kind", "exec_graph"})


@register
class PositionSupervised(_PointwiseStepLocalizer):
    """Logistic regression on the normalized step position (the early-fault prior)."""

    _ncols = 1
    card = MethodCard(
        method_id="step.position_supervised",
        name="Early-fault position prior",
        family="step",
        fit_scope="inductive",
        supervised=True,
        output_kind="score",
        fidelity="original",
        requires=_REQUIRES,
        cite=(_GRADE, _WHO_WHEN),
        extras=("tabular-ad",),
        board_aliases=("position",),
        benchmark_role="baseline",
        summary="Logistic regression on step position alone; faults that cluster early rank first.",
    )


@register
class ExecRank(_PointwiseStepLocalizer):
    """Logistic regression on the full Who&When step matrix: position, tool flag, agent
    activity share and rank, agents seen so far, and agent handoff degrees."""

    _ncols = 8
    card = MethodCard(
        method_id="step.exec_rank",
        name="Execution-feature step ranker",
        family="step",
        fit_scope="inductive",
        supervised=True,
        output_kind="score",
        fidelity="original",
        requires=_REQUIRES,
        cite=(_GRADE, _WHO_WHEN),
        extras=("tabular-ad",),
        board_aliases=("exec-rank (sup.)",),
        benchmark_role="baseline",
        summary="Logistic regression on position, agent activity and handoff centrality of each step.",
    )
