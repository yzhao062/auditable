"""Step-localization evaluation protocols of the CatchBench board.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.3,
``catchbench/via_auditable/protocols.py``, ``rules.py`` (``PostRandom``, ``PostBlast``),
``run_tabular.py`` (``SeedLocalization``), and ``graph.py`` (``PyGODLocalization``), and GRADE
(https://github.com/yzhao062/grade, MIT License) at commit 3839a57
(``experiment/agent_failure_localization.py``: ``SEEDS``, ``_grouped_folds``, ``_oof_scores``,
and ``_rank_metrics``). The folds, seeds, and ranking calls stay unchanged; only the localizer
inside each protocol comes from the library entry.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

SEEDS = range(5)  # GRADE agent_failure_localization.SEEDS
METRICS = ("top1", "top3", "mrr")


def _grouped_folds(groups: Any, n_splits: int, seed: int):
    """Test masks for ``n_splits`` folds, partitioning whole runs with a seeded shuffle."""
    import numpy as np

    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    np.random.RandomState(seed).shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    fold = np.array([fold_of[g] for g in groups])
    return [fold == k for k in range(n_splits)]


def _rank_metrics(scores: Any, groups: Any, mistake_row: Mapping[int, int]):
    """Mean per-run top-1, top-3, and MRR of the true fault step given pooled step scores."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    top1, top3, rr = [], [], []
    for ri in sorted(set(groups.tolist())):
        rows = np.where(groups == ri)[0]
        order = rows[np.argsort(-scores[rows], kind="stable")]  # highest score first
        rank = int(np.where(order == mistake_row[ri])[0][0]) + 1
        top1.append(rank == 1)
        top3.append(rank <= 3)
        rr.append(1.0 / rank)
    return np.array([np.mean(top1), np.mean(top3), np.mean(rr)])


grouped_folds = _grouped_folds
rank_metrics = _rank_metrics


def step_oof_scores(make_localizer: Callable[[], Any], traces: Sequence[Any], mistakes: Sequence[int],
                    groups: Any, seed: int):
    """Out-of-fold step scores in pooled order (runs in order, steps within a run in order).

    Mirrors GRADE ``_oof_scores``: grouped folds over runs, one model per fold fit on the
    training runs, and one batched prediction over the held-out runs in pooled order.
    ``mistakes[r]`` is the ``idx`` of run ``r``'s decisive step.
    """
    import numpy as np

    groups = np.asarray(groups)
    scores = np.full(len(groups), np.nan)
    for test in _grouped_folds(groups, 5, seed):
        test_runs = sorted(set(groups[test].tolist()))
        train_runs = sorted(set(groups[~test].tolist()))
        loc = make_localizer()
        loc.fit([traces[r] for r in train_runs], [mistakes[r] for r in train_runs])
        per_run = loc.step_scores_batch([traces[r] for r in test_runs])
        scores[test] = np.concatenate([np.asarray(v, dtype=float) for v in per_run])
    return scores


def _as_dict(values: Any) -> Dict[str, float]:
    return dict(zip(METRICS, (float(v) for v in values)))


def random_step_scores(make_localizer: Callable[..., Any], runs: Sequence[Any], groups: Any,
                       mistake_row: Mapping[int, int], seeds=SEEDS) -> Dict[str, float]:
    """Mean top-1, top-3, and MRR over ``seeds``: per seed ``s``, one ``RandomState(s)`` stream
    drawn run by run over the pooled steps."""
    import numpy as np

    per_seed = []
    for s in seeds:
        rule = make_localizer(random_state=np.random.RandomState(s))
        scores = np.concatenate([rule.step_scores(run) for run in runs])
        per_seed.append(_rank_metrics(scores, groups, mistake_row))
    return _as_dict(np.array(per_seed).mean(axis=0))


def stateless_step_scores(make_localizer: Callable[[], Any], runs: Sequence[Any], groups: Any,
                          mistake_row: Mapping[int, int]) -> Dict[str, float]:
    """Top-1, top-3, and MRR of one stateless ``step_scores`` pass over each run."""
    import numpy as np

    rule = make_localizer()
    scores = np.concatenate([rule.step_scores(run) for run in runs])
    return _as_dict(_rank_metrics(scores, groups, mistake_row))


def supervised_step_scores(make_localizer: Callable[[], Any], runs: Sequence[Any], mistakes: Sequence[int],
                           groups: Any, mistake_row: Mapping[int, int], seeds=SEEDS) -> Dict[str, float]:
    """Mean top-1, top-3, and MRR over ``seeds`` of 5-fold grouped cross-validation."""
    import numpy as np

    per_seed = np.array([
        _rank_metrics(step_oof_scores(make_localizer, runs, mistakes, groups, s), groups, mistake_row)
        for s in seeds
    ])
    return _as_dict(per_seed.mean(axis=0))


def transductive_step_scores(make_localizer: Callable[[], Any], runs: Sequence[Any], groups: Any,
                             mistake_row: Mapping[int, int]) -> Dict[str, float]:
    """Top-1, top-3, and MRR of one whole-batch ``fit_step_scores`` call across all runs."""
    import numpy as np

    per_run = make_localizer().fit_step_scores(runs)
    return _as_dict(_rank_metrics(np.concatenate(per_run), groups, mistake_row))


def step_localization_metrics(entry: Any, runs: Sequence[Any], mistakes: Sequence[int], groups: Any,
                              mistake_row: Mapping[int, int]) -> Dict[str, float]:
    """The CatchBench board protocol for one step-localizer entry."""
    card = entry.card
    if card.method_id == "step.random":
        return random_step_scores(entry.make, runs, groups, mistake_row)
    if card.supervised:
        return supervised_step_scores(entry.make, runs, mistakes, groups, mistake_row)
    if card.fit_scope in ("transductive", "inductive"):
        return transductive_step_scores(entry.make, runs, groups, mistake_row)
    return stateless_step_scores(entry.make, runs, groups, mistake_row)
