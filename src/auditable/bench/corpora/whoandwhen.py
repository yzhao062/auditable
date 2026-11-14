"""Who&When failed multi-agent trajectories: the 126-run decisive-step localization population.

Ported from GRADE (https://github.com/yzhao062/grade, MIT License) at commit 3839a57,
``experiment/agent_graph_characterization.load_tasks`` and
``experiment/agent_failure_localization.load_failed_runs`` and ``_pool``. Runs are failed
traces (``label_failed == 1``) with at least three steps and an in-range human
``mistake_step`` label, read through ``from_agentpile(..., deps="grade", convention="grade")``.

GRADE reads files in lexicographical order of path (``1.json``, ``10.json``, ``100.json``),
whereas ``agentpile.load`` yields files in numerical stem order (``1.json``, ``2.json``).
Sorting by ``source["file"]`` preserves GRADE's run order so grouped cross-validation folds
and seeded random streams match the CatchBench board.

At the dataset revision CatchBench pins (``59b9fcba``) this is 126 runs, 1099 steps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

DATASET = "who_and_when"
MIN_STEPS = 3


def _run_steps(run: Any) -> Sequence[Dict[str, Any]]:
    if hasattr(run, "steps"):
        return run.steps
    if isinstance(run, dict) and "steps" in run:
        return run["steps"]
    return run


def pool_steps(runs: Sequence[Any], mistakes: Sequence[int]) -> Tuple[Any, Dict[int, int]]:
    """Pool step rows across ``runs`` and locate each run's decisive-step row.

    Returns ``(groups, mistake_row)`` where ``groups`` is a 1D integer array mapping each
    pooled step row to its run index and ``mistake_row`` maps each run index to the pooled
    row index whose step ``idx`` equals ``mistakes[ri]``.
    """
    import numpy as np

    if len(runs) != len(mistakes):
        raise ValueError(f"got {len(runs)} runs and {len(mistakes)} mistakes")
    groups: List[int] = []
    mistake_row: Dict[int, int] = {}
    for ri, (run, mistake) in enumerate(zip(runs, mistakes)):
        steps = _run_steps(run)
        base = len(groups)
        for j, step in enumerate(steps):
            groups.append(ri)
            if step["idx"] == mistake and ri not in mistake_row:
                mistake_row[ri] = base + j
        if ri not in mistake_row:
            raise ValueError(f"run {ri} has no step with idx={mistake!r}")
    return np.array(groups, dtype=int), mistake_row


def load_runs(**load_kwargs: Any) -> Tuple[List[Any], List[int], Dict[str, Any]]:
    """``(views, mistakes, info)``: the 126-run localization population as
    :class:`~auditable.featurize.TraceView` objects (``convention="grade"``, carrying both
    raw step lists and step texts), the decisive-step ``idx`` per run from
    :func:`~auditable.featurize.step_targets`, and corpus summary metadata."""
    import agentpile as ap

    from auditable.featurize.agentpile import from_agentpile, step_targets

    raw_runs = sorted(ap.load(DATASET, **load_kwargs), key=lambda r: (r.source or {}).get("file", ""))
    if not raw_runs:
        raise RuntimeError(f"agentpile returned no {DATASET} runs")
    views_all = from_agentpile(raw_runs, deps="grade", convention="grade")
    targets = step_targets(views_all)

    views: List[Any] = []
    mistakes: List[int] = []
    files: List[str] = []
    for raw, view, target in zip(raw_runs, views_all, targets):
        if view.meta.get("label_failed") != 1:
            continue
        if len(view.steps) < MIN_STEPS:
            continue
        ms = int(target)
        if ms < 0:
            continue
        views.append(view)
        mistakes.append(ms)
        files.append((raw.source or {}).get("file", ""))

    info = {
        "dataset": DATASET,
        "n_total": len(raw_runs),
        "n_runs": len(views),
        "n_steps": sum(len(v.steps) for v in views),
        "files": files,
    }
    return views, mistakes, info
