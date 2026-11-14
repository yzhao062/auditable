"""SWE-Gym OpenHands trajectories: the balanced single-scaffold population of the board.

Ported from GRADE (https://github.com/yzhao062/grade, MIT License) at commit 3839a57,
``experiment/agent_graph_swegym.load_runs`` and ``agent_failure_detection.load_swegym``.
Shards are read in order and a run with fewer than two steps is dropped; reading stops after
the first shard at which the per-``run_id`` balanced count reaches 400. Of the ``run_id``s seen,
the one with the most runs of its rarer class is kept (one model, one iteration cap, one
temperature), and its first ``k`` resolved and first ``k`` unresolved runs form the population,
resolved first. Pooling several ``run_id``s would let step count separate scaffolds rather than
outcomes, which is why GRADE keeps one.

At the dataset revision CatchBench pins (``baf3a4e4``) this is 376 runs, 188 failed.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

DATASET = "swe_gym_openhands"
N_PER_CLASS = 400


def load_runs(n_per_class: int = N_PER_CLASS, **load_kwargs: Any) -> Tuple[List[Any], List[int], Dict[str, Any]]:
    """``(views, y, info)``: the population as :class:`~auditable.featurize.TraceView` objects
    (``convention="grade"``), labels with 1 for an unresolved run, and how it was selected."""
    import agentpile as ap

    from auditable.featurize.agentpile import from_agentpile

    by_run: Dict[Any, Dict[int, List[Any]]] = {}
    files: List[str] = []
    for run in ap.load(DATASET, **load_kwargs):
        shard = run.source.get("file")
        if not files or shard != files[-1]:
            if files and sum(min(len(v[0]), len(v[1])) for v in by_run.values()) >= n_per_class:
                break
            files.append(shard)
        with warnings.catch_warnings():
            # Runs are converted one at a time so reading can stop at a shard boundary; the
            # adapter's "no edge on any of the runs" warning then fires for every single run
            # without edges, which says nothing about the population.
            warnings.filterwarnings("ignore", message="the dependency rules produced no edge", category=UserWarning)
            (view,) = from_agentpile([run], deps="grade", convention="grade")
        if len(view.steps) < 2:
            continue
        failed = int(view.meta["label_failed"])
        by_run.setdefault(run.source.get("run_id", "all"), {0: [], 1: []})[failed].append(view)
    if not by_run:
        raise RuntimeError(f"agentpile returned no {DATASET} runs")
    run_id, best = max(by_run.items(), key=lambda kv: min(len(kv[1][0]), len(kv[1][1])))
    k = min(len(best[0]), len(best[1]))
    views = best[0][:k] + best[1][:k]
    info = {"run_id": run_id, "shards": files, "per_class": k,
            "run_ids_seen": {str(r): [len(v[0]), len(v[1])] for r, v in by_run.items()}}
    return views, [0] * k + [1] * k, info
