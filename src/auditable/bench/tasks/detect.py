"""Run-level failure detection: did the run fail? (CatchBench ``post_detection``.)

Each run is given to a method as the typed graph :func:`auditable.graph.build_graph` makes from
its steps (``dependency="explicit"``, ``shared_resource=False``), the input the board's
detectors read; :func:`auditable.featurize.normalize_trace` accepts it like any other trace.
The protocol per method kind is in :mod:`auditable.bench.protocols`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from auditable.bench._core import Entry, Task, register_task
from auditable.bench.protocols import run_detection_auc


@dataclass
class DetectionPopulation:
    runs: List[Any]  # typed graphs, one per run
    y: Any  # 1 failed, 0 succeeded
    info: Dict[str, Any]


class DetectionTask(Task):
    family = "run"
    metrics = ("roc_auc",)

    def __init__(self, task_id: str, title: str, loader: Callable[[], Any]) -> None:
        super().__init__()
        self.task_id = task_id
        self.title = title
        self._loader = loader

    def load(self) -> DetectionPopulation:
        import numpy as np

        from auditable.graph import build_graph

        views, y, info = self._loader()
        runs = [build_graph(v.steps, dependency="explicit", shared_resource=False) for v in views]
        return DetectionPopulation(runs=runs, y=np.asarray(y), info=info)

    def describe(self, population: DetectionPopulation) -> str:
        n, failed = len(population.y), int(population.y.sum())
        return f"{self.title}, {n} runs ({failed} failed, {n - failed} succeeded)"

    def score(self, entry: Entry, population: DetectionPopulation) -> Dict[str, float]:
        return {"roc_auc": run_detection_auc(entry, population.runs, population.y)}


def _swegym():
    from auditable.bench.corpora import swegym

    return swegym.load_runs()


def _tau():
    from auditable.bench.corpora import tau

    return tau.load_runs()


register_task(DetectionTask("detect/swegym", "SWE-Gym OpenHands trajectories (one scaffold, balanced)", _swegym))
register_task(DetectionTask("detect/tau", "tau-bench retail and airline trajectories", _tau))
