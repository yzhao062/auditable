"""LIVE streaming early-warning tasks: predict run failure from a growing prefix of steps.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.3,
``catchbench/live.py`` (``LiveStreaming``, ``live_streaming_methods``).

Runs with at least four steps are kept as raw step lists. At each prefix fraction (25%, 50%,
75%, 100%), ``run``-family detectors are scored on typed graphs rebuilt from the prefix, while
``online``-family monitors score each run directly from its own raw step prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from auditable.bench._core import Entry, Task, _extras_note, _missing_extras, register_task
from auditable.bench.protocols_live import MIN_STEPS, PREFIXES, streaming_scores


@dataclass
class StreamingPopulation:
    runs: List[List[Dict[str, Any]]]  # raw step lists, at least MIN_STEPS steps each
    y: Any  # 1 failed, 0 succeeded
    prefixes: Tuple[float, ...]
    info: Dict[str, Any]


class StreamingTask(Task):
    """Streaming early-warning task mixing ``run`` detectors on prefix graphs and ``online``
    monitors on raw step prefixes."""

    family = "online"
    metrics = ("prefix_auc", "auc@25", "auc@50", "auc@75", "auc@100")

    _DEFAULT_METHODS = (
        "run.random",
        "run.lr_flat",
        "run.lr_spline_flat",
        "run.lr_flatdep",
        "run.lr_full",
        "run.pyod_ecod",
        "online.dep_span",
    )

    def __init__(self, task_id: str, title: str, loader: Callable[[], Any],
                 prefixes: Sequence[float] = PREFIXES) -> None:
        super().__init__()
        self.task_id = task_id
        self.title = title
        self._loader = loader
        self.prefixes = tuple(prefixes)
        if self.prefixes != PREFIXES:
            self.metrics = ("prefix_auc",) + tuple(f"auc@{int(round(p * 100))}" for p in self.prefixes)

    def load(self) -> StreamingPopulation:
        import numpy as np

        views, y, info = self._loader()
        runs: List[List[Dict[str, Any]]] = []
        labels: List[int] = []
        for view, label in zip(views, y):
            steps = view.steps if hasattr(view, "steps") else list(view)
            if len(steps) >= MIN_STEPS:
                runs.append(list(steps))
                labels.append(int(label))
        return StreamingPopulation(runs=runs, y=np.asarray(labels), prefixes=self.prefixes, info=info)

    def describe(self, population: StreamingPopulation) -> str:
        n, failed = len(population.y), int(population.y.sum())
        pcts = ", ".join(f"{int(round(p * 100))}%" for p in population.prefixes)
        return (f"{self.title}, {n} runs (>={MIN_STEPS} steps; {failed} failed, "
                f"{n - failed} succeeded), prefixes [{pcts}]")

    def default_entries(self) -> List[Entry]:
        from auditable.methods import get_method

        return [Entry(get_method(mid)) for mid in self._DEFAULT_METHODS]

    def check(self, entry: Entry) -> Optional[str]:
        """Accept ``run`` detectors (scored on prefix graphs) and ``online`` monitors."""
        if entry.card.family not in ("run", "online"):
            return f"family {entry.card.family!r}, task needs 'run' or 'online'"
        missing = _missing_extras(entry.card)
        if missing:
            return _extras_note(missing)
        return None

    def score(self, entry: Entry, population: StreamingPopulation) -> Dict[str, float]:
        return streaming_scores(entry, population.runs, population.y, prefixes=population.prefixes)


def _swegym():
    from auditable.bench.corpora import swegym

    return swegym.load_runs()


def _tau():
    from auditable.bench.corpora import tau

    return tau.load_runs()


register_task(StreamingTask(
    "live/swegym",
    "SWE-Gym OpenHands trajectories, streaming early warning",
    _swegym,
))
register_task(StreamingTask(
    "live/tau",
    "tau-bench retail and airline trajectories, streaming early warning",
    _tau,
))
