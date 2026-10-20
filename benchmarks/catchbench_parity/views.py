"""Raw-input views of CatchBench tasks: the whitelist of what an adapter may read.

An adapter sees what a user would hold for their own traces: configurations, step lists,
step text, the typed graphs a POST detection corpus ships as, labels, and protocol
parameters. Precomputed arrays (``task.layers``, ``task.layers_at``, ``task.X``,
``GoldLocalization.graphs``) and the loader's localization graphs (``runs[i]["G"]``,
which the library rebuilds from the steps) are left out, so every feature the library
uses is computed by ``auditable.featurize``. A view holds references to the task's own
objects; adapters must not mutate them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class PreView:
    configs: Sequence[dict]          # PreOverPrivilege.method_view(), one dict per instance
    labels: Sequence[dict]           # per-instance labels, for supervised protocols and scoring


@dataclass(frozen=True)
class LocalizationView:
    runs: Sequence[dict]             # {"steps": [...]} plus "texts" when requested, in task order
    mistakes: Sequence[int]          # idx of each run's decisive step
    groups: Any                      # pooled run index per step row
    mistake_row: Dict[int, int]


@dataclass(frozen=True)
class DetectionView:
    graphs: Sequence[Any]            # typed networkx graphs, as the corpus loader built them
    y: Any                           # 1 = failed


@dataclass(frozen=True)
class GoldView:
    steps: Sequence[list]            # injected step lists
    targets: Sequence[int]
    kinds: Sequence[str]
    groups: Any
    injected_row: Dict[int, int]


@dataclass(frozen=True)
class AttributionView:
    runs: Sequence[list]             # stale-state then dropped-dependency injected step lists
    y: Any                           # 1 = stale-state


@dataclass(frozen=True)
class StreamingView:
    runs: Sequence[list]             # raw step lists
    y: Any
    prefixes: Sequence[float]


@dataclass(frozen=True)
class StaleView:
    injected: Sequence[list]
    clean: Sequence[list]
    fprs: Sequence[float]


def as_task(view):
    """A stand-in task exposing only ``view``'s fields, for CatchBench evaluation code an
    adapter reuses (thresholds, ranking metrics, the prefix rule). Reading any other task
    field raises ``AttributeError``, so the whitelist holds for that code too."""
    from dataclasses import fields
    from types import SimpleNamespace

    return SimpleNamespace(setup=lambda: None, **{f.name: getattr(view, f.name) for f in fields(view)})


def pre_view(task) -> PreView:
    task.setup()
    return PreView(configs=task.method_view(), labels=[o.labels for o in task.instances])


def localization_view(task, *, texts: bool = False) -> LocalizationView:
    task.setup()
    step_texts: Optional[Sequence[Sequence[str]]] = task.step_texts if texts else None
    runs = []
    for i, run in enumerate(task.runs):
        r = {"steps": run["steps"]}
        if step_texts is not None:
            r["texts"] = list(step_texts[i])
        runs.append(r)
    return LocalizationView(runs=runs, mistakes=[r["mistake"] for r in task.runs],
                            groups=task.groups, mistake_row=dict(task.mistake_row))


def detection_view(task) -> DetectionView:
    task.setup()
    return DetectionView(graphs=list(task.graphs), y=task.y)


def gold_view(task) -> GoldView:
    task.setup()
    return GoldView(steps=task.steps, targets=task.targets, kinds=task.kinds, groups=task.groups,
                    injected_row=dict(task.injected_row))


def attribution_view(task) -> AttributionView:
    task.setup()
    return AttributionView(runs=task.runs, y=task.y)


def streaming_view(task) -> StreamingView:
    task.setup()
    return StreamingView(runs=task.runs, y=task.y, prefixes=tuple(task.prefixes))


def stale_view(task) -> StaleView:
    task.setup()
    return StaleView(injected=task.injected, clean=task.clean, fprs=tuple(task.fprs))
