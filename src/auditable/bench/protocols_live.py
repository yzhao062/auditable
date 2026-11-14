"""Evaluation protocols for LIVE streaming early warning on growing trace prefixes.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.3,
``catchbench/live.py`` (``LiveStreaming``, ``_prefix_steps``, ``live_breakdown``),
``catchbench/via_auditable/run_tabular.py`` (``StreamingLayer``, ``StreamingRandom``,
``StreamingPyOD``, ``_prefix_graphs``), and ``catchbench/via_auditable/online.py``
(``StreamingOnline``).

At each prefix fraction (by default 25%, 50%, 75%, and 100%), runs with at least four steps are
truncated via :func:`prefix_steps` (at least two steps so a dependency graph can form). Methods of
the ``run`` family are scored with :func:`auditable.bench.protocols.run_detection_auc` on typed
graphs rebuilt from each prefix, while ``online`` monitors score each run directly from its own
raw step prefix via ``prefix_score(prefix, len(prefix))``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

PREFIXES = (0.25, 0.5, 0.75, 1.0)
MIN_STEPS = 4  # a run needs enough steps for an early-versus-late prefix distinction
DETECT_BAR = 0.70  # ROC-AUC a prefix must clear to count as detected for time-to-detection


def prefix_steps(steps: Sequence[Any], frac: float) -> List[Any]:
    """The first ``frac`` of ``steps``, at least two steps so ``build_graph`` can form an edge."""
    import numpy as np

    return list(steps[: max(2, int(np.ceil(frac * len(steps))))])


def prefix_graphs(runs: Sequence[Sequence[Any]], frac: float) -> List[Any]:
    """Build typed graphs (``dependency="explicit"``, ``shared_resource=False``) on the
    ``frac`` prefix of each run."""
    from auditable.graph import build_graph

    return [build_graph(prefix_steps(s, frac), dependency="explicit", shared_resource=False) for s in runs]


def streaming_run_auc_curve(entry: Any, runs: Sequence[Sequence[Any]], y: Any,
                            prefixes: Sequence[float] = PREFIXES) -> Dict[float, float]:
    """Per-prefix ROC-AUC for a ``run``-family entry on prefix graphs rebuilt at each fraction.

    For ``run.random``, each prefix uses a fresh ``RandomState(0)`` (five draws), matching
    ``StreamingRandom`` on CatchBench; all other ``run`` entries follow
    :func:`auditable.bench.protocols.run_detection_auc`.
    """
    from auditable.bench.protocols import random_auc, run_detection_auc

    out: Dict[float, float] = {}
    is_random = entry.card.method_id == "run.random"
    for p in prefixes:
        fp = float(p)
        if is_random:
            out[fp] = random_auc(entry.make, runs, y)
        else:
            out[fp] = run_detection_auc(entry, prefix_graphs(runs, fp), y)
    return out


def streaming_online_auc_curve(entry: Any, runs: Sequence[Sequence[Any]], y: Any,
                               prefixes: Sequence[float] = PREFIXES) -> Dict[float, float]:
    """Per-prefix ROC-AUC for an ``online`` monitor: score each run from its own prefix via
    ``prefix_score(prefix, len(prefix))``, with no labels and no other runs."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_arr = np.asarray(y)
    mon = entry.make()
    out: Dict[float, float] = {}
    for p in prefixes:
        fp = float(p)
        scores = []
        for s in runs:
            sub = prefix_steps(s, fp)
            scores.append(mon.prefix_score(sub, len(sub)))
        out[fp] = float(roc_auc_score(y_arr, np.asarray(scores, dtype=float)))
    return out


def streaming_auc_curve(entry: Any, runs: Sequence[Sequence[Any]], y: Any,
                        prefixes: Sequence[float] = PREFIXES) -> Dict[float, float]:
    """Per-prefix ROC-AUC curve for either a ``run`` detector or an ``online`` monitor."""
    family = entry.card.family
    if family == "online":
        return streaming_online_auc_curve(entry, runs, y, prefixes=prefixes)
    if family == "run":
        return streaming_run_auc_curve(entry, runs, y, prefixes=prefixes)
    raise ValueError(f"unsupported family {family!r} for streaming early warning")


def streaming_scores(entry: Any, runs: Sequence[Sequence[Any]], y: Any,
                     prefixes: Sequence[float] = PREFIXES) -> Dict[str, float]:
    """Return ``prefix_auc`` (mean over prefixes) and ``auc@<pct>`` for each prefix."""
    import numpy as np

    curve = streaming_auc_curve(entry, runs, y, prefixes=prefixes)
    out: Dict[str, float] = {"prefix_auc": float(np.mean(list(curve.values())))}
    for p, auc in curve.items():
        out[f"auc@{int(round(p * 100))}"] = float(auc)
    return out


def time_to_detection(curve: Mapping[float, float], bar: float = DETECT_BAR) -> str:
    """Earliest prefix fraction whose ROC-AUC is at least ``bar``, formatted like ``'25%'``,
    or ``'>100%'`` when no prefix clears ``bar``."""
    for p in sorted(curve):
        if curve[p] >= bar:
            return f"{int(round(p * 100))}%"
    return ">100%"
