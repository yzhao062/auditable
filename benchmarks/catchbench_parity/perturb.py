"""Controlled outputs for the consumption check.

With perturbation on, every library method an adapter builds is wrapped so its public
scoring entry points return altered values: seeded Gaussian noise added to scores, a
shifted step for single-step predictions, flipped flags for config findings. A board entry
that changes under perturbation reads the library's output; one that does not is reading
something else. Noise (rather than negation) also moves zero-signal controls, whose rows a
monotone transform can leave unchanged.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

ENV = "CATCHBENCH_PARITY_PERTURB"
_SCORE_ENTRIES = ("decision_function", "fit_score", "step_scores", "step_alarm_scores")
_BATCH_ENTRIES = ("step_scores_batch", "fit_step_scores")


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def _noisy(rng, out):
    arr = np.asarray(out, dtype=float)
    scale = float(arr.std()) + 1.0 if arr.size else 1.0
    return arr + rng.standard_normal(arr.shape) * scale


class Perturbed:
    """A pass-through proxy that alters the scoring outputs of ``inner``."""

    def __init__(self, inner: Any, seed: int = 20261019) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_rng", np.random.RandomState(seed))

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        rng = self._rng
        if name in _SCORE_ENTRIES:
            return lambda *a, **k: _noisy(rng, attr(*a, **k))
        if name in _BATCH_ENTRIES:
            return lambda *a, **k: [_noisy(rng, v) for v in attr(*a, **k)]
        if name == "prefix_score":
            return lambda *a, **k: float(attr(*a, **k)) + float(rng.standard_normal()) + 1.0
        if name == "predict_step":
            def shifted(trace, *a, **k):
                top = attr(trace, *a, **k)
                return 0 if top is None else top + 1
            return shifted
        if name == "predict":
            def moved(trace, *a, **k):
                p = dict(attr(trace, *a, **k))
                p["top"] = 0 if p.get("top") is None else p["top"] + 1
                p["ranking"] = list(reversed(p.get("ranking") or []))
                return p
            return moved
        if name in ("audit", "audit_many"):
            def flipped(*a, **k):
                res = attr(*a, **k)
                return [_flip(f) for f in res] if isinstance(res, list) else _flip(res)
            return flipped
        if name == "fit":
            def fit(*a, **k):
                attr(*a, **k)
                return self
            return fit
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


def _flip(finding):
    from dataclasses import replace

    flag, excess = finding.flag, finding.excess
    return replace(finding, flag=None if flag is None else not flag,
                   excess=type(excess)() if excess else excess)


def wrap(method: Any) -> Any:
    """``method`` itself, or its perturbed proxy when perturbation is on."""
    return Perturbed(method) if enabled() else method
