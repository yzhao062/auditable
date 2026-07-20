"""Fault injection for dependency-snapshot detector evaluation.

Labels are injection-site ground truth fixed when a corpus is constructed. They are
independent of any detector that later scores the rows, so evaluation labels stay
non-circular.

Signed ``DecisionRecord`` objects are out of scope by design: mutating one would break
its content digest. Injection applies only to snapshot copies used in evaluation
corpora, never to audit evidence. This module is evaluation tooling, not a shipped
detection method.
"""

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from ..record import DependencySnapshot


@dataclass(frozen=True)
class Injection:
    """One constructed fault and its injection-site ground truth."""

    kind: str
    key: str
    original: Any
    injected: Any
    detail: dict = field(default_factory=dict)


@dataclass
class InjectionResult:
    """A faulted snapshot copy and the faults applied to it."""

    snapshot: DependencySnapshot
    injections: List[Injection]


@dataclass
class CorpusRow:
    """One clean or faulted row in a detector-evaluation corpus."""

    snapshot: DependencySnapshot
    label: str
    injections: List[Injection]


def _random(rng: Optional[random.Random]) -> random.Random:
    return random.Random() if rng is None else rng


def _numeric_keys(state: dict, keys: Optional[Sequence[str]]) -> List[str]:
    candidates = state.keys() if keys is None else keys
    return [
        key
        for key in candidates
        if key in state
        and not isinstance(state[key], bool)
        and isinstance(state[key], (int, float))
    ]


def _ood_keys(state: dict, keys: Optional[Sequence[str]]) -> List[str]:
    candidates = state.keys() if keys is None else keys
    return [
        key
        for key in candidates
        if key in state
        and not isinstance(state[key], bool)
        and isinstance(state[key], (str, int, float))
    ]


def inject_stale(
    snapshot: DependencySnapshot,
    *,
    min_age_seconds: float = 7 * 86400.0,
    max_age_seconds: float = 30 * 86400.0,
    now: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> InjectionResult:
    """Backdate a copy of ``snapshot`` into the requested stale-age range."""
    faulted = copy.deepcopy(snapshot)
    rng = _random(rng)
    now = time.time() if now is None else now
    age = rng.uniform(min_age_seconds, max_age_seconds)
    original = faulted.captured_at
    injected = now - age
    faulted.captured_at = injected
    return InjectionResult(
        snapshot=faulted,
        injections=[
            Injection(
                kind="stale",
                key="captured_at",
                original=original,
                injected=injected,
                detail={"age_seconds": age},
            )
        ],
    )


def inject_drift(
    snapshot: DependencySnapshot,
    *,
    keys: Optional[Sequence[str]] = None,
    fraction=(0.3, 0.9),
    direction: str = "down",
    rng: Optional[random.Random] = None,
) -> InjectionResult:
    """Scale one numeric value in a copy of ``snapshot`` up or down."""
    faulted = copy.deepcopy(snapshot)
    rng = _random(rng)
    eligible = _numeric_keys(faulted.state, keys)
    if not eligible:
        raise ValueError("inject_drift needs at least one numeric state key.")
    if direction not in ("down", "up"):
        raise ValueError("direction must be 'down' or 'up'.")

    key = rng.choice(eligible)
    original = faulted.state[key]
    amount = rng.uniform(*fraction)
    multiplier = 1 - amount if direction == "down" else 1 + amount
    injected = original * multiplier
    faulted.state[key] = injected
    return InjectionResult(
        snapshot=faulted,
        injections=[
            Injection(
                kind="drift",
                key=key,
                original=original,
                injected=injected,
                detail={"fraction": amount, "direction": direction},
            )
        ],
    )


def inject_ood(
    snapshot: DependencySnapshot,
    *,
    keys: Optional[Sequence[str]] = None,
    rng: Optional[random.Random] = None,
) -> InjectionResult:
    """Replace one scalar value with an out-of-distribution value in a copy."""
    faulted = copy.deepcopy(snapshot)
    rng = _random(rng)
    eligible = _ood_keys(faulted.state, keys)
    if not eligible:
        raise ValueError("inject_ood needs at least one string or numeric state key.")

    key = rng.choice(eligible)
    original = faulted.state[key]
    if isinstance(original, str):
        injected = original + "__ood"
        rule = "append_ood_suffix"
    elif original == 0:
        injected = 1e6
        rule = "replace_zero_with_1e6"
    else:
        injected = original * 100.0
        rule = "multiply_by_100"
    faulted.state[key] = injected
    return InjectionResult(
        snapshot=faulted,
        injections=[
            Injection(
                kind="ood",
                key=key,
                original=original,
                injected=injected,
                detail={"rule": rule},
            )
        ],
    )


def build_injected_corpus(
    snapshots: Sequence[DependencySnapshot],
    *,
    kinds: Sequence[str] = ("stale", "drift", "ood"),
    rate: float = 0.5,
    seed: int = 0,
    now: Optional[float] = None,
) -> List[CorpusRow]:
    """Build deterministic clean and faulted snapshot copies for evaluation."""
    rng = random.Random(seed)
    shared_now = time.time() if now is None else now
    kind_order = list(kinds)
    rows = []

    for snapshot in snapshots:
        if rng.random() >= rate or not kind_order:
            rows.append(CorpusRow(copy.deepcopy(snapshot), "clean", []))
            continue

        drawn = rng.choice(kind_order)
        candidates = [drawn] + [kind for kind in kind_order if kind != drawn]
        result = None
        label = "clean"
        for kind in candidates:
            try:
                if kind == "stale":
                    result = inject_stale(snapshot, now=shared_now, rng=rng)
                elif kind == "drift":
                    result = inject_drift(snapshot, rng=rng)
                elif kind == "ood":
                    result = inject_ood(snapshot, rng=rng)
                else:
                    continue
            except ValueError:
                continue
            label = kind
            break

        if result is None:
            rows.append(CorpusRow(copy.deepcopy(snapshot), "clean", []))
        else:
            rows.append(CorpusRow(result.snapshot, label, result.injections))

    return rows
