"""Per-sample comparisons between the library and the CatchBench originals.

Each adapter module registers one check per manifest item (``manifest.check``). A check
receives a :class:`Context`, which builds and caches CatchBench tasks, and returns a dict
with ``compared`` and ``mismatches`` counts. Equality is exact: same shape, same values,
bit for bit (``numpy.array_equal`` on float arrays; ``==`` otherwise).
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np

from catchbench_parity import manifest


def _task_classes():
    from catchbench.detection import PostDetection
    from catchbench.gold import GoldAttribution, GoldLocalization
    from catchbench.live import LiveStaleState, LiveStreaming
    from catchbench.post import PostLocalization
    from catchbench.pre import PreOverPrivilege

    return {"pre_over_privilege": PreOverPrivilege, "post_localization": PostLocalization,
            "post_detection": PostDetection, "gold_localization": GoldLocalization,
            "gold_attribution": GoldAttribution, "live_streaming": LiveStreaming,
            "live_stale_state": LiveStaleState}


class Context:
    def __init__(self) -> None:
        self._tasks: Dict[Tuple, Any] = {}
        self._classes = _task_classes()

    def task(self, task_id: str, **kw):
        key = (task_id, tuple(sorted(kw.items())))
        if key not in self._tasks:
            t = self._classes[task_id](**kw)
            t.setup()
            self._tasks[key] = t
        return self._tasks[key]

    @staticmethod
    def same(a: Any, b: Any) -> bool:
        if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
            a, b = np.asarray(a), np.asarray(b)
            return a.shape == b.shape and bool(np.array_equal(a, b))
        return a == b

    @staticmethod
    def merge(parts: Dict[str, dict]) -> dict:
        """Sum ``compared`` and ``mismatches`` over parts and keep each part's detail."""
        return {"compared": sum(p["compared"] for p in parts.values()),
                "mismatches": sum(p["mismatches"] for p in parts.values()),
                "parts": parts}


def run_checks(only: str = "", modules=None) -> Dict[str, dict]:
    manifest.load_adapters(modules)
    ctx = Context()
    items = {it.key: it for it in manifest.ITEMS}
    results: Dict[str, dict] = {}
    for key, fn in manifest.CHECKS.items():
        label = " | ".join(k for k in key if k)
        if only and only not in label:
            continue
        it = items.get(key) or SimpleNamespace(key=key, task_id=key[0], board_alias=key[1], variant=key[2],
                                               method_id="", factory="")
        t0 = time.time()
        try:
            res = fn(ctx, it)
        except Exception as exc:  # a crashing check is a failed check, reported, not skipped
            res = {"compared": 0, "mismatches": -1, "error": f"{type(exc).__name__}: {exc}"}
        res["seconds"] = round(time.time() - t0, 1)
        res["passed"] = res.get("mismatches") == 0 and res.get("compared", 0) > 0
        results[label] = res
        print(f"{'PASS' if res['passed'] else 'FAIL'}  {label}  compared={res.get('compared')} "
              f"mismatches={res.get('mismatches')}  ({res['seconds']}s)"
              + (f"  {res['error']}" if "error" in res else ""), flush=True)
    return results


def missing_checks() -> list:
    manifest.load_adapters()
    return [it.key for it in manifest.ITEMS if it.key not in manifest.CHECKS]
