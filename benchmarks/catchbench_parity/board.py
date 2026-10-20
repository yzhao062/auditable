"""Run the full CatchBench board with the manifest applied.

``cli.main`` imports its method factories when it runs, so replacing the factory
attributes before the call routes the main table and every breakdown and robustness table
through the adapters. Original scoring cores named in ``SENTINELS`` are replaced, in every
module that binds them, by functions that raise; a board that completes therefore never
fell back to an original scoring path.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
from typing import Any, Callable, Dict, List, Set, Tuple

from catchbench_parity import manifest

_CATCHBENCH_MODULES = (
    "catchbench.cli", "catchbench.pre", "catchbench.pre_baselines", "catchbench.post",
    "catchbench.llm_judge", "catchbench.detection", "catchbench.pyod_extra", "catchbench.pygod_extra",
    "catchbench.graph_ad", "catchbench.agent_detectors", "catchbench.gold", "catchbench.live",
    "catchbench.namedvalue",
)
_REFERENCE_PREFIXES = ("catchbench", "agent_failure_", "agent_graph_", "grade")

USED: Set[Tuple[str, str, str]] = set()
SENTINEL_HITS: List[str] = []


class SentinelError(RuntimeError):
    pass


def _resolve(spec: str) -> Tuple[Any, str]:
    mod, _, attr = spec.partition(":")
    owner = importlib.import_module(mod)
    for part in attr.split(".")[:-1]:
        owner = getattr(owner, part)
    return owner, attr.split(".")[-1]


def _wrap_factory(name: str, original: Callable) -> Callable:
    items = [it for it in manifest.ITEMS if it.factory == name]

    def factory(*args, **kwargs):
        out = []
        for obj in original(*args, **kwargs):
            hits = [it for it in items if it.matches(obj)]
            if len(hits) > 1:
                raise RuntimeError(f"{name}: {getattr(obj, 'method_id', obj)!r} matches {len(hits)} items")
            if hits:
                USED.add(hits[0].key)
                out.append(hits[0].build(obj))
            else:
                out.append(obj)
        return out

    factory.__wrapped__ = original
    return factory


def _sentinel(spec: str) -> Callable:
    def raiser(*args, **kwargs):
        SENTINEL_HITS.append(spec)
        raise SentinelError(f"original scoring core {spec} was called with the manifest applied")

    return raiser


def _rebind_everywhere(original: Any, replacement: Any) -> int:
    n = 0
    for modname, mod in list(sys.modules.items()):
        if mod is None or not modname.startswith(_REFERENCE_PREFIXES):
            continue
        for attr, value in list(vars(mod).items()):
            if value is original:
                setattr(mod, attr, replacement)
                n += 1
    return n


def apply_manifest(*, sentinels: bool = True) -> Dict[str, int]:
    """Patch factories, feature functions and sentinels in place. Returns binding counts."""
    for m in _CATCHBENCH_MODULES:
        importlib.import_module(m)
    manifest.load_adapters()
    counts: Dict[str, int] = {}
    for name, modules in manifest.FACTORIES.items():
        if not any(it.factory == name for it in manifest.ITEMS):
            continue
        original = getattr(importlib.import_module(modules[0]), name)
        counts[name] = _rebind_everywhere(original, _wrap_factory(name, original))
    for fi in manifest.FEATURE_ITEMS:
        owner, attr = _resolve(fi.name)
        lib_owner, lib_attr = _resolve(fi.library)
        counts[fi.name] = _rebind_everywhere(getattr(owner, attr), fi.adapt(getattr(lib_owner, lib_attr)))
    if sentinels:
        # Scoped to the named module (or class): task construction in GRADE may still call
        # the same function object through its own module, and that is allowed.
        for spec in manifest.SENTINELS:
            owner, attr = _resolve(spec)
            if not hasattr(owner, attr):
                raise RuntimeError(f"sentinel {spec} names nothing")
            setattr(owner, attr, _sentinel(spec))
            counts[spec] = 1
    return counts


def run_board(out_path: str, *, apply: bool = True) -> Dict[str, Any]:
    """Run ``catchbench.cli.main`` (the full board) and write its stdout to ``out_path``."""
    counts = apply_manifest() if apply else {}
    from catchbench import cli

    argv = sys.argv
    sys.argv = ["catchbench"]
    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh, contextlib.redirect_stdout(fh):
            cli.main()
    finally:
        sys.argv = argv
    expected = {it.key for it in manifest.ITEMS}
    return {"bindings": counts, "used": sorted(USED), "unused": sorted(expected - USED),
            "sentinel_hits": list(SENTINEL_HITS)}
