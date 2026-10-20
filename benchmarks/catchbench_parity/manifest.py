"""The migration manifest: which CatchBench board entries run through the library.

Each :class:`Item` is one board entry, identified by ``(task_id, board_alias, variant)``.
It names the CatchBench factory that produces the entry, the library method it now runs
through, and how to build its adapter from the original method object. Adapter modules
register their items with :func:`item`; ``board.py`` swaps the matching objects in every
factory's output, and ``checks.py`` runs each item's per-sample comparison.

``SENTINELS`` lists original scoring cores that no board entry may call once the manifest
is applied, each as ``"module:attr"`` or ``"module:Class.attr"``. A sentinel replaces that
one binding only, so GRADE's own modules keep the function objects their task
construction calls; task construction may use original feature code (that belongs to
the benchmark), scoring may not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# factory name -> modules whose attribute must be replaced (cli binds pre_methods at import)
FACTORIES: Dict[str, Tuple[str, ...]] = {
    "pre_methods": ("catchbench.pre", "catchbench.cli"),
    "post_localization_methods": ("catchbench.post",),
    "discovered_llm_judge_methods": ("catchbench.llm_judge",),
    "post_detection_methods": ("catchbench.detection",),
    "pyod_extra_methods": ("catchbench.pyod_extra",),
    "pygod_extra_methods": ("catchbench.pygod_extra",),
    "gold_localization_methods": ("catchbench.gold",),
    "gold_attribution_methods": ("catchbench.gold",),
    "live_streaming_methods": ("catchbench.live",),
    "live_stale_methods": ("catchbench.live",),
}

# task_id -> board section headers (prefixes) where that task's entries appear
SECTIONS: Dict[str, Tuple[str, ...]] = {
    "pre_over_privilege": ("[PRE] pre_over_privilege :: multi", "[PRE] pre_over_privilege :: F1 by source"),
    "post_localization": ("[POST] post_localization ::",),
    "post_detection": ("[POST] post_detection ::",),
    "gold_localization": ("[POST] gold_localization ::", "Gold per-fault breakdown",
                          "Gold eligibility-matched control", "Gold seed robustness"),
    "gold_attribution": ("[POST] gold_attribution ::", "Gold attribution seed robustness"),
    "live_streaming": ("[LIVE] live_streaming ::", "LIVE streaming early-warning"),
    "live_stale_state": ("[LIVE] live_stale_state ::", "LIVE stale-state online detection",
                         "LIVE stale-state seed robustness"),
}


@dataclass(frozen=True)
class Item:
    task_id: str
    board_alias: str                     # the board row label
    variant: str                         # "" or, for LLM judges, the model name
    factory: str                         # a key of FACTORIES
    method_id: str                       # the library method the entry runs through
    build: Callable[[Any], Any]          # original method object -> adapter object
    match: Optional[Callable[[Any], bool]] = None  # default: obj.method_id == row label

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.task_id, self.board_alias, self.variant)

    @property
    def row_label(self) -> str:
        return f"{self.board_alias} ({self.variant})" if self.variant else self.board_alias

    def matches(self, obj: Any) -> bool:
        if self.match is not None:
            return self.match(obj)
        return getattr(obj, "method_id", None) == self.row_label


@dataclass(frozen=True)
class FeatureItem:
    """A board-side feature function replaced by a library primitive (no method row of its
    own; it feeds a report such as ``gold_report``)."""

    name: str                            # "catchbench.gold:_run_maxspan"
    library: str                         # "auditable.featurize:run_max_span"
    sections: Tuple[str, ...]
    adapt: Callable[[Callable], Callable]  # library function -> drop-in for the original


# Report lines without a method label of their own, computed from a migrated entry's
# scores: (section header prefix, line prefix, the manifest key they derive from). The
# consumption check requires them to change with that entry.
DERIVED_ROWS: List[Tuple[str, str, Tuple[str, str, str]]] = [
    ("Gold seed robustness", "matched stale max-span", ("gold_localization", "max-span (control)", "")),
]

# Original cores whose board callers span several adapter modules; applied only when
# every module is loaded (the board run), because a subset would leave callers unported.
SHARED_SENTINELS: Tuple[str, ...] = (
    "catchbench.post:_seed_metrics",            # random (rules), position and exec-rank (run_tabular)
    "catchbench.post:_SeedModel.evaluate",
)

ITEMS: List[Item] = []
FEATURE_ITEMS: List[FeatureItem] = []
SENTINELS: List[str] = []                # "module:attr" of original scoring cores
CHECKS: Dict[Tuple[str, str, str], Callable[..., dict]] = {}


def item(**kw) -> Item:
    it = Item(**kw)
    if any(o.key == it.key for o in ITEMS):
        raise ValueError(f"duplicate manifest item {it.key}")
    ITEMS.append(it)
    return it


def feature_item(**kw) -> FeatureItem:
    fi = FeatureItem(**kw)
    FEATURE_ITEMS.append(fi)
    return fi


def sentinel(*names: str) -> None:
    for n in names:
        if n not in SENTINELS:
            SENTINELS.append(n)


def check(*keys: Tuple[str, str, str]):
    """Register a per-sample comparison for one or more manifest keys."""
    def deco(fn):
        for k in keys:
            CHECKS[k] = fn
        return fn
    return deco


ADAPTER_MODULES = ("online", "rules", "run_tabular", "graph", "llm", "config")


def load_adapters(modules=None) -> None:
    """Import the adapter modules (all of them by default) so their items register; with
    all of them loaded, also register the shared sentinels."""
    import importlib

    for name in modules or ADAPTER_MODULES:
        importlib.import_module(f"catchbench_parity.adapters.{name}")
    if not modules:
        sentinel(*SHARED_SENTINELS)
