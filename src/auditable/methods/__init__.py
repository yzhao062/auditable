"""The auditing-method registry.

Every method class is registered under its card's ``method_id``. Method modules import
their numerical dependencies (NumPy, scikit-learn, PyOD, PyGOD, torch) inside functions,
so listing the registry works on the core install; calling a method needs its extras.

>>> from auditable.methods import list_methods, get_method
>>> [c.method_id for c in list_methods(family="online")]      # doctest: +SKIP
>>> Detector = get_method("run.pyod_ecod")                    # doctest: +SKIP
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Type

from auditable.base import BaseAuditingMethod, MethodCard

# Modules that define registered methods, one per family file. Adding a module here is the
# only step needed to expose its methods through list_methods / get_method.
_METHOD_MODULES = (
    "auditable.methods.config.rules",
    "auditable.methods.config.llm",
    "auditable.methods.run.baselines",
    "auditable.methods.run.supervised",
    "auditable.methods.run.pyod",
    "auditable.methods.run.graph",
    "auditable.methods.step.rules",
    "auditable.methods.step.supervised",
    "auditable.methods.step.graph",
    "auditable.methods.step.llm",
    "auditable.methods.online.rules",
)

_REGISTRY: Dict[str, Type[BaseAuditingMethod]] = {}
_LOADED = False


def register(cls: Type[BaseAuditingMethod]) -> Type[BaseAuditingMethod]:
    """Class decorator: add ``cls`` to the registry under ``cls.card.method_id``."""
    card = getattr(cls, "card", None)
    if not isinstance(card, MethodCard):
        raise TypeError(f"{cls.__name__} must define a MethodCard as its 'card' attribute")
    existing = _REGISTRY.get(card.method_id)
    if existing is not None and existing is not cls:
        raise ValueError(f"duplicate method_id {card.method_id!r}: {existing.__name__} and {cls.__name__}")
    _REGISTRY[card.method_id] = cls
    return cls


def _load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    for name in _METHOD_MODULES:
        # Every module imports its third-party dependencies lazily, so this succeeds on the
        # core install; an ImportError here is a bug in the module.
        importlib.import_module(name)
    _LOADED = True


def list_methods(family: Optional[str] = None, information_state: Optional[str] = None,
                 fit_scope: Optional[str] = None) -> List[MethodCard]:
    """Cards of the registered methods, filtered and sorted by ``method_id``."""
    _load_all()
    cards = [cls.card for cls in _REGISTRY.values()]
    if family is not None:
        cards = [c for c in cards if c.family == family]
    if information_state is not None:
        cards = [c for c in cards if c.information_state == information_state]
    if fit_scope is not None:
        cards = [c for c in cards if c.fit_scope == fit_scope]
    return sorted(cards, key=lambda c: c.method_id)


def get_method(method_id: str) -> Type[BaseAuditingMethod]:
    """The method class registered under ``method_id``."""
    _load_all()
    try:
        return _REGISTRY[method_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown method_id {method_id!r}; registered: {known}") from None


_FAMILY_ORDER = {f: i for i, f in enumerate(("config", "run", "step", "online"))}
# The Type column of the method table: the module that defines a method names its kind
# (methods/<family>/<group>.py), and the random floors are baselines wherever they live.
_TYPES = {"rules": "Rule", "supervised": "Supervised", "pyod": "PyOD detector",
          "graph": "Graph detector", "llm": "LLM judge", "baselines": "Baseline"}
_TYPE_ORDER = {t: i for i, t in enumerate(("Rule", "Supervised", "PyOD detector", "Graph detector",
                                           "LLM judge", "Baseline"))}


def _method_type(card: MethodCard) -> str:
    if card.method_id.endswith(".random"):
        return "Baseline"
    group = get_method(card.method_id).__module__.rsplit(".", 1)[-1]
    return _TYPES.get(group, group)


def _reference(card: MethodCard) -> str:
    if not card.cite:
        return ""
    c = card.cite[0]
    link = f"https://arxiv.org/abs/{c.arxiv}" if c.arxiv else c.url
    return f"[{c.key}]({link})" if link else c.key


def method_table(family: Optional[str] = None, detailed: bool = False) -> str:
    """The registry as a Markdown table, in family order.

    By default each row gives a method's type, ID, name, and main reference, grouped by
    type; the README's per-family tables are this output for each ``family``, and a test
    keeps them equal. ``detailed=True`` lists the card fields instead: family, fit scope,
    fidelity, extras, and the CatchBench rows the method reproduces.
    """
    cards = list_methods(family=family)
    if detailed:
        rows = ["| Method | Family | Fit | Fidelity | Extras | CatchBench row |", "|---|---|---|---|---|---|"]
        for c in sorted(cards, key=lambda c: (_FAMILY_ORDER[c.family], c.method_id)):
            fit = c.fit_scope + (", supervised" if c.supervised else "")
            extras = ", ".join(c.extras) or "none"
            rows.append(f"| `{c.method_id}` | {c.family} ({c.information_state}) | {fit} | {c.fidelity} | "
                        f"{extras} | {'; '.join(c.board_aliases) or 'none'} |")
        return "\n".join(rows)
    typed = [(c, _method_type(c)) for c in cards]
    typed.sort(key=lambda ct: (_FAMILY_ORDER[ct[0].family], _TYPE_ORDER.get(ct[1], len(_TYPE_ORDER)),
                               ct[0].method_id))
    rows = ["| Type | Method | Algorithm | Reference |", "|---|---|---|---|"]
    rows += [f"| {t} | `{c.method_id}` | {c.name} | {_reference(c)} |" for c, t in typed]
    return "\n".join(rows)


__all__ = ["list_methods", "get_method", "method_table", "register"]
