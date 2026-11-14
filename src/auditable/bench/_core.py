"""The leaderboard core: tasks, entries, boards, and the task registry.

A :class:`Task` fixes a population (which runs, steps or configurations are scored) and a
protocol (how each method is fit and scored on it). Both are frozen so that a number computed
here can be compared with the same row of the CatchBench board, and with any other method
scored on the same task. An :class:`Entry` is one row of a board: a method class, optional
constructor arguments, and the label the row is printed under. :func:`leaderboard` scores the
entries of a task and returns a :class:`Board`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

from auditable.base import BaseAuditingMethod

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


@dataclass
class Entry:
    """One board row: a method class, its constructor arguments, and its printed label."""

    method: Type[BaseAuditingMethod]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    key: Optional[str] = None  # the row's golden key; the method_id when the row has no variant

    def __post_init__(self) -> None:
        card = getattr(self.method, "card", None)
        if card is None:
            raise TypeError(f"{self.method.__name__} has no MethodCard 'card' attribute")
        if self.key is None:
            self.key = card.method_id
        if self.label is None:
            self.label = self.key

    @property
    def card(self):
        return self.method.card

    def make(self, **overrides: Any) -> BaseAuditingMethod:
        """A fresh instance with the entry's arguments, updated with ``overrides``."""
        return self.method(**{**dict(self.kwargs), **overrides})


@dataclass
class Row:
    """A scored entry. ``status`` is ``"ok"``, ``"skipped"`` (a missing extra or input) or
    ``"error"``; ``note`` says why a row was not scored."""

    key: str
    label: str
    method_id: str
    family: str
    scores: Dict[str, float]
    seconds: float
    status: str = "ok"
    note: str = ""
    golden: Dict[str, float] = field(default_factory=dict)
    role: Optional[str] = None  # the card's benchmark_role: "baseline", "control" or "oracle"
    missing_extras: Tuple[str, ...] = ()  # set when the row was skipped only for these extras

    @property
    def is_reference(self) -> bool:
        """An oracle row (it reads the labels): shown as an upper bound, never ranked."""
        return self.role == "oracle"


class Task:
    """A frozen population and protocol. Subclasses set the class attributes and implement
    :meth:`load`, :meth:`default_entries` and :meth:`score`."""

    task_id: str = ""
    family: str = ""
    metrics: Tuple[str, ...] = ()  # the first is the ranking metric; higher is better
    title: str = ""
    extras: Tuple[str, ...] = ("bench",)

    def __init__(self) -> None:
        self._population = None

    # -- population -----------------------------------------------------------------------
    def load(self) -> Any:
        """Build the population (download and convert on first use)."""
        raise NotImplementedError

    def population(self) -> Any:
        if self._population is None:
            self._population = self.load()
        return self._population

    def describe(self, population: Any) -> str:
        """One line on the population, printed above the board."""
        return self.title

    # -- entries and scoring --------------------------------------------------------------
    def default_entries(self) -> List[Entry]:
        """The rows a board shows when no ``methods`` are named: every registered method of the
        task's family, with any variants the task defines (an LLM judge per cached model)."""
        from auditable.methods import get_method, list_methods

        return [Entry(get_method(card.method_id)) for card in list_methods(family=self.family)]

    def score(self, entry: Entry, population: Any) -> Dict[str, float]:
        """Fit and score one entry under the task's protocol; returns ``{metric: value}``."""
        raise NotImplementedError

    def check(self, entry: Entry) -> Optional[str]:
        """``None`` when the entry can be scored, else the reason it is skipped."""
        if entry.card.family != self.family:
            return f"family {entry.card.family!r}, task needs {self.family!r}"
        missing = _missing_extras(entry.card)
        if missing:
            return _extras_note(missing)
        return None

    # -- golden ---------------------------------------------------------------------------
    def golden(self) -> Dict[str, Any]:
        """The CatchBench board rows this task reproduces, keyed by entry key, plus their source."""
        path = os.path.join(_GOLDEN_DIR, self.task_id.replace("/", "__") + ".json")
        if not os.path.exists(path):
            return {"source": None, "rows": {}}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)


_TASKS: Dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Add ``task`` to the registry under ``task.task_id``."""
    if not task.task_id or task.task_id in _TASKS:
        raise ValueError(f"task id {task.task_id!r} is empty or already registered")
    _TASKS[task.task_id] = task
    return task


def _load_tasks() -> None:
    import_module("auditable.bench.tasks")


def list_tasks() -> List[Task]:
    """The registered tasks, sorted by id."""
    _load_tasks()
    return [_TASKS[k] for k in sorted(_TASKS)]


def get_task(task_id: str) -> Task:
    _load_tasks()
    if task_id not in _TASKS:
        raise KeyError(f"unknown task {task_id!r}; available: {sorted(_TASKS)}")
    return _TASKS[task_id]


# Probe modules for each extra of pyproject.toml: an extra counts as installed when all of its
# modules can be found. An extra not listed here counts as missing, so a method that names one is
# skipped with the name in the reason rather than scored on a guess.
_EXTRA_MODULES = {
    "graph": ("networkx",),
    "langchain": ("langchain_core",),
    "anomaly": ("pyod",),
    "tabular-ad": ("pyod", "sklearn", "numpy", "networkx"),
    "graph-ad": ("torch", "torch_geometric", "pygod"),
    "langgraph": ("langgraph", "networkx"),
    "report": ("fpdf", "matplotlib", "networkx"),
    "mcp": ("mcp",),
    "llm": ("openai",),
    "agentpile": ("agentpile", "numpy"),
    "bench": ("pyod", "sklearn", "numpy", "networkx", "agentpile"),
}
# PyGOD's neighbour sampling needs a compiled backend that pip cannot resolve from the graph-ad
# extra (README, Install): either one counts.
_PYG_SAMPLING = ("pyg_lib", "torch_sparse")


def _missing_extras(card: Any) -> Tuple[str, ...]:
    return tuple(x for x in card.extras if not _extra_available(x))


def _extras_note(missing: Sequence[str]) -> str:
    """The skip reason :meth:`Task.check` gives when only extras are missing."""
    return "needs auditable[" + ",".join(missing) + "]"


def _extra_available(extra: str) -> bool:
    from importlib.util import find_spec

    modules = _EXTRA_MODULES.get(extra)
    if modules is None:
        return False
    if not all(find_spec(m) is not None for m in modules):
        return False
    if extra == "graph-ad":
        return any(find_spec(m) is not None for m in _PYG_SAMPLING)
    return True


MethodSpec = Union[str, Type[BaseAuditingMethod], Entry]


def _as_entry(spec: MethodSpec) -> Entry:
    if isinstance(spec, Entry):
        return spec
    if isinstance(spec, str):
        from auditable.methods import get_method

        return Entry(get_method(spec))
    if isinstance(spec, type) and issubclass(spec, BaseAuditingMethod):
        return Entry(spec)
    raise TypeError(f"cannot make a board entry from {spec!r}")


def leaderboard(task: Union[str, Task], methods: Optional[Sequence[MethodSpec]] = None, *,
                extra: Sequence[MethodSpec] = (), progress: Optional[Callable[[Row], None]] = None) -> "Board":
    """Score methods on a task and return the ranked :class:`Board`.

    ``methods`` names the rows (method ids, method classes, or :class:`Entry` objects); by
    default every registered method of the task's family. ``extra`` adds rows to the default
    set, which is how a new method is placed on the board. A method whose extra is not installed
    is listed as skipped with the reason; a method that raises is listed as an error.
    ``progress``, when given, is called with each row as soon as it is scored.
    """
    task = get_task(task) if isinstance(task, str) else task
    entries = [_as_entry(m) for m in methods] if methods is not None else task.default_entries()
    entries = entries + [_as_entry(m) for m in extra]
    population = task.population()
    golden = task.golden().get("rows", {})
    rows: List[Row] = []
    for entry in entries:
        card = entry.card
        reason = task.check(entry)
        t0 = time.time()
        if reason is not None:
            row = Row(entry.key, entry.label, card.method_id, card.family, {}, 0.0, "skipped", reason)
            missing = _missing_extras(card)
            if missing and reason == _extras_note(missing):
                row.missing_extras = missing
        else:
            try:
                scores = task.score(entry, population)
                row = Row(entry.key, entry.label, card.method_id, card.family,
                          {k: float(v) for k, v in scores.items()}, time.time() - t0)
            except Exception as exc:  # one failing method must not hide the rest of the board
                row = Row(entry.key, entry.label, card.method_id, card.family, {}, time.time() - t0,
                          "error", f"{type(exc).__name__}: {exc}")
        row.golden = {k: v for k, v in golden.get(entry.key, {}).items() if k in task.metrics}
        row.role = card.benchmark_role
        rows.append(row)
        if progress is not None:
            progress(row)
    return Board(task, task.describe(population), rows, default_rows=methods is None)


class Board:
    """A ranked board: scored rows by the task's first metric (higher first), then the oracle
    rows (upper bounds that read the labels, never ranked), then the rows that were skipped or
    failed. Control rows (random scores, flag everything) are ranked and marked."""

    def __init__(self, task: Task, description: str, rows: Iterable[Row], default_rows: bool = False) -> None:
        self.task = task
        self.description = description
        self.default_rows = default_rows  # True when the task chose the rows (no ``methods=``)
        rows = list(rows)
        primary = task.metrics[0]
        ok = sorted((r for r in rows if r.status == "ok"), key=lambda r: (-r.scores.get(primary, float("-inf")), r.label))
        self.rows: List[Row] = ([r for r in ok if not r.is_reference] + [r for r in ok if r.is_reference]
                                + [r for r in rows if r.status != "ok"])

    def __getitem__(self, key: str) -> Row:
        for row in self.rows:
            if row.key == key or row.label == key:
                return row
        raise KeyError(key)

    def scored(self) -> List[Row]:
        """Every row with scores, references included."""
        return [r for r in self.rows if r.status == "ok"]

    def ranked(self) -> List[Row]:
        """The scored rows that compete for a rank (oracle rows excluded)."""
        return [r for r in self.scored() if not r.is_reference]

    def references(self) -> List[Row]:
        return [r for r in self.scored() if r.is_reference]

    def rank_of(self, key: str) -> Optional[int]:
        """The 1-based rank of a row by its key or label, ties at three decimals sharing the
        better rank; ``None`` for a reference row or a row that was not scored."""
        ranked = self.ranked()
        primary = self.task.metrics[0]
        for row in ranked:
            if row.key == key or row.label == key:
                value = round(row.scores[primary], 3)
                return 1 + sum(1 for r in ranked if round(r.scores[primary], 3) > value)
        return None

    @staticmethod
    def _shown(row: Row) -> str:
        return row.label + (" (control)" if row.role == "control" else "")

    def compare_golden(self, tol: float = 0.0005) -> List[Dict[str, Any]]:
        """Each scored metric that has a golden value, with ``ok`` true when the score rounds to
        the golden value within ``tol`` (golden values are printed to three decimals). A golden
        row may carry its own ``tol`` (torch rows), a ``note``, and ``accept``: values known to
        come out on another platform (``{"roc_auc": [0.582]}``), reported with
        ``platform: True`` instead of as a mismatch."""
        golden = self.task.golden().get("rows", {})
        out = []
        for row in self.scored():
            spec = golden.get(row.key, {})
            row_tol = float(spec.get("tol", tol))
            accept = spec.get("accept", {})
            for metric in self.task.metrics:
                if metric not in spec or metric not in row.scores:
                    continue
                got, want = round(row.scores[metric], 3), float(spec[metric])
                exact = abs(got - want) <= row_tol + 1e-12
                platform = not exact and any(abs(got - float(a)) <= tol + 1e-12 for a in accept.get(metric, ()))
                out.append({"key": row.key, "metric": metric, "got": got, "golden": want,
                            "ok": exact or platform, "platform": platform, "note": spec.get("note", "")})
        return out

    def golden_coverage(self, allow_skipped: bool = False) -> List[Dict[str, Any]]:
        """The golden values this board leaves unchecked, one entry per row with a ``reason``.

        A golden row requested on this board counts as unchecked when it was skipped (unless
        ``allow_skipped`` and the row was skipped only for a missing extra), raised (``error``), or lacks one of its golden
        metrics. When the task chose the rows (no ``methods=``), a golden row that is absent
        from the board also counts, for example an LLM-judge row whose cache was not found.
        Any error row is reported too, golden or not, as are a golden row that names none of the
        task's metrics and a golden file the task expects but cannot find (``missing_files``). :meth:`compare_golden` checks values;
        this method checks that every value it should have checked was there to check."""
        spec_all = self.task.golden()
        golden = spec_all.get("rows", {})
        by_key = {r.key: r for r in self.rows}
        out = [{"key": f, "reason": "golden file missing"} for f in spec_all.get("missing_files", ())]
        for key, spec in golden.items():
            if not any(m in spec for m in self.task.metrics):
                out.append({"key": key, "reason": "golden row names no metric of this task"})
                continue
            row = by_key.get(key)
            if row is None:
                if self.default_rows:
                    out.append({"key": key, "reason": "absent: not among the task's default rows"})
                continue
            if row.status == "skipped":
                if not (allow_skipped and row.missing_extras):
                    out.append({"key": key, "reason": f"skipped: {row.note}"})
                continue
            if row.status != "ok":
                continue  # reported below with every other error row
            missing = [m for m in self.task.metrics if m in spec and m not in row.scores]
            if missing:
                out.append({"key": key, "reason": "metric missing: " + ", ".join(missing)})
        out.extend({"key": r.key, "reason": f"error: {r.note}"} for r in self.rows if r.status == "error")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task.task_id, "description": self.description, "metrics": list(self.task.metrics),
                "rows": [{"rank": self.rank_of(r.key), "key": r.key, "label": r.label, "method_id": r.method_id,
                          "role": r.role, "scores": r.scores, "seconds": round(r.seconds, 2),
                          "status": r.status, "note": r.note, "golden": r.golden} for r in self.rows]}

    def to_markdown(self, golden: bool = False) -> str:
        """A Markdown table of the ranked rows, then the reference rows ranked ``ref``; with
        ``golden``, a last column holds the CatchBench board value of the ranking metric."""
        metrics = list(self.task.metrics)
        primary = metrics[0]
        head = ["Rank", "Method"] + metrics + ([f"CatchBench {primary}"] if golden else [])
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in self.ranked() + self.references():
            rank = str(self.rank_of(r.key)) if not r.is_reference else "ref"
            label = f"`{r.label}`" + (" (control)" if r.role == "control" else "") + (" (oracle)" if r.is_reference else "")
            cells = [rank, label] + [f"{r.scores[m]:.3f}" for m in metrics]
            if golden:
                cells.append(f"{r.golden[primary]:.3f}" if primary in r.golden else "")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def __str__(self) -> str:
        metrics = list(self.task.metrics)
        primary = metrics[0]
        width = max([len(self._shown(r)) for r in self.rows] + [6])
        show_golden = any(primary in r.golden for r in self.rows)
        out = [f"{self.task.task_id}: {self.description}",
               f"ranked by {primary}, higher is better"
               + (f"; the CatchBench reference {primary} in brackets" if show_golden else ""),
               "", f"{'rank':>4}  {'method':<{width}}  " + "  ".join(f"{m:>8}" for m in metrics)]

        def line(rank: str, r: Row) -> str:
            cells = [f"{r.scores[m]:8.3f}" for m in metrics]
            if primary in r.golden:
                cells[0] += f" [{r.golden[primary]:.3f}]"
            elif show_golden:
                cells[0] += " " * 8
            return f"{rank:>4}  {self._shown(r):<{width}}  " + "  ".join(cells)

        out.extend(line(str(self.rank_of(r.key)), r) for r in self.ranked())
        refs = self.references()
        if refs:
            out.append("")
            out.append("reference rows (read the labels; an upper bound, not ranked):")
            out.extend(line("ref", r) for r in refs)
        rest = [r for r in self.rows if r.status != "ok"]
        if rest:
            out.append("")
            out.extend(f"   -  {r.label:<{width}}  {r.status}: {r.note}" for r in rest)
        return "\n".join(out)

    def __repr__(self) -> str:
        return (f"<Board {self.task.task_id}: {len(self.ranked())} ranked, {len(self.references())} reference, "
                f"{len(self.rows) - len(self.scored())} not scored>")
