"""Step-level decisive-mistake localization: which step caused the failure? (CatchBench ``post_localization``.)

Each run is given to a method as a :class:`~auditable.featurize.TraceView` holding its raw
step list (with full-context dependency edges) and step texts. The protocol per method kind
is in :mod:`auditable.bench.protocols_step`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

from auditable.bench._core import Entry, Task, register_task
from auditable.bench.corpora.whoandwhen import pool_steps
from auditable.bench.protocols_step import step_localization_metrics

_BASE_ENTRIES: Tuple[Tuple[str, str], ...] = (
    ("step.random", "random"),
    ("step.blast", "auditable (blast)"),
    ("step.position_supervised", "position"),
    ("step.pygod_dominant_ww8", "pygod (graph AD)"),
    ("step.exec_rank", "exec-rank (sup.)"),
)

# Extension hook for additional default board entries (for example cached LLM-judge rows).
# Each callable returns a sequence of Entry objects appended to LocalizationTask.default_entries().
EXTRA_ENTRY_FACTORIES: List[Callable[[], Sequence[Entry]]] = []


@dataclass
class LocalizationPopulation:
    runs: List[Any]
    mistakes: List[int]
    groups: Any
    mistake_row: Dict[int, int]
    info: Dict[str, Any] = field(default_factory=dict)


class LocalizationTask(Task):
    family = "step"
    metrics = ("top1", "top3", "mrr")

    def __init__(self, task_id: str, title: str, loader: Callable[[], Any]) -> None:
        super().__init__()
        self.task_id = task_id
        self.title = title
        self._loader = loader

    def load(self) -> LocalizationPopulation:
        runs, mistakes, info = self._loader()
        groups, mistake_row = pool_steps(runs, mistakes)
        return LocalizationPopulation(runs=list(runs), mistakes=list(mistakes), groups=groups,
                                      mistake_row=mistake_row, info=info)

    def describe(self, population: LocalizationPopulation) -> str:
        n_runs = len(population.runs)
        n_steps = len(population.groups)
        frac = f"{n_runs / n_steps:.0%}" if n_steps else "0%"
        return f"{self.title}, {n_runs} failed runs, {n_steps} steps ({frac} faults)"

    def default_entries(self) -> List[Entry]:
        from auditable.methods import get_method

        entries = [Entry(get_method(method_id), label=label) for method_id, label in _BASE_ENTRIES]
        for factory in EXTRA_ENTRY_FACTORIES:
            entries.extend(factory())
        return entries

    def score(self, entry: Entry, population: LocalizationPopulation) -> Dict[str, float]:
        from auditable.bench.llm_rows import is_llm_entry, score_llm_entry

        if is_llm_entry(entry):  # replayed from the committed prediction caches, no API call
            return score_llm_entry(entry, population)
        return step_localization_metrics(entry, population.runs, population.mistakes,
                                         population.groups, population.mistake_row)

    def golden(self) -> Dict[str, Any]:
        """The non-LLM rows and the LLM rows, which are kept in two golden files."""
        import json
        import os

        from auditable.bench._core import _GOLDEN_DIR

        out = super().golden()
        base = self.task_id.replace("/", "__") + ".json"
        path = os.path.join(_GOLDEN_DIR, self.task_id.replace("/", "__") + "__llm.json")
        has_base = os.path.exists(os.path.join(_GOLDEN_DIR, base))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                out = {**out, "rows": {**out.get("rows", {}), **json.load(fh).get("rows", {})}}
            if not has_base:  # the LLM half is there, the board's golden file is not
                out = {**out, "missing_files": [base]}
        elif has_base:  # the board's golden file is there, the LLM half is not
            out = {**out, "missing_files": [os.path.basename(path)]}
        return out


def _whoandwhen():
    from auditable.bench.corpora import whoandwhen

    return whoandwhen.load_runs()


def _llm_entries():
    from auditable.bench.llm_rows import llm_entries

    return llm_entries()


EXTRA_ENTRY_FACTORIES.append(_llm_entries)


register_task(LocalizationTask("localize/whoandwhen", "Who&When failed runs (human mistake_step labels)", _whoandwhen))
