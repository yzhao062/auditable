"""PRE configuration audit before a run: is the agent over-privileged? (CatchBench ``pre_over_privilege``.)

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.3,
``src/catchbench/pre.py`` and ``src/catchbench/pre_baselines.py``.

The task fixes the 1,187-configuration population across six logical sources (``crewai``, ``n8n``,
``mcp``, ``injecagent``, ``sweagent``, and ``synthetic``) and scores every ``config`` auditor on
pooled micro-averaged F1, precision, recall, coverage, and per-source F1.

Three benchmark-only controls (``flag_all``, ``flag_none``, and ``oracle_privilege_diff``) are
defined privately here with their own :class:`~auditable.base.MethodCard` objects and are not
added to the global method registry. The held-out LLM judge (``config.llm_needed``) replays
recorded votes from ``src/auditable/bench/data/pre/llm_judge_method/`` with ``replay_only=True``
so no network call is made.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from auditable.base import BaseConfigAuditor, ConfigFinding, MethodCard
from auditable.bench._core import Entry, Task, register_task
from auditable.bench.corpora.pre import SOURCE_ORDER
from auditable.bench.protocols_config import score_config_auditor
from auditable.featurize.spec import as_agent_config

_PRE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pre", "llm_judge_method")


class _FlagAll(BaseConfigAuditor):
    """Upper-recall control that flags every declared capability as excess."""

    card = MethodCard(
        method_id="config.flag_all",
        name="Flag all capabilities",
        family="config",
        fit_scope="stateless",
        supervised=False,
        output_kind="finding",
        fidelity="original",
        requires=frozenset({"declared_capabilities"}),
        board_aliases=("flag_all",),
        benchmark_role="control",
        summary="Upper-recall control: flag every declared capability as excess.",
    )

    def audit(self, config: Any) -> ConfigFinding:
        cfg = as_agent_config(config)
        excess = frozenset(cfg.capability_names)
        return ConfigFinding(flag=bool(excess), excess=excess, rules=("flag_all",) if excess else ())


class _FlagNone(BaseConfigAuditor):
    """Null control that flags no declared capabilities as excess."""

    card = MethodCard(
        method_id="config.flag_none",
        name="Flag no capabilities",
        family="config",
        fit_scope="stateless",
        supervised=False,
        output_kind="finding",
        fidelity="original",
        requires=frozenset({"declared_capabilities"}),
        board_aliases=("flag_none",),
        benchmark_role="control",
        summary="Null control: flag no declared capabilities as excess.",
    )

    def audit(self, config: Any) -> ConfigFinding:
        as_agent_config(config)
        return ConfigFinding(flag=False, excess=frozenset(), rules=())


class _OraclePrivilegeDiff(BaseConfigAuditor):
    """Oracle ceiling: declared capabilities minus the ground-truth minimal reference."""

    card = MethodCard(
        method_id="config.oracle_privilege_diff",
        name="Privilege-diff oracle",
        family="config",
        fit_scope="stateless",
        supervised=False,
        output_kind="finding",
        fidelity="original",
        requires=frozenset({"declared_capabilities", "minimal_reference"}),
        board_aliases=("oracle_privilege_diff",),
        benchmark_role="oracle",
        summary="Oracle ceiling: declared capabilities minus the minimal reference.",
    )

    def __init__(self, minimal_references: Optional[Mapping[str, Sequence[str]]] = None) -> None:
        self.minimal_references = minimal_references

    def audit(self, config: Any) -> ConfigFinding:
        if isinstance(config, Mapping) and "minimal_reference" in config:
            minimal = set(config["minimal_reference"])
            cfg = as_agent_config(config)
        else:
            cfg = as_agent_config(config)
            if self.minimal_references is None or cfg.instance_id not in self.minimal_references:
                raise ValueError(
                    "config.oracle_privilege_diff requires minimal_references for every configuration"
                )
            minimal = set(self.minimal_references[cfg.instance_id])
        excess = frozenset(set(cfg.capability_names) - minimal)
        return ConfigFinding(
            flag=bool(excess),
            excess=excess,
            rules=("oracle_privilege_diff",) if excess else (),
        )


def _judge_cache_files(cache_dir: str = _PRE_DATA_DIR) -> List[str]:
    return sorted(glob.glob(os.path.join(cache_dir, "*.json")))


def _load_judge_needed(cache_path: str) -> Dict[str, Any]:
    with open(cache_path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class PrePopulation:
    """Loaded configurations, per-instance targets, and summary metadata for PRE."""

    configs: List[Any]
    targets: List[Dict[str, Any]]
    info: Dict[str, Any]
    minimal_references: Dict[str, List[str]]


class PreOverPrivilegeTask(Task):
    """CatchBench ``pre_over_privilege`` task over declared agent configurations."""

    family = "config"
    metrics = (
        "f1",
        "precision",
        "recall",
        "coverage",
        *(f"f1@{src}" for src in SOURCE_ORDER),
    )

    def __init__(self, task_id: str, title: str, loader: Callable[[], Any]) -> None:
        super().__init__()
        self.task_id = task_id
        self.title = title
        self._loader = loader

    def load(self) -> PrePopulation:
        configs, labels, info = self._loader()
        minimal_refs = {rec["instance_id"]: list(rec["minimal_reference"]) for rec in labels}
        return PrePopulation(
            configs=configs,
            targets=labels,
            info=info,
            minimal_references=minimal_refs,
        )

    def describe(self, population: PrePopulation) -> str:
        by_src = population.info.get("by_source", {})
        return f"{self.title}, {len(population.configs)} configs across {len(by_src)} corpora {by_src}"

    def default_entries(self) -> List[Entry]:
        from auditable.methods import get_method, list_methods

        entries: List[Entry] = [Entry(_FlagAll), Entry(_FlagNone)]
        for card in list_methods(family=self.family):
            if card.method_id == "config.llm_needed":
                continue
            entries.append(Entry(get_method(card.method_id)))
        entries.append(Entry(_OraclePrivilegeDiff))
        llm_cls = get_method("config.llm_needed")
        for path in _judge_cache_files():
            model = os.path.splitext(os.path.basename(path))[0]
            entries.append(
                Entry(
                    llm_cls,
                    kwargs={"needed": _load_judge_needed(path), "model": model, "replay_only": True},
                    label=f"config.llm_needed({model})",
                    key="config.llm_needed",
                )
            )
        return entries

    def score(self, entry: Entry, population: PrePopulation) -> Dict[str, float]:
        if entry.card.method_id == "config.oracle_privilege_diff" and "minimal_references" not in entry.kwargs:
            auditor = entry.make(minimal_references=population.minimal_references)
        elif (
            entry.card.method_id == "config.llm_needed"
            and all(k not in entry.kwargs for k in ("backend", "needed", "cache"))
        ):
            files = _judge_cache_files()
            if not files:
                raise FileNotFoundError(f"no cached PRE judge files in {_PRE_DATA_DIR}")
            model = os.path.splitext(os.path.basename(files[0]))[0]
            auditor = entry.make(needed=_load_judge_needed(files[0]), model=model, replay_only=True)
        else:
            auditor = entry.make()
        return score_config_auditor(auditor, population.configs, population.targets, sources=SOURCE_ORDER)


def _pre_over_privilege():
    from auditable.bench.corpora import pre

    return pre.load_configs()


register_task(
    PreOverPrivilegeTask(
        "pre/over_privilege",
        "PRE over-privilege configuration audit (CatchBench multi-corpus)",
        _pre_over_privilege,
    )
)
