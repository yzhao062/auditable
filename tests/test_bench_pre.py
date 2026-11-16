"""No-download tests for the ``pre/over_privilege`` leaderboard task and its helpers.

Covers:
- task registration, metrics order, and golden file keys;
- private control and oracle classes (cards, behavior, and non-registration);
- ``pre_score`` and ``score_config_auditor`` including abstentions and coverage;
- ``corpora.pre.load_configs`` with a monkeypatched ``agentpile.load``;
- end-to-end ``leaderboard`` run on a synthetic population (including a custom ``extra`` method);
- cached judge vote integrity and replay-only entry configuration;
- no top-level third-party imports in the new bench modules.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

import auditable
from auditable.base import BaseConfigAuditor, ConfigFinding, MethodCard
from auditable.bench import Entry, get_task, leaderboard, list_tasks
from auditable.bench.corpora import pre as pre_corpus
from auditable.bench.protocols_config import pre_score, score_config_auditor
from auditable.bench.tasks.pre import (
    PreOverPrivilegeTask,
    _FlagAll,
    _FlagNone,
    _OraclePrivilegeDiff,
)
from auditable.featurize.spec import AgentConfig, as_agent_config
from auditable.methods import list_methods


def _cap(name: str, level: str = "read", ctype: str = "") -> Dict[str, str]:
    return {"name": name, "type": ctype, "permission_level": level}


# --- registration and golden metadata --------------------------------------------------------


def test_pre_task_is_registered_with_expected_metrics_and_golden():
    ids = [t.task_id for t in list_tasks()]
    assert "pre/over_privilege" in ids
    task = get_task("pre/over_privilege")
    assert task.family == "config"
    assert task.metrics == (
        "f1",
        "precision",
        "recall",
        "coverage",
        "f1@crewai",
        "f1@n8n",
        "f1@mcp",
        "f1@injecagent",
        "f1@sweagent",
        "f1@synthetic",
    )
    golden = task.golden()
    assert golden["source"]["catchbench"] == "0.1.3"
    entries = task.default_entries()
    assert len(entries) == 11
    assert set(golden["rows"]) == {e.key for e in entries}
    for spec in golden["rows"].values():
        assert set(task.metrics) <= set(spec)


# --- private controls and oracle -------------------------------------------------------------


def test_controls_are_private_and_have_valid_cards():
    registered_ids = {c.method_id for c in list_methods(family="config")}
    assert "config.flag_all" not in registered_ids
    assert "config.flag_none" not in registered_ids
    assert "config.oracle_privilege_diff" not in registered_ids

    assert _FlagAll.card.benchmark_role == "control"
    assert _FlagNone.card.benchmark_role == "control"
    assert _OraclePrivilegeDiff.card.benchmark_role == "oracle"
    for cls in (_FlagAll, _FlagNone, _OraclePrivilegeDiff):
        assert cls.card.family == "config"
        assert cls.card.fit_scope == "stateless"
        assert cls.card.output_kind == "finding"


def test_controls_behavior_and_oracle_lookup():
    cfg = AgentConfig(
        declared_capabilities=[_cap("read_file", "read"), _cap("write_file", "write")],
        spec_tokens=["read"],
        instance_id="inst-1",
    )
    assert _FlagAll().audit(cfg).excess == {"read_file", "write_file"}
    assert _FlagAll().audit(cfg).flag is True

    empty_cfg = AgentConfig(declared_capabilities=[], spec_tokens=[], instance_id="empty")
    assert _FlagAll().audit(empty_cfg).flag is False

    assert _FlagNone().audit(cfg).excess == frozenset()
    assert _FlagNone().audit(cfg).flag is False

    oracle = _OraclePrivilegeDiff(minimal_references={"inst-1": ["read_file"]})
    assert oracle.audit(cfg).excess == {"write_file"}
    assert oracle.audit(cfg).flag is True

    inline = {
        "instance_id": "inst-2",
        "declared_capabilities": [_cap("a"), _cap("b")],
        "minimal_reference": ["a", "b"],
    }
    assert _OraclePrivilegeDiff().audit(inline).flag is False

    with pytest.raises(ValueError, match="minimal_references"):
        _OraclePrivilegeDiff().audit(cfg)


# --- cached judge votes and replay-only entry ------------------------------------------------


def test_cached_judge_votes_present_and_replay_only():
    bench_root = pathlib.Path(auditable.__file__).parent / "bench"
    votes_path = bench_root / "data" / "pre" / "llm_judge_method" / "llama-3.3-70b.json"
    readme_path = bench_root / "data" / "pre" / "README.md"
    assert votes_path.is_file()
    assert readme_path.is_file()

    digest = hashlib.sha256(votes_path.read_bytes()).hexdigest()
    assert digest in readme_path.read_text(encoding="utf-8")

    data = json.loads(votes_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and len(data) > 1000

    task = get_task("pre/over_privilege")
    llm_entry = next(e for e in task.default_entries() if e.key == "config.llm_needed")
    assert llm_entry.kwargs["replay_only"] is True
    assert llm_entry.kwargs.get("backend") is None
    assert llm_entry.kwargs["model"] == "llama-3.3-70b"
    assert llm_entry.kwargs["needed"] == data


# --- protocol functions ----------------------------------------------------------------------


def test_pre_score_and_abstentions():
    targets = [
        {"instance_id": "a", "logical_source": "crewai", "excess_set": ["x", "y"]},
        {"instance_id": "b", "logical_source": "n8n", "excess_set": ["z"]},
    ]
    # Predict {x, w} on "a" (tp=1, fp=1, fn=1) and {z} on "b" (tp=1, fp=0, fn=0).
    # Total: tp=2, fp=1, fn=1 -> P=2/3, R=2/3, F1=2/3, cov=1.0.
    full = pre_score({"a": ["x", "w"], "b": ["z"]}, targets)
    assert full["precision"] == pytest.approx(2 / 3)
    assert full["recall"] == pytest.approx(2 / 3)
    assert full["f1"] == pytest.approx(2 / 3)
    assert full["coverage"] == 1.0

    # Abstain on "b": evaluable={"a"} -> tp=1, fp=1, fn=1 -> P=0.5, R=0.5, F1=0.5, cov=0.5.
    partial = pre_score({"a": ["x", "w"]}, targets, evaluable={"a"})
    assert partial["precision"] == pytest.approx(0.5)
    assert partial["recall"] == pytest.approx(0.5)
    assert partial["f1"] == pytest.approx(0.5)
    assert partial["coverage"] == pytest.approx(0.5)

    assert pre_score({}, []) == {"f1": 0.0, "precision": 0.0, "recall": 0.0, "coverage": 0.0}


def test_score_config_auditor_handles_abstention_per_source():
    from auditable.methods import get_method

    configs = [
        AgentConfig(declared_capabilities=[_cap("a"), _cap("b")], spec_tokens=[], instance_id="c1"),
        AgentConfig(declared_capabilities=[_cap("c")], spec_tokens=[], instance_id="c2"),
    ]
    targets = [
        {"instance_id": "c1", "logical_source": "crewai", "excess_set": ["b"]},
        {"instance_id": "c2", "logical_source": "mcp", "excess_set": ["c"]},
    ]
    # c2 is absent from needed -> abstains, leaving only c1 evaluable.
    judge = get_method("config.llm_needed")(needed={"c1": ["a"]}, replay_only=True)
    scores = score_config_auditor(judge, configs, targets)
    assert scores["coverage"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(1.0)
    assert scores["f1@crewai"] == pytest.approx(1.0)
    assert scores["f1@mcp"] == pytest.approx(0.0)


# --- monkeypatched agentpile loader and synthetic leaderboard --------------------------------


@dataclass
class _FakeAgentpileRun:
    id: str
    source: Dict[str, Any]
    dataset: Dict[str, Any]
    tools: Dict[str, Dict[str, Any]]
    agents: List[Dict[str, Any]]
    task: Dict[str, Any]
    record: Dict[str, Any]

    def events(self):
        return []


_FakeAgentpileRun.__module__ = "agentpile.fake"


def _make_fake_run(iid: str, source: str, caps: List[Dict[str, str]], minimal: List[str], tokens: List[str]):
    declared = [{"name": c["name"], "type": c["type"], "permission_level": c["permission_level"]} for c in caps]
    excess = [c["name"] for c in caps if c["name"] not in set(minimal)]
    tools = {
        c["name"]: {
            "name": c["name"],
            "type": c["type"],
            "permission": c["permission_level"],
            "permission_from": "test",
        }
        for c in caps
    }
    ds = {
        "instance_id": iid,
        "source": source,
        "declared_capabilities": declared,
        "minimal_reference": minimal,
        "labels": {"excess_set": excess, "label_source": "synthetic_inject"},
        "spec_tokens": tokens,
        "spec_token_overrides": {},
    }
    return _FakeAgentpileRun(
        id=iid,
        source={"reader": "catchbench_pre", "dataset": "yzhao062/catchbench"},
        dataset=ds,
        tools=tools,
        agents=[],
        task={},
        record={},
    )


def test_load_configs_maps_synthetic_and_preserves_order(monkeypatch):
    ap = pytest.importorskip("agentpile")

    fake_runs = [
        _make_fake_run("id-1", "crewai", [_cap("read_file", "read"), _cap("write_file", "write")], ["read_file"], ["read"]),
        _make_fake_run("id-2", "synth_crewai", [_cap("http_get", "network")], [], ["read"]),
    ]
    monkeypatch.setattr(ap, "load", lambda name, **kw: list(fake_runs) if name == "catchbench_pre" else [])

    configs, labels, info = pre_corpus.load_configs()
    assert [c.instance_id for c in configs] == ["id-1", "id-2"]
    assert [lab["logical_source"] for lab in labels] == ["crewai", "synthetic"]
    assert labels[0]["excess_set"] == ["write_file"]
    assert labels[1]["minimal_reference"] == []
    assert info == {"dataset": "catchbench_pre", "n_configs": 2, "by_source": {"crewai": 1, "synthetic": 1}}


def test_synthetic_pre_task_leaderboard_with_extra_method(monkeypatch):
    ap = pytest.importorskip("agentpile")

    fake_runs = [
        _make_fake_run("id-1", "crewai", [_cap("read_file", "read"), _cap("write_file", "write")], ["read_file"], ["read"]),
        _make_fake_run("id-2", "n8n", [_cap("read_file", "read"), _cap("delete_db", "write")], ["read_file"], ["read"]),
        _make_fake_run("id-3", "synth_mcp", [_cap("grant_admin", "admin")], [], ["read"]),
    ]
    monkeypatch.setattr(ap, "load", lambda name, **kw: list(fake_runs) if name == "catchbench_pre" else [])

    class _OnlyWrite(BaseConfigAuditor):
        card = MethodCard(
            method_id="config.test_only_write",
            name="Only write",
            family="config",
            fit_scope="stateless",
            supervised=False,
            output_kind="finding",
            fidelity="original",
        )

        def audit(self, config: Any) -> ConfigFinding:
            cfg = as_agent_config(config)
            excess = frozenset(c["name"] for c in cfg.declared_capabilities if c.get("permission_level") == "write")
            return ConfigFinding(flag=bool(excess), excess=excess)

    task = PreOverPrivilegeTask("pre/synth_test", "Synthetic PRE", pre_corpus.load_configs)
    entries = [
        Entry(_FlagAll),
        Entry(_FlagNone),
        Entry(_OraclePrivilegeDiff),
    ]
    board = leaderboard(task, methods=entries, extra=[_OnlyWrite])
    assert board["config.oracle_privilege_diff"].scores["f1"] == pytest.approx(1.0)
    assert board["config.flag_none"].scores["f1"] == pytest.approx(0.0)
    assert board["config.test_only_write"].scores["f1@crewai"] == pytest.approx(1.0)
    assert board["config.test_only_write"].scores["f1@n8n"] == pytest.approx(1.0)
    assert board["config.test_only_write"].scores["f1@synthetic"] == pytest.approx(0.0)
    # The oracle reads the labels: it is listed as a reference row and never takes a rank.
    assert board.rank_of("config.oracle_privilege_diff") is None
    assert [r.key for r in board.references()] == ["config.oracle_privilege_diff"]
    assert "config.oracle_privilege_diff" not in [r.key for r in board.ranked()]
    assert "Synthetic PRE, 3 configs across 3 corpora" in board.description
    # The controls, the oracle, and an unregistered method print their kind on the board.
    assert {r.key: r.type for r in board.rows} == {
        "config.flag_all": "Baseline", "config.flag_none": "Baseline",
        "config.oracle_privilege_diff": "Oracle", "config.test_only_write": "Custom"}
    assert board["config.oracle_privilege_diff"].algorithm == "Privilege-diff oracle"
    md = board.to_markdown().splitlines()
    assert md[0].startswith("| Rank | Type | Method | Algorithm | f1 |")
    assert any(line.startswith("| ref | Oracle | `config.oracle_privilege_diff` (oracle) | Privilege-diff oracle |")
               for line in md)


# --- no top-level third-party imports --------------------------------------------------------


def test_new_bench_modules_have_no_top_level_third_party_imports():
    root = pathlib.Path(auditable.__file__).parent / "bench"
    allowed = {
        "__future__",
        "dataclasses",
        "glob",
        "json",
        "os",
        "typing",
        "auditable",
    }
    for rel in ("corpora/pre.py", "protocols_config.py", "tasks/pre.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                top.add((node.module or "").split(".")[0])
        assert top <= allowed, (rel, top - allowed)
