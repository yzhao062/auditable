"""Tests for the SWE-Gym population selection rule in auditable.bench.corpora.swegym.

Uses synthetic agentpile-like runs with monkeypatched agentpile.load and
auditable.featurize.agentpile.from_agentpile, requiring no download.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

ap = pytest.importorskip("agentpile")

import auditable.featurize.agentpile as ag_adapter  # noqa: E402
from auditable.bench.corpora import swegym  # noqa: E402


def _fake_run(tag: str, *, shard: str, run_id: str, n_steps: int, failed: int):
    return SimpleNamespace(
        tag=tag,
        source={"file": shard, "run_id": run_id},
        n_steps=n_steps,
        failed=failed,
    )


def _fake_from_agentpile(runs, *, deps, convention):
    assert deps == "grade" and convention == "grade"
    return [
        SimpleNamespace(
            tag=r.tag,
            steps=[{"idx": i} for i in range(r.n_steps)],
            meta={"label_failed": r.failed},
        )
        for r in runs
    ]


def test_stops_after_first_shard_reaching_target(monkeypatch):
    runs = [
        # Shard 1: reaches balanced target 2 (2 resolved, 2 unresolved across run_ids).
        _fake_run("s1_0", shard="shard_1.parquet", run_id="r1", n_steps=3, failed=0),
        _fake_run("s1_1", shard="shard_1.parquet", run_id="r1", n_steps=3, failed=1),
        _fake_run("s1_2", shard="shard_1.parquet", run_id="r1", n_steps=3, failed=0),
        _fake_run("s1_3", shard="shard_1.parquet", run_id="r1", n_steps=3, failed=1),
        # Extra run in shard 1 after target is reached: still read before the shard boundary.
        _fake_run("s1_4", shard="shard_1.parquet", run_id="r1", n_steps=3, failed=0),
        # Shard 2: must trigger break before converting or recording shard_2.
        _fake_run("s2_0", shard="shard_2.parquet", run_id="r2", n_steps=3, failed=0),
        _fake_run("s2_1", shard="shard_2.parquet", run_id="r2", n_steps=3, failed=1),
        _fake_run("s2_2", shard="shard_2.parquet", run_id="r2", n_steps=3, failed=0),
        _fake_run("s2_3", shard="shard_2.parquet", run_id="r2", n_steps=3, failed=1),
        _fake_run("s2_4", shard="shard_2.parquet", run_id="r2", n_steps=3, failed=0),
        _fake_run("s2_5", shard="shard_2.parquet", run_id="r2", n_steps=3, failed=1),
    ]
    converted_tags = []

    def _tracked_convert(batch, *, deps, convention):
        converted_tags.extend(r.tag for r in batch)
        return _fake_from_agentpile(batch, deps=deps, convention=convention)

    monkeypatch.setattr(ap, "load", lambda dataset, **kw: iter(runs))
    monkeypatch.setattr(ag_adapter, "from_agentpile", _tracked_convert)

    views, y, info = swegym.load_runs(n_per_class=2)

    assert info["shards"] == ["shard_1.parquet"]
    assert info["run_id"] == "r1"
    assert info["per_class"] == 2
    assert converted_tags == ["s1_0", "s1_1", "s1_2", "s1_3", "s1_4"]
    assert len(views) == 4 and y == [0, 0, 1, 1]


def test_drops_runs_with_fewer_than_two_steps(monkeypatch):
    runs = [
        _fake_run("zero_step", shard="s0.parquet", run_id="r1", n_steps=0, failed=0),
        _fake_run("one_step", shard="s0.parquet", run_id="r1", n_steps=1, failed=1),
        _fake_run("two_step_ok", shard="s0.parquet", run_id="r1", n_steps=2, failed=0),
        _fake_run("three_step_fail", shard="s0.parquet", run_id="r1", n_steps=3, failed=1),
    ]
    monkeypatch.setattr(ap, "load", lambda dataset, **kw: iter(runs))
    monkeypatch.setattr(ag_adapter, "from_agentpile", _fake_from_agentpile)

    views, y, info = swegym.load_runs(n_per_class=10)

    assert [v.tag for v in views] == ["two_step_ok", "three_step_fail"]
    assert y == [0, 1]
    assert info["run_ids_seen"] == {"r1": [1, 1]}


def test_keeps_run_id_with_largest_rarer_class_resolved_first(monkeypatch):
    runs = [
        # run_a: 5 resolved, 1 unresolved -> rarer class = 1
        _fake_run("a_0", shard="s0.parquet", run_id="run_a", n_steps=4, failed=0),
        _fake_run("a_1", shard="s0.parquet", run_id="run_a", n_steps=4, failed=0),
        _fake_run("a_2", shard="s0.parquet", run_id="run_a", n_steps=4, failed=0),
        _fake_run("a_3", shard="s0.parquet", run_id="run_a", n_steps=4, failed=0),
        _fake_run("a_4", shard="s0.parquet", run_id="run_a", n_steps=4, failed=0),
        _fake_run("a_f0", shard="s0.parquet", run_id="run_a", n_steps=4, failed=1),
        # run_b: 3 resolved, 2 unresolved -> rarer class = 2 (winner)
        _fake_run("b_0", shard="s0.parquet", run_id="run_b", n_steps=4, failed=0),
        _fake_run("b_f0", shard="s0.parquet", run_id="run_b", n_steps=4, failed=1),
        _fake_run("b_1", shard="s0.parquet", run_id="run_b", n_steps=4, failed=0),
        _fake_run("b_f1", shard="s0.parquet", run_id="run_b", n_steps=4, failed=1),
        _fake_run("b_2", shard="s0.parquet", run_id="run_b", n_steps=4, failed=0),
    ]
    monkeypatch.setattr(ap, "load", lambda dataset, **kw: iter(runs))
    monkeypatch.setattr(ag_adapter, "from_agentpile", _fake_from_agentpile)

    views, y, info = swegym.load_runs(n_per_class=10)

    assert info["run_id"] == "run_b"
    assert info["per_class"] == 2
    assert [v.tag for v in views] == ["b_0", "b_1", "b_f0", "b_f1"]
    assert y == [0, 0, 1, 1]


def test_empty_runs_raises_and_forwards_load_kwargs(monkeypatch):
    seen = {}

    def _empty_load(dataset, **kw):
        seen["dataset"] = dataset
        seen["kw"] = kw
        return iter([])

    monkeypatch.setattr(ap, "load", _empty_load)
    monkeypatch.setattr(ag_adapter, "from_agentpile", _fake_from_agentpile)

    with pytest.raises(RuntimeError, match="no swe_gym_openhands runs"):
        swegym.load_runs(n_per_class=5, revision="baf3a4e4")

    assert seen == {"dataset": "swe_gym_openhands", "kw": {"revision": "baf3a4e4"}}
