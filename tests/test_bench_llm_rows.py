"""Tests for Who&When LLM-judge leaderboard rows without downloading datasets or calling APIs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from auditable.bench.llm_rows import (
    ReplayBackend,
    discover_llm_caches,
    is_llm_entry,
    llm_entries,
    llm_entry_key,
    llm_entry_label,
    score_llm_entry,
)
from auditable.llm import FixedResponseBackend
from auditable.llm.who_when import all_at_once_prompt


def _synthetic_trace(n_steps: int = 4, tag: str = "a"):
    return [
        {
            "idx": i,
            "agent": f"agent{i % 2}",
            "kind": "decision" if i % 2 == 0 else "tool_call",
            "text": f"trace {tag} step {i}",
        }
        for i in range(n_steps)
    ]


def test_discovery_finds_all_31_committed_caches():
    entries = llm_entries()
    assert len(entries) == 31
    by_method = {}
    for entry in entries:
        by_method.setdefault(entry.card.method_id, []).append(entry)
    assert len(by_method["step.llm_all_at_once"]) == 11
    assert len(by_method["step.llm_binary_search"]) == 10
    assert len(by_method["step.llm_step_by_step"]) == 10
    assert all(is_llm_entry(e) for e in entries)


def test_key_and_label_format_match_golden_file():
    entries = llm_entries()
    import auditable.bench

    # the copy the package carries, so the test also runs against an installed wheel
    golden_path = os.path.join(os.path.dirname(auditable.bench.__file__), "golden", "localize__whoandwhen__llm.json")
    with open(golden_path, encoding="utf-8") as fh:
        golden_rows = json.load(fh)["rows"]

    assert {e.key for e in entries} == set(golden_rows)
    for entry in entries:
        protocol = entry.card.method_id.removeprefix("step.llm_")
        model = entry.kwargs["model"]
        assert entry.key == llm_entry_key(protocol, model) == f"step.llm_{protocol}:{model}"
        assert entry.label == llm_entry_label(protocol, model)
        assert entry.label == golden_rows[entry.key]["board_label"]
        assert isinstance(entry.kwargs["backend"], ReplayBackend)
        assert isinstance(entry.kwargs["backend"], FixedResponseBackend)


def test_discovery_skips_partial_and_unknown_caches(tmp_path):
    legacy = {"format": "v1", "keys": {"000-aaa": "sha256-aaa", "001-bbb": "sha256-bbb"}}
    (tmp_path / "legacy_run_keys.json").write_text(json.dumps(legacy), encoding="utf-8")

    valid_pred = {"top": 0, "ranking": [0, 1], "raw": "MISTAKE_STEP: 0\nRANKING: 0, 1"}
    good = {"predictions": {"000-aaa": valid_pred, "001-bbb": valid_pred}}
    partial = {"predictions": {"000-aaa": valid_pred}}
    bad_shape = {"predictions": {"000-aaa": {"top": "0"}, "001-bbb": valid_pred}}

    (tmp_path / "whoandwhen__all_at_once__m-good.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "whoandwhen__all_at_once__m-partial.json").write_text(json.dumps(partial), encoding="utf-8")
    (tmp_path / "whoandwhen__all_at_once__m-bad.json").write_text(json.dumps(bad_shape), encoding="utf-8")
    (tmp_path / "whoandwhen__unknown_proto__m-good.json").write_text(json.dumps(good), encoding="utf-8")

    found = discover_llm_caches(cache_dir=str(tmp_path))
    assert [(proto, model) for proto, model, _ in found] == [("all_at_once", "m-good")]


def test_replay_backend_refuses_uncached_prompt():
    empty_backend = ReplayBackend({}, method="all_at_once", model="gpt-5.5")
    with pytest.raises(LookupError, match="uncached prompt"):
        empty_backend("any prompt")

    trace = _synthetic_trace(4, tag="uncached")
    prompt = all_at_once_prompt({"steps": trace, "texts": [s["text"] for s in trace]})
    mapped = ReplayBackend({prompt: "MISTAKE_STEP: 1\nRANKING: 1, 0"})
    assert mapped(prompt) == "MISTAKE_STEP: 1\nRANKING: 1, 0"
    with pytest.raises(LookupError, match="uncached prompt"):
        mapped("a different uncached prompt")

    seq = ReplayBackend(["VERDICT: NO"])
    assert seq("p0") == "VERDICT: NO"
    with pytest.raises(LookupError, match="uncached prompt"):
        seq("p1")

    entry = llm_entries()[0]
    judge = entry.make()
    with pytest.raises(LookupError, match="uncached prompt"):
        judge.predict(trace)


def test_score_llm_entry_synthetic_all_three_protocols_no_download(tmp_path):
    # Two runs with identical steps and texts to verify per-run isolation (no cache collision).
    t0 = _synthetic_trace(4, tag="dup")
    t1 = _synthetic_trace(4, tag="dup")
    legacy = {"format": "v1", "keys": {"000-r0": "sha256-r0", "001-r1": "sha256-r1"}}
    (tmp_path / "legacy_run_keys.json").write_text(json.dumps(legacy), encoding="utf-8")

    aao = {
        "predictions": {
            "000-r0": {"top": 1, "ranking": [1, 2, 0, 3], "raw": "MISTAKE_STEP: 1\nRANKING: 1, 2, 0, 3"},
            "001-r1": {"top": 2, "ranking": [2, 1, 0, 3], "raw": "MISTAKE_STEP: 2\nRANKING: 2, 1, 0, 3"},
        }
    }
    sbs = {
        "predictions": {
            "000-r0": {"top": 1, "ranking": [1], "raw": "[0..0] VERDICT: NO\n[0..1] VERDICT: 1"},
            "001-r1": {"top": 2, "ranking": [2], "raw": "[0..0] VERDICT: NO\n[0..1] VERDICT: NO\n[0..2] VERDICT: 2"},
        }
    }
    bs = {
        "predictions": {
            # [0-3 mid1] FIRST -> [0-1 mid0] SECOND -> step 1
            "000-r0": {"top": 1, "ranking": [1], "raw": "[0-3 mid1] FIRST :: x\n[0-1 mid0] SECOND :: y"},
            # [0-3 mid1] SECOND -> [2-3 mid2] FIRST -> step 2
            "001-r1": {"top": 2, "ranking": [2], "raw": "[0-3 mid1] SECOND :: x\n[2-3 mid2] FIRST :: y"},
        }
    }
    (tmp_path / "whoandwhen__all_at_once__m.json").write_text(json.dumps(aao), encoding="utf-8")
    (tmp_path / "whoandwhen__step_by_step__m.json").write_text(json.dumps(sbs), encoding="utf-8")
    (tmp_path / "whoandwhen__binary_search__m.json").write_text(json.dumps(bs), encoding="utf-8")

    # Run 0 mistake is 1 (hit for both); run 1 mistake is 1 (aao ranks step 1 second, single-step misses).
    pop = SimpleNamespace(runs=[t0, t1], mistakes=[1, 1])
    by_key = {e.key: e for e in llm_entries(cache_dir=str(tmp_path))}
    assert set(by_key) == {"step.llm_all_at_once:m", "step.llm_step_by_step:m", "step.llm_binary_search:m"}

    aao_scores = score_llm_entry(by_key["step.llm_all_at_once:m"], pop, cache_dir=str(tmp_path))
    assert aao_scores == pytest.approx({"top1": 0.5, "top3": 1.0, "mrr": 0.75})

    sbs_scores = score_llm_entry(by_key["step.llm_step_by_step:m"], pop, cache_dir=str(tmp_path))
    assert sbs_scores == pytest.approx({"top1": 0.5, "top3": 0.5, "mrr": 0.5})

    bs_scores = score_llm_entry(by_key["step.llm_binary_search:m"], pop, cache_dir=str(tmp_path))
    assert bs_scores == pytest.approx({"top1": 0.5, "top3": 0.5, "mrr": 0.5})

    # Separate step_texts attribute on population also works.
    bare = [[{"idx": s["idx"], "agent": s["agent"], "kind": s["kind"]} for s in t] for t in (t0, t1)]
    texts = [[s["text"] for s in t] for t in (t0, t1)]
    pop_sep = SimpleNamespace(runs=bare, step_texts=texts, mistakes=[1, 1])
    assert score_llm_entry(by_key["step.llm_all_at_once:m"], pop_sep, cache_dir=str(tmp_path)) == pytest.approx(aao_scores)


def test_score_llm_entry_raises_on_replay_mismatch(tmp_path):
    t0 = _synthetic_trace(4, tag="m")
    legacy = {"format": "v1", "keys": {"000-r0": "sha256-r0"}}
    (tmp_path / "legacy_run_keys.json").write_text(json.dumps(legacy), encoding="utf-8")
    bad = {
        "predictions": {
            # Recorded top=3 disagrees with parsed MISTAKE_STEP: 1
            "000-r0": {"top": 3, "ranking": [1, 2], "raw": "MISTAKE_STEP: 1\nRANKING: 1, 2"},
        }
    }
    (tmp_path / "whoandwhen__all_at_once__m.json").write_text(json.dumps(bad), encoding="utf-8")
    entry = llm_entries(cache_dir=str(tmp_path))[0]
    pop = SimpleNamespace(runs=[t0], mistakes=[1])
    with pytest.raises(ValueError, match="library replay gives top"):
        score_llm_entry(entry, pop, cache_dir=str(tmp_path))


def test_llm_rows_defers_third_party_imports():
    import ast

    import auditable.bench.llm_rows

    llm_rows_path = os.path.abspath(auditable.bench.llm_rows.__file__)
    tree = ast.parse(open(llm_rows_path, encoding="utf-8").read())
    top_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    assert not (top_imports & {"numpy", "sklearn", "agentpile", "torch", "pygod"})

    # the parent of the imported package, so the subprocess imports the same copy (a checkout or site-packages)
    src = os.path.dirname(os.path.dirname(os.path.abspath(auditable.__file__)))
    code = (
        "import sys; import auditable.bench.llm_rows as m; "
        "loaded = [k for k in ('agentpile', 'torch', 'pygod') if k in sys.modules]; "
        "assert not loaded, loaded"
    )
    env = dict(os.environ, PYTHONPATH=src)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
