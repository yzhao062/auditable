"""No-download tests for ``detect/tau`` and ``auditable.bench.corpora.tau``."""
from __future__ import annotations

import json

import pytest

ap = pytest.importorskip("agentpile")
pytest.importorskip("numpy")
pytest.importorskip("networkx")

from auditable.bench import get_task, list_tasks  # noqa: E402
from auditable.bench.corpora import tau  # noqa: E402


def _call(cid: str, name: str, args: dict) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _valid_messages():
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "exchange my order"},
        {"role": "assistant", "content": None, "tool_calls": [_call("a", "get_order_details", {"id": "#1"})]},
        {"role": "tool", "tool_call_id": "a", "name": "get_order_details", "content": "order #1"},
        {"role": "assistant", "content": None,
         "tool_calls": [_call("b", "exchange_delivered_order_items", {"id": "#1"})]},
        {"role": "tool", "tool_call_id": "b", "name": "exchange_delivered_order_items", "content": "ok"},
    ]


def _make_tau_run(model: str, row: int, messages, eval_result, task_id: str = "0"):
    raw = {
        "messages": messages,
        "model_path": model,
        "task_name": "retail",
        "eval_result": eval_result,
        "meta": {"id": f"retail_{task_id}"},
    }
    where = {"file": f"{model}.jsonl", "config": "default", "split": "train", "model": model, "row": row}
    return ap.datasets.get("tau_bench").convert(f"{model}/retail/{task_id}", raw, where)


def test_tau_selection_filters_labels_and_order(monkeypatch):
    valid = _valid_messages()
    # Emit in agentpile's sorted-filename order (Kimi first) and include one failure per filter rule.
    records = [
        _make_tau_run("Kimi-K2-Instruct", 0, valid, {"db_match": False, "score": 0.0}, "0"),
        _make_tau_run("Kimi-K2-Instruct", 1, valid, {"db_match": True, "score": 1.0}, "1"),
        _make_tau_run("claude-4.5-sonnet-thinking-on-10k", 0, valid, {"db_match": True, "score": 1.0}, "2"),
        # Fewer than 3 messages even though non-system steps == 2: must be dropped.
        _make_tau_run("claude-4.5-sonnet-thinking-on-10k", 1,
                      [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                      {"db_match": False, "score": 0.0}, "3"),
        # Empty messages list: must be dropped.
        _make_tau_run("claude-4.5-sonnet-thinking-on-10k", 2, [], {"db_match": False, "score": 0.0}, "4"),
        # Missing db_match in eval_result: must be dropped.
        _make_tau_run("gpt-4.1", 0, valid, {"score": 0.0}, "5"),
        _make_tau_run("gpt-4.1", 1, valid, {"db_match": False, "score": 0.0}, "6"),
        # Three messages, but two are system messages so only one non-system step remains: must be dropped.
        _make_tau_run("gpt-4o-mini", 0,
                      [{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                       {"role": "user", "content": "only user"}],
                      {"db_match": True, "score": 1.0}, "7"),
        _make_tau_run("gpt-4o-mini", 1, valid, {"db_match": True, "score": 1.0}, "8"),
    ]

    calls = []

    def fake_load(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return iter(records)

    monkeypatch.setattr(ap, "load", fake_load)

    views, y, info = tau.load_runs(root="/fake/root")
    assert calls == [("tau_bench", {"model": list(tau.MODELS), "root": "/fake/root"})]
    assert [v.meta["run_id"] for v in views] == [
        "claude-4.5-sonnet-thinking-on-10k/retail/2",
        "gpt-4.1/retail/6",
        "gpt-4o-mini/retail/8",
        "Kimi-K2-Instruct/retail/0",
        "Kimi-K2-Instruct/retail/1",
    ]
    assert y == [0, 1, 0, 1, 0]
    assert info["dropped"] == {"messages_lt_3": 2, "no_db_match": 1, "steps_lt_2": 1}
    assert info["counts_by_file"] == {
        "claude-4.5-sonnet-thinking-on-10k.jsonl": 1,
        "gpt-4.1.jsonl": 1,
        "gpt-4o-mini.jsonl": 1,
        "Kimi-K2-Instruct.jsonl": 2,
    }


def test_tau_task_registered_and_loads(monkeypatch):
    assert "detect/tau" in {t.task_id for t in list_tasks()}
    task = get_task("detect/tau")

    valid = _valid_messages()
    records = [
        _make_tau_run("gpt-4.1", 0, valid, {"db_match": True, "score": 1.0}, "0"),
        _make_tau_run("Kimi-K2-Instruct", 0, valid, {"db_match": False, "score": 0.0}, "1"),
    ]
    monkeypatch.setattr(ap, "load", lambda dataset, **kwargs: iter(records))

    pop = task.load()
    assert len(pop.runs) == 2
    assert pop.y.tolist() == [0, 1]
    assert "2 runs (1 failed, 1 succeeded)" in task.describe(pop)
