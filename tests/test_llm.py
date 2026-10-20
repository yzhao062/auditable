"""auditable.llm and the Who&When LLM-judge localizers, with scripted backends only."""
import json

import pytest

from auditable.llm import FixedResponseBackend, OpenAICompatibleBackend, PredictionCache, input_key
from auditable.llm.who_when import (
    PROTOCOLS,
    _parse_half,
    _parse_step_answer,
    all_at_once_prompt,
    parse_prediction,
)
from auditable.methods import get_method


def _run(n=4):
    return [{"idx": i, "agent": f"agent{i % 2}", "kind": "decision" if i % 2 else "tool_call",
             "text": f"step {i} says something"} for i in range(n)]


def test_fixed_backend_list_dict_callable_and_default():
    b = FixedResponseBackend(["a", "b"], default="z")
    assert [b("p1"), b("p2"), b("p3")] == ["a", "b", "z"]
    assert b.prompts == ["p1", "p2", "p3"]
    assert FixedResponseBackend({"q": "r"})("q") == "r"
    assert FixedResponseBackend(lambda p: p.upper())("x") == "X"
    with pytest.raises(LookupError):
        FixedResponseBackend([])("anything")


def test_openai_backend_needs_no_network_to_construct(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    b = OpenAICompatibleBackend(model="m", api_key="k", base_url="http://localhost:9/v1/")
    assert b.base_url == "http://localhost:9/v1" and b.temperature == 0.0


def test_input_key_is_order_insensitive_for_dicts_and_sensitive_to_content():
    k1 = input_key("step.x", "v1", "m", {"a": 1, "b": [1, 2]})
    assert k1 == input_key("step.x", "v1", "m", {"b": [1, 2], "a": 1})
    assert k1.startswith("sha256-")
    assert k1 != input_key("step.x", "v2", "m", {"a": 1, "b": [1, 2]})
    assert k1 != input_key("step.x", "v1", "m", {"a": 1, "b": [2, 1]})


def test_prediction_cache_round_trip(tmp_path):
    path = str(tmp_path / "sub" / "cache.json")
    c = PredictionCache(path=path, provenance={"model": "m"})
    c.put("k", {"top": 1})
    assert "k" in c and len(c) == 1 and c.get("missing") is None
    c.save()
    data = json.loads(open(path, encoding="utf-8").read())
    assert data == {"provenance": {"model": "m"}, "predictions": {"k": {"top": 1}}}
    again = PredictionCache(path=path, provenance={"note": "reloaded"})
    assert again.get("k") == {"top": 1} and again.provenance == {"model": "m", "note": "reloaded"}
    with pytest.raises(ValueError):
        PredictionCache().save()


def test_parse_prediction_clamps_and_falls_back():
    assert parse_prediction("MISTAKE_STEP: 2\nRANKING: 2, 0, 9, 0, 1", 4) == {"top": 2, "ranking": [2, 0, 1]}
    assert parse_prediction("RANKING: 3, 1", 4) == {"top": 3, "ranking": [3, 1]}
    assert parse_prediction("mistake_step : 1", 4) == {"top": 1, "ranking": [1]}
    assert parse_prediction("no idea", 4) == {"top": None, "ranking": []}


def test_step_and_half_parsers_read_the_last_verdict():
    assert _parse_step_answer("it may be step 4 ... VERDICT: NO", 5) is None
    assert _parse_step_answer("first VERDICT: 1 then VERDICT: 3", 5) == 3
    assert _parse_step_answer("VERDICT: 9", 5) is None
    assert _parse_step_answer("2", 5) == 2
    assert _parse_half("not the first half. VERDICT: SECOND") == "second"
    assert _parse_half("first") == "first"
    assert _parse_half("first or second") is None


def test_all_at_once_prompt_lists_every_step():
    run = {"steps": _run(3), "texts": ["a", "b  c", "d" * 2000]}
    p = all_at_once_prompt(run)
    assert "[step 0] (agent0, tool_call): a" in p
    assert "[step 1] (agent1, decision): b c" in p
    assert "...[truncated]" in p and "indexed from 0 to 2" in p


def test_all_at_once_judge_scores_and_caches():
    backend = FixedResponseBackend(lambda p: "MISTAKE_STEP: 1\nRANKING: 1, 3")
    judge = get_method("step.llm_all_at_once")(backend=backend, model="scripted")
    trace = _run(4)
    assert judge.predict_step(trace) == 1
    assert judge.step_scores(trace).tolist() == [0.0, 5.0, 0.0, 3.0]
    assert judge.rank(trace) == [1, 3, 0, 2]
    assert len(backend.prompts) == 1  # the later calls replayed the cached prediction
    judge.predict(_run(5))
    assert len(backend.prompts) == 2  # a different trace is a different key


def test_replay_only_raises_on_a_cache_miss_and_needs_text():
    judge = get_method("step.llm_all_at_once")(model="m", replay_only=True)
    with pytest.raises(KeyError):
        judge.predict(_run(3))
    no_text = [{"idx": 0, "agent": "a", "kind": "decision"}]
    with pytest.raises(ValueError, match="step text"):
        get_method("step.llm_all_at_once")(backend=FixedResponseBackend(["x"])).predict(no_text)
    with pytest.raises(ValueError, match="backend"):
        get_method("step.llm_all_at_once")(model="m").predict(_run(2))


def test_step_by_step_stops_at_the_first_flagged_prefix():
    replies = ["VERDICT: NO", "VERDICT: NO", "VERDICT: 1"]
    backend = FixedResponseBackend(replies)
    judge = get_method("step.llm_step_by_step")(backend=backend, model="scripted")
    assert judge.predict_step(_run(5)) == 1
    assert len(backend.prompts) == 3
    assert "You are shown steps 0 through 2" in backend.prompts[-1]


def test_step_by_step_defaults_to_the_last_step():
    judge = get_method("step.llm_step_by_step")(backend=FixedResponseBackend(lambda p: "VERDICT: NO"), model="s")
    assert judge.predict_step(_run(4)) == 3


def test_binary_search_halves_the_interval():
    backend = FixedResponseBackend(["VERDICT: SECOND", "VERDICT: FIRST", "VERDICT: SECOND"])
    judge = get_method("step.llm_binary_search")(backend=backend, model="scripted")
    # [0, 7] -> second [4, 7] -> first [4, 5] -> second [5, 5]
    assert judge.predict_step(_run(8)) == 5
    assert len(backend.prompts) == 3


@pytest.mark.parametrize("mid,replies", [
    ("step.llm_all_at_once", ["MISTAKE_STEP: 2\nRANKING: 2, 0"]),
    ("step.llm_step_by_step", ["VERDICT: NO", "VERDICT: NO", "VERDICT: 2"]),
    ("step.llm_binary_search", ["VERDICT: SECOND", "VERDICT: FIRST"]),  # [0, 3] -> [2, 3] -> [2, 2]
])
def test_judges_label_steps_by_position_not_idx(mid, replies):
    plain = _run(4)
    renumbered = [dict(s, idx=10 * (i + 1)) for i, s in enumerate(plain)]
    outs = []
    for trace in (plain, renumbered):
        backend = FixedResponseBackend(list(replies))
        judge = get_method(mid)(backend=backend, model="scripted")
        outs.append((judge.predict(trace)["top"], backend.prompts))
    assert outs[0] == outs[1]  # same prompts, same answer
    assert outs[0][0] == 2


def test_protocols_return_prediction_and_log():
    pred, raw = PROTOCOLS["binary_search"]({"steps": _run(2), "texts": ["a", "b"]}, lambda p: "VERDICT: FIRST")
    assert pred == {"top": 0, "ranking": [0]} and raw.startswith("[0-1 mid0] FIRST")
