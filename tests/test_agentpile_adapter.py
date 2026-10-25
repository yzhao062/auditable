"""auditable.from_agentpile and its helpers, on records built offline by agentpile's own readers.

Every record here is synthetic: a short OpenAI-style conversation, or a raw row in a dataset's
layout passed to that dataset's ``convert``, so the tests need no download and copy no data.
"""
import json
import warnings
from pathlib import Path

import pytest

ap = pytest.importorskip("agentpile")
np = pytest.importorskip("numpy")
pytest.importorskip("networkx")

import auditable as au  # noqa: E402
from auditable.datasets import load_traces  # noqa: E402
from auditable.featurize import TraceView, grade_layer, normalize_trace, to_pyg  # noqa: E402
from auditable.llm import FixedResponseBackend  # noqa: E402
from auditable.methods import get_method  # noqa: E402


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


MESSAGES = [
    {"role": "system", "content": "You are a booking agent."},
    {"role": "user", "content": "Cancel reservation R1."},
    {"role": "assistant", "content": None, "tool_calls": [_call("c1", "get_reservation", {"id": "R1"})]},
    {"role": "tool", "tool_call_id": "c1", "content": "reservation R1, flight F9"},
    {"role": "assistant", "content": None, "tool_calls": [_call("c2", "cancel_reservation", {"id": "R1"})]},
    {"role": "tool", "tool_call_id": "c2", "content": "cancelled"},
    {"role": "assistant", "content": "Done."},
]


def _messages_run():
    return next(ap.load(MESSAGES))


def _who_and_when_run(sub="Algorithm-Generated", mistake_step=1, outcome=False):
    raw = {"question": "q", "question_ID": "qid", "mistake_agent": "Coder", "mistake_reason": "r",
           "mistake_step": str(mistake_step), "system_prompt": {}}
    if sub == "Algorithm-Generated":
        raw["history"] = [{"name": "Planner", "role": "assistant", "content": "plan"},
                          {"name": "Coder", "role": "assistant", "content": "code"},
                          {"name": "Computer_terminal", "role": "user", "content": "exitcode: 0"},
                          {"name": "Coder", "role": "assistant", "content": "answer"}]
        raw["is_correct"] = outcome
    else:
        raw["history"] = [{"role": "human", "content": "question"},
                          {"role": "Orchestrator (thought)", "content": "think"},
                          {"role": "Orchestrator (-> WebSurfer)", "content": "go"},
                          {"role": "WebSurfer", "content": "page"}]
        raw["is_corrected"] = outcome
    where = {"file": f"Who&When/{sub}/1.json", "config": sub, "split": "train"}
    return ap.datasets.get("who_and_when").convert(f"{sub}/1", raw, where)


def _tau_run(messages, db_match=False):
    raw = {"messages": messages, "model_path": "openai/gpt-x", "task_name": "retail",
           "eval_result": {"db_match": db_match, "score": 1.0 if db_match else 0.0}, "meta": {"id": "retail_7"}}
    where = {"file": "gpt-x.jsonl", "config": "default", "split": "train", "model": "gpt-x", "row": 0}
    return ap.datasets.get("tau_bench").convert("gpt-x/retail/7", raw, where)


# -- generic convention --------------------------------------------------------------------------------

def test_generic_steps_texts_addresses_and_rule_edges():
    run = _messages_run()
    view = au.from_agentpile(run, deps=["tool_call_id_v1"])
    assert isinstance(view, TraceView) and view.source == "agentpile"
    assert [(s["kind"], s["agent"]) for s in view.steps] == [
        ("decision", "user"), ("decision", "agent"), ("tool_call", "env"), ("decision", "agent"),
        ("tool_call", "env"), ("decision", "agent")]
    assert view.texts[2] == 'get_reservation({"id": "R1"}) -> reservation R1, flight F9'
    assert view.meta["report"]["system_messages_skipped"] == 1
    # the tool step stands for its call and its result
    assert [set(a) for a in view.meta["addresses"][2]] == [{"trajectory", "step", "call"},
                                                           {"trajectory", "step", "result"}]
    # tool_call_id: the next agent step after a result depends on the call that produced it
    assert view.steps[3]["deps"] == [2] and view.steps[5]["deps"] == [4]
    assert view.meta["dep_rules"] == [{"requested": "tool_call_id_v1", "rule": "tool_call_id_v1",
                                       "rule_version": "1", "extractor": None, "edges": 2}]
    assert view.shared_resource is True and view.meta["convention"] == "generic"


def test_a_list_gives_a_list_and_a_run_gives_one_view():
    run = _messages_run()
    assert isinstance(au.from_agentpile([run, run], deps="full_context"), list)
    assert len(au.from_agentpile(iter([run]), deps="full_context")) == 1
    once = au.from_agentpile([run, run], deps=(r for r in ["tool_call_id_v1"]))  # a generator serves every run
    assert [v.meta["report"]["edges"] for v in once] == [2, 2]


def test_deps_is_required_and_checked():
    run = _messages_run()
    with pytest.raises(TypeError):
        au.from_agentpile(run)
    with pytest.raises(ValueError, match="needs convention"):
        au.from_agentpile(run, deps="grade")
    with pytest.raises(ValueError, match="convention"):
        au.from_agentpile(run, deps="full_context", convention="auto")
    with pytest.raises(KeyError):
        au.from_agentpile(run, deps=["no_such_rule"])
    with pytest.raises(TypeError):
        au.from_agentpile([{"steps": []}], deps="full_context")


def test_full_context_and_recorded_modes():
    run = _messages_run()
    full = au.from_agentpile(run, deps="full_context")
    assert [s["deps"] for s in full.steps] == [list(range(i)) for i in range(len(full.steps))]
    by_rule = au.from_agentpile(run, deps=["tool_call_id_v1"])
    recorded = au.from_agentpile(run.with_deps(["tool_call_id_v1"]), deps="recorded")
    assert [s["deps"] for s in recorded.steps] == [s["deps"] for s in by_rule.steps]
    assert recorded.meta["dep_rules"][0]["rule"] == "tool_call_id"


def test_warnings_for_unknown_history_and_rules_that_found_nothing():
    run = _messages_run()
    with pytest.warns(UserWarning, match="whether any rule ran is unknown"):
        au.from_agentpile(run, deps="recorded")
    chat = next(ap.load([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]))
    with pytest.warns(UserWarning, match="produced no edge"):
        view = au.from_agentpile(chat, deps=["tool_call_id_v1"])
    assert view.meta["dep_rules"][0]["edges"] == 0  # the rule is recorded although it found nothing
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        au.from_agentpile(chat, deps=[])  # an empty rule list asks for no edges


def test_custom_rule_endpoints_resolve_through_the_step_address_table_and_edges_are_deduplicated():
    run = _messages_run()
    tid = run.id

    # ATIF steps: 1 system, 2 user, 3 assistant with call c1, 4 assistant with call c2, 5 the answer
    def link(r):
        return [{"src": {"trajectory": tid, "step": 3}, "dst": {"trajectory": tid, "step": 5},
                 "rule": "link", "rule_version": "1", "resource": None},
                {"src": {"trajectory": tid, "step": 5}, "dst": {"trajectory": tid, "step": 3},
                 "rule": "link", "rule_version": "1", "resource": None}]

    view = au.from_agentpile(run, deps=[link, link])
    # a bare step address means the step's decision (position 1), not its tool step (position 2)
    assert view.steps[5]["deps"] == [1]
    assert view.meta["report"]["dropped_not_earlier"] == 2 and view.meta["report"]["edges"] == 1


def test_subagents_are_refused_unless_skipped():
    record = _messages_run().to_dict()
    record["subagent_trajectories"] = [dict(record, session_id="child")]
    run = ap.Run(record)
    with pytest.raises(ValueError, match="1 subagent"):
        au.from_agentpile(run, deps="full_context")
    view = au.from_agentpile(run, deps="full_context", subagents="skip")
    assert view.meta["subagents_skipped"] == 1


def test_normalize_trace_points_a_run_at_from_agentpile():
    with pytest.raises(TypeError, match="from_agentpile"):
        normalize_trace(_messages_run())


# -- grade convention -----------------------------------------------------------------------------------

def test_grade_who_and_when_keeps_raw_roles_and_maps_the_label():
    run = _who_and_when_run("Hand-Crafted", mistake_step=2)
    view = au.from_agentpile(run, deps="grade", convention="grade")
    assert [s["agent"] for s in view.steps] == ["human", "Orchestrator (thought)", "Orchestrator (-> WebSurfer)",
                                                "WebSurfer"]
    assert view.shared_resource is True and view.meta["projection"] == "grade-steps-v1"
    assert au.step_targets([view]).tolist() == [2]
    assert au.run_targets([view]).tolist() == [-1]  # Hand-Crafted has is_corrected, which GRADE does not read
    ag = au.from_agentpile(_who_and_when_run("Algorithm-Generated"), deps="grade", convention="grade")
    assert [s["kind"] for s in ag.steps] == ["decision", "decision", "tool_call", "decision"]
    assert ag.steps[3]["deps"] == [0, 1, 2] and au.run_targets([ag]).tolist() == [1]
    assert ag.texts == ["plan", "code", "exitcode: 0", "answer"]


def test_out_of_range_step_label_is_minus_one():
    view = au.from_agentpile(_who_and_when_run(mistake_step=9), deps="grade", convention="grade")
    assert au.step_targets([view]).tolist() == [-1]


def test_grade_tau_bench_uses_model_path_tool_steps_at_results_and_no_shared_node():
    msgs = [{"role": "system", "content": "policy"}, {"role": "user", "content": "exchange my order"},
            {"role": "assistant", "content": None, "tool_calls": [_call("a", "get_order_details", {"id": "#1"})]},
            {"role": "tool", "tool_call_id": "a", "name": "get_order_details", "content": "order #1"},
            {"role": "assistant", "content": None,
             "tool_calls": [_call("b", "exchange_delivered_order_items", {"id": "#1"})]},
            {"role": "tool", "tool_call_id": "b", "name": "exchange_delivered_order_items", "content": "ok"}]
    view = au.from_agentpile(_tau_run(msgs), deps="grade", convention="grade")
    assert [(s["agent"], s["kind"]) for s in view.steps] == [
        ("user", "decision"), ("openai/gpt-x", "decision"), ("env", "tool_call"), ("openai/gpt-x", "decision"),
        ("env", "tool_call")]
    assert view.steps[4]["deps"] == [2]  # the write depends on the earlier read
    assert view.shared_resource is False and au.run_targets([view]).tolist() == [1]
    assert view.meta["dep_rules"][0]["extractor"] == "db_tool_names_v1"


def test_reversed_results_are_reported_not_silently_accepted():
    # one message calls a read then a write; the results arrive write first. GRADE places tool steps at
    # the results and derives no edge; the rule derives read -> write, which then points at a later step.
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None,
             "tool_calls": [_call("r", "get_order_details", {}), _call("w", "cancel_pending_order", {})]},
            {"role": "tool", "tool_call_id": "w", "name": "cancel_pending_order", "content": "done"},
            {"role": "tool", "tool_call_id": "r", "name": "get_order_details", "content": "order"}]
    with pytest.warns(UserWarning, match="produced no edge"):
        view = au.from_agentpile(_tau_run(msgs), deps="grade", convention="grade")
    assert all(s["deps"] == [] for s in view.steps)
    assert view.meta["report"]["dropped_not_earlier"] == 1


def test_grade_needs_a_covered_reader():
    with pytest.raises(ValueError, match="covers the readers"):
        au.from_agentpile(_messages_run(), deps=["tool_call_id_v1"], convention="grade")


# -- grade convention on agentpile's GRADE fixtures (tests/fixtures/agentpile_grade/README.md) ------------

FIX = Path(__file__).parent / "fixtures" / "agentpile_grade"


def _sample(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def _openhands_run(raw):
    where = {"file": "trajectories.parquet", "config": "default", "split": "train", "row": 0, "tool_definitions": []}
    return ap.datasets.get("swe_rebench_openhands").convert(raw["trajectory_id"], raw, where)


def _swe_agent_run(raw):
    where = {"file": "data/train-00000-of-00012.parquet", "config": "default", "split": "train"}
    return ap.datasets.get("swe_agent_nebius").convert("train-00000-of-00012:0", raw, where)


def _arb_run(raw):
    key = f"{raw['benchmark']}/{raw['model_name']}/{raw['task_id']}"
    where = {"file": f"cleaned/{raw['benchmark']}/{raw['model_name']}/run/{raw['task_id']}.json",
             "config": raw["benchmark"], "split": "full", "task_id": raw["task_id"]}
    head = {"goal": "Find shoes.", "agent": raw["model_name"], "model": "demo-model"}
    return ap.datasets.get("agent_reward_bench").convert(
        key, {"head": head, "steps": raw["records"], "annotations": []}, where)


def _grade_fields(view):
    assert all(view.meta["addresses"])  # every step stands for at least one record address
    return [{k: s[k] for k in ("idx", "agent", "kind", "deps")} for s in view.steps]


def test_grade_openhands_gives_each_call_of_a_turn_its_own_tool_step():
    sample = _sample("b_openhands_multi_call_turn")
    view = au.from_agentpile(_openhands_run(sample["raw"]), deps="grade", convention="grade")
    assert _grade_fields(view) == sample["expected_steps_hand"]
    assert [s["kind"] for s in view.steps[:4]] == ["decision", "tool_call", "tool_call", "tool_call"]
    assert [sorted(a) for a in view.meta["addresses"][2]] == [["call", "step", "trajectory"],
                                                              ["result", "step", "trajectory"]]
    assert view.texts[1] == 'str_replace_editor({"command": "view", "path": "/repo/a.py"}) -> a.py contents'


def test_grade_swe_agent_reads_the_command_and_resolves_bare_step_endpoints_by_rule():
    sample = _sample("swe_agent_synthetic")
    run = _swe_agent_run(sample["raw"])
    view = au.from_agentpile(run, deps="grade", convention="grade")
    # GRADE's rule names the turn by its step address and means the command it ran: the tool step
    assert _grade_fields(view) == sample["expected_steps"]
    assert view.texts[1].startswith("ls -a -> a.py b.py") and view.texts[33] == "submit -> "
    assert view.meta["addresses"][33] == [{"trajectory": run.id, "step": 20}]  # the last command has no result

    # a rule of the caller's own: the same bare step address means the turn's decision
    def link(r):
        return [{"src": {"trajectory": run.id, "step": 3}, "dst": {"trajectory": run.id, "step": 4},
                 "rule": "link", "rule_version": "1", "resource": None}]

    custom = au.from_agentpile(run, deps=[link], convention="grade")
    assert custom.steps[2]["deps"] == [0] and custom.meta["report"]["edges"] == 1


def test_grade_agent_reward_bench_drops_the_page_after_the_last_action_and_keeps_actions_in_texts():
    sample = _sample("agent_reward_bench_records")
    view = au.from_agentpile(_arb_run(sample["raw"]), deps="grade", convention="grade")
    assert _grade_fields(view) == sample["expected_steps"][:-2]  # GRADE's last two steps are the E17 tail
    assert view.texts[9] == "fill('3', 'shoes') -> page text"
    backend = FixedResponseBackend(lambda p: "MISTAKE_STEP: 9\nRANKING: 9")
    get_method("step.llm_all_at_once")(backend=backend, model="scripted").predict_step(view)
    assert "fill('3', 'shoes')" in backend.prompts[0]


# -- shared_resource follows the trace ------------------------------------------------------------------

def test_shared_resource_survives_normalization_and_reaches_the_layer_and_graph_entry_points():
    steps = [{"idx": 0, "agent": "a", "kind": "decision", "deps": []},
             {"idx": 1, "agent": "env", "kind": "tool_call", "deps": [0]}]
    off = TraceView(steps=[dict(s) for s in steps], shared_resource=False)
    on = TraceView(steps=[dict(s) for s in steps])
    assert "resource::shared_state" not in off.typed_graph() and "resource::shared_state" in on.typed_graph()
    assert normalize_trace(off, dependency="full_context").shared_resource is False
    assert grade_layer(off) == grade_layer(on, shared_resource=False)
    assert to_pyg(off.typed_graph())[0].shape[0] == to_pyg(on.typed_graph())[0].shape[0] - 1
    built = on.typed_graph()
    assert TraceView(steps=steps, graph=built, shared_resource=False).typed_graph() is built


# -- step.blast warns only when full context changes the recorded dependencies ----------------------------

def test_blast_warns_only_when_full_context_overrides_recorded_deps():
    blast = get_method("step.blast")()
    explicit = [{"idx": 0, "agent": "a", "kind": "decision", "deps": []},
                {"idx": 1, "agent": "a", "kind": "decision", "deps": []},
                {"idx": 2, "agent": "a", "kind": "decision", "deps": [1]}]
    with pytest.warns(UserWarning, match="dependency='explicit'"):
        blast.step_scores(explicit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        blast.step_scores(au.from_agentpile(_messages_run(), deps="full_context"))
        blast.step_scores([{"idx": i, "agent": "a", "kind": "decision"} for i in range(3)])


# -- labels -------------------------------------------------------------------------------------------

def test_run_targets_reads_the_outcome_or_a_named_label():
    ok = au.from_agentpile(_tau_run([{"role": "user", "content": "hi"}], db_match=True), deps=[])
    bad = au.from_agentpile(_tau_run([{"role": "user", "content": "hi"}], db_match=False), deps=[])
    plain = au.from_agentpile(_messages_run(), deps=[])
    y = au.run_targets([ok, bad, plain])
    assert y.dtype.kind == "i" and y.tolist() == [0, 1, -1]
    assert au.run_targets([ok, bad], label="score", failed=lambda v: v == 0.0).tolist() == [0, 1]
    with pytest.raises(TypeError, match="failed="):
        au.run_targets([ok], label="score")
    assert au.step_targets([plain]).tolist() == [-1]


def test_load_traces_matches_make_traces_shapes(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(MESSAGES), encoding="utf-8")
    traces, y, mistakes = load_traces(str(path), deps=["tool_call_id_v1"])
    assert len(traces) == 1 and y.tolist() == [-1] and mistakes.tolist() == [-1]
    assert get_method("step.dep_anomaly")().rank(traces[0])


def test_llm_judge_reads_the_step_texts_in_position_order():
    view = au.from_agentpile(_who_and_when_run(), deps="grade", convention="grade")
    backend = FixedResponseBackend(lambda p: "MISTAKE_STEP: 2\nRANKING: 2, 1")
    judge = get_method("step.llm_all_at_once")(backend=backend, model="scripted")
    assert judge.predict_step(view) == 2
    prompt = backend.prompts[0]
    assert prompt.index("plan") < prompt.index("code") < prompt.index("exitcode: 0") < prompt.index("answer")


# -- configurations -------------------------------------------------------------------------------------

def _config(agents, tools, task=None):
    record = {"schema": "agentpile-config/0", "source": {"record_key": "cfg", "reader": "test"},
              "task": {"id": "t", "text": task}, "agents": agents, "tools": tools, "labels": []}
    return ap.Run(record)


TOOLS = {"read_file": {"name": "read_file", "type": "fs", "permission": "read", "permission_from": "annotation"},
         "send_email": {"name": "send_email", "type": "net", "permission": "write", "permission_from": "annotation"}}


def test_configs_follow_tool_ownership():
    (one,) = au.configs_from_agentpile(_config([], TOOLS, task="read the report"))
    assert one.instance_id == "cfg" and one.capability_names == ("read_file", "send_email")
    assert one.declared_capabilities[0]["permission_from"] == "annotation"
    assert one.task_or_role_spec == "read the report"
    a, b = au.configs_from_agentpile(_config([{"id": "a", "tools": []}, {"id": "b", "tools": ["send_email"]}],
                                             TOOLS, task="email it"))
    assert (a.instance_id, a.capability_names) == ("cfg:a", ())
    assert (b.instance_id, b.capability_names) == ("cfg:b", ("send_email",))
    (all_tools,) = au.configs_from_agentpile(_config([{"id": "a"}], TOOLS, task="x"))
    assert all_tools.capability_names == ("read_file", "send_email")
    with pytest.raises(ValueError, match="does not define"):
        au.configs_from_agentpile(_config([{"id": "a", "tools": ["nope"]}], TOOLS, task="x"))


def test_a_configuration_without_task_evidence_is_refused_unless_allowed():
    run = _config([], TOOLS)
    with pytest.raises(ValueError, match="no task text"):
        au.configs_from_agentpile(run)
    (cfg,) = au.configs_from_agentpile(run, allow_no_task=True)
    assert cfg.spec_tokens == ()
    (given,) = au.configs_from_agentpile(run, task="send the weekly email")
    assert given.task_or_role_spec == "send the weekly email"
    finding = get_method("config.flag_risky_perms")().audit(given)
    assert finding.excess == frozenset({"send_email"})


def test_the_example_runs_offline(capsys):
    import runpy
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "example_agentpile.py"
    runpy.run_path(str(example), run_name="example")["main"]()
    out = capsys.readouterr().out
    assert "('value_flow_v1', 2)" in out and "config.owasp_asi_combined" in out


def test_catchbench_pre_rows_carry_their_tokens_even_when_empty():
    raw = {"instance_id": "crewai-9999", "source": "synth_crewai", "spec_tokens": [], "spec_token_overrides": {},
           "declared_capabilities": [{"name": "SerperDevTool", "type": "search", "permission_level": "network"}],
           "minimal_reference": [], "labels": {"excess_set": [], "label_source": "test"}}
    run = ap.datasets.get("catchbench_pre").convert("crewai-9999", raw, {"file": "data/pre/synthetic.json",
                                                                        "config": "synthetic", "split": None})
    (cfg,) = au.configs_from_agentpile(run)
    assert cfg.instance_id == "crewai-9999" and cfg.spec_tokens == ()
    assert cfg.declared_capabilities[0]["permission_level"] == "network"


# -- generic convention on content parts, source-step coverage, and refusals --------------------------

PARTS = [
    {"role": "user", "content": [{"type": "text", "text": "look"},
                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUFBQQ=="}}]},
    {"role": "assistant", "content": None, "tool_calls": [_call("c1", "read", {}), _call("c2", "list", {})]},
    {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": "ok"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,QkJCQg=="}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "done"},
    {"role": "assistant", "content": "fine"},
]


def test_content_parts_keep_their_text_and_name_every_other_part():
    view = au.from_agentpile(next(ap.load(PARTS)), deps="full_context")
    assert "base64" not in "".join(view.texts)
    assert view.texts[0] == "look\n[image_url]"
    assert view.texts[2] == "read({}) -> ok\n[image_url]" and view.texts[3] == "list({}) -> done"
    report = view.meta["report"]
    assert report["message_parts_replaced"] == 1 and report["result_parts_replaced"] == 1


def test_a_plain_json_list_is_not_read_as_content_parts():
    from auditable.featurize.agentpile import _content_text

    report = {"message_parts_replaced": 0}
    assert _content_text([1, {"a": 1}], report, "message") == '[1, {"a": 1}]'
    assert _content_text([], report, "message") == "[]"
    assert _content_text(["a", "b"], report, "message") == '["a", "b"]'  # strings alone are a JSON array
    assert _content_text([{"type": "text", "text": "x"}, {"no_type": 1}], report, "message").startswith("[{")
    assert report["message_parts_replaced"] == 0


def test_content_parts_may_mix_in_bare_strings_and_keep_only_the_type_of_other_parts():
    from auditable.featurize.agentpile import _content_text

    report = {"result_parts_replaced": 0}
    image = {"type": "image", "source": {"type": "base64", "data": "QUFBQQ=="}}
    assert _content_text(["caption", image], report, "result") == "caption\n[image]"
    assert _content_text([{"type": "file", "path": "/tmp/a.txt", "size": 42}], report, "result") == "[file]"
    assert report["result_parts_replaced"] == 2


def _with_steps(run, *steps):
    record = run.to_dict()
    record["steps"] = record["steps"] + list(steps)
    return ap.Run(record)


def test_every_source_step_is_mapped_or_counted_by_role():
    run = next(ap.load(PARTS))
    report = au.from_agentpile(run, deps="full_context").meta["report"]
    # three ATIF steps (user, agent with two calls and their results, agent) give five view steps
    assert (report["source_steps"], report["source_steps_mapped"], report["source_steps_unmapped"]) == (3, 3, 0)
    assert report["view_steps"] == 5 and report["unmapped_by_source"] == {}
    report = au.from_agentpile(_messages_run(), deps="full_context").meta["report"]
    assert report["source_steps_mapped"] + report["source_steps_unmapped"] == report["source_steps"]
    assert report["unmapped_by_source"] == {"system": 1} and report["system_messages_skipped"] == 1


def test_coverage_counts_steps_with_no_event_and_maps_a_step_holding_only_an_orphan_result():
    missing = {"agentpile": {"message_state": "missing"}}
    run = _with_steps(
        next(ap.load(PARTS)),
        # a step whose only event is a result naming no call it can be joined to
        {"step_id": 4, "source": "agent", "message": "", "extra": missing,
         "observation": {"results": [{"source_call_id": None, "content": "late"}]}},
        {"step_id": 5, "source": "system", "message": "", "extra": missing},  # no event at all
        {"step_id": 6, "source": "agent", "message": "", "extra": missing},  # no event at all
    )
    view = au.from_agentpile(run, deps=[])
    report = view.meta["report"]
    assert (report["source_steps"], report["source_steps_mapped"], report["source_steps_unmapped"]) == (6, 4, 2)
    assert report["unmapped_by_source"] == {"system": 1, "agent": 1}
    assert report["results_without_call"] == 1 and report["view_steps"] == 6 == len(view.steps)
    assert view.texts[-1] == "None() -> late"
    assert [a["step"] for a in view.meta["addresses"][-1]] == [4] and "result" in view.meta["addresses"][-1][0]


def test_the_subagent_refusal_names_both_ways_to_read_the_run():
    record = _messages_run().to_dict()
    record["subagent_trajectories"] = [dict(record, session_id="child")]
    run = ap.Run(record)
    with pytest.raises(ValueError, match=r"from_agentpile\(run\.subagents, \.\.\.\)") as err:
        au.from_agentpile(run, deps="full_context")
    assert 'subagents="skip"' in str(err.value)
    view = au.from_agentpile(run, deps="full_context", subagents="skip")
    assert view.meta["subagent_steps_skipped"] == len(run.subagents[0].steps)
    children = au.from_agentpile(run.subagents, deps="full_context")
    assert len(children) == 1 and len(children[0].steps) == len(view.steps)


def _with_revision(run, revision):
    record = run.to_dict()
    record["extra"]["agentpile"]["source"]["revision"] = revision
    return ap.Run(record)


def test_grade_refuses_an_untested_dataset_revision_and_names_the_generic_route():
    run = _tau_run(MESSAGES[1:])
    assert au.from_agentpile(run, deps="grade", convention="grade").meta["revision_tested"] is True
    other = _with_revision(run, "0" * 40)
    with pytest.raises(ValueError, match="unverified") as err:
        au.from_agentpile(other, deps="grade", convention="grade")
    message = str(err.value)
    assert "0" * 40 in message and run.source["revision"] in message
    assert 'convention="generic"' in message and 'deps="full_context"' in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        au.from_agentpile(other, deps="full_context")  # the generic route reads it


def test_a_missing_agentpile_names_the_extra(monkeypatch):
    import builtins

    from auditable.featurize import agentpile as adapter

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "agentpile" or name.startswith("agentpile."):
            raise ImportError("no agentpile")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match=r'pip install "auditable\[agentpile\]"'):
        adapter._agentpile()
