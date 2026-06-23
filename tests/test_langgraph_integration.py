"""Integration tests for the LangGraph capture adapter (auditable.integrations.langgraph).

These build small real ``StateGraph``s of pure-function nodes (no LLM, no network,
no key) and assert that instrumenting them yields OBSERVED dependency edges over
the state channels, correctly attributed across the superstep barrier. They skip
when ``langgraph`` is not installed.
"""
import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph

from auditable.graph.session import Grade
from auditable.integrations.langgraph import instrument


class _State(TypedDict):
    budget: int
    log: Annotated[list, operator.add]  # reducer channel


def test_instrument_captures_observed_read_after_write_over_channels():
    def fund(state):
        return {"budget": 100, "log": ["fund"]}

    def spend(state):
        return {"budget": state["budget"] - 30, "log": ["spend"]}

    def audit_node(state):
        _ = state["budget"]
        return {"log": ["audit"]}

    builder = instrument(StateGraph(_State))
    builder.add_node("fund", fund)
    builder.add_node("spend", spend)
    builder.add_node("audit", audit_node)
    builder.add_edge(START, "fund")
    builder.add_edge("fund", "spend")
    builder.add_edge("spend", "audit")
    builder.add_edge("audit", END)
    graph = builder.compile()
    graph.invoke({"budget": 0, "log": []})

    steps = builder.to_steps()
    by_agent = {s.agent: s for s in steps}
    assert set(by_agent) == {"fund", "spend", "audit"}

    # spend read budget, which fund wrote -> one OBSERVED edge spend -> fund
    spend_budget_edges = [
        e for e in by_agent["spend"].deps
        if e.resource.resource_id == "budget"
    ]
    assert [e.src_idx for e in spend_budget_edges] == [by_agent["fund"].idx]
    e = spend_budget_edges[0]
    assert e.grade is Grade.OBSERVED
    assert e.resource.namespace == "langgraph_state"
    assert e.evidence["mode"] == "overwrite"

    # audit read budget, which spend last wrote -> OBSERVED edge audit -> spend
    audit_budget_edges = [
        e for e in by_agent["audit"].deps
        if e.resource.resource_id == "budget"
    ]
    assert [e.src_idx for e in audit_budget_edges] == [by_agent["spend"].idx]


def test_parallel_branch_does_not_fabricate_same_superstep_edges():
    # fund -> {branch_a, branch_b} in one superstep -> merge. The branches both read
    # budget (committed by fund earlier) but must NOT depend on each other.
    class S(TypedDict):
        budget: int
        a_out: int
        b_out: int

    def fund(state):
        return {"budget": 100}

    def branch_a(state):
        return {"a_out": state["budget"] + 1}

    def branch_b(state):
        return {"b_out": state["budget"] + 2}

    def merge(state):
        return {"budget": state["a_out"] + state["b_out"]}

    builder = instrument(StateGraph(S))
    for nm, fn in [("fund", fund), ("branch_a", branch_a), ("branch_b", branch_b), ("merge", merge)]:
        builder.add_node(nm, fn)
    builder.add_edge(START, "fund")
    builder.add_edge("fund", "branch_a")
    builder.add_edge("fund", "branch_b")
    builder.add_edge("branch_a", "merge")
    builder.add_edge("branch_b", "merge")
    builder.add_edge("merge", END)
    builder.compile().invoke({"budget": 0, "a_out": 0, "b_out": 0})

    steps = {s.agent: s for s in builder.to_steps()}
    # both branches read budget written by fund (an earlier superstep)
    assert [e.src_idx for e in steps["branch_a"].deps if e.resource.resource_id == "budget"] == [steps["fund"].idx]
    assert [e.src_idx for e in steps["branch_b"].deps if e.resource.resource_id == "budget"] == [steps["fund"].idx]
    # the parallel siblings do NOT depend on each other (same superstep)
    assert steps["branch_b"].idx not in {e.src_idx for e in steps["branch_a"].deps}
    assert steps["branch_a"].idx not in {e.src_idx for e in steps["branch_b"].deps}
    # merge reads a_out and b_out, committed by the two branches at the barrier
    merge_edges = {(e.src_idx, e.resource.resource_id) for e in steps["merge"].deps}
    assert (steps["branch_a"].idx, "a_out") in merge_edges
    assert (steps["branch_b"].idx, "b_out") in merge_edges


def test_reducer_channel_fans_in_on_a_real_graph():
    class S(TypedDict):
        log: Annotated[list, operator.add]
        done: bool

    def a(state):
        return {"log": ["a"]}

    def b(state):
        return {"log": ["b"]}

    def reader(state):
        _ = state["log"]
        return {"done": True}

    builder = instrument(StateGraph(S))
    for nm, fn in [("a", a), ("b", b), ("reader", reader)]:
        builder.add_node(nm, fn)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "reader")
    builder.add_edge("reader", END)
    builder.compile().invoke({"log": [], "done": False})

    steps = {s.agent: s for s in builder.to_steps()}
    log_edges = [e for e in steps["reader"].deps if e.resource.resource_id == "log"]
    assert sorted(e.src_idx for e in log_edges) == sorted([steps["a"].idx, steps["b"].idx])
    for e in log_edges:
        assert e.evidence["mode"] == "reducer"
        assert e.evidence["modeled"] == "reducer_writer_set"


def test_async_node_is_captured():
    import asyncio

    class S(TypedDict):
        x: int
        y: int

    async def producer(state):
        return {"x": 5}

    async def consumer(state):
        return {"y": state["x"] + 1}

    builder = instrument(StateGraph(S))
    builder.add_node("producer", producer)
    builder.add_node("consumer", consumer)
    builder.add_edge(START, "producer")
    builder.add_edge("producer", "consumer")
    builder.add_edge("consumer", END)
    asyncio.run(builder.compile().ainvoke({"x": 0, "y": 0}))

    steps = {s.agent: s for s in builder.to_steps()}
    assert [e.src_idx for e in steps["consumer"].deps if e.resource.resource_id == "x"] == [steps["producer"].idx]


def test_instrumented_graph_drives_analyze_run():
    from auditable import analyze_run

    class S(TypedDict):
        v: int

    def seed(state):
        return {"v": 1}

    def use(state):
        return {"v": state["v"] + 1}

    def use_again(state):
        return {"v": state["v"] + 1}

    builder = instrument(StateGraph(S))
    builder.add_node("seed", seed)
    builder.add_node("use", use)
    builder.add_node("use_again", use_again)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", "use_again")
    builder.add_edge("use_again", END)
    builder.compile().invoke({"v": 0})

    report = analyze_run(builder, adapter=builder, ground=False)
    assert len(report.ranked) == 3


# --- regression tests from the execution review -------------------------------


class _VState(TypedDict):
    v: int


def _seed(state):
    return {"v": 1}


def _use(state):
    return {"v": state["v"] + 1}


def _build_seed_use():
    builder = instrument(StateGraph(_VState))
    builder.add_node("seed", _seed)
    builder.add_node("use", _use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    return builder


def test_reusing_compiled_graph_captures_only_the_latest_run():
    # H1: a compiled graph is reusable; langgraph_step restarts each invoke, so a
    # cumulative recorder would merge runs and fabricate cross-run edges.
    builder = _build_seed_use()
    graph = builder.compile()
    graph.invoke({"v": 0})
    graph.invoke({"v": 0})
    steps = builder.to_steps()
    assert len(steps) == 2  # the latest run only, not 4 accumulated
    by_agent = {s.agent: s for s in steps}
    assert [e.src_idx for e in by_agent["use"].deps if e.resource.resource_id == "v"] == [by_agent["seed"].idx]


def test_add_node_keyword_action_is_captured():
    # H2: add_node("name", action=fn) passes the action as a keyword
    builder = instrument(StateGraph(_VState))
    builder.add_node("seed", action=_seed)
    builder.add_node("use", action=_use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    builder.compile().invoke({"v": 0})
    by_agent = {s.agent: s for s in builder.to_steps()}
    assert set(by_agent) == {"seed", "use"}
    assert [e.src_idx for e in by_agent["use"].deps if e.resource.resource_id == "v"] == [by_agent["seed"].idx]


def test_doc_annotated_channel_is_overwrite_not_reducer():
    # H3: Annotated[int, "doc"] carries metadata but is NOT a reducer; a read of it
    # must bind only the last writer, not fan in from every writer.
    class S(TypedDict):
        note: Annotated[int, "just documentation, not a reducer"]

    def a(state):
        return {"note": 1}

    def b(state):
        return {"note": 2}

    def reader(state):
        _ = state["note"]
        return {}

    builder = instrument(StateGraph(S))
    for nm, fn in [("a", a), ("b", b), ("reader", reader)]:
        builder.add_node(nm, fn)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "reader")
    builder.add_edge("reader", END)
    builder.compile().invoke({"note": 0})
    by_agent = {s.agent: s for s in builder.to_steps()}
    note_srcs = sorted(e.src_idx for e in by_agent["reader"].deps if e.resource.resource_id == "note")
    assert note_srcs == [by_agent["b"].idx]  # overwrite: last writer only


def test_keyword_only_config_node_is_captured():
    # M2: a node with keyword-only config must be called fn(proxy, config=config)
    def seed(state):
        return {"v": 1}

    def use(state, *, config):
        return {"v": state["v"] + 1}

    builder = instrument(StateGraph(_VState))
    builder.add_node("seed", seed)
    builder.add_node("use", use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    builder.compile().invoke({"v": 0})
    by_agent = {s.agent: s for s in builder.to_steps()}
    assert [e.src_idx for e in by_agent["use"].deps if e.resource.resource_id == "v"] == [by_agent["seed"].idx]


def test_dataclass_state_schema_fails_fast():
    # M2: v1 supports mapping/TypedDict state only; a non-mapping schema must raise a
    # clear error at instrument time, not a confusing AttributeError mid-run.
    from dataclasses import dataclass

    @dataclass
    class DState:
        v: int = 0

    with pytest.raises(TypeError, match="(?i)state"):
        instrument(StateGraph(DState))


def test_runnable_node_passthrough_warns_about_incomplete_capture():
    # M3: an uncaptured (Runnable) node makes the dependency graph incomplete; to_steps
    # must warn rather than silently return a partial capture that looks complete.
    from langchain_core.runnables import RunnableLambda

    builder = instrument(StateGraph(_VState))
    builder.add_node("seed", RunnableLambda(lambda state: {"v": 1}))
    builder.add_node("use", _use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    builder.compile().invoke({"v": 0})
    with pytest.warns(UserWarning, match="(?i)not captured"):
        builder.to_steps()


# --- regression tests from the round-2 execution review -----------------------


def test_fluent_add_node_chaining_is_captured():
    # add_node returns Self for chaining; the proxy must keep instrumenting later nodes
    builder = instrument(StateGraph(_VState))
    builder.add_node("seed", _seed).add_node("use", _use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    builder.compile().invoke({"v": 0})
    by_agent = {s.agent: s for s in builder.to_steps()}
    assert set(by_agent) == {"seed", "use"}
    assert [e.src_idx for e in by_agent["use"].deps if e.resource.resource_id == "v"] == [by_agent["seed"].idx]


def test_command_list_update_writes_are_captured():
    # Command(update=[(channel, value)]) is a valid public write form
    from langgraph.types import Command

    def seed(state):
        return Command(update=[("v", 1)])

    builder = instrument(StateGraph(_VState))
    builder.add_node("seed", seed)
    builder.add_node("use", _use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    builder.compile().invoke({"v": 0})
    by_agent = {s.agent: s for s in builder.to_steps()}
    assert [e.src_idx for e in by_agent["use"].deps if e.resource.resource_id == "v"] == [by_agent["seed"].idx]


def test_runtime_context_node_is_not_broken_and_is_captured():
    # a node that wants `runtime` (not config) must still run and be captured
    def seed(state):
        return {"v": 1}

    def use(state, runtime):
        _ = runtime.context  # a real Runtime has .context; a config dict does not
        return {"v": state["v"] + 1}

    builder = instrument(StateGraph(_VState))
    builder.add_node("seed", seed)
    builder.add_node("use", use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    builder.compile().invoke({"v": 0})  # must not raise AttributeError
    by_agent = {s.agent: s for s in builder.to_steps()}
    assert [e.src_idx for e in by_agent["use"].deps if e.resource.resource_id == "v"] == [by_agent["seed"].idx]


def test_interleaved_streams_do_not_mix_captures():
    # two stream generators created before either is drained must not mix into one run
    builder = _build_seed_use()
    graph = builder.compile()
    s1 = graph.stream({"v": 0})
    s2 = graph.stream({"v": 10})
    list(s1)
    list(s2)
    steps = builder.to_steps()
    assert len(steps) == 2  # the latest run only, not 4 accumulated across both streams


def test_doc_then_reducer_annotated_is_overwrite():
    # Annotated[int, add, "doc"]: LangGraph keys on the LAST metadata item, so this is
    # overwrite (LastValue), not a reducer.
    class S(TypedDict):
        ch: Annotated[int, operator.add, "doc"]

    def a(state):
        return {"ch": 1}

    def b(state):
        return {"ch": 2}

    def reader(state):
        _ = state["ch"]
        return {}

    builder = instrument(StateGraph(S))
    for nm, fn in [("a", a), ("b", b), ("reader", reader)]:
        builder.add_node(nm, fn)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "reader")
    builder.add_edge("reader", END)
    builder.compile().invoke({"ch": 0})
    by_agent = {s.agent: s for s in builder.to_steps()}
    ch_srcs = sorted(e.src_idx for e in by_agent["reader"].deps if e.resource.resource_id == "ch")
    assert ch_srcs == [by_agent["b"].idx]  # overwrite: last writer only


def test_notrequired_reducer_annotated_fans_in():
    # NotRequired[Annotated[list, add]]: LangGraph unwraps NotRequired, so it IS a reducer
    try:
        from typing import NotRequired
    except ImportError:
        from typing_extensions import NotRequired

    class S(TypedDict):
        log: NotRequired[Annotated[list, operator.add]]

    def a(state):
        return {"log": ["a"]}

    def b(state):
        return {"log": ["b"]}

    def reader(state):
        _ = state["log"]
        return {}

    builder = instrument(StateGraph(S))
    for nm, fn in [("a", a), ("b", b), ("reader", reader)]:
        builder.add_node(nm, fn)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "reader")
    builder.add_edge("reader", END)
    builder.compile().invoke({"log": []})
    by_agent = {s.agent: s for s in builder.to_steps()}
    log_srcs = sorted(e.src_idx for e in by_agent["reader"].deps if e.resource.resource_id == "log")
    assert log_srcs == sorted([by_agent["a"].idx, by_agent["b"].idx])  # reducer fan-in


# --- regression tests from the round-3 execution review -----------------------


def test_annotation_inferred_return_schema_is_preserved():
    # LangGraph infers extra channels from a node's return annotation; the wrapper
    # must keep the node's type hints so those channels register and writes survive.
    class Overall(TypedDict):
        public: int

    class Private(TypedDict):
        public: int
        secret: int

    def seed(state: Overall) -> Private:
        return {"public": 1, "secret": 7}

    def use(state: Private):
        return {"public": state["secret"] + 1}

    builder = instrument(StateGraph(Overall))
    builder.add_node("seed", seed)
    builder.add_node("use", use)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "use")
    builder.add_edge("use", END)
    out = builder.compile().invoke({"public": 0})
    assert out["public"] == 8  # no KeyError: the 'secret' channel was registered
    by_agent = {s.agent: s for s in builder.to_steps()}
    assert [e.src_idx for e in by_agent["use"].deps if e.resource.resource_id == "secret"] == [by_agent["seed"].idx]


def test_duplicate_reducer_command_writes_make_one_edge():
    # Command(update=[("log", ...), ("log", ...)]) writes the same channel twice;
    # the matcher must not emit two identical OBSERVED edges.
    from langgraph.types import Command

    class S(TypedDict):
        log: Annotated[list, operator.add]
        done: bool

    def seed(state):
        return Command(update=[("log", ["a"]), ("log", ["b"])])

    def reader(state):
        _ = state["log"]
        return {"done": True}

    builder = instrument(StateGraph(S))
    builder.add_node("seed", seed)
    builder.add_node("reader", reader)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "reader")
    builder.add_edge("reader", END)
    builder.compile().invoke({"log": [], "done": False})
    by_agent = {s.agent: s for s in builder.to_steps()}
    log_edges = [e for e in by_agent["reader"].deps if e.resource.resource_id == "log"]
    assert len(log_edges) == 1  # one edge to seed, not two


def test_two_compiled_proxies_share_one_active_run_guard():
    # the active-run guard must be shared (on the recorder), so two compiled graphs
    # from one instrumented builder cannot run concurrently and mix captures.
    builder = _build_seed_use()
    g1 = builder.compile()
    g2 = builder.compile()
    s1 = g1.stream({"v": 0})
    next(s1)  # begin run on g1, leave it active (not drained)
    with pytest.raises(RuntimeError):
        g2.invoke({"v": 10})  # a run on g2 while g1 is active must raise


def test_input_schema_reducer_channel_is_detected():
    # add_node(..., input_schema=...) adds channels after instrument(); reducer
    # detection must run at to_steps (post-construction), not be snapshotted early.
    class Overall(TypedDict):
        done: bool

    class WithLog(TypedDict):
        log: Annotated[list, operator.add]
        done: bool

    def a(state):
        return {"log": ["a"]}

    def b(state):
        return {"log": ["b"]}

    def reader(state):
        _ = state.get("log")
        return {"done": True}

    builder = instrument(StateGraph(Overall))
    builder.add_node("a", a, input_schema=WithLog)
    builder.add_node("b", b, input_schema=WithLog)
    builder.add_node("reader", reader, input_schema=WithLog)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "reader")
    builder.add_edge("reader", END)
    builder.compile().invoke({"done": False})
    by_agent = {s.agent: s for s in builder.to_steps()}
    log_srcs = sorted(e.src_idx for e in by_agent["reader"].deps if e.resource.resource_id == "log")
    assert log_srcs == sorted([by_agent["a"].idx, by_agent["b"].idx])  # reducer fan-in


def test_batch_fails_closed_rather_than_mixing_captures():
    # batch would run the wrapped nodes for several inputs through one recorder,
    # mixing captures across items; v1 fails closed instead of capturing wrongly.
    builder = _build_seed_use()
    graph = builder.compile()
    with pytest.raises(RuntimeError):
        graph.batch([{"v": 0}, {"v": 10}])


def test_event_and_log_stream_methods_fail_closed():
    # astream_events / astream_log / stream_events also run wrapped nodes without
    # begin_run; they must fail closed rather than mix captures.
    builder = _build_seed_use()
    graph = builder.compile()
    with pytest.raises(RuntimeError):
        graph.astream_events({"v": 0}, version="v2")
    with pytest.raises(RuntimeError):
        graph.astream_log({"v": 0})


def test_record_only_captures_inside_an_active_run():
    # backstop: if any unguarded surface ran wrapped nodes without begin_run, the
    # recorder must not capture (so a missed surface fails safe, not fabricating).
    from auditable.integrations.langgraph import _Recorder

    rec = _Recorder()
    rec.record("n", 0, {"x"}, set(), {"x": 1})  # no begin_run first
    assert rec.to_touches() == []
