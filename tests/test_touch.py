"""Tests for the runtime touch matcher and TouchRecorder (auditable.graph.touch).

The matcher turns ordered per-step resource touches into OBSERVED dependency
edges. It is superstep-aware (same-superstep writes are invisible, matching
LangGraph's BSP barrier) and reducer-aware (a reducer channel fans in from every
accumulated writer; an overwrite channel binds the last writer only).
"""
import pytest

from auditable.graph.session import DependencyEdge, Grade, ResourceRef, SessionGraph, Step
from auditable.graph.touch import StepTouch, TouchRecorder, match_observed_deps, touches_to_steps


def _ref(name, ns="langgraph_state", key=""):
    return ResourceRef(ns, name, key)


def test_read_after_write_in_earlier_superstep_is_one_observed_edge():
    touches = [
        StepTouch(idx=0, superstep=0, agent="setup", kind="decision", writes=[_ref("budget")]),
        StepTouch(idx=1, superstep=1, agent="pay", kind="decision", reads=[_ref("budget")]),
    ]
    deps = match_observed_deps(touches)
    assert deps[0] == []                        # the writer depends on nothing
    assert [e.src_idx for e in deps[1]] == [0]  # the reader depends on the writer
    edge = deps[1][0]
    assert edge.grade is Grade.OBSERVED
    assert edge.resource == _ref("budget")
    assert edge.evidence["relation"] == "read_after_committed_write"
    assert edge.evidence["granularity"] == "channel"
    assert edge.evidence["mode"] == "overwrite"


def test_same_superstep_write_is_invisible_to_a_parallel_reader():
    # LangGraph commits a superstep's writes at the barrier, so two nodes in the
    # same superstep never see each other's writes; a later superstep does.
    touches = [
        StepTouch(idx=0, superstep=0, agent="root", kind="decision", writes=[_ref("seed")]),
        StepTouch(idx=1, superstep=1, agent="branch_a", kind="decision",
                  reads=[_ref("seed")], writes=[_ref("shared")]),
        StepTouch(idx=2, superstep=1, agent="branch_b", kind="decision", reads=[_ref("shared")]),
        StepTouch(idx=3, superstep=2, agent="merge", kind="decision", reads=[_ref("shared")]),
    ]
    deps = match_observed_deps(touches)
    assert [e.src_idx for e in deps[2]] == []   # parallel sibling write is invisible
    assert [e.src_idx for e in deps[3]] == [1]  # committed at the barrier, visible next superstep
    assert [e.src_idx for e in deps[1]] == [0]  # branch_a still sees root's earlier-superstep write


def test_reducer_channel_read_fans_in_to_all_accumulated_writers():
    # a reducer channel (Annotated[..., add_messages]) accumulates writes across
    # supersteps, so a later read depends on every prior writer, marked modeled.
    msgs = _ref("messages")
    touches = [
        StepTouch(idx=0, superstep=0, agent="a", kind="decision", writes=[msgs]),
        StepTouch(idx=1, superstep=1, agent="b", kind="decision", writes=[msgs]),
        StepTouch(idx=2, superstep=2, agent="reader", kind="decision", reads=[msgs]),
    ]
    deps = match_observed_deps(touches, reducer_channels={("langgraph_state", "messages", "")})
    assert sorted(e.src_idx for e in deps[2]) == [0, 1]
    for e in deps[2]:
        assert e.evidence["mode"] == "reducer"
        assert e.evidence["modeled"] == "reducer_writer_set"


def test_overwrite_channel_read_binds_only_the_last_writer():
    # an overwrite channel keeps only the most recent committed writer
    ch = _ref("budget")
    touches = [
        StepTouch(idx=0, superstep=0, agent="a", kind="decision", writes=[ch]),
        StepTouch(idx=1, superstep=1, agent="b", kind="decision", writes=[ch]),
        StepTouch(idx=2, superstep=2, agent="reader", kind="decision", reads=[ch]),
    ]
    deps = match_observed_deps(touches)  # no reducer channels -> overwrite
    assert [e.src_idx for e in deps[2]] == [1]
    assert "modeled" not in deps[2][0].evidence


def test_read_modify_write_depends_on_prior_writer_and_feeds_later_readers():
    ch = _ref("counter")
    touches = [
        StepTouch(idx=0, superstep=0, agent="init", kind="decision", writes=[ch]),
        StepTouch(idx=1, superstep=1, agent="rmw", kind="decision", reads=[ch], writes=[ch]),
        StepTouch(idx=2, superstep=2, agent="reader", kind="decision", reads=[ch]),
    ]
    deps = match_observed_deps(touches)
    assert [e.src_idx for e in deps[1]] == [0]  # the read sees the prior writer
    assert [e.src_idx for e in deps[2]] == [1]  # the modify becomes the writer for later readers


def test_read_with_no_prior_writer_makes_no_edge():
    # a channel set only by the initial input has no prior step to bind to
    touches = [
        StepTouch(idx=0, superstep=0, agent="reader", kind="decision", reads=[_ref("user_input")]),
    ]
    deps = match_observed_deps(touches)
    assert deps[0] == []


def test_overcaptured_read_flags_the_edge():
    # a read taken from a whole-state access (**state / keys()) is flagged honestly
    ch = _ref("state_blob")
    touches = [
        StepTouch(idx=0, superstep=0, agent="w", kind="decision", writes=[ch]),
        StepTouch(idx=1, superstep=1, agent="r", kind="decision", reads=[ch],
                  overcaptured=frozenset({ch})),
    ]
    deps = match_observed_deps(touches)
    assert deps[1][0].evidence["overcaptured"] is True


# --- the generic TouchRecorder (manual runtime capture, any framework) -------


def test_touch_recorder_lowers_read_after_write_to_an_observed_step_edge():
    rec = TouchRecorder()
    with rec.step(agent="fetch", kind="tool_call", node_attrs={"tool": "get_order"}) as st:
        st.writes("db", "orders", "row_7")
    with rec.step(agent="pay", kind="decision") as st:
        st.reads("db", "orders", "row_7")
    steps = rec.to_steps()
    assert [s.idx for s in steps] == [0, 1]
    assert steps[0].agent == "fetch" and steps[0].kind == "tool_call"
    assert steps[0].node_attrs["tool"] == "get_order"
    assert [e.src_idx for e in steps[1].deps] == [0]
    assert steps[1].deps[0].grade is Grade.OBSERVED
    assert steps[1].deps[0].resource == ResourceRef("db", "orders", "row_7")


def test_touch_recorder_conforms_to_adapter_and_ignores_source_arg():
    from auditable.graph.adapters import Adapter

    rec = TouchRecorder()
    with rec.step(agent="a", kind="decision") as st:
        st.writes("kv", "x")
    with rec.step(agent="b", kind="decision") as st:
        st.reads("kv", "x")
    assert isinstance(rec, Adapter)            # name + version + to_steps
    assert rec.id == "touch_recorder_v1"
    # to_steps(source) ignores the source (the recorder IS the source)
    by_arg = rec.to_steps(rec)
    by_none = rec.to_steps()
    assert [(s.idx, [e.src_idx for e in s.deps]) for s in by_arg] == \
           [(s.idx, [e.src_idx for e in s.deps]) for s in by_none]


def test_touch_recorder_drives_analyze_run():
    pytest.importorskip("networkx")
    from auditable import analyze_run

    rec = TouchRecorder()
    with rec.step(agent="env", kind="tool_call") as st:      # the keystone read source
        st.writes("db", "reservation", "5")
    with rec.step(agent="agent", kind="decision") as st:
        st.reads("db", "reservation", "5")
        st.writes("db", "reservation", "5")
    with rec.step(agent="agent", kind="decision") as st:
        st.reads("db", "reservation", "5")
    report = analyze_run(rec, adapter=rec, ground=False)
    assert len(report.ranked) == 3


def test_core_import_does_not_pull_langgraph():
    # the framework-agnostic core (and the touch spine) must not import langgraph;
    # the LangGraph adapter is the only place that does, behind the optional extra
    import subprocess
    import sys

    code = (
        "import sys, auditable, auditable.graph.touch\n"
        "pulled = sorted(m for m in sys.modules if m == 'langgraph' or m.startswith('langgraph.'))\n"
        "assert not pulled, pulled\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_touches_to_steps_orders_output_so_projection_keeps_edges():
    # the matcher sorts internally, but to_steps must emit steps in (superstep, idx)
    # order too: SessionGraph.to_networkx only wires a dep to an already-seen step, so
    # an out-of-order reader-before-writer would silently drop the matched edge.
    pytest.importorskip("networkx")
    x = ResourceRef("kv", "x", "")
    touches = [
        StepTouch(idx=1, superstep=1, agent="reader", kind="decision", reads=[x]),
        StepTouch(idx=0, superstep=0, agent="writer", kind="decision", writes=[x]),
    ]
    steps = touches_to_steps(touches)
    assert [s.idx for s in steps] == [0, 1]  # causal order, writer before reader
    graph = SessionGraph.from_steps(steps).to_networkx()
    assert graph.has_edge("step::1", "step::0")  # the depends_on edge survived projection


def test_duplicate_reads_of_one_resource_make_one_edge():
    # a dependency is an edge between two steps over a resource, not one per access;
    # repeated reads of the same resource must not inflate the graph
    rec = TouchRecorder()
    with rec.step(agent="w", kind="decision") as st:
        st.writes("kv", "x")
    with rec.step(agent="r", kind="decision") as st:
        st.reads("kv", "x")
        st.reads("kv", "x")  # duplicate read of the same resource
    steps = rec.to_steps()
    x_edges = [e for e in steps[1].deps if e.resource.resource_id == "x"]
    assert len(x_edges) == 1
