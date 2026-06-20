"""Tests for the ingestion adapters (auditable.graph.adapters).

Three parts, matching the task:

- **protocol**: the public ``Adapter`` extension point (name + version + to_steps),
  and that the two shipped singletons conform and are callable;
- **tau-bench corpus adapter**: a pure messages-in / steps-out mapping with each
  consequential write depending on every prior DB read, graded OBSERVED but marked
  ``modeled`` in evidence, and no ``huggingface_hub`` in the pure module;
- **own-record adapter**: DecisionRecords to steps, execution edges from the
  prev_digest backbone, model attributes on each node, identity carried, and sparse
  DECLARED (never fabricated-observed) dependency edges that read as low coverage.

All fixtures are small and inline; nothing touches the network.
"""
import sys

import pytest

from auditable.graph.adapters import (
    Adapter,
    OwnRecordAdapter,
    TauBenchPriorDBReadsAdapter,
    own_record_v1,
    tau_bench_prior_db_reads_v1,
)
from auditable.graph.session import Grade, SessionGraph, Step


# --- the Adapter protocol ----------------------------------------------------


def test_shipped_adapters_conform_to_protocol():
    for a in (tau_bench_prior_db_reads_v1, own_record_v1):
        assert isinstance(a.name, str) and a.name
        assert isinstance(a.version, str) and a.version
        assert callable(a.to_steps)
        # runtime_checkable structural conformance, no subclassing required
        assert isinstance(a, Adapter)


def test_adapter_id_is_name_and_version():
    assert tau_bench_prior_db_reads_v1.id == "tau_bench_prior_db_reads_v1"
    assert own_record_v1.id == "own_record_v1"
    assert tau_bench_prior_db_reads_v1.version == "v1"
    assert own_record_v1.name == "own_record"


def test_adapter_instance_is_callable_shorthand():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_order_details"}}]},
        {"role": "tool", "name": "get_order_details"},
    ]
    called = tau_bench_prior_db_reads_v1(msgs)        # __call__
    explicit = tau_bench_prior_db_reads_v1.to_steps(msgs)
    assert [s.idx for s in called] == [s.idx for s in explicit]
    assert all(isinstance(s, Step) for s in called)


# --- tau-bench corpus adapter ------------------------------------------------


def _tau_run():
    """A small tau-bench-style trajectory: read, then a non-DB utility, then a write."""
    return [
        {"role": "system", "content": "you are a retail agent; follow policy"},
        {"role": "user", "content": "cancel my reservation"},
        {"role": "assistant", "content": "let me check",
         "tool_calls": [{"function": {"name": "get_reservation_details"}}]},
        {"role": "tool", "name": "get_reservation_details", "content": "{...}"},
        {"role": "assistant", "content": "thinking", "tool_calls": [{"function": {"name": "think"}}]},
        {"role": "tool", "name": "think", "content": ""},
        {"role": "assistant", "content": "cancelling",
         "tool_calls": [{"function": {"name": "cancel_reservation"}}]},
        {"role": "tool", "name": "cancel_reservation", "content": "ok"},
    ]


def test_tau_bench_maps_roles_to_typed_steps():
    steps = tau_bench_prior_db_reads_v1.to_steps(_tau_run())
    # the system message is context, not a step; the other 7 messages each map
    assert [s.idx for s in steps] == [0, 1, 2, 3, 4, 5, 6]
    assert steps[0].agent == "user" and steps[0].kind == "decision"
    assert steps[1].agent == "assistant" and steps[1].kind == "decision"
    assert steps[2].kind == "tool_call" and steps[2].agent == "env"
    # tool steps are tagged with their tool name and DB role
    assert steps[2].node_attrs["tool"] == "get_reservation_details"
    assert steps[2].node_attrs["is_db_read"] is True and steps[2].node_attrs["is_write"] is False
    assert steps[4].node_attrs["tool"] == "think"          # a non-DB utility call
    assert steps[4].node_attrs["is_db_read"] is False and steps[4].node_attrs["is_write"] is False
    assert steps[6].node_attrs["tool"] == "cancel_reservation"
    assert steps[6].node_attrs["is_write"] is True


def test_tau_bench_write_depends_on_prior_reads_observed_but_modeled():
    steps = tau_bench_prior_db_reads_v1.to_steps(_tau_run())
    write = steps[6]
    # the write rests on the one prior DB read (idx 2), not the non-DB utility (idx 4)
    assert [e.src_idx for e in write.deps] == [2]
    e = write.deps[0]
    assert e.grade is Grade.OBSERVED          # the read/write events are observed
    assert e.resource is None                 # corpus adapters leave resource unset
    # the honesty seam: the write-to-prior-read edge set is MODELED, not causal
    assert e.evidence["modeled"] is True
    assert e.evidence["adapter"] == "tau_bench_prior_db_reads_v1"
    assert e.evidence["write_tool"] == "cancel_reservation"
    assert "not a causal label" in e.evidence["note"]


def test_tau_bench_write_depends_on_all_prior_db_reads():
    # two reads precede the write; the write depends on both (the conservative
    # prior-read upper bound over the observed reads)
    msgs = [
        {"role": "user", "content": "modify my order"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_order_details"}}]},
        {"role": "tool", "name": "get_order_details"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_user_details"}}]},
        {"role": "tool", "name": "get_user_details"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "modify_pending_order_items"}}]},
        {"role": "tool", "name": "modify_pending_order_items"},
    ]
    steps = tau_bench_prior_db_reads_v1.to_steps(msgs)
    write = steps[6]
    assert [e.src_idx for e in write.deps] == [2, 4]
    assert all(e.grade is Grade.OBSERVED and e.evidence["modeled"] is True for e in write.deps)


def test_tau_bench_write_with_no_prior_reads_has_no_deps():
    msgs = [
        {"role": "user", "content": "book it"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "book_reservation"}}]},
        {"role": "tool", "name": "book_reservation"},  # a write with nothing read before it
    ]
    steps = tau_bench_prior_db_reads_v1.to_steps(msgs)
    assert steps[2].node_attrs["is_write"] is True
    assert steps[2].deps == []


def test_tau_bench_tool_name_resolved_from_pending_when_omitted():
    # a tool result without its own name resolves from the assistant's pending call,
    # mirroring the experiment; the resolved name still drives the write classification
    msgs = [
        {"role": "user", "content": "book it"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "book_reservation"}}]},
        {"role": "tool", "content": "ok"},  # name omitted
    ]
    steps = tau_bench_prior_db_reads_v1.to_steps(msgs)
    assert steps[2].node_attrs["tool"] == "book_reservation"
    assert steps[2].node_attrs["is_write"] is True


def test_tau_bench_tool_call_name_fallback_without_function_key():
    # a tool_call may carry a bare "name" rather than a nested "function"
    msgs = [
        {"role": "user", "content": "look it up"},
        {"role": "assistant", "tool_calls": [{"name": "get_product_details"}]},
        {"role": "tool", "name": "get_product_details"},
    ]
    steps = tau_bench_prior_db_reads_v1.to_steps(msgs)
    assert steps[2].node_attrs["tool"] == "get_product_details"
    assert steps[2].node_attrs["is_db_read"] is True


def test_tau_bench_empty_messages_is_empty():
    assert tau_bench_prior_db_reads_v1.to_steps([]) == []
    assert tau_bench_prior_db_reads_v1.to_steps(None) == []


def test_tau_bench_optional_model_id_tags_assistant_nodes():
    tagged = TauBenchPriorDBReadsAdapter(assistant_agent="gpt-4.1", model_id="gpt-4.1")
    steps = tagged.to_steps(_tau_run())
    assert steps[1].agent == "gpt-4.1"
    assert steps[1].node_attrs["model_id"] == "gpt-4.1"
    # the default singleton does not invent a model id the trace does not carry
    assert "model_id" not in tau_bench_prior_db_reads_v1.to_steps(_tau_run())[1].node_attrs


def test_corpus_adapter_module_has_no_download_helper():
    # the pure adapter must not carry the huggingface_hub fetch path; that lives in
    # an examples-only helper or the optional 'corpora' extra
    import auditable.graph.adapters.tau_bench as tb

    assert not hasattr(tb, "hf_hub_download")
    assert not hasattr(tb, "_ensure_files")
    assert not hasattr(tb, "load_runs")
    assert "huggingface_hub" not in sys.modules  # importing it pulled in no network dep


# --- tau-bench steps drive the scorer (offline, no network) ------------------


def test_tau_bench_steps_score_with_observed_coverage():
    pytest.importorskip("networkx")
    from auditable.graph.risk import STATE_SCORED, structural_risk

    steps = tau_bench_prior_db_reads_v1.to_steps(_tau_run())
    g = SessionGraph.from_steps(steps)
    cov = g.coverage()
    assert cov.n_dep_edges == 1
    assert cov.observed_fraction == 1.0           # the one dependency edge is observed
    r = structural_risk(g)
    assert r.state == STATE_SCORED                # observed and unsaturated, so it scores
    # the read (idx 2) is the keystone the cancel write rests on
    assert r.per_decision[2] == max(r.per_decision.values()) > 0


# --- own-record adapter ------------------------------------------------------


def _own_records(n=3):
    """Capture n chained, signed DecisionRecords via a MemorySink (offline)."""
    from auditable import Action, DependencySnapshot, MemorySink, audit

    sink = MemorySink()
    for i in range(n):
        with audit(f"pay-{i}", snapshot=DependencySnapshot(state={"budget": 100}), sink=sink) as d:
            d.read(invoice=f"INV-{100 + i}")
            d.model("gpt-x", decision_basis=f"invoice INV-{100 + i} matches an approved PO")
            d.act(Action(f"pay-{i}", {"amount": 100 + i}, cost=1.0))
    return sink.records


def test_own_record_one_step_per_record_with_model_attrs():
    records = _own_records(3)
    steps = own_record_v1.to_steps(records)
    assert len(steps) == 3
    assert all(s.kind == "decision" for s in steps)
    # the model attributes ride on each node
    assert steps[1].node_attrs["model_id"] == "gpt-x"
    assert steps[1].node_attrs["decision_basis"] == "invoice INV-101 matches an approved PO"
    assert steps[1].node_attrs["action_type"] == "pay-1"


def test_own_record_exec_preds_follow_prev_digest_backbone():
    records = _own_records(3)
    steps = own_record_v1.to_steps(records)
    assert steps[0].exec_preds == []      # genesis: no execution predecessor
    assert steps[1].exec_preds == [0]     # linked by prev_digest, not by position alone
    assert steps[2].exec_preds == [1]


def test_own_record_carries_identity():
    records = _own_records(2)
    steps = own_record_v1.to_steps(records)
    for s, rec in zip(steps, records):
        assert s.record_id == rec.record_id          # the sealed digest
        assert s.correlation_id == rec.record_id     # offline, the digest is the correlation key


def test_own_record_dependency_edges_are_sparse_declared_never_observed():
    records = _own_records(3)
    steps = own_record_v1.to_steps(records)
    assert steps[0].deps == []                        # genesis declares nothing
    assert [e.src_idx for e in steps[1].deps] == [0]
    assert [e.src_idx for e in steps[2].deps] == [1]
    for s in steps:
        for e in s.deps:
            assert e.grade is Grade.DECLARED          # never OBSERVED without the touch contract
            assert e.resource is None
            assert e.evidence["observed"] is False
            assert e.evidence["declared"] is True
            assert e.evidence["adapter"] == "own_record_v1"


def test_own_record_reports_low_coverage_not_scored():
    pytest.importorskip("networkx")
    from auditable.graph.risk import STATE_LOW_COVERAGE, structural_risk

    records = _own_records(3)
    g = SessionGraph.from_steps(own_record_v1.to_steps(records))
    cov = g.coverage()
    assert cov.by_grade[Grade.DECLARED] == 2
    assert cov.by_grade[Grade.OBSERVED] == 0
    assert cov.observed_fraction == 0.0
    # the honesty contract: own records this round are not paper-validated structure
    assert structural_risk(g).state == STATE_LOW_COVERAGE


def test_own_record_link_sequential_off_yields_empty_dependency_layer():
    pytest.importorskip("networkx")
    from auditable.graph.risk import STATE_LOW_COVERAGE, structural_risk

    records = _own_records(3)
    adapter = OwnRecordAdapter(link_sequential=False)
    steps = adapter.to_steps(records)
    assert all(s.deps == [] for s in steps)
    g = SessionGraph.from_steps(steps)
    assert g.coverage().n_dep_edges == 0
    # the execution backbone is still wired from prev_digest
    assert steps[1].exec_preds == [0] and steps[2].exec_preds == [1]
    assert structural_risk(g).state == STATE_LOW_COVERAGE


def test_own_record_exec_backbone_projects_to_handoffs():
    pytest.importorskip("networkx")
    records = _own_records(3)
    G = SessionGraph.from_steps(own_record_v1.to_steps(records)).to_networkx()
    handoffs = {(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "handoff_to"}
    assert handoffs == {("step::0", "step::1"), ("step::1", "step::2")}


def test_own_record_duck_types_stub_records():
    # a lightweight stub with the same shape works (no heavy DecisionRecord import)
    from types import SimpleNamespace as NS

    recs = [
        NS(record_id="a", prev_digest=None, action_type="x",
           model=NS(model_id="m", decision_basis="b0")),
        NS(record_id="b", prev_digest="a", action_type="y",
           model=NS(model_id="m", decision_basis="b1")),
    ]
    steps = own_record_v1.to_steps(recs)
    assert steps[0].exec_preds == [] and steps[1].exec_preds == [0]
    assert steps[1].node_attrs["model_id"] == "m"
    assert steps[1].deps[0].src_idx == 0 and steps[1].deps[0].grade is Grade.DECLARED
    assert steps[1].record_id == "b" and steps[1].correlation_id == "b"


def test_own_record_empty_input_is_empty():
    assert own_record_v1.to_steps([]) == []
    assert own_record_v1.to_steps(None) == []


def test_own_record_unsigned_records_are_roots_with_no_identity():
    # records without a sealed digest (assembled but not yet signed) carry
    # prev_digest None, so each is a genesis root and identity stays None
    from auditable.record import DecisionRecord

    recs = [DecisionRecord(action_type=f"a-{i}") for i in range(3)]  # record_id "", prev None
    steps = own_record_v1.to_steps(recs)
    assert all(s.exec_preds == [] for s in steps)            # every prev_digest is None
    assert all(s.record_id is None and s.correlation_id is None for s in steps)
    assert all(s.deps == [] for s in steps)                  # a genesis declares nothing


def test_own_record_unresolved_prev_digest_falls_back_to_prior_position():
    # a non-genesis record whose prev_digest names an id absent from this batch
    # (an out-of-order or partial slice): keep the backbone connected to the prior
    # position rather than inventing a non-adjacent link or dropping the edge
    from types import SimpleNamespace as NS

    recs = [
        NS(record_id="", prev_digest=None, action_type="x",
           model=NS(model_id="m", decision_basis="b0")),     # unsigned genesis, not mapped
        NS(record_id="b", prev_digest="GHOST", action_type="y",
           model=NS(model_id="m", decision_basis="b1")),     # prev set but unresolved, i=1
    ]
    steps = own_record_v1.to_steps(recs)
    assert steps[0].exec_preds == []        # i == 0, nothing prior to fall back to
    assert steps[1].exec_preds == [0]       # fall back to the immediate prior position
    assert steps[1].deps[0].src_idx == 0 and steps[1].deps[0].grade is Grade.DECLARED
