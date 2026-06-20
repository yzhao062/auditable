"""Tests for the public ``analyze_run`` entry and the ``AnalysisReport`` it returns.

Covers the three things the report must show: a ranked structural-risk list naming
the keystone, dependency coverage with rho, and model-basis grounding (where a basis
is stated) -- plus the honest no-score states. All fixtures are small and inline; the
scorer needs the NetworkX projection but nothing touches the network.
"""
import math

import pytest

pytest.importorskip("networkx")  # analyze_run scores over the NetworkX projection

from auditable import AnalysisReport, analyze_run
from auditable.analysis import DecisionRisk
from auditable.graph.adapters import own_record_v1, tau_bench_prior_db_reads_v1
from auditable.graph.risk import (
    STATE_LOW_COVERAGE,
    STATE_SCORED,
    STATE_SINGLE_DECISION,
)
from auditable.graph.session import GraphCompleteness


# --- corpus path: tau-bench, observed-but-modeled edges, scored keystone --------


def _tau_messages():
    """A retail run: read the order, modify it, read the user, send a refund cert.

    Two consequential writes (modify, send) both rest on the first read
    (get_order_details), so that read is the unique structural keystone.
    """
    return [
        {"role": "system", "content": "you are a retail agent; follow policy"},
        {"role": "user", "content": "Modify pending order #W512 and refund the difference."},
        {"role": "assistant", "content": "Pulling up the order.",
         "tool_calls": [{"function": {"name": "get_order_details"}}]},
        {"role": "tool", "name": "get_order_details",
         "content": '{"order_id": "#W512", "status": "pending"}'},
        {"role": "assistant", "content": "Updating the item.",
         "tool_calls": [{"function": {"name": "modify_pending_order_items"}}]},
        {"role": "tool", "name": "modify_pending_order_items", "content": "ok"},
        {"role": "assistant", "content": "Checking your account.",
         "tool_calls": [{"function": {"name": "get_user_details"}}]},
        {"role": "tool", "name": "get_user_details",
         "content": '{"user_id": "sara_doe_496"}'},
        {"role": "assistant", "content": "Issuing the refund certificate.",
         "tool_calls": [{"function": {"name": "send_certificate"}}]},
        {"role": "tool", "name": "send_certificate", "content": "gc-001"},
    ]


def test_analyze_run_returns_report_with_scored_keystone():
    report = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)
    assert isinstance(report, AnalysisReport)
    assert report.state == STATE_SCORED
    assert report.completeness is GraphCompleteness.COMPLETE
    assert report.adapter == "tau_bench_prior_db_reads_v1"
    assert report.n_steps == 9

    # the first DB read (idx 2, get_order_details) is the unique keystone: the modify
    # (idx 4) and the send (idx 8) both transitively rest on it
    assert report.keystone is not None
    assert report.keystone.idx == 2
    assert report.keystone.label == "tool_call get_order_details"
    assert math.isclose(report.keystone.score, 0.25)
    assert math.isclose(report.per_session, 0.25)

    # ranked highest-first; the second read (idx 6) carries half the blast
    assert [d.idx for d in report.ranked][:2] == [2, 6]
    assert report.ranked[0].score >= report.ranked[1].score
    by_idx = {d.idx: d for d in report.ranked}
    assert math.isclose(by_idx[6].score, 0.125)


def test_report_coverage_is_observed_but_modeled_and_unsaturated():
    report = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)
    cov = report.coverage
    assert cov.n_dep_edges == 3            # modify->read, send->read, send->read2
    assert cov.observed_fraction == 1.0
    assert cov.rho < 0.9
    # the honesty contract: corpus observed edges are MODELED, said in the notes
    assert any("MODELED" in n for n in report.notes)
    assert any("not calibrated" in n for n in report.notes)


def test_corpus_run_has_no_model_basis_grounding():
    # a corpus tool trace states no model basis, so grounding is empty and the report
    # says so honestly rather than inventing a score
    report = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)
    assert report.grounding == {}
    assert report.keystone.grounding is None
    assert any("grounding is empty" in n for n in report.notes)


def test_summary_renders_keystone_coverage_and_notes():
    report = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)
    text = report.summary()
    assert "structural risk analysis" in text
    assert "keystone decision: step 2" in text
    assert "get_order_details" in text
    assert "rho=" in text
    assert str(report) == text  # print(report) renders the summary


# --- own-record path: declared edges, low coverage, but grounding lights up -----


def _own_records(n=3):
    from auditable import Action, DependencySnapshot, MemorySink, audit

    sink = MemorySink()
    for i in range(n):
        snap = DependencySnapshot(state={"budget_remaining": 5000, "policy_id": "kyc-2026-03"})
        with audit(f"pay-{i}", snapshot=snap, sink=sink) as d:
            d.read(invoice=f"INV-{100 + i}", vendor="acme")
            d.model(
                "gpt-x",
                decision_basis=(
                    f"Invoice INV-{100 + i} from acme is within the 5000 budget "
                    "under policy kyc-2026-03."
                ),
            )
            d.act(Action(f"pay-{i}", {"to": "acme"}, cost=100.0))
    return sink.records


def test_own_records_low_coverage_but_basis_is_grounded():
    records = _own_records(3)
    report = analyze_run(records, adapter=own_record_v1)

    # declared, non-observed dependency edges -> the honest no-score gate
    assert report.state == STATE_LOW_COVERAGE
    assert report.keystone is None
    assert report.per_session is None
    assert all(d.score is None for d in report.ranked)
    assert any("DECLARED" in n for n in report.notes)
    assert any("withheld" in n for n in report.notes)

    # but grounding lights up from the records' own basis + read context (a real score)
    assert set(report.grounding) == {0, 1, 2}
    g0 = report.grounding[0]
    assert g0.state == "scored"
    assert g0.score is not None and g0.score > 0.5
    # the checkable anchors in the basis are found in what was read
    assert "inv-100" in g0.matched
    assert "kyc-2026-03" in g0.matched
    # the per-step grounding also rides on the ranked row
    assert {d.idx: d.grounding for d in report.ranked}[0] is g0


def test_ground_false_skips_the_grounding_pass():
    records = _own_records(2)
    report = analyze_run(records, adapter=own_record_v1, ground=False)
    assert report.grounding == {}
    assert all(d.grounding is None for d in report.ranked)


# --- the no-score gates surface through the report ------------------------------


def test_single_decision_is_no_score():
    records = _own_records(1)
    report = analyze_run(records, adapter=own_record_v1)
    assert report.state == STATE_SINGLE_DECISION
    assert report.keystone is None
    assert report.per_session is None
    # the single decision still carries its grounding
    assert set(report.grounding) == {0}
    assert any("single-decision" in n for n in report.notes)


def test_empty_source_is_single_decision_safe():
    # an empty corpus run yields no steps; the report must not crash and must withhold
    report = analyze_run([], adapter=tau_bench_prior_db_reads_v1)
    assert report.n_steps == 0
    assert report.keystone is None
    assert report.ranked == []
    assert isinstance(report.summary(), str)


def test_ranked_rows_are_decision_risk_with_labels():
    report = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)
    assert all(isinstance(d, DecisionRisk) for d in report.ranked)
    user_row = next(d for d in report.ranked if d.agent == "user")
    assert user_row.kind == "decision"
    assert user_row.score == 0.0  # a user turn nothing rests on
