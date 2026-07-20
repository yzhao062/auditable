"""TDD for the LIVE streaming analysis surface (the v0.3b live path).

`analyze_run` scores a finished run (POST, completeness=complete). `LiveSession`
grows the same typed graph as steps stream in and re-scores the prefix, so a caller
gets a running keystone and per-step blast share while the run is still going
(completeness=prefix). Behavioral only: this is the same structural kernel run on a
growing graph. It carries NO failure-label validation and makes no early-warning
claim; the prefix-AUC curve that would validate early warning stays private.
"""
import math
import threading
import time

import pytest

pytest.importorskip("networkx")  # LiveSession scores over the NetworkX projection

from auditable import (  # noqa: E402
    DataAuditor,
    DependencySnapshot,
    LiveSession,
    Report,
    analyze_run,
)
from auditable.graph.adapters import tau_bench_prior_db_reads_v1  # noqa: E402
from auditable.graph.risk import STATE_SCORED  # noqa: E402
from auditable.graph.session import GraphCompleteness, Step  # noqa: E402


def _tau_messages():
    """A retail run: read the order, modify it, read the user, send a refund cert.

    Two consequential writes (modify, send) both rest on the first read, so that read
    is the unique structural keystone once the whole run is seen.
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


def _steps():
    return list(tau_bench_prior_db_reads_v1.to_steps(_tau_messages()))


class _StateScoreAuditor:
    """Small deterministic scorer for checking captured state ownership."""

    def assess(self, snapshot):
        return Report(
            stage="data",
            name="state-score",
            score=snapshot.state["risk"]["score"],
            flag="ok",
        )


def test_observe_grows_prefix_and_marks_completeness_prefix():
    live = LiveSession()
    report = None
    for s in _steps():
        report = live.observe(s)
    assert report.completeness is GraphCompleteness.PREFIX
    assert report.state == STATE_SCORED
    assert report.adapter == "live_session_v1"
    assert report.n_steps == 9
    assert report.keystone is not None
    assert report.keystone.idx == 2
    assert math.isclose(report.keystone.score, 0.25)


def test_live_after_all_steps_agrees_with_batch_ranking():
    live = LiveSession()
    for s in _steps():
        live.observe(s)
    live_report = live.report()
    batch = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)

    # same structural scoring; only the completeness label differs
    assert [d.idx for d in live_report.ranked] == [d.idx for d in batch.ranked]
    assert live_report.keystone.idx == batch.keystone.idx
    assert math.isclose(live_report.per_session, batch.per_session)
    assert live_report.completeness is GraphCompleteness.PREFIX
    assert batch.completeness is GraphCompleteness.COMPLETE


def test_keystone_emerges_as_the_prefix_grows():
    reports = []
    live = LiveSession()
    for s in _steps():
        reports.append(live.observe(s))

    keystones = [r.keystone.idx if r.keystone else None for r in reports]
    # the read (idx 2) is the keystone only after a later write rests on it; early
    # prefixes have no dependent yet, so the running keystone evolves.
    assert keystones[-1] == 2
    assert any(k != 2 for k in keystones)


def test_short_prefix_is_withheld_then_scored():
    live = LiveSession()
    first = live.observe(_steps()[0])
    assert first.state != STATE_SCORED  # a one-step prefix cannot be scored

    last = None
    for s in _steps()[1:]:
        last = live.observe(s)
    assert last.state == STATE_SCORED


def test_prefix_report_carries_running_caveat_and_no_early_warning_claim():
    live = LiveSession()
    for s in _steps():
        live.observe(s)
    notes = " ".join(live.report().notes).lower()
    assert "prefix" in notes
    # honest framing: a running triage signal, not a validated early-warning probability
    assert "triage" in notes or "early-warning" in notes or "early warning" in notes


def test_live_data_auditor_assesses_snapshot_state_and_state_fallback():
    live = LiveSession(data_auditor=DataAuditor())
    live.observe(
        Step(
            idx=0,
            agent="agent",
            kind="decision",
            node_attrs={"snapshot_state": {"policy_version": 7}},
        )
    )
    report = live.observe(
        Step(
            idx=1,
            agent="agent",
            kind="decision",
            node_attrs={"state": {"budget_remaining": 100}},
        )
    )

    assert set(report.data_reports) == {0, 1}
    assert all(r.flag == "ok" for r in report.data_reports.values())
    assert all(
        r.evidence["mode"] == "freshness_fallback"
        for r in report.data_reports.values()
    )


def test_node_attr_snapshot_is_deep_copied_at_observation_time():
    state = {"risk": {"score": 0.2}}
    step = Step(
        idx=8,
        agent="agent",
        kind="decision",
        node_attrs={"snapshot_state": state},
    )
    live = LiveSession(data_auditor=_StateScoreAuditor())

    first = live.observe(step)
    state["risk"]["score"] = 0.9
    repeated = live.report()

    assert set(first.data_reports) == {8}
    assert first.data_reports[8].score == 0.2
    assert repeated.data_reports[8].score == 0.2


def test_explicit_snapshot_wins_and_is_deep_copied_at_observation_time():
    node_state = {"risk": {"score": 0.9}}
    snapshot = DependencySnapshot(state={"risk": {"score": 0.25}})
    step = Step(
        idx=12,
        agent="agent",
        kind="decision",
        node_attrs={"snapshot_state": node_state},
    )
    live = LiveSession(data_auditor=_StateScoreAuditor())

    first = live.observe(step, snapshot=snapshot)
    snapshot.state["risk"]["score"] = 0.75
    node_state["risk"]["score"] = 0.1
    repeated = live.report()

    assert set(first.data_reports) == {12}
    assert first.data_reports[12].score == 0.25
    assert repeated.data_reports[12].score == 0.25


def test_stale_empty_live_snapshot_flags_step_adds_honesty_note_and_summary_line():
    live = LiveSession(data_auditor=DataAuditor(max_age_seconds=1))
    report = live.observe(
        Step(
            idx=4,
            agent="agent",
            kind="decision",
            node_attrs={
                "snapshot_state": {},
                "snapshot_captured_at": time.time() - 60,
            },
        )
    )

    assert report.data_reports[4].flag == "stale"
    data_notes = [n for n in report.notes if "data score" in n]
    assert len(data_notes) == 1
    assert "[4]" in data_notes[0]
    assert "separate from the structural score" in data_notes[0]
    assert "not calibrated" in data_notes[0]
    summary = report.summary()
    assert "data-stage triage" in summary
    assert "step 4: stale" in summary


def test_live_data_auditor_skips_steps_without_snapshot_state():
    report = LiveSession(data_auditor=DataAuditor()).observe(
        Step(
            idx=0,
            agent="agent",
            kind="decision",
            node_attrs={"model_id": "example-model"},
        )
    )

    assert report.data_reports == {}


def test_raising_data_auditor_reports_unavailable_step_without_breaking_structure():
    class RaisingDataAuditor:
        def __init__(self):
            self._fallback = DataAuditor(max_age_seconds=1)

        def assess(self, snapshot):
            if snapshot.state.get("raise"):
                raise TimeoutError("scorer unavailable")
            return self._fallback.assess(snapshot)

    live = LiveSession(data_auditor=RaisingDataAuditor())
    first = live.observe(
        Step(
            idx=0,
            agent="agent",
            kind="decision",
            node_attrs={"snapshot_state": {"raise": True}},
        )
    )
    report = live.observe(
        Step(
            idx=1,
            agent="agent",
            kind="decision",
            node_attrs={
                "snapshot_state": {"raise": False},
                "snapshot_captured_at": time.time() - 60,
            },
        )
    )

    assert first.n_steps == 1
    assert first.data_reports == {}
    assert len(first.ranked) == 1
    assert any(
        "steps 0 (TimeoutError)" in note and "no data score was produced" in note
        for note in first.notes
    )
    assert report.n_steps == 2
    assert len(report.ranked) == 2
    assert set(report.data_reports) == {1}
    assert report.data_reports[1].flag == "stale"
    assert any("flagged dependency snapshots" in note for note in report.notes)
    assert any(
        "steps 0 (TimeoutError)" in note and "no data score was produced" in note
        for note in report.notes
    )


def test_offline_analysis_keeps_data_reports_empty():
    report = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)

    assert report.data_reports == {}


def test_caller_step_mutation_after_observe_cannot_rewrite_evidence():
    state = {"risk": {"score": 0.3}}
    step = Step(idx=3, agent="agent", kind="decision", node_attrs={"snapshot_state": state})
    live = LiveSession(data_auditor=_StateScoreAuditor())

    first = live.observe(step)
    step.node_attrs["snapshot_state"]["risk"]["score"] = 0.95
    step.idx = 4  # the caller's object identity is not trusted either
    repeated = live.report()

    assert first.data_reports[3].score == 0.3
    assert repeated.data_reports[3].score == 0.3
    assert set(repeated.data_reports) == {3}


def test_mutating_report_session_graph_does_not_change_future_reports():
    live = LiveSession(data_auditor=_StateScoreAuditor())
    first = live.observe(
        Step(
            idx=5,
            agent="agent",
            kind="decision",
            node_attrs={"snapshot_state": {"risk": {"score": 0.4}}},
        )
    )

    graph_step = first.session_graph.steps[0]
    graph_step.node_attrs["snapshot_state"] = {"risk": {"score": 0.99}}
    repeated = live.report()

    assert repeated.data_reports[5].score == 0.4


def test_duplicate_step_idx_raises_before_committing_and_session_stays_usable():
    live = LiveSession(data_auditor=_StateScoreAuditor())
    live.observe(
        Step(idx=0, agent="agent", kind="decision", node_attrs={}),
        snapshot=DependencySnapshot(state={"risk": {"score": 0.5}}),
    )

    with pytest.raises(ValueError, match="duplicate Step.idx"):
        live.observe(
            Step(idx=0, agent="agent", kind="decision", node_attrs={}),
            snapshot=DependencySnapshot(state={"risk": {"score": 0.9}}),
        )

    report = live.report()
    assert [d.idx for d in report.ranked] == [0]
    assert report.data_reports[0].score == 0.5


def test_concurrent_same_idx_observations_commit_exactly_once():
    live = LiveSession(data_auditor=_StateScoreAuditor())
    barrier = threading.Barrier(2)
    outcomes = []

    def worker(score):
        step = Step(idx=7, agent="agent", kind="decision", node_attrs={})
        barrier.wait()
        try:
            live.observe(
                step, snapshot=DependencySnapshot(state={"risk": {"score": score}})
            )
            outcomes.append(("ok", score))
        except ValueError:
            outcomes.append(("dup", score))

    threads = [threading.Thread(target=worker, args=(s,)) for s in (0.2, 0.8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcome for outcome, _ in outcomes) == ["dup", "ok"]
    winner = next(score for outcome, score in outcomes if outcome == "ok")
    report = live.report()
    assert len(report.ranked) == 1
    assert report.data_reports[7].score == winner
