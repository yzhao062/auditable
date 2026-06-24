"""TDD for the LIVE streaming analysis surface (the v0.3b live path).

`analyze_run` scores a finished run (POST, completeness=complete). `LiveSession`
grows the same typed graph as steps stream in and re-scores the prefix, so a caller
gets a running keystone and per-step blast share while the run is still going
(completeness=prefix). Behavioral only: this is the same structural kernel run on a
growing graph. It carries NO failure-label validation and makes no early-warning
claim; the prefix-AUC curve that would validate early warning stays private.
"""
import math

from auditable import LiveSession, analyze_run
from auditable.graph.adapters import tau_bench_prior_db_reads_v1
from auditable.graph.risk import STATE_SCORED
from auditable.graph.session import GraphCompleteness


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
