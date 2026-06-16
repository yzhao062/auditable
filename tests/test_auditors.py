import time

import pytest

from auditable import (
    Action,
    DataAuditor,
    DependencySnapshot,
    HarnessAuditor,
    ModelAuditor,
    ModelSpan,
    Report,
)


def test_data_auditor_flags_stale_snapshot():
    now = time.time()
    snap = DependencySnapshot(state={"budget": 100}, captured_at=now - 1000)
    report = DataAuditor(max_age_seconds=500).assess(snap, now=now)
    assert report.stage == "data"
    assert report.flag == "stale"
    assert report.score == 1.0  # clipped to the normalized ceiling
    assert report.evidence["raw_ratio"] == 2.0


def test_data_auditor_passes_fresh_snapshot():
    now = time.time()
    snap = DependencySnapshot(state={"budget": 100}, captured_at=now - 100)
    report = DataAuditor(max_age_seconds=500).assess(snap, now=now)
    assert report.flag == "ok"
    assert report.score < 1.0


def test_data_auditor_no_capture_time_is_ok():
    now = time.time()
    snap = DependencySnapshot(state={"budget": 100})
    report = DataAuditor(max_age_seconds=500).assess(snap, now=now)
    assert report.flag == "ok"


def test_data_auditor_boundary_is_stale_at_exactly_budget():
    now = time.time()
    snap = DependencySnapshot(state={"budget": 100}, captured_at=now - 500)
    report = DataAuditor(max_age_seconds=500).assess(snap, now=now)
    assert report.flag == "stale"  # age == budget counts as stale, so score 1.0 lines up
    assert report.score == 1.0


def test_data_auditor_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        DataAuditor(max_age_seconds=0)
    with pytest.raises(ValueError):
        DataAuditor(max_age_seconds=-5)


def test_model_auditor_flags_missing_basis():
    assert ModelAuditor().assess(ModelSpan(model_id="m")).flag == "low_trust"
    assert ModelAuditor().assess(ModelSpan(model_id="m", decision_basis="x")).flag == "ok"


def test_harness_auditor_flags_over_cap():
    over = HarnessAuditor(cost_cap=100).assess(Action("pay", {}, cost=150))
    assert over.flag == "over_cap"
    assert over.score == 0.5
    under = HarnessAuditor(cost_cap=100).assess(Action("pay", {}, cost=80))
    assert under.flag == "ok"
    assert under.score == 0.0


def test_report_digest_is_stable_and_content_addressed():
    r1 = Report("data", "x", 0.5, "ok", "r")
    r2 = Report("data", "x", 0.5, "ok", "r")
    r3 = Report("data", "x", 0.6, "ok", "r")
    assert r1.digest() == r2.digest()
    assert r1.digest() != r3.digest()
