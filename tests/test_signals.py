import time

from auditable import (
    DependencySnapshot,
    ModelSpan,
    score_model_trust,
    score_snapshot_freshness,
)


def test_freshness_flags_stale_snapshot():
    now = time.time()
    snap = DependencySnapshot(state={"budget": 100}, captured_at=now - 1000)
    sig = score_snapshot_freshness(snap, now=now, max_age_seconds=500)
    assert sig.stale is True
    assert sig.anomaly_score > 1.0


def test_freshness_passes_fresh_snapshot():
    now = time.time()
    snap = DependencySnapshot(state={"budget": 100}, captured_at=now - 100)
    sig = score_snapshot_freshness(snap, now=now, max_age_seconds=500)
    assert sig.stale is False
    assert sig.anomaly_score < 1.0


def test_freshness_no_capture_time_is_not_stale():
    now = time.time()
    snap = DependencySnapshot(state={"budget": 100})
    sig = score_snapshot_freshness(snap, now=now, max_age_seconds=500)
    assert sig.stale is False


def test_model_trust_flags_missing_basis():
    assert score_model_trust(ModelSpan(model_id="m")).flag == "low_trust"
    assert score_model_trust(ModelSpan(model_id="m", decision_basis="x")).flag == "ok"
