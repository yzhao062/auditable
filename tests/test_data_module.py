import json
import random
import time

import pytest

pytest.importorskip("pyod")  # the learned path needs the optional 'anomaly' extra

from auditable import DataAuditor, DependencySnapshot, FileSink, MemorySink, audit


def _normal_snapshots(n=200, now=None, seed=0):
    now = time.time() if now is None else now
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        state = {
            "budget_remaining": round(rng.uniform(200, 2000), 2),
            "allow_list_version": 7,
            "config_version": "cfg-12",
            "policy_id": "kyc-2026-03",
            "allow_list": ["acme", "globex", "initech"][: rng.randint(2, 3)],
        }
        out.append(DependencySnapshot(state=state, captured_at=now - rng.uniform(0, 3600)))
    return out


def _capture_snapshots(snapshots, sink):
    for index, snapshot in enumerate(snapshots):
        with audit("vendor_payment", snapshot=snapshot, sink=sink) as decision:
            decision.read(request_id=f"req-{index:03d}")


def test_unfitted_uses_freshness_fallback_mode():
    now = time.time()
    rep = DataAuditor().assess(
        DependencySnapshot(state={"budget_remaining": 100}, captured_at=now - 10), now=now
    )
    assert rep.evidence["mode"] == "freshness_fallback"
    assert rep.evidence["reason_code"] == "freshness_rule"


def test_fit_then_learned_mode_scores_in_unit_range():
    now = time.time()
    auditor = DataAuditor().fit(_normal_snapshots(now=now), now=now)
    rep = auditor.assess(_normal_snapshots(n=1, seed=99, now=now)[0], now=now)
    assert rep.evidence["mode"] == "learned"
    assert rep.evidence["detector"] == "ECOD"
    assert 0.0 <= rep.score <= 1.0


def test_learned_flags_out_of_envelope_state():
    now = time.time()
    auditor = DataAuditor().fit(_normal_snapshots(now=now), now=now)
    weird = DependencySnapshot(
        state={
            "budget_remaining": 10_000_000,
            "allow_list_version": 1,
            "config_version": "cfg-00",
            "policy_id": "unknown-policy",
            "allow_list": [],
        },
        captured_at=now - 30 * 86400,
    )
    rep = auditor.assess(weird, now=now)
    assert rep.flag == "anomalous"
    assert rep.score > 0.5
    assert rep.evidence["top_features"]  # names the features that drove the score


def test_detector_injection_uses_custom_detector():
    from pyod.models.iforest import IForest

    now = time.time()
    auditor = DataAuditor(detector=IForest(random_state=0)).fit(_normal_snapshots(now=now), now=now)
    rep = auditor.assess(_normal_snapshots(n=1, seed=7, now=now)[0], now=now)
    assert rep.evidence["mode"] == "learned"
    assert rep.evidence["detector"] == "IForest"


def test_fit_requires_snapshots():
    with pytest.raises(ValueError):
        DataAuditor().fit([])


def test_fit_records_uses_captured_memory_history():
    now = time.time()
    sink = MemorySink()
    _capture_snapshots(_normal_snapshots(n=50, now=now), sink)

    auditor = DataAuditor().fit_records(sink.records, now=now)
    rep = auditor.assess(_normal_snapshots(n=1, now=now, seed=99)[0], now=now)

    assert rep.evidence["mode"] == "learned"
    assert 0.0 <= rep.score <= 1.0


def test_fit_records_rejects_history_with_only_empty_states():
    now = time.time()
    sink = MemorySink()
    _capture_snapshots(
        [DependencySnapshot(state={}, captured_at=now) for _ in range(3)], sink
    )

    with pytest.raises(ValueError, match="non-empty snapshot state"):
        DataAuditor().fit_records(sink.records, now=now)


def test_fit_from_log_uses_verified_file_history(tmp_path):
    now = time.time()
    path = tmp_path / "decisions.jsonl"
    _capture_snapshots(
        _normal_snapshots(n=50, now=now), FileSink(str(path))
    )

    auditor = DataAuditor().fit_from_log(path, now=now)
    rep = auditor.assess(_normal_snapshots(n=1, now=now, seed=101)[0], now=now)

    assert rep.evidence["mode"] == "learned"
    assert 0.0 <= rep.score <= 1.0


def test_fit_from_log_fails_closed_on_tampered_history(tmp_path):
    now = time.time()
    path = tmp_path / "tampered.jsonl"
    _capture_snapshots(
        _normal_snapshots(n=3, now=now), FileSink(str(path))
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["data"]["snapshot"]["state"]["budget_remaining"] = -1
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="record_id mismatch"):
        DataAuditor().fit_from_log(path, now=now)


def test_fitted_auditor_still_returns_a_report_for_empty_state():
    now = time.time()
    auditor = DataAuditor().fit(_normal_snapshots(now=now), now=now)
    rep = auditor.assess(DependencySnapshot(state={}, captured_at=now - 100), now=now)
    assert rep.stage == "data"
    assert 0.0 <= rep.score <= 1.0
