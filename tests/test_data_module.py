import random
import time

import pytest

pytest.importorskip("pyod")  # the learned path needs the optional 'anomaly' extra

from auditable import DataAuditor, DependencySnapshot


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


def test_unfitted_uses_freshness_fallback_mode():
    now = time.time()
    rep = DataAuditor().assess(
        DependencySnapshot(state={"budget_remaining": 100}, captured_at=now - 10), now=now
    )
    assert rep.evidence["mode"] == "freshness_fallback"
    assert rep.evidence["reason_code"] in ("unfit_detector", "pyod_absent")


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


def test_fitted_auditor_still_returns_a_report_for_empty_state():
    now = time.time()
    auditor = DataAuditor().fit(_normal_snapshots(now=now), now=now)
    rep = auditor.assess(DependencySnapshot(state={}, captured_at=now - 100), now=now)
    assert rep.stage == "data"
    assert 0.0 <= rep.score <= 1.0
