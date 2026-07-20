import copy
import random

import pytest

from auditable import DependencySnapshot
from auditable.data.inject import (
    build_injected_corpus,
    inject_drift,
    inject_ood,
    inject_stale,
)


NOW = 1_800_000_000.0


def _snapshot(index=0):
    return DependencySnapshot(
        state={
            "budget": 100.0 + index,
            "version": 7,
            "config": "cfg-12",
            "allow_list": ["acme", "globex"],
            "enabled": True,
        },
        captured_at=NOW - index,
    )


def test_corpus_is_deterministic_for_the_same_seed():
    snapshots = [_snapshot(index) for index in range(40)]

    first = build_injected_corpus(snapshots, rate=0.75, seed=27, now=NOW)
    second = build_injected_corpus(snapshots, rate=0.75, seed=27, now=NOW)

    assert first == second
    assert [row.label for row in first] == [row.label for row in second]
    assert [row.snapshot.state for row in first] == [row.snapshot.state for row in second]


def test_injectors_do_not_mutate_the_input_snapshot():
    snapshot = _snapshot()
    before_state = copy.deepcopy(snapshot.state)
    before_captured_at = snapshot.captured_at

    results = [
        inject_stale(snapshot, now=NOW, rng=random.Random(1)),
        inject_drift(snapshot, keys=["budget"], rng=random.Random(1)),
        inject_ood(snapshot, keys=["config"], rng=random.Random(1)),
    ]

    assert snapshot.state == before_state
    assert snapshot.captured_at == before_captured_at
    for result in results:
        assert result.snapshot is not snapshot
        assert result.snapshot.state is not snapshot.state
        assert result.snapshot.state["allow_list"] is not snapshot.state["allow_list"]


def test_stale_backdates_by_the_recorded_age():
    snapshot = _snapshot()
    result = inject_stale(
        snapshot,
        min_age_seconds=7 * 86400.0,
        max_age_seconds=30 * 86400.0,
        now=NOW,
        rng=random.Random(3),
    )
    injection = result.injections[0]

    assert injection.kind == "stale"
    assert injection.key == "captured_at"
    assert injection.original == snapshot.captured_at
    assert NOW - result.snapshot.captured_at >= 7 * 86400.0
    assert NOW - result.snapshot.captured_at <= 30 * 86400.0
    assert NOW - result.snapshot.captured_at == pytest.approx(
        injection.detail["age_seconds"]
    )
    assert injection.injected == result.snapshot.captured_at


def test_stale_handles_a_missing_capture_time():
    snapshot = DependencySnapshot(state={"budget": 100}, captured_at=None)
    result = inject_stale(
        snapshot,
        min_age_seconds=10.0,
        max_age_seconds=10.0,
        now=NOW,
        rng=random.Random(0),
    )
    injection = result.injections[0]

    assert injection.original is None
    assert injection.injected == NOW - 10.0
    assert result.snapshot.captured_at == NOW - 10.0
    assert snapshot.captured_at is None


@pytest.mark.parametrize("direction, multiplier", [("down", 0.6), ("up", 1.4)])
def test_drift_changes_exactly_one_numeric_key_by_its_fraction(direction, multiplier):
    snapshot = _snapshot()
    result = inject_drift(
        snapshot,
        fraction=(0.4, 0.4),
        direction=direction,
        rng=random.Random(4),
    )
    injection = result.injections[0]
    changed = [
        key
        for key in snapshot.state
        if result.snapshot.state[key] != snapshot.state[key]
    ]

    assert injection.kind == "drift"
    assert changed == [injection.key]
    assert injection.original == snapshot.state[injection.key]
    assert injection.injected == pytest.approx(injection.original * multiplier)
    assert result.snapshot.state[injection.key] == pytest.approx(injection.injected)
    assert injection.detail == {"fraction": 0.4, "direction": direction}
    assert injection.key != "enabled"


def test_ood_string_adds_an_unseen_token_suffix():
    snapshot = _snapshot()
    result = inject_ood(snapshot, keys=["config"], rng=random.Random(5))
    injection = result.injections[0]

    assert injection.kind == "ood"
    assert injection.key == "config"
    assert injection.original == "cfg-12"
    assert injection.injected == "cfg-12__ood"
    assert result.snapshot.state["config"].endswith("__ood")
    assert injection.detail["rule"]


@pytest.mark.parametrize("value, expected", [(12, 1200.0), (0, 1e6)])
def test_ood_numeric_value_leaves_the_fit_envelope(value, expected):
    snapshot = DependencySnapshot(state={"amount": value}, captured_at=NOW)
    result = inject_ood(snapshot, rng=random.Random(6))
    injection = result.injections[0]

    assert injection.kind == "ood"
    assert injection.original == value
    assert injection.injected == expected
    assert result.snapshot.state["amount"] == expected
    assert injection.detail["rule"]


def test_drift_rejects_a_snapshot_without_numeric_keys():
    snapshot = DependencySnapshot(
        state={"config": "cfg-12", "enabled": True, "allow_list": []}
    )

    with pytest.raises(ValueError):
        inject_drift(snapshot, rng=random.Random(0))


def test_ood_rejects_a_snapshot_without_eligible_keys():
    snapshot = DependencySnapshot(state={"enabled": True, "allow_list": [], "meta": {}})

    with pytest.raises(ValueError):
        inject_ood(snapshot, rng=random.Random(0))


def test_corpus_injection_rate_tracks_the_requested_rate():
    rate = 0.4
    snapshots = [_snapshot(index) for index in range(500)]
    rows = build_injected_corpus(snapshots, rate=rate, seed=11, now=NOW)
    observed = sum(row.label != "clean" for row in rows) / len(rows)

    assert observed == pytest.approx(rate, abs=0.07)


def test_eligible_corpus_contains_all_fault_kinds():
    snapshots = [_snapshot(index) for index in range(150)]
    rows = build_injected_corpus(snapshots, rate=1.0, seed=19, now=NOW)

    assert {row.label for row in rows} == {"stale", "drift", "ood"}
    assert all(row.injections[0].kind == row.label for row in rows)


def test_ineligible_draw_falls_through_and_clean_rows_have_no_injections():
    string_only = [DependencySnapshot(state={"config": "cfg-12"}) for _ in range(30)]
    faulted = build_injected_corpus(
        string_only, kinds=("drift", "ood"), rate=1.0, seed=9, now=NOW
    )
    clean = build_injected_corpus(string_only, rate=0.0, seed=9, now=NOW)

    assert all(row.label == "ood" for row in faulted)
    assert all(row.injections and row.injections[0].kind == "ood" for row in faulted)
    assert all(row.label == "clean" and row.injections == [] for row in clean)
    assert all(row.snapshot is not source for row, source in zip(clean, string_only))
