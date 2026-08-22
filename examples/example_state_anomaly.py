"""Data-module concept demo: capture clean state, fit history, inject faults.

This offline example captures a normal dependency-state history as signed, chained
``DecisionRecord`` rows in a durable ``FileSink`` JSONL log. ``DataAuditor`` verifies
that log and fits its learned envelope directly from the captured snapshots. The
injector then makes deterministic stale, drifted, and out-of-distribution copies so
the same auditor can score clean and faulted rows side by side.

The injected faults are synthetic and the resulting table is demonstration evidence,
not a detector benchmark. No API key or network access is needed.

Needs:  pip install "auditable[anomaly]"
Run:    python examples/example_state_anomaly.py
"""
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics
import tempfile
import time

from auditable import Action, DataAuditor, DependencySnapshot, FileSink, audit


def _pyod_skip_reason():
    try:
        import pyod  # noqa: F401
    except ImportError:
        return 'the pyod package is not installed (pip install "auditable[anomaly]")'
    return None


def _capture_clean_history(path: Path, *, now: float):
    rng = random.Random(17)
    sink = FileSink(str(path))
    snapshots = []
    vendors = ["acme-supplies", "globex", "initech"]

    for index in range(60):
        allow_count = 2 + index % 2
        snapshot = DependencySnapshot(
            state={
                "budget_remaining": round(rng.uniform(8_000, 12_000), 2),
                "allow_list_version": 17 + index // 30,
                "config_version": f"payments-{4 + index // 30}",
                "policy_id": "vendor-payments-2026-07",
                "allow_list": vendors[:allow_count],
            },
            captured_at=now - rng.uniform(60, 3_600),
        )
        snapshots.append(snapshot)
        vendor = vendors[index % allow_count]
        cost = round(rng.uniform(100, 1_200), 2)
        with audit("vendor_payment", snapshot=snapshot, sink=sink) as decision:
            decision.read(request_id=f"pay-{index:03d}", vendor=vendor)
            decision.model(
                "offline-policy-v1",
                decision_basis="Vendor is allow-listed and the payment is in budget.",
                output="approve",
            )
            decision.act(Action("vendor_payment", {"recipient": vendor}, cost=cost))

    return snapshots


def _print_score_table(rows, auditor, *, now: float):
    scores = defaultdict(list)
    flags = defaultdict(Counter)
    for row in rows:
        report = auditor.assess(row.snapshot, now=now)
        scores[row.label].append(report.score)
        flags[row.label][report.flag] += 1

    print("\nLearned-envelope scores:")
    print(f"{'label':<8} {'rows':>5} {'mean score':>12}  flag counts")
    print(f"{'-' * 8} {'-' * 5} {'-' * 12}  {'-' * 20}")
    for label in ("clean", "stale", "drift", "ood"):
        if label not in scores:
            continue
        flag_counts = ", ".join(
            f"{flag}={count}" for flag, count in sorted(flags[label].items())
        )
        print(
            f"{label:<8} {len(scores[label]):>5} "
            f"{statistics.fmean(scores[label]):>12.3f}  {flag_counts}"
        )


def main():
    skip = _pyod_skip_reason()
    if skip:
        print(f"example_state_anomaly: skipping because {skip}.")
        print("This example is offline once the optional anomaly extra is installed.")
        return

    # Imported after the optional-dependency guard so a missing anomaly extra skips
    # cleanly. The injector itself is pure and deep-copies every source snapshot.
    from auditable.data.inject import build_injected_corpus

    now = time.time()
    with tempfile.TemporaryDirectory(prefix="auditable-state-") as temp_dir:
        log_path = Path(temp_dir) / "clean-decisions.jsonl"
        clean_snapshots = _capture_clean_history(log_path, now=now)
        print(f"Captured {len(clean_snapshots)} clean decisions in {log_path}")

        auditor = DataAuditor().fit_from_log(str(log_path), now=now)
        rows = build_injected_corpus(
            clean_snapshots,
            kinds=("stale", "drift", "ood"),
            rate=0.75,
            seed=23,
            now=now,
        )
        _print_score_table(rows, auditor, now=now)

    print("\nHonesty note: this is a concept demo on synthetic faults, not benchmark evidence.")
    print(
        "An artifact-controlled CatchBench Gold board on a named-value substrate "
        "remains the validation target."
    )


if __name__ == "__main__":
    main()
