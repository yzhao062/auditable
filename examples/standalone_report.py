"""Standalone module API: score one stage on its own, get a signed report.

Lower-level than the full-chain record. Here we score only the data a decision relied on,
with no agent, no chain, and no replay. This is the data signal that feeds the data span
of a full-chain record (the snapshot-freshness leaf in payment_audit.py's signed record).
Each module is an input to the record, not the headline product; composition across the
three lifecycle pillars is what leads.

Run:  python examples/standalone_report.py
"""
import time

from auditable import DataAuditor, DependencySnapshot

SEVEN_DAYS = 7 * 24 * 60 * 60


def main():
    now = time.time()
    snapshot = DependencySnapshot(
        state={"budget_remaining": 1000}, captured_at=now - SEVEN_DAYS
    )
    report = DataAuditor(max_age_seconds=24 * 60 * 60).assess(snapshot, now=now)

    print("Standalone data report (no agent, no chain):")
    print(f"  stage={report.stage} name={report.name}")
    print(f"  flag={report.flag}  score={report.score}")
    print(f"  reason: {report.reason}")
    print(f"  evidence: {report.evidence}")
    print(f"  leaf digest: {report.digest()[:16]}...")


if __name__ == "__main__":
    main()
