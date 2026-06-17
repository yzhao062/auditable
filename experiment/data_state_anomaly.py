"""v0.2 data experiment: does a fitted anomaly score on the dependency state pre-flag
stale-state failures that the v0.1 freshness rule misses?

Honest framing, two separate claims:

1. Stress test (synthetic). Dependency-state drift (stale snapshots, version downgrades,
   budget drift) is constructed over the real public transaction-amount distribution (ULB
   credit-card, OpenML), because public tabular data has no live dependency-state drift.
   The amounts are real; the drift is synthetic and labeled as such.
2. Measurement (the promotable result). At a fixed false-positive rate on clean states,
   how many replay-failing decisions does the fitted detector pre-flag, versus freshness?

Independence guard: the detector scores only the dependency state it would see at decision
time (budget, versions, allow-list, snapshot age). It never sees the action cost or the
drifted live budget. The label comes from replay (the cost was justified on the snapshot
budget but not on the live budget), so the detector is independent of the label-generating
replay. The claim is a pre-replay triage gain, not "detector plus replay beats freshness
plus replay," which replay would dominate.

Run:  python experiment/explore_creditcard.py   # once, caches the real ULB amounts
      python experiment/data_state_anomaly.py
"""
import csv
import os
import random
import time

from auditable import DataAuditor, DependencySnapshot

CACHE = os.path.join(os.path.dirname(__file__), ".cache", "creditcard_amounts_10k.csv")


def load_amounts():
    if not os.path.exists(CACHE):
        raise FileNotFoundError(
            f"Missing amount cache {CACHE}. Generate it once with: "
            "python experiment/explore_creditcard.py"
        )
    with open(CACHE) as handle:
        return [float(r["Amount"]) for r in csv.DictReader(handle) if float(r["Amount"]) > 0]


def build(amounts, n, now, seed=0):
    """Each row: a dependency snapshot, its kind, and a replay-derived label.

    The detector sees only the snapshot state. The label uses the action cost and the
    drifted live budget, which the detector never sees.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        cost = rng.choice(amounts)
        base_budget = cost * rng.uniform(1.2, 6.0)  # snapshot budget, usually covers the cost
        kind = rng.choices(["clean", "stale", "version_drift"], weights=[0.6, 0.2, 0.2])[0]
        if kind == "clean":
            age, ver, cfg, drift = rng.uniform(0, 3600), 7, "cfg-12", rng.uniform(0.0, 0.1)
        elif kind == "stale":
            age, ver, cfg, drift = rng.uniform(7 * 86400, 30 * 86400), 7, "cfg-12", rng.uniform(0.4, 0.9)
        else:  # version_drift: fresh snapshot, but a downgraded version/config
            age, ver, cfg, drift = rng.uniform(0, 3600), rng.choice([2, 3]), "cfg-09", rng.uniform(0.4, 0.9)
        live_budget = base_budget * (1 - drift)
        state = {
            "budget_remaining": round(base_budget, 2),
            "allow_list_version": ver,
            "config_version": cfg,
            "allow_list": ["acme", "globex", "initech"][: rng.randint(2, 3)],
        }
        snap = DependencySnapshot(state=state, captured_at=now - age)
        label = "stale_failure" if (cost <= base_budget and cost > live_budget) else "clean"
        rows.append({"snap": snap, "kind": kind, "label": label})
    return rows


def threshold_at_fpr(scored, target_fpr):
    # Admit at most floor(target_fpr * N_clean) clean rows at score >= threshold. With clean
    # sorted descending, clean[allowed - 1] is the largest threshold that keeps the realized
    # false-positive count at `allowed` (inclusive comparison). allowed == 0 means no clean
    # row may exceed it, so the threshold sits above the maximum clean score.
    clean = sorted((s for s, l in scored if l == "clean"), reverse=True)
    allowed = int(target_fpr * len(clean))
    if allowed == 0:
        return float("inf")
    return clean[allowed - 1]


def tpr_at_fpr(scored, target_fpr):
    thr = threshold_at_fpr(scored, target_fpr)
    fails = [s for s, l in scored if l == "stale_failure"]
    if not fails:
        return 0.0
    return sum(1 for s in fails if s >= thr) / len(fails)


def main():
    now = time.time()
    amounts = load_amounts()

    # Fit on the clean training states only: the agent's normal operating envelope.
    train = build(amounts, 800, now, seed=1)
    clean_train = [r["snap"] for r in train if r["kind"] == "clean"]
    learned = DataAuditor().fit(clean_train, now=now)
    freshness = DataAuditor(max_age_seconds=3 * 86400)  # v0.1 rule (unfit -> freshness fallback)

    ev = build(amounts, 6000, now, seed=2)
    learned_rows = [(learned.assess(r["snap"], now=now).score, r["label"], r["kind"]) for r in ev]
    fresh_rows = [
        (freshness.assess(r["snap"], now=now).evidence.get("raw_ratio", 0.0), r["label"], r["kind"])
        for r in ev
    ]
    n_fail = sum(1 for r in ev if r["label"] == "stale_failure")
    n_clean = sum(1 for r in ev if r["label"] == "clean")

    print(
        f"amounts: real ULB sample (n={len(amounts)}); fit corpus: {len(clean_train)} clean states; "
        f"eval scenarios: {len(ev)} (stale_failures {n_fail}, clean {n_clean})"
    )
    print(
        "drift is synthetic over real amounts; labels come from replay (cost vs live budget); "
        "the detector never sees cost or live budget.\n"
    )

    print(f"{'method':<24}{'TPR@FPR=5%':>13}{'TPR@FPR=10%':>13}")
    for name, rows in (("freshness (age only)", fresh_rows), ("learned ECOD on state", learned_rows)):
        scored = [(s, l) for s, l, _ in rows]
        print(f"{name:<24}{100 * tpr_at_fpr(scored, 0.05):>12.1f}%{100 * tpr_at_fpr(scored, 0.10):>12.1f}%")

    print("\nwhere the failures are caught at FPR=10% (by failure kind):")
    print(f"{'method':<24}{'stale':>10}{'version_drift':>16}")
    for name, rows in (("freshness (age only)", fresh_rows), ("learned ECOD on state", learned_rows)):
        thr = threshold_at_fpr([(s, l) for s, l, _ in rows], 0.10)
        cells = []
        for kind in ("stale", "version_drift"):
            fails = [s for s, l, k in rows if l == "stale_failure" and k == kind]
            caught = sum(1 for s in fails if s >= thr) / len(fails) if fails else 0.0
            cells.append(f"{100 * caught:.1f}%")
        print(f"{name:<24}{cells[0]:>10}{cells[1]:>16}")

    print(
        "\nReading: freshness catches stale snapshots through age but is blind to version "
        "and config drift; the learned state detector catches the version_drift failures "
        "freshness misses, and more failures overall. The two are complementary, which is the "
        "case for the v0.3 calibrated compound."
    )


if __name__ == "__main__":
    main()
