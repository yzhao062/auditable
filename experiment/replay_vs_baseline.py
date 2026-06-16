"""Harness experiment: does replay-under-live-state catch stale-state payment failures
that a static spend cap and a snapshot-only policy structurally miss?

Real public data: payment amounts are sampled from the ULB credit-card transaction
dataset (OpenML), using the real amount distribution, not the fraud labels. The one
constructed dimension is the temporal drift (snapshot budget vs live budget), because
stale-state failure is the new risk and no public dataset captures it.

Honest framing: replay is not a smarter classifier. It checks the right thing (the live
budget) where the incumbents check a proxy: a fixed spend cap (AgentCore / AP2 style) or
the stale snapshot the agent saw. The point the numbers make is structural: a large
fraction of stale-state failures sit under any fixed cap and so are invisible to
cap-based controls, while replay re-evaluates live state and catches them.

A stale-state failure is a payment justified on the snapshot (amount <= snapshot budget)
but wrong on the live state (amount > live budget). We report, on that subset, how many
each method flags, and the false-positive rate on clean payments (justified on both).

Run:  python experiment/replay_vs_baseline.py
"""
import csv
import os
import random

CACHE = os.path.join(os.path.dirname(__file__), ".cache", "creditcard_amounts_10k.csv")


def load_amounts():
    with open(CACHE) as f:
        return [float(r["Amount"]) for r in csv.DictReader(f) if float(r["Amount"]) > 0]


def build_scenarios(amounts, n, seed=0):
    """Each scenario: amount A, snapshot budget B_snap, live budget B_live, static cap.

    B_snap usually covers the payment (justified at decision time), occasionally not.
    B_live drifts down from B_snap by a random factor. The cap is a fixed high percentile
    of real amounts, the kind of limit a rail or AgentCore policy sets once.
    """
    rng = random.Random(seed)
    cap = sorted(amounts)[int(0.95 * len(amounts))]
    scenarios = []
    for _ in range(n):
        a = rng.choice(amounts)
        b_snap = a * rng.uniform(0.8, 3.0)
        b_live = b_snap * rng.uniform(0.2, 1.1)
        scenarios.append({"amount": a, "b_snap": b_snap, "b_live": b_live, "cap": cap})
    return scenarios


def classify(s):
    just_snap = s["amount"] <= s["b_snap"]
    just_live = s["amount"] <= s["b_live"]
    if just_snap and not just_live:
        return "stale_failure"
    if just_snap and just_live:
        return "clean"
    return "clearly_bad"


def flags(method, s):
    """True if the method would block or roll back the payment."""
    if method == "static_cap":
        return s["amount"] > s["cap"]
    if method == "snapshot_only":
        return s["amount"] > s["b_snap"]
    if method == "replay":
        return s["amount"] > s["b_live"]
    raise ValueError(method)


def main():
    amounts = load_amounts()
    scenarios = build_scenarios(amounts, n=20000, seed=0)
    labels = [classify(s) for s in scenarios]
    n_stale = labels.count("stale_failure")
    n_clean = labels.count("clean")
    cap = scenarios[0]["cap"]

    print(f"amounts: real ULB credit-card sample (n={len(amounts)})")
    print(
        f"scenarios: {len(scenarios)} | stale_failures: {n_stale} | clean: {n_clean} "
        f"| clearly_bad: {labels.count('clearly_bad')}"
    )
    print(f"static cap (95th-percentile real amount): {cap:.2f}")

    under_cap_stale = sum(
        1 for s, l in zip(scenarios, labels) if l == "stale_failure" and s["amount"] <= cap
    )
    print(
        f"stale failures under the cap (invisible to a cap-based control): "
        f"{100 * under_cap_stale / n_stale:.1f}%\n"
    )

    print(f"{'method':<16}{'catch% on stale-failures':>26}{'false-positive% on clean':>28}")
    for method in ("static_cap", "snapshot_only", "replay"):
        caught = sum(1 for s, l in zip(scenarios, labels) if l == "stale_failure" and flags(method, s))
        fp = sum(1 for s, l in zip(scenarios, labels) if l == "clean" and flags(method, s))
        catch = 100 * caught / n_stale if n_stale else 0.0
        fpr = 100 * fp / n_clean if n_clean else 0.0
        print(f"{method:<16}{catch:>25.1f}%{fpr:>27.1f}%")


if __name__ == "__main__":
    main()
