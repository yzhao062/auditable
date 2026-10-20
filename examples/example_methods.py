"""One method from each family on synthetic traces, with no benchmark data.

Run from anywhere after ``pip install "auditable[tabular-ad]"``:

    python examples/example_methods.py

The traces are generated here: each run is a sequence of agent steps, each step relies on
one or two recent steps, and a failed run contains one stale read, a step that relies on
state from much earlier in the run. The numbers printed describe this toy data only.
"""
import numpy as np

from auditable.featurize import AgentConfig
from auditable.methods import get_method, list_methods

AGENTS = ("planner", "coder", "tester")


def make_trace(n_steps, rng, stale_at=None):
    """A run of ``n_steps`` steps; ``stale_at`` (a step index) makes that step read step 0."""
    steps = []
    for i in range(n_steps):
        if i == 0:
            deps = []
        elif i == stale_at:
            deps = [0]
        else:
            deps = sorted({i - 1, max(0, i - 1 - int(rng.randint(0, 2)))})
        steps.append({"idx": i, "agent": AGENTS[int(rng.randint(0, 3))],
                      "kind": "tool_call" if rng.rand() < 0.5 else "decision", "deps": deps})
    return steps


def main():
    rng = np.random.RandomState(0)
    runs, labels, stale_steps = [], [], []
    for r in range(80):
        n = int(rng.randint(10, 20))
        failed = r % 2 == 1
        stale = int(rng.randint(n // 2, n)) if failed else None
        runs.append(make_trace(n, rng, stale_at=stale))
        labels.append(int(failed))
        stale_steps.append(stale)
    labels = np.array(labels)

    print("Registered methods by family:")
    for family in ("config", "run", "step", "online"):
        print(f"  {family:7s}", ", ".join(c.method_id for c in list_methods(family=family)))

    # PRE: audit a configuration against its own task text.
    config = AgentConfig(
        declared_capabilities=[
            {"name": "read_report", "type": "tool", "permission_level": "read"},
            {"name": "delete_files", "type": "tool", "permission_level": "write"},
            {"name": "run_shell", "type": "tool", "permission_level": "execute"},
        ],
        task_or_role_spec="Read the quarterly report and summarize the revenue section.",
    )
    finding = get_method("config.owasp_asi_combined")().audit(config)
    print("\nPRE  config.owasp_asi_combined: flag =", finding.flag, "excess =", sorted(finding.excess))

    # LIVE: a causal alarm per step on one failed run; the stale read should be the peak.
    failed_run = next(i for i, s in enumerate(stale_steps) if s is not None)
    alarms = get_method("online.span_z")().step_alarm_scores(runs[failed_run])
    print(f"LIVE online.span_z: peak alarm at step {int(np.argmax(alarms))}, "
          f"stale read at step {stale_steps[failed_run]}")

    # POST, step level: rank the steps of each failed run; how often is the stale read first?
    locator = get_method("step.dep_anomaly")()
    hits = [locator.rank(runs[i])[0] == s for i, s in enumerate(stale_steps) if s is not None]
    print(f"POST step.dep_anomaly: stale read ranked first in {sum(hits)} of {len(hits)} failed runs")

    # POST, run level: a supervised detector fit on half the runs, scored on the other half,
    # next to an unsupervised one fit on all runs without labels.
    train, test = np.arange(0, 80, 2), np.arange(1, 80, 2)
    train, test = np.r_[train[:20], test[:20]], np.r_[train[20:], test[20:]]
    detector = get_method("run.lr_flatdep")().fit([runs[i] for i in train], labels[train])
    scores = detector.decision_function([runs[i] for i in test])
    ecod = get_method("run.pyod_ecod")().fit_score(runs)
    print(f"POST run.lr_flatdep: mean held-out score {scores[labels[test] == 1].mean():+.2f} on failed runs, "
          f"{scores[labels[test] == 0].mean():+.2f} on the rest")
    print(f"POST run.pyod_ecod: mean outlier score {ecod[labels == 1].mean():.2f} on failed runs, "
          f"{ecod[labels == 0].mean():.2f} on the rest")


if __name__ == "__main__":
    main()
