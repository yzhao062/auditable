"""One method from each family on synthetic traces, with no benchmark data.

Run from anywhere after ``pip install ".[tabular-ad]"`` in a checkout:

    python examples/example_methods.py

The traces come from ``auditable.datasets.make_traces``: each run is a sequence of agent
steps, each step relies on one or two recent steps, and a failed run contains one stale
read, a step that relies on state from much earlier in the run. The numbers printed
describe this toy data only.
"""
import numpy as np

from auditable.datasets import make_traces
from auditable.featurize import AgentConfig
from auditable.methods import get_method, list_methods


def main():
    runs, labels, mistakes = make_traces(n_runs=80, contamination=0.5, random_state=0)
    failed = [i for i in range(len(runs)) if labels[i] == 1]

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
    alarms = get_method("online.span_z")().step_alarm_scores(runs[failed[0]])
    print(f"LIVE online.span_z: peak alarm at step {int(np.argmax(alarms))}, "
          f"stale read at step {mistakes[failed[0]]}")

    # POST, step level: rank the steps of each failed run; how often is the stale read first?
    locator = get_method("step.dep_anomaly")()
    hits = sum(locator.rank(runs[i])[0] == mistakes[i] for i in failed)
    print(f"POST step.dep_anomaly: stale read ranked first in {hits} of {len(failed)} failed runs")

    # POST, run level: a supervised detector fit on half the runs and scored on the other
    # half, next to an unsupervised one fit on all runs without labels.
    detector = get_method("run.lr_flatdep")().fit(runs[:40], labels[:40])
    scores = detector.decision_function(runs[40:])
    held_out = labels[40:]
    ecod = get_method("run.pyod_ecod")().fit_score(runs)
    print(f"POST run.lr_flatdep: mean held-out score {scores[held_out == 1].mean():+.2f} on failed runs, "
          f"{scores[held_out == 0].mean():+.2f} on the rest")
    print(f"POST run.pyod_ecod: mean outlier score {ecod[labels == 1].mean():.2f} on failed runs, "
          f"{ecod[labels == 0].mean():.2f} on the rest")


if __name__ == "__main__":
    main()
