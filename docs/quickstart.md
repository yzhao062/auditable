# Quickstart

This page installs `auditable` and shows the three smallest runnable snippets, one per lifecycle pillar. For the narrative behind the three pillars, see [Lifecycle](lifecycle.md).

## Install

```bash
pip install auditable
```

The core is dependency-free. The graph layer needs the `graph` extra (NetworkX):

```bash
pip install "auditable[graph]"
```

The `graph` extra is required by `analyze_plan` (PRE), `analyze_run` (POST), and any graph projection. The REAL-TIME capture, replay, and gate path runs on the core alone. The library is torch-free; only NetworkX is pulled in by the extra.

## REAL-TIME: Replay Under Live State, Then Recover

This is the sharpest demo. An agent approves a payment against a budget snapshot, the budget moves, and `replay` re-derives that the payment no longer holds, so the gate reverses it through a reference rail. The full runnable script is [`examples/payment_audit.py`](https://github.com/yzhao062/auditable/blob/main/examples/payment_audit.py).

```python
import time
from auditable import (
    Action, DependencySnapshot, ActionGate, ReferenceLedger, audit, replay,
)


def budget_policy(state, action):
    """Justified when the recipient is allow-listed and the amount fits the budget."""
    if action.arguments["recipient"] not in state.get("allow_list", []):
        return False, "Recipient not on the allow-list."
    if action.cost > state.get("budget_remaining", 0):
        return False, "Amount exceeds the remaining budget."
    return True, "Within budget and allow-list."


now = time.time()
snapshot = DependencySnapshot(
    state={"budget_remaining": 10000, "allow_list": ["acme-supplies"]},
    captured_at=now - 6 * 24 * 60 * 60,  # six days old
)
action = Action("vendor_payment", {"recipient": "acme-supplies"}, cost=4200)

ledger = ReferenceLedger(balance=10000)
gate = ActionGate(ledger)

with audit("vendor_payment", snapshot=snapshot) as d:
    d.read(invoice="INV-4471")
    d.model("gpt-x", decision_basis="Invoice matches an approved PO; within budget.")
    d.act(action)
record = d.record

receipt = gate.commit(action)  # the agent actually pays

# Later the live budget has dropped below the payment amount.
live_state = {"budget_remaining": 3000, "allow_list": ["acme-supplies"]}
verdict = replay(record, live_state=live_state, policy=budget_policy)
outcome = gate.enforce_post_commit(verdict, receipt=receipt)

print(verdict.action.value, "->", outcome.executed)  # rollback -> rolled_back
```

`ReferenceLedger` is an in-process reference rail for demos and tests (commit spends, compensate refunds). It is not a production payment rail and moves no real money. See [REAL-TIME Replay and Recovery](realtime-replay.md) for the verdicts and gate semantics.

## PRE: Lint the Plan Before Deploy

PRE runs read-only lints over a declared plan before any step executes. The entry is `analyze_plan`, imported from `auditable.graph.pre` (it is not a top-level `auditable` export). This needs the `graph` extra. The full runnable script is [`examples/analyze_plan.py`](https://github.com/yzhao062/auditable/blob/main/examples/analyze_plan.py).

```python
from auditable.graph.pre import analyze_plan
from auditable.graph.adapters import declared_plan_v1

plan = {
    "nodes": [
        # 0: fetch the live pricing policy (a volatile resource).
        {"idx": 0, "agent": "planner", "kind": "tool_call",
         "writes": ["pricing_policy"]},
        # 1: a decision reads the volatile pricing_policy, unpinned and not revalidated.
        {"idx": 1, "agent": "planner", "kind": "decision", "control_preds": [0],
         "reads": [{"id": "pricing_policy", "producer": 0, "volatile": True}]},
        # 2: a consequential write to 'ledger' with the volatile read reaching it
        #    and no intervening re-read.
        {"idx": 2, "agent": "executor", "kind": "tool_call", "control_preds": [1],
         "reads": [{"id": "pricing_policy", "producer": 0, "volatile": True}],
         "writes": ["ledger"]},
        # 3: granted scope over 'admin_api' but only read 'pricing_policy'.
        {"idx": 3, "agent": "executor", "kind": "decision", "control_preds": [2],
         "reads": [{"id": "pricing_policy", "producer": 0}],
         "scope": ["pricing_policy", "admin_api"]},
    ]
}

report = analyze_plan(plan, adapter=declared_plan_v1)
print(report)  # all four lints, the execution keystone, and the preflight coverage report
```

The rendered report shows the four lints firing, the execution-topology keystone (labeled a structural chokepoint, distinct from the POST keystone), and the Preflight Coverage Report, with dependency-state risk withheld at PRE. See [PRE Rules](pre-rules.md) for what each lint flags.

## POST: Rank a Finished Run, Find the Keystone

POST reads a recorded run and ranks every step by structural blast share. The entry is `analyze_run`, a top-level export, and it also needs the `graph` extra. The full runnable script is [`examples/analyze_run.py`](https://github.com/yzhao062/auditable/blob/main/examples/analyze_run.py).

```python
from auditable import analyze_run
from auditable.graph.adapters import tau_bench_prior_db_reads_v1

# A recorded trajectory: read one reservation, then two writes that rest on it.
messages = [
    {"role": "assistant", "content": "Pulling up the reservation.",
     "tool_calls": [{"function": {"name": "get_reservation_details"}}]},
    {"role": "tool", "name": "get_reservation_details",
     "content": '{"reservation_id": "ZFA04Y", "cabin": "economy", "baggages": 0}'},
    {"role": "assistant", "content": "Rebooking onto the morning flights.",
     "tool_calls": [{"function": {"name": "update_reservation_flights"}}]},
    {"role": "tool", "name": "update_reservation_flights", "content": "ok"},
    {"role": "assistant", "content": "Adding the checked bag.",
     "tool_calls": [{"function": {"name": "update_reservation_baggages"}}]},
    {"role": "tool", "name": "update_reservation_baggages", "content": "ok"},
]

report = analyze_run(messages, adapter=tau_bench_prior_db_reads_v1)
print(report)  # ranked steps, the keystone, coverage, and the honesty notes
```

The score is an uncalibrated triage ranking, and the corpus write-to-read edges are modeled (a conservative prior-read upper bound, not a causal label). Both caveats appear in the report's own notes. See [POST Analysis](post-analysis.md) for how to read the report.

## Using a Single Layer

Each span auditor is usable on its own, with no agent and no chain. The data auditor scores a dependency snapshot and returns a signed `Report`. The full runnable script is [`examples/standalone_report.py`](https://github.com/yzhao062/auditable/blob/main/examples/standalone_report.py).

```python
import time
from auditable import DataAuditor, DependencySnapshot

snapshot = DependencySnapshot(
    state={"budget_remaining": 1000}, captured_at=time.time() - 7 * 24 * 60 * 60,
)
report = DataAuditor(max_age_seconds=24 * 60 * 60).assess(snapshot)
print(report.flag, report.score, report.reason)  # stale 1.0 ...
```

The standalone auditors are inputs to the record; the composed full-chain record is the main line. See [Architecture](architecture.md) for how the three spans compose into one signed record.
