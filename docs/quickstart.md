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

The `graph` extra is required by `analyze_plan` (PRE), `analyze_run` (POST), and any graph projection. The LIVE capture, replay, and gate path runs on the core alone. The library is torch-free; only NetworkX is pulled in by the extra.

## LIVE: Replay Under Live State, Then Recover

This is the sharpest demo. An agent approves a payment against a budget snapshot, the budget moves, and `replay` re-derives that the payment no longer holds, so the gate reverses it through a reference rail. The full runnable script is [`examples/example_payment_audit.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_payment_audit.py).

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

`ReferenceLedger` is an in-process reference rail for demos and tests (commit spends, compensate refunds). It is not a production payment rail and moves no real money. See [LIVE Replay and Recovery](realtime-replay.md) for the verdicts and gate semantics.

## PRE: Lint the Plan Before Deploy

PRE runs read-only lints over a declared plan before any step executes. The entry is `analyze_plan`, imported from `auditable.graph.pre` (it is not a top-level `auditable` export). This needs the `graph` extra. The full runnable script is [`examples/example_pre_lint_plan.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_pre_lint_plan.py).

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

POST reads a recorded run and ranks every step by structural blast share. The entry is `analyze_run`, a top-level export, and it also needs the `graph` extra. The full runnable script is [`examples/example_post_rank_run.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_post_rank_run.py).

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

## Render a Report as Markdown

`print(report)` gives terse indented plaintext. For a clean Markdown form to paste into a pull request, an issue, or a design doc, call `render_report(report)`. It is a top-level export that dispatches on type: a PRE `PreReport` and a POST `AnalysisReport` each render through the matching renderer. The same string is available as a method, `report.to_markdown()`. The renderer formats the typed fields the report already carries; it computes nothing new, and it is dependency-free (standard library only, no NetworkX or templating engine).

```python
from auditable import render_report

# `report` is the POST AnalysisReport from the section above.
print(render_report(report))          # the dispatcher form
print(report.to_markdown(level=2))    # the method form; level sets the heading depth
```

The POST run above renders to this Markdown:

```markdown
# Auditable POST Report: Structural Risk Analysis

- stage: POST (runtime, after the run completes)
- adapter: tau_bench_prior_db_reads_v1
- completeness: complete
- state: scored
- steps: 6
- coverage: 2 dependency edge(s), rho=0.133, observed=100%, grades: observed=2

## Keystone (Blast-Radius)

- step 1 [tool_call get_reservation_details]: structural risk 0.400 (2 of 5 other steps transitively rest on it)
- grounding: n/a (this step states no checkable model basis)

## What Is Risky on the Graph

ranked decisions (structural blast share):

| score | step | kind | label |
| --- | --- | --- | --- |
| 0.400 | 1 | tool_call | tool_call get_reservation_details |
| 0.000 | 0 | decision | decision |
| 0.000 | 2 | decision | decision |
| 0.000 | 3 | tool_call | tool_call update_reservation_flights |
| 0.000 | 4 | decision | decision |
| 0.000 | 5 | tool_call | tool_call update_reservation_baggages |

## Recommended Action

- triage the keystone step first (it carries the highest blast share); the score is a ranking signal, not a calibrated probability

## Notes

- all 2 observed dependency edge(s) are MODELED: a conservative prior-read upper bound over the observed reads, not a causal label.
- the structural signal is the dependency-DAG blast structure (how much of the run transitively rests on a step); it is a ranking / triage signal, not calibrated.
- grounding is empty: no step states a checkable model basis (a corpus tool trace does not). It lights up on records that carry a basis, such as auditable's own runs.
```

The blast-share scores are within-run structural fractions (how much of this one run transitively rests on a step), not a benchmark or a calibrated probability. A withheld score renders as `n/a`, never `0.000`, so a withheld value never reads as no risk. A PRE `PreReport` renders the same way through `render_report(pre_report)`, with an execution-topology keystone section in place of the blast-radius one.

## End-to-End: One Payment Across All Three Pillars

The snippets above show one pillar each. [`examples/example_end_to_end.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_end_to_end.py) carries a single vendor payment through PRE, then LIVE, then POST on one dataset and one state, and closes by printing the aggregate audit report through `AuditReport.to_markdown`. Run it with one command:

```bash
python examples/example_end_to_end.py
```

The amount is one real value sampled from the ULB credit-card dataset; the only constructed dimension is a temporal budget drift (the snapshot budget covers the payment, the live budget has since dropped below it). See [Lifecycle](lifecycle.md#one-payment-across-all-three) for the narrative.

## Using a Single Layer

Each span auditor is usable on its own, with no agent and no chain. The data auditor scores a dependency snapshot and returns a signed `Report`. The full runnable script is [`examples/example_standalone_report.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_standalone_report.py).

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
