# LIVE: Replay and Recovery Reference

The LIVE pillar captures one consequential decision with the dependency state it relied on, re-derives whether that decision still holds under the state that is live now, and executes a routed fix through a rail. This page is the reference for the capture API, replay semantics, the four verdicts, the policy contract, and gate execution. The capture, replay, and gate path runs on the core install; no extra is required.

## Capture: the `audit` Context Manager

`audit(action_type, *, snapshot, sink=None)` is a context manager that captures one decision. It yields a `Decision` handle you fill in, then signs the record and appends it to a sink when the block exits.

```python
from auditable import audit, Action, DependencySnapshot

snapshot = DependencySnapshot(
    state={"budget_remaining": 10000, "allow_list": ["acme-supplies"]},
)
with audit("vendor_payment", snapshot=snapshot) as d:
    d.read(invoice="INV-4471")
    d.model("gpt-x", decision_basis="Invoice matches an approved PO; within budget.")
    d.act(Action("vendor_payment", {"recipient": "acme-supplies"}, cost=4200))
record = d.record
```

The `Decision` handle has four chained methods:

| Method | What it records |
|---|---|
| `read(**inputs)` | What the agent read (merged into the data span inputs). |
| `model(model_id, decision_basis="", output=None)` | Which model produced the output, and its stated basis. |
| `act(action)` | The action the agent executed (type, arguments, cost). |
| `attach(report)` | Routes a leaf `Report` to its stage span (data, model, or harness). |

The `snapshot` is a `DependencySnapshot`: the versioned dependency state the decision relied on (for example, a remaining budget, an allow-list version, a policy id), with an optional `captured_at` timestamp. A stale snapshot is the failure mode replay is built to surface. When the block exits, the record is signed and chained by the sink; `sink` defaults to an in-process `MemorySink`. See [Architecture](architecture.md) for the sinks and the record shape.

## Replay: Re-Derive the Decision Under Live State

`replay(record, *, live_state, policy) -> Verdict` re-derives whether a recorded decision still holds under the live dependency state. The agent acted under `record.data.snapshot.state`; replay re-evaluates the same action against `live_state`.

Replay is pure. It mutates nothing, and it executes nothing. It hands the policy a deep copy of both the live state and the action, so a mutating policy cannot alter the signed record. It returns a `Verdict` and leaves all side effects to the gate.

The re-derivation logic is direct. Replay first asks the policy whether the action is justified under the live state. If yes, the verdict is `ALLOW`. If not, replay asks whether the action was justified under the original snapshot. If it was justified on the snapshot but not on live state, the action relied on stale or drifted state, and the verdict is `ROLLBACK`. If it was justified under neither, the verdict is `BLOCK`.

## The Four Routed Verdicts

A `Verdict` carries a `FixAction`, a `justified` flag, a `reason`, and the `record_id`. The `FixAction` is one of four:

| Verdict | Returned when | Meaning |
|---|---|---|
| `ALLOW` | The action is justified under live state. | The decision still holds; let it stand. |
| `ROLLBACK` | Justified on the snapshot, but not on live state. | The decision relied on stale or drifted state; reverse it. |
| `BLOCK` | Justified under neither the snapshot nor live state. | The decision does not hold even on its own snapshot. |
| `HUMAN_REVIEW` | The policy raised `ReplayUndecidable`. | The policy could not decide; route to a human. |

The distinction between `ROLLBACK` and `BLOCK` is the useful one. `ROLLBACK` is the stale-state case: the decision was sound when it was made and the world moved underneath it. `BLOCK` is the case where the decision did not hold even on the state it read.

## The Policy Contract

A `Policy` is any callable with the signature `(state, action) -> (justified, reason)`. It returns a boolean and a short reason string. You write the policy; it encodes what "still justified" means for your decision.

```python
def budget_policy(state, action):
    if action.arguments["recipient"] not in state.get("allow_list", []):
        return False, "Recipient not on the allow-list."
    if action.cost > state.get("budget_remaining", 0):
        return False, "Amount exceeds the remaining budget."
    return True, "Within budget and allow-list."
```

A policy that cannot decide deterministically under a given state raises `ReplayUndecidable`. When it does, replay does not guess: it returns a `HUMAN_REVIEW` verdict carrying the exception message. This keeps an undecidable case out of the automated allow and block paths.

## Execute the Fix: `ActionGate` Over a `Rail`

A verdict on its own is a recommendation. `ActionGate` executes it through a rail, so a routed fix reverses a committed action rather than printing a suggestion. The gate is constructed over a `Rail` and exposes three methods.

`commit(action)` runs the action through the rail and returns a receipt.

`enforce_pre_commit(verdict, action)` runs before the action. `ALLOW` allows it, `HUMAN_REVIEW` holds it, and `ROLLBACK` or `BLOCK` block it (the action has not run yet, so there is nothing to reverse).

`enforce_post_commit(verdict, *, receipt=None)` runs after the action committed through the rail. `ALLOW` lets the committed action stand, `HUMAN_REVIEW` holds, and `ROLLBACK` or `BLOCK` call `rail.compensate(receipt)` to reverse the committed action. The gate preserves which verdict drove the compensation: a `ROLLBACK` reports `rolled_back`, a post-commit `BLOCK` reports `reversed`.

Each call returns a `GateOutcome` with the `fix` that was applied, an `executed` string (for example, `allowed`, `blocked`, `held`, `rolled_back`, `reversed`), and a `detail`. When a post-commit `ROLLBACK` or `BLOCK` arrives with no receipt, the gate cannot reverse the action and reports `compensation_unavailable`. That outcome is observability, not control: it records that a fix was routed but could not execute, rather than silently dropping it.

## The Rail Abstraction

`Rail` is a protocol with two methods, `commit(action)` and `compensate(receipt)`. Any commit/compensate backend satisfies it (a payment rail, a record store, a ledger), so the gate is rail-neutral.

`ReferenceLedger` is the in-process reference rail that ships for demos and tests: `commit` spends from a balance and returns a receipt, and `compensate` refunds the receipt amount. It is a reference and demo rail, and it moves no real money; a production integration supplies its own `Rail`. The [`examples/payment_audit.py`](https://github.com/yzhao062/auditable/blob/main/examples/payment_audit.py) script shows the full capture, replay, and post-commit rollback path over `ReferenceLedger`.
