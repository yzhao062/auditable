"""Flagship demo: a payment agent that pays on a stale budget snapshot.

An agent approves a $4,200 vendor payment. At decision time it read a budget snapshot
captured six days earlier, under which the payment was in policy, and it commits the
payment through a reference ledger (the money moves). auditable captures one signed
record binding all three layers: the data the agent read plus a freshness signal, the
deciding model plus a trust signal, and the action. Six days later the live budget has
dropped below the payment amount. A replay re-derives that the payment is no longer
justified under the live state, and the gate EXECUTES a rollback through the ledger (the
money moves back), rather than printing a verdict and stopping.

Run:  python examples/payment_audit.py
"""
import time

from auditable import (
    Action,
    ActionGate,
    DependencySnapshot,
    ReferenceLedger,
    audit,
    replay,
    score_model_trust,
    score_snapshot_freshness,
)

SIX_DAYS = 6 * 24 * 60 * 60


def budget_policy(state, action):
    """A payment is justified if the recipient is allow-listed and the amount is within
    the remaining budget, evaluated against whatever dependency state is supplied."""
    if action.arguments["recipient"] not in state.get("allow_list", []):
        return False, f"Recipient {action.arguments['recipient']} not on the allow-list."
    if action.cost > state.get("budget_remaining", 0):
        return False, (
            f"${action.cost:,.0f} exceeds remaining budget "
            f"${state.get('budget_remaining', 0):,.0f}."
        )
    return True, "Within budget and allow-list."


def main():
    now = time.time()
    # The snapshot the agent read at decision time (six days old).
    snapshot = DependencySnapshot(
        state={"budget_remaining": 10000, "allow_list": ["acme-supplies", "globex"]},
        captured_at=now - SIX_DAYS,
    )
    action = Action("vendor_payment", {"recipient": "acme-supplies"}, cost=4200)

    # A reference rail. The agent commits the payment through the gate; money moves.
    ledger = ReferenceLedger(balance=10000)
    gate = ActionGate(ledger)

    # Capture the decision across all three layers in one signed record.
    with audit("vendor_payment", snapshot=snapshot) as d:
        d.read(invoice="INV-4471", vendor="acme-supplies")
        d.model("gpt-x", decision_basis="Invoice matches an approved PO; within budget.")
        d.act(action)
        # Attach the thin data and model signals before the record is signed.
        d.record.data.signal = score_snapshot_freshness(
            snapshot, now=now, max_age_seconds=SIX_DAYS / 2
        )
        d.record.model.signal = score_model_trust(d.record.model)
    record = d.record

    receipt = gate.commit(action)  # the agent actually pays
    print(f"Captured decision {record.record_id[:12]} | model {record.model.model_id}")
    print(
        f"  data signal:  stale={record.data.signal.stale} "
        f"anomaly={record.data.signal.anomaly_score}"
    )
    print(
        f"  model signal: trust={record.model.signal.trust} "
        f"flag={record.model.signal.flag}"
    )
    print(f"  agent paid ${action.cost:,.0f}; ledger balance now ${ledger.balance:,.0f}")

    # Six days later the live budget has dropped below the payment amount.
    live_state = {"budget_remaining": 3000, "allow_list": ["acme-supplies", "globex"]}
    verdict = replay(record, live_state=live_state, policy=budget_policy)
    outcome = gate.enforce(verdict, receipt=receipt)  # execute the routed fix

    print(f"Replay verdict: {verdict.action.value.upper()} | {verdict.reason}")
    print(
        f"  gate executed: {outcome.executed}; "
        f"ledger balance restored to ${ledger.balance:,.0f}"
    )


if __name__ == "__main__":
    main()
