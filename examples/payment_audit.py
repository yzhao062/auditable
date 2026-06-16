"""Flagship demo: a payment agent that pays on a stale budget snapshot.

An agent approves a $4,200 vendor payment. At decision time it read a budget snapshot
captured six days earlier, under which the payment was in policy. The live budget has
since dropped below the payment amount. auditable captured the snapshot the agent
relied on, and a replay re-derives that the payment was not justified under the live
state, then routes a rollback with the reason.

Run:  python examples/payment_audit.py
"""
from auditable import Action, DependencySnapshot, audit, replay


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
    # The snapshot the agent read at decision time (six days old).
    snapshot = DependencySnapshot(
        state={"budget_remaining": 10000, "allow_list": ["acme-supplies", "globex"]},
        captured_at=0.0,
    )

    # Capture the decision across all three layers in one record.
    with audit("vendor_payment", snapshot=snapshot) as d:
        d.read(invoice="INV-4471", vendor="acme-supplies")
        d.model("gpt-x", decision_basis="Invoice matches an approved PO; within budget.")
        d.act(Action("vendor_payment", {"recipient": "acme-supplies"}, cost=4200))

    record = d.record
    print(
        f"Captured decision {record.record_id[:12]} "
        f"(cost ${record.harness.cost:,.0f}, model {record.model.model_id})"
    )

    # Six days later the live budget has dropped below the payment amount.
    live_state = {"budget_remaining": 3000, "allow_list": ["acme-supplies", "globex"]}

    verdict = replay(record, live_state=live_state, policy=budget_policy)
    print(f"Replay verdict: {verdict.action.value.upper()}  justified={verdict.justified}")
    print(f"Reason: {verdict.reason}")


if __name__ == "__main__":
    main()
