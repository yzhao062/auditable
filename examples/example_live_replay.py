"""LIVE pillar demo: capture a payment decision, replay it live, reverse it.

auditable attaches at three points in an agent's lifecycle. This is the LIVE
pillar: capture one consequential decision with the dependency state it relied
on, replay it under the state that is live now, and execute a fix. The companion
pillars are example_pre_lint_plan.py (PRE) and example_post_rank_run.py (POST). No
API key, no network. This is a capability demo, NOT a benchmark; it carries no
percentage.

The story in one record: an agent approves a vendor payment against a six-day-old
budget snapshot under which it was in policy, and commits the payment through a
reference ledger (a demo rail, not a production payment rail). The live budget has
since dropped below the amount. replay() re-decides on the live state and routes a
rollback; the ActionGate EXECUTES it through the ledger, restoring the balance.

Run:  python examples/example_live_replay.py
"""
import time

from auditable import (
    Action,
    ActionGate,
    DependencySnapshot,
    ReferenceLedger,
    audit,
    replay,
)

SIX_DAYS = 6 * 24 * 60 * 60


def budget_policy(state, action):
    """Justified if the recipient is allow-listed and the amount fits the budget."""
    if action.arguments["recipient"] not in state.get("allow_list", []):
        return False, f"Recipient {action.arguments['recipient']} not allow-listed."
    if action.cost > state.get("budget_remaining", 0):
        return False, (
            f"${action.cost:,.0f} exceeds remaining budget "
            f"${state.get('budget_remaining', 0):,.0f}."
        )
    return True, "Within budget and allow-list."


def main():
    now = time.time()
    # The budget snapshot the agent read at decision time (six days old); the
    # payment was in policy against it.
    snapshot = DependencySnapshot(
        state={"budget_remaining": 10000, "allow_list": ["acme-supplies"]},
        captured_at=now - SIX_DAYS,
    )
    action = Action("vendor_payment", {"recipient": "acme-supplies"}, cost=4200)
    ledger = ReferenceLedger(balance=10000)  # reference / demo rail
    gate = ActionGate(ledger)

    # Capture the decision with the snapshot it relied on, then commit the payment.
    with audit("vendor_payment", snapshot=snapshot) as d:
        d.read(invoice="INV-4471", budget_remaining=10000)
        d.model("gpt-x", decision_basis="Invoice within the $10,000 remaining budget.")
        d.act(action)
    receipt = gate.commit(action)
    print(f"Committed ${action.cost:,.0f}; ledger balance now ${ledger.balance:,.0f}.")

    # Six days on, the live budget has drifted below the amount. Replay re-decides
    # on the live state and the gate EXECUTES the rollback (recovery, not a flag).
    live_state = {"budget_remaining": 3000, "allow_list": ["acme-supplies"]}
    verdict = replay(d.record, live_state=live_state, policy=budget_policy)
    gate.enforce_post_commit(verdict, receipt=receipt)
    print(f"Replay verdict: {verdict.action.value.upper()} -- {verdict.reason}")
    print(f"Balance restored to ${ledger.balance:,.0f}.")


if __name__ == "__main__":
    main()
