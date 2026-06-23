"""LIVE pillar demo: capture a decision, replay it under live state, recover.

auditable attaches at three points in an agent's lifecycle. This is the LIVE
pillar: capture one consequential decision with the dependency state it relied on,
replay it under the state that is live now, and route and execute a fix. The other
two pillars are example_pre_lint_plan.py (PRE: lint a declared plan before deploy)
and example_post_rank_run.py (POST: rank a finished run). All three run over the
same typed two-layer decision graph.

The full chain in one record. An agent approves a $4,200 vendor payment. At decision time
it read a budget snapshot captured six days earlier, under which the payment was in
policy, and it commits the payment through a reference ledger (a reference / demo rail,
not a production payment rail; here the demo balance moves). auditable captures one signed
record binding all three layers: a data report (snapshot freshness), a model report
(decision-basis trust), and a harness report (a static cost cap). Six days later the live
budget has dropped below the payment amount. replay() re-derives that the payment is no
longer justified under the live state, and the gate EXECUTES a rollback through the ledger
(the demo balance moves back), rather than printing a verdict and stopping.

Run:  python examples/example_payment_audit.py
"""
import time

from auditable import (
    Action,
    ActionGate,
    CompoundReport,
    DataAuditor,
    DependencySnapshot,
    HarnessAuditor,
    ModelAuditor,
    ReferenceLedger,
    audit,
    replay,
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

    # A reference / demo rail (not a production payment rail). The agent commits the
    # payment through the gate; the demo balance is spent.
    ledger = ReferenceLedger(balance=10000)
    gate = ActionGate(ledger)

    # Capture the decision across all three layers in one signed record.
    with audit("vendor_payment", snapshot=snapshot) as d:
        d.read(invoice="INV-4471", vendor="acme-supplies")
        d.model("gpt-x", decision_basis="Invoice matches an approved PO; within budget.")
        d.act(action)
        # Three thin audit modules, each producing a signed report leaf.
        d.attach(DataAuditor(max_age_seconds=SIX_DAYS / 2).assess(snapshot, now=now))
        d.attach(ModelAuditor().assess(d.record.model))
        d.attach(HarnessAuditor(cost_cap=5000).assess(action))
        # v0.1 compound is a transparent bundle: per-stage reports plus a debug score.
        d.record.compound = CompoundReport.of(
            [d.record.data.report, d.record.model.report, d.record.harness.report]
        )
    record = d.record

    receipt = gate.commit(action)  # the agent commits the payment on the demo rail
    print(f"Captured decision {record.record_id[:12]} | model {record.model.model_id}")
    print(
        f"  data report:    {record.data.report.flag} "
        f"(score {record.data.report.score})  {record.data.report.reason}"
    )
    print(f"  model report:   {record.model.report.flag} (score {record.model.report.score})")
    print(f"  harness report: {record.harness.report.flag} (score {record.harness.report.score})")
    print(f"  compound (uncalibrated debug score): {record.compound.uncalibrated_score}")
    print(f"  agent paid ${action.cost:,.0f}; ledger balance now ${ledger.balance:,.0f}")

    # Six days later the live budget has dropped below the payment amount.
    live_state = {"budget_remaining": 3000, "allow_list": ["acme-supplies", "globex"]}
    verdict = replay(record, live_state=live_state, policy=budget_policy)
    outcome = gate.enforce_post_commit(verdict, receipt=receipt)  # execute the routed fix

    print(f"Replay verdict: {verdict.action.value.upper()} | {verdict.reason}")
    print(
        f"  gate executed: {outcome.executed}; "
        f"ledger balance restored to ${ledger.balance:,.0f}"
    )

    # The quotable line: recovery, not just observability.
    print(
        f"\nObservability would only have flagged this. auditable re-decided on the live "
        f"budget and reversed the payment: {outcome.executed}, balance restored to "
        f"${ledger.balance:,.0f}."
    )


if __name__ == "__main__":
    main()
