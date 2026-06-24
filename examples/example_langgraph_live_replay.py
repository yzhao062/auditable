"""Capture a real LangGraph agent, then replay and reverse its decision under drift.

This joins the two halves auditable shipped separately. ``example_langgraph_capture.py``
captures a real LangGraph run for POST analysis (observed dependency edges). This
example does that AND lowers the captured decision into a replayable record, so the
LIVE pillar (``replay`` + routed recovery) runs on a real captured agent rather than
a hand-built dict. Wrap once, get both:

    builder = instrument(StateGraph(State))   # 1) wrap; build / compile / invoke normally
    report = analyze_run(builder, adapter=builder)        # POST: rank the run, name the keystone
    records = builder.to_records(                          # LIVE: replayable records
        decisions={"approve": "vendor_payment"},
        action_args={"approve": {"recipient": "recipient"}},  # action payload from captured fields
        action_costs={"approve": "amount"},                   # action cost from the captured amount
    )

The scenario is a payment approver: ``fetch_budget`` reads the account budget,
``approve`` decides a vendor payment against it, and ``record_ledger`` rests on the
approval. The agent is real (a LangGraph ``StateGraph`` of plain function nodes); the
relied-on budget and the decision are captured from the run, not hand-written.

The drift is fault injection, stated plainly: after the run is captured, we lower the
live budget below the approved amount and ``replay`` re-decides on that live state.
This is a controlled post-capture change, not a fault the corpus supplied (real agent
corpora carry no intra-run state drift). It is a capability demo on one run, not the
comparative replay-versus-baseline experiment, which is not shown here. No API key, no
network.

Needs:  pip install "auditable[langgraph]"
Run:    python examples/example_langgraph_live_replay.py
"""
from typing import TypedDict

from auditable import Action, ActionGate, ReferenceLedger, analyze_run, replay
from auditable.integrations.langgraph import instrument

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise SystemExit('This example needs langgraph: pip install "auditable[langgraph]"') from exc


class PaymentState(TypedDict):
    invoice: str
    recipient: str
    amount: int
    budget: int
    allow_list: list
    approved: bool
    recorded: bool


def budget_policy(state, action):
    """Re-decide the payment: the action's recipient must be allow-listed, and the action's
    cost must fit the budget that is live now. The recipient and cost ride on the captured
    action (``to_records(action_args=..., action_costs=...)``); the allow-list and budget come
    from the dependency state. A cost-based check like this is why the lowered action must
    carry the real cost, not ``0``."""
    recipient = action.arguments.get("recipient")
    allow_list = state.get("allow_list", [])
    budget = state.get("budget", 0)
    if recipient not in allow_list:
        return False, f"recipient {recipient!r} not allow-listed"
    if action.cost > budget:
        return False, f"payment ${action.cost:,.0f} exceeds budget ${budget:,.0f}"
    return True, f"payment ${action.cost:,.0f} within budget ${budget:,.0f}"


def build_agent():
    """A real LangGraph payment approver of plain function nodes, instrumented once."""
    builder = instrument(StateGraph(PaymentState))

    def fetch_budget(state):
        """Read the account budget the approval will rest on (the keystone source).
        The allow-list is static config (read free in the initial state), so the budget
        is the single fetched dependency the approval rests on."""
        return {"budget": 10000}

    def approve(state):
        """Decide the payment against the budget and allow-list this node read."""
        ok = state["recipient"] in state["allow_list"] and state["amount"] <= state["budget"]
        return {"approved": ok}

    def record_ledger(state):
        """Rest on the approval the approve node wrote."""
        return {"recorded": state["approved"]}

    builder.add_node("fetch_budget", fetch_budget)
    builder.add_node("approve", approve)
    builder.add_node("record_ledger", record_ledger)
    builder.add_edge(START, "fetch_budget")
    builder.add_edge("fetch_budget", "approve")
    builder.add_edge("approve", "record_ledger")
    builder.add_edge("record_ledger", END)
    return builder


def main():
    builder = build_agent()
    graph = builder.compile()
    final = graph.invoke(
        {
            "invoice": "INV-4471",
            "recipient": "acme-supplies",
            "amount": 4200,
            "budget": 0,
            "allow_list": ["acme-supplies"],
            "approved": False,
            "recorded": False,
        }
    )
    print(f"agent run: approved={final['approved']}\n")

    # POST: the same captured run, ranked. fetch_budget is the structural keystone.
    report = analyze_run(builder, adapter=builder)
    print(report)
    k = report.keystone
    if k is not None:
        print(f"\nPOST keystone: step {k.idx} ({k.node_attrs.get('langgraph_node')})\n")

    # LIVE: lower the captured approval into a replayable record. The relied-on budget,
    # amount, recipient, and allow-list become its DependencySnapshot; action_args and
    # action_costs lift the recipient and the real cost onto the action itself, so the
    # lowered record is the executable payment, not a state-only stub.
    record = builder.to_records(
        decisions={"approve": "vendor_payment"},
        action_args={"approve": {"recipient": "recipient"}},
        action_costs={"approve": "amount"},
    )[0]
    snap = record.data.snapshot.state
    print("captured decision record:")
    print(f"  action:   {record.harness.action_type} -> {record.harness.arguments}")
    print(f"  cost:     ${record.harness.cost:,.0f} (the captured amount)")
    print(f"  relied-on budget: ${snap['budget']:,}; amount: ${snap['amount']:,}\n")

    # Commit the captured payment through a reference ledger (a demo rail, not a production
    # rail). The action committed is the one the record carries, not a hand-rebuilt one.
    ledger = ReferenceLedger(balance=10000)
    gate = ActionGate(ledger)
    pay = Action(record.harness.action_type, dict(record.harness.arguments), cost=record.harness.cost)
    receipt = gate.commit(pay)
    print(f"committed ${pay.cost:,.0f}; ledger balance now ${ledger.balance:,.0f}")

    # Fault injection: after capture, the live budget drops below the approved amount.
    # replay re-decides on the live state and routes a rollback; the gate executes it.
    live_state = {**snap, "budget": 3000}
    verdict = replay(record, live_state=live_state, policy=budget_policy)
    gate.enforce_post_commit(verdict, receipt=receipt)
    print(f"replay under live budget $3,000: {verdict.action.value.upper()} -- {verdict.reason}")
    print(f"recovery executed; ledger balance restored to ${ledger.balance:,.0f}")


if __name__ == "__main__":
    main()
