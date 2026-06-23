"""Plug in a real LangGraph agent: capture its run into the typed decision graph.

What this shows, in plain terms: you point `auditable` at an actual LangGraph
`StateGraph` you already built, run it as usual, and get back the typed two-layer
decision graph for that run, with the dependency layer captured as real OBSERVED
edges over the state channels each node read and wrote. Then `analyze_run` ranks
the run and names the keystone, exactly as the POST pillar does for corpus traces.

This is the first real-agent capture path: no hand-written plan dict, no corpus
fixture. The only change to your code is two lines, wrapping the builder and
pulling the report:

    builder = instrument(StateGraph(State))   # 1) wrap once, build normally
    ...
    report = analyze_run(builder, adapter=builder)   # 2) after invoke

The scenario is a small payment approver: read a budget, approve a payment
against it, then record the result. `approve` reads the `budget` channel that
`fetch_budget` wrote, and `record` reads the `approved` channel that `approve`
wrote, so both downstream steps rest on the budget read. `fetch_budget` is the
structural keystone: a stale or wrong budget there propagates to the approval and
the ledger entry.

Honesty holds in the output. The edges are OBSERVED as a channel-level
read-after-committed-write touch (the proxy logged the node reading the channel
and the prior node writing it), matched across LangGraph's superstep barrier. It
is an observed touch match, not a precise causal claim, and the score is a triage
ranking, not a calibrated probability. The corpus-scale GRADE numbers
(arXiv:2606.22741) are not produced here; this is a capability demo on one run.

Needs the langgraph extra:  pip install "auditable[langgraph]"
Run:  python examples/example_langgraph_capture.py
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from auditable import analyze_run
from auditable.integrations.langgraph import instrument


class PaymentState(TypedDict):
    budget: int
    amount: int
    approved: bool
    paid: bool
    log: Annotated[list, operator.add]  # a reducer channel: the running audit log


def fetch_budget(state):
    """Read the account budget the approval will rely on (the keystone source)."""
    return {"budget": 5_000, "log": ["fetched budget 5000"]}


def approve_payment(state):
    """Decide against the budget channel that fetch_budget wrote."""
    ok = state["amount"] <= state["budget"]
    return {"approved": ok, "log": [f"approved={ok} for amount {state['amount']}"]}


def record_ledger(state):
    """Record the result, resting on the approval channel that approve wrote."""
    return {"paid": state["approved"], "log": ["ledger updated"]}


def main():
    builder = instrument(StateGraph(PaymentState))
    builder.add_node("fetch_budget", fetch_budget)
    builder.add_node("approve_payment", approve_payment)
    builder.add_node("record_ledger", record_ledger)
    builder.add_edge(START, "fetch_budget")
    builder.add_edge("fetch_budget", "approve_payment")
    builder.add_edge("approve_payment", "record_ledger")
    builder.add_edge("record_ledger", END)

    graph = builder.compile()
    final = graph.invoke({"budget": 0, "amount": 4_200, "approved": False, "paid": False, "log": []})
    print(f"agent result: approved={final['approved']}, paid={final['paid']}\n")

    # One public call: the instrumented builder is its own source and adapter.
    report = analyze_run(builder, adapter=builder)
    print(report)

    print("\nObserved dependency edges captured from the live run:")
    steps = {s.idx: s for s in builder.to_steps()}
    for s in steps.values():
        for e in s.deps:
            src = steps[e.src_idx]
            print(
                f"  {s.node_attrs['langgraph_node']} -> {src.node_attrs['langgraph_node']} "
                f"via channel '{e.resource.resource_id}' ({e.grade.value}, {e.evidence['mode']})"
            )

    k = report.keystone
    if k is not None:
        print(
            f"\nKeystone: step {k.idx} ({k.node_attrs.get('langgraph_node')}) -- "
            f"the approval and the ledger entry both rest on this read, so a stale "
            f"budget here is the first thing to review."
        )


if __name__ == "__main__":
    main()
