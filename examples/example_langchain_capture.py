"""Plug in a real LangChain runnable: capture its run into the typed decision graph.

What this shows, in plain terms: you attach one `auditable` callback handler to
an actual LangChain runnable you already built, run it as usual, and get back the
typed two-layer decision graph for that run. Chain/runnable and model callbacks
become decision steps, tool callbacks become tool-call steps, and `analyze_run`
ranks the run and names the keystone, exactly as the POST pillar does for corpus
traces.

This is a real-agent capture path: no hand-written plan dict, no corpus fixture,
no network, and no API key. The only change to your runnable call is the callback,
followed by one analysis line:

    handler = LangChainCallbackHandler()                         # 1) make once
    chain.invoke(request, config={"callbacks": [handler]})       # 2) attach
    report = analyze_run(handler, adapter=handler)               # 3) analyze

The scenario is a small payment approver. `fetch_budget` prepares the amount and
budget, the `approve_payment` tool decides whether the payment fits, and
`record_ledger` records the result. LangChain's RunnableSequence callback tags
declare both handoffs, and the adjacent output/input payload digests match, so
those two State-A value-flow edges are graded OBSERVED.

Honesty holds in the output. Callbacks expose State A -- runnable/model reasoning
and tool flow -- but not State B, the live version or contents of the external
systems behind a tool. This example therefore makes no observed claim about an
account database; its OBSERVED edges mean exact callback-level value handoffs,
not external-state reads or precise causality. The score is a triage ranking, not
a calibrated probability. The corpus-scale GRADE numbers (arXiv:2606.22741) are
not produced here; this is a capability demo on one offline run.

Needs the graph + langchain extras:  pip install "auditable[graph,langchain]"
Run:  python examples/example_langchain_capture.py
"""
try:
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool
except ImportError:
    print(
        'example skipped: install langchain-core with pip install "auditable[graph,langchain]"'
    )
    raise SystemExit(0)

from auditable import analyze_run
from auditable.integrations.langchain import LangChainCallbackHandler


def fetch_budget(request):
    """Prepare the payment and the illustrative budget it will be checked against."""
    return {"amount": request["amount"], "budget": 5_000}


@tool
def approve_payment(amount: int, budget: int) -> bool:
    """Approve a payment only when it fits within the supplied budget."""
    return amount <= budget


def record_ledger(approved):
    """Record the tool's approval result."""
    return {"approved": approved, "paid": approved}


def main():
    chain = (
        RunnableLambda(fetch_budget, name="fetch_budget")
        | approve_payment
        | RunnableLambda(record_ledger, name="record_ledger")
    ).with_config(run_name="payment_chain")

    handler = LangChainCallbackHandler()
    final = chain.invoke({"amount": 4_200}, config={"callbacks": [handler]})
    print(f"agent result: approved={final['approved']}, paid={final['paid']}\n")

    # One public call: the callback handler is its own source and adapter.
    report = analyze_run(handler, adapter=handler)
    print(report)

    print("\nDependency edges captured from the live run:")
    steps = {step.idx: step for step in handler.steps()}
    for step in steps.values():
        for edge in step.deps:
            source = steps[edge.src_idx]
            print(
                f"  {step.agent} -> {source.agent} "
                f"({edge.grade.value}, {edge.evidence['relation']})"
            )

    keystone = report.keystone
    if keystone is not None:
        print(
            f"\nKeystone: step {keystone.idx} "
            f"({keystone.node_attrs.get('runnable') or keystone.label}) -- "
            "the approval and ledger result both rest on this captured handoff, "
            "so it is the first step to review."
        )


if __name__ == "__main__":
    main()
