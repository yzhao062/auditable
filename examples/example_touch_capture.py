"""Capture any agent loop with the generic touch recorder (no framework needed).

What this shows, in plain terms: when your agent is a plain tool-calling loop
(raw OpenAI or Anthropic calls, your own scheduler, anything), you can still
capture the typed two-layer decision graph by declaring, per step, which
resources it read and which it wrote. The `TouchRecorder` records those touches
and matches a read to the writer it relied on, producing the same OBSERVED
dependency edges and the same `analyze_run` keystone the framework adapters give.

The pattern is one `with` block per consequential step:

    rec = TouchRecorder()
    with rec.step(agent="decide", kind="decision") as st:
        st.reads("market", "price", "AAPL")     # what this step relied on
        st.writes("plan", "order")              # what this step produced
    report = analyze_run(rec, adapter=rec)

The scenario is a tiny trading loop: look up a price, decide an order against that
price, then place the order. The decision reads the price the lookup wrote, and
the order placement reads the decision, so the price lookup is the keystone: a
stale price there propagates to the order.

Honesty holds in the output. Each edge is OBSERVED as a resource read-after-write
match over what you declared, not a precise causal claim, and the score is a
triage ranking, not a calibrated probability. Declaring touches is manual here;
the LangGraph adapter (example_langgraph_capture.py) derives them automatically
from state channels.

Run:  python examples/example_touch_capture.py
"""
from auditable import analyze_run
from auditable.graph.touch import TouchRecorder


def main():
    rec = TouchRecorder()

    # 1) look up a price (the keystone source): a tool step that writes the price.
    with rec.step(agent="lookup_price", kind="tool_call") as st:
        st.writes("market", "price", "AAPL")

    # 2) decide an order against the price it just read.
    with rec.step(agent="decide_order", kind="decision") as st:
        st.reads("market", "price", "AAPL")
        st.writes("plan", "order")

    # 3) place the order, resting on the decision.
    with rec.step(agent="place_order", kind="tool_call") as st:
        st.reads("plan", "order")

    report = analyze_run(rec, adapter=rec)
    print(report)

    print("\nObserved dependency edges captured from the loop:")
    steps = {s.idx: s for s in rec.to_steps()}
    for s in steps.values():
        for e in s.deps:
            print(
                f"  {s.agent} -> {steps[e.src_idx].agent} "
                f"via {e.resource.namespace}:{e.resource.resource_id} ({e.grade.value})"
            )

    k = report.keystone
    if k is not None:
        print(
            f"\nKeystone: step {k.idx} ({k.agent}) -- the decision and the order "
            f"placement both rest on this lookup, so a stale price here is the first "
            f"thing to review."
        )


if __name__ == "__main__":
    main()
