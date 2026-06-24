"""Plug in a REAL LLM agent: capture a model-driven LangGraph run into the typed graph.

This is the sibling of example_langgraph_capture.py, with the one difference that
matters: every node here makes a real LLM call (an OpenAI-compatible chat
completion) instead of running a pure function. The capture is identical, because
`instrument(...)` records the state channels each node read and wrote, not what
happened inside the node. So the OBSERVED dependency edges come out the same
whether a node's write came from a pure function or a frontier model. Proving that
is the whole point of this example.

Supported v1 shape: plain sync function nodes over TypedDict state, with the model
call inside the function. A LangChain Runnable node or a prebuilt
`create_react_agent` is a documented v1 non-goal and would not be captured; write
the node as a function that calls the model, as below.

The example needs a model, reached through any OpenAI-compatible endpoint via three
environment variables:

    OPENAI_API_KEY    required; the example prints a notice and exits 0 if unset
    OPENAI_BASE_URL   optional; point at a gateway or local server (default: OpenAI)
    OPENAI_MODEL      optional; default "gpt-4o-mini"

That covers a plain OpenAI key, a local model server (vLLM, Ollama, LM Studio), or
any OpenAI-compatible gateway, by setting OPENAI_BASE_URL and OPENAI_MODEL.

The scenario is a support-ticket agent: understand the request, look up account
state, decide an action, compose an internal note, then review it against the
original intent. Each step rests on the one before, and `review` re-reads the
intent, so `understand` is the structural keystone: a wrong reading there
propagates through the whole run.

Honesty holds in the output, the same as the pure-function capture example. The
edges are OBSERVED channel-level read-after-committed-write touches, matched across
LangGraph's superstep barrier; the keystone ranking is a triage signal, not a
calibrated probability; and the GRADE corpus numbers (arXiv:2606.22741) are not
produced here. This is a capability demo on one live run.

Needs:  pip install "auditable[langgraph,llm]"
Run:    OPENAI_API_KEY=sk-... python examples/example_langgraph_llm_agent.py
"""
import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from auditable import analyze_run
from auditable.integrations.langgraph import instrument


def _resolve_client():
    """Return (client, model, None) to run live, or (None, None, reason) to skip."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, None, "OPENAI_API_KEY is not set"
    try:
        from openai import OpenAI
    except ImportError:
        return None, None, 'the openai package is not installed (pip install "auditable[llm]")'
    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> the OpenAI default endpoint
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return client, model, None


class TicketState(TypedDict):
    ticket: str
    intent: str
    account_state: str
    action: str
    draft: str
    final: str


def _build_agent(client, model):
    """Build the instrumented agent. Each node is a plain function that calls the model."""

    def ask(prompt):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return (resp.choices[0].message.content or "").strip().replace("\n", " ")

    def understand(state):
        out = ask("In one short sentence, state what this customer actually wants. "
                  f"Ticket: {state['ticket']}")
        print(f"  [understand] {out[:80]}")
        return {"intent": out}

    def lookup(state):
        out = ask("Return a plausible one-line account status for this intent "
                  f"(invent realistic details). Intent: {state['intent']}")
        print(f"  [lookup]     {out[:80]}")
        return {"account_state": out}

    def decide(state):
        out = ask("Decide the single next action in one short sentence. "
                  f"Account status: {state['account_state']}")
        print(f"  [decide]     {out[:80]}")
        return {"action": out}

    def compose(state):
        out = ask(f"Draft a one-sentence internal note for this action: {state['action']}")
        print(f"  [compose]    {out[:80]}")
        return {"draft": out}

    def review(state):
        out = ask("Rewrite this draft into a two-sentence customer reply that addresses "
                  f"the original intent. Draft: {state['draft']}. Intent: {state['intent']}")
        print(f"  [review]     {out[:80]}")
        return {"final": out}

    builder = instrument(StateGraph(TicketState))
    builder.add_node("understand", understand)
    builder.add_node("lookup", lookup)
    builder.add_node("decide", decide)
    builder.add_node("compose", compose)
    builder.add_node("review", review)
    builder.add_edge(START, "understand")
    builder.add_edge("understand", "lookup")
    builder.add_edge("lookup", "decide")
    builder.add_edge("decide", "compose")
    builder.add_edge("compose", "review")
    builder.add_edge("review", END)
    return builder


def main():
    client, model, skip = _resolve_client()
    if skip:
        print(f"example_langgraph_llm_agent: skipping the live run because {skip}.")
        print("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, OPENAI_MODEL) to run it; "
              "see examples/README.md.")
        return

    ticket = ("My account got locked after I changed my password and now I cannot log "
              "in to download my invoices before the tax deadline tomorrow.")
    print(f"Model: {model}")
    print(f"Ticket: {ticket}\n")
    print("Running the real LLM agent (each node makes one model call):")

    builder = _build_agent(client, model)
    graph = builder.compile()
    graph.invoke({"ticket": ticket, "intent": "", "account_state": "",
                  "action": "", "draft": "", "final": ""})

    report = analyze_run(builder, adapter=builder)
    print("\n" + "=" * 70)
    print(report)

    print("\nObserved dependency edges captured from the live LLM run:")
    steps = {s.idx: s for s in builder.to_steps()}
    n_edges = observed = 0
    for s in steps.values():
        for e in s.deps:
            src = steps[e.src_idx]
            n_edges += 1
            observed += e.grade.value == "observed"
            print(f"  {s.node_attrs['langgraph_node']} -> {src.node_attrs['langgraph_node']} "
                  f"via channel '{e.resource.resource_id}' ({e.grade.value}, {e.evidence['mode']})")
    if n_edges:
        print(f"\n{n_edges} dependency edges, {observed} observed "
              f"({100 * observed // n_edges}% observed).")

    k = report.keystone
    if k is not None:
        print(f"Keystone: step {k.idx} ({k.node_attrs.get('langgraph_node')}) -- "
              "the rest of the run transitively rests on this step.")


if __name__ == "__main__":
    main()
