"""Methods on agent records read through agentpile: a conversation, a tool list, and a public dataset.

Run from anywhere after ``pip install ".[agentpile,tabular-ad]"`` in a checkout:

    python examples/example_agentpile.py              # offline: a conversation and a tool list
    python examples/example_agentpile.py --dataset    # also downloads the first 20 tau-bench runs

agentpile loads datasets, trace files and captured runs into one record; ``from_agentpile``
turns trajectories into the traces the run, step and online methods read, and
``configs_from_agentpile`` turns a record into the configurations the config methods read.
"""
import json
import sys

import agentpile as ap

import auditable as au
from auditable.methods import get_method


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


CONVERSATION = [
    {"role": "user", "content": "Cancel reservation RES-48213 and email me the confirmation."},
    {"role": "assistant", "content": None, "tool_calls": [_call("c1", "get_reservation", {"id": "RES-48213"})]},
    {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"reservation": "RES-48213", "flight": "UA2210", "passenger": "PAX-7731"})},
    {"role": "assistant", "content": None, "tool_calls": [_call("c2", "cancel_reservation", {"id": "RES-48213"})]},
    {"role": "tool", "tool_call_id": "c2", "content": "RES-48213 cancelled"},
    {"role": "assistant", "content": None, "tool_calls": [_call("c3", "send_email", {"to": "PAX-7731", "body": "RES-48213 cancelled"})]},
    {"role": "tool", "tool_call_id": "c3", "content": "sent"},
    {"role": "assistant", "content": "Done: RES-48213 is cancelled and the confirmation is on its way."},
]

TOOLS = {"tools": [
    {"name": "get_reservation", "description": "Look up a reservation.", "inputSchema": {"type": "object"},
     "annotations": {"readOnlyHint": True}},
    {"name": "cancel_reservation", "description": "Cancel a reservation.", "inputSchema": {"type": "object"},
     "annotations": {"readOnlyHint": False, "destructiveHint": True}},
    {"name": "delete_account", "description": "Delete a customer account.", "inputSchema": {"type": "object"},
     "annotations": {"destructiveHint": True}},
]}


def main(dataset=False):
    # POST and LIVE on one conversation: tool_call_id ties each reply to the result before it, and
    # value_flow ties a call to the earlier result whose id it reuses (results that are JSON objects).
    run = next(ap.load(CONVERSATION))
    trace = au.from_agentpile(run, deps=["tool_call_id_v1", "value_flow_v1"])
    print("steps:", [f"{s['idx']}:{s['kind']}" for s in trace.steps])
    print("dependencies:", {s["idx"]: s["deps"] for s in trace.steps if s["deps"]})
    print("rules:", [(r["rule"], r["edges"]) for r in trace.meta["dep_rules"]])
    print("step.dep_anomaly ranking:", get_method("step.dep_anomaly")().rank(trace))
    print("online.span_z alarms:", get_method("online.span_z")().step_alarm_scores(trace).round(2).tolist())

    # PRE on an MCP tools/list result: the permission comes from the tool annotations.
    (config,) = au.configs_from_agentpile(next(ap.load(TOOLS)), task="Cancel the reservation the user names.")
    finding = get_method("config.owasp_asi_combined")().audit(config)
    print("config.owasp_asi_combined: flag =", finding.flag, "excess =", sorted(finding.excess))

    if dataset:
        from auditable.datasets import load_traces

        traces, y, _ = load_traces("tau_bench", deps=["tool_call_id_v1", "value_flow_v1"], limit=20)
        scores = get_method("run.pyod_ecod")().fit_score(traces)
        print(f"tau-bench: {len(traces)} runs, {int((y == 1).sum())} failed; ECOD scores", scores.round(2)[:5])


if __name__ == "__main__":
    main(dataset="--dataset" in sys.argv[1:])
