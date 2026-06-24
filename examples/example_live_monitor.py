"""Monitor an agent run as it streams: watch the keystone emerge before the run ends.

`analyze_run` scores a finished run (POST). `LiveSession` scores the run as it grows:
feed each step as it happens, and the running report names the keystone and its blast
share over the run seen so far (`completeness=prefix`). A caller can flag, gate, or
checkpoint a high-blast step before the run completes.

The run here is the retail trajectory from the POST example, replayed one step at a
time so the output shows the running score change. The first read (get_order_details)
carries no blast when it happens and the prefix is too sparse to score. When the first
write lands on the read, the read becomes the keystone; its blast share then moves as
later steps arrive (the share is downstream reach over the run size seen so far), and it
settles at the keystone value once the final write lands. Watching that value move
mid-run is the live signal: you see which already-taken step the rest of the run is
coming to rest on, and the dips show why a prefix score is not yet the final score.

Honest framing: this is the same structural kernel as POST, run on a growing graph. It
is a live triage and gating signal, not a validated early-warning probability. Whether a
prefix score predicts failure early is a separate, unshipped validation (the prefix-AUC
curve), not a claim this example makes. No API key, no network.

Needs:  pip install "auditable[graph]"
Run:    python examples/example_live_monitor.py
"""
from auditable import LiveSession
from auditable.graph.adapters import tau_bench_prior_db_reads_v1


def trajectory():
    """A retail run: read the order, modify it, read the user, send a refund cert. Two
    writes (modify, send) both rest on the first read, so that read is the keystone."""
    return [
        {"role": "system", "content": "you are a retail agent; follow policy"},
        {"role": "user", "content": "Modify pending order #W512 and refund the difference."},
        {"role": "assistant", "content": "Pulling up the order.",
         "tool_calls": [{"function": {"name": "get_order_details"}}]},
        {"role": "tool", "name": "get_order_details",
         "content": '{"order_id": "#W512", "status": "pending"}'},
        {"role": "assistant", "content": "Updating the item.",
         "tool_calls": [{"function": {"name": "modify_pending_order_items"}}]},
        {"role": "tool", "name": "modify_pending_order_items", "content": "ok"},
        {"role": "assistant", "content": "Checking your account.",
         "tool_calls": [{"function": {"name": "get_user_details"}}]},
        {"role": "tool", "name": "get_user_details",
         "content": '{"user_id": "sara_doe_496"}'},
        {"role": "assistant", "content": "Issuing the refund certificate.",
         "tool_calls": [{"function": {"name": "send_certificate"}}]},
        {"role": "tool", "name": "send_certificate", "content": "gc-001"},
    ]


def main():
    # In production each step would arrive as the agent runs; here we lower a recorded
    # run to steps and feed them in order so the running score is visible.
    steps = list(tau_bench_prior_db_reads_v1.to_steps(trajectory()))

    live = LiveSession(adapter=tau_bench_prior_db_reads_v1)
    print("streaming the run; the keystone and its blast share update per step:\n")
    for step in steps:
        report = live.observe(step)
        k = report.keystone
        if k is not None:
            head = f"keystone -> step {k.idx} {k.label} (blast {k.score:.3f})"
        else:
            head = f"keystone -> none yet ({report.state})"
        print(f"  + step {step.idx:>2} {_label(step):<34} | {head}")

    final = live.report()
    print("\nfinal running report (a prefix; the run could still continue):")
    print(final.summary())


def _label(step):
    return f"{step.kind} {step.node_attrs.get('tool_name', step.agent or '')}".strip()


if __name__ == "__main__":
    main()
