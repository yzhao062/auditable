"""POST pillar demo: rank a finished agent run, name the keystone.

What this shows, in plain terms: given a run that already finished, auditable
builds one typed decision graph over it, scores every step by how much of the
rest of the run rests on it, and names the single step to review first (the
keystone). This is the lifecycle pillar where the GRADE results live.

GRADE relation (cite arXiv:2606.22741): the graph's two layers are what GRADE
measures on a six-corpus benchmark. The dependency layer PREDICTS run failure
(ROC-AUC 0.805 on SWE-Gym, +0.142 over the run-length baseline), and the
execution layer LOCALIZES the faulting step (Top-3 0.614 on Who&When). IMPORTANT:
those are GRADE full-corpus numbers, NOT outputs of this example. The trajectory
below is a small illustrative tau-bench-style run; it prints a keystone ranking,
not a benchmark score. The corpus-scale evaluation lives in GRADE.

auditable attaches at three points in an agent's lifecycle. This is the POST
pillar: read a recorded run after it completes, build one typed session decision
graph, score every step by how much of the rest of the run transitively rests on
it, and report the riskiest decision to review first. The other two pillars are
example_pre_lint_plan.py (PRE: lint a declared plan before deploy) and
example_live_replay.py (LIVE: replay under live state and execute a fix). All
three run over the same typed two-layer graph. No live agent, no API key, no
network.

The trajectory below is modeled on a tau-bench airline task (tau-bench: Sierra
Research, MIT, github.com/sierra-research/tau-bench). The agent reads one
reservation, then makes two consequential writes against it (rebook the flights,
add a checked bag). Both writes rest on that single read, so
`get_reservation_details` is the structural keystone: a fault there propagates to
both writes, which is why auditable ranks it first.

Honesty holds in the output: the read/write events are observed, but the
write-to-prior-read edges are MODELED (a conservative upper bound, not a causal
label), and the score is a triage ranking, not a calibrated probability.

Run:  python examples/example_post_rank_run.py
"""
from auditable import analyze_run
from auditable.graph.adapters import tau_bench_prior_db_reads_v1
from auditable.graph.grounding import ground_basis


def airline_run():
    """A tau-bench-style airline trajectory (role / tool messages).

    Modeled on the common real pattern (read a reservation, then update its flights
    and its baggages); the two updates both rest on the one reservation read.
    """
    return [
        {"role": "system", "content": "you are an airline agent; follow the policy"},
        {"role": "user",
         "content": "On reservation ZFA04Y, move me to the morning flights and add one checked bag."},
        {"role": "assistant", "content": "Pulling up the reservation.",
         "tool_calls": [{"function": {"name": "get_reservation_details"}}]},
        {"role": "tool", "name": "get_reservation_details",
         "content": '{"reservation_id": "ZFA04Y", "cabin": "economy", '
                    '"origin": "SFO", "destination": "JFK", "baggages": 0}'},
        {"role": "assistant", "content": "Rebooking onto the morning flights.",
         "tool_calls": [{"function": {"name": "update_reservation_flights"}}]},
        {"role": "tool", "name": "update_reservation_flights", "content": "ok"},
        {"role": "assistant", "content": "Adding the checked bag.",
         "tool_calls": [{"function": {"name": "update_reservation_baggages"}}]},
        {"role": "tool", "name": "update_reservation_baggages", "content": "ok"},
    ]


def main():
    messages = airline_run()

    # One public call: adapter -> session graph -> structural risk (+ grounding).
    report = analyze_run(messages, adapter=tau_bench_prior_db_reads_v1)
    print(report)

    # The model-consistency helper: a corpus tool step states no basis, so the report's
    # grounding is empty above. The basis below is an example we supply (the corpus trace
    # records none), scored by the same deterministic helper against the reservation read.
    print("\n  model-consistency grounding (an example basis vs. the reservation the run read):")
    basis = "Reservation ZFA04Y is an economy booking, SFO to JFK, with 0 checked bags."
    read_back = [
        '{"reservation_id": "ZFA04Y", "cabin": "economy", "origin": "SFO", '
        '"destination": "JFK", "baggages": 0}',
    ]
    g = ground_basis(basis, retrieved=read_back)
    print(f"    basis:     {basis}")
    print(f"    grounded:  {g.score} supported ({g.state})")
    print(f"    supported: {', '.join(g.matched)}")
    if g.unmatched:
        print(f"    unsupported: {', '.join(g.unmatched)}")

    k = report.keystone
    print(
        f"\nKeystone: step {k.idx} ({k.node_attrs['tool']}) -- "
        f"two consequential writes (rebook, add bag) rest on this one read, "
        f"so auditable ranks it first to review."
    )


if __name__ == "__main__":
    main()
