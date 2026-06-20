"""v0.3 offline analysis: rank an agent run by structural risk, name the keystone.

Point auditable at a recorded agent run and it builds one session decision graph,
scores every step by how much of the rest of the run transitively rests on it (the
keystone signal the graph-view study validated), and reports the riskiest decision
to review first. No live agent, no API key, no network.

The trajectory below is modeled on a tau-bench airline task (tau-bench: Sierra
Research, MIT, github.com/sierra-research/tau-bench), and its shape is grounded in a
survey of the 660 public tau-bench trajectories: 38% of runs issue two or more
consequential writes that rest on prior reads, exactly the structure this example
shows. The agent reads one reservation, then makes two consequential writes against
it (rebook the flights, add a checked bag). Both writes rest on that single read, so
`get_reservation_details` is the structural keystone: a fault there propagates to
both writes, which is why auditable ranks it first.

Honesty holds in the output: the read/write events are observed, but the
write-to-prior-read edges are MODELED (a conservative upper bound, not a causal
label), and the score is a triage ranking, not a calibrated probability.

Run:  python examples/analyze_run.py
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
