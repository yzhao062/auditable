"""Characterize agent graphs on a corpus with REAL auditable decisions.

Where Who&When is multi-agent debate (dependencies latent, nothing externally
consequential), tau-bench is a tool-using agent acting against a backend database
(retail / airline). The DB read and write events are in the trace, so the read/write
structure is observed rather than inferred (the write-to-read edges themselves are a
conservative model), and some calls are consequential writes (book / cancel / modify
a reservation or order) whose correctness tau-bench scores with `db_match`. This is
the textbook case the product targets: audit a decision against the state it relied on.

What this adds over the Who&When pass (research/agent-graph-characterization.md):
  - the consequential subset: what fraction of steps actually mutate state
  - the audit surface: how many prior DB reads each write could have rested on (what
    replay would have to revalidate per consequential decision)
  - State B events are observable here (tool I/O is in the trace) vs fully inferred in
    Who&When; the write-to-read edges are still a model
  - does graph size / shape track failure (db_match)?

Run:  python experiment/agent_graph_tau_bench.py
      (downloads a few model trajectory files once via huggingface_hub)
"""
import json
import statistics as st
from collections import Counter

from auditable.graph import build_graph, characterize

REPO = "AgentSuite/tau-bench-trajectories"
# a spread across capability, so run-length / shape varies (weak models retry more)
MODELS = [
    "claude-4.5-sonnet-thinking-on-10k",
    "gpt-4.1",
    "gpt-4o-mini",
    "Kimi-K2-Instruct",
]
# consequential DB mutations vs reads, by tool-name prefix (tau-bench retail + airline)
_WRITE_PREFIXES = ("book_", "cancel_", "modify_", "update_", "return_", "exchange_", "send_")
# tool calls that touch no DB state (local compute, no-op thought, human handoff,
# static reference lookup); they are tool steps but not part of any write's audit surface
_NON_DB_TOOLS = {"calculate", "think", "transfer_to_human_agents", "list_all_airports"}


def _is_write(name):
    return bool(name) and name.startswith(_WRITE_PREFIXES)


def _is_db_read(name):
    return bool(name) and not _is_write(name) and name not in _NON_DB_TOOLS


def _ensure_files():
    from huggingface_hub import hf_hub_download
    paths = []
    for m in MODELS:
        try:
            paths.append(hf_hub_download(REPO, f"{m}.jsonl", repo_type="dataset"))
        except Exception as exc:  # pragma: no cover
            print(f"  (skip {m}: {exc})")
    if not paths:
        raise SystemExit("Could not fetch any tau-bench trajectory file.")
    return paths


def to_steps(messages, model_name):
    """Normalize one tau-bench task run into typed steps with modeled deps.

    Read and write tool events are observed in the trace. For each consequential
    write we attach every prior DB-read tool step as a conservative temporal upper
    bound on the audit surface; tau-bench does not label the exact causal read
    subset a write used. Non-DB utility calls (calculate, think, handoff) are tool
    steps but are not reads and never enter the audit surface.
    """
    steps, pending, reads_so_far, idx = [], [], [], 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue  # the policy text: context, not a step
        if role == "user":
            steps.append({"idx": idx, "agent": "user", "kind": "decision", "deps": []})
            idx += 1
        elif role == "assistant":
            steps.append({"idx": idx, "agent": model_name, "kind": "decision", "deps": []})
            idx += 1
            for tc in (m.get("tool_calls") or []):
                name = (tc.get("function") or {}).get("name") or tc.get("name")
                pending.append(name)
        elif role == "tool":
            name = m.get("name")
            if name is None and pending:
                name = pending.pop(0)
            elif name in pending:
                pending.remove(name)
            write = _is_write(name)
            db_read = _is_db_read(name)
            steps.append({"idx": idx, "agent": "env", "kind": "tool_call",
                          "deps": list(reads_so_far) if write else [],
                          "tool": name, "is_write": write, "is_db_read": db_read})
            if db_read:
                reads_so_far.append(idx)
            idx += 1
    return steps


def load_runs(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def main():
    paths = _ensure_files()
    rows, audit_surface, by_domain = [], [], Counter()
    n_writes_total = 0
    failed_sizes, ok_sizes = [], []
    for rec in load_runs(paths):
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 3:
            continue
        model = rec.get("model_path", "model")
        steps = to_steps(msgs, model)
        if len(steps) < 2:
            continue
        G = build_graph(steps, dependency="explicit", shared_resource=False)
        c = characterize(G)
        writes = [s for s in steps if s.get("is_write")]
        reads = [s for s in steps if s.get("is_db_read")]
        c["n_writes"], c["n_reads"] = len(writes), len(reads)
        rows.append(c)
        by_domain[rec.get("task_name", "?")] += 1
        n_writes_total += len(writes)
        for w in writes:
            audit_surface.append(len(w["deps"]))
        # does size track failure? db_match = did the consequential writes land correctly
        db_match = bool((rec.get("eval_result") or {}).get("db_match"))
        (ok_sizes if db_match else failed_sizes).append(c["n_steps"])

    def dist(key):
        xs = [r[key] for r in rows]
        return f"median {st.median(xs):.0f}  mean {st.mean(xs):.1f}  min {min(xs)}  max {max(xs)}"

    dom = "  ".join(f"{k}={v}" for k, v in by_domain.items())
    print(f"corpus: tau-bench trajectories ({REPO}), {len(rows)} task runs across "
          f"{len(paths)} models ({dom})\n")

    print("=== representation coverage (do our types capture tool-using runs?) ===")
    tool_frac = st.mean([r["n_tool_calls"] / r["n_steps"] for r in rows])
    print(f"every non-system message maps to a typed step; tool_call fraction = {tool_frac:.0%} "
          f"(user/assistant turns are the rest)")
    with_tool = sum(1 for r in rows if r["n_tool_calls"])
    print(f"node types used across the corpus: agent, decision, tool_call (DB reads, writes, "
          f"and utility calls) ({with_tool}/{len(rows)} runs contain a tool_call)\n")

    print("=== size and shape (settles: lightweight stack vs heavy) ===")
    print(f"steps per run     : {dist('n_steps')}")
    print(f"distinct agents   : {dist('n_agents')}   (user + assistant model + env)")
    print(f"nodes per run     : {dist('n_nodes')}")

    print("\n=== the consequential subset (what is actually worth auditing) ===")
    w = [r["n_writes"] for r in rows]
    r = [r["n_reads"] for r in rows]
    write_frac = n_writes_total / sum(x["n_steps"] for x in rows)
    runs_with_write = sum(1 for x in rows if x["n_writes"])
    print(f"DB reads per run  : median {st.median(r):.0f}  mean {st.mean(r):.1f}")
    print(f"DB writes per run : median {st.median(w):.0f}  mean {st.mean(w):.1f}  max {max(w)}")
    print(f"consequential writes are {write_frac:.0%} of all steps; "
          f"{runs_with_write}/{len(rows)} runs make at least one write")

    print("\n=== the audit surface (prior DB reads each write could have rested on) ===")
    if audit_surface:
        print(f"reads a write depended on: median {st.median(audit_surface):.0f}  "
              f"mean {st.mean(audit_surface):.1f}  max {max(audit_surface)}  "
              f"(n={len(audit_surface)} writes)")
        print(f"modeled dependency depth  : {dist('dep_depth')}   "
              f"(one hop by construction: writes point at prior reads)")

    print("\n=== State B: observed tool I/O, modeled write->read edges ===")
    dep_e = st.mean([r["n_dep_edges"] for r in rows])
    exec_e = st.mean([r["n_exec_edges"] for r in rows])
    print(f"execution edges per run (observed): mean {exec_e:.1f}")
    print(f"dependency edges per run (modeled write->prior-read): mean {dep_e:.1f}")
    print("unlike Who&When, the DB read/write events are in the trace; the write->read "
          "edge set is a conservative prior-read model, not a causal label.")

    if failed_sizes and ok_sizes:
        print("\n=== does graph size track failure (db_match)? ===")
        print(f"runs with correct final DB state : {len(ok_sizes)}  median {st.median(ok_sizes):.0f} steps")
        print(f"runs with wrong   final DB state : {len(failed_sizes)}  median {st.median(failed_sizes):.0f} steps")

    print("\nReading: the typed graph fits tool-using agent runs as well as it fit debate "
          "traces, and these runs are still small (median ~"
          f"{st.median([r['n_steps'] for r in rows]):.0f} steps). The point this corpus adds "
          "over Who&When: the consequential subset is sparse (writes are a small fraction of "
          "steps), and under a conservative prior-DB-read model each write has a small audit "
          "surface (median above). The read/write events are observed, while the exact "
          "write->read dependencies are modeled. The one-hop dependency depth follows from "
          "this construction, so it is a useful baseline but not independent evidence against "
          "deeper message-passing; data with true multi-hop dependencies (a write read back "
          "and re-used) needs a filesystem corpus (SWE-agent), the remaining stress test for "
          "the depth question.")


if __name__ == "__main__":
    main()
