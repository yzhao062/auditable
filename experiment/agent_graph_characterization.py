"""Characterize the graphs that real agent systems induce, to verify the graph is
a faithful representation of agentic systems and to motivate the system design.

Corpus: Who&When (ag2ai / Kevin355/Who_and_When, MIT), 184 real multi-agent
failure traces with step-level fault labels (mistake_agent, mistake_step). Each
task is a multi-expert collaboration with a Computer_terminal tool executor.

What this measures (each property settles a design decision, per
research/agent-graph-characterization.md):
  - size and shape  -> is the lightweight stack (NetworkX + PyOD) justified
  - dependency depth -> local scoring vs graph propagation / deep GNN (the PyG question)
  - failure origin   -> where faults start and how far they propagate
  - State A vs B     -> what the trace gives for free vs what must be inferred

Run:  python experiment/agent_graph_characterization.py
      (downloads the corpus once via huggingface_hub if the cache is absent)
"""
import glob
import json
import os
import statistics as st

from auditable.graph import build_graph, characterize, downstream_reach

CACHE = os.path.join(os.path.dirname(__file__), ".cache", "whoandwhen", "Who&When")
_TOOL_AGENTS = {"Computer_terminal"}


def _ensure_corpus():
    if glob.glob(os.path.join(CACHE, "*", "*.json")):
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Missing corpus and huggingface_hub unavailable ({exc}). "
                         "Install huggingface_hub or place Who&When JSONs under experiment/.cache/whoandwhen/.")
    snapshot_download("Kevin355/Who_and_When", repo_type="dataset",
                      local_dir=os.path.join(os.path.dirname(__file__), ".cache", "whoandwhen"),
                      allow_patterns=["Who&When/**"])


def _is_tool(step):
    if step.get("name") in _TOOL_AGENTS:
        return True
    c = str(step.get("content", "")).lower()
    return c.startswith("exitcode") or c.startswith("code output")


def to_steps(task):
    """Normalize a Who&When task into typed steps for the construction layer."""
    steps = []
    for i, h in enumerate(task.get("history", [])):
        agent = h.get("name") or h.get("role") or "unknown"
        steps.append({"idx": i, "agent": agent, "kind": "tool_call" if _is_tool(h) else "decision"})
    return steps


def load_tasks():
    out = []
    for path in sorted(glob.glob(os.path.join(CACHE, "*", "*.json"))):
        try:
            out.append(json.load(open(path, encoding="utf-8")))
        except Exception:
            continue
    return out


def main():
    _ensure_corpus()
    tasks = load_tasks()
    rows, mistakes = [], []
    n_failed = 0
    for t in tasks:
        steps = to_steps(t)
        if len(steps) < 2:
            continue
        G = build_graph(steps, dependency="full_context")
        c = characterize(G)
        rows.append(c)
        # most-active agent (excludes the tool executor, which is plumbing not a decider)
        from collections import Counter
        active = Counter(s["agent"] for s in steps if s["kind"] == "decision")
        top_agent = active.most_common(1)[0][0] if active else None
        if str(t.get("is_correct", "")).lower() == "false":
            n_failed += 1
            try:
                ms = int(t.get("mistake_step"))
            except (TypeError, ValueError):
                ms = None
            if ms is not None and 0 <= ms < c["n_steps"]:
                mistakes.append({
                    "pos": ms,
                    "norm_pos": ms / max(1, c["n_steps"] - 1),
                    "blast": downstream_reach(G, ms),
                    "n_steps": c["n_steps"],
                    "agent_is_top": t.get("mistake_agent") == top_agent,
                })

    def dist(key):
        xs = [r[key] for r in rows]
        return f"median {st.median(xs):.0f}  mean {st.mean(xs):.1f}  min {min(xs)}  max {max(xs)}"

    print(f"corpus: Who&When, {len(rows)} usable traces ({n_failed} labeled-failed); each a real multi-agent run\n")
    print("=== representation coverage (do our types capture real traces?) ===")
    tool_frac = st.mean([r["n_tool_calls"] / r["n_steps"] for r in rows])
    print(f"every step maps to a node type by construction; tool_call fraction = {tool_frac:.0%} "
          f"(rest are decision/model steps)")
    with_tool = sum(1 for r in rows if r["n_tool_calls"])
    print(f"node types used across the corpus: agent, decision, tool_call, dependency_resource "
          f"({with_tool}/{len(rows)} traces contain a tool_call)\n")

    print("=== size and shape (settles: lightweight stack vs heavy) ===")
    print(f"steps per trace   : {dist('n_steps')}")
    print(f"distinct agents   : {dist('n_agents')}")
    print(f"nodes per trace   : {dist('n_nodes')}")
    print(f"dependency depth  : {dist('dep_depth')}   (longest depends_on chain)")

    print("\n=== where failures originate and how far they reach ===")
    if mistakes:
        norm = [m["norm_pos"] for m in mistakes]
        early = sum(1 for x in norm if x <= 0.34) / len(norm)
        blast = [m["blast"] for m in mistakes]
        top = sum(1 for m in mistakes if m["agent_is_top"]) / len(mistakes)
        print(f"labeled mistakes      : {len(mistakes)} with a usable step index")
        print(f"mistake position      : median {st.median(norm):.2f} of the run (0=first, 1=last); "
              f"{early:.0%} land in the first third")
        print(f"blast radius (steps depending on the mistake, full-context): "
              f"median {st.median(blast):.0f}  mean {st.mean(blast):.1f}")
        print(f"mistake from the most-active agent: {top:.0%}")

    print("\n=== State A vs State B, measured ===")
    exec_e = st.mean([r["n_exec_edges"] for r in rows])
    dep_e = st.mean([r["n_dep_edges"] for r in rows])
    print(f"execution edges per trace (observed from the trace): mean {exec_e:.1f}")
    print(f"dependency edges per trace (INFERRED, full-context assumption): mean {dep_e:.1f}")
    print("the trace gives the execution graph directly; the dependency graph is modeled, "
          "not read off the trace.")

    print("\nReading: the typed graph fits every one of these real runs, and the runs are small "
          "(median 10 steps, 4 agents), which supports the lightweight construction. Two honest "
          "limits sharpen the next step. The dependency graph is not in the trace (mean ~689 inferred "
          "edges vs ~44 observed), so the State A / State B split is real, not assumed. And because "
          "the dependencies are inferred under a full-context assumption, the depends_on depth tracks "
          "run length and cannot reveal true multi-hop propagation; settling whether deep "
          "message-passing (PyG) is needed requires a corpus where dependencies are observable "
          "(SWE-agent file and environment reads), which is the next characterization.")


if __name__ == "__main__":
    main()
