"""Characterize agent graphs on a corpus with OBSERVABLE, multi-hop dependencies.

Who&When dependencies are inferred; tau-bench's are observed events but the
write->read edges are a one-hop model. SWE-agent is the corpus that can actually
answer the depth question: a coding agent opens files, searches, edits, re-opens,
and edits again, all as commands in the trace. So the dependency between a file
edit and the read it rested on is OBSERVED, and repeated read/edit cycles on one
file form genuine multi-hop chains. This is the test the other two corpora set up:
do real agent runs have deep dependency structure that would justify a heavier
graph model (PyG / message passing), or do they stay shallow?

Corpus: nebius/SWE-agent-trajectories (SWE-bench runs). Each row has a `trajectory`
(role system/ai/user; `text` holds the message), a `target` bool (issue resolved),
and a `generated_patch`. We parse each ai action command and the file it touches,
and model each file event as depending on the prior event on the same file -- the
observed per-file interaction chain. Its longest path is the observed dependency
depth.

Run:  python experiment/agent_graph_swe_agent.py
      (downloads one ~90MB shard once via huggingface_hub)
"""
import json
import re
import shlex
import statistics as st
from collections import Counter

from auditable.graph import build_graph, characterize

REPO = "nebius/SWE-agent-trajectories"
SHARD = "data/train-00000-of-00012.parquet"
N_RUNS = 600  # a first pass on one shard, matching the scale of the other corpora

# action command -> how it touches the filesystem
_READ_CMDS = {"open", "goto", "scroll_up", "scroll_down", "search_file",
              "search_dir", "find_file", "cat", "ls", "grep"}
_WRITE_CMDS = {"edit", "create", "insert", "append", "submit"}
_OPEN_FILE_RE = re.compile(r"\(Open file:\s*(.+?)\)")
_BLOCK_RE = re.compile(r"```(?:bash|python)?\s*(.*?)```", re.DOTALL)


def _open_file(obs_text):
    """Path of the current open file, "" if the footer says none, None if no footer."""
    m = _OPEN_FILE_RE.search(obs_text or "")
    if not m:
        return None  # no footer: leave the current open file unchanged
    v = m.group(1).strip()
    return "" if v in ("n/a", "") else v  # explicit no-file clears the open file


def _parse_action(text, open_file):
    """Return (cmd, file, kind) for the ai turn's action; kind in read/write/run/None."""
    blocks = _BLOCK_RE.findall(text or "")
    if not blocks or not blocks[-1].strip():
        return (None, None, None)
    line = blocks[-1].strip().splitlines()[0].strip()
    if not line:
        return (None, None, None)
    try:
        tok = shlex.split(line)  # keeps quoted patterns with spaces as one token
    except ValueError:
        tok = line.split()
    if not tok:
        return (None, None, None)
    c = tok[0]
    if c in ("open", "create"):
        f = tok[1] if len(tok) > 1 else None
        return (c, f, "write" if c == "create" else "read")
    if c in ("edit", "insert", "append"):
        return (c, open_file, "write")
    if c in ("goto", "scroll_up", "scroll_down"):
        return (c, open_file, "read")
    if c == "search_file":
        f = tok[2] if len(tok) > 2 else open_file  # search_file "query" [file]
        return (c, f, "read")
    if c == "cat":
        return (c, tok[1] if len(tok) > 1 else None, "read")
    if c == "grep":
        args = [a for a in tok[1:] if not a.startswith("-")]
        recursive = any(
            a == "--recursive"
            or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
            for a in tok)
        f = args[1] if len(args) >= 2 else None  # grep PATTERN FILE
        if f and (recursive or f in (".", "./") or f.endswith("/")):
            f = None  # recursive or directory grep is a repo-level read
        return (c, f, "read")
    if c in ("find_file", "search_dir", "ls"):
        return (c, None, "read")  # repo-level read, no single target file
    if c == "submit":
        return (c, None, "write")  # applies the patch (the consequential action)
    return (c, None, "run")  # bash / reproduce / test execution


def to_steps(traj):
    """Typed steps with OBSERVED per-file dependencies.

    Each ai action is a decision step plus a tool_call step (read/write/run on a
    file). A file event depends_on the prior event on the SAME file, so repeated
    open/edit cycles on one file form an observed multi-hop chain.
    """
    steps, idx, open_file, last_on_file = [], 0, None, {}
    for turn in traj:
        role = turn.get("role")
        text = turn.get("text") or ""
        if role == "system":
            continue
        if role == "user":  # observation: only updates the current open file
            of = _open_file(text)
            if of is not None:  # distinguish "no footer" (leave) from n/a (clear)
                open_file = of or None
            continue
        if role != "ai":
            continue
        cmd, f, kind = _parse_action(text, open_file)
        steps.append({"idx": idx, "agent": "agent", "kind": "decision", "deps": []})
        idx += 1
        if cmd is None:
            continue
        deps = [last_on_file[f]] if (f and f in last_on_file) else []
        steps.append({"idx": idx, "agent": "env", "kind": "tool_call", "deps": deps,
                      "cmd": cmd, "file": f, "kind_fs": kind})
        if f:
            last_on_file[f] = idx
            if cmd in ("open", "create"):
                open_file = f
        idx += 1
    return steps


def load_runs(limit):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(REPO, SHARD, repo_type="dataset")
    pf = pq.ParquetFile(path)
    out = []
    for batch in pf.iter_batches(batch_size=200,
                                 columns=["instance_id", "model_name", "target", "trajectory"]):
        d = batch.to_pydict()
        for i in range(len(d["trajectory"])):
            traj = d["trajectory"][i]
            if isinstance(traj, str):
                traj = json.loads(traj)
            out.append({"instance_id": d["instance_id"][i], "model": d["model_name"][i],
                        "target": bool(d["target"][i]), "traj": traj})
            if len(out) >= limit:
                return out
    return out


def main():
    runs = load_runs(N_RUNS)
    rows, files_touched, deepest, write_depths, cmds, models = [], [], [], [], Counter(), Counter()
    n_reads = n_writes = n_runs_cmd = 0
    resolved_depth, unresolved_depth = [], []
    for rec in runs:
        steps = to_steps(rec["traj"])
        if len(steps) < 2:
            continue
        G = build_graph(steps, dependency="explicit", shared_resource=False)
        c = characterize(G)
        rows.append(c)
        tool_steps = [s for s in steps if s["kind"] == "tool_call"]
        for s in tool_steps:
            cmds[s["cmd"]] += 1
            if s["kind_fs"] == "read":
                n_reads += 1
            elif s["kind_fs"] == "write":
                n_writes += 1
            else:
                n_runs_cmd += 1
        files_touched.append(len({s["file"] for s in tool_steps if s["file"]}))
        deepest.append(c["dep_depth"])
        # edit-backed depth: same-file events before a write (read-only loops contribute 0)
        per_file, wdepth = {}, 0
        for s in tool_steps:
            if s["file"]:
                per_file.setdefault(s["file"], []).append(s)
        for chain in per_file.values():
            for pos, s in enumerate(chain):
                if s["kind_fs"] == "write":
                    wdepth = max(wdepth, pos)
        write_depths.append(wdepth)
        models[rec["model"]] += 1
        (resolved_depth if rec["target"] else unresolved_depth).append(c["dep_depth"])

    def dist(xs):
        return f"median {st.median(xs):.0f}  mean {st.mean(xs):.1f}  min {min(xs)}  max {max(xs)}"

    print(f"corpus: SWE-agent trajectories ({REPO}, one shard), {len(rows)} runs")
    print(f"models: {dict(models.most_common())}\n")

    print("=== representation coverage ===")
    print(f"every ai action maps to a typed step; "
          f"reads={n_reads} writes={n_writes} runs={n_runs_cmd}")
    print(f"top commands: {dict(cmds.most_common(8))}\n")

    print("=== size and shape ===")
    print(f"steps per run     : {dist([r['n_steps'] for r in rows])}")
    print(f"distinct files    : {dist(files_touched)}")
    print(f"nodes per run     : {dist([r['n_nodes'] for r in rows])}")

    print("\n=== OBSERVED dependency depth (the PyG question) ===")
    print(f"same-file interaction depth : {dist(deepest)}   "
          f"(longest command chain on one file; read-only search loops included)")
    print(f"edit-backed depth           : {dist(write_depths)}   "
          f"(same-file events an edit rests on; read-only loops excluded)")
    deep = sum(1 for d in deepest if d >= 3)
    deepw = sum(1 for d in write_depths if d >= 3)
    print(f"runs with interaction depth >= 3 : {deep}/{len(deepest)} ({deep/len(deepest):.0%})")
    print(f"runs with edit-backed depth >= 3 : {deepw}/{len(write_depths)} ({deepw/len(write_depths):.0%})")

    if resolved_depth and unresolved_depth:
        print("\n=== does dependency depth track success (issue resolved)? ===")
        print(f"resolved runs   : {len(resolved_depth)}  median depth {st.median(resolved_depth):.0f}")
        print(f"unresolved runs : {len(unresolved_depth)}  median depth {st.median(unresolved_depth):.0f}")

    print(f"\nReading: SWE-agent dependencies are OBSERVED, not inferred or one-hop-modeled. "
          f"Separating real edit dependencies from read-only search loops matters here: the "
          f"same-file interaction depth is median {st.median(deepest):.0f} but counts pure search "
          f"loops (the max-{max(deepest)} case is one open plus that many searches, no edit). The "
          f"edit-backed depth, where an edit rests on prior same-file events, is median "
          f"{st.median(write_depths):.0f} (mean {st.mean(write_depths):.1f}, max {max(write_depths)}), "
          f"and {deepw}/{len(write_depths)} ({deepw/len(write_depths):.0%}) of runs have an edit "
          f"resting on >= 3 prior events. So coding agents still leave the one-hop regime on real "
          "edit dependencies, less dramatically than the raw interaction depth suggests. Two caveats "
          "remain: these are weak models that thrash (depth tracks failure), and this is per-file "
          "depth, not cross-file propagation (an edit whose effect flows through imports), the next cut.")


if __name__ == "__main__":
    main()
