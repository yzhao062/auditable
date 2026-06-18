"""The cross-file dependency cut: does a real coding fix propagate across files?

The SWE-agent corpus measured per-file re-edit depth (a file read, edited,
re-read, edited again). The open question it left is cross-file propagation: an
edit whose effect flows through an import or a call to another file. That is the
case a heavier graph model (PyG / message passing) would actually target, and
per-file depth cannot see it.

This corpus answers it with GOLD patches, not agent trajectories. SWE-bench_Verified
is 500 human-validated instances; each `patch` is the real human fix (source only;
tests live in a separate `test_patch`). So the cross-file structure here is the
intrinsic structure of the fix, independent of any agent's weak-model thrashing,
which removes the SWE-agent corpus's biggest caveat.

The measurement separates two kinds of multi-file fix that a naive "touched >= 2
files" count conflates:

  - dependency-coupled: file A references a symbol that file B provides, so the
    edit in B propagates to A. This is a real cross-file dependency edge, the
    multi-hop case.
  - parallel-sibling: several files edited together with no symbol shared between
    their hunks (e.g. the mysql / oracle / sqlite backends each overriding one
    interface). Edited together, but no propagation path, so a graph model has
    nothing to message-pass along.

Coupling is read from the patch hunks only and by symbol NAME, so it has errors in
both directions: it misses a dependency on a symbol whose `def` is unchanged and
outside the hunk (false negative), and a bare name can match across unrelated
receivers (false positive: two backends calling `gc.get_antialiased()` while a
third file defines `Text.get_antialiased`). It is therefore a heuristic screen, not
a calibrated coupling rate. Two guards bound the name-collision error:

  - A name a file DEFINES on its own def/class signature line is not counted as a
    use of that name, so two files overriding a same-named method do not look
    mutually coupled.
  - The unambiguous-provider variant counts a shared symbol only when exactly one
    edited file in the patch provides it. This drops receiver-blind collisions, but
    it also drops legitimate multi-provider edges such as base/override methods and
    shared interfaces, which a patch-only screen cannot type-resolve. The
    receiver-blind variant keeps every name match and is reported alongside so the
    ambiguity is visible; the dropped edges are a mix, not all false.

Provision is read as the `def` / `class` names in a hunk; widening it to include
module-level (column-0) assignments was tested and adds no edges on this dataset,
so the def/class basis is used. The proper coupling census needs an import / symbol
graph from the repo source at `base_commit` and is deferred; this script is a
patch-only screen.

Run:  python experiment/agent_graph_swebench_crossfile.py
      (downloads one ~5MB parquet once via huggingface_hub)
"""
import keyword
import re
import statistics as st
from collections import Counter

import networkx as nx

REPO = "princeton-nlp/SWE-bench_Verified"
SHARD = "data/test-00000-of-00001.parquet"

_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)
_DEF = re.compile(r"(?:def|class)\s+([A-Za-z_]\w*)")
_DEFNAME = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+[A-Za-z_]\w*")
_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)")
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_ATTR = re.compile(r"\.([A-Za-z_]\w*)")
_WORD = re.compile(r"[A-Za-z_]\w*")
# names that are method/attribute noise: short or ubiquitous, so a match across
# two files is far more likely coincidence than a real dependency edge.
_NOISE = {"get", "set", "run", "name", "value", "data", "self", "cls", "type",
          "key", "keys", "items", "values", "len", "str", "int", "list", "dict",
          "args", "kwargs", "result", "obj", "node", "test", "main", "init",
          "append", "format", "join", "split", "strip", "update", "add", "pop",
          "true", "false", "none", "super", "print", "return", "import", "from"}


def _ok(name):
    return len(name) >= 4 and name not in _NOISE and name not in keyword.kwlist


def _is_test(path):
    p = path.lower()
    return (p.startswith("test") or "/test" in p or "tests/" in p
            or "_test" in p or "/conftest" in p)


def parse_patch(patch):
    """{source_file: {'added': [line], 'removed': [line]}} from a unified diff.

    Diff sections with no +/- content (mode-only, rename-only, binary) are
    skipped so they do not inflate the file counts as zero-edit "touched" files.
    """
    out = {}
    matches = list(_DIFF_GIT.finditer(patch or ""))
    for i, m in enumerate(matches):
        path = m.group(2)
        if _is_test(path):
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(patch or "")
        added, removed = [], []
        for line in patch[start:end].splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added.append(line[1:])
            elif line.startswith("-"):
                removed.append(line[1:])
        if not added and not removed:
            continue  # no edit content: mode change, rename, or binary blob
        out[path] = {"added": added, "removed": removed}
    return out


def file_symbols(sections):
    """Per file: symbols it provides (strict / wide) and symbols it references."""
    strict, wide, referenced = {}, {}, {}
    for path, hunks in sections.items():
        both = hunks["added"] + hunks["removed"]
        defs = {s for s in _DEF.findall("\n".join(both)) if _ok(s)}
        # wide adds only MODULE-LEVEL (column-0) assignments: the constants and
        # aliases a file genuinely provides. Indented assignments are locals,
        # keyword arguments, and class attributes -- not cross-file providers.
        assigned = set()
        for line in both:
            if line[:1].isspace():
                continue
            m = _ASSIGN.match(line)
            if m and _ok(m.group(1)):
                assigned.add(m.group(1))
        # references: calls / attribute access / imports in ADDED lines, but the
        # name a def/class line DEFINES is not a use, so strip it before scanning.
        r = set()
        for line in hunks["added"]:
            body = _DEFNAME.sub("", line, count=1)
            r |= set(_CALL.findall(body)) | set(_ATTR.findall(body))
            s = line.strip()
            if s.startswith(("import ", "from ")):
                r |= set(_WORD.findall(s))
        strict[path] = defs
        wide[path] = defs | assigned
        referenced[path] = {n for n in r if _ok(n)}
    return strict, wide, referenced


def coupling_graph(referenced, provides, *, unambiguous=False):
    """DiGraph over files; edge A -> B if A references a symbol B provides.

    With unambiguous=True, only symbols provided by exactly one edited file count.
    This is a conservative ambiguity filter: it drops receiver-blind collisions
    (two files calling a bare `foo()` while a third defines a different `foo`), but
    a dropped multi-provider symbol can also be a real base/override or interface
    dependency, so this is not proof the dropped edges are false.
    """
    files = list(provides)
    counts = Counter(s for names in provides.values() for s in names)
    G = nx.DiGraph()
    G.add_nodes_from(files)
    for a in files:
        for b in files:
            if a == b:
                continue
            shared = referenced[a] & provides[b]
            if unambiguous:
                shared = {s for s in shared if counts[s] == 1}
            if shared:
                G.add_edge(a, b, via=sorted(shared))
    return G


def cross_file_depth(G):
    """Longest cross-file dependency chain (edges); a cyclic coupling counts as 1."""
    if G.number_of_edges() == 0:
        return 0
    return max(1, nx.dag_longest_path_length(nx.condensation(G)))


def load_runs():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(REPO, SHARD, repo_type="dataset")
    pf = pq.ParquetFile(path)
    out = []
    for batch in pf.iter_batches(batch_size=200,
                                 columns=["instance_id", "repo", "patch", "difficulty"]):
        d = batch.to_pydict()
        for i in range(len(d["patch"])):
            out.append({"instance_id": d["instance_id"][i], "repo": d["repo"][i],
                        "patch": d["patch"][i], "difficulty": d["difficulty"][i]})
    return out


def summarize(multi, key):
    """(coupled count, depth list over coupled) for graphs stored under `key`."""
    coupled = [r for r in multi if r[key].number_of_edges() > 0]
    depths = [cross_file_depth(r[key]) for r in coupled]
    return len(coupled), depths


def main():
    runs = load_runs()
    span = Counter()
    multi = []
    n_single = n_empty = wide_adds = 0
    coupled_examples, ambiguous_examples, parallel_examples = [], [], []
    for rec in runs:
        sections = parse_patch(rec["patch"])
        nfiles = len(sections)
        span[nfiles] += 1
        if nfiles == 0:
            n_empty += 1  # patch touched only tests / no-hunk sections
            continue
        if nfiles == 1:
            n_single += 1
            continue
        strict, wide, referenced = file_symbols(sections)
        Gb = coupling_graph(referenced, strict, unambiguous=False)  # receiver-blind
        Gu = coupling_graph(referenced, strict, unambiguous=True)   # unambiguous provider
        Gw = coupling_graph(referenced, wide, unambiguous=False)    # provision robustness
        if Gw.number_of_edges() > Gb.number_of_edges():
            wide_adds += 1
        rowrec = {"iid": rec["instance_id"], "nfiles": nfiles, "blind": Gb, "uniq": Gu,
                  "edges": [(a, b, d["via"]) for a, b, d in Gu.edges(data=True)]}
        multi.append(rowrec)
        if Gu.number_of_edges() > 0 and len(coupled_examples) < 6:
            coupled_examples.append(rowrec)
        elif Gb.number_of_edges() > 0 and Gu.number_of_edges() == 0 and len(ambiguous_examples) < 3:
            ambiguous_examples.append((rec["instance_id"],
                                       [(a, b, d["via"]) for a, b, d in Gb.edges(data=True)][:3]))
        elif Gb.number_of_edges() == 0 and len(parallel_examples) < 4:
            parallel_examples.append((rec["instance_id"], sorted(sections)))

    n = len(runs)
    n_multi = len(multi)
    cb, db = summarize(multi, "blind")
    cu, du = summarize(multi, "uniq")

    print(f"corpus: SWE-bench_Verified gold patches (human-validated), {n} instances\n")

    print("=== fix size (files per gold patch, source only) ===")
    print(f"distribution     : {dict(sorted(span.items()))}")
    print(f"single-file fixes: {n_single}/{n} ({n_single / n:.0%})")
    print(f"multi-file fixes : {n_multi}/{n} ({n_multi / n:.0%})")
    if n_empty:
        print(f"no-source-hunk   : {n_empty}/{n} (only tests / mode / binary; excluded)")
    print()

    print("=== cross-file dependency structure (the PyG question) ===")
    print(f"of the {n_multi} multi-file fixes, dependency-coupled:")
    print(f"  receiver-blind (any shared symbol)         : {cb} ({cb / n_multi:.0%} of multi, {cb / n:.0%} of all)")
    print(f"  unambiguous-provider (symbol from 1 file)  : {cu} ({cu / n_multi:.0%} of multi, {cu / n:.0%} of all)")
    print(f"  parallel-sibling (no shared symbol)        : {n_multi - cb} ({(n_multi - cb) / n_multi:.0%} of multi)")
    print(f"  (widening provides to module-level assignments adds an edge in {wide_adds} instances)")
    for label, depths, c in (("receiver-blind", db, cb), ("unambiguous", du, cu)):
        if depths:
            deep = sum(1 for d in depths if d >= 2)
            print(f"\ncross-file dependency depth, {label} ({c} coupled fixes):")
            print(f"  median {st.median(depths):.0f}  mean {st.mean(depths):.1f}  "
                  f"min {min(depths)}  max {max(depths)};  >= 2-hop chains: {deep}/{c}")

    print("\n=== example dependency-coupled fixes, unambiguous (edge: A depends on B via symbol) ===")
    for r in coupled_examples:
        print(f"  {r['iid']}  ({r['nfiles']} files, depth {cross_file_depth(r['uniq'])})")
        for a, b, via in r["edges"][:3]:
            print(f"      {a.split('/')[-1]} -> {b.split('/')[-1]}   via {via[:3]}")

    print("\n=== example receiver-blind-only edges (multi-provider; dropped by the unambiguous filter) ===")
    for iid, edges in ambiguous_examples:
        print(f"  {iid}")
        for a, b, via in edges:
            print(f"      {a.split('/')[-1]} -> {b.split('/')[-1]}   via {via[:3]}")

    print("\n=== example parallel-sibling fixes (multi-file, no shared symbol) ===")
    for iid, files in parallel_examples:
        print(f"  {iid}: {['/'.join(f.split('/')[-2:]) for f in files]}")

    deep_u = sum(1 for d in du if d >= 2)
    print(f"\nReading: {n_single / n:.0%} of real gold fixes are single-file, so the "
          f"consequential edit is local for most coding tasks, consistent with the "
          f"lightweight per-node stack. Among the {n_multi} multi-file fixes, a "
          f"receiver-blind symbol match links {cb} of {n} ({cb / n:.1%}); requiring "
          f"the symbol to be provided by exactly one edited file leaves {cu} "
          f"({cu / n:.1%}). That filter removes receiver-blind collisions, but it "
          f"also drops same-interface and base-override cases when more than one "
          f"edited file provides the method name, so the {cb - cu} receiver-blind-only "
          f"fixes are ambiguous rather than automatically false. The "
          f"other {n_multi - cb} multi-file fixes share no symbol at all and are "
          f"parallel-sibling edits, where message passing has no path. Cross-file chains "
          f"are one hop: the unambiguous coupling has depth median "
          f"{st.median(du) if du else 0:.0f}, max {max(du) if du else 0} "
          f"({deep_u}/{cu} at 2 hops); the only 2-hop case under receiver-blind matching "
          f"was a method-name collision (two matplotlib backends calling "
          f"gc.get_antialiased() while a text module defines Text.get_antialiased) and it "
          f"disappears once the match must be unambiguous. This is a patch-local heuristic "
          f"SCREEN with errors both ways: it misses dependencies on unchanged defs and, "
          f"under blind matching, adds same-name collisions; the proper census needs a "
          f"repo-source import graph at base_commit (deferred). What it supports is narrow "
          f"but decision-relevant: observable cross-file dependency propagation in real "
          f"fixes is uncommon ({cu}-{cb} of {n} fixes, under 5%) and shallow (one hop), "
          f"so a heavy cross-file graph model is not justified by gold-patch fix structure "
          f"alone. That is evidence for a lightweight default, not a general rejection of "
          f"cross-file graph modeling.")


if __name__ == "__main__":
    main()
