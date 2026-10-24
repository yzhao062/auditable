"""Cross-check auditable's agentpile adapter against GRADE 3839a57.

    PYTHONHASHSEED=0 GRADE_DIR=<grade@3839a57> python benchmarks/agentpile_crosscheck/run_crosscheck.py \\
        --expected <dir with the GRADE expected *.jsonl or *.jsonl.gz> \\
        --exceptions <agentpile checkout>/tests/fixtures/grade/exceptions.json \\
        --data <dir with one folder per dataset holding its pinned files, e.g. ~/.cache/agentpile> \\
        [--corpus who_and_when tau_bench ...] [--out <dir for report.json and manifests>]

For each corpus it loads the runs an expected file names with ``agentpile.load``, converts them
with ``auditable.from_agentpile(run, deps="grade", convention="grade")``, and compares:

1. the expected file itself with ``baseline.json`` (hash of the decompressed bytes, counts,
   GRADE commit markers), and the source files the comparison reads with ``baseline.json`` (one
   SHA-256 per file, one digest over the list): the run files the expected runs name, plus any
   file the reader joins into every run, such as AgentRewardBench's annotation CSV or SWE-rebench's
   tool definitions. A changed reference or a changed input thus fails before anything is
   compared. The per-file hashes, the keys of loaded runs GRADE's selection left out (its filters,
   caps and balancing are not ported, so no reason is recorded per run), and the feature column
   names go to the report;
2. every run's steps (``idx``, ``agent``, ``kind``, ``deps``) with GRADE's, and its run identity
   and failure label. A difference passes only when the agentpile exception list holds an entry
   with the same run, step, field, expected and got values; an unused entry for the corpus fails;
3. on every run without a step difference, the inputs the methods read: ``who_when_step_matrix``,
   the six GRADE layers, and ``to_pyg`` of the graph, computed from GRADE's ``build_graph`` (with
   its loader's parameters) on the expected steps and through the adapter's public entry points on
   the converted trace. An AgentRewardBench run whose only differences are its listed E17 tail is
   compared after the listed tail steps are removed from the reference, and counted apart;
4. for Who&When, the 126-run localization population of GRADE and CatchBench: ``step_targets``
   against the raw ``mistake_step`` of each source file;
5. CatchBench PRE: ``configs_from_agentpile`` on every row against CatchBench's ``method_view``, on
   the fields the config methods read, and the findings of the seven rule methods.

The adapter is never given the expected steps; the reference side never calls the adapter. Exit
status 0 means every selected comparison passed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
GRADE_PIN = "3839a57ac165d58a807fce0a3ff38346732ee936"
CORPORA = ("who_and_when", "tau_bench", "tau2_bench", "swe_rebench_openhands", "swe_gym_openhands",
           "swe_agent_nebius", "agent_reward_bench")
#: GRADE's loaders: how each corpus's graphs are built (agent_failure_detection.py, agent_failure_localization.py)
GRAPH = {name: ({"dependency": "full_context"} if name == "who_and_when"
                else {"dependency": "explicit", "shared_resource": False}) for name in CORPORA}
LAYERS = ("flat", "exec", "full", "flatdens", "flatdep", "depnorm")
#: files a reader joins into every run besides the run files the expected lines name
AUXILIARY = {"agent_reward_bench": ("data/annotations.csv",), "swe_rebench_openhands": ("tools.json",)}


# -- reference and environment ----------------------------------------------------------------------

def grade_build_graph():
    grade_dir = os.environ.get("GRADE_DIR")
    if not grade_dir or not os.path.isdir(grade_dir):
        sys.exit(f"set GRADE_DIR to a GRADE checkout at {GRADE_PIN}")
    head = subprocess.run(["git", "-C", grade_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != GRADE_PIN:
        sys.exit(f"GRADE_DIR is at {head or 'an unknown commit'}, expected {GRADE_PIN}")
    sys.path.insert(0, str(Path(grade_dir) / "src"))
    from grade import build_graph

    return build_graph


def read_expected(directory: Path, name: str, baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The expected lines of one corpus, after checking the file against its frozen record."""
    plain, packed = directory / f"{name}.jsonl", directory / f"{name}.jsonl.gz"
    raw = plain.read_bytes() if plain.exists() else gzip.open(packed, "rb").read()
    lines = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    got = {"sha256": hashlib.sha256(raw).hexdigest(), "n_runs": len(lines),
           "n_steps": sum(len(e["steps"]) for e in lines),
           "n_dep_edges": sum(len(s["deps"]) for e in lines for s in e["steps"]),
           "grade_commit": sorted({str(e.get("grade_commit")) for e in lines})}
    want = {k: baseline[k] for k in got}
    if got != want:
        raise SystemExit(f"{name}: the expected file differs from baseline.json: {got} != {want}")
    return lines


def source_manifest(root: Path, files: Iterable[str]) -> Dict[str, Any]:
    """SHA-256 of each source file (path relative to ``root``) and one digest over the sorted list."""
    lines = []
    for rel in sorted(set(files)):
        h = hashlib.sha256()
        with open(Path(root) / rel, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{rel}\t{h.hexdigest()}")
    return {"n_files": len(lines), "digest": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
            "files": lines}


def check_inputs(name: str, manifest: Dict[str, Any], frozen: Optional[Dict[str, Any]]) -> None:
    """Refuse source files that differ from the ones the frozen comparison read."""
    want = {k: (frozen or {}).get(k) for k in ("n_files", "digest")}
    got = {k: manifest[k] for k in ("n_files", "digest")}
    if got != want:
        raise SystemExit(f"{name}: the source files differ from baseline.json: {got} != {want}")


# -- run identity (ported from agentpile tests/compat/grade.py, the form the expected files use) -----

def identity(corpus: str, run: Any) -> Tuple[str, int, List[Any]]:
    src = run.source
    meta = run.dataset.get("meta") or {}
    if corpus == "who_and_when":
        return src["file"], 0, [run.task.get("id")]
    if corpus in ("tau_bench", "tau2_bench"):
        return src["file"], src["row"], [run.dataset.get("task_name"), meta.get("id")]
    if corpus == "swe_rebench_openhands":
        return src["file"], src["row_index"], [run.id, run.task.get("id")]
    if corpus == "swe_gym_openhands":
        return src["file"], src["row_in_file"], [src.get("instance_id"), src.get("run_id")]
    if corpus == "swe_agent_nebius":
        return src["file"], int(src["record_key"].rsplit(":", 1)[1]), [run.dataset.get("instance_id"),
                                                                       run.dataset.get("model_name")]
    if corpus == "agent_reward_bench":
        return src["file"], 0, [run.dataset.get("benchmark"), run.id.split("/")[1], run.task.get("id")]
    raise KeyError(corpus)


def run_key(corpus: str, run: Any) -> str:
    file, row, ids = identity(corpus, run)
    return ":".join([corpus, quote(str(file), safe=""), str(row)] + [quote(str(i), safe="") for i in ids])


def load_runs(corpus: str, root: Optional[str], expected: Sequence[Dict[str, Any]]) -> Iterator[Any]:
    """The loaded runs that can hold the expected ones; ``root`` is the local copy of the pinned files."""
    import agentpile as ap

    files = sorted({e["source"]["file"] for e in expected})
    rows = 1 + max(e["source"]["row"] for e in expected)
    if corpus == "tau_bench":
        return ap.load(corpus, root=root, model=[Path(f).stem for f in files])
    if corpus == "tau2_bench":
        if len(files) != 1:
            raise ValueError(f"the tau2 expectation is one model file, got {files}")
        return ap.load(corpus, root=root, model=Path(files[0]).stem)
    if corpus in ("swe_rebench_openhands", "swe_agent_nebius"):
        return ap.load(corpus, root=root, limit=rows)
    if corpus == "agent_reward_bench":
        return ap.load(corpus, root=root, benchmark=sorted({f.split("/")[1] for f in files}))
    return ap.load(corpus, root=root)


# -- comparison ------------------------------------------------------------------------------------

def _shape(step: Dict[str, Any]) -> Dict[str, Any]:
    return {k: step[k] for k in ("idx", "agent", "kind", "deps")}


def diff_steps(label: str, key: str, expected: Sequence[Dict[str, Any]], got: Sequence[Dict[str, Any]]):
    """Differences in the form of agentpile's exception list (``n_steps``, ``tail``, and per-step fields)."""
    out: List[Dict[str, Any]] = []

    def add(step, fld, exp, have):
        out.append({"corpus": label, "run_key": key, "step": step, "field": fld, "expected": exp, "got": have})

    if len(expected) != len(got):
        add(None, "n_steps", len(expected), len(got))
        n = min(len(expected), len(got))
        add(n, "tail", [_shape(s) for s in expected[n:]], [_shape(s) for s in got[n:]])
    for i, (e, g) in enumerate(zip(expected, got)):
        if e["idx"] != g["idx"]:
            add(i, "idx", e["idx"], g["idx"])
        for fld in ("agent", "kind", "deps"):
            if e[fld] != g[fld]:
                add(e["idx"], fld, e[fld], g[fld])
    return out


def _same(entry: Dict[str, Any], diff: Dict[str, Any]) -> bool:
    return all(entry.get(k) == diff[k] for k in ("corpus", "run_key", "step", "field", "expected", "got"))


def _equal(a: Any, b: Any) -> bool:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def method_inputs(trace: Any) -> Dict[str, Any]:
    """What the run and step methods read from a trace, through their public entry points."""
    from auditable.featurize import grade_layer, to_pyg, who_when_step_matrix
    from auditable.featurize import normalize_trace

    out = {"who_when_step_matrix": who_when_step_matrix(trace)}
    for layer in LAYERS:
        out[f"layer:{layer}"] = grade_layer(trace, layer)[1]
    x, edges = to_pyg(normalize_trace(trace).typed_graph())
    out["pyg:x"], out["pyg:edge_index"] = x, edges
    return out


def reference_inputs(corpus: str, steps: Sequence[Dict[str, Any]], build_graph) -> Dict[str, Any]:
    """The same inputs computed from GRADE's graph of the expected steps."""
    import copy

    steps = copy.deepcopy(list(steps))
    G = build_graph(steps, **GRAPH[corpus])
    run = {"steps": steps, "G": G}
    if corpus != "who_and_when":  # the step matrix reads the full-context handoff graph GRADE builds for it
        run_ww = {"steps": steps, "G": build_graph(steps, dependency="full_context", shared_resource=False)}
    else:
        run_ww = run
    from auditable.featurize import grade_layer, to_pyg, who_when_step_matrix

    out = {"who_when_step_matrix": who_when_step_matrix(run_ww)}
    for layer in LAYERS:
        out[f"layer:{layer}"] = grade_layer(G, layer)[1]
    x, edges = to_pyg(G)
    out["pyg:x"], out["pyg:edge_index"] = x, edges
    return out


def compare_corpus(corpus: str, root: Optional[str], expected: Sequence[Dict[str, Any]],
                   exceptions: Sequence[Dict[str, Any]], build_graph) -> Dict[str, Any]:
    import auditable as au

    label = corpus
    by_key = {e["run_key"]: e for e in expected}
    rep: Dict[str, Any] = {"corpus": corpus, "runs_expected": len(expected), "runs_compared": 0, "steps_compared": 0,
                           "join_errors": [], "differences": [], "inputs_equal": 0, "inputs_equal_e17_aligned": 0,
                           "inputs_failed": [], "warnings": 0}
    views: Dict[str, Any] = {}
    seen = set()
    pending: List[Tuple[str, Any, Dict[str, Any], List[Dict[str, Any]]]] = []
    # loaded runs GRADE's selection left out (its filters, caps and balancing, which this harness does not port)
    not_expected: List[str] = []
    for run in load_runs(corpus, root, expected):
        key = run_key(corpus, run)
        want = by_key.get(key)
        if want is None:
            not_expected.append(key)
            continue
        seen.add(key)
        file, row, ids = identity(corpus, run)
        src = want["source"]
        if (src["file"], src["row"], list(src["ids"].values())) != (file, row, ids):
            rep["join_errors"].append(f"{key}: source {src['file']!r} {src['row']} {list(src['ids'].values())} != "
                                      f"{file!r} {row} {ids}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            view = au.from_agentpile(run, deps="grade", convention="grade")
        rep["warnings"] += len(caught)
        if src.get("label_failed") != view.meta["label_failed"]:
            rep["join_errors"].append(f"{key}: label_failed {src.get('label_failed')!r} != {view.meta['label_failed']!r}")
        rep["runs_compared"] += 1
        rep["steps_compared"] += len(want["steps"])
        diffs = diff_steps(label, key, want["steps"], view.steps)
        rep["differences"].extend(diffs)
        pending.append((key, view, want, diffs))
        views[key] = view
    rep["missing_runs"] = sorted(set(by_key) - seen)
    rep["runs_loaded_not_expected"] = len(not_expected)
    rep["runs_loaded_not_expected_keys"] = sorted(not_expected)
    listed, unlisted, used = [], [], set()
    for diff in rep["differences"]:
        hit = next((i for i, e in enumerate(exceptions) if _same(e, diff)), None)
        if hit is None:
            unlisted.append(diff)
        else:
            used.add(hit)
            listed.append(dict(diff, exception=exceptions[hit].get("id")))
    rep["listed"] = len(listed)
    rep["listed_by_exception"] = {}
    for d in listed:
        rep["listed_by_exception"][d["exception"]] = rep["listed_by_exception"].get(d["exception"], 0) + 1
    rep["unlisted"] = unlisted[:20]
    rep["n_unlisted"] = len(unlisted)
    stale = [e for i, e in enumerate(exceptions) if i not in used and e.get("corpus") == label]
    rep["n_stale"] = len(stale)
    rep["stale"] = stale[:5]
    for key, view, want, diffs in pending:
        ref_steps = want["steps"]
        aligned = False
        if diffs:
            fields = sorted(d["field"] for d in diffs)
            ok = (fields == ["n_steps", "tail"] and all(any(_same(e, d) for e in exceptions) for d in diffs)
                  and next(d for d in diffs if d["field"] == "tail")["got"] == []
                  and all(e.get("id") == "E17" for d in diffs for e in exceptions if _same(e, d)))
            if not ok:
                continue  # a step difference: the step comparison above already decides this run
            n = next(d for d in diffs if d["field"] == "tail")["step"]
            ref_steps, aligned = ref_steps[:n], True
        ref = reference_inputs(corpus, ref_steps, build_graph)
        got = method_inputs(view)
        bad = [k for k in ref if not _equal(ref[k], got[k])]
        if bad:
            rep["inputs_failed"].append({"run_key": key, "fields": bad, "e17_aligned": aligned})
        elif aligned:
            rep["inputs_equal_e17_aligned"] += 1
        else:
            rep["inputs_equal"] += 1
    rep["n_inputs_failed"] = len(rep["inputs_failed"])
    rep["inputs_failed"] = rep["inputs_failed"][:20]
    rep["controls"] = controls(corpus, [p for p in pending if not p[3]][:20], build_graph)
    rep["clean"] = not (rep["missing_runs"] or rep["join_errors"] or unlisted or stale or rep["n_inputs_failed"]
                        or not rep["controls"]["sensitive"])
    rep["differences"] = len(rep["differences"])
    rep["_views"] = views
    return rep


def controls(corpus: str, sample, build_graph) -> Dict[str, Any]:
    """Two mutations of the adapter side that must make the comparison fail on some sampled run: the
    other shared-resource setting, and the dependencies removed. Zero differences above mean something
    only if these are caught."""
    from auditable.featurize import TraceView

    flipped = emptied = 0
    for _key, view, want, _diffs in sample:
        ref = reference_inputs(corpus, want["steps"], build_graph)
        flip = TraceView(steps=view.steps, texts=view.texts, meta=view.meta, shared_resource=not view.shared_resource)
        flipped += any(not _equal(ref[k], v) for k, v in method_inputs(flip).items())
        bare = TraceView(steps=[dict(s, deps=[]) for s in view.steps], texts=view.texts, meta=view.meta,
                         shared_resource=view.shared_resource)
        emptied += any(not _equal(ref[k], v) for k, v in method_inputs(bare).items())
    has_edges = any(s["deps"] for _k, v, _w, _d in sample for s in v.steps)
    return {"sampled": len(sample), "caught_shared_resource_flip": flipped, "caught_deps_removed": emptied,
            "sensitive": bool(sample) and flipped > 0 and (emptied > 0 or not has_edges)}


def who_and_when_localization(expected: Sequence[Dict[str, Any]], views: Dict[str, Any], root: str) -> Dict[str, Any]:
    """GRADE's localization population (agent_failure_localization.load_failed_runs) and its labels."""
    import auditable as au

    manifest, mismatches = [], []
    for line in expected:  # the expected file keeps GRADE's sorted-glob order
        raw = json.loads((Path(root) / line["source"]["file"]).read_text(encoding="utf-8"))
        if str(raw.get("is_correct", "")).lower() != "false":
            continue
        n = len(line["steps"])
        if n < 3:
            continue
        try:
            ms = int(raw.get("mistake_step"))
        except (TypeError, ValueError):
            continue
        if not 0 <= ms < n:
            continue
        got = int(au.step_targets([views[line["run_key"]]])[0])
        manifest.append({"run_key": line["run_key"], "mistake_step": ms})
        if got != ms:
            mismatches.append({"run_key": line["run_key"], "expected": ms, "got": got})
    return {"runs": len(manifest), "mismatches": mismatches, "manifest": manifest,
            "clean": len(manifest) == 126 and not mismatches}


def catchbench_pre(root: str) -> Dict[str, Any]:
    import agentpile as ap

    import auditable as au
    from auditable.featurize import AgentConfig
    from auditable.methods import get_method, list_methods

    from catchbench.pre import PreOverPrivilege

    views = {v["instance_id"]: v for v in PreOverPrivilege().method_view()}
    rules = [c.method_id for c in list_methods(family="config") if "llm" not in c.method_id]
    rep = {"rows": 0, "input_mismatches": [], "finding_mismatches": [], "permission_from_missing": 0, "no_tokens": 0,
           "rules": rules}
    seen = set()
    for run in ap.load("catchbench_pre", root=root):
        (cfg,) = au.configs_from_agentpile(run)
        view = views.get(cfg.instance_id)
        rep["rows"] += 1
        if view is None:
            rep["input_mismatches"].append({"instance_id": cfg.instance_id, "field": "missing in CatchBench"})
            continue
        seen.add(cfg.instance_id)
        ref = AgentConfig.from_view(view)
        if not cfg.spec_tokens and not cfg.task_or_role_spec:
            rep["no_tokens"] += 1
        proj = lambda c: [(d["name"], d.get("type"), d.get("permission_level")) for d in c.declared_capabilities]  # noqa: E731
        for fld, a, b in (("spec_tokens", cfg.spec_tokens, ref.spec_tokens),
                          ("spec_token_overrides", cfg.spec_token_overrides, ref.spec_token_overrides),
                          ("capabilities", proj(cfg), proj(ref))):
            if a != b:
                rep["input_mismatches"].append({"instance_id": cfg.instance_id, "field": fld})
        rep["permission_from_missing"] += sum(not d.get("permission_from") for d in cfg.declared_capabilities)
        for method_id in rules:
            m = get_method(method_id)()
            if m.audit(cfg) != m.audit(ref):
                rep["finding_mismatches"].append({"instance_id": cfg.instance_id, "method": method_id})
    rep["missing_rows"] = sorted(set(views) - seen)
    rep["clean"] = rep["rows"] == len(views) == 1187 and not (rep["input_mismatches"] or rep["finding_mismatches"]
                                                              or rep["missing_rows"] or rep["permission_from_missing"])
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--exceptions", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--corpus", nargs="*", default=list(CORPORA) + ["catchbench_pre"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("run with PYTHONHASHSEED=0")
    build_graph = grade_build_graph()
    frozen = json.loads((HERE / "baseline.json").read_text(encoding="utf-8"))
    baseline, inputs = frozen["corpora"], frozen.get("inputs") or {}
    exceptions = json.loads(args.exceptions.read_text(encoding="utf-8"))["exceptions"]
    from auditable.featurize import NODE_FEATURES, STEP_FEATURES, grade_layer
    from auditable.featurize._grade_projection import REVISIONS

    tiny = [{"idx": 0, "agent": "a", "kind": "decision", "deps": []}]
    report: Dict[str, Any] = {
        "grade": GRADE_PIN, "exceptions_sha256": hashlib.sha256(args.exceptions.read_bytes()).hexdigest(),
        "columns": {"who_when_step_matrix": list(STEP_FEATURES), "pyg_node_features": list(NODE_FEATURES),
                    "layers": {layer: list(grade_layer(tiny, layer)[0]) for layer in LAYERS}}}
    ok = True
    for corpus in args.corpus:
        if corpus == "catchbench_pre":
            root = args.data / "catchbench_pre" / "30a516c77cda902de2f8e56fb27be09ea3034591"
            files = sorted(f.relative_to(root).as_posix() for f in (root / "data" / "pre").glob("*.json"))
        else:
            root = args.data / corpus / REVISIONS[corpus]
            expected = read_expected(args.expected, corpus, baseline[corpus])
            files = [e["source"]["file"] for e in expected] + list(AUXILIARY.get(corpus, ()))
        manifest = source_manifest(root, files)
        check_inputs(corpus, manifest, inputs.get(corpus))
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"inputs-{corpus}.tsv").write_text("\n".join(manifest["files"]) + "\n", encoding="utf-8",
                                                          newline="\n")
        if corpus == "catchbench_pre":
            rep = catchbench_pre(str(root))
        else:
            rep = compare_corpus(corpus, str(root), expected, exceptions, build_graph)
            views = rep.pop("_views")
            if corpus == "who_and_when":
                loc = who_and_when_localization(expected, views, str(root))
                rep["localization"] = {k: v for k, v in loc.items() if k != "manifest"}
                rep["clean"] = rep["clean"] and loc["clean"]
                if args.out:
                    args.out.mkdir(parents=True, exist_ok=True)
                    (args.out / "who_and_when_localization_manifest.json").write_text(
                        json.dumps(loc["manifest"], indent=1) + "\n", encoding="utf-8", newline="\n")
        rep["inputs"] = {k: manifest[k] for k in ("n_files", "digest")}
        left_out = rep.pop("runs_loaded_not_expected_keys", None)
        if args.out and left_out is not None:
            (args.out / f"not-expected-{corpus}.txt").write_text("".join(k + "\n" for k in left_out),
                                                                 encoding="utf-8", newline="\n")
        report[corpus] = rep
        ok = ok and rep["clean"]
        brief = {k: v for k, v in rep.items() if k not in ("unlisted", "stale", "inputs_failed", "manifest")}
        print(json.dumps(brief, default=str), flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8",
                                              newline="\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
