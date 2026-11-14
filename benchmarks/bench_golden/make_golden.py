"""Write ``src/auditable/bench/golden/<task>.json`` from CatchBench's committed golden board.

    python benchmarks/bench_golden/make_golden.py --board <catchbench>/tests/golden/board.txt --catchbench <catchbench>

Each task below names the board sections it reproduces and how a board row label maps to the
entry key of a library row. The values are copied as printed (three decimals). Rows whose
numbers depend on torch carry ``tol`` 0.005, the tolerance CatchBench's own board check applies;
rows with a known platform difference carry a ``note`` and the value measured on the other
platform under ``accept``. The board file's SHA-256 and the CatchBench version go into ``source``
so a reader can trace every number. The LLM-judge rows of ``localize/whoandwhen`` are written by
``make_golden_llm.py`` into their own file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "src", "auditable", "bench", "golden")

TORCH_TOL = 0.005
TORCH_METHODS = {"run.pygod_dominant", "run.pygod_conad", "run.pygod_anomalydae", "run.pygod_gaan",
                 "run.guardian_lite", "run.gsafeguard_lite", "step.pygod_dominant_ww8"}

# Known Mac versus Linux differences of the board (PLAN-auditable-release-m3 gate 8): the golden
# board is the Linux one.
PLATFORM_NOTES = {
    "pyod-lof": "macOS arm64 prints 0.582 on SWE-Gym; the golden board is Linux x86-64",
    "pyod-knn": "macOS arm64 may differ by up to 0.002; the golden board is Linux x86-64",
    "pygod-anomalydae": "torch row; macOS arm64 may differ by up to 0.04",
}
# (task_id, board label) -> {metric: [values measured on another platform]}
PLATFORM_ACCEPT = {
    ("detect/swegym", "pyod-lof"): {"roc_auc": [0.582]},
    ("detect/tau", "pyod-knn"): {"roc_auc": [0.574]},
    ("detect/tau", "pygod-anomalydae"): {"roc_auc": [0.529]},
}

_LIVE_KEYS = {
    "random": "run.random",
    "size (flat)": "run.lr_flat",
    "auditable (size+deps)": "run.lr_flatdep",
    "full": "run.lr_full",
    "pyod (ECOD)": "run.pyod_ecod",
    "dep-span (online)": "online.dep_span",
}
_DETECT = {"roc_auc": "roc_auc"}

# task_id -> spec. ``sections``: [(board header, {board column: library metric})]; ``family``:
# the card family whose aliases name the rows when ``keys`` is None; ``keys``: an explicit
# board-label -> entry-key map; ``partial``: rows not in ``keys`` are left to another file;
# ``live_block``: which "LIVE streaming early-warning" block holds the per-prefix values.
TASKS = {
    "detect/swegym": {"sections": [("[POST] post_detection :: swegym", _DETECT)], "family": "run"},
    "detect/tau": {"sections": [("[POST] post_detection :: tau", _DETECT)], "family": "run"},
    "localize/whoandwhen": {
        "sections": [("[POST] post_localization :: whoandwhen", {"top1": "top1", "top3": "top3", "mrr": "mrr"})],
        "keys": {
            "random": "step.random",
            "auditable (blast)": "step.blast",
            "position": "step.position_supervised",
            "pygod (graph AD)": "step.pygod_dominant_ww8",
            "exec-rank (sup.)": "step.exec_rank",
        },
        "partial": True,  # the LLM-judge rows: make_golden_llm.py
    },
    "live/swegym": {"sections": [("[LIVE] live_streaming :: swegym", {"prefix_auc": "prefix_auc"})],
                    "keys": _LIVE_KEYS, "live_block": 0},
    "live/tau": {"sections": [("[LIVE] live_streaming :: tau", {"prefix_auc": "prefix_auc"})],
                 "keys": _LIVE_KEYS, "live_block": 1},
    "pre/over_privilege": {
        "sections": [
            ("[PRE] pre_over_privilege :: multi",
             {"f1": "f1", "precision": "precision", "recall": "recall", "coverage": "coverage"}),
            ("[PRE] pre_over_privilege :: F1 by source",
             {"crewai": "f1@crewai", "n8n": "f1@n8n", "mcp": "f1@mcp",
              "injecagent": "f1@injecagent", "sweagent": "f1@sweagent", "synthetic": "f1@synthetic"}),
        ],
        "keys": {
            "flag_all": "config.flag_all",
            "flag_none": "config.flag_none",
            "flag_risky_perms": "config.flag_risky_perms",
            "owasp_excess_permissions": "config.owasp_excess_permissions",
            "owasp_excess_functionality": "config.owasp_excess_functionality",
            "owasp_privilege_escalation": "config.owasp_privilege_escalation",
            "unrequested_high_impact": "config.unrequested_high_impact",
            "sensitive_access": "config.sensitive_access",
            "owasp_asi_combined": "config.owasp_asi_combined",
            "oracle_privilege_diff": "config.oracle_privilege_diff",
            "llm_judge_needed(llama-3.3-70b)": "config.llm_needed",
        },
    },
}

# Rows that are not on the board but come from CatchBench's committed size audits: the flexible
# size control. (task_id, entry key) -> (audit, corpus). The audits ran on Windows, and the spline
# moves in the fourth decimal across platforms, so these rows carry a 0.001 tolerance.
AUDIT_TOL = 0.001
AUDIT_ROWS = {
    ("detect/swegym", "run.lr_spline_flat"): ("detection", "swegym"),
    ("detect/tau", "run.lr_spline_flat"): ("detection", "tau"),
    ("live/swegym", "run.lr_spline_flat"): ("live", "swegym"),
    ("live/tau", "run.lr_spline_flat"): ("live", "tau"),
}
AUDIT_FILES = {"detection": "tools/detection_audit_results.json", "live": "tools/live_size_control_audit_results.json"}
_PREFIXES = ("25", "50", "75", "100")


def audit_row(kind: str, corpus: str, audit: dict, source: dict):
    """The ``size (spline)`` arm of one CatchBench size audit, rounded as the board prints."""
    if kind == "detection":
        row = {"roc_auc": round(audit["corpora"][corpus]["arms"]["size (spline)"]["mean_roc_auc"], 3)}
    else:
        by = audit["corpora"][corpus]["by_prefix"]
        aucs = [by[p]["arms"]["size (spline)"]["mean_fold_roc_auc"] for p in _PREFIXES]
        row = {"prefix_auc": round(sum(aucs) / len(aucs), 3)}
        row.update({f"auc@{p}": round(a, 3) for p, a in zip(_PREFIXES, aucs)})
    row.update({"tol": AUDIT_TOL, "board_label": "size (spline)", "source": source,
                "note": "a CatchBench size-audit arm, not a board row; the audit ran on Windows"})
    return row


_TRAILERS = ("label source per column:", "pooled F1 ", "abstained ")


def parse_section(text: str, header: str):
    """``{label: {column: value}}`` for the table under ``header`` (e.g. ``[POST] post_detection :: swegym``)."""
    lines = text.splitlines()
    start = lines.index(header)
    cols = lines[start + 1].split()[1:]
    rows = {}
    for line in lines[start + 2:]:
        stripped = line.strip()
        if not stripped or line.startswith("[") or stripped.startswith(_TRAILERS):
            break
        parts = line.rstrip().rsplit(None, len(cols))
        label, values = parts[0].strip(), parts[1:]
        rows[label] = {c: float(v) for c, v in zip(cols, values)}
    return cols, rows


def parse_live_breakdown(text: str, occurrence: int):
    """The ``occurrence``-th "LIVE streaming early-warning" block as
    ``{label: {"auc@25": float, ..., "t2d": str}}``; ``t2d`` is printed as text (``25%``, ``>100%``)."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("LIVE streaming early-warning (")]
    if occurrence >= len(starts):
        raise SystemExit(f"missing LIVE streaming early-warning block #{occurrence}")
    start = starts[occurrence]
    cols = lines[start + 1].split()[1:]
    rows = {}
    for line in lines[start + 2:]:
        if not line.strip() or line.startswith("["):
            break
        parts = line.rstrip().rsplit(None, len(cols))
        label, values = parts[0].strip(), parts[1:]
        row = {}
        for col, val in zip(cols, values):
            if col == "t2d":
                row["t2d"] = val
            elif col.endswith("%"):
                row[f"auc@{col[:-1]}"] = float(val)
        rows[label] = row
    return rows


def card_keys(family: str):
    """``{board label: method_id}`` from the cards' ``board_aliases``, refusing ambiguous labels."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    from auditable.methods import list_methods

    out = {}
    for card in list_methods(family=family):
        for alias in card.board_aliases:
            if alias in out:
                raise SystemExit(f"board label {alias!r} maps to {out[alias]} and {card.method_id}; give the task explicit keys")
            out[alias] = card.method_id
    return out


def build(task_id: str, board_text: str):
    spec = TASKS[task_id]
    keys = spec.get("keys") or card_keys(spec["family"])
    partial = spec.get("partial", False)
    breakdown = parse_live_breakdown(board_text, spec["live_block"]) if "live_block" in spec else {}
    out = {}
    first_labels = None
    for header, columns in spec["sections"]:
        _, rows = parse_section(board_text, header)
        if first_labels is None:
            first_labels = set(rows)
            if spec.get("keys"):
                missing = sorted(set(spec["keys"]) - set(rows))
                if missing:
                    raise SystemExit(f"{task_id}: mapped board rows missing from {header!r}: {missing}")
        elif set(rows) != first_labels:
            raise SystemExit(f"{task_id}: section {header!r} rows differ from {spec['sections'][0][0]!r}")
        for label, values in rows.items():
            if label not in keys:
                if partial:
                    continue
                raise SystemExit(f"{task_id}: board row {label!r} has no library method")
            key = keys[label]
            row = out.setdefault(key, {})
            for c, v in values.items():
                if c in columns:
                    row[columns[c]] = v
            row.update(breakdown.get(label, {}))
            if key in TORCH_METHODS:
                row["tol"] = TORCH_TOL
            if label in PLATFORM_NOTES:
                row["note"] = PLATFORM_NOTES[label]
            if (task_id, label) in PLATFORM_ACCEPT:
                row["accept"] = PLATFORM_ACCEPT[(task_id, label)]
            row["board_label"] = label
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--catchbench-version", default="0.1.3")
    ap.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    ap.add_argument("--catchbench", help="a CatchBench 0.1.3 checkout: adds the size-audit rows (AUDIT_ROWS)")
    args = ap.parse_args()
    if not args.catchbench and any(tid in args.tasks for tid, _ in AUDIT_ROWS):
        ap.error("--catchbench is required to write the size-audit rows of " +
                 ", ".join(sorted({tid for tid, _ in AUDIT_ROWS if tid in args.tasks})))
    raw = open(args.board, "rb").read()
    text = raw.decode("utf-8")
    source = {"catchbench": args.catchbench_version, "file": "tests/golden/board.txt",
              "sha256": hashlib.sha256(raw).hexdigest()}
    os.makedirs(OUT, exist_ok=True)
    audits = {}
    if args.catchbench:
        for kind, rel in AUDIT_FILES.items():
            raw_audit = open(os.path.join(args.catchbench, rel), "rb").read()
            audits[kind] = (json.loads(raw_audit), {"catchbench": args.catchbench_version, "file": rel,
                                                    "sha256": hashlib.sha256(raw_audit).hexdigest()})
    for task_id in args.tasks:
        rows = build(task_id, text)
        for (tid, key), (kind, corpus) in AUDIT_ROWS.items():
            if tid == task_id and kind in audits:
                rows[key] = audit_row(kind, corpus, *audits[kind])
        headers = [h for h, _ in TASKS[task_id]["sections"]]
        path = os.path.join(OUT, task_id.replace("/", "__") + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"task": task_id, "source": source, "section": headers[0] if len(headers) == 1 else headers,
                       "rows": rows}, fh, indent=1)
            fh.write("\n")
        print(f"{task_id}: {len(rows)} rows -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
