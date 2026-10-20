"""Command-line entry for the CatchBench parity harness.

    PYTHONHASHSEED=0 GRADE_DIR=<grade@3839a57> python benchmarks/catchbench_parity/run_parity.py <command>

Commands:
  env                          check the reference stack and print it
  coverage                     list manifest items and those without a per-sample check
  checks [--only S] [--modules M,N] [--json P]
                               run the per-sample comparisons (substring filter on the label)
  board --out P [--perturb]    run the full board with the manifest applied (sentinels on)
  consumption LIB PERTURBED    compare a library board with a perturbed one
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catchbench_parity._env import check_environment  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    sub.add_parser("coverage")
    c = sub.add_parser("checks")
    c.add_argument("--only", default="")
    c.add_argument("--json", default="")
    c.add_argument("--modules", default="", help="comma-separated adapter modules (default: all)")
    b = sub.add_parser("board")
    b.add_argument("--out", required=True)
    b.add_argument("--perturb", action="store_true")
    b.add_argument("--report", default="")
    k = sub.add_parser("consumption")
    k.add_argument("library_board")
    k.add_argument("perturbed_board")
    k.add_argument("--json", default="")
    args = p.parse_args(argv)

    env = check_environment()
    print(json.dumps(env), file=sys.stderr)
    if args.cmd == "env":
        return 0
    if args.cmd == "coverage":
        from catchbench_parity import manifest
        from catchbench_parity.checks import missing_checks

        manifest.load_adapters()
        for it in manifest.ITEMS:
            print(f"{it.task_id:20s} {it.row_label:44s} -> {it.method_id}")
        print(f"{len(manifest.ITEMS)} items, {len(manifest.SENTINELS)} sentinels, "
              f"{len(manifest.FEATURE_ITEMS)} feature items")
        missing = missing_checks()
        for key in missing:
            print("NO CHECK:", key)
        return 1 if missing else 0
    if args.cmd == "checks":
        from catchbench_parity.checks import run_checks

        results = run_checks(args.only, [m for m in args.modules.split(",") if m] or None)
        if args.json:
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"env": env, "results": results}, fh, indent=1, default=str)
        failed = [k for k, v in results.items() if not v["passed"]]
        print(f"{len(results) - len(failed)}/{len(results)} checks passed")
        return 1 if failed or not results else 0
    if args.cmd == "board":
        from catchbench_parity import perturb

        if args.perturb:
            os.environ[perturb.ENV] = "1"
        from catchbench_parity.board import run_board

        report = run_board(args.out)
        report["env"] = env
        report["perturb"] = bool(args.perturb)
        if args.report:
            with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(report, fh, indent=1, default=str)
        print(json.dumps({k: report[k] for k in ("unused", "sentinel_hits")}), file=sys.stderr)
        return 1 if report["unused"] or report["sentinel_hits"] else 0
    if args.cmd == "consumption":
        from catchbench_parity.consumption import consumption

        with open(args.library_board, encoding="utf-8") as fh:
            lib = fh.read()
        with open(args.perturbed_board, encoding="utf-8") as fh:
            per = fh.read()
        res = consumption(lib, per)
        for label, r in res.get("items", {}).items():
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {label}  changed {r['changed']}/{r['sections']} sections"
                  + (f"  unchanged in: {r['unchanged_in']}" if r["unchanged_in"] else ""))
        for u in res.get("unexpected_changes", []):
            print("UNEXPECTED CHANGE:", u)
        if args.json:
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(res, fh, indent=1)
        print("consumption", "PASS" if res["passed"] else "FAIL")
        return 0 if res["passed"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
