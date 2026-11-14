"""``python -m auditable.bench [task_id ...] [--methods ID ...] [--golden [--allow-skipped] [--exact]] [--markdown] [--json]``."""
from __future__ import annotations

import argparse
import json
import sys

from auditable.bench._core import leaderboard, list_tasks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auditable.bench",
                                     description="Score agent-auditing methods on the CatchBench board tasks.")
    parser.add_argument("tasks", nargs="*", help="task ids; none lists the tasks")
    parser.add_argument("--methods", nargs="+", help="method ids to score (default: the task's family)")
    parser.add_argument("--golden", action="store_true",
                        help="compare with the CatchBench board; exit 1 on a mismatch, an error row, or a "
                             "golden row that was not scored")
    parser.add_argument("--allow-skipped", action="store_true",
                        help="with --golden, accept golden rows skipped for a missing extra")
    parser.add_argument("--exact", action="store_true",
                        help="with --golden, count a known platform value as a mismatch (the Linux reference run)")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--json", action="store_true", help="print the board as JSON")
    args = parser.parse_args(argv)

    if not args.tasks:
        for task in list_tasks():
            print(f"{task.task_id:24s} {task.family:7s} {', '.join(task.metrics):20s} {task.title}")
        return 0
    status = 0
    for task_id in args.tasks:
        board = leaderboard(task_id, args.methods)
        if args.json:
            print(json.dumps(board.to_dict(), indent=1))
        elif args.markdown:
            print(board.to_markdown(golden=args.golden))
        else:
            print(board)
        if args.golden:
            checks = board.compare_golden()
            if args.exact:
                for c in checks:
                    if c["platform"]:
                        c["ok"] = False
            for c in checks:
                if c["platform"] and not args.exact:
                    print(f"golden platform value: {c['key']} {c['metric']} {c['got']:.3f} "
                          f"(Linux board {c['golden']:.3f}; {c['note']})", file=sys.stderr)
            bad = [c for c in checks if not c["ok"]]
            for c in bad:
                note = f" ({c['note']})" if c["note"] else ""
                print(f"golden mismatch: {c['key']} {c['metric']} {c['got']:.3f} != {c['golden']:.3f}{note}",
                      file=sys.stderr)
            unchecked = board.golden_coverage(allow_skipped=args.allow_skipped)
            if not checks and not unchecked:
                unchecked = [{"key": task_id, "reason": "no golden value was compared"}]
            for u in unchecked:
                print(f"golden unchecked: {u['key']} ({u['reason']})", file=sys.stderr)
            status = status or (1 if bad or unchecked else 0)
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
