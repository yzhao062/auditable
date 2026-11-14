"""Write ``src/auditable/bench/golden/localize__whoandwhen__llm.json`` from CatchBench's board.

    python benchmarks/bench_golden/make_golden_llm.py --board <catchbench>/tests/golden/board.txt

Emits the 31 LLM-judge rows of ``[POST] post_localization :: whoandwhen`` keyed as
``step.llm_<protocol>:<model>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from make_golden import OUT, parse_section  # noqa: E402

SECTION = "[POST] post_localization :: whoandwhen"
_LLM_LABEL_RE = re.compile(r"^llm-judge (all-at-once|binary-search|step-by-step) \((.+)\)$")
_COLUMNS = ("top1", "top3", "mrr")


def build_llm_rows(board_text: str) -> dict:
    """Extract the LLM-judge rows from ``[POST] post_localization :: whoandwhen``."""
    _, rows = parse_section(board_text, SECTION)
    out = {}
    for label, values in rows.items():
        match = _LLM_LABEL_RE.match(label)
        if not match:
            continue
        protocol = match.group(1).replace("-", "_")
        model = match.group(2)
        key = f"step.llm_{protocol}:{model}"
        spec = {col: values[col] for col in _COLUMNS if col in values}
        spec["board_label"] = label
        out[key] = spec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--catchbench-version", default="0.1.3")
    args = ap.parse_args()
    raw = open(args.board, "rb").read()
    text = raw.decode("utf-8")
    source = {
        "catchbench": args.catchbench_version,
        "file": "tests/golden/board.txt",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    rows = build_llm_rows(text)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "localize__whoandwhen__llm.json")
    payload = {"task": "localize/whoandwhen", "source": source, "section": SECTION, "rows": rows}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    print(f"localize/whoandwhen (llm): {len(rows)} rows -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
