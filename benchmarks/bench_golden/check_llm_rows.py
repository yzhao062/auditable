"""Verify all 31 cached LLM-judge rows of ``localize/whoandwhen`` against CatchBench's golden table.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONHASHSEED=0 PYTHONPATH=src python benchmarks/bench_golden/check_llm_rows.py

Blocks outbound network connections (``socket.socket.connect`` and ``urllib.request.urlopen``)
and sets ``HF_HUB_OFFLINE=1`` so any accidental download or API call fails immediately.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _forbid_network() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def _blocked(*args, **kwargs):
        raise RuntimeError("network access is forbidden during check_llm_rows")

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    urllib.request.urlopen = _blocked  # type: ignore[assignment]


def main() -> int:
    _forbid_network()

    from auditable.bench.llm_rows import _load_whoandwhen_population, llm_entries, score_llm_entry

    golden_path = os.path.join(SRC, "auditable", "bench", "golden", "localize__whoandwhen__llm.json")
    with open(golden_path, encoding="utf-8") as fh:
        golden = json.load(fh)["rows"]

    population = _load_whoandwhen_population()
    entries = llm_entries()
    assert len(entries) == 31, f"expected 31 LLM entries, got {len(entries)}"
    assert len(population.runs) == 126, f"expected 126 runs, got {len(population.runs)}"

    header = (
        f"{'key':<44} {'label':<42} "
        f"{'top1 (gold)':>15} {'top3 (gold)':>15} {'mrr (gold)':>15}  status"
    )
    print(header)
    print("-" * len(header))

    mismatches = 0
    for entry in entries:
        scores = score_llm_entry(entry, population)
        spec = golden.get(entry.key)
        if spec is None:
            mismatches += 1
            print(f"{entry.key:<44} {entry.label:<42} MISSING GOLDEN")
            continue
        ok = True
        cells = []
        for metric in ("top1", "top3", "mrr"):
            got = round(scores[metric], 3)
            want = round(float(spec[metric]), 3)
            if abs(got - want) > 1e-9:
                ok = False
            cells.append(f"{got:.3f} ({want:.3f})")
        if not ok:
            mismatches += 1
        status = "OK" if ok else "MISMATCH"
        print(f"{entry.key:<44} {entry.label:<42} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15}  {status}")

    print()
    print(f"Checked {len(entries)}/31 combinations on {len(population.runs)} Who&When runs; mismatches={mismatches}.")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
