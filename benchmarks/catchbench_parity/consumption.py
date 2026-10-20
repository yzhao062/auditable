"""Consumption check: compare a library board with a perturbed library board.

Every manifest item's row must change in every board section of its task where it appears
(main table, breakdowns, robustness tables), and every row outside the manifest must stay
the same. Sections are matched by order and header, rows by label. The two boards must
have the same sections and, within each section, the same row labels in the same order, so
a row that disappears from the perturbed board fails the check instead of counting as a
change.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from catchbench_parity import manifest

Section = Tuple[str, List[str]]


def parse_board(text: str) -> List[Section]:
    sections: List[Section] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("  "):
            if sections:
                sections[-1][1].append(line)
        else:
            sections.append((line, []))
    return sections


_VALUE = re.compile(r"\s+[-+>.]?\d")


def _label(row: str) -> str:
    """The row's text up to its first value (the whole row when it has none)."""
    s = row.strip()
    m = _VALUE.search(s)
    return s[:m.start()] if m else s


def _row(rows: List[str], label: str):
    """The row whose label is exactly ``label``: the label is followed by the end of the
    line or by its first value, so "random" does not match "random (matched)"."""
    for r in rows:
        s = r.strip()
        if s.startswith(label) and (len(s) == len(label) or _VALUE.match(s[len(label):])):
            return r
    return None


def _row_identity_mismatch(lib: List[Section], per: List[Section]) -> Optional[str]:
    """The header of the first section whose row labels differ between the boards in
    content, order, or count, or None when every section has the same rows."""
    for (h, rows_l), (_, rows_p) in zip(lib, per):
        if [_label(r) for r in rows_l] != [_label(r) for r in rows_p]:
            return h
    return None


def _labelled_by_manifest(row: str, task_sections: Dict[str, List[str]], header: str) -> bool:
    for label, prefixes in task_sections.items():
        if any(header.startswith(p) for p in prefixes) and _row([row], label) is not None:
            return True
    return False


def consumption(library_board: str, perturbed_board: str) -> dict:
    manifest.load_adapters()
    lib, per = parse_board(library_board), parse_board(perturbed_board)
    if [h for h, _ in lib] != [h for h, _ in per]:
        return {"passed": False, "error": "the two boards have different section headers"}
    mismatch = _row_identity_mismatch(lib, per)
    if mismatch is not None:
        return {"passed": False, "error": f"the two boards have different rows in section {mismatch!r}"}
    results, failed = {}, 0
    for it in manifest.ITEMS:
        prefixes = manifest.SECTIONS[it.task_id]
        seen = changed = 0
        unchanged_in = []
        for (h, rows_l), (_, rows_p) in zip(lib, per):
            if not any(h.startswith(p) for p in prefixes):
                continue
            a, b = _row(rows_l, it.row_label), _row(rows_p, it.row_label)
            if a is None:
                continue
            seen += 1
            if a != b:
                changed += 1
            else:
                unchanged_in.append(h)
        main = any(h.startswith("[") and _row(r, it.row_label) is not None
                   for h, r in lib if any(h.startswith(p) for p in prefixes))
        ok = main and seen > 0 and not unchanged_in
        failed += not ok
        results[" | ".join(k for k in it.key if k)] = {"sections": seen, "changed": changed,
                                                         "unchanged_in": unchanged_in, "in_main_table": main,
                                                         "passed": ok}
    for section, label, source in manifest.DERIVED_ROWS:
        rows = [(r_l, r_p) for (h, rows_l), (_, rows_p) in zip(lib, per) if h.startswith(section)
                for r_l, r_p in zip(rows_l, rows_p) if _row([r_l], label) is not None]
        ok = bool(rows) and all(a != b for a, b in rows)
        failed += not ok
        results[f"derived | {section} | {label}"] = {"sections": len(rows), "changed": sum(a != b for a, b in rows),
                                                     "unchanged_in": [] if ok else [section], "in_main_table": True,
                                                     "passed": ok, "source": " | ".join(k for k in source if k)}
    labels: Dict[str, List[str]] = {}
    for it in manifest.ITEMS:
        labels.setdefault(it.row_label, []).extend(manifest.SECTIONS[it.task_id])
    unexpected = []
    for (h, rows_l), (_, rows_p) in zip(lib, per):
        for a, b in zip(rows_l, rows_p):
            derived = any(h.startswith(sec) and _row([a], lab) is not None for sec, lab, _ in manifest.DERIVED_ROWS)
            if a != b and not derived and not _labelled_by_manifest(a, labels, h):
                unexpected.append(f"{h} :: {a.strip()[:60]}")
    return {"passed": failed == 0 and not unexpected, "items": results, "failed_items": failed,
            "unexpected_changes": unexpected}
