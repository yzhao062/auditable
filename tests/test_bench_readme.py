"""The README leaderboard blocks are generated, never typed.

Each task has a block between ``<!-- board:<task_id>:start -->`` and ``<!-- board:<task_id>:end -->``
in README.md. Its content is exactly ``Board.to_markdown()`` of a board built from the task's default
rows and the golden values (the CatchBench reference). Building that board downloads no data and
fits no model, so the check is fast and offline.

After a change to a golden file, the method registry, a task's default rows, or the board format,
regenerate the README blocks (the ten generated blocks: six boards and four method tables) with::

    python tests/test_bench_readme.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

from auditable.base import _FAMILY_STATE, FAMILIES
from auditable.bench._core import Board, Row, _entry_type, list_tasks
from auditable.methods import list_methods, method_table

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
HERO_BUILDER = README.parent / "figure-src" / "hero" / "build_hero.py"
REGENERATE = "regenerate the README blocks with: python tests/test_bench_readme.py"
TASK_IDS = [task.task_id for task in list_tasks()]


def _readme():
    """The README text; skip when the tests run outside a checkout (the weekly platform job copies
    only tests/ beside an installed package)."""
    if not README.is_file():
        pytest.skip("README.md is not beside the tests (installed-package run)")
    return README.read_text(encoding="utf-8")


def _task(task_id):
    return next(t for t in list_tasks() if t.task_id == task_id)


def _markers(name):
    return f"<!-- {name}:start -->", f"<!-- {name}:end -->"


def _entries(task):
    """The task's default rows, after checking that they are the rows the golden file holds."""
    entries = task.default_entries()
    entry_keys = [e.key for e in entries]
    golden_keys = list(task.golden().get("rows", {}))
    assert len(entry_keys) == len(set(entry_keys)), f"{task.task_id}: duplicate default entry keys {entry_keys}"
    assert set(entry_keys) == set(golden_keys), (
        f"{task.task_id}: the golden rows and the default entries differ\n"
        f"  golden keys only: {sorted(set(golden_keys) - set(entry_keys))}\n"
        f"  entry keys only: {sorted(set(entry_keys) - set(golden_keys))}\n"
        f"  golden keys: {sorted(golden_keys)}\n  entry keys: {sorted(entry_keys)}")
    return entries


def golden_board(task) -> Board:
    """The board of ``task`` with the golden value for every metric of every default row."""
    golden = task.golden().get("rows", {})
    rows = []
    for entry in _entries(task):
        spec = golden[entry.key]
        missing = [m for m in task.metrics if m not in spec]
        assert not missing, f"{task.task_id}: golden row {entry.key!r} lacks the metrics {missing}"
        card = entry.card
        rows.append(Row(key=entry.key, label=entry.label, method_id=card.method_id, family=card.family,
                        scores={m: float(spec[m]) for m in task.metrics}, seconds=0.0,
                        role=card.benchmark_role, type=_entry_type(entry), algorithm=card.name))
    return Board(task, "", rows, default_rows=True)


def _block(text, name):
    start, end = _markers(name)
    assert text.count(start) == 1 and text.count(end) == 1, f"README needs exactly one {start} ... {end} pair"
    head, rest = text.split(start, 1)
    body, tail = rest.split(end, 1)
    return head, body.strip(), tail


def _replace_block(text, name, content):
    head, _, tail = _block(text, name)
    start, end = _markers(name)
    return f"{head}{start}\n{content}\n{end}{tail}"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_golden_rows_are_the_default_entries(task_id):
    task = _task(task_id)
    entries = _entries(task)
    assert entries, f"{task_id} has no default entries"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_default_rows_have_no_custom_type(task_id):
    board = golden_board(_task(task_id))
    assert all(r.type and r.type != "Custom" for r in board.rows), [(r.key, r.type) for r in board.rows]
    assert all(r.algorithm for r in board.rows)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_readme_board_matches_golden_board(task_id):
    text = _readme()
    _, shown, _ = _block(text, f"board:{task_id}")
    expected = golden_board(_task(task_id)).to_markdown()
    assert shown == expected, f"README board {task_id} is stale; {REGENERATE}"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_stage_column_appears_only_on_boards_that_mix_stages(task_id):
    board = golden_board(_task(task_id))
    header = board.to_markdown().splitlines()[0]
    stages_present = {_FAMILY_STATE[r.family] for r in board.ranked() + board.references()}
    mixed = len(stages_present) > 1
    assert ("| Stage |" in header) == mixed, header
    if mixed:
        assert header.startswith("| Rank | Stage | Type | Method | Algorithm |")
        shown = {line.split("|")[2].strip() for line in board.to_markdown().splitlines()[2:]}
        assert shown == stages_present
        single = Board(board.task, "", [r for r in board.rows if r.family == "run"], default_rows=True)
        assert single.to_markdown().splitlines()[0].startswith("| Rank | Type | Method | Algorithm |")


def test_readme_method_counts_match_the_registry():
    text = _readme()
    assert f"The {len(list_methods())} methods fall into" in text
    for family in FAMILIES:
        row = re.search(rf"^\|[^|]*\|[^|]*\| `{family}` \|[^|]*\| \[(\d+)\]\(", text, re.M)
        assert row, f"the README category table has no row for the {family} family"
        assert int(row.group(1)) == len(list_methods(family=family)), family




def test_hero_figure_and_alt_text_counts_match_the_registry():
    text = _readme()
    counts = {family: len(list_methods(family=family)) for family in FAMILIES}
    total, n_tasks = len(list_methods()), len(TASK_IDS)
    swegym_rows = len(_task("detect/swegym").golden()["rows"])
    alt = re.search(r'assets/hero\.png" alt="([^"]+)"', text).group(1)
    for literal in (f"run-level detection ({counts['run']} methods)", f"localization ({counts['step']})",
                    f"{total} methods", f"{n_tasks} leaderboards"):
        assert literal in alt, f"the hero alt text lacks {literal!r}: {alt}"
    chart_alt = re.search(r'assets/leaderboard-swegym\.png" alt="([^"]+)"', text).group(1)
    assert f"ROC-AUC of {swegym_rows} methods" in chart_alt, chart_alt
    if not HERO_BUILDER.is_file():
        pytest.skip("figure-src/ is not beside the tests")
    source = HERO_BUILDER.read_text(encoding="utf-8")
    literals = [f"{family} \u00b7 {n} methods" for family, n in counts.items()]
    literals += [f"{total} methods", f"{n_tasks} leaderboards", f"list the {n_tasks} boards", f"of {swegym_rows} rows"]
    for literal in literals:
        assert literal in source, f"figure-src/hero/build_hero.py lacks {literal!r}; update it and rebuild the hero"


def test_readme_has_a_board_block_for_each_task_and_no_other():
    found = re.findall(r"<!-- board:(\S+?):start -->", _readme())
    assert sorted(found) == sorted(TASK_IDS), f"README board blocks {sorted(found)} != tasks {sorted(TASK_IDS)}"


def regenerate() -> None:
    """Rewrite the board blocks and the method tables of README.md in place."""
    text = original = README.read_text(encoding="utf-8")
    blocks = {f"board:{tid}": golden_board(_task(tid)).to_markdown() for tid in TASK_IDS}
    blocks.update({f"methods:{family}": method_table(family) for family in FAMILIES})
    for name, content in blocks.items():
        updated = _replace_block(text, name, content)
        print(f"{'rewrote' if updated != text else 'kept'} {name}")
        text = updated
    if text != original:
        with open(README, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


if __name__ == "__main__":
    regenerate()
    sys.exit(0)
