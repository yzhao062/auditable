"""The consumption check of the CatchBench parity harness, on synthetic boards.

The harness (benchmarks/catchbench_parity) is development-only and not in the wheel. Its
board comparison needs neither CatchBench nor the corpora, so it is tested here.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from catchbench_parity import consumption as cons  # noqa: E402
from catchbench_parity import manifest  # noqa: E402

LIB = ("[POST] post_detection :: metrics\n"
       "  method        AUC\n"
       "  example       0.700\n"
       "  random (matched) 0.500\n"
       "  control       0.500\n")


@pytest.fixture
def one_item(monkeypatch):
    monkeypatch.setattr(manifest, "load_adapters", lambda modules=None: None)
    monkeypatch.setattr(manifest, "ITEMS", [
        manifest.Item("post_detection", "example", "", "post_detection_methods", "run.example", lambda o: o)])
    monkeypatch.setattr(manifest, "DERIVED_ROWS", [])


def test_a_changed_item_row_passes(one_item):
    res = cons.consumption(LIB, LIB.replace("0.700", "0.650"))
    assert res["passed"], res
    assert res["items"]["post_detection | example"]["changed"] == 1


def test_an_unchanged_item_row_fails(one_item):
    res = cons.consumption(LIB, LIB)
    assert not res["passed"]
    assert res["items"]["post_detection | example"]["unchanged_in"] == ["[POST] post_detection :: metrics"]


def test_a_changed_row_outside_the_manifest_fails(one_item):
    res = cons.consumption(LIB, LIB.replace("0.700", "0.650").replace("control       0.500", "control       0.400"))
    assert not res["passed"] and res["unexpected_changes"]


@pytest.mark.parametrize("perturbed", [
    "[POST] post_detection :: metrics\n",                                  # every row removed
    LIB.replace("  example       0.700\n", ""),                            # the item row removed
    LIB.replace("0.700", "0.650").replace("  control       0.500\n", ""),  # a control removed
    LIB.replace("0.700", "0.650") + "  extra         0.100\n",             # a row added
])
def test_missing_or_added_rows_fail(one_item, perturbed):
    res = cons.consumption(LIB, perturbed)
    assert not res["passed"]
    assert "different rows" in res["error"]


def test_labels_match_exactly(one_item, monkeypatch):
    monkeypatch.setattr(manifest, "ITEMS", [
        manifest.Item("post_detection", "random", "", "post_detection_methods", "run.random", lambda o: o)])
    board = LIB.replace("control       0.500", "random        0.500")
    res = cons.consumption(board, board.replace("random        0.500", "random        0.450"))
    assert res["passed"], res  # "random (matched)" is a different row and stays unchanged
    assert cons._row(board.splitlines(), "random").strip() == "random        0.500"
