"""The golden gate fails when a value it should check was never there to check."""
import pytest

from auditable.bench import Board, Row, Task
from auditable.bench import __main__ as cli

GOLDEN = {"rows": {"run.a": {"roc_auc": 0.8}, "run.b": {"roc_auc": 0.7}}}


class _Task(Task):
    task_id = "detect/coverage-test"
    family = "run"
    metrics = ("roc_auc",)

    def golden(self):
        return GOLDEN


def _row(key, status="ok", scores=None, note="", missing_extras=()):
    return Row(key, key, key, "run", {"roc_auc": 0.8} if scores is None and status == "ok" else (scores or {}),
               0.0, status=status, note=note, missing_extras=tuple(missing_extras))


_EXTRA_SKIP = dict(status="skipped", note="needs auditable[graph-ad]", missing_extras=("graph-ad",))


def _board(rows, default_rows=True):
    return Board(_Task(), "coverage test", rows, default_rows=default_rows)


def test_complete_board_leaves_nothing_unchecked():
    board = _board([_row("run.a"), _row("run.b", scores={"roc_auc": 0.7})])
    assert board.golden_coverage() == []
    assert all(c["ok"] for c in board.compare_golden())


def test_error_row_is_unchecked_even_without_a_golden_value():
    board = _board([_row("run.a"), _row("run.b", status="error", note="RuntimeError: failed"),
                    _row("run.user", status="error", note="ValueError: bad")])
    reasons = {u["key"]: u["reason"] for u in board.golden_coverage()}
    assert reasons == {"run.b": "error: RuntimeError: failed", "run.user": "error: ValueError: bad"}


def test_skipped_golden_row_is_unchecked_unless_allowed():
    board = _board([_row("run.a"), _row("run.b", **_EXTRA_SKIP)])
    assert [u["key"] for u in board.golden_coverage()] == ["run.b"]
    assert board.golden_coverage(allow_skipped=True) == []


def test_allow_skipped_waives_only_a_missing_extra():
    board = _board([_row("run.a"), _row("run.b", status="skipped", note="family 'step', task needs 'run'")])
    assert [u["key"] for u in board.golden_coverage(allow_skipped=True)] == ["run.b"]


def test_leaderboard_marks_only_extra_skips(monkeypatch):
    from auditable.bench import _core

    class _Card:
        method_id, name, family, extras, benchmark_role = "run.x", "X", "run", ("graph-ad",), None

    class _Entry:
        key = label = "run.x"
        method = None  # a stand-in class that is not in the registry
        card = _Card()

    class _Pop(_Task):
        def population(self):
            return None

        def describe(self, population):
            return "test"

    monkeypatch.setattr(_core, "_extra_available", lambda extra: False)
    monkeypatch.setattr(_core, "_as_entry", lambda spec: spec)
    assert _core.leaderboard(_Pop(), [_Entry()]).rows[0].missing_extras == ("graph-ad",)
    _Card.family = "step"
    row = _core.leaderboard(_Pop(), [_Entry()]).rows[0]
    assert row.status == "skipped" and row.missing_extras == ()


def test_absent_golden_row_counts_only_when_the_task_chose_the_rows():
    rows = [_row("run.a")]
    assert [u["reason"] for u in _board(rows, default_rows=True).golden_coverage()] == \
        ["absent: not among the task's default rows"]
    assert _board(rows, default_rows=False).golden_coverage() == []  # methods= named the rows


def test_missing_golden_metric_is_unchecked():
    board = _board([_row("run.a", scores={"other": 0.5}), _row("run.b", scores={"roc_auc": 0.7})])
    assert board.golden_coverage() == [{"key": "run.a", "reason": "metric missing: roc_auc"}]


@pytest.mark.parametrize("rows, allow, expected", [
    ([_row("run.a"), _row("run.b", scores={"roc_auc": 0.7})], False, 0),
    ([_row("run.a"), _row("run.b", status="error", note="RuntimeError: x")], False, 1),
    ([_row("run.a"), _row("run.b", **_EXTRA_SKIP)], False, 1),
    ([_row("run.a"), _row("run.b", **_EXTRA_SKIP)], True, 0),
    ([_row("run.a"), _row("run.b", status="skipped", note="family 'step', task needs 'run'")], True, 1),
    ([_row("run.a")], False, 1),
], ids=["complete", "error-row", "skipped-row", "skipped-allowed", "family-skip-not-waived", "absent-row"])
def test_cli_golden_exit_status(monkeypatch, capsys, rows, allow, expected):
    monkeypatch.setattr(cli, "leaderboard", lambda task_id, methods=None: _board(rows, default_rows=methods is None))
    argv = ["detect/coverage-test", "--golden"] + (["--allow-skipped"] if allow else [])
    assert cli.main(argv) == expected
    if expected:
        assert "golden unchecked" in capsys.readouterr().err


def _platform_board():
    class _PlatformTask(_Task):
        def golden(self):
            return {"rows": {"run.a": {"roc_auc": 0.8, "accept": {"roc_auc": [0.79]}, "note": "macOS arm64"}}}

    return Board(_PlatformTask(), "platform test", [_row("run.a", scores={"roc_auc": 0.79})])


@pytest.mark.parametrize("exact, expected", [(False, 0), (True, 1)], ids=["default", "exact"])
def test_cli_exact_counts_a_platform_value_as_a_mismatch(monkeypatch, capsys, exact, expected):
    board = _platform_board()
    assert [c["platform"] for c in board.compare_golden()] == [True]
    monkeypatch.setattr(cli, "leaderboard", lambda task_id, methods=None: board)
    argv = ["detect/coverage-test", "--golden"] + (["--exact"] if exact else [])
    assert cli.main(argv) == expected


def test_cli_fails_when_no_golden_value_was_compared(monkeypatch, capsys):
    board = _board([_row("run.user")], default_rows=False)  # --methods named only a row without golden
    monkeypatch.setattr(cli, "leaderboard", lambda task_id, methods=None: board)
    assert cli.main(["detect/coverage-test", "--methods", "run.user", "--golden"]) == 1
    assert "no golden value was compared" in capsys.readouterr().err


def test_golden_row_without_a_task_metric_is_unchecked():
    class _NoMetric(_Task):
        def golden(self):
            return {"rows": {"run.a": {"roc_auc": 0.8}, "run.b": {"note": "no metric"}}}

    board = Board(_NoMetric(), "t", [_row("run.a"), _row("run.b")])
    assert board.golden_coverage() == [{"key": "run.b", "reason": "golden row names no metric of this task"}]


def test_localize_reports_a_missing_llm_golden_file(monkeypatch, tmp_path):
    import json

    from auditable.bench import _core, get_task

    task = get_task("localize/whoandwhen")
    for name, missing in (("localize__whoandwhen.json", "localize__whoandwhen__llm.json"),
                          ("localize__whoandwhen__llm.json", "localize__whoandwhen.json")):
        with open(f"{_core._GOLDEN_DIR}/{name}", encoding="utf-8") as fh:
            spec = json.load(fh)
        half = tmp_path / name
        half.write_text(json.dumps(spec), encoding="utf-8")
        monkeypatch.setattr(_core, "_GOLDEN_DIR", str(tmp_path))
        assert task.golden()["missing_files"] == [missing]
        board = Board(task, "t", [], default_rows=False)
        assert {"key": missing, "reason": "golden file missing"} in board.golden_coverage()
        half.unlink()
        monkeypatch.undo()


def test_live_check_marks_extra_skips_for_both_families(monkeypatch):
    from auditable.bench import _core
    from auditable.bench.tasks.live import StreamingTask

    live = StreamingTask("live/coverage-test", "t", lambda: None)
    entries = live.default_entries()
    assert {"run", "online"} <= {e.card.family for e in entries}
    monkeypatch.setattr(live, "population", lambda: None)
    monkeypatch.setattr(live, "describe", lambda population: "t")
    monkeypatch.setattr(_core, "_extra_available", lambda extra: False)
    board = _core.leaderboard(live, entries)
    for row, entry in zip(board.rows, entries):
        assert row.status == "skipped"
        assert row.missing_extras == tuple(entry.card.extras)
