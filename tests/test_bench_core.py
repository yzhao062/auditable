"""Tests for the auditable.bench core: registry, leaderboard ranking, errors, skips, golden
checks, formatting, and the CLI. All tests use a synthetic task over auditable.datasets.make_traces
and require no download.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")
pytest.importorskip("pyod")

import auditable.bench._core as bench_core  # noqa: E402
from auditable.base import BaseRunDetector, MethodCard  # noqa: E402
from auditable.bench import Entry, Task, leaderboard, list_tasks, register_task  # noqa: E402
from auditable.bench.__main__ import main as bench_main  # noqa: E402
from auditable.bench.protocols import run_detection_auc  # noqa: E402
from auditable.datasets import make_traces  # noqa: E402
from auditable.graph import build_graph  # noqa: E402
from auditable.methods import get_method  # noqa: E402


def _make_card(method_id: str, *, supervised: bool = False, fit_scope: str = "transductive",
               extras: tuple = ()) -> MethodCard:
    return MethodCard(
        method_id=method_id,
        name=method_id,
        family="run",
        fit_scope=fit_scope,
        supervised=supervised,
        output_kind="score",
        fidelity="original",
        extras=extras,
    )


class _ConstantScoreDetector(BaseRunDetector):
    """Unregistered detector returning deterministic linear scores so ROC-AUC is exact."""

    card = _make_card("run.synth_const")

    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse

    def fit_score(self, X, y=None):
        scores = np.arange(len(X), dtype=float)
        if self.reverse:
            scores = -scores
        self.decision_scores_ = scores
        return self.decision_scores_


class _FixedAucDetector(BaseRunDetector):
    """Unregistered detector whose fit_score yields a predictable ROC-AUC on sorted labels."""

    card = _make_card("run.synth_fixed")

    def __init__(self, mode: str = "high") -> None:
        self.mode = mode

    def fit_score(self, X, y=None):
        n = len(X)
        if self.mode == "high":
            # Matches the half-split synthetic labels used by _SyntheticTask when ordered,
            # or produces a deterministic fixed array for score-controlled tests.
            scores = np.linspace(0.0, 1.0, n)
        elif self.mode == "tie":
            scores = np.zeros(n, dtype=float)
        else:
            scores = np.linspace(1.0, 0.0, n)
        self.decision_scores_ = scores
        return self.decision_scores_


class _SyntheticTask(Task):
    """Synthetic run-detection task built from auditable.datasets.make_traces."""

    task_id = "detect/synthetic_core_test"
    family = "run"
    metrics = ("roc_auc",)
    title = "Synthetic make_traces run detection"

    def __init__(self, golden_rows=None) -> None:
        super().__init__()
        self._golden_rows = golden_rows or {}

    def load(self):
        traces, y, _ = make_traces(n_runs=30, contamination=0.5, n_steps=(6, 10), random_state=0)
        runs = [build_graph(t, dependency="explicit", shared_resource=False) for t in traces]
        return runs, np.asarray(y)

    def describe(self, population) -> str:
        runs, y = population
        return f"{self.title}, {len(runs)} runs ({int(y.sum())} failed)"

    def default_entries(self):
        return [
            Entry(get_method("run.lr_flat")),
            Entry(get_method("run.lr_flatdep")),
            Entry(get_method("run.random")),
        ]

    def score(self, entry: Entry, population):
        runs, y = population
        return {"roc_auc": run_detection_auc(entry, runs, y)}

    def golden(self):
        return {"source": {"synthetic": True}, "rows": self._golden_rows}


@pytest.fixture
def synthetic_task():
    list_tasks()
    task = _SyntheticTask()
    register_task(task)
    try:
        yield task
    finally:
        bench_core._TASKS.pop(task.task_id, None)


def test_leaderboard_ranks_and_ties_share_rank(synthetic_task):
    high = Entry(_ConstantScoreDetector, {"reverse": False}, label="high_det", key="high_det")
    tie_b = Entry(_FixedAucDetector, {"mode": "tie"}, label="tie_b", key="tie_b")
    tie_a = Entry(_FixedAucDetector, {"mode": "tie"}, label="tie_a", key="tie_a")
    low = Entry(_ConstantScoreDetector, {"reverse": True}, label="low_det", key="low_det")

    # Arrange population labels so high > tie_a == tie_b == 0.5 > low.
    runs, _ = synthetic_task.population()
    y = np.array([0] * 15 + [1] * 15)
    synthetic_task._population = (runs, y)

    board = leaderboard(synthetic_task, methods=[low, tie_b, high, tie_a])

    assert [r.key for r in board.ranked()] == ["high_det", "tie_a", "tie_b", "low_det"]
    assert board.rank_of("high_det") == 1
    assert board.rank_of("tie_a") == 2
    assert board.rank_of("tie_b") == 2
    assert board.rank_of("low_det") == 4
    assert board.rank_of("nonexistent") is None
    assert board["tie_a"].scores["roc_auc"] == pytest.approx(0.5)


def test_extra_places_unregistered_method_on_board(synthetic_task):
    class MyDetector(BaseRunDetector):
        card = _make_card("run.user_custom", supervised=False, fit_scope="transductive", extras=())

        def fit_score(self, X, y=None):
            self.decision_scores_ = np.array([len(g.nodes) for g in X], dtype=float)
            return self.decision_scores_

    with pytest.raises(KeyError):
        get_method("run.user_custom")

    board = leaderboard(synthetic_task.task_id, extra=[MyDetector])

    keys = [r.key for r in board.ranked()]
    assert "run.user_custom" in keys
    assert {"run.lr_flat", "run.lr_flatdep", "run.random"}.issubset(keys)
    assert board["run.user_custom"].status == "ok"


def test_missing_extra_lists_row_as_skipped_with_extra_name(synthetic_task, monkeypatch):
    class NeedsGraphExtra(BaseRunDetector):
        card = _make_card("run.needs_graph", extras=("graph-ad",))

        def fit_score(self, X, y=None):
            raise AssertionError("skipped method should not be scored")

    real_extra_available = bench_core._extra_available
    monkeypatch.setattr(
        bench_core,
        "_extra_available",
        lambda extra: False if extra == "graph-ad" else real_extra_available(extra),
    )

    board = leaderboard(synthetic_task.task_id, methods=["run.lr_flat", NeedsGraphExtra])

    skipped = board["run.needs_graph"]
    assert skipped.status == "skipped"
    assert "graph-ad" in skipped.note
    assert board["run.lr_flat"].status == "ok"


def test_raising_method_is_listed_as_error_and_other_rows_score(synthetic_task):
    class FailingDetector(BaseRunDetector):
        card = _make_card("run.failing_detector")

        def fit_score(self, X, y=None):
            raise RuntimeError("detector crashed")

    board = leaderboard(synthetic_task.task_id, methods=["run.lr_flat", FailingDetector, "run.random"])

    err = board["run.failing_detector"]
    assert err.status == "error"
    assert "RuntimeError: detector crashed" in err.note
    assert board["run.lr_flat"].status == "ok"
    assert board["run.random"].status == "ok"
    assert len(board.ranked()) == 2


def test_compare_golden_mismatch_tol_and_platform(synthetic_task):
    class _ScoreTask(_SyntheticTask):
        task_id = "detect/synthetic_golden_test"

        def score(self, entry: Entry, population):
            return {"roc_auc": float(entry.kwargs["fixed_score"])}

    task = _ScoreTask(
        golden_rows={
            "m_tol": {"roc_auc": 0.750, "tol": 0.005},
            "m_plat": {"roc_auc": 0.584, "accept": {"roc_auc": [0.582]}, "note": "macOS arm64"},
            "m_bad": {"roc_auc": 0.900, "note": "expected higher"},
        }
    )
    e_tol = Entry(_ConstantScoreDetector, {"fixed_score": 0.7534}, label="m_tol", key="m_tol")
    e_plat = Entry(_ConstantScoreDetector, {"fixed_score": 0.5821}, label="m_plat", key="m_plat")
    e_bad = Entry(_ConstantScoreDetector, {"fixed_score": 0.400}, label="m_bad", key="m_bad")

    board = leaderboard(task, methods=[e_tol, e_plat, e_bad])
    checks = {c["key"]: c for c in board.compare_golden()}

    assert checks["m_tol"]["ok"] is True and checks["m_tol"]["platform"] is False
    assert checks["m_plat"]["ok"] is True and checks["m_plat"]["platform"] is True
    assert checks["m_plat"]["note"] == "macOS arm64"
    assert checks["m_bad"]["ok"] is False and checks["m_bad"]["platform"] is False


def test_to_markdown_and_to_dict_roundtrip_ranks(synthetic_task):
    tie_a = Entry(_FixedAucDetector, {"mode": "tie"}, label="tie_a", key="tie_a")
    tie_b = Entry(_FixedAucDetector, {"mode": "tie"}, label="tie_b", key="tie_b")
    high = Entry(_ConstantScoreDetector, {"reverse": False}, label="high", key="high")

    class FailingDetector(BaseRunDetector):
        card = _make_card("run.failing")

        def fit_score(self, X, y=None):
            raise ValueError("bad")

    runs, _ = synthetic_task.population()
    synthetic_task._population = (runs, np.array([0] * 15 + [1] * 15))
    synthetic_task._golden_rows = {"high": {"roc_auc": 1.000}}

    board = leaderboard(synthetic_task, methods=[tie_b, high, tie_a, FailingDetector])
    as_dict = board.to_dict()

    assert as_dict["task"] == synthetic_task.task_id
    assert as_dict["metrics"] == ["roc_auc"]
    dict_ranks = {r["key"]: r["rank"] for r in as_dict["rows"]}
    assert dict_ranks == {"high": 1, "tie_a": 2, "tie_b": 2, "run.failing": None}

    md = board.to_markdown(golden=True)
    data_lines = md.splitlines()[2:]
    md_ranks = []
    for line in data_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        md_ranks.append((int(cells[0]), cells[1].strip("`")))
    assert md_ranks == [(1, "high"), (2, "tie_a"), (2, "tie_b")]


def test_cli_lists_tasks_and_golden_exits_one_on_mismatch(synthetic_task, capsys):
    assert bench_main([]) == 0
    listed = capsys.readouterr().out
    assert "detect/swegym" in listed
    assert synthetic_task.task_id in listed

    # Mismatch triggers exit code 1 and writes details to stderr.
    synthetic_task._golden_rows = {"run.random": {"roc_auc": 0.999}}
    rc = bench_main([synthetic_task.task_id, "--methods", "run.random", "--golden"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "golden mismatch: run.random roc_auc" in err

    # Matching via platform accept exits 0 and reports the platform note on stderr.
    board = leaderboard(synthetic_task.task_id, methods=["run.random"])
    actual = round(board["run.random"].scores["roc_auc"], 3)
    synthetic_task._golden_rows = {
        "run.random": {"roc_auc": 0.111, "accept": {"roc_auc": [actual]}, "note": "synthetic platform"}
    }
    rc_ok = bench_main([synthetic_task.task_id, "--methods", "run.random", "--golden"])
    err_ok = capsys.readouterr().err
    assert rc_ok == 0
    assert "golden platform value: run.random roc_auc" in err_ok
