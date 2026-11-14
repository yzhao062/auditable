"""Download-free tests for ``auditable.bench`` step localization (protocols, pooling, and task)."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")

from auditable.bench import Entry, get_task, leaderboard, list_tasks  # noqa: E402
from auditable.bench.corpora.whoandwhen import pool_steps  # noqa: E402
from auditable.bench.protocols_step import (  # noqa: E402
    SEEDS,
    _grouped_folds,
    _rank_metrics,
    grouped_folds,
    rank_metrics,
    step_oof_scores,
)
from auditable.bench.tasks.localize import EXTRA_ENTRY_FACTORIES, LocalizationTask  # noqa: E402
from auditable.featurize import TraceView  # noqa: E402
from auditable.methods import get_method  # noqa: E402


def _synthetic_runs(n_runs: int = 10):
    """Synthetic multi-agent runs with full-context dependencies and an early mistake step."""
    runs = []
    mistakes = []
    for r in range(n_runs):
        n_steps = 4 + (r % 3)
        steps = []
        for i in range(n_steps):
            agent = "Computer_terminal" if i % 3 == 2 else ("planner" if i % 2 == 0 else "coder")
            kind = "tool_call" if agent == "Computer_terminal" else "decision"
            steps.append({"idx": i, "agent": agent, "kind": kind, "deps": list(range(i))})
        runs.append(TraceView(steps=steps, texts=[f"step {i}" for i in range(n_steps)], source="synthetic"))
        mistakes.append(0 if r % 2 == 0 else 1)
    return runs, mistakes


# --- task registry and defaults --------------------------------------------------------------

def test_task_registered_and_default_entries():
    ids = [t.task_id for t in list_tasks()]
    assert "localize/whoandwhen" in ids

    task = get_task("localize/whoandwhen")
    assert task.family == "step"
    assert task.metrics == ("top1", "top3", "mrr")

    entries = task.default_entries()
    assert [(e.key, e.label) for e in entries[:5]] == [
        ("step.random", "random"),
        ("step.blast", "auditable (blast)"),
        ("step.position_supervised", "position"),
        ("step.pygod_dominant_ww8", "pygod (graph AD)"),
        ("step.exec_rank", "exec-rank (sup.)"),
    ]
    # The cached LLM-judge rows follow, one per (protocol, model) combination.
    llm = entries[5:]
    assert len(llm) == 31
    assert all(e.key.startswith("step.llm_") and ":" in e.key for e in llm)


def test_extra_entry_factories_hook():
    task = get_task("localize/whoandwhen")
    n_default = len(task.default_entries())
    extra_entry = Entry(get_method("step.degree"), label="degree-extra", key="step.degree")
    EXTRA_ENTRY_FACTORIES.append(lambda: [extra_entry])
    try:
        entries = task.default_entries()
        assert len(entries) == n_default + 1
        assert entries[-1] is extra_entry
    finally:
        EXTRA_ENTRY_FACTORIES.pop()


def test_golden_rows_are_the_five_rule_rows_and_the_llm_rows():
    golden = get_task("localize/whoandwhen").golden()
    assert golden["section"] == "[POST] post_localization :: whoandwhen"
    rule = {"step.random", "step.blast", "step.position_supervised", "step.pygod_dominant_ww8", "step.exec_rank"}
    assert rule <= set(golden["rows"])
    llm = set(golden["rows"]) - rule
    assert len(llm) == 31 and all(k.startswith("step.llm_") for k in llm)
    assert golden["rows"]["step.exec_rank"]["top1"] == pytest.approx(0.211)


# --- pooling, grouped folds, and ranking metrics ---------------------------------------------

def test_pool_steps_formats_and_errors():
    runs_tv, mistakes = _synthetic_runs(3)
    g1, m1 = pool_steps(runs_tv, mistakes)
    g2, m2 = pool_steps([{"steps": r.steps} for r in runs_tv], mistakes)
    g3, m3 = pool_steps([r.steps for r in runs_tv], mistakes)
    assert np.array_equal(g1, g2)
    assert np.array_equal(g1, g3)
    assert m1 == m2 == m3

    with pytest.raises(ValueError):
        pool_steps(runs_tv, mistakes[:-1])
    with pytest.raises(ValueError):
        pool_steps(runs_tv, [99, 0, 0])


def test_rank_metrics_exact_values_and_stable_ties():
    # Three runs of lengths 4, 5, 3; mistake rows at pooled indices 0, 5, 11.
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    mistake_row = {0: 0, 1: 5, 2: 11}
    # Ranks: run 0 -> rank 1, run 1 -> rank 2, run 2 -> rank 3 (on tie at 0.5, earlier idx wins).
    scores = np.array([
        0.9, 0.2, 0.1, 0.0,        # mistake at row 0 -> rank 1
        0.8, 0.7, 0.3, 0.2, 0.1,   # mistake at row 5 -> rank 2
        0.5, 0.5, 0.5,             # all tied -> mistake at row 11 gets rank 3 (stable)
    ])
    top1, top3, mrr = _rank_metrics(scores, groups, mistake_row)
    assert top1 == pytest.approx(1.0 / 3.0)
    assert top3 == pytest.approx(1.0)
    assert mrr == pytest.approx((1.0 + 0.5 + 1.0 / 3.0) / 3.0)
    assert np.array_equal(rank_metrics(scores, groups, mistake_row), [top1, top3, mrr])

    # Move run 2's mistake to step 0 (row 9) and add a 4-step run with rank 4.
    groups4 = np.concatenate([groups, np.array([3, 3, 3, 3])])
    mistake_row4 = {0: 0, 1: 5, 2: 9, 3: 15}
    scores4 = np.concatenate([scores, np.array([0.9, 0.8, 0.7, 0.1])])
    top1_4, top3_4, mrr_4 = _rank_metrics(scores4, groups4, mistake_row4)
    assert top1_4 == pytest.approx(2.0 / 4.0)  # runs 0 and 2 are rank 1
    assert top3_4 == pytest.approx(3.0 / 4.0)  # run 3 is rank 4
    assert mrr_4 == pytest.approx((1.0 + 0.5 + 1.0 + 0.25) / 4.0)


def test_grouped_folds_partition_runs_without_leakage():
    runs, mistakes = _synthetic_runs(12)
    groups, _ = pool_steps(runs, mistakes)
    masks = _grouped_folds(groups, 5, seed=0)
    assert len(masks) == 5
    # Every step row belongs to exactly one test fold.
    covered = sum(m.astype(int) for m in masks)
    assert np.array_equal(covered, np.ones(len(groups), dtype=int))
    # Whole runs stay together: train and test run sets are disjoint.
    for test in masks:
        train_runs = set(groups[~test].tolist())
        test_runs = set(groups[test].tolist())
        assert train_runs.isdisjoint(test_runs)
    # Deterministic per seed, varies across seeds.
    masks_again = grouped_folds(groups, 5, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(masks, masks_again))
    masks_seed1 = grouped_folds(groups, 5, seed=1)
    assert not all(np.array_equal(a, b) for a, b in zip(masks, masks_seed1))


def test_step_oof_scores_and_synthetic_leaderboard():
    runs, mistakes = _synthetic_runs(10)
    groups, _ = pool_steps(runs, mistakes)

    for method_id in ("step.position_supervised", "step.exec_rank"):
        cls = get_method(method_id)
        oof = step_oof_scores(cls, runs, mistakes, groups, seed=0)
        assert oof.shape == (len(groups),)
        assert np.all(np.isfinite(oof))
        assert np.all((oof >= 0.0) & (oof <= 1.0))

    assert list(SEEDS) == [0, 1, 2, 3, 4]

    task = LocalizationTask("test/localize", "Synthetic", lambda: (runs, mistakes, {}))
    board = leaderboard(task, methods=["step.random", "step.blast", "step.position_supervised", "step.exec_rank"])
    assert len(board.ranked()) == 4
    for row in board.ranked():
        for metric in ("top1", "top3", "mrr"):
            assert 0.0 <= row.scores[metric] <= 1.0
    assert "10 failed runs" in board.description
    assert "49 steps" in board.description


def test_synthetic_leaderboard_pygod():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("pygod")

    runs, mistakes = _synthetic_runs(6)
    task = LocalizationTask("test/localize_pygod", "Synthetic PyGOD", lambda: (runs, mistakes, {}))
    board = leaderboard(task, methods=["step.pygod_dominant_ww8"])
    assert len(board.ranked()) == 1
    row = board["step.pygod_dominant_ww8"]
    for metric in ("top1", "top3", "mrr"):
        assert 0.0 <= row.scores[metric] <= 1.0
