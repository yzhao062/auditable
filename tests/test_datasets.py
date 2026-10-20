"""auditable.datasets.make_traces, and the README quick start that uses it."""
import pathlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")

from auditable.datasets import make_traces  # noqa: E402

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_shapes_labels_and_planted_mistakes():
    traces, y, mistakes = make_traces(n_runs=40, contamination=0.25, n_steps=(6, 9), random_state=3)
    assert len(traces) == 40 and y.shape == mistakes.shape == (40,)
    assert y.sum() == 10
    for trace, label, m in zip(traces, y, mistakes):
        assert 6 <= len(trace) <= 9
        assert [s["idx"] for s in trace] == list(range(len(trace)))
        assert all(d < s["idx"] for s in trace for d in s["deps"])
        if label:
            assert len(trace) // 2 <= m < len(trace) and trace[m]["deps"] == [0]
        else:
            assert m == -1


def test_seeded_runs_repeat_and_a_shared_stream_continues():
    a, b = make_traces(n_runs=5, random_state=0), make_traces(n_runs=5, random_state=0)
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    rng = np.random.RandomState(0)
    first, second = make_traces(n_runs=5, random_state=rng), make_traces(n_runs=5, random_state=rng)
    assert first[0] == a[0] and second[0] != first[0]


@pytest.mark.parametrize("kw", [{"n_steps": (3, 9)}, {"n_steps": (9, 6)}, {"contamination": 1.5}])
def test_rejects_invalid_arguments(kw):
    with pytest.raises(ValueError):
        make_traces(n_runs=5, **kw)


def test_readme_quick_start_runs(capsys):
    text = README.read_text(encoding="utf-8")
    code = text.split("## Quick Start", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(code, "README quick start", "exec"), {})
    out = capsys.readouterr().out
    assert "run-level ROC-AUC:" in out and "mistake ranked first in" in out
