"""DOMINANT step localizers (step.pygod_dominant_ww8, step.pygod_dominant_sg4): real PyGOD fits."""
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("networkx")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pygod")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from auditable.featurize import full_context_edges, step_graph, who_when_step_matrix  # noqa: E402
from auditable.methods import get_method  # noqa: E402
from auditable.methods._pygod import chunked_node_scores  # noqa: E402


def _trace(n, seed):
    r = np.random.RandomState(seed)
    return [{"idx": i, "agent": f"a{r.randint(0, 3)}", "kind": ["decision", "tool_call"][r.randint(0, 2)],
             "deps": sorted(r.choice(i, size=min(i, 2), replace=False).tolist()) if i else []} for i in range(n)]


TRACES = [_trace(5 + s % 7, s) for s in range(12)]


@pytest.mark.parametrize("mid", ["step.pygod_dominant_ww8", "step.pygod_dominant_sg4"])
def test_fit_step_scores_shapes_and_determinism(mid):
    a = get_method(mid)(epoch=5).fit_step_scores(TRACES)
    b = get_method(mid)(epoch=5).fit_step_scores(TRACES)
    assert [len(x) for x in a] == [len(t) for t in TRACES]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert all(np.isfinite(x).all() for x in a)


def test_ww8_and_sg4_read_their_own_representations():
    ww8 = chunked_node_scores("DOMINANT", [(who_when_step_matrix(t), full_context_edges(len(t))) for t in TRACES],
                              epoch=5)
    sg4 = chunked_node_scores("DOMINANT", [step_graph(t) for t in TRACES], epoch=5)
    lib_ww8 = get_method("step.pygod_dominant_ww8")(epoch=5).fit_step_scores(TRACES)
    lib_sg4 = get_method("step.pygod_dominant_sg4")(epoch=5).fit_step_scores(TRACES)
    assert all(np.array_equal(x, y) for x, y in zip(lib_ww8, ww8))
    assert all(np.array_equal(x, y) for x, y in zip(lib_sg4, sg4))


def test_chunking_keeps_traces_whole_and_ordered():
    scores = chunked_node_scores("DOMINANT", [step_graph(t) for t in TRACES], epoch=3, max_chunk_nodes=20)
    assert [len(x) for x in scores] == [len(t) for t in TRACES]
