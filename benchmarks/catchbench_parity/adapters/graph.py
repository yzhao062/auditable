"""Graph anomaly detectors (PyGOD, GUARDIAN, G-Safeguard) through the library.

POST localization reads raw Who&When step lists: the library rebuilds each run's typed
graph and the eight-column step matrix itself. POST detection reads the corpus's typed
graphs, which the library encodes with ``to_pyg``; the five unsupervised detectors score
all runs in one fit, and G-Safeguard runs inside GRADE-style cross-validation
(``protocols.gsafeguard_cv_aucs``). Gold localization reads the injected step lists and the
library builds the step graphs.
"""
from __future__ import annotations

import contextlib
import importlib

import numpy as np

from catchbench_parity import perturb, views
from catchbench_parity.manifest import check, item, sentinel

_METRICS = ("top1", "top3", "mrr")


class _PyGODLocalization:
    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def evaluate(self, task):
        from agent_failure_localization import _rank_metrics
        from auditable.methods import get_method

        view = views.localization_view(task)
        per_run = perturb.wrap(get_method(self.library_method)()).fit_step_scores(view.runs)
        metrics = _rank_metrics(np.concatenate(per_run), view.groups, view.mistake_row)
        return dict(zip(_METRICS, (float(v) for v in metrics)))


item(task_id="post_localization", board_alias="pygod (graph AD)", variant="",
     factory="post_localization_methods", method_id="step.pygod_dominant_ww8",
     build=lambda o: _PyGODLocalization(o, "step.pygod_dominant_ww8"))

# post.py binds GRADE's _step_matrix only for PyGODLocalization; GRADE's own _pool keeps its
# binding for task construction.
sentinel("catchbench.post:_step_matrix", "catchbench.post:PyGODLocalization.evaluate")


@check(("post_localization", "pygod (graph AD)", ""))
def check_pygod_ww8(ctx, it):
    """Step matrices and edges per run, then per-step DOMINANT scores of one batch fit."""
    from agent_failure_localization import _step_matrix
    from auditable.featurize import full_context_edges as lib_edges
    from auditable.featurize import normalize_trace, who_when_step_matrix
    from auditable.methods import get_method
    from catchbench.graph_ad import full_context_edges, pygod_node_scores

    task = ctx.task("post_localization")
    view = views.localization_view(task)
    feat_bad = 0
    for run, raw in zip(task.runs, view.runs):
        feat_bad += not ctx.same(who_when_step_matrix(normalize_trace(raw)), _step_matrix(run))
        feat_bad += not ctx.same(lib_edges(len(raw["steps"])), full_context_edges(len(run["steps"])))
    ref = pygod_node_scores([(_step_matrix(r), full_context_edges(len(r["steps"]))) for r in task.runs])
    lib = get_method(it.method_id)().fit_step_scores(view.runs)
    n = sum(len(r) for r in ref)
    bad = sum(int(np.sum(np.asarray(a) != np.asarray(b))) if len(a) == len(b) else len(b)
              for a, b in zip(lib, ref))
    return {"compared": n, "mismatches": bad + feat_bad, "feature_mismatches": feat_bad,
            "unit": "per-step DOMINANT scores (126 runs, one fit)"}


# --- POST detection --------------------------------------------------------------------------

# board row -> (library method, the factory that produces the original row)
_UNSUPERVISED = {
    "pygod (graph AD)": ("run.pygod_dominant", "post_detection_methods"),
    "guardian (recon-AE)": ("run.guardian_lite", "post_detection_methods"),
    "pygod-conad": ("run.pygod_conad", "pygod_extra_methods"),
    "pygod-anomalydae": ("run.pygod_anomalydae", "pygod_extra_methods"),
    "pygod-gaan": ("run.pygod_gaan", "pygod_extra_methods"),
}
_GSAFEGUARD = "g-safeguard (sup GNN)"


class _GraphADDetection:
    """A ``post_detection`` entry scored by an unsupervised library detector: one fit over
    the corpus's typed graphs, then ROC-AUC of the run scores against the failure labels."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def evaluate(self, task):
        from auditable.methods import get_method
        from sklearn.metrics import roc_auc_score

        view = views.detection_view(task)
        scores = perturb.wrap(get_method(self.library_method)()).fit_score(view.graphs)
        return {"roc_auc": float(roc_auc_score(view.y, scores))}


class _GSafeguardDetection:
    """``g-safeguard (sup GNN)``: the library network inside CatchBench's cross-validation
    (3 seeds x 5 stratified folds), mean held-out ROC-AUC, 0.5 when no fold is usable."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def evaluate(self, task):
        from auditable.methods import get_method
        from catchbench_parity.protocols import gsafeguard_cv_aucs

        view = views.detection_view(task)
        cls = get_method(self.library_method)
        graphs = cls().features(view.graphs)
        aucs = gsafeguard_cv_aucs(lambda fold_seed: perturb.wrap(cls(seed=fold_seed)), graphs, view.y)
        return {"roc_auc": float(np.mean(aucs)) if aucs else 0.5}


for _alias, (_mid, _factory) in _UNSUPERVISED.items():
    item(task_id="post_detection", board_alias=_alias, variant="", factory=_factory, method_id=_mid,
         build=lambda o, m=_mid: _GraphADDetection(o, m))
item(task_id="post_detection", board_alias=_GSAFEGUARD, variant="", factory="post_detection_methods",
     method_id="run.gsafeguard_lite", build=lambda o: _GSafeguardDetection(o, "run.gsafeguard_lite"))


# --- Gold localization -----------------------------------------------------------------------

class _GoldPyGOD:
    """``gold.GoldPyGOD``: ``scores`` is what the breakdowns read and ``evaluate`` ranks the
    injected step through CatchBench's tie-aware ``_scored``. The library builds each step
    graph from the injected step list and fits one DOMINANT over all of them."""

    def __init__(self, original) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)

    def scores(self, task):
        from auditable.methods import get_method

        view = views.gold_view(task)
        return perturb.wrap(get_method("step.pygod_dominant_sg4")()).fit_step_scores(view.steps)

    def evaluate(self, task):
        from catchbench.gold import _scored

        return _scored(views.as_task(views.gold_view(task)), self.scores(task))


item(task_id="gold_localization", board_alias="pygod (graph AD)", variant="", factory="gold_localization_methods",
     method_id="step.pygod_dominant_sg4", build=lambda o: _GoldPyGOD(o))

# Every board caller of these cores is a migrated item: PyGODLocalization (above),
# PyGODDetection, GuardianDetection, GSafeguardDetection, _PyGODExtraDetection and GoldPyGOD.
# graph_ad.pygod_node_scores and nx_to_graph are also read late by the detection classes, and
# gold.py binds its own copy of pygod_node_scores at import.
sentinel("catchbench.graph_ad:pygod_node_scores", "catchbench.graph_ad:nx_to_graph",
         "catchbench.gold:pygod_node_scores", "catchbench.pygod_extra:_pygod_extra_node_scores",
         "catchbench.agent_detectors:guardian_run_scores", "catchbench.agent_detectors:gsafeguard_cv_auc",
         "catchbench.detection:PyGODDetection.evaluate", "catchbench.detection:GuardianDetection.evaluate",
         "catchbench.detection:GSafeguardDetection.evaluate", "catchbench.pygod_extra:_PyGODExtraDetection.evaluate",
         "catchbench.gold:GoldPyGOD.scores", "catchbench.gold:GoldPyGOD.evaluate")


# --- per-sample comparisons ------------------------------------------------------------------

@contextlib.contextmanager
def _recording(module: str, attr: str):
    """Wrap ``module.attr`` for the duration of the block and yield the list of its return
    values, one per call. The original keeps running; only its outputs are read."""
    mod = importlib.import_module(module)
    original = getattr(mod, attr)
    calls = []

    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append((args, out))
        return out

    setattr(mod, attr, wrapper)
    try:
        yield calls
    finally:
        setattr(mod, attr, original)


def _same_array(ctx, a, b) -> bool:
    return ctx.same(a, b) and np.asarray(a).dtype == np.asarray(b).dtype


def _array_lists(ctx, lib, ref, unit: str) -> dict:
    """Exact comparison of two lists of per-run arrays, counting individual values."""
    n = sum(len(r) for r in ref)
    if len(lib) != len(ref):
        return {"compared": n, "mismatches": n, "unit": unit}
    bad = sum(int(np.sum(np.asarray(a) != np.asarray(b))) if len(a) == len(b) else len(b)
              for a, b in zip(lib, ref))
    return {"compared": n, "mismatches": bad, "unit": unit}


def _encodings(ctx, typed_graphs) -> dict:
    """``to_pyg`` and the detectors' own encoder (``features``) against ``nx_to_graph``."""
    from auditable.featurize import to_pyg
    from auditable.methods import get_method
    from catchbench.graph_ad import nx_to_graph

    pairs = get_method("run.gsafeguard_lite")().features(typed_graphs)
    bad = 0
    for g, lib in zip(typed_graphs, pairs):
        ref = nx_to_graph(g)
        direct = to_pyg(g)
        for got in (direct, lib):
            bad += not _same_array(ctx, got[0], ref[0])
            bad += not _same_array(ctx, got[1], ref[1])
    return {"compared": 4 * len(typed_graphs), "mismatches": bad, "unit": "graph encodings (features, edges)"}


def _original_row(alias: str):
    from catchbench.detection import GSafeguardDetection, GuardianDetection, PyGODDetection
    from catchbench.pygod_extra import pygod_extra_methods

    rows = {"pygod (graph AD)": PyGODDetection(), "guardian (recon-AE)": GuardianDetection(),
            _GSAFEGUARD: GSafeguardDetection()}
    rows.update({m.method_id: m for m in pygod_extra_methods()})
    return rows[alias]


_RECORDED_CORE = {"pygod (graph AD)": ("catchbench.graph_ad", "pygod_node_scores"),
                  "guardian (recon-AE)": ("catchbench.agent_detectors", "guardian_run_scores"),
                  "pygod-conad": ("catchbench.pygod_extra", "_pygod_extra_node_scores"),
                  "pygod-anomalydae": ("catchbench.pygod_extra", "_pygod_extra_node_scores"),
                  "pygod-gaan": ("catchbench.pygod_extra", "_pygod_extra_node_scores")}


@check(*[("post_detection", a, "") for a in _UNSUPERVISED])
def check_unsupervised(ctx, it):
    """On both corpora: encodings; per-node scores (PyGOD detectors) and per-run scores of
    one fit against the original core's return value, recorded while the original ``evaluate``
    runs; the ROC-AUC of the original ``evaluate`` against the library's, both through the
    library detector directly and through the board adapter."""
    from auditable.methods import get_method
    from sklearn.metrics import roc_auc_score

    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("post_detection", corpus=corpus)
        view = views.detection_view(task)
        res[f"{corpus}:encoding"] = _encodings(ctx, view.graphs)

        original = _original_row(it.board_alias)
        with _recording(*_RECORDED_CORE[it.board_alias]) as calls:
            ref_auc = original.evaluate(task)["roc_auc"]
        if len(calls) != 1:
            raise RuntimeError(f"expected one call of the original core, saw {len(calls)}")
        ref_out = calls[0][1]

        det = get_method(it.method_id)()
        lib_runs = det.fit_score(view.graphs)
        if it.board_alias == "guardian (recon-AE)":
            ref_runs = np.asarray(ref_out)
        else:
            res[f"{corpus}:nodes"] = _array_lists(ctx, det.node_scores_, ref_out, "per-node anomaly scores")
            ref_runs = np.array([scores.mean() for scores in ref_out])
        res[f"{corpus}:runs"] = _array_lists(ctx, [lib_runs], [ref_runs], "per-run failure scores")
        res[f"{corpus}:auc"] = {
            "compared": 3,
            "mismatches": int(float(roc_auc_score(view.y, lib_runs)) != ref_auc)
            + int(_GraphADDetection(original, it.method_id).evaluate(task)["roc_auc"] != ref_auc)
            + int(float(roc_auc_score(task.y, ref_runs)) != ref_auc),
            "unit": "ROC-AUC (library, adapter, recomputed original)"}
    return ctx.merge(res)


class _OriginalGSafeguard:
    """The network and training loop inside ``agent_detectors.gsafeguard_cv_auc``, transcribed
    from the function body into a fit / decision_function estimator (it has no other entry
    point) with CatchBench's own ``_to_data``. It stands in for the original model when the
    rewritten fold loop is checked before any library model is put inside it."""

    def __init__(self, seed: int, hid: int = 32, epochs: int = 60) -> None:
        self.seed, self.hid, self.epochs = seed, hid, epochs

    def fit(self, graphs, y):
        import torch
        from catchbench.agent_detectors import _to_data
        from sklearn.preprocessing import StandardScaler
        from torch_geometric.data import Batch
        from torch_geometric.nn import GCNConv, global_mean_pool

        hid, in_dim = self.hid, np.asarray(graphs[0][0]).shape[1]

        class _GNN(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = GCNConv(in_dim, hid)
                self.c2 = GCNConv(hid, hid)
                self.lin = torch.nn.Linear(hid, 1)

            def forward(self, b):
                h = torch.relu(self.c1(b.x, b.edge_index))
                h = torch.relu(self.c2(h, b.edge_index))
                return self.lin(global_mean_pool(h, b.batch)).squeeze(-1)

        self._to_data, self._Batch = _to_data, Batch
        self.scaler = StandardScaler().fit(np.vstack([np.asarray(g[0], dtype=float) for g in graphs]))
        train_batch = Batch.from_data_list([_to_data(g, self.scaler) for g in graphs])
        train_y = torch.tensor(np.asarray(y), dtype=torch.float)
        torch.manual_seed(self.seed)
        self.model = _GNN()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(train_batch), train_y)
            loss.backward()
            opt.step()
        self.model.eval()
        return self

    def decision_function(self, graphs):
        import torch

        test_batch = self._Batch.from_data_list([self._to_data(g, self.scaler) for g in graphs])
        with torch.no_grad():
            return torch.sigmoid(self.model(test_batch)).numpy()


class _Tap:
    """Pass-through estimator that keeps every ``decision_function`` output."""

    def __init__(self, inner, sink: list) -> None:
        self._inner, self._sink = inner, sink

    def fit(self, *args, **kwargs):
        self._inner.fit(*args, **kwargs)
        return self

    def decision_function(self, X):
        out = self._inner.decision_function(X)
        self._sink.append(np.asarray(out))
        return out


def _compare_folds(ctx, tag: str, aucs, taps, rec_aucs, rec_probs) -> dict:
    """Per-fold held-out predictions and AUCs of a protocol run against the original's."""
    if len(aucs) != len(rec_aucs) or len(taps) != len(rec_probs):
        n = max(len(rec_aucs), 1)
        return {"compared": n, "mismatches": n, "unit": f"{tag}: fold count differs"}
    n = sum(len(p) for p in rec_probs) + len(rec_aucs) + 1
    bad = sum(int(np.sum(np.asarray(a, dtype=float) != np.asarray(b, dtype=float))) if len(a) == len(b) else len(b)
              for a, b in zip(taps, rec_probs))
    bad += sum(int(a != b) for a, b in zip(aucs, rec_aucs))
    mean = float(np.mean(aucs)) if aucs else 0.5
    bad += int(mean != (float(np.mean(rec_aucs)) if rec_aucs else 0.5))
    return {"compared": n, "mismatches": bad, "unit": f"{tag}: held-out probabilities, fold AUCs, mean AUC"}


@check(("post_detection", _GSAFEGUARD, ""))
def check_gsafeguard(ctx, it):
    """On both corpora: encodings, then the original ``gsafeguard_cv_auc`` with its fold
    predictions and AUCs recorded (through ``roc_auc_score``). The rewritten fold loop with
    the transcribed original network must reproduce them (protocol check), and so must the
    loop with the library network (3 seeds x 5 folds, probabilities bit for bit); the board
    adapter's final value is compared with the original's as well."""
    import sklearn.metrics
    from auditable.methods import get_method
    from catchbench.agent_detectors import gsafeguard_cv_auc
    from catchbench.graph_ad import nx_to_graph
    from catchbench_parity.protocols import gsafeguard_cv_aucs

    cls = get_method(it.method_id)
    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("post_detection", corpus=corpus)
        view = views.detection_view(task)
        res[f"{corpus}:encoding"] = _encodings(ctx, view.graphs)

        graphs = [nx_to_graph(g) for g in task.graphs]
        real_auc = sklearn.metrics.roc_auc_score
        rec = []

        def recording_auc(y_true, y_score, *args, **kwargs):
            out = real_auc(y_true, y_score, *args, **kwargs)
            rec.append((np.asarray(y_score).copy(), out))
            return out

        sklearn.metrics.roc_auc_score = recording_auc
        try:
            ref = gsafeguard_cv_auc(graphs, task.y)
        finally:
            sklearn.metrics.roc_auc_score = real_auc
        rec_probs, rec_aucs = [p for p, _ in rec], [a for _, a in rec]
        if float(np.mean(rec_aucs)) != ref:
            raise RuntimeError("recorded fold AUCs do not average to the original's return value")

        taps = []
        aucs = gsafeguard_cv_aucs(lambda s: _Tap(_OriginalGSafeguard(s), taps), graphs, task.y)
        res[f"{corpus}:protocol+original model"] = _compare_folds(ctx, "protocol+original model", aucs, taps,
                                                                  rec_aucs, rec_probs)
        taps = []
        pairs = cls().features(view.graphs)
        aucs = gsafeguard_cv_aucs(lambda s: _Tap(cls(seed=s), taps), pairs, view.y)
        res[f"{corpus}:protocol+library model"] = _compare_folds(ctx, "protocol+library model", aucs, taps,
                                                                 rec_aucs, rec_probs)
        board = _GSafeguardDetection(_original_row(_GSAFEGUARD), it.method_id).evaluate(task)["roc_auc"]
        res[f"{corpus}:adapter"] = {"compared": 1, "mismatches": int(board != ref), "unit": "board value"}
    return ctx.merge(res)


@check(("gold_localization", "pygod (graph AD)", ""))
def check_gold_pygod(ctx, it):
    """For each of the five injection seeds: the library step graphs against ``_step_graph``
    on every injected run, per-step DOMINANT scores of one fit against ``pygod_node_scores``
    over the original graphs, and the tie-aware metrics of the board adapter against
    ``GoldPyGOD.evaluate``."""
    from auditable.featurize import step_graph
    from auditable.methods import get_method
    from catchbench.gold import GoldPyGOD, _step_graph
    from catchbench.graph_ad import pygod_node_scores

    res = {}
    for seed in range(5):
        task = ctx.task("gold_localization", seed=seed)
        view = views.gold_view(task)
        feat_bad = 0
        for steps in view.steps:
            (xa, ea), (xb, eb) = step_graph(steps), _step_graph(steps)
            feat_bad += not _same_array(ctx, xa, xb)
            feat_bad += not _same_array(ctx, ea, eb)
        res[f"seed{seed}:graphs"] = {"compared": 2 * len(view.steps), "mismatches": feat_bad,
                                     "unit": "step graphs (features, edges)"}
        ref = pygod_node_scores(task.graphs)
        lib = get_method(it.method_id)().fit_step_scores(view.steps)
        res[f"seed{seed}:scores"] = _array_lists(ctx, lib, ref, "per-step DOMINANT scores")
        ref_metrics = GoldPyGOD().evaluate(task)
        lib_metrics = _GoldPyGOD(GoldPyGOD()).evaluate(task)
        res[f"seed{seed}:metrics"] = {"compared": 3, "mismatches": sum(
            int(ref_metrics[k] != lib_metrics[k]) for k in ("top1", "top3", "mrr")), "unit": "top1, top3, mrr"}
    return ctx.merge(res)
