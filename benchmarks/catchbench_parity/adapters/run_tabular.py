"""Run-level tabular detectors: the GRADE-layer logistic regressions (``run.lr_*``).

POST detection reads the corpus's typed graphs; LIVE streaming builds each prefix graph
with the library's ``build_graph`` from the raw steps, as ``LiveStreaming.setup`` does
with GRADE's. Features come from the library detector's ``features``; the folds are
GRADE's (``protocols.run_cv_auc``).
"""
from __future__ import annotations

import numpy as np

from catchbench_parity import perturb, views
from catchbench_parity.manifest import check, item, sentinel
from catchbench_parity.protocols import run_cv_auc

_LAYERS = (("size (flat)", "run.lr_flat", "flat"), ("auditable (size+deps)", "run.lr_flatdep", "flatdep"),
           ("full", "run.lr_full", "full"))


def _make(method_id: str):
    from auditable.methods import get_method

    return lambda: perturb.wrap(get_method(method_id)())


def _prefix_graphs(runs, frac):
    from auditable.graph import build_graph
    from catchbench.live import _prefix_steps  # the LIVE protocol's prefix rule

    return [build_graph(_prefix_steps(s, frac), dependency="explicit", shared_resource=False) for s in runs]


class _DetectionLayer:
    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def evaluate(self, task):
        view = views.detection_view(task)
        make = _make(self.library_method)
        X = make().features(view.graphs)
        return {"roc_auc": float(run_cv_auc(make, X, view.y).mean())}


class _StreamingLayer:
    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def auc_curve(self, task):
        view = views.streaming_view(task)
        make = _make(self.library_method)
        out = {}
        for p in view.prefixes:
            X = make().features(_prefix_graphs(view.runs, p))
            out[p] = float(run_cv_auc(make, X, view.y).mean())
        return out

    def evaluate(self, task):
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


for _alias, _mid, _layer in _LAYERS:
    item(task_id="post_detection", board_alias=_alias, variant="", factory="post_detection_methods",
         method_id=_mid, build=lambda o, m=_mid: _DetectionLayer(o, m))
    item(task_id="live_streaming", board_alias=_alias, variant="", factory="live_streaming_methods",
         method_id=_mid, build=lambda o, m=_mid: _StreamingLayer(o, m))

# _cv's only board callers are the six entries above.
sentinel("catchbench.detection:_cv", "catchbench.live:_cv")


# --- per-sample comparisons ------------------------------------------------------------------

def _compare_cv(ctx, method_id, X_lib, X_ref, y):
    """Three comparisons on one feature matrix: library features against the task's;
    the rewritten fold loop with GRADE's own model against ``_cv``; and per-fold decision
    values of the library model against GRADE's model on the same fold."""
    from agent_failure_detection import _cv
    from auditable.methods import get_method
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def grade_model():
        return make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight="balanced"))

    out = {"features_equal": bool(ctx.same(X_lib, X_ref))}
    ref = _cv(X_ref, y)
    out["protocol_with_grade_model"] = bool(ctx.same(run_cv_auc(grade_model, X_ref, y), ref))
    out["protocol_with_library_model"] = bool(
        ctx.same(run_cv_auc(lambda: get_method(method_id)(), X_lib, y), ref))
    n = bad = 0
    for s in range(5):
        for train, test in StratifiedKFold(5, shuffle=True, random_state=s).split(X_ref, y):
            lib = get_method(method_id)().fit(X_lib[train], y[train]).decision_function(X_lib[test])
            ref_dec = grade_model().fit(X_ref[train], y[train]).decision_function(X_ref[test])
            n += len(test)
            bad += int(np.sum(lib != ref_dec))
    out.update(compared=n, mismatches=bad + sum(not v for k, v in out.items() if isinstance(v, bool)),
               unit="out-of-fold decision values")
    return out


@check(*[("post_detection", a, "") for a, _, _ in _LAYERS])
def check_detection(ctx, it):
    from auditable.methods import get_method

    layer = next(L for a, _, L in _LAYERS if a == it.board_alias)
    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("post_detection", corpus=corpus)
        X_lib = get_method(it.method_id)().features(views.detection_view(task).graphs)
        res[corpus] = _compare_cv(ctx, it.method_id, X_lib, task.layers[layer], task.y)
    return ctx.merge(res)


@check(*[("live_streaming", a, "") for a, _, _ in _LAYERS])
def check_streaming(ctx, it):
    from auditable.methods import get_method

    layer = next(L for a, _, L in _LAYERS if a == it.board_alias)
    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("live_streaming", corpus=corpus)
        for p in task.prefixes:
            X_lib = get_method(it.method_id)().features(_prefix_graphs(task.runs, p))
            res[f"{corpus}@{p}"] = _compare_cv(ctx, it.method_id, X_lib, task.layers_at[p][layer], task.y)
    return ctx.merge(res)


# === random floor, PyOD detectors and supervised step localizers =================================
#
# run.random and run.pyod_* score the same typed graphs the detection board reads (and the
# prefix graphs of LIVE); the supervised step localizers read raw Who&When step lists and
# run inside GRADE's grouped folds (``protocols.step_oof_scores``). Evaluation shells (the
# ROC-AUC call, ``_rank_metrics``, the folds) stay CatchBench and GRADE code.

_LOCALIZATION_METRICS = ("top1", "top3", "mrr")

# board row label -> (library method, factory, the original class whose scoring core it replaces)
_PYOD_DETECTION = (
    ("pyod-flatten (ECOD)", "run.pyod_ecod", "post_detection_methods", "catchbench.detection:PyODFlatten"),
    ("pyod-iforest", "run.pyod_iforest", "pyod_extra_methods", "catchbench.pyod_extra:IForestDetection"),
    ("pyod-knn", "run.pyod_knn", "pyod_extra_methods", "catchbench.pyod_extra:KNNDetection"),
    ("pyod-lof", "run.pyod_lof", "pyod_extra_methods", "catchbench.pyod_extra:LOFDetection"),
    ("pyod-copod", "run.pyod_copod", "pyod_extra_methods", "catchbench.pyod_extra:COPODDetection"),
    ("pyod-hbos", "run.pyod_hbos", "pyod_extra_methods", "catchbench.pyod_extra:HBOSDetection"),
)
_SEED_LOCALIZATION = (("position", "step.position_supervised", 1),
                      ("exec-rank (sup.)", "step.exec_rank", 8))


class _DetectionRandom:
    """``RandomDetection``: one RandomState(0), five sequential draws, mean ROC-AUC."""

    def __init__(self, original) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)

    def evaluate(self, task):
        from auditable.methods import get_method
        from sklearn.metrics import roc_auc_score

        view = views.detection_view(task)
        det = perturb.wrap(get_method("run.random")(random_state=np.random.RandomState(0)))
        aucs = [roc_auc_score(view.y, det.fit_score(view.graphs)) for _ in range(5)]
        return {"roc_auc": float(np.mean(aucs))}


class _DetectionPyOD:
    """``PyODFlatten`` and the ``pyod_extra`` rows: fit over the whole batch, ROC-AUC of the
    training-time scores against the failure labels."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def evaluate(self, task):
        from sklearn.metrics import roc_auc_score

        view = views.detection_view(task)
        scores = _make(self.library_method)().fit_score(view.graphs)
        return {"roc_auc": float(roc_auc_score(view.y, scores))}


class _StreamingRandom:
    """``RandomLive``: a fresh RandomState(0) per prefix, five draws, mean ROC-AUC."""

    def __init__(self, original) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)

    def auc_curve(self, task):
        from auditable.methods import get_method
        from sklearn.metrics import roc_auc_score

        view = views.streaming_view(task)
        out = {}
        for p in view.prefixes:
            det = perturb.wrap(get_method("run.random")(random_state=np.random.RandomState(0)))
            out[p] = float(np.mean([roc_auc_score(view.y, det.fit_score(view.runs)) for _ in range(5)]))
        return out

    def evaluate(self, task):
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


class _StreamingPyOD:
    """``_PrefixECOD``: ECOD fit over the population of prefix graphs at each prefix."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def auc_curve(self, task):
        from sklearn.metrics import roc_auc_score

        view = views.streaming_view(task)
        out = {}
        for p in view.prefixes:
            scores = _make(self.library_method)().fit_score(_prefix_graphs(view.runs, p))
            out[p] = float(roc_auc_score(view.y, scores))
        return out

    def evaluate(self, task):
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


class _SeedLocalization:
    """``post._SeedModel`` for the position and exec-rank rows: GRADE's grouped folds with the
    library model inside, metrics averaged over GRADE's seeds."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def evaluate(self, task):
        from agent_failure_localization import SEEDS, _rank_metrics
        from catchbench_parity.protocols import step_oof_scores

        view = views.localization_view(task)
        make = _make(self.library_method)
        per_seed = np.array([
            _rank_metrics(step_oof_scores(make, view.runs, view.mistakes, view.groups, s),
                          view.groups, view.mistake_row)
            for s in SEEDS])
        return dict(zip(_LOCALIZATION_METRICS, (float(v) for v in per_seed.mean(axis=0))))


item(task_id="post_detection", board_alias="random", variant="", factory="post_detection_methods",
     method_id="run.random", build=lambda o: _DetectionRandom(o))
item(task_id="live_streaming", board_alias="random", variant="", factory="live_streaming_methods",
     method_id="run.random", build=lambda o: _StreamingRandom(o))
for _alias, _mid, _factory, _cls in _PYOD_DETECTION:
    item(task_id="post_detection", board_alias=_alias, variant="", factory=_factory, method_id=_mid,
         build=lambda o, m=_mid: _DetectionPyOD(o, m))
item(task_id="live_streaming", board_alias="pyod (ECOD)", variant="", factory="live_streaming_methods",
     method_id="run.pyod_ecod", build=lambda o: _StreamingPyOD(o, "run.pyod_ecod"))
for _alias, _mid, _ncols in _SEED_LOCALIZATION:
    item(task_id="post_localization", board_alias=_alias, variant="", factory="post_localization_methods",
         method_id=_mid, build=lambda o, m=_mid: _SeedLocalization(o, m))

# Scoring cores that only these rows reach. post._seed_metrics is shared with the "random"
# localization row, so it is not registered here.
sentinel("catchbench.detection:RandomDetection.evaluate",
         "catchbench.live:RandomLive.auc_curve", "catchbench.live:RandomLive.evaluate",
         "catchbench.live:_PrefixECOD.auc_curve", "catchbench.live:_PrefixECOD.evaluate")
for _alias, _mid, _factory, _cls in _PYOD_DETECTION:
    sentinel(f"{_cls}.evaluate")


# --- per-sample comparisons for the rows above -----------------------------------------------

def _part(ctx, lib, ref, unit):
    """One exact comparison: ``compared`` is the reference size, ``mismatches`` the positions
    where the library differs (every position when the shapes differ)."""
    a, b = np.asarray(lib), np.asarray(ref)
    bad = int(b.size) if a.shape != b.shape else int(np.sum(a != b))
    if bad == 0 and not ctx.same(a, b):
        bad = 1
    return {"compared": int(b.size), "mismatches": bad, "unit": unit}


def _merged(ctx, parts, unit):
    return {**ctx.merge(parts), "unit": unit}


def _original(it):
    """The CatchBench object a manifest item replaces, built by its own factory."""
    import importlib

    from catchbench_parity.manifest import FACTORIES

    factory = getattr(importlib.import_module(FACTORIES[it.factory][0]), it.factory)
    return next(o for o in factory() if it.matches(o))


class _capture_auc:
    """Context manager that records every (labels, scores) pair reaching sklearn's
    ``roc_auc_score`` while it runs. The originals and the adapters both import the function
    inside the scoring call, so this reads the scores each one actually evaluates."""

    def __enter__(self):
        import sklearn.metrics as metrics

        self._metrics, self._real, calls = metrics, metrics.roc_auc_score, []

        def spy(y_true, y_score, *args, **kwargs):
            calls.append((np.array(y_true, copy=True), np.array(y_score, copy=True)))
            return self._real(y_true, y_score, *args, **kwargs)

        metrics.roc_auc_score = spy
        return calls

    def __exit__(self, *exc):
        self._metrics.roc_auc_score = self._real
        return False


def _compare_board_path(ctx, it, task, method):
    """Run ``method`` ("evaluate" or "auc_curve") on the original board entry and on the adapter,
    and compare the ROC-AUC inputs (labels and per-run scores, call by call) and the returned
    metrics exactly."""
    original = _original(it)
    adapter = it.build(original)
    with _capture_auc() as ref_calls:
        ref_out = getattr(original, method)(task)
    with _capture_auc() as lib_calls:
        lib_out = getattr(adapter, method)(task)
    n = bad = 0
    for (ly, ls), (ry, rs) in zip(lib_calls, ref_calls):
        for lib, ref in ((ls, rs), (ly, ry)):
            p = _part(ctx, lib, ref, "")
            n += p["compared"]
            bad += p["mismatches"]
    bad += abs(len(lib_calls) - len(ref_calls))
    keys = sorted(ref_out, key=str)
    bad += int(sorted(lib_out, key=str) != keys) + sum(int(not ctx.same(lib_out.get(k), ref_out[k])) for k in keys)
    return {"compared": n + len(keys), "mismatches": bad, "calls": len(ref_calls),
            "unit": "ROC-AUC inputs (scores and labels) and returned metrics"}


@check(("post_detection", "random", ""))
def check_detection_random(ctx, it):
    from auditable.methods import get_method

    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("post_detection", corpus=corpus)
        view = views.detection_view(task)
        res[f"{corpus} board path"] = _compare_board_path(ctx, it, task, "evaluate")
        # the library stream against the original's, drawn straight from a RandomState(0)
        ref_rng = np.random.RandomState(0)
        det = get_method(it.method_id)(random_state=np.random.RandomState(0))
        lib = [det.fit_score(view.graphs) for _ in range(5)]
        ref = [ref_rng.rand(len(task.y)) for _ in range(5)]
        res[f"{corpus} five draws"] = _part(ctx, lib, ref, "random scores")
    return _merged(ctx, res, "random scores, ROC-AUC inputs and returned metrics")


@check(("live_streaming", "random", ""))
def check_streaming_random(ctx, it):
    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("live_streaming", corpus=corpus)
        res[corpus] = _compare_board_path(ctx, it, task, "auc_curve")
        res[f"{corpus} evaluate"] = _compare_board_path(ctx, it, task, "evaluate")
    return _merged(ctx, res, "ROC-AUC inputs (random scores, labels) and returned metrics")


_PYOD_REFERENCE = {  # the PyOD constructors the CatchBench originals call
    "run.pyod_ecod": ("pyod.models.ecod", "ECOD", {}),
    "run.pyod_iforest": ("pyod.models.iforest", "IForest", {"random_state": 0}),
    "run.pyod_knn": ("pyod.models.knn", "KNN", {}),
    "run.pyod_lof": ("pyod.models.lof", "LOF", {}),
    "run.pyod_copod": ("pyod.models.copod", "COPOD", {}),
    "run.pyod_hbos": ("pyod.models.hbos", "HBOS", {}),
}


def _check_holdout(ctx, method_id, graphs, flat):
    """The inductive entry points: fit on a random 70% of the runs and score the rest, against
    PyOD driven by hand (scaler fit on the training rows, the same constructor)."""
    import importlib

    from auditable.methods import get_method
    from sklearn.preprocessing import StandardScaler

    module, name, kwargs = _PYOD_REFERENCE[method_id]
    order = np.random.RandomState(0).permutation(len(graphs))
    cut = int(0.7 * len(graphs))
    train, test = np.sort(order[:cut]), np.sort(order[cut:])
    scaler = StandardScaler()
    ref = getattr(importlib.import_module(module), name)(**kwargs)
    ref.fit(scaler.fit_transform(flat[train]))
    lib = get_method(method_id)().fit([graphs[i] for i in train])
    return {
        "train scores": _part(ctx, lib.decision_scores_, ref.decision_scores_, "training-time scores"),
        "held-out scores": _part(ctx, lib.decision_function([graphs[i] for i in test]),
                                 ref.decision_function(scaler.transform(flat[test])), "held-out scores"),
    }


@check(*[("post_detection", a, "") for a, _, _, _ in _PYOD_DETECTION])
def check_detection_pyod(ctx, it):
    from auditable.featurize import grade_layer_matrix

    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("post_detection", corpus=corpus)
        view = views.detection_view(task)
        res[f"{corpus} flat features"] = _part(ctx, grade_layer_matrix(view.graphs, "flat"),
                                               task.layers["flat"], "flat-layer values")
        res[f"{corpus} board path"] = _compare_board_path(ctx, it, task, "evaluate")
        for name, part in _check_holdout(ctx, it.method_id, view.graphs, task.layers["flat"]).items():
            res[f"{corpus} {name}"] = part
    return _merged(ctx, res, "flat-layer values, per-run scores, ROC-AUC inputs and returned metrics")


@check(("live_streaming", "pyod (ECOD)", ""))
def check_streaming_pyod(ctx, it):
    from auditable.featurize import grade_layer_matrix

    res = {}
    for corpus in ("swegym", "tau"):
        task = ctx.task("live_streaming", corpus=corpus)
        for p in task.prefixes:
            res[f"{corpus}@{p} flat features"] = _part(
                ctx, grade_layer_matrix(_prefix_graphs(task.runs, p), "flat"), task.layers_at[p]["flat"],
                "flat-layer values")
        res[f"{corpus} auc_curve"] = _compare_board_path(ctx, it, task, "auc_curve")
        res[f"{corpus} evaluate"] = _compare_board_path(ctx, it, task, "evaluate")
    return _merged(ctx, res, "flat-layer values, per-run scores, ROC-AUC inputs and returned metrics")


class _GradeLocalizer:
    """GRADE's own model (``_oof_scores``) behind the localizer interface, so the rewritten fold
    loop can be compared with the original before any library model is put inside it. It reads
    the loader's runs (with their typed graphs) through GRADE's ``_step_matrix``."""

    def __init__(self, ncols: int) -> None:
        self.ncols = ncols

    def fit(self, runs, mistakes):
        from agent_failure_localization import _step_matrix
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.vstack([_step_matrix(r) for r in runs])
        y = np.array([1 if s["idx"] == m else 0 for r, m in zip(runs, mistakes) for s in r["steps"]])
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight="balanced"))
        self.clf.fit(X[:, :self.ncols], y)
        return self

    def step_scores_batch(self, runs):
        from agent_failure_localization import _step_matrix

        X = np.vstack([_step_matrix(r) for r in runs])
        proba = self.clf.predict_proba(X[:, :self.ncols])[:, 1]
        out, start = [], 0
        for r in runs:
            out.append(proba[start:start + len(r["steps"])])
            start += len(r["steps"])
        return out


@check(*[("post_localization", a, "") for a, _, _ in _SEED_LOCALIZATION])
def check_seed_localization(ctx, it):
    """Per seed: the out-of-fold step scores of GRADE's ``_oof_scores``, the rewritten fold loop
    with GRADE's model inside it, and the rewritten loop with the library model inside it, all
    on the same pooled steps. Also the step matrices and the seed-averaged metrics."""
    from agent_failure_localization import SEEDS, _oof_scores
    from auditable.featurize import normalize_trace, who_when_step_matrix
    from catchbench_parity.protocols import step_oof_scores

    ncols = next(c for a, _, c in _SEED_LOCALIZATION if a == it.board_alias)
    task = ctx.task("post_localization")
    view = views.localization_view(task)
    res = {"step matrices": _part(ctx, np.vstack([who_when_step_matrix(normalize_trace(r)) for r in view.runs]),
                                  task.X, "step-matrix values")}
    for s in SEEDS:
        ref = _oof_scores(task.X, task.y, task.groups, ncols, s)
        res[f"seed {s} protocol + GRADE model"] = _part(
            ctx, step_oof_scores(lambda: _GradeLocalizer(ncols), task.runs, view.mistakes, task.groups, s),
            ref, "out-of-fold step scores")
        res[f"seed {s} protocol + library model"] = _part(
            ctx, step_oof_scores(_make(it.method_id), view.runs, view.mistakes, view.groups, s),
            ref, "out-of-fold step scores")
    original = _original(it)
    ref_out, lib_out = original.evaluate(task), it.build(original).evaluate(task)
    res["seed-averaged metrics"] = {
        "compared": len(ref_out),
        "mismatches": sum(int(not ctx.same(lib_out.get(k), v)) for k, v in ref_out.items()),
        "unit": "top1, top3, mrr"}
    return _merged(ctx, res, "step-matrix values, out-of-fold step scores and seed-averaged metrics")
