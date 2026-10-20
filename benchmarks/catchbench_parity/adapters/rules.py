"""Structural step rules: the random streams, the blast baseline, the Gold rows, and the Gold
attribution statistics through the library.

POST localization keeps GRADE's ``_rank_metrics`` and Gold keeps CatchBench's tie-aware
``_scored`` / ``_ranks``; only the per-step score function comes from the library, and it
receives raw step lists. The Gold attribution rows score a per-run statistic from
``auditable.featurize``, which is not a method, so a small shim lets the consumption check
perturb it like one. ``catchbench.gold._run_maxspan``, ``_run_edges`` and ``_step_graph``
are replaced by their library ports for the whole board, which also routes the
distributional check of ``gold_report`` (it calls ``_step_graph`` directly) through the
library.
"""
from __future__ import annotations

import numpy as np

from catchbench_parity import perturb, views
from catchbench_parity.manifest import check, feature_item, item, sentinel

_METRICS = ("top1", "top3", "mrr")
_SEEDS = range(5)  # injection seeds the Gold checks cover


def _step_rule(method_id: str, **kw):
    from auditable.methods import get_method

    return perturb.wrap(get_method(method_id)(**kw))


# --- POST localization -----------------------------------------------------------------------

class _PostRandom:
    """``post._SeedModel("random")``: per seed ``s`` one ``RandomState(s)`` stream over the
    pooled steps. The library draws it run by run, which is the same stream."""

    def __init__(self, original) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)

    def evaluate(self, task):
        from agent_failure_localization import SEEDS, _rank_metrics

        view = views.localization_view(task)
        per_seed = []
        for s in SEEDS:
            rule = _step_rule("step.random", random_state=np.random.RandomState(s))
            scores = np.concatenate([rule.step_scores(run) for run in view.runs])
            per_seed.append(_rank_metrics(scores, view.groups, view.mistake_row))
        mean = np.array(per_seed).mean(axis=0)
        return dict(zip(_METRICS, (float(v) for v in mean)))


class _PostBlast:
    def __init__(self, original) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)

    def evaluate(self, task):
        from agent_failure_localization import _rank_metrics

        view = views.localization_view(task)
        rule = _step_rule("step.blast")
        scores = np.concatenate([rule.step_scores(run) for run in view.runs])
        metrics = _rank_metrics(scores, view.groups, view.mistake_row)
        return dict(zip(_METRICS, (float(v) for v in metrics)))


item(task_id="post_localization", board_alias="random", variant="", factory="post_localization_methods",
     method_id="step.random", build=lambda o: _PostRandom(o))
item(task_id="post_localization", board_alias="auditable (blast)", variant="",
     factory="post_localization_methods", method_id="step.blast", build=lambda o: _PostBlast(o))


# --- Gold localization -----------------------------------------------------------------------

_GOLD_RULES = (("position", "step.position_early_rule"), ("degree", "step.degree"),
               ("has-dep (control)", "step.has_dep"), ("max-span (control)", "step.max_span"),
               ("auditable (dep-anomaly)", "step.dep_anomaly"))


class _GoldRule:
    """A ``gold._ScoreMethod`` row: ``scores(task)`` is what the breakdowns read, and
    ``evaluate`` ranks the injected step through CatchBench's tie-aware ``_scored``."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def scores(self, task):
        view = views.gold_view(task)
        rule = _step_rule(self.library_method)
        return [np.asarray(rule.step_scores(steps)) for steps in view.steps]

    def evaluate(self, task):
        from catchbench.gold import _scored

        return _scored(views.as_task(views.gold_view(task)), self.scores(task))


class _GoldRandom:
    """``gold.GoldRandom``: ``scores`` is one ``RandomState(0)`` stream over the runs and
    ``evaluate`` averages 50 seeds; the breakdowns skip this row."""

    def __init__(self, original) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)

    def scores(self, task):
        view = views.gold_view(task)
        rule = _step_rule("step.random", random_state=np.random.RandomState(0))
        return [rule.step_scores(steps) for steps in view.steps]

    def evaluate(self, task):
        from catchbench.gold import _kind_summary, _ranks

        view = views.gold_view(task)
        groups = sorted(set(view.groups.tolist()))
        rows = []
        for seed in range(50):
            rule = _step_rule("step.random", random_state=np.random.RandomState(seed))
            draw = [rule.step_scores(steps) for steps in view.steps]
            rows.append(_kind_summary(_ranks(views.as_task(view), draw), groups))
        mean = np.mean(rows, axis=0)
        return dict(zip(_METRICS, (float(v) for v in mean)))


item(task_id="gold_localization", board_alias="random", variant="", factory="gold_localization_methods",
     method_id="step.random", build=lambda o: _GoldRandom(o))
for _alias, _mid in _GOLD_RULES:
    item(task_id="gold_localization", board_alias=_alias, variant="", factory="gold_localization_methods",
         method_id=_mid, build=lambda o, m=_mid: _GoldRule(o, m))


# --- Gold attribution ------------------------------------------------------------------------

_ATTRIBUTION = (("max-span (higher=stale)", "run_max_span"), ("edge-count (higher=stale)", "run_edge_count"))


class _RunStatistic:
    """A per-run statistic from ``auditable.featurize`` behind a scoring entry point, so
    ``perturb.wrap`` can alter it as it alters a method's score."""

    def __init__(self, name: str) -> None:
        self.name = name

    def decision_function(self, steps):
        from auditable import featurize

        return getattr(featurize, self.name)(steps)


class _AttribStat:
    """A ``gold._AttribScore`` row. ``_fn`` stays: ``gold_attribution_robustness`` calls it on
    each run."""

    def __init__(self, original, statistic: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.statistic = statistic
        scorer = perturb.wrap(_RunStatistic(statistic))
        self._fn = lambda steps: float(scorer.decision_function(steps))

    def evaluate(self, task):
        from sklearn.metrics import roc_auc_score

        view = views.attribution_view(task)
        scores = np.array([self._fn(s) for s in view.runs])
        return {"roc_auc": float(roc_auc_score(view.y, scores))}


for _alias, _stat in _ATTRIBUTION:
    item(task_id="gold_attribution", board_alias=_alias, variant="", factory="gold_attribution_methods",
         method_id=f"featurize.{_stat}", build=lambda o, s=_stat: _AttribStat(o, s))

# Every board caller of these cores is one of the rows above. `_seed_metrics` and
# `_SeedModel.evaluate` also serve the position and exec-rank rows (unit run_tabular), so
# they are not registered here.
sentinel("catchbench.post:_auditable_blast", "catchbench.post:AuditableBlast.evaluate",
         "catchbench.gold:_degrees", "catchbench.gold:_zmax",
         "catchbench.gold:_ScoreMethod.scores", "catchbench.gold:_ScoreMethod.evaluate",
         "catchbench.gold:GoldRandom.scores", "catchbench.gold:GoldRandom.evaluate",
         "catchbench.gold:_AttribScore.evaluate")

_SECTIONS = ("Gold distributional check", "[POST] gold_attribution ::", "Gold attribution seed robustness")
feature_item(name="catchbench.gold:_step_graph", library="auditable.featurize:step_graph",
             sections=("Gold distributional check",), adapt=lambda lib: lib)
feature_item(name="catchbench.gold:_run_maxspan", library="auditable.featurize:run_max_span",
             sections=_SECTIONS, adapt=lambda lib: (lambda steps: float(lib(steps))))
feature_item(name="catchbench.gold:_run_edges", library="auditable.featurize:run_edge_count",
             sections=_SECTIONS, adapt=lambda lib: (lambda steps: float(lib(steps))))


# --- per-sample comparisons ------------------------------------------------------------------

@check(("post_localization", "random", ""))
def check_post_random(ctx, it):
    """The pooled draw GRADE makes per seed against the library's run-by-run draws from one
    stream, then the averaged board metrics."""
    from auditable.methods import get_method
    from catchbench.post import post_localization_methods

    task = ctx.task("post_localization")
    view = views.localization_view(task)
    n = bad = 0
    for s in range(5):
        ref = np.random.RandomState(s).rand(len(task.y))  # post._SeedModel -> GRADE _seed_metrics
        rule = get_method("step.random")(random_state=np.random.RandomState(s))
        offset = 0
        for run in view.runs:
            lib = rule.step_scores(run)
            n += 1
            bad += not ctx.same(lib, ref[offset:offset + len(lib)])
            offset += len(lib)
        bad += offset != len(ref)
    orig = next(m for m in post_localization_methods() if m.method_id == "random")
    n += 1
    bad += not ctx.same(_PostRandom(orig).evaluate(task), dict(orig.evaluate(task)))
    return {"compared": n, "mismatches": bad, "unit": "(seed, run) score vectors and the board metrics"}


@check(("post_localization", "auditable (blast)", ""))
def check_blast(ctx, it):
    """Per-step blast on all Who&When runs, then the board metrics."""
    from auditable.methods import get_method
    from catchbench.post import _auditable_blast, post_localization_methods

    task = ctx.task("post_localization")
    view = views.localization_view(task)
    rule = get_method("step.blast")()
    n = bad = 0
    for run, raw in zip(task.runs, view.runs):
        n += 1
        bad += not ctx.same(rule.step_scores(raw), _auditable_blast(run))
    orig = next(m for m in post_localization_methods() if m.method_id == "auditable (blast)")
    n += 1
    bad += not ctx.same(_PostBlast(orig).evaluate(task), dict(orig.evaluate(task)))
    return {"compared": n, "mismatches": bad, "unit": "run blast vectors and the board metrics"}


@check(*[("gold_localization", a, "") for a, _ in _GOLD_RULES])
def check_gold_rule(ctx, it):
    """Per-step scores on every Gold run at injection seeds 0..4, the adapter's scores, and
    the board metrics, against the original row."""
    from auditable.methods import get_method
    from catchbench.gold import gold_localization_methods

    n = bad = 0
    for seed in _SEEDS:
        task = ctx.task("gold_localization", seed=seed)
        view = views.gold_view(task)
        orig = next(m for m in gold_localization_methods() if m.method_id == it.board_alias)
        ref = orig.scores(task)
        rule = get_method(it.method_id)()
        adapter = _GoldRule(orig, it.method_id)
        bad += len(ref) != len(view.steps)
        for steps, r, a in zip(view.steps, ref, adapter.scores(task)):
            n += 1
            bad += not (ctx.same(rule.step_scores(steps), r) and ctx.same(a, r))
        n += 1
        bad += not ctx.same(adapter.evaluate(task), dict(orig.evaluate(task)))
    return {"compared": n, "mismatches": bad, "unit": "Gold run score vectors (5 seeds) and board metrics"}


@check(("gold_localization", "random", ""))
def check_gold_random(ctx, it):
    """The ``RandomState(0)`` stream of ``scores`` and each of the 50 streams ``evaluate``
    averages, run by run, then the board metrics."""
    from auditable.methods import get_method
    from catchbench.gold import gold_localization_methods

    n = bad = 0
    for seed in _SEEDS:
        task = ctx.task("gold_localization", seed=seed)
        view = views.gold_view(task)
        orig = next(m for m in gold_localization_methods() if m.method_id == "random")
        adapter = _GoldRandom(orig)
        bad += len(view.steps) != len(task.graphs)
        for steps, r, a in zip(view.steps, orig.scores(task), adapter.scores(task)):
            n += 1
            bad += not ctx.same(a, r)
        for stream in range(50):
            ref_rng = np.random.RandomState(stream)
            rule = get_method("step.random")(random_state=np.random.RandomState(stream))
            for steps, (X, _) in zip(view.steps, task.graphs):
                n += 1
                bad += not ctx.same(rule.step_scores(steps), ref_rng.rand(len(X)))
        n += 1
        bad += not ctx.same(adapter.evaluate(task), dict(orig.evaluate(task)))
    return {"compared": n, "mismatches": bad, "unit": "Gold run score vectors (5 seeds, 51 streams) and board metrics"}


@check(*[("gold_attribution", a, "") for a, _ in _ATTRIBUTION])
def check_attribution(ctx, it):
    """The per-run statistic on every attribution run at injection seeds 0..4, through the
    adapter's ``_fn``, and the board ROC-AUC."""
    from auditable import featurize
    from catchbench.gold import gold_attribution_methods

    stat = getattr(featurize, next(s for a, s in _ATTRIBUTION if a == it.board_alias))
    n = bad = 0
    for seed in _SEEDS:
        task = ctx.task("gold_attribution", seed=seed)
        view = views.attribution_view(task)
        orig = next(m for m in gold_attribution_methods() if m.method_id == it.board_alias)
        adapter = _AttribStat(orig, next(s for a, s in _ATTRIBUTION if a == it.board_alias))
        for steps in view.runs:
            n += 1
            ref = orig._fn(steps)
            bad += not (ctx.same(stat(steps), ref) and ctx.same(adapter._fn(steps), ref))
        n += 1
        bad += not ctx.same(adapter.evaluate(task), dict(orig.evaluate(task)))
    return {"compared": n, "mismatches": bad, "unit": "attribution run statistics (5 seeds) and board ROC-AUC"}


@check(("gold_localization", "gold_report features", ""))
def check_gold_report_features(ctx, it):
    """What ``gold_report`` reads from every clean and injected Gold run at seeds 0..4: the
    step graph (features and edges), the run's max span and its edge count, which the report
    takes from ``_step_graph`` and the attribution rows from ``_run_maxspan`` / ``_run_edges``."""
    from auditable.featurize import run_edge_count, run_max_span, step_graph
    from catchbench import gold

    n = bad = 0
    for seed in _SEEDS:
        task = ctx.task("gold_localization", seed=seed)
        for steps in list(task.clean_steps) + list(task.steps):
            X, e = gold._step_graph(steps)
            span = float(X[:, 3].max()) if len(X) else 0.0  # gold_report.stats
            lib_X, lib_e = step_graph(steps)
            n += 1
            bad += not (ctx.same(lib_X, X) and ctx.same(lib_e, e) and lib_X.dtype == X.dtype
                        and lib_e.dtype == e.dtype
                        and ctx.same(run_max_span(steps), span) and ctx.same(run_max_span(steps), gold._run_maxspan(steps))
                        and ctx.same(run_edge_count(steps), float(e.shape[1]))
                        and ctx.same(run_edge_count(steps), gold._run_edges(steps)))
    return {"compared": n, "mismatches": bad, "unit": "clean and injected Gold runs (5 seeds)"}
