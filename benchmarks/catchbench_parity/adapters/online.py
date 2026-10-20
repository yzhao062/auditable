"""LIVE online rules: ``online.*`` through CatchBench's own evaluation shells.

The stale-state entries keep ``live._OnlineDetector`` (run peak, clean-run FPR threshold)
and the streaming entry keeps ``live._OnlineScore`` (prefix rule, ROC-AUC); only the
per-step score function comes from the library, and it receives raw step lists. The
shells see a stand-in task built from the view (``views.as_task``), never the task itself.
"""
from __future__ import annotations

import numpy as np

from catchbench_parity import perturb, views
from catchbench_parity.manifest import check, item, sentinel


def _monitor(method_id: str, **kw):
    from auditable.methods import get_method

    return perturb.wrap(get_method(method_id)(**kw))


class _StaleDetector:
    """A ``live_stale_state`` entry scored by a library online monitor."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def _det(self):
        from catchbench.live import _OnlineDetector

        # online.random gets a fresh RandomState(0) per call, as RandomOnline._det does
        kw = {"random_state": np.random.RandomState(0)} if self.library_method == "online.random" else {}
        return _OnlineDetector(self.method_id, _monitor(self.library_method, **kw).step_alarm_scores)

    def detail(self, task):
        return self._det().detail(views.as_task(views.stale_view(task)))

    def evaluate(self, task):
        return self._det().evaluate(views.as_task(views.stale_view(task)))


class _StreamingOnline:
    """A ``live_streaming`` entry whose run score is the monitor's score after the prefix.
    ``_fn`` mirrors ``live._OnlineScore._fn``: the run score of one prefix."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method
        mon = _monitor(library_method)
        self._fn = lambda prefix: mon.prefix_score(prefix, len(prefix))

    def auc_curve(self, task):
        from catchbench.live import _OnlineScore

        mon = _monitor(self.library_method)
        shell = _OnlineScore(self.method_id, lambda prefix: mon.prefix_score(prefix, len(prefix)))
        return shell.auc_curve(views.as_task(views.streaming_view(task)))

    def evaluate(self, task):
        return {"prefix_auc": float(np.mean(list(self.auc_curve(task).values())))}


_STALE = (("random", "online.random"), ("dep-count (control)", "online.dep_count_z"),
          ("raw-span", "online.raw_span"), ("auditable (span z-score)", "online.span_z"))

for _alias, _mid in _STALE:
    item(task_id="live_stale_state", board_alias=_alias, variant="", factory="live_stale_methods",
         method_id=_mid, build=lambda o, m=_mid: _StaleDetector(o, m))

item(task_id="live_streaming", board_alias="dep-span (online)", variant="", factory="live_streaming_methods",
     method_id="online.dep_span", build=lambda o: _StreamingOnline(o, "online.dep_span"))

sentinel("catchbench.live:_span_series", "catchbench.live:_count_series", "catchbench.live:_online_z",
         "catchbench.live:_mean_span", "catchbench.live:RandomOnline._det")


# --- per-sample comparisons ------------------------------------------------------------------

def _original(factory: str, alias: str):
    from catchbench import live

    return next(m for m in getattr(live, factory)() if m.method_id == alias)


@check(*[("live_stale_state", a, "") for a, _ in _STALE])
def check_stale(ctx, it):
    """At all five injection seeds: the library's per-step alarm scores against the original
    score function on every clean and injected run, in the order the original draws them
    (clean runs first, which matters for the shared random stream), and the adapter's
    ``detail`` (TPR and realized FPR at each target FPR) against the original entry's."""
    from auditable.methods import get_method
    from catchbench import live

    original = _original("live_stale_methods", it.board_alias)
    n = bad = 0
    for seed in range(5):
        task = ctx.task("live_stale_state", seed=seed)
        runs = [s for s in list(task.clean) + list(task.injected) if len(s)]
        if it.board_alias == "random":
            ref_fn = live.RandomOnline._det()._score_fn  # the original stream, RandomState(0)
            mon = get_method("online.random")(random_state=np.random.RandomState(0))
        else:
            ref_fn = original._score_fn
            mon = get_method(it.method_id)()
        for s in runs:
            n += 1
            bad += not ctx.same(mon.step_alarm_scores(s), ref_fn(s))
        n += 1
        bad += not ctx.same(_StaleDetector(original, it.method_id).detail(task), original.detail(task))
    return {"compared": n, "mismatches": bad, "unit": "run alarm-score vectors and per-seed detail tables"}


@check(("live_streaming", "dep-span (online)", ""))
def check_dep_span(ctx, it):
    """Run score after each prefix against ``live._mean_span``, both corpora and every prefix
    fraction, then the adapter's ROC-AUC curve against the original entry's."""
    from auditable.methods import get_method
    from catchbench import live

    original = _original("live_streaming_methods", it.board_alias)
    mon = get_method("online.dep_span")()
    n = bad = 0
    for corpus in ("swegym", "tau"):
        task = ctx.task("live_streaming", corpus=corpus)
        for s in task.runs:
            for p in task.prefixes:
                prefix = live._prefix_steps(s, p)
                n += 1
                bad += not ctx.same(mon.prefix_score(prefix, len(prefix)), live._mean_span(prefix))
        n += 1
        bad += not ctx.same(_StreamingOnline(original, it.method_id).auc_curve(task), original.auc_curve(task))
    return {"compared": n, "mismatches": bad, "unit": "(run, prefix) scores and per-corpus AUC curves"}
