"""LLM-judge step localization (Who&When protocols) through the library.

Every protocol runs through the library's own driver, prompts and parser. All-at-once
replays each recorded reply: the library builds the prompt, a fixed-response backend
returns the reply CatchBench recorded for that run, and the library parses it and scores
the steps. The single-step protocols record a log with one line per query, holding the
prefix examined (step-by-step) or the half chosen (binary search) and the reply cut to 120
characters, so the full replies are not available. Their replay feeds the library driver,
query by query, a reply that states the decision the log recorded; the driver must then
make the same number of queries, write the same log headers, and name the recorded step.
``check_protocols`` also runs both drivers against a deterministic scripted backend and
compares every prompt.
"""
from __future__ import annotations

import re

import numpy as np

from catchbench_parity import perturb, views
from catchbench_parity.manifest import check, item, sentinel

_METRICS = ("top1", "top3", "mrr")
_LIBRARY = {"all_at_once": "step.llm_all_at_once", "step_by_step": "step.llm_step_by_step",
            "binary_search": "step.llm_binary_search"}


_SBS_LINE = re.compile(r"(?m)^\[0\.\.(\d+)\] ")
_BS_LINE = re.compile(r"(?m)^\[(\d+)-(\d+) mid(\d+)\] (FIRST|SECOND) :: ")


def _headers(method: str, log: str):
    """The per-query header of a protocol log: the prefix for step-by-step, the interval,
    midpoint and decision for binary search."""
    if method == "step_by_step":
        return [int(m.group(1)) for m in _SBS_LINE.finditer(log)]
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)) for m in _BS_LINE.finditer(log)]


def _decision_replies(method: str, rec: dict):
    """Replies that state, query by query, the decisions a recorded single-step log holds.

    Step-by-step stopped at its last logged prefix (or ran to the end and defaulted to the
    last step), so every earlier reply named no step and the last one names the recorded
    step. Binary search logged FIRST or SECOND per query. A log whose headers are not one
    consecutive query sequence raises: it cannot be replayed faithfully."""
    headers = _headers(method, rec["raw"])
    if method == "step_by_step":
        if not headers or headers != list(range(len(headers))) or rec["top"] > headers[-1]:
            raise ValueError("step-by-step log is not a consecutive prefix sequence ending at the named step")
        return ["VERDICT: NO"] * (len(headers) - 1) + [f"VERDICT: {rec['top']}"], headers
    return [f"VERDICT: {h[3]}" for h in headers], headers


def _recorded(method: str, model: str):
    """CatchBench's recorded predictions aligned with ``PostLocalization.runs`` (data access)."""
    from catchbench.llm_judge import load_cache, load_judge_runs, run_key

    preds = load_cache(method, model)["predictions"]
    return [preds[run_key(r)] for r in load_judge_runs()]


class _LLMJudge:
    def __init__(self, original) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.method = original.method
        self.model = original.model

    def evaluate(self, task):
        from agent_failure_localization import _rank_metrics
        from auditable.llm import FixedResponseBackend
        from auditable.methods import get_method

        view = views.localization_view(task, texts=True)
        recorded = _recorded(self.method, self.model)
        cls = get_method(_LIBRARY[self.method])
        if self.method == "all_at_once":
            chunks = []
            for run, rec in zip(view.runs, recorded):
                judge = perturb.wrap(cls(backend=FixedResponseBackend([rec["raw"]]), model=self.model))
                chunks.append(judge.step_scores(run))
            metrics = _rank_metrics(np.concatenate(chunks), view.groups, view.mistake_row)
            return dict(zip(_METRICS, (float(v) for v in metrics)))
        # One judge (and so one in-memory cache) per run: the library keys predictions by input
        # content, and Who&When holds two pairs of runs with identical steps and text whose
        # recorded answers (and, for one pair, gold steps) differ.
        hits = []
        for run, rec, mistake in zip(view.runs, recorded, view.mistakes):
            replies, _ = _decision_replies(self.method, rec)
            judge = perturb.wrap(cls(backend=FixedResponseBackend(replies), model=self.model))
            hits.append(float(judge.predict_step(run) == mistake))
        hit = float(np.mean(hits))
        return dict(zip(_METRICS, (hit, hit, hit)))


def _register():
    from catchbench.llm_judge import discovered_llm_judge_methods

    for m in discovered_llm_judge_methods():
        alias = f"llm-judge {m.method.replace('_', '-')}"
        item(task_id="post_localization", board_alias=alias, variant=m.model,
             factory="discovered_llm_judge_methods", method_id=_LIBRARY[m.method],
             build=lambda o: _LLMJudge(o),
             match=lambda o, a=alias, mod=m.model: getattr(o, "method_id", None) == f"{a} ({mod})")


_register()
sentinel("catchbench.llm_judge:_score_vector", "catchbench.llm_judge:LLMJudgeLocalization.evaluate")


# --- per-sample comparisons ------------------------------------------------------------------

def _keys():
    from catchbench_parity import manifest

    return [it.key for it in manifest.ITEMS if it.factory == "discovered_llm_judge_methods"]


def _scripted(prompt: str) -> str:
    """A deterministic stand-in for a model: the reply depends only on the prompt text."""
    import hashlib
    import re

    h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
    if "FIRST half" in prompt:
        return "VERDICT: " + ("FIRST" if h % 2 else "SECOND")
    m = re.search(r"You are shown steps 0 through (\d+)", prompt)
    if m:
        upto = int(m.group(1))
        return "VERDICT: NO" if h % 3 else f"VERDICT: {h % (upto + 1)}"
    n = len(re.findall(r"^\[step ", prompt, flags=re.M))
    ranking = sorted(range(n), key=lambda i: (h >> (i % 64)) & 7)
    return f"MISTAKE_STEP: {ranking[0]}\nRANKING: " + ", ".join(map(str, ranking))


@check(*_keys())
def check_llm(ctx, it):
    """Per run: the library's prediction against the recorded one. All-at-once re-parses the
    recorded reply (prediction and score vector). Single-step replays the recorded decisions
    through the library driver (named step, query count, and log headers)."""
    from auditable.llm import FixedResponseBackend
    from auditable.llm.who_when import _score_vector as lib_score
    from auditable.methods import get_method
    from catchbench.llm_judge import _score_vector

    task = ctx.task("post_localization")
    view = views.localization_view(task, texts=True)
    method = next(k for k, v in _LIBRARY.items() if v == it.method_id)
    recorded = _recorded(method, it.variant)
    cls = get_method(it.method_id)
    n = bad = 0
    for run, rec in zip(view.runs, recorded):
        n += 1
        if method == "all_at_once":
            backend = FixedResponseBackend([rec["raw"]])
            pred = cls(backend=backend, model=it.variant).predict(run)
            same = (pred["top"] == rec["top"] and pred["ranking"] == rec["ranking"]
                    and ctx.same(lib_score(pred, len(run["steps"])), _score_vector(rec, len(run["steps"]))))
        else:
            replies, headers = _decision_replies(method, rec)
            backend = FixedResponseBackend(replies)
            pred = cls(backend=backend, model=it.variant).predict(run)
            same = (pred["top"] == rec["top"] and len(backend.prompts) == len(replies)
                    and _headers(method, pred["raw"]) == headers)
        bad += not same
    return {"compared": n, "mismatches": bad, "unit": "runs (prediction, and score vector or query log)"}


@check(("post_localization", "llm-judge protocols", "scripted"))
def check_protocols(ctx, it):
    """Both implementations of each protocol driven by the same scripted backend: prompts in
    call order, the prediction, and the raw log must all be equal, run by run."""
    from auditable.llm import who_when as lib
    from catchbench import llm_judge as ref

    task = ctx.task("post_localization")
    view = views.localization_view(task, texts=True)
    n = bad = 0
    for name in ("all_at_once", "step_by_step", "binary_search"):
        for run in view.runs:
            lib_prompts, ref_prompts = [], []
            lib_out = lib.PROTOCOLS[name](run, lambda p: (lib_prompts.append(p), _scripted(p))[1])
            ref_out = ref._PROTOCOLS[name](run, lambda p: (ref_prompts.append(p), _scripted(p))[1])
            n += 1
            bad += not (lib_out == ref_out and lib_prompts == ref_prompts)
    return {"compared": n, "mismatches": bad, "unit": "(protocol, run) prompt sequences and outputs"}
