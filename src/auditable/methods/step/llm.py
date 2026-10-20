"""LLM-judge step localization: the three Who&When query protocols.

Each method asks an LLM which step of a failed run is the decisive mistake. Predictions go
through a :class:`~auditable.llm.PredictionCache`, so a run is queried once and replayed
afterwards; with ``replay_only=True`` a cache miss raises instead of calling the backend.
The cache key is the input content (method, prompt version, model, steps and texts), so
two identical traces share one entry; keep one cache per trace to hold distinct recorded
answers for identical inputs.

- ``step.llm_all_at_once`` returns a ranking and therefore per-step scores;
- ``step.llm_step_by_step`` and ``step.llm_binary_search`` name one step
  (``output_kind="single"``). Read them with :meth:`predict_step`. They have no
  ``step_scores``: ranking metrics computed over a one-step answer padded with zeros
  would credit guesses the method never made.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from auditable.base import BaseStepLocalizer, Citation, MethodCard, NotSupported
from auditable.methods import register

PROMPT_VERSION = "who-when-v1"  # bump when a prompt or parser changes; it is part of every cache key

_WHO_WHEN = Citation(key="Who&When (Zhang et al., 2025)",
                     title="Which Agent Causes Task Failures and When? On Automated Failure Attribution "
                           "of LLM Multi-Agent Systems",
                     arxiv="2505.00212")


class _LLMJudgeLocalizer(BaseStepLocalizer):
    _strategy: str = ""

    def __init__(self, backend: Optional[Any] = None, model: str = "", cache: Optional[Any] = None,
                 replay_only: bool = False) -> None:
        self.backend = backend
        self.model = model
        self.cache = cache
        self.replay_only = replay_only

    def _cache(self):
        from auditable.llm import PredictionCache

        if isinstance(self.cache, PredictionCache):
            return self.cache
        if not hasattr(self, "_own_cache"):
            self._own_cache = PredictionCache(path=self.cache if isinstance(self.cache, str) else None,
                                              provenance={"method": self.card.method_id,
                                                          "prompt_version": PROMPT_VERSION,
                                                          "model": self.model})
        return self._own_cache

    @staticmethod
    def _run_dict(trace: Any) -> Dict[str, Any]:
        """The protocol input. Steps are labelled by position, which the prompts describe as
        0..n-1 and the parsers read back; a trace's own ``idx`` values may be any unique ints."""
        from auditable.featurize import normalize_trace

        view = normalize_trace(trace)
        texts = view.require_texts()
        steps = [{"idx": i, "agent": s["agent"], "kind": s["kind"]} for i, s in enumerate(view.steps)]
        return {"steps": steps, "texts": [str(t) for t in texts]}

    def cache_key(self, trace: Any) -> str:
        """The cache key of ``trace``: method, prompt version, model, and the exact step list and
        texts the prompt is built from."""
        from auditable.llm import input_key

        return input_key(self.card.method_id, PROMPT_VERSION, self.model, self._run_dict(trace))

    def predict(self, trace: Any) -> Dict[str, Any]:
        """``{"top": step position or None, "ranking": [positions], "raw": reply or log}``."""
        from auditable.llm.who_when import PROTOCOLS

        key = self.cache_key(trace)
        cache = self._cache()
        hit = cache.get(key)
        if hit is not None:
            return hit
        if self.replay_only:
            raise KeyError(f"{self.card.method_id}: no cached prediction for this trace (replay_only=True)")
        if self.backend is None:
            raise ValueError(f"{self.card.method_id} needs a backend (a callable prompt -> reply) or a "
                             "cache holding this trace")
        pred, raw = PROTOCOLS[self._strategy](self._run_dict(trace), self.backend)
        record = {**pred, "raw": raw}
        cache.put(key, record)
        return record

    def predict_step(self, trace: Any) -> Optional[int]:
        """The step position the judge names as the decisive mistake, or None."""
        return self.predict(trace)["top"]


@register
class LLMAllAtOnce(_LLMJudgeLocalizer):
    """Show the whole trace once; the judge names the decisive step and ranks the rest."""

    _strategy = "all_at_once"
    card = MethodCard(
        method_id="step.llm_all_at_once",
        name="LLM judge, all-at-once",
        family="step",
        fit_scope="stateless",
        supervised=False,
        output_kind="ranking",
        fidelity="faithful",
        requires=frozenset({"step_text", "agent", "kind"}),
        cite=(_WHO_WHEN,),
        extras=("tabular-ad",),  # NumPy builds the score vector; predict_step needs no extra
        board_aliases=("llm-judge all-at-once",),
        benchmark_role="baseline",
        summary="One query over the full trace; returns the decisive step and a ranked shortlist.",
    )

    def step_scores(self, trace: Any):
        from auditable.featurize import normalize_trace
        from auditable.llm.who_when import _score_vector

        return _score_vector(self.predict(trace), len(normalize_trace(trace).steps))


class _SingleStepJudge(_LLMJudgeLocalizer):
    def step_scores(self, trace: Any):
        raise NotSupported(f"{self.card.method_id} names one step; use predict_step(trace)")

    def step_scores_batch(self, traces):
        raise NotSupported(f"{self.card.method_id} names one step; use predict_step(trace)")


@register
class LLMStepByStep(_SingleStepJudge):
    """Walk a growing prefix and stop at the first prefix the judge says contains the mistake."""

    _strategy = "step_by_step"
    card = MethodCard(
        method_id="step.llm_step_by_step",
        name="LLM judge, step-by-step",
        family="step",
        fit_scope="stateless",
        supervised=False,
        output_kind="single",
        fidelity="faithful",
        requires=frozenset({"step_text", "agent", "kind"}),
        cite=(_WHO_WHEN,),
        board_aliases=("llm-judge step-by-step",),
        benchmark_role="baseline",
        summary="Up to n queries over growing prefixes; names one step.",
    )


@register
class LLMBinarySearch(_SingleStepJudge):
    """Halve the suspect interval with each query until one step remains."""

    _strategy = "binary_search"
    card = MethodCard(
        method_id="step.llm_binary_search",
        name="LLM judge, binary-search",
        family="step",
        fit_scope="stateless",
        supervised=False,
        output_kind="single",
        fidelity="faithful",
        requires=frozenset({"step_text", "agent", "kind"}),
        cite=(_WHO_WHEN,),
        board_aliases=("llm-judge binary-search",),
        benchmark_role="baseline",
        summary="About log2(n) interval queries over the full trace; names one step.",
    )
