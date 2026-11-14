"""LLM-judge rows for Who&When step localization (``localize/whoandwhen``).

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.3, ``src/catchbench/llm_judge.py`` and ``src/catchbench/via_auditable/llm.py``. Every
protocol runs through the library's own driver, prompts, and parser, and every run is
replayed from the committed prediction caches under ``data/llm_judge/`` with no network
or API calls.

Expected localization population interface (for joining with ``tasks/localize.py``):

- ``population.runs``: sequence of 126 failed Who&When traces in GRADE run order
  (lexicographic ``source["file"]`` order, not raw ``agentpile.load`` numeric order),
  each readable by :func:`auditable.featurize.normalize_trace` with step texts (such as
  :class:`~auditable.featurize.TraceView` from ``from_agentpile`` or ``{"steps": ...,
  "texts": ...}`` dicts; if ``population`` carries ``step_texts`` separately,
  ``score_llm_entry`` attaches ``step_texts[i]`` when ``runs[i]`` lacks texts).
- ``population.mistakes``: sequence of 126 integer mistake step positions (0-based index
  into each run's steps), aligned with ``population.runs``.
- ``population.groups``: 1D integer array of pooled run indices (``0..125``), one entry per
  step across ``population.runs``.
- ``population.mistake_row``: mapping ``{run_index: pooled_row_index}`` giving the pooled
  row of each run's mistake step.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from auditable.bench._core import Entry
from auditable.llm import FixedResponseBackend

_METRICS = ("top1", "top3", "mrr")
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "llm_judge")
_PROTOCOLS = ("all_at_once", "binary_search", "step_by_step")
_SINGLE_STEP_PROTOCOLS = frozenset({"step_by_step", "binary_search"})
_METHOD_IDS: Dict[str, str] = {
    "all_at_once": "step.llm_all_at_once",
    "binary_search": "step.llm_binary_search",
    "step_by_step": "step.llm_step_by_step",
}
_PROTOCOL_BY_METHOD: Dict[str, str] = {v: k for k, v in _METHOD_IDS.items()}

_SBS_LINE = re.compile(r"(?m)^\[0\.\.(\d+)\] ")
_BS_LINE = re.compile(r"(?m)^\[(\d+)-(\d+) mid(\d+)\] (FIRST|SECOND) :: ")


class ReplayBackend(FixedResponseBackend):
    """Fixed-response backend that replays cached replies and refuses uncached prompts."""

    def __init__(
        self,
        responses: Union[Sequence[str], Mapping[str, str]] = (),
        *,
        method: str = "",
        model: str = "",
        cache_path: Optional[str] = None,
    ) -> None:
        stored: Union[List[str], Dict[str, str]] = (
            dict(responses) if isinstance(responses, Mapping) else list(responses)
        )
        super().__init__(stored, default=None)
        self.method = method
        self.model = model
        self.cache_path = cache_path

    def __call__(self, prompt: str) -> str:
        try:
            return super().__call__(prompt)
        except LookupError as exc:
            target = f" for {self.method}/{self.model}" if (self.method or self.model) else ""
            raise LookupError(f"uncached prompt{target}; replay backend makes no network calls") from exc


def cache_path(method: str, model: str, cache_dir: Optional[str] = None) -> str:
    """Path to the committed JSON prediction cache for ``(method, model)``."""
    root = cache_dir or _DEFAULT_CACHE_DIR
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    return os.path.join(root, f"whoandwhen__{method}__{safe_model}.json")


def legacy_key_map(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Bridge from order-prefixed v1 keys to full-record SHA-256 content addresses."""
    root = cache_dir or _DEFAULT_CACHE_DIR
    path = os.path.join(root, "legacy_run_keys.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    mapping = payload.get("keys") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not all(
        isinstance(old, str) and isinstance(new, str) for old, new in mapping.items()
    ):
        raise ValueError(f"invalid legacy LLM-judge key map: {path}")
    return mapping


def ordered_run_keys(cache_dir: Optional[str] = None) -> List[str]:
    """The 126 canonical run keys in GRADE run order, from ``legacy_run_keys.json``."""
    mapping = legacy_key_map(cache_dir)
    if not mapping:
        root = cache_dir or _DEFAULT_CACHE_DIR
        raise FileNotFoundError(f"missing legacy_run_keys.json in {root}")
    return [mapping[k] for k in sorted(mapping)]


def _normalize_predictions(cache: dict, mapping: Mapping[str, str]) -> dict:
    predictions = cache.get("predictions") if isinstance(cache, dict) else None
    if not isinstance(predictions, dict):
        return cache
    normalized: Dict[str, Any] = {}
    for key, prediction in predictions.items():
        content_key = mapping.get(key, key)
        if content_key in normalized and normalized[content_key] != prediction:
            raise ValueError(f"legacy cache keys collide at {content_key}")
        normalized[content_key] = prediction
    return {**cache, "predictions": normalized}


def load_cache(method: str, model: str, cache_dir: Optional[str] = None) -> Optional[dict]:
    """Load and key-normalize one committed LLM-judge cache, or return None if absent."""
    path = cache_path(method, model, cache_dir=cache_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        cache = json.load(fh)
    return _normalize_predictions(cache, legacy_key_map(cache_dir))


def load_cache_file(path: str, cache_dir: Optional[str] = None) -> dict:
    """Load and key-normalize one cache JSON from an explicit file path."""
    with open(path, encoding="utf-8") as fh:
        cache = json.load(fh)
    root = cache_dir or os.path.dirname(path)
    return _normalize_predictions(cache, legacy_key_map(root))


def _pred_shape_error(pred: Any, n_steps: Optional[int]) -> Optional[str]:
    if not isinstance(pred, dict) or not {"top", "ranking", "raw"} <= set(pred):
        return "not a {top, ranking, raw} dict"
    top = pred["top"]
    if top is not None and (isinstance(top, bool) or not isinstance(top, int) or top < 0 or (n_steps is not None and top >= n_steps)):
        return "top is not None or an in-range int"
    ranking = pred["ranking"]
    if not isinstance(ranking, list) or any(
        isinstance(i, bool) or not isinstance(i, int) or i < 0 or (n_steps is not None and i >= n_steps)
        for i in ranking
    ):
        return "ranking is not a list of in-range step indices"
    if not isinstance(pred["raw"], str):
        return "raw is not a string"
    return None


def _cache_errors(
    cache: Any,
    expected_keys: Optional[Sequence[str]] = None,
    step_counts: Optional[Mapping[str, int]] = None,
) -> List[str]:
    if not isinstance(cache, dict):
        return ["cache root is not a dict"]
    predictions = cache.get("predictions")
    if not isinstance(predictions, dict):
        return ["missing predictions object"]
    errors: List[str] = []
    if expected_keys is not None:
        expected = set(expected_keys)
        actual = set(predictions)
        if expected - actual:
            errors.append(f"missing {len(expected - actual)} expected run predictions")
        if actual - expected:
            errors.append(f"has {len(actual - expected)} unknown run predictions")
        keys_to_check = expected & actual
    else:
        if not predictions:
            return ["predictions dict is empty"]
        keys_to_check = set(predictions)
    for key in sorted(keys_to_check):
        n_steps = step_counts.get(key) if step_counts is not None else None
        shape_error = _pred_shape_error(predictions[key], n_steps)
        if shape_error:
            errors.append(f"{key}: {shape_error}")
            break
    return errors


def discover_llm_caches(cache_dir: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """Discover valid ``(protocol, model, path)`` caches under ``cache_dir``.

    Ordered lexicographically by filename (method then model), matching CatchBench
    ``discovered_llm_judge_methods``. Incomplete or malformed caches are skipped.
    """
    root = cache_dir or _DEFAULT_CACHE_DIR
    mapping = legacy_key_map(root)
    expected_keys = [mapping[k] for k in sorted(mapping)] if mapping else None
    out: List[Tuple[str, str, str]] = []
    for path in sorted(glob.glob(os.path.join(root, "whoandwhen__*__*.json"))):
        stem = os.path.basename(path)[: -len(".json")]
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        _, method, model = parts
        if method not in _PROTOCOLS:
            continue
        cache = load_cache_file(path, cache_dir=root)
        if _cache_errors(cache, expected_keys=expected_keys):
            continue
        out.append((method, model, path))
    return out


def llm_entry_key(protocol: str, model: str) -> str:
    """Golden entry key for an LLM-judge board row: ``step.llm_<protocol>:<model>``."""
    return f"step.llm_{protocol}:{model}"


def llm_entry_label(protocol: str, model: str) -> str:
    """Board display label matching CatchBench: ``llm-judge <protocol> (<model>)``."""
    return f"llm-judge {protocol.replace('_', '-')} ({model})"


def llm_entries(cache_dir: Optional[str] = None) -> List[Entry]:
    """Return one :class:`Entry` per discovered cached ``(protocol, model)`` combination."""
    from auditable.methods import get_method

    entries: List[Entry] = []
    for protocol, model, path in discover_llm_caches(cache_dir=cache_dir):
        cls = get_method(_METHOD_IDS[protocol])
        backend = ReplayBackend({}, method=protocol, model=model, cache_path=path)
        entries.append(
            Entry(
                method=cls,
                kwargs={"backend": backend, "model": model},
                label=llm_entry_label(protocol, model),
                key=llm_entry_key(protocol, model),
            )
        )
    return entries


def is_llm_entry(entry: Entry) -> bool:
    """True when ``entry`` is one of the three Who&When LLM-judge step localizers."""
    return entry.card.method_id in _PROTOCOL_BY_METHOD


def _headers(method: str, log: str) -> list:
    if method == "step_by_step":
        return [int(m.group(1)) for m in _SBS_LINE.finditer(log)]
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)) for m in _BS_LINE.finditer(log)]


def _decision_replies(method: str, rec: Mapping[str, Any]) -> Tuple[List[str], list]:
    headers = _headers(method, rec["raw"])
    if method == "step_by_step":
        if not headers or headers != list(range(len(headers))) or rec["top"] > headers[-1]:
            raise ValueError("step-by-step log is not a consecutive prefix sequence ending at the named step")
        top = rec["top"]
        return ["VERDICT: NO"] * (len(headers) - 1) + [f"VERDICT: {top}"], headers
    return [f"VERDICT: {h[3]}" for h in headers], headers


def _rank_metrics(scores: Any, groups: Any, mistake_row: Mapping[int, int]) -> Tuple[float, float, float]:
    """Ported from GRADE ``agent_failure_localization._rank_metrics`` (stable descending sort)."""
    import numpy as np

    top1: List[bool] = []
    top3: List[bool] = []
    rr: List[float] = []
    for ri in sorted(set(groups.tolist())):
        rows = np.where(groups == ri)[0]
        order = rows[np.argsort(-scores[rows], kind="stable")]
        rank = int(np.where(order == mistake_row[ri])[0][0]) + 1
        top1.append(rank == 1)
        top3.append(rank <= 3)
        rr.append(1.0 / rank)
    return float(np.mean(top1)), float(np.mean(top3)), float(np.mean(rr))


def _runs_with_texts(population: Any) -> List[Any]:
    from auditable.featurize import normalize_trace

    step_texts = getattr(population, "step_texts", None)
    runs = list(population.runs)
    if step_texts is None:
        return runs
    out: List[Any] = []
    for i, run in enumerate(runs):
        view = normalize_trace(run)
        if view.texts is None:
            out.append({"steps": view.steps, "texts": list(step_texts[i])})
        else:
            out.append(run)
    return out


def _pooling_for(runs: Sequence[Any], mistakes: Sequence[int], population: Any):
    import numpy as np
    from auditable.featurize import normalize_trace

    groups = getattr(population, "groups", None)
    mistake_row = getattr(population, "mistake_row", None)
    if groups is not None and mistake_row is not None:
        return np.asarray(groups), dict(mistake_row)
    built_groups: List[int] = []
    built_mistake_row: Dict[int, int] = {}
    base = 0
    for ri, (run, ms) in enumerate(zip(runs, mistakes)):
        n_steps = len(normalize_trace(run).steps)
        built_groups.extend([ri] * n_steps)
        built_mistake_row[ri] = base + int(ms)
        base += n_steps
    return np.asarray(built_groups), built_mistake_row


def _mismatch(key: str, run_index: int, what: str, library: Any, recorded: Any) -> ValueError:
    return ValueError(
        f"{key}: run {run_index}: the library replay gives {what} {library!r}, "
        f"the cache records {recorded!r}"
    )


def score_llm_entry(entry: Entry, population: Any, cache_dir: Optional[str] = None) -> Dict[str, float]:
    """Score one LLM-judge :class:`Entry` on a Who&When localization population via cache replay."""
    import numpy as np
    from auditable.featurize import normalize_trace

    method_id = entry.card.method_id
    if method_id not in _PROTOCOL_BY_METHOD:
        raise ValueError(f"{method_id!r} is not an LLM-judge step localization method")
    protocol = _PROTOCOL_BY_METHOD[method_id]
    model = str(entry.kwargs.get("model", ""))
    backend_obj = entry.kwargs.get("backend")

    if isinstance(backend_obj, ReplayBackend) and backend_obj.cache_path:
        c_dir = cache_dir or os.path.dirname(backend_obj.cache_path)
        if not os.path.exists(backend_obj.cache_path):
            raise FileNotFoundError(f"no LLM-judge cache at {backend_obj.cache_path}")
        cache = load_cache_file(backend_obj.cache_path, cache_dir=c_dir)
    else:
        c_dir = cache_dir or _DEFAULT_CACHE_DIR
        cache = load_cache(protocol, model, cache_dir=c_dir)
        if cache is None:
            raise FileNotFoundError(f"no LLM-judge cache for {protocol}/{model} in {c_dir}")

    runs = _runs_with_texts(population)
    mistakes = list(population.mistakes)
    if len(runs) != len(mistakes):
        raise ValueError(f"population runs ({len(runs)}) and mistakes ({len(mistakes)}) differ in length")

    pop_keys = getattr(population, "run_keys", None)
    if pop_keys is not None:
        keys = list(pop_keys)
    else:
        keys = ordered_run_keys(c_dir)
    if len(keys) != len(runs):
        raise ValueError(f"expected {len(keys)} runs to match cache keys, got {len(runs)}")

    step_counts = {k: len(normalize_trace(r).steps) for k, r in zip(keys, runs)}
    errors = _cache_errors(cache, expected_keys=keys, step_counts=step_counts)
    if errors:
        raise ValueError(f"unscoreable LLM-judge cache for {protocol}/{model}: " + "; ".join(errors))

    preds = cache["predictions"]
    recorded = [preds[k] for k in keys]
    groups, mistake_row = _pooling_for(runs, mistakes, population)

    # One judge (and one in-memory PredictionCache) per run: Who&When has two pairs of runs
    # with identical steps and text whose recorded answers differ.
    if protocol == "all_at_once":
        chunks = []
        for i, (run, rec) in enumerate(zip(runs, recorded)):
            judge = entry.make(backend=ReplayBackend([rec["raw"]], method=protocol, model=model))
            pred = judge.predict(run)
            if pred["top"] != rec["top"]:
                raise _mismatch(entry.key, i, "top", pred["top"], rec["top"])
            if list(pred["ranking"]) != list(rec["ranking"]):
                raise _mismatch(entry.key, i, "ranking", pred["ranking"], rec["ranking"])
            chunks.append(judge.step_scores(run))
        metrics = _rank_metrics(np.concatenate(chunks), groups, mistake_row)
        return dict(zip(_METRICS, (float(v) for v in metrics)))

    hits: List[float] = []
    for i, (run, rec, mistake) in enumerate(zip(runs, recorded, mistakes)):
        replies, headers = _decision_replies(protocol, rec)
        backend = ReplayBackend(replies, method=protocol, model=model)
        judge = entry.make(backend=backend)
        pred = judge.predict(run)
        if len(backend.prompts) != len(replies):
            raise _mismatch(entry.key, i, "query count", len(backend.prompts), len(replies))
        if _headers(protocol, pred["raw"]) != headers:
            raise _mismatch(entry.key, i, "log headers", _headers(protocol, pred["raw"]), headers)
        if pred["top"] != rec["top"]:
            raise _mismatch(entry.key, i, "step", pred["top"], rec["top"])
        hits.append(float(judge.predict_step(run) == mistake))
    # Single-step protocols name one step, so Top-1, Top-3, and MRR equal the hit rate.
    hit = float(np.mean(hits))
    return dict(zip(_METRICS, (hit, hit, hit)))


@dataclass
class _WhoAndWhenPopulation:
    """Internal Who&When 126-run population matching the LocalizationView interface."""

    runs: List[Any]
    mistakes: List[int]
    groups: Any
    mistake_row: Dict[int, int]


def _load_whoandwhen_population(**load_kwargs: Any) -> _WhoAndWhenPopulation:
    """The 126-run Who&When population of ``localize/whoandwhen`` (runs in GRADE order, with step
    texts), built by the task's own corpus loader so the LLM rows and the board read one population."""
    from auditable.bench.corpora.whoandwhen import load_runs, pool_steps

    runs, mistakes, _info = load_runs(**load_kwargs)
    groups, mistake_row = pool_steps(runs, mistakes)
    return _WhoAndWhenPopulation(runs=list(runs), mistakes=list(mistakes), groups=groups,
                                 mistake_row=mistake_row)
