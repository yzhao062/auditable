"""tau-bench retail and airline trajectories: the tool-using population of the board.

Ported from GRADE (https://github.com/yzhao062/grade, MIT License) at commit 3839a57,
``experiment/agent_graph_tau_bench.py`` (``_ensure_files``, ``load_runs``, ``to_steps``) and
``experiment/agent_failure_detection.load_tau``. Four model trajectory files are loaded in the
same order as GRADE's ``MODELS`` list (``claude-4.5-sonnet-thinking-on-10k``, ``gpt-4.1``,
``gpt-4o-mini``, ``Kimi-K2-Instruct``), keeping each file's line order. Runs with fewer than
three original messages, without ``db_match`` in ``eval_result``, or with fewer than two steps
after skipping system messages are dropped; the failure label is 1 when ``db_match`` is false.

At the dataset revision CatchBench pins (``382e57d1``) this is 660 runs, 363 failed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

DATASET = "tau_bench"
MODELS = (
    "claude-4.5-sonnet-thinking-on-10k",
    "gpt-4.1",
    "gpt-4o-mini",
    "Kimi-K2-Instruct",
)


def _model_stem(run: Any) -> str:
    source = getattr(run, "source", None) or {}
    file_name = source.get("file") or ""
    if file_name.endswith(".jsonl"):
        return file_name[:-6]
    if file_name:
        return file_name
    dataset = getattr(run, "dataset", None) or {}
    return str(dataset.get("model_path") or "")


def _n_messages(run: Any) -> int:
    """Count original messages from ATIF steps and attached tool-result observations."""
    steps = getattr(run, "steps", None) or ()
    if len(steps) == 1 and ((steps[0].get("extra") or {}).get("agentpile") or {}).get("empty_source"):
        return 0
    return sum(1 + len(((step.get("observation") or {}).get("results")) or ()) for step in steps)


def load_runs(models: Sequence[str] = MODELS, **load_kwargs: Any) -> Tuple[List[Any], List[int], Dict[str, Any]]:
    """``(views, y, info)``: the population as :class:`~auditable.featurize.TraceView` objects
    (``convention="grade"``), labels with 1 when ``db_match`` is false, and selection counts."""
    import agentpile as ap

    from auditable.featurize.agentpile import from_agentpile

    model_list = list(models)
    runs = list(ap.load(DATASET, model=model_list, **load_kwargs))
    if not runs:
        raise RuntimeError(f"agentpile returned no {DATASET} runs")

    # agentpile iterates files in sorted filename order, whereas GRADE reads MODELS in list order.
    order = {name: i for i, name in enumerate(model_list)}
    runs.sort(key=lambda r: order.get(_model_stem(r), len(order)))

    eligible = []
    dropped = {"messages_lt_3": 0, "no_db_match": 0, "steps_lt_2": 0}
    for run in runs:
        if _n_messages(run) < 3:
            dropped["messages_lt_3"] += 1
            continue
        eligible.append(run)

    views: List[Any] = []
    labels: List[int] = []
    by_file: Dict[str, int] = {}
    if eligible:
        for view in from_agentpile(eligible, deps="grade", convention="grade"):
            eval_result = ((view.meta.get("outcome") or {}).get("raw") or {}).get("eval_result") or {}
            if "db_match" not in eval_result:
                dropped["no_db_match"] += 1
                continue
            if len(view.steps) < 2:
                dropped["steps_lt_2"] += 1
                continue
            views.append(view)
            labels.append(int(view.meta["label_failed"]))
            file_name = str((view.meta.get("source") or {}).get("file") or "")
            by_file[file_name] = by_file.get(file_name, 0) + 1

    info = {
        "models": model_list,
        "files": list(by_file),
        "counts_by_file": by_file,
        "dropped": dropped,
    }
    return views, labels, info
