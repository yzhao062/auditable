"""Agent traces to try the methods on: synthetic ones, or public datasets through agentpile.

>>> from auditable.datasets import load_traces, make_traces
>>> traces, y, mistakes = make_traces(n_runs=100, random_state=0)   # doctest: +SKIP
>>> traces, y, mistakes = load_traces("tau_bench", deps=["tool_call_id_v1"], limit=50)   # doctest: +SKIP

The synthetic construction is a toy, not a benchmark: numbers measured on it describe these
traces only. For real corpora and evaluation protocols, use CatchBench.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

AGENTS = ("planner", "coder", "tester")


def make_traces(n_runs: int = 100, contamination: float = 0.5, n_steps: Tuple[int, int] = (10, 20),
                random_state: Optional[Any] = None):
    """Synthetic multi-agent traces with one planted mistake in each failed run.

    Each run is a sequence of steps by three agents, each a decision or a tool call, and
    each step relies on one or two of the steps just before it. A failed run contains one
    stale read: a step in the second half of the run that relies on step 0 alone, as an
    agent acting on state recorded long before would.

    Parameters
    ----------
    n_runs : number of runs.
    contamination : fraction of runs that fail, rounded to a whole number of runs.
    n_steps : the smallest and largest run length, inclusive; the smallest is at least 4.
    random_state : an int seed, a ``numpy.random.RandomState``, or None.

    Returns
    -------
    traces : a list of ``n_runs`` traces, each a list of step dicts with ``idx``, ``agent``,
        ``kind`` and ``deps``, the input every method family reads.
    y : int array of shape (n_runs,), 1 for a failed run and 0 otherwise.
    mistakes : int array of shape (n_runs,), the ``idx`` of the stale read in a failed run
        and -1 in the others.
    """
    import numpy as np

    lo, hi = n_steps
    if not 4 <= lo <= hi:
        raise ValueError("n_steps must be (smallest, largest) with 4 <= smallest <= largest")
    if not 0.0 <= contamination <= 1.0:
        raise ValueError("contamination must be between 0 and 1")
    rng = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    y = np.zeros(n_runs, dtype=int)
    y[rng.permutation(n_runs)[:int(round(contamination * n_runs))]] = 1
    mistakes = np.full(n_runs, -1, dtype=int)
    traces: List[List[Dict[str, Any]]] = []
    for r in range(n_runs):
        n = int(rng.randint(lo, hi + 1))
        if y[r]:
            mistakes[r] = int(rng.randint(n // 2, n))
        steps = []
        for i in range(n):
            if i == 0:
                deps = []
            elif i == mistakes[r]:
                deps = [0]
            else:
                deps = sorted({i - 1, max(0, i - 1 - int(rng.randint(0, 2)))})
            steps.append({"idx": i, "agent": AGENTS[int(rng.randint(0, 3))],
                          "kind": "tool_call" if rng.rand() < 0.5 else "decision", "deps": deps})
        traces.append(steps)
    return traces, y, mistakes


def load_traces(name: str, *, deps: Any, convention: str = "generic", limit: Optional[int] = None,
                subagents: str = "error", **load_kwargs: Any):
    """One agentpile dataset as ``(traces, y, mistakes)``, the shapes :func:`make_traces` returns.

    ``name`` is an agentpile dataset (``agentpile.datasets.names()``) or a trace file path;
    ``limit`` and ``load_kwargs`` (``subset=``, ``root=``, ``model=`` and the like) go to
    ``agentpile.load``. ``deps``, ``convention`` and ``subagents`` go to
    :func:`auditable.from_agentpile`; ``deps`` has no default.

    Returns
    -------
    traces : a list of :class:`~auditable.featurize.TraceView`, one per run.
    y : int array, 1 for a failed run, 0 for a successful one, -1 where the dataset gives no
        outcome (:func:`auditable.run_targets`). A dataset whose runs all failed, such as
        Who&When, has no second class to detect.
    mistakes : int array, the ``idx`` of the labeled decisive-mistake step, -1 where there is
        none (:func:`auditable.step_targets`).
    """
    from auditable.featurize.agentpile import _agentpile, from_agentpile, run_targets, step_targets

    ap = _agentpile()

    if limit is not None:
        load_kwargs["limit"] = limit
    traces = from_agentpile(ap.load(name, **load_kwargs), deps=deps, convention=convention, subagents=subagents)
    return traces, run_targets(traces), step_targets(traces)
