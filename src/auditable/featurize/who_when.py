"""Per-step structural features for step-level localization (the Who&When step matrix).

Eight columns per step, position first so the one-column position model is a strict
prefix of the full set. Ported from GRADE (https://github.com/yzhao062/grade, MIT
License), ``experiment/agent_failure_localization.py`` at commit 3839a57, with the same
traversal and tie order (``Counter.most_common`` ranks agents by decision count, ties in
first-seen order).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Tuple

STEP_FEATURES = ("norm_pos", "is_tool", "agent_share", "is_most_active",
                 "agent_rank", "agents_so_far", "handoff_out", "handoff_in")


def agent_handoff_degrees(G) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Distinct-agent out and in handoff degree per agent, from the typed graph's
    ``handoff_to`` edges projected to the agents that own the steps. A handoff between two
    steps of the same agent is the agent acting again and is not counted."""
    agent_of = {n: d["agent"] for n, d in G.nodes(data=True) if d["ntype"] in ("decision", "tool_call")}
    out: Dict[str, set] = {}
    inn: Dict[str, set] = {}
    for u, v, d in G.edges(data=True):
        if d["etype"] != "handoff_to":
            continue
        au, av = agent_of.get(u), agent_of.get(v)
        if au is None or av is None or au == av:
            continue
        out.setdefault(au, set()).add(av)
        inn.setdefault(av, set()).add(au)
    return {a: len(s) for a, s in out.items()}, {a: len(s) for a, s in inn.items()}


def who_when_step_matrix(trace: Any):
    """An (n_steps, 8) float array aligned to the trace's steps, columns ``STEP_FEATURES``.

    ``trace`` is anything :func:`auditable.featurize.normalize_trace` reads. The two
    handoff columns come from the trace's typed graph, so parallel branches and merges in
    the execution topology are reflected; when no graph is supplied one is built from the
    steps with the full-context dependency assumption the Who&When corpus uses.
    """
    import numpy as np

    from auditable.featurize.trace import normalize_trace

    view = normalize_trace(trace)
    steps = view.steps
    G = view.typed_graph(dependency="full_context")
    n = len(steps)
    dec = Counter(s["agent"] for s in steps if s["kind"] == "decision")
    rank = {a: i for i, (a, _) in enumerate(dec.most_common())}  # 0 = most active decider
    n_ag = max(1, len({s["agent"] for s in steps}))
    n_dec = max(1, sum(dec.values()))
    hout, hin = agent_handoff_degrees(G)
    seen, rows = set(), []
    for pos, s in enumerate(steps):  # the position, not the step's idx, which may be any unique int
        a = s["agent"]
        seen.add(a)
        rows.append([
            pos / max(1, n - 1),
            1.0 if s["kind"] == "tool_call" else 0.0,
            dec.get(a, 0) / n_dec,
            1.0 if rank.get(a, 10 ** 9) == 0 else 0.0,
            rank.get(a, n_ag) / n_ag,
            len(seen) / n_ag,
            hout.get(a, 0) / n_ag,
            hin.get(a, 0) / n_ag,
        ])
    return np.array(rows, dtype=float)
