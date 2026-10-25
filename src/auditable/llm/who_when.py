"""Who&When LLM-judge query protocols: prompts, response parsing, and the three drivers.

Zhang et al. (2025, Who&When, arXiv:2505.00212) define three ways to ask an LLM which step
of a failed multi-agent run is the decisive mistake: show the whole trace once
(all-at-once), walk a growing prefix (step-by-step), or halve the suspect interval
(binary-search). Ported unchanged from CatchBench (https://github.com/yzhao062/catchbench,
MIT License) at release 0.1.2, ``llm_judge.py``; the function bodies and prompt wording
are kept character for character so a prediction cache made by either copy stays valid.

Six functions are frozen: ``_all_at_once``, ``all_at_once_prompt``, ``_step_by_step``,
``_step_prompt``, ``_binary_search`` and ``_interval_prompt``. CatchBench hashes their source
text (``inspect.getsource``) into the metadata of its LLM caches, so any edit to them, a
docstring or whitespace included, makes those caches stale; ``tests/test_llm.py`` pins the
hashes, and the parsers' behaviour separately.

A ``run`` here is a dict with ordered ``steps`` (each with ``idx``, ``agent``, ``kind``)
and aligned ``texts``. Each driver takes ``(run, complete)``, where ``complete`` maps a
prompt to a reply, and returns ``(prediction, raw_log)`` with
``prediction = {"top": step or None, "ranking": [steps]}``.
"""
from __future__ import annotations

import re
from typing import Callable, List, Mapping, Optional


def _trace_block(run: dict, max_chars: int = 1500) -> str:
    """Render the run as a numbered step list (index, agent, content) for the prompt. Long step
    contents are truncated to keep the whole trace within a sane budget; the index and agent, which
    the answer is keyed to, are never truncated."""
    lines = []
    for s, text in zip(run["steps"], run["texts"]):
        body = " ".join(text.split())
        if len(body) > max_chars:
            body = body[:max_chars] + " ...[truncated]"
        lines.append(f"[step {s['idx']}] ({s['agent']}, {s['kind']}): {body}")
    return "\n".join(lines)


def all_at_once_prompt(run: dict) -> str:
    """Who&When all-at-once: the whole trace, one query, name the decisive mistake step. We also ask
    for a short ranked shortlist so the board's Top-3 and MRR columns have a ranking to score, while
    Top-1 uses the single decisive step the method is defined to produce."""
    n = len(run["steps"])
    return (
        "You are auditing a failed multi-agent run. Each step is one agent action, indexed from 0 to "
        f"{n - 1}. Exactly one step is the decisive mistake: the first step whose error caused the "
        "run to fail (not a downstream symptom of an earlier error).\n\n"
        "TRACE:\n" + _trace_block(run) + "\n\n"
        "Identify the decisive mistake step. Then list the steps from most to least likely to be the "
        "decisive mistake (most likely first), as a comma-separated list of step indices.\n"
        "Answer in exactly this format, nothing else:\n"
        "MISTAKE_STEP: <index>\n"
        "RANKING: <index>, <index>, <index>, ...\n"
    )


def parse_prediction(text: str, n_steps: int) -> dict:
    """Pull MISTAKE_STEP and RANKING out of an LLM response, clamped to valid step indices. Robust to
    extra prose: takes the first integer after each label. A missing ranking falls back to the single
    mistake step; a missing mistake step falls back to the ranking head; total failure yields an empty
    prediction the scorer treats as a miss (score 0 for that run)."""
    top: Optional[int] = None
    ranking: List[int] = []
    m = re.search(r"MISTAKE_STEP\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 0 <= v < n_steps:
            top = v
    m = re.search(r"RANKING\s*:\s*([0-9,\s]+)", text, re.IGNORECASE)
    if m:
        seen = set()
        for tok in re.findall(r"\d+", m.group(1)):
            v = int(tok)
            if 0 <= v < n_steps and v not in seen:
                seen.add(v)
                ranking.append(v)
    if top is None and ranking:
        top = ranking[0]
    if top is not None and not ranking:
        ranking = [top]
    return {"top": top, "ranking": ranking}


def _score_vector(pred: dict, n_steps: int):
    """Turn a parsed prediction into a per-step score (higher = more suspected), so the board's
    ranking metric can read it. The ranked shortlist gets descending scores; the decisive step is
    forced to the top; unranked steps get zero."""
    import numpy as np

    scores = np.zeros(n_steps, dtype=float)
    ranking = pred.get("ranking") or []
    for rank, idx in enumerate(ranking):
        if 0 <= idx < n_steps:
            scores[idx] = max(scores[idx], float(n_steps - rank))
    top = pred.get("top")
    if top is not None and 0 <= top < n_steps:
        scores[top] = float(n_steps + 1)  # the single decisive step outranks the shortlist
    return scores


def _all_at_once(run: dict, complete: Callable[[str], str]):
    raw = complete(all_at_once_prompt(run))
    return parse_prediction(raw, len(run["steps"])), raw


def _step_prompt(run: dict, upto: int, max_chars: int = 1200) -> str:
    lines = []
    for s, text in list(zip(run["steps"], run["texts"]))[:upto + 1]:
        body = " ".join(str(text).split())
        if len(body) > max_chars:
            body = body[:max_chars] + " ...[truncated]"
        lines.append(f"[step {s['idx']}] ({s['agent']}, {s['kind']}): {body}")
    return (
        "You are auditing a failed multi-agent run, examined incrementally. The decisive mistake is the "
        "FIRST step whose error caused the run to fail. You are shown steps 0 through "
        f"{upto}; all later steps are hidden. Has the decisive mistake ALREADY occurred within steps 0 "
        f"to {upto}?\n\nSTEPS:\n" + "\n".join(lines) +
        f"\n\nEnd your reply with one line in exactly this format: 'VERDICT: <index>', where <index> is "
        f"an integer 0 to {upto} if the decisive mistake has occurred, or 'VERDICT: NO' if it has not.")


def _parse_step_answer(raw: str, upto: int):
    """Read one step-by-step reply via the required ``VERDICT:`` line and return an in-range step index,
    or None when the verdict is NO or absent (treated as not-yet). The structured verdict makes parsing
    immune to verbose reasoning ("the mistake has not occurred yet, the error in step 4 is ...") that a
    free-form scan would misread. A bare integer reply is accepted as a fallback for non-compliant
    models; anything else is None."""
    low = raw.strip().lower()
    if not low:
        return None
    verdicts = re.findall(r"verdict\s*[:=]\s*(no|\d+)", low)  # last match = the final-line answer
    if verdicts:
        tok = verdicts[-1]
        if tok == "no":
            return None
        v = int(tok)
        return v if 0 <= v <= upto else None
    if re.fullmatch(r"\d+", low):  # non-compliant fallback: a bare index
        v = int(low)
        return v if 0 <= v <= upto else None
    return None


def _step_by_step(run: dict, complete: Callable[[str], str]):
    """Walk the growing prefix; the first prefix where the judge says the mistake has occurred yields
    the step it names (which may be earlier than the current step). If no prefix flags a mistake,
    default to the last step (the failure has surfaced by the end)."""
    n = len(run["steps"])
    log = []
    for i in range(n):
        raw = complete(_step_prompt(run, i)).strip()
        log.append(f"[0..{i}] {raw[:120]}")
        ans = _parse_step_answer(raw, i)
        if not isinstance(ans, int):
            continue  # no step named (mistake not yet visible, or reply unparseable); extend prefix
        return {"top": ans, "ranking": [ans]}, "\n".join(log)
    return {"top": n - 1, "ranking": [n - 1]}, "\n".join(log)


def _interval_prompt(run: dict, lo: int, hi: int, mid: int, max_chars: int = 900) -> str:
    lines = []
    for s, text in zip(run["steps"], run["texts"]):
        body = " ".join(str(text).split())
        if len(body) > max_chars:
            body = body[:max_chars] + " ...[truncated]"
        lines.append(f"[step {s['idx']}] ({s['agent']}, {s['kind']}): {body}")
    return (
        "You are auditing a failed multi-agent run (full trace below). The decisive mistake, the first "
        f"step whose error caused the failure, is within steps {lo} to {hi}. Is it in the FIRST half "
        f"(steps {lo} to {mid}) or the SECOND half (steps {mid + 1} to {hi})?\n\nTRACE:\n"
        + "\n".join(lines) +
        f"\n\nEnd your reply with one line in exactly this format: 'VERDICT: FIRST' (steps {lo}-{mid}) "
        f"or 'VERDICT: SECOND' (steps {mid + 1}-{hi}).")


def _parse_half(ans: str) -> Optional[str]:
    """Read an interval reply via the required ``VERDICT:`` line: ``"first"``, ``"second"``, or None
    when absent. The structured verdict avoids misreading verbose replies that name both halves ("not
    in the first half; it is in the second"). Fallback for a non-compliant reply: a single unambiguous
    half token, else None (the caller defaults to the first half)."""
    low = ans.lower()
    verdicts = re.findall(r"verdict\s*[:=]\s*(first|second)", low)  # last match = the final answer
    if verdicts:
        return verdicts[-1]
    has_f, has_s = "first" in low, "second" in low  # non-compliant fallback: only if unambiguous
    if has_f and not has_s:
        return "first"
    if has_s and not has_f:
        return "second"
    return None


def _binary_search(run: dict, complete: Callable[[str], str]):
    """Halve the suspect interval each query until one step remains (about log2(n) calls). An
    unparseable reply defaults to the first half, keeping the search total."""
    n = len(run["steps"])
    lo, hi, log = 0, n - 1, []
    while lo < hi:
        mid = (lo + hi) // 2
        raw = complete(_interval_prompt(run, lo, hi, mid)).strip()
        go_second = _parse_half(raw) == "second"
        log.append(f"[{lo}-{hi} mid{mid}] {'SECOND' if go_second else 'FIRST'} :: {raw[:120]}")
        lo, hi = (mid + 1, hi) if go_second else (lo, mid)
    return {"top": lo, "ranking": [lo]}, "\n".join(log)


PROTOCOLS: Mapping[str, Callable] = {
    "all_at_once": _all_at_once,
    "step_by_step": _step_by_step,
    "binary_search": _binary_search,
}

# Drivers that name one step (no ranked shortlist). A benchmark scoring them must credit only
# that step: Top-1 == Top-3 == MRR. Expanding the one step into a full score vector and ranking
# it would let zero-scored steps fill ranks 2..n and inflate Top-3 and MRR.
SINGLE_STEP_PROTOCOLS = frozenset({"step_by_step", "binary_search"})
