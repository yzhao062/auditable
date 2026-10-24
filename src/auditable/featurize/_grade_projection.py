"""GRADE's step conventions for agentpile records (projection ``grade-steps-v1``).

Ported from agentpile (https://github.com/yzhao062/agentpile, Apache-2.0 License),
``tests/compat/grade.py`` at release 0.3.0 (commit 9b9905e; the module was last changed in
8072495, SHA-256 a32b186ab2f7747c359ebff0701780cf8af6601e52421540fe5ab031e23e47ca): the step
functions (``_who_and_when_steps``, ``_tau_bench_steps``, ``_decision_then_calls``,
``_action_turn_steps`` and the two that call it), ``CORPUS_RULES`` and ``label_failed``. They
reproduce the step lists of GRADE's converters at commit 3839a57 from agentpile record fields:
order, agent names and kinds. Dependencies are never computed here; they come from agentpile
rules and are mapped onto the steps through the tables below.

Changes from the source:

- every step carries ``text``, which GRADE's step dicts do not, so the LLM-judge methods can read
  a projected trace: the message for a decision, ``name(arguments) -> result`` for a call, and,
  in the text-action corpora, the executed action (the call as the source wrote it, or the last
  code block of the turn) followed by its observation;
- every step lists the record addresses it stands for (``owners``). The source's endpoint table
  (``nodes``) is kept unchanged: it is how GRADE's own rules resolve, and for the text-action
  corpora it sends a bare step address to the action's tool step. ``owners`` sends a bare step
  address to the step's decision, which is what any other rule, and a step label, means; the
  tool step of a text action lists that step address too, after the decision, so every emitted
  step has an address;
- ScienceWorld is left out: agentpile has no reader for it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentpile.record import MISSING  # this module is imported only once agentpile is in use

Node = Tuple[Any, Any]  # (step id, call id or None)

_TOOL_AGENTS = {"Computer_terminal"}
_BLOCK_RE = re.compile(r"```(?:bash|python)?\s*(.*?)```", re.DOTALL)

#: reader name -> the dataset id its records carry
DATASETS = {
    "who_and_when": "Kevin355/Who_and_When",
    "tau_bench": "AgentSuite/tau-bench-trajectories",
    "tau2_bench": "AgentSuite/tau2-bench-trajectories",
    "swe_rebench_openhands": "nebius/SWE-rebench-openhands-trajectories",
    "swe_gym_openhands": "SWE-Gym/OpenHands-Sampled-Trajectories",
    "swe_agent_nebius": "nebius/SWE-agent-trajectories",
    "agent_reward_bench": "McGill-NLP/agent-reward-bench",
}

#: reader name -> the revision GRADE's expected outputs were checked against (agentpile 0.3.0 manifests)
REVISIONS = {
    "who_and_when": "59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c",
    "tau_bench": "382e57d1784b55c5155f4ef394ef48f1c747a287",
    "tau2_bench": "07264f1d253a618fc20966670eddc44246212916",
    "swe_rebench_openhands": "35455389ab51bf5e2306bfd436ef72d0f98bf882",
    "swe_gym_openhands": "baf3a4e4bff514d48ddc08a93a2ade5c126212c7",
    "swe_agent_nebius": "68195a1450865274106246d0d0296a1d6807b88e",
    "agent_reward_bench": "b6d17e646009d6cb63d5dd7be78807b680693f61",
}

#: GRADE's loaders build Who&When graphs with the shared-resource node and every other corpus without
SHARED_RESOURCE = {name: name == "who_and_when" for name in DATASETS}


def corpus_rules(reader: str) -> List[Any]:
    """The agentpile rules that carry the dependency semantics of the corpus's GRADE converter."""
    from agentpile import deps as D

    rules = {
        "who_and_when": ["full_context_v1"],
        "tau_bench": ["prior_reads_v1"],
        "tau2_bench": ["value_flow_v1"],
        "swe_rebench_openhands": ["resource_chain_v1"],
        "swe_gym_openhands": ["resource_chain_v1"],
        "swe_agent_nebius": [D.resource_chain(D.shell_commands_v1)],
        "agent_reward_bench": [D.resource_chain(D.web_urls_v1)],
    }
    return rules[reader]


def _ext(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (((obj or {}).get("extra") or {}).get("agentpile")) or {}


def _text(value: Any) -> str:
    """Event content as step text: a string as it is, nothing as '', anything else as JSON."""
    if value is None or value is MISSING:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _message_text(step: Dict[str, Any]) -> str:
    sx = _ext(step)
    if sx.get("message_state") in ("null", "missing"):
        return ""
    return _text(sx["message_raw"] if "message_raw" in sx else step.get("message"))


def _call_text(step: Dict[str, Any], call: Dict[str, Any]) -> str:
    cx = (_ext(step).get("calls") or {}).get(call.get("tool_call_id")) or {}
    if cx.get("arguments_state") == "missing":
        args = ""
    else:
        args = _text(cx["arguments_raw"] if "arguments_raw" in cx else call.get("arguments"))
    return f"{call.get('function_name')}({args})"


def _result_text(step: Dict[str, Any], i: int) -> str:
    results = (step.get("observation") or {}).get("results") or []
    fields = (_ext(step).get("result_fields") or [])
    rf = (fields[i] if i < len(fields) else None) or {}
    if rf.get("content_state") in ("null", "missing"):
        return ""
    return _text(rf["content_raw"] if "content_raw" in rf else results[i].get("content"))


def _address(tid: Any, step: Dict[str, Any], **part: Any) -> Dict[str, Any]:
    return dict({"trajectory": tid, "step": step["step_id"]}, **part)


def _results_of(step: Dict[str, Any], cid: Any) -> List[int]:
    results = (step.get("observation") or {}).get("results") or []
    return [i for i, r in enumerate(results) if r.get("source_call_id") == cid]


def _calls_by_id(step: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    return {c.get("tool_call_id"): c for c in step.get("tool_calls") or []}


def _decision(agent: str, text: str) -> Dict[str, Any]:
    return {"agent": agent, "kind": "decision", "deps": [], "text": text}


def _tool(text: str, agent: str = "env") -> Dict[str, Any]:
    return {"agent": agent, "kind": "tool_call", "deps": [], "text": text}


Projected = Tuple[List[Dict[str, Any]], Dict[Node, int], List[List[Dict[str, Any]]]]


def _assistant_steps(run: Any):
    """Steps that came from an assistant message (a user message promoted to an agent step is not one)."""
    for step in run.steps:
        if step.get("source") == "agent" and _ext(step).get("role", "assistant") == "assistant":
            yield step


def _decision_then_calls(run: Any) -> Projected:
    """OpenHands and tau2: one decision per assistant message, then one tool step per call, at the call."""
    tid = run.id
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    owners: List[List[Dict[str, Any]]] = []
    for step in _assistant_steps(run):
        steps.append(dict(_decision("agent", _message_text(step)), idx=len(steps)))
        owners.append([_address(tid, step)])
        for call in step.get("tool_calls") or []:
            cid = call["tool_call_id"]
            nodes[(step["step_id"], cid)] = len(steps)
            res = _results_of(step, cid)
            text = _call_text(step, call) + " -> " + "\n".join(_result_text(step, i) for i in res)
            steps.append(dict(_tool(text), idx=len(steps)))
            owners.append([_address(tid, step, call=cid)] + [_address(tid, step, result=i) for i in res])
    return steps, nodes, owners


def _tau_bench_steps(run: Any) -> Projected:
    """tau-bench: a decision per user and assistant message, and a tool step at each tool result (message order)."""
    tid = run.id
    model = run.dataset.get("model_path", "model")
    items: List[Tuple[Tuple[int, int, int], Dict[str, Any], Optional[Node], List[Dict[str, Any]]]] = []
    for pos, step in enumerate(run.steps):
        sx = _ext(step)
        ref = sx["source_ref"] if isinstance(sx.get("source_ref"), int) else pos
        source = step.get("source")
        if source == "user" and sx.get("role") in ("tool", "function"):  # a tool output before any agent turn
            items.append(((ref, 0, 0), _tool(_message_text(step)), None, [_address(tid, step)]))
        elif source == "user":
            items.append(((ref, 0, 0), _decision("user", _message_text(step)), None, [_address(tid, step)]))
        elif source == "agent":
            items.append(((ref, 0, 0), _decision(model, _message_text(step)), None, [_address(tid, step)]))
            refs = sx.get("result_refs") or []
            fields = sx.get("result_fields") or []
            calls = _calls_by_id(step)
            for i, res in enumerate((step.get("observation") or {}).get("results") or []):
                at = refs[i] if i < len(refs) and isinstance(refs[i], int) else ref
                cid = res.get("source_call_id")
                call = calls.get(cid)
                name = ((fields[i] if i < len(fields) else None) or {}).get("name")
                head = _call_text(step, call) if call is not None else f"{name}()"
                own = [_address(tid, step, result=i)] + ([_address(tid, step, call=cid)] if cid is not None else [])
                items.append(((at, 1, i), _tool(head + " -> " + _result_text(step, i)),
                              (step["step_id"], cid) if cid is not None else None, own))
    items.sort(key=lambda item: item[0])
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    owners: List[List[Dict[str, Any]]] = []
    for _key, proto, node, own in items:
        if node is not None:
            nodes[node] = len(steps)
        steps.append(dict(proto, idx=len(steps)))
        owners.append(own)
    return steps, nodes, owners


def _who_and_when_steps(run: Any) -> Projected:
    """Who&When: one step per history entry; the speaker is the raw name, else the raw role."""
    tid = run.id
    named = run.source.get("config") == "Algorithm-Generated"  # only these entries carry a name
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    owners: List[List[Dict[str, Any]]] = []
    for pos, step in enumerate(run.steps):
        sx = _ext(step)
        name = sx.get("actor") if named else None
        agent = name or sx.get("role") or "unknown"
        text = str(step.get("message", ""))
        tool = name in _TOOL_AGENTS or text.lower().startswith("exitcode") or text.lower().startswith("code output")
        steps.append({"idx": pos, "agent": agent, "kind": "tool_call" if tool else "decision", "deps": [], "text": text})
        nodes[(step["step_id"], None)] = pos
        owners.append([_address(tid, step)])
    return steps, nodes, owners


def _action_text(step: Dict[str, Any], text: str) -> str:
    """The action a text-action turn executed: its calls as the source wrote them, else the last code block."""
    calls = step.get("tool_calls") or []
    if calls:
        fields = _ext(step).get("calls") or {}
        raw = [(fields.get(c.get("tool_call_id")) or {}).get("arguments_raw") for c in calls]
        return "\n".join(r if isinstance(r, str) else _call_text(step, c) for r, c in zip(raw, calls))
    blocks = _BLOCK_RE.findall(text)
    return blocks[-1].strip() if blocks else _message_text(step)


def _action_turn_steps(run: Any, has_action: Callable[[Dict[str, Any], str], Optional[bool]]) -> Projected:
    """Text-action corpora: a decision per agent turn, and a tool step when the turn holds an action."""
    tid = run.id
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    owners: List[List[Dict[str, Any]]] = []
    for step in run.steps:
        if step.get("source") != "agent":
            continue
        text = step.get("message") if isinstance(step.get("message"), str) else ""
        wants = has_action(step, text)
        if wants is None:
            continue  # the turn emits nothing
        results = list(range(len((step.get("observation") or {}).get("results") or [])))
        steps.append(dict(_decision("agent", _message_text(step)), idx=len(steps)))
        owners.append([_address(tid, step)])
        if wants:
            calls = step.get("tool_calls") or []
            nodes[(step["step_id"], None)] = len(steps)
            for call in calls:
                nodes[(step["step_id"], call["tool_call_id"])] = len(steps)
            steps.append(dict(_tool(_action_text(step, text) + " -> "
                                    + "\n".join(_result_text(step, i) for i in results)), idx=len(steps)))
            # the step address comes after the decision's, so a bare step address still resolves to the decision
            owners.append([_address(tid, step)] + [_address(tid, step, call=c["tool_call_id"]) for c in calls]
                          + [_address(tid, step, result=i) for i in results])
        else:
            owners[-1].extend(_address(tid, step, result=i) for i in results)
    return steps, nodes, owners


def _swe_agent_steps(run: Any) -> Projected:
    def has_action(step: Dict[str, Any], text: str):
        blocks = _BLOCK_RE.findall(text)
        return bool(blocks and blocks[-1].strip() and blocks[-1].strip().splitlines()[0].strip())

    return _action_turn_steps(run, has_action)


def _agent_reward_bench_steps(run: Any) -> Projected:
    return _action_turn_steps(run, lambda step, text: True)


STEPS: Dict[str, Callable[[Any], Projected]] = {
    "who_and_when": _who_and_when_steps,
    "tau_bench": _tau_bench_steps,
    "tau2_bench": _decision_then_calls,
    "swe_rebench_openhands": _decision_then_calls,
    "swe_gym_openhands": _decision_then_calls,
    "swe_agent_nebius": _swe_agent_steps,
    "agent_reward_bench": _agent_reward_bench_steps,
}


def label_failed(reader: str, run: Any) -> Optional[int]:
    """The failure label as GRADE's loaders read it, from the record's outcome; None when GRADE counts none."""
    raw = run.outcome.get("raw") or {}
    if reader == "swe_rebench_openhands":
        return 1 - int(raw["resolved"])
    if reader == "swe_gym_openhands":
        return 0 if raw.get("resolved") else 1
    if reader == "tau_bench":
        return 0 if (raw.get("eval_result") or {}).get("db_match") else 1
    if reader == "tau2_bench":
        return 1 if float((raw or {})["score"]) == 0.0 else 0
    if reader == "who_and_when":
        return (1 if str(raw["is_correct"]).lower() == "false" else 0) if "is_correct" in raw else None
    if reader == "swe_agent_nebius":  # GRADE: target is resolved
        return 0 if run.outcome.get("success") else 1
    if reader == "agent_reward_bench":  # GRADE: the first annotator's success starting with "unsuccess" is a failure
        return 1 if run.outcome.get("success") is False else 0
    raise KeyError(reader)
