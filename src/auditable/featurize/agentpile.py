"""Read agentpile records into the traces and configurations the methods take.

agentpile (https://pypi.org/project/agentpile/) loads public agent-trace datasets, trace files
and captured runs into one record, an ATIF v1.7 trajectory, or a config record for a
configuration that never ran. This module is the seam on auditable's side:

- :func:`from_agentpile` turns trajectory records into :class:`TraceView` objects, which every
  run, step and online method reads;
- :func:`configs_from_agentpile` turns a record into the :class:`AgentConfig` objects the
  config methods read;
- :func:`step_targets` and :func:`run_targets` turn the record labels into ``y``.

Two step conventions exist. ``"generic"`` reads ``run.events()``: one decision per user or
agent message and one tool step per call, carrying its result. Content that a source gives as a
list of parts (dicts with a string ``type``, possibly mixed with bare strings) keeps its text and
writes every other part as ``[<type>]``, dropping that part's other fields; any other list stays
JSON. A ``user`` step is a step the record attributes to the user role, which is not
always text a person wrote (a session's compact summary or a task notification also has that
role). ``"grade"`` reproduces the step lists of GRADE's converters (commit 3839a57) for the
seven corpora they cover, through the ``grade-steps-v1`` projection in
:mod:`auditable.featurize._grade_projection`, on the dataset revisions it was checked on. Dependencies are
never guessed: ``deps`` names the agentpile rules (or ``"recorded"``, ``"full_context"``,
``"grade"``), and the rules that ran are kept in each view's ``meta``.

agentpile is imported only when one of these functions is called (``pip install
"auditable[agentpile]"``).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from auditable.featurize.spec import AgentConfig
from auditable.featurize.trace import TraceView

ADAPTER_ID = "agentpile_v1"
PROJECTION = "grade-steps-v1"
CONVENTIONS = ("generic", "grade")
_DEP_MODES = ("recorded", "full_context", "grade")
_PRE = ("catchbench_pre", "yzhao062/catchbench")

Key = Tuple[Any, ...]


def is_agentpile_run(x: Any) -> bool:
    """True for an agentpile ``Run``, recognized without importing agentpile."""
    return type(x).__module__.startswith("agentpile") and hasattr(x, "events") and hasattr(x, "record")


def _agentpile():
    try:
        import agentpile
    except ImportError as err:
        raise ImportError('reading agentpile records needs agentpile: pip install "auditable[agentpile]"') from err
    return agentpile


def _subagent_steps(run: Any) -> int:
    """ATIF steps in a run's subagent trajectories, nested ones included."""
    return sum(len(sub.steps) + _subagent_steps(sub) for sub in run.subagents)


def _as_runs(runs: Any) -> Tuple[List[Any], bool]:
    if is_agentpile_run(runs):
        return [runs], True
    items = list(runs)
    for item in items:
        if not is_agentpile_run(item):
            raise TypeError(f"expected agentpile Run objects, found {type(item).__name__}")
    return items, False


def _ext(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (((obj or {}).get("extra") or {}).get("agentpile")) or {}


def _key(address: Dict[str, Any]) -> Key:
    """The lookup key of a record address: a call, a result, or a whole step."""
    tid, sid = address.get("trajectory"), address.get("step")
    if "call" in address:
        return ("call", tid, sid, address["call"])
    if "result" in address:
        return ("result", tid, sid, address["result"])
    return ("step", tid, sid)


def _text(value: Any) -> str:
    from auditable.featurize._grade_projection import _text as text

    return text(value)


def _is_parts(value: Any) -> bool:
    """True for a list of content parts as chat APIs send them: every item a dict with a string ``type``
    or a bare string (which some sources mix in as text), and at least one dict. A list of strings alone,
    or a list with any other item, is ordinary JSON."""
    return isinstance(value, list) and any(isinstance(p, dict) for p in value) and all(
        isinstance(p, str) or (isinstance(p, dict) and isinstance(p.get("type"), str)) for p in value)


def _content_text(value: Any, report: Dict[str, Any], kind: str) -> str:
    """Step text of message or result content. A list of content parts keeps its text parts and bare
    strings, in order and one per line, and writes every other part as ``[<type>]`` (its other fields,
    an image's base64 payload or a file's path alike, are not kept); ``report`` counts the parts written
    that way. Any other value reads as :func:`_text` does."""
    if not _is_parts(value):
        return _text(value)
    lines = []
    for part in value:
        if isinstance(part, str):
            lines.append(part)
        elif part["type"] == "text" and isinstance(part.get("text"), str):
            lines.append(part["text"])
        else:
            lines.append(f"[{part['type']}]")
            report[f"{kind}_parts_replaced"] += 1
    return "\n".join(lines)


# -- the two step conventions -----------------------------------------------------------------------

def _generic_steps(run: Any) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], Dict[str, Any]]:
    """One decision per user or agent message, one tool step per call (with its results), in source order."""
    steps: List[Dict[str, Any]] = []
    owners: List[List[Dict[str, Any]]] = []
    tool_of_call: Dict[Key, int] = {}
    report: Dict[str, Any] = {"system_messages_skipped": 0, "results_without_call": 0,
                              "message_parts_replaced": 0, "result_parts_replaced": 0}
    for ev in run.events():
        if ev.kind == "message":
            if ev.role == "system":
                report["system_messages_skipped"] += 1
                continue
            steps.append({"idx": len(steps), "agent": ev.actor or ev.role, "kind": "decision", "deps": [],
                          "text": _content_text(ev.content, report, "message")})
            owners.append([dict(ev.address)])
        elif ev.kind == "call":
            tool_of_call[_key(ev.address)] = len(steps)
            steps.append({"idx": len(steps), "agent": "env", "kind": "tool_call", "deps": [], "tool": ev.name,
                          "text": f"{ev.name}({_text(ev.content)})"})
            owners.append([dict(ev.address)])
        else:  # a result
            call = dict(ev.address, call=ev.call_id)
            call.pop("result", None)
            pos = tool_of_call.get(_key(call)) if ev.call_id is not None else None
            if pos is None:
                report["results_without_call"] += 1
                pos = len(steps)
                steps.append({"idx": pos, "agent": "env", "kind": "tool_call", "deps": [], "tool": ev.name,
                              "text": f"{ev.name}()"})
                owners.append([])
            steps[pos]["text"] += " -> " + _content_text(ev.content, report, "result")
            owners[pos].append(dict(ev.address))
    report.update(_coverage(run, owners))
    return steps, owners, report


def _coverage(run: Any, owners: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """How the run's own ATIF steps map into the view: every source step is either mapped (some view step
    holds one of its addresses) or unmapped, counted by its ``source`` role. One source step can give
    several view steps (a message and its calls), so the view's step count is reported apart."""
    mapped = {(a.get("trajectory"), a.get("step")) for addresses in owners for a in addresses}
    unmapped: Dict[str, int] = {}
    n_mapped = 0
    for step in run.steps:
        if (run.id, step.get("step_id")) in mapped:
            n_mapped += 1
        else:
            role = str(step.get("source"))
            unmapped[role] = unmapped.get(role, 0) + 1
    return {"source_steps": len(run.steps), "source_steps_mapped": n_mapped,
            "source_steps_unmapped": len(run.steps) - n_mapped, "unmapped_by_source": unmapped,
            "view_steps": len(owners)}


def _grade_steps(run: Any):
    from auditable.featurize import _grade_projection as G

    reader = (run.source or {}).get("reader")
    if reader not in G.STEPS:
        raise ValueError(f'convention="grade" covers the readers {sorted(G.STEPS)}; this run comes from '
                         f"reader {reader!r}")
    dataset = run.source.get("dataset")
    if dataset != G.DATASETS[reader]:
        raise ValueError(f"source.dataset of a {reader} run should be {G.DATASETS[reader]!r}, got {dataset!r}")
    if reader == "tau_bench" and not isinstance(run.dataset.get("model_path", "model"), str):
        raise ValueError("run.dataset['model_path'] of a tau_bench run should be a string")
    revision = run.source.get("revision")
    if revision != G.REVISIONS[reader]:
        raise ValueError(f"run {run.id!r} comes from revision {revision!r} of {reader}, but the {PROJECTION} "
                         f"projection was checked against GRADE only on revision {G.REVISIONS[reader]!r}, so its "
                         f"steps on this one are unverified; read it with convention=\"generic\" and a generic "
                         f"dependency setting (deps=\"full_context\" or a list of agentpile rules) instead")
    steps, nodes, owners = G.STEPS[reader](run)
    return reader, steps, nodes, owners, G.SHARED_RESOURCE[reader]


# -- dependencies -------------------------------------------------------------------------------------

def _rule_name(rule: Any) -> str:
    """The versioned name agentpile resolves a rule to: a registered name, its latest version, or a callable's name."""
    if callable(rule):
        return getattr(rule, "__name__", repr(rule))
    from agentpile.deps import RULES

    if rule in RULES:
        return rule
    versions = [r for r in RULES if r.rsplit("_v", 1)[0] == rule]
    if not versions:
        raise KeyError(f"unknown dependency rule {rule!r} (known: {', '.join(RULES)})")
    return max(versions, key=lambda r: int(r.rsplit("_v", 1)[1]))


def _dep_entries(run: Any, deps: Any, reader: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(entries, rules)``: the dependency entries and, per rule asked for, what it resolved to and produced."""
    ap = _agentpile()

    if deps == "recorded":
        entries = [d for step in run.steps for d in (_ext(step).get("deps") or [])]
        seen = sorted({(d.get("rule"), d.get("rule_version"), d.get("extractor")) for d in entries}, key=str)
        return entries, [{"requested": "recorded", "rule": r, "rule_version": v, "extractor": e} for r, v, e in seen]
    if deps == "grade":
        from auditable.featurize._grade_projection import corpus_rules

        rules = corpus_rules(reader)
    else:
        rules = list(deps)
    entries: List[Dict[str, Any]] = []
    record: List[Dict[str, Any]] = []
    for rule in rules:
        got = ap.derive_deps(run, rules=[rule])
        entries.extend(got)
        extractors = sorted({d.get("extractor") for d in got if d.get("extractor")})
        versions = sorted({str(d.get("rule_version")) for d in got})
        record.append({"requested": rule if isinstance(rule, str) else _rule_name(rule), "rule": _rule_name(rule),
                       "rule_version": versions[0] if len(versions) == 1 else (versions or None),
                       "extractor": extractors[0] if len(extractors) == 1 else (extractors or None),
                       "edges": len(got)})
    return entries, record


def _apply(steps: List[Dict[str, Any]], entries: Sequence[Dict[str, Any]], resolve) -> Dict[str, int]:
    found: Dict[int, set] = {}
    report = {"dep_entries": len(entries), "dropped_unmapped": 0, "dropped_not_earlier": 0}
    for entry in entries:
        src, dst = resolve(entry.get("src") or {}), resolve(entry.get("dst") or {})
        if src is None or dst is None:
            report["dropped_unmapped"] += 1
        elif src >= dst:
            report["dropped_not_earlier"] += 1
        else:
            found.setdefault(dst, set()).add(src)
    for pos, srcs in found.items():
        steps[pos]["deps"] = sorted(steps[i]["idx"] for i in srcs)
    report["edges"] = sum(len(s["deps"]) for s in steps)
    return report


def _lookup(owners: Sequence[Sequence[Dict[str, Any]]]):
    """``address -> step position`` over the per-step owner lists; a bare step address falls back to the
    first step that owns any part of that step."""
    table: Dict[Key, int] = {}
    first: Dict[Key, int] = {}
    for pos, addresses in enumerate(owners):
        for address in addresses:
            table.setdefault(_key(address), pos)
            first.setdefault(("step", address.get("trajectory"), address.get("step")), pos)

    def resolve(address: Dict[str, Any]) -> Optional[int]:
        key = _key(address)
        pos = table.get(key)
        if pos is None and key[0] == "step":
            pos = first.get(key)
        return pos

    return resolve


# -- labels -------------------------------------------------------------------------------------------

def _labels(run: Any, steps: List[Dict[str, Any]], resolve) -> List[Dict[str, Any]]:
    out = []
    for label in run.labels:
        target = label.get("target") or {}
        pos = None
        if target.get("kind") in ("step", "call", "result"):
            pos = resolve(target.get("address") or {})
        ref = label.get("source_ref") or {}
        out.append(dict(label, position=pos, idx=None if pos is None else steps[pos]["idx"],
                        in_range=ref.get("in_range") if isinstance(ref, dict) else None))
    return out


# -- public functions ---------------------------------------------------------------------------------

def from_agentpile(runs: Any, *, deps: Union[str, Iterable[Any]], convention: str = "generic",
                   subagents: str = "error") -> Union[TraceView, List[TraceView]]:
    """Traces of agentpile trajectory records: one :class:`TraceView` for a ``Run``, a list for an iterable.

    Parameters
    ----------
    runs : an agentpile ``Run`` or an iterable of them (``ap.load(...)``).
    deps : required; there is no default, because a method that reads dependencies returns zeros
        on a trace that has none. A list of agentpile rules (names such as ``"tool_call_id_v1"`` or
        rule objects), passed to ``agentpile.derive_deps``; ``"recorded"`` for the entries the record
        already holds (from the source or ``run.with_deps``); ``"full_context"`` for every earlier
        step; ``"grade"`` for the rules of the corpus's GRADE converter (with ``convention="grade"``).
    convention : ``"generic"`` (default) or ``"grade"``; see the module docstring.
    subagents : ``"error"`` (default) refuses a run with subagent trajectories; ``"skip"`` reads
        the top-level trajectory and counts what it skipped.

    Each view's ``meta`` records the adapter, agentpile version, convention, the dependency rules
    asked for and what they produced, the record address of every step (``addresses``, by
    position), the labels with the step they point at, and the run's source and outcome. With
    ``convention="generic"``, ``meta["report"]`` also says how the run's ATIF steps map into the
    view (``source_steps``, ``source_steps_mapped``, ``source_steps_unmapped`` by role in
    ``unmapped_by_source``, and ``view_steps``, which differs because one message can give a
    decision and its calls) and how many non-text content parts became ``[<type>]`` placeholders.
    With ``convention="grade"``, a run from a dataset revision other than the one the projection
    was checked on raises ``ValueError``.
    """
    ap = _agentpile()

    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    if subagents not in ("error", "skip"):
        raise ValueError(f'subagents must be "error" or "skip", got {subagents!r}')
    if deps is None:
        raise ValueError('deps is required: a list of agentpile rules, "recorded", "full_context" or "grade"')
    if isinstance(deps, str):
        if deps not in _DEP_MODES:
            deps = [deps]  # one rule name
    else:
        deps = list(deps)  # read once, for every run
    if deps == "grade" and convention != "grade":
        raise ValueError('deps="grade" needs convention="grade"')
    items, single = _as_runs(runs)
    views: List[TraceView] = []
    any_entries = False
    for run in items:
        if run.kind != "trajectory":
            raise ValueError(f"run {run.id!r} is a config record with no steps; use configs_from_agentpile")
        meta: Dict[str, Any] = {"adapter": ADAPTER_ID, "agentpile_version": ap.__version__, "run_id": run.id,
                                "source": dict(run.source), "outcome": dict(run.outcome), "convention": convention}
        n_sub = len(run.subagents)
        if n_sub and subagents == "error":
            raise ValueError(f"run {run.id!r} has {n_sub} subagent trajectories, which auditable does not flatten "
                             f"into one trace. Convert each subagent as a trace of its own with "
                             f"from_agentpile(run.subagents, ...), or pass subagents=\"skip\" to read the top-level "
                             f"trajectory only (the subagents' steps are then left out)")
        meta["subagents_skipped"] = n_sub
        meta["subagent_steps_skipped"] = _subagent_steps(run)
        reader = None
        if convention == "grade":
            reader, steps, nodes, owners, shared = _grade_steps(run)
            meta.update(projection=PROJECTION, revision_tested=True)
            report: Dict[str, Any] = {}
            from auditable.featurize import _grade_projection as G

            meta["label_failed"] = G.label_failed(reader, run)
        else:
            steps, owners, report = _generic_steps(run)
            nodes, shared = None, True
        resolve = _lookup(owners)
        if deps == "full_context":
            for pos, step in enumerate(steps):
                step["deps"] = [s["idx"] for s in steps[:pos]]
            meta["dep_rules"] = [{"requested": "full_context", "rule": "full_context", "rule_version": None,
                                  "extractor": None}]
            report.update(edges=sum(len(s["deps"]) for s in steps))
        else:
            entries, meta["dep_rules"] = _dep_entries(run, deps, reader)
            any_entries = any_entries or bool(entries)
            if deps == "grade":  # GRADE's own rules resolve through the converter's endpoint table
                def by_node(address, nodes=nodes):
                    return nodes.get((address.get("step"), address.get("call")))

                report.update(_apply(steps, entries, by_node))
            else:
                report.update(_apply(steps, entries, resolve))
        meta["report"] = report
        meta["addresses"] = [list(a) for a in owners]
        meta["labels"] = _labels(run, steps, resolve)
        texts = [s.pop("text") for s in steps]
        views.append(TraceView(steps=steps, texts=texts, source="agentpile", meta=meta, shared_resource=shared))
    if items and deps != "full_context":
        edges = sum(v.meta["report"].get("edges", 0) for v in views)
        if deps == "recorded" and not any_entries:
            warnings.warn("none of the runs holds a recorded dependency entry, so whether any rule ran is unknown; "
                          "derive them with run.with_deps(rules) or pass deps=[...]", stacklevel=2)
        elif deps != "recorded" and deps != [] and edges == 0:  # an empty rule list asks for no edges
            warnings.warn(f"the dependency rules produced no edge on any of the {len(items)} runs; methods that read "
                          f"dependencies will score every step alike", stacklevel=2)
    return views[0] if single else views


def step_targets(traces: Sequence[TraceView], name: str = "mistake_step"):
    """The ``idx`` of each trace's decisive-mistake step, from its step label ``name``, as an int array.

    -1 marks a trace with no such label, a label the source marks out of range, or one that points
    at no step of the trace. The output has one entry per trace, in input order.
    """
    import numpy as np

    out = []
    for view in traces:
        value = -1
        for label in (view.meta or {}).get("labels") or []:
            if label.get("name") == name and label.get("value") and label.get("in_range") is not False \
                    and label.get("idx") is not None:
                value = int(label["idx"])
                break
        out.append(value)
    return np.array(out, dtype=int)


def run_targets(traces: Sequence[TraceView], *, label: Optional[str] = None, failed=None):
    """Failure labels as an int array: 1 failed, 0 succeeded, -1 unknown, one entry per trace.

    By default the label is the record's ``outcome["success"]``, or, for a trace read with
    ``convention="grade"``, the label GRADE's loaders read. ``label`` names a run-level label
    instead and needs ``failed``, a function of its value that says whether the run failed.
    """
    import numpy as np

    if label is not None and failed is None:
        raise TypeError("a run label needs failed=, a function of the label value that returns True for a failure")
    out = []
    for view in traces:
        meta = view.meta or {}
        if label is not None:
            hits = [lb for lb in meta.get("labels") or [] if lb.get("name") == label
                    and (lb.get("target") or {}).get("kind") == "run"]
            out.append(int(bool(failed(hits[0]["value"]))) if hits else -1)
        elif "label_failed" in meta:
            out.append(-1 if meta["label_failed"] is None else int(meta["label_failed"]))
        else:
            success = (meta.get("outcome") or {}).get("success")
            out.append(-1 if not isinstance(success, bool) else int(not success))
    return np.array(out, dtype=int)


def configs_from_agentpile(run: Any, *, task: Optional[str] = None, allow_no_task: bool = False) -> List[AgentConfig]:
    """The :class:`AgentConfig` of each agent in an agentpile record, as a list.

    A record with no agent roster gives one configuration with all its tools. With a roster, each
    agent gets the tools its ``tools`` list names; an empty list stays empty, and an agent whose
    entry records no tool list gets all tools. The task text is ``task``, else ``run.task["text"]``,
    else the agent's instructions, and goes into ``task_or_role_spec``. A CatchBench PRE row (reader
    ``catchbench_pre``) carries its spec tokens instead, which count as evidence even when empty.
    With neither text nor tokens the call raises ``ValueError`` unless ``allow_no_task=True``.
    """
    if not is_agentpile_run(run):
        raise TypeError(f"expected an agentpile Run, got {type(run).__name__}")
    tools = run.tools
    source = run.source or {}
    pre = (source.get("reader"), source.get("dataset")) == _PRE
    tokens = overrides = None
    if pre:
        ds = run.dataset
        tokens, overrides = ds.get("spec_tokens"), ds.get("spec_token_overrides") or {}
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError(f"run.dataset['spec_tokens'] of CatchBench PRE row {run.id!r} should be a list of strings")
        if not isinstance(overrides, dict):
            raise ValueError(f"run.dataset['spec_token_overrides'] of CatchBench PRE row {run.id!r} should be a dict")
    agents = run.agents
    groups: List[Tuple[Optional[Dict[str, Any]], List[str]]] = []
    if not agents:
        groups.append((None, list(tools)))
    for agent in agents:
        ids = agent.get("tools")
        if ids is None:
            ids = list(tools)
        unknown = [i for i in ids if i not in tools]
        if unknown:
            raise ValueError(f"agent {agent.get('id')!r} of run {run.id!r} names tools the record does not define: {unknown}")
        groups.append((agent, list(ids)))
    out = []
    for agent, ids in groups:
        caps = [{"name": tools[i].get("name") or i, "type": tools[i].get("type") or "",
                 "permission_level": tools[i].get("permission") or "unknown",
                 "permission_from": tools[i].get("permission_from")} for i in ids]
        text = task or (run.task or {}).get("text") or (agent or {}).get("instructions") or None  # "" is no text
        if text is None and tokens is None and not allow_no_task:
            raise ValueError(f"run {run.id!r} has no task text and no spec tokens, so the config methods would judge "
                             f"its tools against an empty task; pass task=..., or allow_no_task=True to accept that")
        instance = run.id if len(groups) == 1 else f"{run.id}:{(agent or {}).get('id')}"
        if text is not None:
            out.append(AgentConfig(declared_capabilities=caps, task_or_role_spec=text, instance_id=instance))
        else:
            out.append(AgentConfig(declared_capabilities=caps, spec_tokens=tokens or [],
                                   spec_token_overrides=overrides, instance_id=instance))
    return out


__all__ = ["ADAPTER_ID", "CONVENTIONS", "PROJECTION", "configs_from_agentpile", "from_agentpile",
           "is_agentpile_run", "run_targets", "step_targets"]
