"""LangChain callback capture into auditable's typed session graph.

Attach :class:`LangChainCallbackHandler` to a real ``langchain-core`` invocation,
then use the handler as both source and adapter::

    handler = LangChainCallbackHandler()
    result = chain.invoke(value, config={"callbacks": [handler]})
    report = analyze_run(handler, adapter=handler)

The handler pairs chain/runnable, tool, and LLM start/end callbacks.  Starts fix
the run order and input digest; ends add a capture-safe output snapshot.  Chain
and LLM runs become ``decision`` steps and tools become ``tool_call`` steps.  A
callback ``parent_run_id`` or RunnableSequence tag supplies the execution layer.

Dependency grades deliberately say no more than the callback payload supports:

* ``OBSERVED`` means a declared ``seq:step:N`` handoff had an exact normalized
  prior-output/current-input digest match.
* ``DECLARED`` means the RunnableSequence topology declares the handoff but the
  callback representations did not yield that exact value match.
* ``INFERRED`` means two otherwise unstructured, sequential sibling callbacks had
  matching payload digests; equality and callback order alone do not prove flow.

This surface sees State A -- model/runnable reasoning and tool-call flow exposed
by callbacks.  It does *not* see State B -- the version or contents of external
databases, files, APIs, or other dependencies behind a tool.  A tool result is
therefore captured as an output, never fabricated into an OBSERVED external-state
edge.  Even an OBSERVED edge here is a callback-level value handoff, not a precise
causal claim about the computation that produced the value.

Scored support is scoped to flat ``RunnableSequence`` topologies.  A nested
``RunnableParallel`` branch is captured, but its fan-in to the downstream consumer
is not modeled as dependency edges, so downstream reach through a parallel
container understates the true value flow.  Treat a run containing parallel
containers as capture plus inspection, not as ranking evidence.

One handler captures the latest top-level invocation.  Reusing it sequentially
resets capture when the next root starts; concurrent top-level invocations through
one handler are rejected by :meth:`steps` rather than merged into a false graph.
Use one handler per concurrent invocation.  ``langchain-core`` is an optional
extra, so importing this module without it raises a focused installation message.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..graph.session import DependencyEdge, Grade, ResourceRef, Step

try:  # the import doubles as the optional-extra presence check
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "auditable.integrations.langchain requires langchain-core: "
        "pip install auditable[langchain]"
    ) from exc

__all__ = ["LangChainCallbackHandler"]

_NS = "langchain_callback_value"
_SEQ_TAG = re.compile(r"^seq:step:(\d+)$")
_UNSET = object()


def _safe_copy(value: Any) -> Any:
    """Copy a callback snapshot without ever disrupting the observed run."""
    try:
        return deepcopy(value)
    except Exception:
        return value


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _normalize_payload(value: Any, _seen: Optional[set] = None, _depth: int = 0) -> Any:
    """Return a deterministic, JSON-shaped view of a callback payload.

    LangChain intentionally exposes different wrapper types on adjacent callbacks:
    a prompt runnable ends with ``StringPromptValue``, for example, while the LLM
    starts with a plain prompt string.  The small LangChain-aware cases below lower
    those wrappers to the value actually handed across the seam.  The function is
    defensive because observability must not make an otherwise valid run fail.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (UUID, date, datetime, time)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    if isinstance(value, Enum):
        return _normalize_payload(value.value, _seen, _depth + 1)
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if _depth >= 30:
        return {"__type__": _type_name(value), "__truncated__": True}

    if _seen is None:
        _seen = set()
    identity = id(value)
    if identity in _seen:
        return {"__type__": _type_name(value), "__cycle__": True}
    _seen.add(identity)
    try:
        module = type(value).__module__
        class_name = type(value).__name__

        # PromptValue wrappers and their callback-facing LLM/chat equivalents.
        if module.startswith("langchain_core.prompt_values"):
            try:
                if class_name == "StringPromptValue":
                    return _normalize_payload(value.to_string(), _seen, _depth + 1)
                return _normalize_payload(value.to_messages(), _seen, _depth + 1)
            except Exception:
                pass

        # Keep only semantic message fields. Provider response/usage metadata is not
        # part of the value a downstream runnable reasons over and can differ across
        # the LLM-end and next-chain-start callback views of the same message.
        if module.startswith("langchain_core.messages"):
            message: Dict[str, Any] = {
                "type": getattr(value, "type", class_name),
                "content": _normalize_payload(
                    getattr(value, "content", None), _seen, _depth + 1
                ),
            }
            for attr in (
                "name",
                "additional_kwargs",
                "tool_calls",
                "invalid_tool_calls",
                "tool_call_id",
                "artifact",
                "status",
            ):
                item = getattr(value, attr, None)
                if item not in (None, "", [], {}):
                    message[attr] = _normalize_payload(item, _seen, _depth + 1)
            return message

        if isinstance(value, Mapping):
            if all(isinstance(key, str) for key in value):
                return {
                    key: _normalize_payload(item, _seen, _depth + 1)
                    for key, item in value.items()
                }
            # Preserve non-string key types instead of collapsing (for example)
            # integer key 1 and string key "1" to the same digest.
            items = [
                [
                    _normalize_payload(key, _seen, _depth + 1),
                    _normalize_payload(item, _seen, _depth + 1),
                ]
                for key, item in value.items()
            ]
            items.sort(key=lambda item: json.dumps(item[0], sort_keys=True, default=str))
            return {"__mapping__": items}
        if isinstance(value, (list, tuple)):
            return [_normalize_payload(item, _seen, _depth + 1) for item in value]
        if isinstance(value, (set, frozenset)):
            items = [_normalize_payload(item, _seen, _depth + 1) for item in value]
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            try:
                return _normalize_payload(dataclasses.asdict(value), _seen, _depth + 1)
            except Exception:
                pass
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return _normalize_payload(model_dump(mode="python"), _seen, _depth + 1)
            except Exception:
                pass

        try:
            rendered = repr(value)
        except Exception:
            rendered = "<unrepresentable>"
        return {"__type__": _type_name(value), "repr": rendered}
    except Exception:
        # This is intentionally broad: malformed third-party callback values should
        # reduce capture fidelity, never change the application outcome.
        return {"__type__": _type_name(value), "repr": "<normalization failed>"}
    finally:
        _seen.discard(identity)


def _digest(normalized: Any) -> str:
    try:
        payload = json.dumps(
            normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    except Exception:
        payload = repr(normalized)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _error_fingerprint(error: BaseException) -> Tuple[str, str]:
    """An exception type and content digest, never the raw message.

    Validation errors, provider exceptions, and tool errors routinely embed
    rejected values or response bodies, so the raw text stays out of the graph:
    the step retains only the exception type and a digest over type plus message.
    """
    error_type = _type_name(error)
    try:
        message = str(error)
    except BaseException:
        message = "<unprintable>"
    return error_type, _digest({"type": error_type, "message": message})


def _run_id(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _serialized_name(serialized: Any) -> str:
    if not isinstance(serialized, Mapping):
        return ""
    name = serialized.get("name")
    if name:
        return str(name)
    identifier = serialized.get("id")
    if isinstance(identifier, Sequence) and not isinstance(identifier, (str, bytes)):
        return str(identifier[-1]) if identifier else ""
    return str(identifier or "")


def _name(run_type: str, serialized: Any, kwargs: Mapping) -> str:
    return str(kwargs.get("name") or _serialized_name(serialized) or run_type)


def _model_id(serialized: Any, metadata: Any, kwargs: Mapping, fallback: str) -> str:
    candidates: List[Any] = []
    if isinstance(metadata, Mapping):
        candidates += [
            metadata.get("ls_model_name"),
            metadata.get("model_id"),
            metadata.get("model_name"),
        ]
    params = kwargs.get("invocation_params")
    if isinstance(params, Mapping):
        candidates += [params.get("model"), params.get("model_name")]
    if isinstance(serialized, Mapping):
        serialized_kwargs = serialized.get("kwargs")
        if isinstance(serialized_kwargs, Mapping):
            candidates += [
                serialized_kwargs.get("model"),
                serialized_kwargs.get("model_name"),
            ]
        candidates += [serialized.get("name"), _serialized_name(serialized)]
    candidates.append(fallback)
    return next((str(item) for item in candidates if item), fallback)


def _sequence_step(tags: Any) -> Optional[int]:
    for tag in tags or ():
        match = _SEQ_TAG.match(str(tag))
        if match:
            return int(match.group(1))
    return None


def _parallel_tag(tags: Tuple[str, ...]) -> bool:
    return any(tag.startswith("map:key:") for tag in tags)


def _llm_output(response: Any) -> Any:
    """Extract the value ``invoke`` hands downstream from an ``LLMResult``."""
    generations = getattr(response, "generations", None)
    if not isinstance(generations, Sequence):
        return response
    outputs: List[Any] = []
    for group in generations:
        if not group:
            outputs.append(None)
            continue
        generation = group[0]
        message = getattr(generation, "message", None)
        outputs.append(message if message is not None else getattr(generation, "text", generation))
    return outputs[0] if len(outputs) == 1 else outputs


@dataclass
class _PendingDependency:
    src_idx: int
    grade: Grade
    digest: Optional[str]
    evidence: Dict[str, Any]


@dataclass
class _Capture:
    idx: int
    start_event: int
    run_id: str
    parent_run_id: Optional[str]
    run_type: str
    name: str
    kind: str
    tags: Tuple[str, ...]
    sequence_step: Optional[int]
    inputs_digest: str
    model_id: str = ""
    tool_name: str = ""
    exec_pred_idx: Optional[int] = None
    dependency: Optional[_PendingDependency] = None
    outputs: Any = _UNSET
    outputs_digest: Optional[str] = None
    status: str = "running"
    error_type: str = ""
    error_digest: str = ""
    end_event: Optional[int] = None

    @property
    def complete(self) -> bool:
        return self.outputs is not _UNSET


class LangChainCallbackHandler(BaseCallbackHandler):
    """Capture one LangChain invocation as typed :class:`~auditable.graph.Step`s.

    Pass an instance through ``config={"callbacks": [handler]}``.  After the
    invocation, :meth:`steps` returns the ordered snapshot.  :meth:`to_steps` is
    the adapter-protocol spelling, so ``analyze_run(handler, adapter=handler)``
    works directly.  Completed steps may also be fed to ``LiveSession.observe``.

    Callback handlers run in application code, so all payload normalization is
    fail-soft and the handler does not raise into the LangChain run.  The one
    fail-closed case is graph extraction after concurrent top-level uses of the
    same handler: :meth:`steps` raises instead of presenting mixed traces.
    """

    name = "langchain_callback"
    version = "v1"
    run_inline = True
    raise_error = False

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._captures: Dict[str, _Capture] = {}
        self._next_idx = 0
        self._next_event = 0
        self._active_roots: set = set()
        self._invalid_reason: Optional[str] = None

    @property
    def id(self) -> str:
        """Stable adapter id recorded on dependency-edge evidence."""
        return f"{self.name}_{self.version}"

    def reset(self) -> None:
        """Clear a finished capture explicitly.

        Sequential top-level invocation reuse resets automatically.  An explicit
        reset during an active root would make later end callbacks ambiguous and is
        therefore rejected.
        """
        with self._lock:
            if self._active_roots:
                raise RuntimeError("cannot reset LangChain capture during an active run")
            self._clear_unlocked()

    def _clear_unlocked(self) -> None:
        self._captures = {}
        self._next_idx = 0
        self._next_event = 0
        self._active_roots = set()
        self._invalid_reason = None

    def _event_unlocked(self) -> int:
        event = self._next_event
        self._next_event += 1
        return event

    def _sequence_predecessor_unlocked(
        self, parent_run_id: Optional[str], sequence_step: Optional[int]
    ) -> Optional[_Capture]:
        if sequence_step is None or sequence_step <= 1:
            return None
        candidates = [
            capture
            for capture in self._captures.values()
            if capture.parent_run_id == parent_run_id
            and capture.sequence_step == sequence_step - 1
            and capture.complete
        ]
        return max(candidates, key=lambda item: item.end_event or -1) if candidates else None

    def _matching_sibling_unlocked(
        self, parent_run_id: Optional[str], inputs_digest: str
    ) -> Optional[_Capture]:
        candidates = [
            capture
            for capture in self._captures.values()
            if capture.parent_run_id == parent_run_id
            and capture.complete
            and capture.outputs_digest == inputs_digest
        ]
        return max(candidates, key=lambda item: item.end_event or -1) if candidates else None

    def _start(
        self,
        run_type: str,
        kind: str,
        serialized: Any,
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any,
        tags: Any,
        metadata: Any,
        kwargs: Mapping,
    ) -> None:
        rid = _run_id(run_id)
        if rid is None:
            return
        parent_id = _run_id(parent_run_id)
        tag_tuple = tuple(str(tag) for tag in (tags or ()))
        normalized_inputs = _normalize_payload(inputs)
        inputs_digest = _digest(normalized_inputs)

        with self._lock:
            # Some callback managers may defensively retry an event. Do not create a
            # second step for the same framework run id.
            if rid in self._captures:
                return
            if parent_id is None:
                if self._active_roots:
                    self._invalid_reason = (
                        "one LangChainCallbackHandler observed concurrent top-level runs; "
                        "use one handler per concurrent invocation"
                    )
                elif self._captures:
                    self._clear_unlocked()  # latest sequential top-level run wins
                self._active_roots.add(rid)

            event = self._event_unlocked()
            sequence_step = _sequence_step(tag_tuple)
            predecessor = self._sequence_predecessor_unlocked(parent_id, sequence_step)
            dependency: Optional[_PendingDependency] = None
            exec_pred_idx: Optional[int] = None

            if predecessor is not None:
                exec_pred_idx = predecessor.idx
                if predecessor.outputs_digest == inputs_digest:
                    dependency = _PendingDependency(
                        src_idx=predecessor.idx,
                        grade=Grade.OBSERVED,
                        digest=inputs_digest,
                        evidence={
                            "observed": True,
                            "relation": "callback_output_to_input",
                            "granularity": "normalized_payload_digest",
                            "topology": "runnable_sequence",
                            "inputs_digest": inputs_digest,
                            "source_run_id": predecessor.run_id,
                            "target_run_id": rid,
                            "adapter": self.id,
                            "note": (
                                "exact normalized output/input match on a callback-declared "
                                "RunnableSequence handoff; State A only"
                            ),
                        },
                    )
                else:
                    dependency = _PendingDependency(
                        src_idx=predecessor.idx,
                        grade=Grade.DECLARED,
                        digest=None,
                        evidence={
                            "declared": True,
                            "observed": False,
                            "relation": "runnable_sequence_adjacency",
                            "source_run_id": predecessor.run_id,
                            "target_run_id": rid,
                            "adapter": self.id,
                            "note": (
                                "seq:step tags declare the handoff, but callback payloads "
                                "did not expose an exact normalized value match"
                            ),
                        },
                    )
            else:
                parent = self._captures.get(parent_id or "")
                if parent is not None:
                    exec_pred_idx = parent.idx

                # Payload equality without sequence topology is weaker evidence.
                # Parallel-map siblings are explicitly excluded: callback completion
                # order must not turn parallel branches into fake dependencies.
                if not _parallel_tag(tag_tuple):
                    match = self._matching_sibling_unlocked(parent_id, inputs_digest)
                    if match is not None:
                        dependency = _PendingDependency(
                            src_idx=match.idx,
                            grade=Grade.INFERRED,
                            digest=None,
                            evidence={
                                "inferred": True,
                                "observed": False,
                                "relation": "callback_payload_equality",
                                "inputs_digest": inputs_digest,
                                "source_run_id": match.run_id,
                                "target_run_id": rid,
                                "adapter": self.id,
                                "note": (
                                    "matching normalized payloads and callback order without "
                                    "a declared RunnableSequence handoff; not observed flow"
                                ),
                            },
                        )

            run_name = _name(run_type, serialized, kwargs)
            model_id = (
                _model_id(serialized, metadata, kwargs, run_name)
                if run_type in ("llm", "chat_model")
                else ""
            )
            # A configured run_name is useful as the step/agent label, but the
            # typed `tool` facet should retain the tool's actual serialized name.
            tool_name = ""
            if run_type == "tool":
                tool_name = _serialized_name(serialized) or run_name
            capture = _Capture(
                idx=self._next_idx,
                start_event=event,
                run_id=rid,
                parent_run_id=parent_id,
                run_type=run_type,
                name=run_name,
                kind=kind,
                tags=tag_tuple,
                sequence_step=sequence_step,
                inputs_digest=inputs_digest,
                model_id=model_id,
                tool_name=tool_name,
                exec_pred_idx=exec_pred_idx,
                dependency=dependency,
            )
            self._next_idx += 1
            self._captures[rid] = capture

    def _end(self, run_id: Any, output: Any, *, error: Optional[BaseException] = None) -> None:
        rid = _run_id(run_id)
        if rid is None:
            return
        normalized = _normalize_payload(output)
        with self._lock:
            capture = self._captures.get(rid)
            if capture is None or capture.complete:
                return
            capture.outputs = normalized
            capture.outputs_digest = _digest(normalized) if error is None else None
            capture.status = "error" if error is not None else "ok"
            if error is not None:
                capture.error_type, capture.error_digest = _error_fingerprint(error)
            capture.end_event = self._event_unlocked()
            if capture.parent_run_id is None:
                self._active_roots.discard(rid)

    # -- BaseCallbackHandler surface -----------------------------------------

    def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            "chain",
            "decision",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    def on_chain_end(
        self, outputs: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self._end(run_id, outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._end(run_id, None, error=error)

    def on_tool_start(
        self,
        serialized: Any,
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        logical_inputs = inputs if inputs is not None else input_str
        self._start(
            "tool",
            "tool_call",
            serialized,
            logical_inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    def on_tool_end(
        self, output: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self._end(run_id, output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._end(run_id, None, error=error)

    def on_llm_start(
        self,
        serialized: Any,
        prompts: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        logical_inputs = prompts[0] if isinstance(prompts, list) and len(prompts) == 1 else prompts
        self._start(
            "llm",
            "decision",
            serialized,
            logical_inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        logical_inputs = (
            messages[0] if isinstance(messages, list) and len(messages) == 1 else messages
        )
        self._start(
            "chat_model",
            "decision",
            serialized,
            logical_inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    def on_llm_end(
        self, response: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self._end(run_id, _llm_output(response))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._end(run_id, None, error=error)

    # -- auditable Adapter surface ------------------------------------------

    def steps(self) -> List[Step]:
        """Return completed callbacks as ordered, typed, independent snapshots."""
        with self._lock:
            if self._invalid_reason:
                raise RuntimeError(self._invalid_reason)
            captures = sorted(
                (capture for capture in self._captures.values() if capture.complete),
                key=lambda capture: capture.idx,
            )
            completed_idxs = {capture.idx for capture in captures}
            result: List[Step] = []
            for capture in captures:
                attrs: Dict[str, Any] = {
                    "langchain_run_type": capture.run_type,
                    "langchain_run_id": capture.run_id,
                    "inputs_digest": capture.inputs_digest,
                    "outputs": _safe_copy(capture.outputs),
                    "status": capture.status,
                }
                if capture.parent_run_id is not None:
                    attrs["parent_run_id"] = capture.parent_run_id
                if capture.tags:
                    attrs["tags"] = list(capture.tags)
                if capture.outputs_digest is not None:
                    attrs["outputs_digest"] = capture.outputs_digest
                if capture.run_type == "tool":
                    attrs["tool"] = capture.tool_name or capture.name
                elif capture.run_type in ("llm", "chat_model"):
                    attrs["model_id"] = capture.model_id
                    attrs["runnable"] = capture.name
                else:
                    attrs["runnable"] = capture.name
                if capture.error_type:
                    attrs["error_type"] = capture.error_type
                    attrs["error_digest"] = capture.error_digest

                deps: List[DependencyEdge] = []
                pending = capture.dependency
                if pending is not None and pending.src_idx in completed_idxs:
                    resource = (
                        ResourceRef(_NS, pending.digest or "", "")
                        if pending.grade is Grade.OBSERVED and pending.digest
                        else None
                    )
                    deps.append(
                        DependencyEdge(
                            src_idx=pending.src_idx,
                            grade=pending.grade,
                            resource=resource,
                            evidence=dict(pending.evidence),
                        )
                    )

                exec_preds = (
                    [capture.exec_pred_idx]
                    if capture.exec_pred_idx is not None
                    and capture.exec_pred_idx in completed_idxs
                    else []
                )
                result.append(
                    Step(
                        idx=capture.idx,
                        agent=capture.name,
                        kind=capture.kind,
                        deps=deps,
                        node_attrs=attrs,
                        exec_preds=exec_preds,
                        correlation_id=capture.run_id,
                    )
                )
            return result

    def to_steps(self, source: Any = None) -> List[Step]:
        """Adapter-protocol alias for :meth:`steps`; ``source`` is ignored."""
        return self.steps()

    def __call__(self, source: Any = None) -> List[Step]:
        return self.to_steps(source)
