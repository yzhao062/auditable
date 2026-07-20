"""MCP tool-boundary capture: record every tool call through a FastMCP server.

``instrument(FastMCP(...))`` patches the server's tool dispatch in place and
returns a handle that is its own source and adapter. The server keeps serving
through any transport (stdio, streamable HTTP, or the SDK's in-memory test
transport), because the capture sits below them all, at the one dispatch point
every ``tools/call`` request passes through. This is the ops-side mount: the
team hosting the server records the traffic without touching agent (client)
code. After some traffic::

    server = FastMCP("support-desk")
    ... register tools as usual ...
    mount = instrument(server)                     # once, before or after tools
    ... serve as usual; each tools/call lands as one typed step ...
    report = analyze_run(mount, adapter=mount)     # the calls captured so far

Scope, stated plainly: MCP capture sees tool traffic only (State A at the tool
boundary). Each call becomes one ``tool_call`` step carrying the tool name, an
arguments digest, a result digest, and timing. The boundary does not expose
which dependency state a tool read or wrote, so dependency-state providers
(State B) still have to be registered separately, and the captured steps carry
no dependency edges at all: no read/write touch is observed here, none is
fabricated, and nothing is presented as an OBSERVED edge. ``idx`` is
dispatch-start ordering metadata only. Each step has explicit empty execution
predecessors, because unrelated or overlapping client calls do not establish a
handoff at this boundary. Consequently, analysis yields an honest no-score /
zero-coverage report until State-B providers supply dependency edges.

Values stay out of the graph by default: arguments and results enter as SHA-256
digests over canonical JSON, plus the argument names, so a step commits to what
was passed without copying payloads into node attributes. Errors retain only
their type and a digest over the type and message, never the raw message. A value
with no canonical JSON form falls back to a digest over ``repr(...)``, marked
with a ``*_digest_mode: "repr"`` attribute (a repr digest is stable only within
one Python implementation; absence of the mode key means canonical JSON).

Coupling: the patch replaces ``server._tool_manager.call_tool`` (FastMCP's one
tool-dispatch point) on the instance. That is a private seam with no public
hook today; the SDK's own in-memory test transport reaches into
``FastMCP._mcp_server`` the same way. Tested against mcp 1.28.x. v1 scope is
FastMCP servers; a low-level ``mcp.server.Server`` registers its own call-tool
handler and is rejected with a clear error. ``mcp`` is an optional extra; the
core package never imports this module.
"""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..graph.session import Step
from ..graph.touch import StepTouch, touches_to_steps
from ..record import _sha256

try:  # the import doubles as the optional-extra presence check
    from mcp.server.fastmcp import FastMCP as _FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "auditable.integrations.mcp requires Python 3.10+ and mcp>=1.28.1,<2: "
        "pip install auditable[mcp]"
    ) from exc

__all__ = ["instrument", "InstrumentedMCPServer"]


def _repr_digest(value: Any) -> str:
    """Fallback digest for a value with no canonical JSON form. Capture must never
    crash the call it is observing, so even a ``repr`` that raises degrades to a
    typed placeholder instead of an exception."""
    try:
        text = repr(value)
    except Exception:
        text = f"<unreprable {type(value).__name__}>"
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _content_digest(value: Any) -> Tuple[str, str]:
    """Digest a captured payload: ``(sha256_hex, mode)``.

    ``mode`` is ``"json"`` when the digest is over canonical JSON (sorted keys,
    compact; see :func:`auditable.record._sha256`), or ``"repr"`` for the
    fallback over ``repr(value)`` when the payload is not JSON-serializable.
    """
    try:
        return _sha256(value), "json"
    except (TypeError, ValueError):
        return _repr_digest(value), "repr"


def _error_fingerprint(exc: BaseException) -> Tuple[str, str]:
    """Return an exception type and content digest without retaining raw text."""
    error_type = type(exc).__name__
    try:
        message = str(exc)
    except BaseException:
        message = "<unprintable>"
    digest, _mode = _content_digest({"type": error_type, "message": message})
    return error_type, digest


def _dump_block(block: Any) -> Any:
    """A content block as plain JSON data (pydantic model), or the raw item."""
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            return block
    return block


def _result_payload(result: Any) -> Any:
    """A JSON-able projection of the tool result at the dispatch seam.

    With ``convert_result=True`` (the wire path) FastMCP returns a content-block
    list, a structured dict, or a ``(content, structured)`` pair; content blocks
    are pydantic models, dumped to plain data. A top-level pydantic result (such
    as ``CallToolResult``) is dumped the same way; anything else passes through
    and the digest helper falls back to a repr digest if it is not JSON-able.
    """
    if isinstance(result, tuple) and len(result) == 2:
        content, structured = result
        return {"content": _result_payload(content), "structured": structured}
    if isinstance(result, (list, tuple)):
        return [_dump_block(item) for item in result]
    return _dump_block(result)


def _request_id_of(context: Any) -> Optional[str]:
    """The MCP request id, when dispatch runs inside a request (the wire path).

    A direct in-process ``server.call_tool`` carries a context with no request
    attached; the SDK's ``Context.request_id`` raises there, so this returns
    ``None`` instead of crashing the capture."""
    if context is None:
        return None
    try:
        return str(context.request_id)
    except Exception:
        return None


class _CallLog:
    """Thread-safe accumulator of tool-call touches across one capture window.

    A hosted server may dispatch tool calls concurrently (asyncio tasks, or
    sessions on more than one thread), so the index counter and touch list are
    guarded by a lock. The index is taken at dispatch start, so ``to_steps``
    (which sorts by it) emits steps in dispatch-start order even when an
    overlapping call finishes first. It is ordering metadata, not an execution
    dependency between calls.
    """

    def __init__(self) -> None:
        self._touches: List[StepTouch] = []
        self._next_idx = 0
        self._window = 0
        self._lock = threading.Lock()

    def reserve(self) -> Tuple[int, int]:
        with self._lock:
            window = self._window
            idx = self._next_idx
            self._next_idx += 1
            return window, idx

    def append(self, window: int, touch: StepTouch) -> None:
        with self._lock:
            if window != self._window:
                return
            self._touches.append(touch)

    def reset(self) -> None:
        with self._lock:
            self._window += 1
            self._touches.clear()
            self._next_idx = 0

    def snapshot(self) -> List[StepTouch]:
        with self._lock:
            return list(self._touches)


def _tool_touch(idx: int, agent: str, attrs: Dict[str, Any]) -> StepTouch:
    # No reads or writes: the boundary observes no resource touches, so the
    # matcher has nothing to lower and the dependency layer stays honestly empty.
    # idx records dispatch-start order only; it does not assert a handoff between
    # independent client calls, hence the explicit empty execution predecessors.
    return StepTouch(
        idx=idx,
        superstep=idx,
        agent=agent,
        kind="tool_call",
        node_attrs=attrs,
        exec_preds=[],
    )


def _make_recording_call_tool(original: Any, log: _CallLog, agent: str, server_name: str) -> Any:
    """Build the replacement for ``ToolManager.call_tool``: same signature and
    behavior, plus one recorded ``tool_call`` touch per dispatch (success or
    error). Errors are recorded and re-raised, so the server's error reporting
    to the client is unchanged."""

    async def _recording_call_tool(
        name: str, arguments: Any, context: Any = None, convert_result: bool = False
    ) -> Any:
        args = dict(arguments or {})
        window, idx = log.reserve()
        attrs: Dict[str, Any] = {"tool": name, "mcp_server": server_name}
        request_id = _request_id_of(context)
        if request_id is not None:
            attrs["mcp_request_id"] = request_id
        digest, mode = _content_digest(args)
        attrs["arguments_digest"] = digest
        if mode != "json":
            attrs["arguments_digest_mode"] = mode
        attrs["argument_keys"] = sorted(str(k) for k in args)
        attrs["started_at"] = time.time()
        t0 = time.perf_counter()
        try:
            result = await original(name, arguments, context=context, convert_result=convert_result)
        except BaseException as exc:
            attrs["duration_s"] = time.perf_counter() - t0
            attrs["is_error"] = True
            attrs["error_type"], attrs["error_digest"] = _error_fingerprint(exc)
            log.append(window, _tool_touch(idx, agent, attrs))
            raise
        attrs["duration_s"] = time.perf_counter() - t0
        attrs["is_error"] = False
        digest, mode = _content_digest(_result_payload(result))
        attrs["result_digest"] = digest
        if mode != "json":
            attrs["result_digest_mode"] = mode
        log.append(window, _tool_touch(idx, agent, attrs))
        return result

    _recording_call_tool._auditable_mcp_capture = True  # the double-instrument marker
    return _recording_call_tool


class InstrumentedMCPServer:
    """The handle over an in-place instrumented FastMCP server.

    ``instrument`` patches the server's tool dispatch and returns this handle,
    which implements the ``Adapter`` protocol (``name`` / ``version`` /
    ``to_steps``) and is its own source, so ``analyze_run(mount, adapter=mount)``
    reports on the calls captured so far. With no dependency edges, structural
    scores are withheld at zero coverage until State-B providers add them. A
    server has no run boundary of its own, so the capture window is cumulative:
    ``to_steps`` lowers everything since ``instrument`` or the last ``reset``.

    ``server`` is the wrapped FastMCP, for handing to transports and helpers
    that type-check on FastMCP (the SDK's in-memory test transport does). Every
    attribute this class does not define delegates to it, so ``mount.run(...)``
    serves as usual; the handle's own ``name`` / ``version`` are the adapter
    identity, and the server's shadowed attributes stay reachable via
    ``mount.server``.
    """

    name = "mcp_capture"
    version = "v1"

    def __init__(self, server: Any, *, agent: Optional[str] = None) -> None:
        if not isinstance(server, _FastMCP):
            raise TypeError(
                "auditable MCP capture (v1) instruments a FastMCP server; got "
                f"{type(server).__name__!r}. A low-level mcp.server.Server registers "
                "its own call-tool handler and is not supported this round."
            )
        manager = getattr(server, "_tool_manager", None)
        original = getattr(manager, "call_tool", None)
        if manager is None or not callable(original):
            raise TypeError(
                "auditable MCP capture could not find the FastMCP tool dispatch "
                "(server._tool_manager.call_tool); the installed mcp SDK may have "
                "moved this private seam. Tested against mcp 1.28.x."
            )
        if getattr(original, "_auditable_mcp_capture", False):
            raise RuntimeError(
                "this FastMCP server is already instrumented; instrument it once and "
                "share the handle (a second wrap would double-record every call)."
            )
        self.server = server
        self._agent = agent or str(getattr(server, "name", "") or "mcp_server")
        self._log = _CallLog()
        manager.call_tool = _make_recording_call_tool(
            original, self._log, self._agent, str(getattr(server, "name", "") or "")
        )

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"

    def reset(self) -> None:
        """Start a new capture window with step indices restarted at 0.

        Captured calls and calls begun in the prior window are discarded, even
        when an older in-flight call finishes after this reset.
        """
        self._log.reset()

    def to_steps(self, source: Any = None) -> List[Step]:
        """Lower the captured calls into typed ``tool_call`` steps in
        dispatch-start order. ``source`` is ignored (the handle is its own
        source), so ``analyze_run(mount, adapter=mount)`` works. The steps carry
        no dependency edges; see the module docstring for why that is the honest
        lowering at this boundary."""
        return touches_to_steps(self._log.snapshot(), adapter=self.id)

    def __getattr__(self, attr: str) -> Any:
        # only fires for attributes not defined above; delegate to the server so
        # the handle can stand in for it (serve, register tools, and so on).
        return getattr(object.__getattribute__(self, "server"), attr)


def instrument(server: Any, *, agent: Optional[str] = None) -> InstrumentedMCPServer:
    """Patch a FastMCP server in place for tool-boundary capture.

    Call once, before or after registering tools; every later ``tools/call``
    through any transport is recorded. ``agent`` is the agent label on the
    captured steps (default: the server's name); the boundary cannot see which
    client called, so the label names the serving boundary, not the caller.
    Returns the :class:`InstrumentedMCPServer` handle; pass it to
    ``analyze_run(mount, adapter=mount)``.
    """
    return InstrumentedMCPServer(server, agent=agent)
