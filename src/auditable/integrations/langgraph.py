"""LangGraph capture: instrument a real StateGraph into the typed decision graph.

``instrument(StateGraph(State))`` returns a thin proxy. Building the graph is
unchanged; on each node run the proxy substitutes a recording proxy for the state
the node receives, so every channel the node reads and every channel its returned
update writes is logged. After the user invokes the compiled graph, the proxy is
its own source and adapter, so::

    builder = instrument(StateGraph(State))
    builder.add_node("pay", pay); ...
    graph = builder.compile()
    graph.invoke(initial_state)
    report = analyze_run(builder, adapter=builder)   # POST, on the real run

The dependency layer is matched superstep-aware and reducer-aware (see
:mod:`auditable.graph.touch`): a read binds to the writer(s) committed in an
*earlier* superstep, so same-superstep parallel writes are never fabricated and
reducer channels fan in from every writer. Edges are graded OBSERVED as a
channel-level read-after-committed-write touch, not a precise causal claim.

Coupling is confined to public LangGraph seams: ``add_node`` (the boundary we
wrap), the node ``(state, config)`` contract, ``config["metadata"]["langgraph_step"]``
(the superstep), reducer detection from the state-schema annotations, and
``Command.update``. Tested against langgraph 1.2.x. ``langgraph`` is an optional
extra; the core package never imports this module.

v1 scope: TypedDict / dict (mapping-shaped) state, plain sync or async function
nodes (with any of LangGraph's injected ``config`` / ``runtime`` / ``store`` /
``writer`` parameters, forwarded by name), and writes from a returned dict or a
``Command(update=...)`` in either mapping or ``[(channel, value), ...]`` form. A
dataclass or Pydantic state schema is rejected at ``instrument`` time with a clear
error, because the recording proxy is mapping-shaped (attribute-style state access
is not captured). Runnable nodes pass through uncaptured and ``to_steps`` warns that
the graph is incomplete. Routing-condition reads (conditional edges /
``Command.goto``) are not captured this round, and one run is captured at a time per
instrumented builder.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import threading
import typing
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from ..graph.session import ResourceRef, Step
from ..graph.touch import StepTouch, _safe_deepcopy, touches_to_records, touches_to_steps

if TYPE_CHECKING:  # the to_records bridge lowers to the record/replay core
    from ..record import DecisionRecord

try:  # the import doubles as the optional-extra presence check
    from langgraph.channels.binop import BinaryOperatorAggregate as _BinaryOperatorAggregate
    from langgraph.types import Command as _Command
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "auditable.integrations.langgraph requires langgraph: pip install auditable[langgraph]"
    ) from exc

__all__ = ["instrument", "InstrumentedStateGraph"]

_NS = "langgraph_state"  # the resource namespace for state-channel touches


class _RecordingState(Mapping):
    """A read-logging proxy over the state mapping a node receives.

    Keyed access (``state["x"]`` / ``state.get("x")``) logs a precise read of that
    channel. A whole-state access (iteration, ``keys`` / ``values`` / ``items``,
    ``**state``, ``dict(state)``) logs every channel and marks them overcaptured, so
    the resulting edges are flagged honestly. Membership (``"x" in state``) is a
    presence check, not a value read, and is not logged.
    """

    def __init__(
        self, raw: Mapping, reads: Set[str], over: Set[str], values: Dict[str, Any]
    ) -> None:
        self._raw = raw
        self._reads = reads
        self._over = over
        self._values = values  # relied-on read values, for the replay (to_records) bridge

    def __getitem__(self, key: str) -> Any:
        self._reads.add(key)
        value = self._raw[key]
        # copy at read time so the snapshot holds the value relied on now, immune to a
        # later in-place mutation of the same object; the node still gets the raw value.
        self._values[key] = _safe_deepcopy(value)
        return value

    def __iter__(self):
        for key in self._raw:  # whole-state scan: every channel read, and overcaptured
            self._reads.add(key)
            self._over.add(key)
            self._values[key] = _safe_deepcopy(self._raw[key])
        return iter(list(self._raw))

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, key: object) -> bool:
        return key in self._raw  # presence check, not a value read

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._raw:
            self._reads.add(key)
            self._values[key] = _safe_deepcopy(self._raw[key])
        return self._raw.get(key, default)

    def copy(self) -> dict:
        # a whole-state copy: every channel read, and overcaptured
        for key in self._raw:
            self._reads.add(key)
            self._over.add(key)
            self._values[key] = _safe_deepcopy(self._raw[key])
        return dict(self._raw)


def _extract_writes(result: Any) -> List[str]:
    """The channels a node wrote: from its returned update or ``Command.update``.

    The update is a mapping (``{channel: value}``) or, in the ``Command`` form, a list
    of ``(channel, value)`` pairs (LangGraph accepts both)."""
    update = result.update if isinstance(result, _Command) else result
    if isinstance(update, Mapping):
        return list(update.keys())
    if isinstance(update, (list, tuple)) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
        for item in update
    ):
        return [item[0] for item in update]
    return []


def _extract_write_values(result: Any) -> Dict[str, Any]:
    """The channel -> value map a node wrote, mirroring ``_extract_writes`` (which
    returns only the keys). The replay (to_records) bridge uses it to fill a marked
    decision node's action arguments from what the node wrote."""
    update = result.update if isinstance(result, _Command) else result
    if isinstance(update, Mapping):
        return dict(update)
    if isinstance(update, (list, tuple)) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
        for item in update
    ):
        return {item[0]: item[1] for item in update}
    return {}


def _lookup_field(node: str, arg_name: str, state_key: str, fields: Dict[str, Any]) -> Any:
    """Resolve a captured field for an action arg/cost, or raise naming what is available.

    A mistyped field name in ``to_records(action_args=..., action_costs=...)`` should fail
    loudly at lowering, not silently lower a wrong (or zero-cost) action."""
    if state_key not in fields:
        raise KeyError(
            f"to_records: node {node!r} maps {arg_name!r} to captured field {state_key!r}, "
            f"which the node did not read or write. Captured fields: {sorted(fields)}."
        )
    return fields[state_key]


@dataclass
class _Capture:
    """One node invocation's raw capture, before lowering to a StepTouch."""

    idx: int
    superstep: int
    node: str
    reads: List[str]
    writes: List[str]
    overcaptured: List[str]
    read_values: Dict[str, Any] = field(default_factory=dict)
    write_values: Dict[str, Any] = field(default_factory=dict)


class _Recorder:
    """Thread-safe accumulator of node-invocation captures across one run.

    LangGraph may run a superstep's nodes on a thread pool, so the index counter
    and capture list are guarded by a lock.
    """

    def __init__(self) -> None:
        self._caps: List[_Capture] = []
        self._next_idx = 0
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._active = False

    def reset(self) -> None:
        """Clear captures before a new run. A compiled graph is reusable and
        LangGraph restarts superstep numbering each invocation, so captures from
        separate runs must not accumulate (they would mix across the shared
        superstep ids and fabricate cross-run edges)."""
        with self._lock:
            self._caps = []
            self._next_idx = 0

    def begin_run(self) -> None:
        """Start a run: reset and mark active. One run at a time per recorder, so two
        compiled graphs from one instrumented builder cannot run concurrently and mix
        captures (the guard lives here, on the shared recorder, not on each proxy)."""
        with self._run_lock:
            if self._active:
                raise RuntimeError(
                    "auditable LangGraph capture supports one active run per instrumented "
                    "builder; finish the current run before starting another."
                )
            self._active = True
            self.reset()

    def end_run(self) -> None:
        with self._run_lock:
            self._active = False

    def record(
        self,
        node: str,
        superstep: int,
        reads: Set[str],
        over: Set[str],
        result: Any,
        read_values: Dict[str, Any],
    ) -> None:
        # Backstop: only capture inside an active run. A supported entry point
        # (invoke / ainvoke / stream / astream) calls begin_run before any node
        # fires; any other execution surface that runs the wrapped nodes without
        # begin_run therefore captures nothing instead of mixing into the recorder.
        if not self._active:
            return
        writes = _extract_writes(result)
        # read_values arrive already copied at read time (see _RecordingState); the node's
        # returned writes are raw, so copy them before they can be mutated downstream.
        write_values = _safe_deepcopy(_extract_write_values(result))
        with self._lock:
            idx = self._next_idx
            self._next_idx += 1
            self._caps.append(
                _Capture(
                    idx=idx,
                    superstep=superstep,
                    node=node,
                    reads=sorted(reads),
                    writes=writes,
                    overcaptured=sorted(over),
                    read_values=dict(read_values),
                    write_values=write_values,
                )
            )

    def to_touches(self) -> List[StepTouch]:
        """Lower captures to StepTouches, setting exec_preds to the prior superstep.

        Execution predecessors are the nodes of the immediately prior superstep
        (the BSP round that handed off to this one): ``[]`` for the first superstep
        (roots), the prior round's node ids otherwise.
        """
        caps = sorted(self._caps, key=lambda c: (c.superstep, c.idx))
        touches: List[StepTouch] = []
        prev_superstep_idxs: List[int] = []
        i, n = 0, len(caps)
        while i < n:
            s = caps[i].superstep
            group: List[_Capture] = []
            while i < n and caps[i].superstep == s:
                group.append(caps[i])
                i += 1
            for c in group:
                touches.append(
                    StepTouch(
                        idx=c.idx,
                        superstep=c.superstep,
                        agent=c.node,
                        kind="decision",
                        reads=[ResourceRef(_NS, k, "") for k in c.reads],
                        writes=[ResourceRef(_NS, k, "") for k in c.writes],
                        node_attrs={"langgraph_node": c.node},
                        overcaptured=frozenset(ResourceRef(_NS, k, "") for k in c.overcaptured),
                        exec_preds=list(prev_superstep_idxs),
                        # channel name is the snapshot.state key (key="" for state channels)
                        read_values=dict(c.read_values),
                        write_values=dict(c.write_values),
                    )
                )
            prev_superstep_idxs = [c.idx for c in group]
        return touches


# The objects LangGraph injects into a node by parameter name. The wrapper forwards
# to a node exactly the ones it declares, mirroring LangGraph's name-based injection,
# so a node that wants `runtime` (or `store` / `writer`) is not handed the config.
_INJECTABLES = ("config", "runtime", "store", "writer", "previous")


def _injectables_wanted(fn: Any) -> tuple:
    try:
        params = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return ()
    return tuple(k for k in _INJECTABLES if k in params)


def _wrappable(fn: Any) -> bool:
    """Plain sync/async function-like nodes are wrapped; Runnables etc. pass through."""
    return (
        inspect.isfunction(fn)
        or inspect.ismethod(fn)
        or asyncio.iscoroutinefunction(fn)
        or isinstance(fn, functools.partial)
    )


def _superstep_of(config: Any) -> int:
    if isinstance(config, Mapping):
        meta = config.get("metadata") or {}
        if isinstance(meta, Mapping):
            value = meta.get("langgraph_step")
            if isinstance(value, int):
                return value
    return 0


def _first_param_name(fn: Any) -> str:
    try:
        return next(iter(inspect.signature(fn).parameters), "state")
    except (TypeError, ValueError):
        return "state"


def _wrapper_signature(fn: Any) -> inspect.Signature:
    """The wrapper's signature: the node's first (state) parameter plus all the
    injectables, so ``RunnableCallable`` still injects ``config`` (and the rest) by name."""
    state_name = _first_param_name(fn)
    params = [inspect.Parameter(state_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    params += [
        inspect.Parameter(inj, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
        for inj in _INJECTABLES
    ]
    return inspect.Signature(parameters=params)


def _copy_langgraph_metadata(wrapper: Any, fn: Any, fn_name: str) -> Any:
    """Carry the node's identity, type hints, and a LangGraph-visible signature onto
    the wrapper, so LangGraph still infers the node's input/return channel schemas
    (it reads the function's type hints in ``add_node``) and still injects ``config``.

    ``functools.wraps(fn)`` alone is wrong here: it sets ``__wrapped__``, so
    ``inspect.signature`` follows the original and LangGraph would stop injecting
    ``config`` into wrappers for nodes that do not declare it."""
    wrapper.__name__ = fn_name
    wrapper.__qualname__ = getattr(fn, "__qualname__", fn_name)
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    wrapper.__module__ = getattr(fn, "__module__", wrapper.__module__)
    try:
        wrapper.__annotations__ = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        wrapper.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    wrapper.__signature__ = _wrapper_signature(fn)
    return wrapper


def _wrap(fn: Any, name: str, rec: _Recorder) -> Any:
    """Wrap a node callable so its reads and writes are recorded into ``rec``.

    The wrapper declares all of LangGraph's injectables so LangGraph injects them by
    name (``config`` always carries the superstep); each is forwarded to the original
    only if the original declares it, so a ``(state, runtime)`` node gets its runtime
    and not the config. The wrapper carries the node's name, type hints, and a
    LangGraph-visible signature (see :func:`_copy_langgraph_metadata`), so the
    single-arg ``add_node(fn)`` name inference and LangGraph's input/return channel
    schema inference both still work.
    """
    fn_name = getattr(fn, "__name__", name)
    wanted = _injectables_wanted(fn)

    if asyncio.iscoroutinefunction(fn):

        async def _awrapper(state, config=None, runtime=None, store=None, writer=None, previous=None):
            reads: Set[str] = set()
            over: Set[str] = set()
            values: Dict[str, Any] = {}
            proxy = _RecordingState(state, reads, over, values)
            injected = {"config": config, "runtime": runtime, "store": store, "writer": writer, "previous": previous}
            result = await fn(proxy, **{k: injected[k] for k in wanted})
            rec.record(name, _superstep_of(config), reads, over, result, values)
            return result

        return _copy_langgraph_metadata(_awrapper, fn, fn_name)

    def _wrapper(state, config=None, runtime=None, store=None, writer=None, previous=None):
        reads: Set[str] = set()
        over: Set[str] = set()
        values: Dict[str, Any] = {}
        proxy = _RecordingState(state, reads, over, values)
        injected = {"config": config, "runtime": runtime, "store": store, "writer": writer, "previous": previous}
        result = fn(proxy, **{k: injected[k] for k in wanted})
        rec.record(name, _superstep_of(config), reads, over, result, values)
        return result

    return _copy_langgraph_metadata(_wrapper, fn, fn_name)


def _is_mapping_state(state_schema: Any) -> bool:
    """Whether the state schema is mapping-shaped (TypedDict / dict).

    The recording proxy is a ``Mapping``, so a node reads state by key. A dataclass
    or Pydantic model is attribute-shaped (``state.x``) and is not supported this
    round; ``None`` (no declared schema) is treated as a plain dict.
    """
    if state_schema is None:
        return True
    if typing.is_typeddict(state_schema):
        return True
    if isinstance(state_schema, type) and issubclass(state_schema, dict):
        return True
    if dataclasses.is_dataclass(state_schema):
        return False
    if hasattr(state_schema, "model_fields") or hasattr(state_schema, "__fields__"):
        return False
    return True  # unknown shape: allow, mapping-style access may still work


def _reducer_channels(builder: Any) -> frozenset:
    """The set of reducer channels, read from LangGraph's own channel map.

    A channel whose built channel object is a ``BinaryOperatorAggregate`` accumulates
    writes (it came from ``Annotated[T, reducer]``); every other channel
    (``LastValue`` and the rest) is overwrite. Reading ``builder.channels`` uses
    LangGraph's authoritative classification, so it matches LangGraph exactly across
    ``Required`` / ``NotRequired`` unwrapping and multi-item ``Annotated`` metadata
    (only the last metadata item, a two-arg callable, is a reducer). Returns the
    matcher's resource-key tuples; a builder without a channel map yields an empty
    set (every channel treated as overwrite).
    """
    channels = getattr(builder, "channels", None) or {}
    return frozenset(
        (_NS, name, "")
        for name, channel in channels.items()
        if isinstance(channel, _BinaryOperatorAggregate)
    )


def _instrument_add_node_call(args: tuple, kwargs: dict, recorder: _Recorder):
    """Wrap the node action in an ``add_node`` call, across its public forms.

    LangGraph's signature is ``add_node(node, action=None, *, ...)``, so the action
    can be a positional second arg, an ``action=`` keyword, or (single-arg form) the
    ``node`` itself. Returns ``(new_args, new_kwargs, skipped, label)``: ``skipped``
    is True when a real node action was present but not wrappable (a Runnable), so
    the caller can record that the capture is incomplete.
    """
    new_args = list(args)
    new_kwargs = dict(kwargs)
    name = None
    if new_args and isinstance(new_args[0], str):
        name = new_args[0]
    elif isinstance(new_kwargs.get("node"), str):
        name = new_kwargs["node"]

    def _label(action: Any) -> str:
        return name or getattr(action, "__name__", "node")

    # explicit action: kwargs['action'] or positional[1]
    if new_kwargs.get("action") is not None:
        action = new_kwargs["action"]
        if _wrappable(action):
            new_kwargs["action"] = _wrap(action, _label(action), recorder)
            return tuple(new_args), new_kwargs, False, _label(action)
        return tuple(new_args), new_kwargs, True, name or "node"
    if len(new_args) >= 2 and new_args[1] is not None:
        action = new_args[1]
        if _wrappable(action):
            new_args[1] = _wrap(action, _label(action), recorder)
            return tuple(new_args), new_kwargs, False, _label(action)
        return tuple(new_args), new_kwargs, True, name or "node"

    # single-callable form: the node itself is the action (positional or node=)
    candidate = new_args[0] if new_args else new_kwargs.get("node")
    if candidate is not None and not isinstance(candidate, str):
        if _wrappable(candidate):
            label = getattr(candidate, "__name__", "node")
            if new_args:
                new_args[0] = _wrap(candidate, label, recorder)
            else:
                new_kwargs["node"] = _wrap(candidate, label, recorder)
            return tuple(new_args), new_kwargs, False, label
        return tuple(new_args), new_kwargs, True, getattr(candidate, "__name__", "node")

    return tuple(new_args), new_kwargs, False, None


class _CompiledGraphProxy:
    """Wraps a compiled LangGraph so each invocation captures exactly one run.

    A compiled graph is reusable and LangGraph restarts superstep numbering each
    run, so the shared recorder is reset at the start of every run, and one run is
    allowed at a time: a second run begun while another is active (on this proxy or
    another compiled from the same instrumented builder) raises rather than silently
    mixing captures. The guard lives on the shared recorder. For ``stream`` /
    ``astream`` the reset happens when iteration starts (inside the generator), not
    when the generator is created, so two stream generators created before either is
    drained do not mix. Only ``invoke`` / ``ainvoke`` / ``stream`` / ``astream`` are
    supported; ``batch`` / ``abatch`` (and the ``*_as_completed`` variants) would run
    the wrapped nodes for several inputs through one recorder and mix captures, so
    they fail closed with a clear error rather than capture wrongly.
    """

    def __init__(self, graph: Any, recorder: _Recorder) -> None:
        self._graph = graph
        self._recorder = recorder

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self._recorder.begin_run()
        try:
            return self._graph.invoke(*args, **kwargs)
        finally:
            self._recorder.end_run()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        self._recorder.begin_run()
        try:
            return await self._graph.ainvoke(*args, **kwargs)
        finally:
            self._recorder.end_run()

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        def _gen():
            self._recorder.begin_run()
            try:
                yield from self._graph.stream(*args, **kwargs)
            finally:
                self._recorder.end_run()

        return _gen()

    def astream(self, *args: Any, **kwargs: Any) -> Any:
        async def _agen():
            self._recorder.begin_run()
            try:
                async for chunk in self._graph.astream(*args, **kwargs):
                    yield chunk
            finally:
                self._recorder.end_run()

        return _agen()

    def _unsupported_run_method(self, method: str) -> Any:
        raise RuntimeError(
            "auditable LangGraph capture v1 supports only invoke / ainvoke / stream / "
            f"astream; {method} would run the wrapped nodes for several inputs through "
            "one recorder and mix captures."
        )

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        self._unsupported_run_method("batch")

    async def abatch(self, *args: Any, **kwargs: Any) -> Any:
        self._unsupported_run_method("abatch")

    # Delegated compiled-graph methods that would execute the wrapped nodes without
    # begin_run (and so mix captures). They fail closed; read-only methods such as
    # get_graph / get_state still delegate. The recorder's record() backstop also
    # drops any node fired outside an active run, so a surface missed here fails safe.
    _BLOCKED_RUN_METHODS = frozenset({
        "batch_as_completed", "abatch_as_completed",
        "stream_events", "astream_events",
        "stream_log", "astream_log",
    })

    def __getattr__(self, attr: str) -> Any:
        if attr in _CompiledGraphProxy._BLOCKED_RUN_METHODS:
            def _blocked(*args: Any, **kwargs: Any) -> Any:
                self._unsupported_run_method(attr)

            return _blocked
        return getattr(object.__getattribute__(self, "_graph"), attr)


class InstrumentedStateGraph:
    """A thin proxy over a LangGraph ``StateGraph`` that records each node's touches.

    Wrap the builder once (``instrument(StateGraph(State))``) and build normally;
    ``add_node`` is intercepted to wrap node actions, every other method delegates
    to the wrapped builder. ``compile`` returns a proxy that resets capture per run.
    The proxy implements the ``Adapter`` protocol and is its own source, so
    ``analyze_run(builder, adapter=builder)`` lowers the most recent captured run.
    """

    name = "langgraph_capture"
    version = "v1"

    def __init__(self, builder: Any) -> None:
        state_schema = getattr(builder, "state_schema", None)
        if not _is_mapping_state(state_schema):
            raise TypeError(
                "auditable LangGraph capture (v1) supports TypedDict / dict state; got "
                f"{getattr(state_schema, '__name__', state_schema)!r}, an attribute-shaped "
                "state schema (dataclass or Pydantic) that is not yet supported."
            )
        self._builder = builder
        self._recorder = _Recorder()
        self._skipped: Set[str] = set()

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"

    def _proxy_result(self, result: Any) -> Any:
        # LangGraph's builder methods return Self for fluent chaining; keep returning
        # this proxy so chained `.add_node(...)` calls stay instrumented.
        return self if result is self._builder else result

    def add_node(self, *args: Any, **kwargs: Any) -> Any:
        """Forward to the wrapped builder with the node action wrapped for capture."""
        new_args, new_kwargs, skipped, label = _instrument_add_node_call(args, kwargs, self._recorder)
        if skipped and label:
            self._skipped.add(label)
        return self._proxy_result(self._builder.add_node(*new_args, **new_kwargs))

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        return _CompiledGraphProxy(self._builder.compile(*args, **kwargs), self._recorder)

    def to_steps(self, source: Any = None) -> List[Step]:
        """Lower the captured run into typed steps with OBSERVED dependency edges."""
        if self._skipped:
            warnings.warn(
                f"{len(self._skipped)} LangGraph node(s) were not captured "
                f"(Runnable or unsupported): {sorted(self._skipped)}. The captured "
                "dependency graph may be incomplete.",
                UserWarning,
                stacklevel=2,
            )
        # reducer channels are read now, after construction, so channels added by
        # add_node(..., input_schema=...) after instrument() are classified correctly.
        touches = self._recorder.to_touches()
        return touches_to_steps(
            touches, reducer_channels=_reducer_channels(self._builder), adapter=self.id
        )

    def to_records(
        self,
        decisions: Any,
        *,
        sink: Any = None,
        action_args: "Optional[Mapping[str, Mapping[str, str]]]" = None,
        action_costs: "Optional[Mapping[str, str]]" = None,
    ) -> "List[DecisionRecord]":
        """Lower the marked decision node(s) of the captured run into replayable
        ``DecisionRecord``s, so ``replay(record, live_state=..., policy=...)`` re-decides
        a real captured action against state that is live now.

        ``decisions`` names the consequential node(s): a mapping ``{node: action_type}``,
        or an iterable of node names (the node name is then the action type). For each
        marked node the relied-on read values become the ``DependencySnapshot``.

        The action a record carries is shaped by two optional maps, both keyed by node:

        - ``action_args`` (``{node: {arg_name: field}}``) sets the action's arguments from
          named captured fields. Without it the action arguments default to the node's
          write values.
        - ``action_costs`` (``{node: field}``) sets the action's cost from a captured
          field. Without it the cost is ``0.0``, so a policy that re-decides from the
          snapshot or live *state* still works, but a policy that inspects ``action.cost``
          would see ``0``. Map the cost field whenever the replay policy is cost-based.

        A field name resolves against the node's captured reads and writes (writes win on a
        name clash); an unknown field raises ``KeyError`` naming the captured fields. This
        is the wrap-once LIVE companion to ``to_steps`` (POST); both read the same run.
        """
        wanted = dict(decisions) if isinstance(decisions, Mapping) else {n: n for n in decisions}
        arg_map = dict(action_args) if action_args else {}
        cost_map = dict(action_costs) if action_costs else {}
        touches = self._recorder.to_touches()
        missing = (set(wanted) | set(arg_map) | set(cost_map)) - {t.agent for t in touches}
        if missing:
            warnings.warn(
                f"to_records: decision node(s) not in the captured run: {sorted(missing)}.",
                UserWarning,
                stacklevel=2,
            )
        for touch in touches:
            if touch.agent not in wanted:
                continue
            fields = {**touch.read_values, **touch.write_values}
            touch.action_type = wanted[touch.agent]
            if touch.agent in arg_map:
                touch.action_args = {
                    arg_name: _lookup_field(touch.agent, arg_name, state_key, fields)
                    for arg_name, state_key in arg_map[touch.agent].items()
                }
            else:
                touch.action_args = dict(touch.write_values)
            if touch.agent in cost_map:
                touch.action_cost = float(
                    _lookup_field(touch.agent, "cost", cost_map[touch.agent], fields)
                )
        return touches_to_records(touches, sink=sink)

    def __getattr__(self, attr: str) -> Any:
        # only fires for attributes not defined above; delegate to the wrapped builder.
        # Wrap callables so a builder method that returns Self for chaining (e.g.
        # add_edge) returns this proxy, keeping a chained `.add_node` instrumented.
        builder = object.__getattribute__(self, "_builder")
        value = getattr(builder, attr)
        if not callable(value):
            return value

        @functools.wraps(value)
        def _delegated(*args: Any, **kwargs: Any) -> Any:
            return self._proxy_result(value(*args, **kwargs))

        return _delegated


def instrument(builder: Any) -> InstrumentedStateGraph:
    """Wrap a LangGraph ``StateGraph`` for capture. Call before adding nodes.

    Returns an :class:`InstrumentedStateGraph`. Build, compile, and invoke as usual;
    then pass the returned object to ``analyze_run(builder, adapter=builder)``.
    """
    return InstrumentedStateGraph(builder)
