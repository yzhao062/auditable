"""Runtime touch capture: turn observed resource reads/writes into OBSERVED edges.

A *touch* is a step's record of which resources it read and which it wrote, each
a typed :class:`~auditable.graph.session.ResourceRef`. The matcher lowers an
ordered list of touches into the per-step ``OBSERVED`` dependency edges the
``SessionGraph`` consumes, so a live run and the offline corpora share one
representation.

This is the framework-agnostic spine the LangGraph source (and any future source)
rides on. It imports nothing beyond the typed schema in
:mod:`auditable.graph.session`; there is no framework dependency here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .session import DependencyEdge, Grade, ResourceRef, Step

__all__ = ["StepTouch", "TouchRecorder", "match_observed_deps", "touches_to_steps"]

_Key = Tuple[str, str, str]


def _key(ref: ResourceRef) -> _Key:
    return (ref.namespace, ref.resource_id, ref.key)


@dataclass
class StepTouch:
    """One step's observed touches: the matcher's input unit.

    ``superstep`` groups steps that ran without seeing each other's writes (a
    LangGraph BSP superstep; the generic recorder gives each step its own). Reads
    and writes are typed resource references. ``overcaptured`` names the reads that
    were taken from a whole-state access (``**state`` / ``keys()``) rather than a
    single keyed access, so the resulting edge can be flagged honestly.
    """

    idx: int
    superstep: int
    agent: str
    kind: str = "decision"
    reads: List[ResourceRef] = field(default_factory=list)
    writes: List[ResourceRef] = field(default_factory=list)
    node_attrs: Dict[str, Any] = field(default_factory=dict)
    overcaptured: frozenset = field(default_factory=frozenset)
    exec_preds: Optional[List[int]] = None


def _edge(
    read_ref: ResourceRef,
    writer_idx: int,
    writer_superstep: int,
    *,
    reducer: bool,
    overcaptured: bool,
    adapter: str,
) -> DependencyEdge:
    evidence: Dict[str, Any] = {
        "observed": True,
        "relation": "read_after_committed_write",
        "granularity": "channel",
        "channel": read_ref.resource_id,
        "writer_superstep": writer_superstep,
        "mode": "reducer" if reducer else "overwrite",
        "adapter": adapter,
    }
    if reducer:
        evidence["modeled"] = "reducer_writer_set"
    if overcaptured:
        evidence["overcaptured"] = True
    return DependencyEdge(
        src_idx=writer_idx,
        grade=Grade.OBSERVED,
        resource=read_ref,
        evidence=evidence,
    )


def match_observed_deps(
    touches: Sequence[StepTouch],
    *,
    reducer_channels: AbstractSet[_Key] = frozenset(),
    adapter: str = "touch_matcher",
) -> Dict[int, List[DependencyEdge]]:
    """Lower ordered touches into per-step OBSERVED dependency edges.

    A read of resource ``R`` binds to the writer(s) of ``R`` committed in an
    *earlier* superstep. Writes commit only at the superstep barrier, so two
    steps in one superstep never see each other's writes (matching LangGraph's
    BSP model). An overwrite channel keeps its single most recent writer; a
    reducer channel (named in ``reducer_channels``) accumulates every writer, so a
    read of it fans in to the whole committed writer set, marked modeled.
    """
    result: Dict[int, List[DependencyEdge]] = {t.idx: [] for t in touches}
    # key -> committed writer list. Overwrite channels hold one entry (the last
    # writer); reducer channels accumulate every writer across supersteps.
    committed: Dict[_Key, List[Tuple[int, int]]] = {}
    ordered = sorted(touches, key=lambda t: (t.superstep, t.idx))
    i, n = 0, len(ordered)
    while i < n:
        superstep = ordered[i].superstep
        group: List[StepTouch] = []
        while i < n and ordered[i].superstep == superstep:
            group.append(ordered[i])
            i += 1
        # 1) match every read in this superstep against earlier-committed writers.
        # Reads are deduplicated per step: a dependency is one edge between two steps
        # over a resource, not one edge per repeated access event.
        for t in group:
            for r in dict.fromkeys(t.reads):
                k = _key(r)
                writers = committed.get(k)
                if not writers:
                    continue  # free / external read: bound by the initial state, no edge
                is_reducer = k in reducer_channels
                targets = writers if is_reducer else [writers[-1]]
                over = r in t.overcaptured
                for wsuper, widx in targets:
                    result[t.idx].append(
                        _edge(r, widx, wsuper, reducer=is_reducer, overcaptured=over, adapter=adapter)
                    )
        # 2) commit this superstep's writes after the barrier, deduplicated per step
        # (a step is one writer of a resource, even if it wrote it more than once).
        for t in group:
            for w in dict.fromkeys(t.writes):
                k = _key(w)
                if k in reducer_channels:
                    committed.setdefault(k, []).append((t.superstep, t.idx))
                else:
                    committed[k] = [(t.superstep, t.idx)]
    return result


def touches_to_steps(
    touches: Sequence[StepTouch],
    *,
    reducer_channels: AbstractSet[_Key] = frozenset(),
    adapter: str = "touch_matcher",
) -> List[Step]:
    """Lower touches into typed :class:`Step`s with matched OBSERVED dependency edges.

    The execution layer (``exec_preds``) is carried through verbatim from each
    touch; the dependency layer is the matcher's output. Steps are emitted in
    ``(superstep, idx)`` order so that ``SessionGraph.to_networkx`` (which only
    wires a dependency to an already-seen step) keeps every matched edge, even if
    the caller passed touches out of causal order.
    """
    deps_by_idx = match_observed_deps(touches, reducer_channels=reducer_channels, adapter=adapter)
    ordered = sorted(touches, key=lambda t: (t.superstep, t.idx))
    return [
        Step(
            idx=t.idx,
            agent=t.agent,
            kind=t.kind,
            deps=deps_by_idx[t.idx],
            node_attrs=dict(t.node_attrs),
            exec_preds=t.exec_preds,
        )
        for t in ordered
    ]


class _StepTouchBuilder:
    """The handle yielded by ``TouchRecorder.step``; records reads and writes.

    ``reads`` / ``writes`` take a resource as ``(namespace, resource_id, key="")``
    and return ``self`` so calls chain. The accumulated touch is sealed when the
    ``with`` block exits.
    """

    def __init__(self, touch: StepTouch) -> None:
        self._touch = touch

    def reads(self, namespace: str, resource_id: str, key: str = "") -> "_StepTouchBuilder":
        self._touch.reads.append(ResourceRef(namespace, resource_id, key))
        return self

    def writes(self, namespace: str, resource_id: str, key: str = "") -> "_StepTouchBuilder":
        self._touch.writes.append(ResourceRef(namespace, resource_id, key))
        return self


class TouchRecorder:
    """Manual runtime touch capture for any framework (or a plain tool loop).

    Wrap each consequential step in ``with rec.step(agent=..., kind=...) as st:``
    and declare what it ``reads`` and ``writes``. Each ``step`` is its own
    superstep (sequential), so the dependency layer is a last-writer-wins match
    over the declared touches. The recorder implements the :class:`Adapter`
    protocol (``name`` / ``version`` / ``to_steps``) and is its own source, so it
    drops straight into ``analyze_run(rec, adapter=rec)``.
    """

    name = "touch_recorder"
    version = "v1"

    def __init__(self) -> None:
        self._touches: List[StepTouch] = []
        self._next_idx = 0

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"

    @contextmanager
    def step(
        self,
        *,
        agent: str,
        kind: str = "decision",
        node_attrs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[_StepTouchBuilder]:
        idx = self._next_idx
        self._next_idx += 1
        touch = StepTouch(
            idx=idx,
            superstep=idx,  # each manual step is its own superstep (sequential)
            agent=agent,
            kind=kind,
            node_attrs=dict(node_attrs or {}),
        )
        yield _StepTouchBuilder(touch)
        self._touches.append(touch)

    def to_steps(self, source: Any = None) -> List[Step]:
        """Lower the recorded touches into typed steps. ``source`` is ignored (the
        recorder is its own source), so ``analyze_run(rec, adapter=rec)`` works."""
        return touches_to_steps(self._touches, adapter=self.id)
