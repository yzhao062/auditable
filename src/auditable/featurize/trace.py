"""One view of an agent trace for every method family.

Methods accept traces in the forms users and benchmarks already hold, and read them
through :func:`normalize_trace`:

- an :class:`auditable.graph.session.SessionGraph`;
- a list of GRADE-style step dicts (``idx``, ``agent``, ``kind``, optional ``deps``);
- a GRADE run dict (``{"steps": ..., "G": ..., "texts": ...}``);
- a typed NetworkX graph from :func:`auditable.graph.build_graph` (or GRADE's).

The view keeps the execution topology. A ``SessionGraph`` contributes its own
``to_networkx()`` projection, so parallel roots and merges (``exec_preds``) survive; a
typed graph passed in directly is kept unchanged, so benchmark corpora enter exactly as
they were built. Only when no graph is available is one built from the step list, with
the linear handoff chain that :func:`auditable.graph.build_graph` assumes.

Two representation differences are worth knowing. ``SessionGraph.to_networkx()`` adds no
``resource::shared_state`` node, which ``build_graph(..., shared_resource=True)`` adds by
default; methods that read every graph node see different node sets for the two. And a
``SessionGraph`` step stores dependencies as ``DependencyEdge`` objects, which the view
flattens to the integer step indices the step-dict form uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_STEP_TYPES = ("decision", "tool_call")


@dataclass
class TraceView:
    """A normalized trace. ``steps`` are ordered dicts with integer ``deps`` (indices of
    the steps each one relied on); ``texts`` aligns with ``steps`` when step text is
    known; ``graph`` is the typed execution-and-dependency graph when one was supplied.
    ``shared_resource`` is whether a graph built from ``steps`` gets the
    ``resource::shared_state`` node; GRADE builds its Who&When graphs with it and every
    other corpus without it."""

    steps: List[Dict[str, Any]]
    texts: Optional[List[str]] = None
    graph: Any = None
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    shared_resource: bool = True

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def step_ids(self) -> List[int]:
        return [s["idx"] for s in self.steps]

    def typed_graph(self, *, dependency: str = "explicit", shared_resource: Optional[bool] = None):
        """The typed graph: the one supplied with the trace, or one built from ``steps``.
        ``shared_resource=None`` uses the trace's own setting."""
        if self.graph is not None:
            return self.graph
        from auditable.graph import build_graph

        if shared_resource is None:
            shared_resource = self.shared_resource
        return build_graph(self.steps, dependency=dependency, shared_resource=shared_resource)

    def require_texts(self) -> List[str]:
        if self.texts is None or len(self.texts) != len(self.steps):
            raise ValueError(
                "this method reads step text; pass a run dict with 'texts', step dicts with "
                "'text', or a SessionGraph whose steps carry node_attrs['text']"
            )
        return self.texts


def full_context_edges(n: int):
    """The full-context dependency edges on ``n`` ordered steps as a (2, m) int64 array:
    step i depends on every prior step j, edge i -> j (current -> prior), the direction of
    GRADE's ``depends_on``. Ported from CatchBench ``graph_ad.full_context_edges``."""
    import numpy as np

    src, dst = [], []
    for i in range(n):
        for j in range(i):
            src.append(i)
            dst.append(j)
    return np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)


def _from_session_graph(graph) -> TraceView:
    steps, texts = [], []
    for s in graph.steps:
        steps.append({
            "idx": s.idx,
            "agent": s.agent,
            "kind": s.kind,
            "deps": [e.src_idx for e in s.deps],
            "exec_preds": None if s.exec_preds is None else list(s.exec_preds),
        })
        texts.append(s.node_attrs.get("text"))
    has_text = all(t is not None for t in texts) and bool(texts)
    return TraceView(steps=steps, texts=[str(t) for t in texts] if has_text else None,
                     graph=graph.to_networkx(), source="session_graph")


def _from_networkx(G) -> TraceView:
    nodes = sorted(((d["idx"], n, d) for n, d in G.nodes(data=True) if d.get("ntype") in _STEP_TYPES),
                   key=lambda t: t[0])
    name_to_idx = {n: idx for idx, n, _ in nodes}
    deps: Dict[int, List[int]] = {idx: [] for idx, _, _ in nodes}
    for u, v, d in G.edges(data=True):
        if d.get("etype") == "depends_on" and u in name_to_idx and v in name_to_idx:
            deps[name_to_idx[u]].append(name_to_idx[v])
    steps = [{"idx": idx, "agent": d.get("agent"), "kind": d.get("ntype"), "deps": sorted(deps[idx])}
             for idx, _, d in nodes]
    return TraceView(steps=steps, graph=G, source="networkx")


def _texts_from_steps(steps: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    if steps and all("text" in s for s in steps):
        return [str(s["text"]) for s in steps]
    return None


def normalize_trace(x: Any, *, dependency: str = "explicit") -> TraceView:
    """Return a :class:`TraceView` of ``x`` without mutating it.

    ``dependency="explicit"`` (the default) keeps the dependencies the trace records.
    ``dependency="full_context"`` makes every step depend on every earlier step and
    replaces any recorded ``deps``, the assumption corpora such as Who&When use; it is the
    same reading as ``build_graph(..., dependency="full_context")``.
    """
    if dependency not in ("explicit", "full_context"):
        raise ValueError("dependency must be 'explicit' or 'full_context'")
    if isinstance(x, TraceView):
        view = x
    elif hasattr(x, "steps") and hasattr(x, "to_networkx") and hasattr(x, "coverage"):
        view = _from_session_graph(x)
    elif isinstance(x, dict) and "steps" in x:
        steps = [dict(s) for s in x["steps"]]
        texts = x.get("texts")
        view = TraceView(steps=steps,
                         texts=[str(t) for t in texts] if texts is not None else _texts_from_steps(steps),
                         graph=x.get("G"), source="grade_run")
    elif isinstance(x, (list, tuple)) and (not x or isinstance(x[0], dict)):
        steps = [dict(s) for s in x]
        view = TraceView(steps=steps, texts=_texts_from_steps(steps), source="grade_steps")
    elif hasattr(x, "nodes") and hasattr(x, "edges"):
        view = _from_networkx(x)
    elif type(x).__module__.startswith("agentpile") and hasattr(x, "events"):
        raise TypeError("an agentpile Run is read through auditable.from_agentpile(runs, deps=...), which needs "
                        "the dependency rules to be named")
    else:
        raise TypeError(f"cannot read a trace from {type(x).__name__}")
    if dependency == "full_context":
        prior: List[int] = []
        filled = []
        for s in view.steps:
            s = dict(s)
            s["deps"] = list(prior)
            filled.append(s)
            prior.append(s["idx"])
        view = TraceView(steps=filled, texts=view.texts, graph=view.graph, source=view.source,
                         meta=dict(view.meta), shared_resource=view.shared_resource)
    return view
