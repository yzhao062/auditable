"""Own-record adapter: auditable's signed DecisionRecords to typed Steps.

A run captured through ``audit(...)`` (a ``MemorySink``) or loaded back offline
(``auditable.graph.loader.load_log``) is a chain of signed ``DecisionRecord``s.
This adapter maps that chain into the typed ``Step`` list the SessionGraph and
the structural scorer consume, so an auditable run and a public corpus share one
representation.

What it sets, and what it deliberately does not:

- **Execution edges (harness)** come from the ``prev_digest`` backbone: each step
  carries explicit ``exec_preds`` pointing at the record it chained from (an empty
  list for the genesis record). This is the observed control-flow order.
- **Model attributes** ride on each node: ``model_id`` and ``decision_basis`` from
  the record's model span (plus ``action_type`` from the harness span).
- **Identity** is carried through: the sealed ``record_id`` becomes the step's
  ``record_id``, and offline it also serves as the ``correlation_id`` (a persisted
  record has no separate pre-digest live event id, and fragments do not arrive out
  of order offline, so the sealed digest is the correlation key).
- **Dependency edges (data)** stay sparse and ``DECLARED``. Without the v0.3b
  runtime resource-touch contract, the run does not log which prior write a read
  matched, so this adapter does not fabricate observed edges. With
  ``link_sequential`` set (the default), it declares one edge from each non-genesis
  record to its immediate predecessor, graded ``DECLARED`` with evidence saying so;
  with it cleared, the dependency layer is empty. Either way the dependency layer
  is not observed, and ``structural_risk`` reports the run as low coverage rather
  than presenting this declared structure as a calibrated signal.

Pure Python: it duck-types the record fields (``getattr``), so a real
``DecisionRecord``, a loaded record, and a lightweight stub all work, with no
network and no torch / networkx import.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from ..session import DependencyEdge, Grade, Step
from .protocol import _BaseAdapter

if TYPE_CHECKING:  # annotation only; to_steps duck-types so there is no runtime import
    from ...record import DecisionRecord

__all__ = ["OwnRecordAdapter", "own_record_v1"]


class OwnRecordAdapter(_BaseAdapter):
    """Map a chain of ``DecisionRecord``s to typed steps (execution + model facet).

    ``agent_label`` is the single actor's label under the homogeneous-model
    assumption (the model identity rides on each node as ``model_id``, not as the
    agent). ``link_sequential`` toggles whether each non-genesis record declares a
    dependency on its immediate predecessor; both settings keep the dependency
    layer non-observed and low coverage this round.
    """

    name = "own_record"
    version = "v1"

    def __init__(self, *, agent_label: str = "agent", link_sequential: bool = True) -> None:
        self.agent_label = agent_label
        self.link_sequential = link_sequential

    def to_steps(self, records: "Sequence[DecisionRecord]") -> List[Step]:
        """Build one decision step per record, in log order.

        Execution predecessors come from the ``prev_digest`` chain; dependency
        edges are sparse and ``DECLARED`` (see the module docstring). Records are
        read by duck-typing, so loaded, live, or stubbed records all work."""
        steps: List[Step] = []
        id_to_idx: Dict[str, int] = {}
        for i, rec in enumerate(records or ()):
            model = getattr(rec, "model", None)
            model_id = getattr(model, "model_id", "") or ""
            decision_basis = getattr(model, "decision_basis", "") or ""
            action_type = getattr(rec, "action_type", "") or ""
            record_id = getattr(rec, "record_id", "") or ""
            prev_digest = getattr(rec, "prev_digest", None)

            exec_preds = self._exec_preds(prev_digest, id_to_idx, i)
            deps = self._declared_deps(exec_preds)

            steps.append(
                Step(
                    idx=i,
                    agent=self.agent_label,
                    kind="decision",
                    deps=deps,
                    node_attrs={
                        "model_id": model_id,
                        "decision_basis": decision_basis,
                        "action_type": action_type,
                    },
                    exec_preds=exec_preds,
                    correlation_id=record_id or None,
                    record_id=record_id or None,
                )
            )
            if record_id:
                id_to_idx[record_id] = i
        return steps

    @staticmethod
    def _exec_preds(prev_digest: Any, id_to_idx: Dict[str, int], i: int) -> List[int]:
        """Execution predecessors from the prev_digest backbone.

        Genesis (``prev_digest is None``) has no execution predecessor. Otherwise
        link to the record the digest names; if it does not resolve (an unsigned or
        out-of-order batch), fall back to the immediate prior position so the
        backbone stays connected without inventing a non-adjacent link."""
        if prev_digest is None:
            return []
        if prev_digest in id_to_idx:
            return [id_to_idx[prev_digest]]
        return [i - 1] if i > 0 else []

    def _declared_deps(self, exec_preds: List[int]) -> List[DependencyEdge]:
        """Sparse, DECLARED dependency edges: one to the immediate predecessor, or
        none. Never OBSERVED, because the resource-touch contract (v0.3b) is what
        turns a logged read/write match into an observed edge."""
        if not self.link_sequential or not exec_preds:
            return []
        return [
            DependencyEdge(
                src_idx=exec_preds[0],
                grade=Grade.DECLARED,
                resource=None,
                evidence={
                    "declared": True,
                    "observed": False,
                    "basis": "sequential record adjacency from the prev_digest backbone",
                    "note": (
                        "declared placeholder pending the v0.3b resource-touch contract; "
                        "not an observed read or write match"
                    ),
                    "adapter": self.id,
                },
            )
        ]


# The versioned singleton, callable as own_record_v1(records).
own_record_v1 = OwnRecordAdapter()
