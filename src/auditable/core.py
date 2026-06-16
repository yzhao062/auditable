"""Capture, replay, and route a fix for a single agent decision.

The public surface is small:

- ``audit(...)``   a context manager that captures one decision (data/model/harness)
                   with the dependency snapshot it relied on, signs it, and appends it
                   to a sink.
- ``replay(...)``  re-derives whether a recorded decision is still justified under the
                   *live* dependency state, versus the snapshot the agent actually used.
                   A decision that passed on a stale snapshot but fails on live state is
                   the category that auditable exists to catch.
- ``Verdict`` / ``FixAction``   the fix the operator can route: allow, block,
                   human_review, rollback, each with the reason and the record it
                   points to.

A ``Policy`` is any callable ``(snapshot_state, action) -> (justified, reason)``.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, List, Optional, Tuple

from .record import DecisionRecord, DependencySnapshot, HarnessSpan, ModelSpan


@dataclass
class Action:
    """What the agent is about to do."""

    type: str
    arguments: dict
    cost: float = 0.0


Policy = Callable[[dict, Action], Tuple[bool, str]]


class FixAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    HUMAN_REVIEW = "human_review"
    ROLLBACK = "rollback"


@dataclass
class Verdict:
    action: FixAction
    justified: bool
    reason: str
    record_id: str = ""


class Decision:
    """Handle yielded by ``audit``. Fill in the model output and the executed action."""

    def __init__(self, action_type: str, snapshot: DependencySnapshot):
        self._record = DecisionRecord(action_type=action_type)
        self._record.data.snapshot = snapshot

    def read(self, **inputs: Any) -> "Decision":
        self._record.data.inputs.update(inputs)
        return self

    def model(self, model_id: str, decision_basis: str = "", output: Any = None) -> "Decision":
        self._record.model = ModelSpan(
            model_id=model_id, decision_basis=decision_basis, output=output
        )
        return self

    def act(self, action: Action) -> "Decision":
        self._record.harness = HarnessSpan(
            action_type=action.type, arguments=action.arguments, cost=action.cost
        )
        return self

    @property
    def record(self) -> DecisionRecord:
        return self._record


class MemorySink:
    """Default in-process sink. Signs each record and chains it to the previous one."""

    def __init__(self) -> None:
        self.records: List[DecisionRecord] = []

    def append(self, record: DecisionRecord) -> None:
        record.prev_digest = self.records[-1].record_id if self.records else None
        record.record_id = record.digest()
        self.records.append(record)


default_sink = MemorySink()


@contextmanager
def audit(
    action_type: str,
    *,
    snapshot: DependencySnapshot,
    sink: Optional[MemorySink] = None,
) -> Iterator[Decision]:
    """Capture one agent decision with the dependency snapshot it relied on."""
    sink = sink or default_sink
    decision = Decision(action_type=action_type, snapshot=snapshot)
    try:
        yield decision
    finally:
        if not decision.record.decided_at:
            decision.record.decided_at = time.time()
        sink.append(decision.record)


def replay(record: DecisionRecord, *, live_state: dict, policy: Policy) -> Verdict:
    """Re-derive whether the decision still holds under the live dependency state.

    The agent acted under ``record.data.snapshot.state``. We re-evaluate the same
    action against ``live_state``. If the action was justified on the snapshot but is
    not on the live state, the decision relied on stale or drifted state, and we route
    a fix.
    """
    action = Action(
        type=record.harness.action_type,
        arguments=record.harness.arguments,
        cost=record.harness.cost,
    )
    justified_live, reason = policy(live_state, action)
    if justified_live:
        return Verdict(FixAction.ALLOW, True, "Justified under live state.", record.record_id)

    justified_snapshot, _ = policy(record.data.snapshot.state, action)
    if justified_snapshot:
        return Verdict(
            FixAction.ROLLBACK,
            False,
            f"Decision relied on stale dependency state: {reason}",
            record.record_id,
        )
    return Verdict(FixAction.BLOCK, False, reason, record.record_id)
