"""Capture, replay, and the verdict the operator can route.

- ``audit(...)``   captures one decision (data / model / harness) with the dependency
                   snapshot it relied on, signs it, and appends it to a sink.
- ``replay(...)``  re-derives whether a recorded decision still holds under the live
                   dependency state. Pure: it returns a verdict and executes nothing.
- ``Verdict`` / ``FixAction``   the routed fix: allow, block, human_review, rollback.

A ``Policy`` is any callable ``(state, action) -> (justified, reason)``. A policy that
cannot decide raises ``ReplayUndecidable`` and replay returns ``HUMAN_REVIEW``. Sinks
(``MemorySink``, ``FileSink``) sign and chain records.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, List, Optional, Tuple

from .record import (
    Action,
    DecisionRecord,
    DependencySnapshot,
    HarnessSpan,
    ModelSpan,
    Report,
    _canonical_json,
)

Policy = Callable[[dict, Action], Tuple[bool, str]]


class ReplayUndecidable(Exception):
    """Raised by a policy that cannot re-decide deterministically under a given state."""


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
    """Handle yielded by ``audit``. Fill in inputs, model, action, and attach reports."""

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

    def attach(self, report: Report) -> "Decision":
        """Route a leaf report to its stage span."""
        if report.stage == "data":
            self._record.data.report = report
        elif report.stage == "model":
            self._record.model.report = report
        elif report.stage == "harness":
            self._record.harness.report = report
        else:
            raise ValueError(f"Unknown report stage: {report.stage!r}")
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


class FileSink:
    """Append-only JSONL sink: one signed record per line, durable across process exit.

    A second concrete sink alongside ``MemorySink`` so the signed log survives the process
    and the pluggable-sink abstraction has more than one implementation.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._last: Optional[str] = self._read_last_digest(path) if os.path.exists(path) else None

    @staticmethod
    def _read_last_digest(path: str) -> Optional[str]:
        """Return the last record's id, failing closed on a corrupt tail.

        Blank lines and an empty file are fine. A corrupt, non-object, or
        ``record_id``-less line raises rather than silently restarting the chain, which
        would otherwise let a damaged tail fork an append-only tamper-evident log.
        """
        last_digest = None
        with open(path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Cannot resume FileSink from corrupt JSONL line {lineno} in {path}."
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Cannot resume FileSink from non-object JSONL line {lineno} in {path}."
                    )
                record_id = row.get("record_id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(
                        f"Cannot resume FileSink from line {lineno} without record_id in {path}."
                    )
                last_digest = record_id
        return last_digest

    def append(self, record: DecisionRecord) -> None:
        record.prev_digest = self._last
        record.record_id = record.digest()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(_canonical_json(record.to_dict()) + "\n")
        self._last = record.record_id


default_sink = MemorySink()


@contextmanager
def audit(
    action_type: str,
    *,
    snapshot: DependencySnapshot,
    sink: Optional[Any] = None,
) -> Iterator[Decision]:
    """Capture one agent decision with the dependency snapshot it relied on."""
    sink = sink if sink is not None else default_sink
    decision = Decision(action_type=action_type, snapshot=snapshot)
    try:
        yield decision
    finally:
        if not decision.record.decided_at:
            decision.record.decided_at = time.time()
        sink.append(decision.record)


def replay(record: DecisionRecord, *, live_state: dict, policy: Policy) -> Verdict:
    """Re-derive whether the decision still holds under the live dependency state.

    Pure: returns a verdict, executes nothing. The agent acted under
    ``record.data.snapshot.state``; we re-evaluate the same action against ``live_state``.
    If the action was justified on the snapshot but not on live state, it relied on stale
    or drifted state and we route a ``ROLLBACK``. A policy that cannot decide raises
    ``ReplayUndecidable`` and we return ``HUMAN_REVIEW``.
    """
    def action_for_policy() -> Action:
        # Hand the policy a deep copy so a mutating policy cannot alter the signed record.
        return Action(
            type=record.harness.action_type,
            arguments=deepcopy(record.harness.arguments),
            cost=record.harness.cost,
        )

    try:
        justified_live, reason = policy(deepcopy(live_state), action_for_policy())
    except ReplayUndecidable as exc:
        return Verdict(FixAction.HUMAN_REVIEW, False, str(exc), record.record_id)
    if justified_live:
        return Verdict(FixAction.ALLOW, True, "Justified under live state.", record.record_id)

    try:
        justified_snapshot, _ = policy(
            deepcopy(record.data.snapshot.state), action_for_policy()
        )
    except ReplayUndecidable as exc:
        return Verdict(FixAction.HUMAN_REVIEW, False, str(exc), record.record_id)
    if justified_snapshot:
        return Verdict(
            FixAction.ROLLBACK,
            False,
            f"Decision relied on stale dependency state: {reason}",
            record.record_id,
        )
    return Verdict(FixAction.BLOCK, False, reason, record.record_id)
