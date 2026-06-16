"""Reference action gate and an in-process rail.

A routed verdict is only observability until something acts on it. The gate sits
before a consequential write and executes the ``FixAction`` through a rail. The rail
is the boundary auditable plugs into; the reference implementation here is a mock
in-process ledger so a rollback runs end to end in the demo and tests. Real rails
(a payment processor, a record store, an internal ledger) implement the same two
methods, ``commit`` and ``compensate``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from .core import Action, FixAction, Verdict


class Rail(Protocol):
    """Anything a consequential action commits through: one method to do, one to undo."""

    def commit(self, action: Action) -> str:
        ...

    def compensate(self, receipt: str) -> None:
        ...


class ReferenceLedger:
    """In-process reference rail. ``commit`` spends; ``compensate`` refunds.

    Stands in for a real payment rail or record store so the gate can execute a
    rollback in the demo and tests. Not for production use.
    """

    def __init__(self, balance: float = 0.0) -> None:
        self.balance = balance
        self._open: Dict[str, float] = {}
        self._count = 0

    def commit(self, action: Action) -> str:
        self.balance -= action.cost
        self._count += 1
        receipt = f"rcpt-{self._count}"
        self._open[receipt] = action.cost
        return receipt

    def compensate(self, receipt: str) -> None:
        amount = self._open.pop(receipt)
        self.balance += amount


@dataclass
class GateOutcome:
    """The executed result of routing a verdict through the gate."""

    fix: FixAction
    executed: str  # "committed" | "blocked" | "held" | "rolled_back"
    detail: str


class ActionGate:
    """Maps a replay verdict to an executed fix through a rail.

    Usage: commit the agent's action through the gate and keep the receipt, then after
    a replay verdict call ``enforce`` to allow, block, hold, or roll back. Rollback runs
    the rail's ``compensate`` so the action is actually undone, not just flagged. This
    is the difference between recovery infrastructure and a logger.
    """

    def __init__(self, rail: Rail) -> None:
        self.rail = rail

    def commit(self, action: Action) -> str:
        """Run the agent's consequential action through the rail; return its receipt."""
        return self.rail.commit(action)

    def enforce(self, verdict: Verdict, *, receipt: Optional[str] = None) -> GateOutcome:
        """Execute the fix the replay verdict routed.

        ``receipt`` is required to roll back an already-committed action; without it a
        rollback degrades to a block (nothing was committed yet).
        """
        if verdict.action == FixAction.ALLOW:
            return GateOutcome(FixAction.ALLOW, "committed", "Action stands under live state.")
        if verdict.action == FixAction.ROLLBACK:
            if receipt is None:
                return GateOutcome(FixAction.BLOCK, "blocked", "Not yet committed; blocked.")
            self.rail.compensate(receipt)
            return GateOutcome(FixAction.ROLLBACK, "rolled_back", verdict.reason)
        if verdict.action == FixAction.HUMAN_REVIEW:
            return GateOutcome(FixAction.HUMAN_REVIEW, "held", verdict.reason)
        return GateOutcome(FixAction.BLOCK, "blocked", verdict.reason)
