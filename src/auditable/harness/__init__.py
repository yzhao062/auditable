"""The harness module: audit the agent's action (static rules) and execute control.

v0.1 ships ``HarnessAuditor`` with one static rule (a cost cap, the kind of forward
static check that exists today) and the concrete ``ActionGate`` control surface. Replay
under live state (in ``chain``) is layered on top of the static rule. Later versions add
agent-audit-style OWASP-Agentic and CWE checks, consumed rather than forked, and a
``HarnessController`` subclassing the v0.4 ``Controller`` base.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # Protocol is stdlib on 3.8+; guard for safety.
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

from ..chain import FixAction, Verdict
from ..record import Action, Auditor, Report


class HarnessAuditor(Auditor):
    """A thin static rule on the action: flag spend over a static cap (v0.1).

    The static cap is the forward, point-in-time check that incumbents already run. Its
    report is the harness audit leaf; replay-under-live-state (in ``chain``) is layered on
    top of it. The score ramps from 0 at the cap to 1 at twice the cap.
    """

    stage = "harness"

    def __init__(self, *, cost_cap: Optional[float] = None, name: str = "static-cost-cap"):
        self.cost_cap = cost_cap
        self.name = name

    def assess(self, subject: Action) -> Report:
        cap = self.cost_cap
        if cap is None or cap <= 0:
            return Report(self.stage, self.name, 0.0, "ok", "No static cap configured.")
        over = subject.cost > cap
        score = min(1.0, (subject.cost - cap) / cap) if over else 0.0
        reason = (
            f"Action cost {subject.cost:,.0f} exceeds the static cap {cap:,.0f}."
            if over
            else f"Action cost {subject.cost:,.0f} within the static cap {cap:,.0f}."
        )
        return Report(
            self.stage, self.name, round(score, 3),
            "over_cap" if over else "ok", reason,
            {"cost": subject.cost, "cost_cap": cap},
        )


class Rail(Protocol):
    """Any commit/compensate backend (a payment rail, a record store, a ledger)."""

    def commit(self, action: Action):
        ...

    def compensate(self, receipt) -> None:
        ...


@dataclass
class GateOutcome:
    fix: FixAction
    executed: str
    detail: str


class ReferenceLedger:
    """In-process reference rail: commit spends, compensate refunds. For demo and tests."""

    def __init__(self, balance: float = 0.0):
        self.balance = balance
        self._open: dict = {}
        self._count = 0

    def commit(self, action: Action):
        self.balance -= action.cost
        self._count += 1
        receipt = f"rcpt-{self._count}"
        self._open[receipt] = action.cost
        return receipt

    def compensate(self, receipt) -> None:
        amount = self._open.pop(receipt)
        self.balance += amount


class ActionGate:
    """The concrete v0.1 control surface. Maps a replay verdict to an executed fix.

    Side-effect timing is explicit. ``enforce_pre_commit`` runs before the action (allow,
    block, or hold). ``enforce_post_commit`` runs after the action committed through the
    rail (allow, hold, or roll back via ``rail.compensate``). A routed verdict that cannot
    execute a compensating action is observability, not control.
    """

    def __init__(self, rail: Rail):
        self.rail = rail

    def commit(self, action: Action):
        return self.rail.commit(action)

    def enforce_pre_commit(self, verdict: Verdict, action: Action) -> GateOutcome:
        if verdict.action == FixAction.ALLOW:
            return GateOutcome(FixAction.ALLOW, "allowed", "Action allowed under live state.")
        if verdict.action == FixAction.HUMAN_REVIEW:
            return GateOutcome(FixAction.HUMAN_REVIEW, "held", verdict.reason)
        # ROLLBACK or BLOCK before the action runs: do not run it.
        return GateOutcome(FixAction.BLOCK, "blocked", verdict.reason)

    def enforce_post_commit(self, verdict: Verdict, *, receipt=None) -> GateOutcome:
        if verdict.action == FixAction.ALLOW:
            return GateOutcome(FixAction.ALLOW, "committed", "Action stands under live state.")
        if verdict.action == FixAction.HUMAN_REVIEW:
            return GateOutcome(FixAction.HUMAN_REVIEW, "held", verdict.reason)
        # ROLLBACK or BLOCK after commit: the action already ran, so reverse it via the
        # rail. Preserve which verdict drove the compensation (ROLLBACK = stale state;
        # BLOCK = invalid even under the snapshot). Post-commit there is nothing to block.
        if receipt is None:
            return GateOutcome(
                verdict.action,
                "compensation_unavailable",
                "Action already committed; no receipt to compensate.",
            )
        self.rail.compensate(receipt)
        executed = "rolled_back" if verdict.action == FixAction.ROLLBACK else "reversed"
        return GateOutcome(verdict.action, executed, verdict.reason)
