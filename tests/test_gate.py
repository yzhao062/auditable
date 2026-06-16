from auditable import (
    Action,
    ActionGate,
    DependencySnapshot,
    FixAction,
    ReferenceLedger,
    Verdict,
    audit,
    replay,
)


def test_ledger_commit_and_compensate():
    ledger = ReferenceLedger(balance=100)
    receipt = ledger.commit(Action("pay", {}, cost=40))
    assert ledger.balance == 60
    ledger.compensate(receipt)
    assert ledger.balance == 100


def test_gate_rollback_executes_compensate():
    ledger = ReferenceLedger(balance=100)
    gate = ActionGate(ledger)
    receipt = gate.commit(Action("pay", {}, cost=40))
    assert ledger.balance == 60
    outcome = gate.enforce(Verdict(FixAction.ROLLBACK, False, "stale", "rec1"), receipt=receipt)
    assert outcome.executed == "rolled_back"
    assert ledger.balance == 100  # money moved back


def test_gate_allow_keeps_action():
    ledger = ReferenceLedger(balance=100)
    gate = ActionGate(ledger)
    gate.commit(Action("pay", {}, cost=40))
    outcome = gate.enforce(Verdict(FixAction.ALLOW, True, "ok", "rec1"))
    assert outcome.executed == "committed"
    assert ledger.balance == 60


def test_gate_rollback_without_receipt_blocks():
    gate = ActionGate(ReferenceLedger(balance=100))
    outcome = gate.enforce(Verdict(FixAction.ROLLBACK, False, "stale", "rec1"))
    assert outcome.fix == FixAction.BLOCK
    assert outcome.executed == "blocked"


def test_replay_is_action_agnostic_record_edit():
    # An agent marks an order shipped based on a stale "paid" status.
    snap = DependencySnapshot(state={"payment_status": "paid"})

    def ship_policy(state, action):
        ok = state.get("payment_status") == "paid"
        return ok, ("paid" if ok else "not paid; cannot ship")

    with audit("set_shipped", snapshot=snap) as d:
        d.act(Action("set_shipped", {"order": "O-1"}, cost=0.0))
    # The live status reverted to refunded before the edit took effect.
    verdict = replay(d.record, live_state={"payment_status": "refunded"}, policy=ship_policy)
    assert verdict.action == FixAction.ROLLBACK
