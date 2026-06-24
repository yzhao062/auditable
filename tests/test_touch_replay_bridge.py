"""TDD for the capture -> replay bridge.

The 0.1.1 touch capture records WHICH resources a step read (observed edges, POST).
This bridge also records the relied-on VALUES and lowers a consequential step into a
replayable ``DecisionRecord``, so ``replay`` runs on a captured run, not a hand-built
dict. Behavioral only; no baseline comparison or detector logic lives here.
"""
from auditable import FixAction, replay
from auditable.graph.touch import TouchRecorder


def _budget_policy(state, action):
    amount = action.arguments["amount"]
    budget = state.get("budget", 0)
    return amount <= budget, f"amount {amount} vs budget {budget}"


def test_consequential_step_lowers_to_record_with_relied_on_snapshot():
    rec = TouchRecorder()
    with rec.step(agent="fetch", kind="tool_call") as st:
        st.writes("acct", "budget")
    with rec.step(agent="approver", kind="decision") as st:
        st.reads("acct", "budget", value=5000)
        st.decides("approve_payment", amount=4200)

    records = rec.to_records()

    assert len(records) == 1  # only the consequential (decides) step becomes a record
    r = records[0]
    assert r.action_type == "approve_payment"
    assert r.data.snapshot.state == {"budget": 5000}
    assert r.harness.action_type == "approve_payment"
    assert r.harness.arguments == {"amount": 4200}


def test_record_replays_allow_then_rollback_under_drift():
    rec = TouchRecorder()
    with rec.step(agent="approver", kind="decision") as st:
        st.reads("acct", "budget", value=5000)
        st.decides("approve_payment", amount=4200)
    r = rec.to_records()[0]

    ok = replay(r, live_state={"budget": 5000}, policy=_budget_policy)
    drift = replay(r, live_state={"budget": 100}, policy=_budget_policy)

    assert ok.action == FixAction.ALLOW
    assert drift.action == FixAction.ROLLBACK


def test_non_consequential_steps_produce_no_records():
    rec = TouchRecorder()
    with rec.step(agent="reader", kind="tool_call") as st:
        st.reads("acct", "budget", value=5000)  # read, but no decides
    assert rec.to_records() == []


def test_to_records_chains_and_signs_each_record():
    rec = TouchRecorder()
    with rec.step(agent="a", kind="decision") as st:
        st.reads("acct", "budget", value=5000)
        st.decides("approve_payment", amount=10)
    with rec.step(agent="b", kind="decision") as st:
        st.reads("acct", "budget", value=5000)
        st.decides("record_ledger", amount=10)
    records = rec.to_records()
    assert len(records) == 2
    assert all(r.record_id for r in records)  # signed
    assert records[1].prev_digest == records[0].record_id  # chained


def test_snapshot_owns_a_copy_immune_to_later_source_mutation():
    rec = TouchRecorder()
    items = ["a"]
    with rec.step(agent="d", kind="decision") as st:
        st.reads("acct", "items", value=items)
        st.decides("act")
    r = rec.to_records()[0]
    items.append("b")  # mutate the source after capture; the snapshot must not change
    assert r.data.snapshot.state["items"] == ["a"]
