from auditable import (
    Action,
    DependencySnapshot,
    FixAction,
    ReplayUndecidable,
    audit,
    replay,
)


def budget_policy(state, action):
    ok = action.cost <= state.get("budget", 0)
    return ok, ("within budget" if ok else "over budget")


def _record(cost, snapshot_state):
    snap = DependencySnapshot(state=snapshot_state)
    with audit("pay", snapshot=snap) as d:
        d.act(Action("pay", {"to": "x"}, cost=cost))
    return d.record


def test_replay_blocks_when_unjustified_on_both():
    record = _record(cost=80, snapshot_state={"budget": 50})  # over even the snapshot budget
    verdict = replay(record, live_state={"budget": 50}, policy=budget_policy)
    assert verdict.action == FixAction.BLOCK


def test_replay_human_review_on_undecidable_policy():
    record = _record(cost=10, snapshot_state={"budget": 100})

    def undecidable(state, action):
        raise ReplayUndecidable("live state incomplete")

    verdict = replay(record, live_state={}, policy=undecidable)
    assert verdict.action == FixAction.HUMAN_REVIEW


def test_replay_is_pure_does_not_mutate_record_or_live_state():
    record = _record(cost=80, snapshot_state={"budget": 100})
    before = record.digest()
    live_state = {"budget": 50}

    def mutating_policy(state, action):
        # Decide first, then mutate the inputs it was handed. If replay passed the real
        # record and live_state instead of copies, these mutations would corrupt them.
        ok = action.cost <= state.get("budget", 0)
        state["budget"] = -999
        action.arguments["to"] = "attacker"
        return ok, "mutated"

    verdict = replay(record, live_state=live_state, policy=mutating_policy)
    assert verdict.action == FixAction.ROLLBACK
    assert record.digest() == before  # the signed record is untouched
    assert record.harness.arguments == {"to": "x"}
    assert record.data.snapshot.state == {"budget": 100}
    assert live_state == {"budget": 50}  # the caller's live_state is untouched
