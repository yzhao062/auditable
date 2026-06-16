from auditable import Action, DependencySnapshot, FixAction, audit, replay


def budget_policy(state, action):
    ok = action.cost <= state.get("budget", 0)
    return ok, ("within budget" if ok else "over budget")


def test_capture_signs_and_chains():
    snap = DependencySnapshot(state={"budget": 100})
    with audit("pay", snapshot=snap) as d:
        d.act(Action("pay", {"to": "x"}, cost=50))
    record = d.record
    assert record.record_id, "record should be signed with a content digest"
    assert record.action_type == "pay"


def test_replay_flags_stale_state():
    snap = DependencySnapshot(state={"budget": 100})
    with audit("pay", snapshot=snap) as d:
        d.act(Action("pay", {"to": "x"}, cost=80))
    # Live budget dropped below the amount; the snapshot allowed it.
    verdict = replay(d.record, live_state={"budget": 50}, policy=budget_policy)
    assert verdict.justified is False
    assert verdict.action == FixAction.ROLLBACK


def test_replay_allows_when_still_justified():
    snap = DependencySnapshot(state={"budget": 100})
    with audit("pay", snapshot=snap) as d:
        d.act(Action("pay", {"to": "x"}, cost=40))
    verdict = replay(d.record, live_state={"budget": 100}, policy=budget_policy)
    assert verdict.justified is True
    assert verdict.action == FixAction.ALLOW
