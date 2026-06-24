"""TDD for the LangGraph capture -> replay bridge (the wrap-once path).

`instrument` already yields OBSERVED edges for POST. This adds the relied-on VALUES,
so `builder.to_records(decisions=...)` lowers a marked decision node of a real
LangGraph run into a replayable `DecisionRecord`, and `replay` re-decides it under
drifted live state. Behavioral only.
"""
from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402

from auditable import FixAction, replay  # noqa: E402
from auditable.integrations.langgraph import instrument  # noqa: E402


class PayState(TypedDict):
    budget: int
    amount: int
    approved: bool


def _policy(state, action):
    return state["amount"] <= state["budget"], "amount vs budget"


def _build():
    builder = instrument(StateGraph(PayState))

    def fetch_budget(state):
        return {"budget": 5000}

    def approve(state):
        return {"approved": state["amount"] <= state["budget"]}

    builder.add_node("fetch_budget", fetch_budget)
    builder.add_node("approve", approve)
    builder.add_edge(START, "fetch_budget")
    builder.add_edge("fetch_budget", "approve")
    builder.add_edge("approve", END)
    return builder


def test_to_records_snapshots_relied_on_values_of_the_decision_node():
    builder = _build()
    builder.compile().invoke({"budget": 0, "amount": 4200, "approved": False})

    records = builder.to_records(decisions={"approve": "approve_payment"})

    assert len(records) == 1  # only the marked decision node
    r = records[0]
    assert r.harness.action_type == "approve_payment"
    # approve read both channels; the relied-on budget is the post-fetch value (5000)
    assert r.data.snapshot.state["budget"] == 5000
    assert r.data.snapshot.state["amount"] == 4200
    assert r.harness.arguments == {"approved": True}  # the node's write is the action arg


def test_captured_langgraph_decision_replays_rollback_under_drift():
    builder = _build()
    builder.compile().invoke({"budget": 0, "amount": 4200, "approved": False})
    r = builder.to_records(decisions=["approve"])[0]

    ok = replay(r, live_state={"budget": 5000, "amount": 4200}, policy=_policy)
    drift = replay(r, live_state={"budget": 100, "amount": 4200}, policy=_policy)

    assert ok.action == FixAction.ALLOW
    assert drift.action == FixAction.ROLLBACK


def test_decisions_list_uses_node_name_as_action_type():
    builder = _build()
    builder.compile().invoke({"budget": 0, "amount": 4200, "approved": False})
    r = builder.to_records(decisions=["approve"])[0]
    assert r.harness.action_type == "approve"


def test_unknown_decision_node_warns():
    builder = _build()
    builder.compile().invoke({"budget": 0, "amount": 4200, "approved": False})
    with pytest.warns(UserWarning, match="not in the captured run"):
        builder.to_records(decisions=["nonexistent_node"])


class _ListState(TypedDict):
    items: list
    done: bool


def test_snapshot_value_is_immune_to_later_in_place_mutation():
    builder = instrument(StateGraph(_ListState))
    seen = {}

    def decide(state):
        seen["obj"] = state["items"]  # the live list the node read
        return {"done": len(state["items"]) >= 1}

    builder.add_node("decide", decide)
    builder.add_edge(START, "decide")
    builder.add_edge("decide", END)
    builder.compile().invoke({"items": ["a"], "done": False})

    r = builder.to_records(decisions=["decide"])[0]
    seen["obj"].append("b")  # mutate the read object after capture; snapshot must not change
    assert r.data.snapshot.state["items"] == ["a"]


def test_to_records_action_cost_from_captured_field_drives_cost_policy():
    builder = _build()
    builder.compile().invoke({"budget": 0, "amount": 4200, "approved": False})

    r = builder.to_records(decisions={"approve": "vendor_payment"}, action_costs={"approve": "amount"})[0]
    assert r.harness.cost == 4200

    def cost_policy(state, action):
        return action.cost <= state.get("budget", 0), "cost vs budget"

    ok = replay(r, live_state={"budget": 5000}, policy=cost_policy)
    drift = replay(r, live_state={"budget": 3000}, policy=cost_policy)
    assert ok.action == FixAction.ALLOW
    assert drift.action == FixAction.ROLLBACK


def test_to_records_action_args_map_captured_fields():
    builder = _build()
    builder.compile().invoke({"budget": 0, "amount": 4200, "approved": False})

    r = builder.to_records(decisions=["approve"], action_args={"approve": {"limit": "budget"}})[0]
    assert r.harness.arguments == {"limit": 5000}
