"""Tests for execution_reach: transitive control-flow followers over handoff_to.

execution_reach is the twin of downstream_reach, but on the EXECUTION projection
(handoff_to, predecessor -> successor) rather than the dependency DAG. A node's
transitive control-flow FOLLOWERS are its descendants in that projection. These
tests pin the edge direction (followers, not ancestors), the linear / branch /
root / absent cases, and the contrast against downstream_reach that proves it
reads the execution layer, not depends_on.
"""
import pytest

pytest.importorskip("networkx")  # the projection needs the optional 'graph' extra

from auditable.graph import downstream_reach, execution_reach
from auditable.graph.session import DependencyEdge, Grade, SessionGraph, Step


def _linear(n):
    # a plain handoff chain 0 -> 1 -> ... -> n-1 (exec_preds None = linear default)
    return SessionGraph.from_steps(
        [Step(idx=i, agent="a", kind="decision") for i in range(n)]
    ).to_networkx()


def test_execution_reach_counts_transitive_followers_in_linear_chain():
    G = _linear(5)
    # step 0 leads the whole chain; every later step follows it transitively
    assert execution_reach(G, 0) == 4
    assert execution_reach(G, 1) == 3
    assert execution_reach(G, 3) == 1
    # the last step has no follower
    assert execution_reach(G, 4) == 0


def test_execution_reach_branch_and_merge_counts_merged_successor_once():
    # 0 -> 1, 0 -> 2, then 3 merges branches 1 and 2 (exec_preds=[1, 2]).
    steps = [
        Step(idx=0, agent="a", kind="decision"),
        Step(idx=1, agent="a", kind="decision", exec_preds=[0]),
        Step(idx=2, agent="b", kind="decision", exec_preds=[0]),
        Step(idx=3, agent="b", kind="decision", exec_preds=[1, 2]),
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    # 0 reaches 1, 2, and 3 -- the merged successor 3 is counted once, not twice
    assert execution_reach(G, 0) == 3
    assert execution_reach(G, 1) == 1  # only 3 follows branch 1
    assert execution_reach(G, 2) == 1  # only 3 follows branch 2
    assert execution_reach(G, 3) == 0


def test_execution_reach_explicit_root_is_not_a_follower_of_earlier_nodes():
    # step 2 is an explicit root (exec_preds=[]), so it has no incoming handoff and
    # is not counted among the followers of steps 0 / 1.
    steps = [
        Step(idx=0, agent="a", kind="decision"),
        Step(idx=1, agent="a", kind="decision"),            # linear from 0
        Step(idx=2, agent="b", kind="decision", exec_preds=[]),   # explicit root
        Step(idx=3, agent="b", kind="decision", exec_preds=[2]),  # follows the root
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    # 0 -> 1 only; the explicit root 2 and its follower 3 are a separate component
    assert execution_reach(G, 0) == 1
    assert execution_reach(G, 2) == 1  # only 3 follows the root
    assert execution_reach(G, 3) == 0


def test_execution_reach_absent_step_returns_zero():
    G = _linear(3)
    assert execution_reach(G, 99) == 0


def test_execution_reach_reads_execution_layer_not_dependency_layer():
    # A chain with control flow but NO data dependencies: every step is a root in the
    # dependency DAG, so downstream_reach == 0 everywhere, while execution_reach
    # follows the handoff chain. This proves the two read different projections.
    G = _linear(4)
    assert execution_reach(G, 0) == 3
    assert downstream_reach(G, 0) == 0

    # And the mirror: pure data deps with each step an explicit exec root (no handoff)
    steps = [
        Step(idx=0, agent="a", kind="decision", exec_preds=[]),
        Step(idx=1, agent="a", kind="decision", exec_preds=[],
             deps=[DependencyEdge(0, grade=Grade.OBSERVED)]),
        Step(idx=2, agent="a", kind="decision", exec_preds=[],
             deps=[DependencyEdge(1, grade=Grade.OBSERVED)]),
    ]
    Gd = SessionGraph.from_steps(steps).to_networkx()
    assert execution_reach(Gd, 0) == 0   # no handoff edges at all
    assert downstream_reach(Gd, 0) == 2  # but 1 and 2 transitively depend on 0
