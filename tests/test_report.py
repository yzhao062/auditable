"""Tests for auditable.report: the dependency-free Markdown renderer.

The renderer turns the EXISTING typed fields of the PRE report (``PreReport`` from
``analyze_plan``) and the POST report (``AnalysisReport`` from ``analyze_run``) into
Markdown. It computes nothing new, so these tests check formatting and the honest
boundaries the plaintext ``summary`` already holds:

- both renderers emit the five labeled parts (lifecycle stage, what is risky on the
  graph, the keystone, the per-finding detail, a "what to do" line);
- PRE keeps its execution-topology chokestone wording; POST keeps its blast-radius
  wording; the two keystone concepts stay distinct;
- a withheld score renders ``n/a``, never ``0.000``;
- the state-driven POST action covers all three states;
- ``render_report`` dispatches by type and rejects anything else;
- ``report.to_markdown()`` returns the same string as the top-level function;
- the additive Markdown surface does NOT change the terse plaintext ``summary`` /
  ``__str__`` (no ``#`` headings leak into the plaintext form).
"""
import pytest

pytest.importorskip("networkx")  # both reports build on the NetworkX projection

import auditable
from auditable import render_report
from auditable.analysis import AnalysisReport, analyze_run
from auditable.graph.adapters import (
    declared_plan_v1,
    own_record_v1,
    tau_bench_prior_db_reads_v1,
)
from auditable.graph.pre import PreReport, analyze_plan
from auditable.report import post_to_markdown, pre_to_markdown


# --- fixtures (mirror the example plans / runs) ------------------------------


def _full_plan():
    """A small declared plan that plants one of each lint and a clear keystone."""
    return {
        "nodes": [
            {"idx": 0, "agent": "kyc_tool", "kind": "tool_call", "writes": ["kyc.tier"]},
            {"idx": 1, "agent": "planner", "kind": "decision",
             "reads": [{"id": "kyc.tier", "producer": 0, "volatile": True}],
             "scope": ["kyc.tier", "order.x"]},
            {"idx": 2, "agent": "exec", "kind": "tool_call",
             "reads": [{"id": "kyc.tier", "producer": 0}], "writes": ["order.x"]},
        ]
    }


def _tau_messages():
    """A tau-bench-style retail run with a scored keystone at the first read."""
    return [
        {"role": "system", "content": "you are a retail agent; follow policy"},
        {"role": "user", "content": "Modify pending order #W512 and refund the difference."},
        {"role": "assistant", "content": "Pulling up the order.",
         "tool_calls": [{"function": {"name": "get_order_details"}}]},
        {"role": "tool", "name": "get_order_details",
         "content": '{"order_id": "#W512", "status": "pending"}'},
        {"role": "assistant", "content": "Updating the item.",
         "tool_calls": [{"function": {"name": "modify_pending_order_items"}}]},
        {"role": "tool", "name": "modify_pending_order_items", "content": "ok"},
        {"role": "assistant", "content": "Checking your account.",
         "tool_calls": [{"function": {"name": "get_user_details"}}]},
        {"role": "tool", "name": "get_user_details",
         "content": '{"user_id": "sara_doe_496"}'},
        {"role": "assistant", "content": "Issuing the refund certificate.",
         "tool_calls": [{"function": {"name": "send_certificate"}}]},
        {"role": "tool", "name": "send_certificate", "content": "gc-001"},
    ]


def _own_records(n=3):
    from auditable import Action, DependencySnapshot, MemorySink, audit

    sink = MemorySink()
    for i in range(n):
        snap = DependencySnapshot(state={"budget_remaining": 5000, "policy_id": "kyc-2026-03"})
        with audit(f"pay-{i}", snapshot=snap, sink=sink) as d:
            d.read(invoice=f"INV-{100 + i}", vendor="acme")
            d.model(
                "gpt-x",
                decision_basis=(
                    f"Invoice INV-{100 + i} from acme is within the 5000 budget "
                    "under policy kyc-2026-03."
                ),
            )
            d.act(Action(f"pay-{i}", {"to": "acme"}, cost=100.0))
    return sink.records


# --- public surface ----------------------------------------------------------


def test_render_report_is_a_top_level_export():
    assert "render_report" in auditable.__all__
    assert auditable.render_report is render_report


# --- PRE markdown ------------------------------------------------------------


def test_pre_markdown_has_the_stage_banner_and_meta():
    md = analyze_plan(_full_plan()).to_markdown()
    assert md.startswith("# Auditable PRE Report: Declared-Plan Analysis")
    assert "stage: PRE" in md
    assert "- adapter: declared_plan_v1" in md
    assert "- steps: 3" in md


def test_pre_markdown_keystone_is_chokepoint_not_blast_radius():
    md = analyze_plan(_full_plan()).to_markdown()
    assert "## Keystone (Execution-Topology Chokepoint)" in md
    # the planted keystone is the linear lead node 0 with 2 followers
    assert "step 0: 2 of 2 other steps transitively follow it in control flow" in md
    assert "not the POST blast-radius keystone" in md
    # POST-only wording must NOT appear in a PRE render
    assert "Blast-Radius" not in md


def test_pre_markdown_findings_table_lists_every_planted_lint():
    md = analyze_plan(_full_plan()).to_markdown()
    assert "## What Is Risky on the Graph" in md
    assert "| severity | lint | step | resource | detail |" in md
    for lint in (
        "write_with_no_prior_read",
        "flippable_dependency_annotation",
        "scope_vs_snapshot",
        "missing_revalidation_barrier",
    ):
        assert lint in md


def test_pre_markdown_states_the_withheld_state_b_boundary():
    md = analyze_plan(_full_plan()).to_markdown()
    assert "State B (dependency-state) blast-share risk: WITHHELD" in md
    # the descriptive coverage block is present and labeled as not a risk score
    assert "## Coverage Readiness" in md
    assert "descriptive coverage-readiness, not a risk score" in md
    assert "runtime no-score reason" in md


def test_pre_markdown_recommended_action_points_at_the_keystone():
    md = analyze_plan(_full_plan()).to_markdown()
    assert "## Recommended Action" in md
    # node 0 carries a write_with_no_prior_read finding and is the keystone
    assert "starting at the keystone step" in md
    assert "PRE never emits a State-B number" in md


def test_pre_markdown_no_findings_branch():
    # an all-explicit-roots 2-step plan trips no lint and has no keystone follower
    plan = {
        "nodes": [
            {"idx": 0, "agent": "a", "kind": "decision", "control_preds": []},
            {"idx": 1, "agent": "a", "kind": "decision", "control_preds": []},
        ]
    }
    md = analyze_plan(plan).to_markdown()
    assert "- no lint findings" in md
    assert "(none): no step has control-flow followers" in md
    assert "no design-time lints fired" in md


def test_pre_markdown_notes_render_as_bullets():
    rep = analyze_plan(_full_plan())
    md = rep.to_markdown()
    assert "## Notes" in md
    for note in rep.notes:
        assert f"- {note}" in md


# --- POST markdown -----------------------------------------------------------


def test_post_markdown_has_the_stage_banner_and_coverage_meta():
    md = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1).to_markdown()
    assert md.startswith("# Auditable POST Report: Structural Risk Analysis")
    assert "stage: POST" in md
    assert "- adapter: tau_bench_prior_db_reads_v1" in md
    assert "- state: scored" in md
    assert "rho=" in md and "dependency edge(s)" in md


def test_post_markdown_keystone_is_blast_radius_not_chokepoint():
    md = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1).to_markdown()
    assert "## Keystone (Blast-Radius)" in md
    # keystone is the first read (idx 2, get_order_details), score 0.25
    assert "step 2 [tool_call get_order_details]: structural risk 0.250" in md
    assert "of 8 other steps transitively rest on it" in md
    # PRE-only wording must NOT appear in a POST render
    assert "Execution-Topology Chokepoint" not in md


def test_post_markdown_ranked_table_and_scored_action():
    md = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1).to_markdown()
    assert "| score | step | kind | label |" in md
    assert "ranked decisions (structural blast share):" in md
    assert "0.250" in md  # the keystone row
    assert "## Recommended Action" in md
    assert "triage the keystone step first" in md


def test_post_markdown_withheld_score_renders_na_not_zero():
    # low-coverage own-record run: every score is withheld (None)
    md = analyze_run(_own_records(3), adapter=own_record_v1).to_markdown()
    assert "## Keystone (Blast-Radius)" in md
    assert "(none): structural risk withheld" in md
    # a withheld score is "n/a", never "0.000" (zero would read as no-risk)
    assert "| n/a |" in md
    assert "0.000" not in md
    assert "gather more observed dependency coverage" in md


def test_post_markdown_single_decision_action():
    md = analyze_run(_own_records(1), adapter=own_record_v1).to_markdown()
    assert "- state: no_score:single_decision" in md
    assert "single-decision run has no cross-decision structure" in md


def test_post_markdown_grounding_section_when_basis_present():
    # own records carry a decision basis, so model-basis grounding lights up
    md = analyze_run(_own_records(2), adapter=own_record_v1).to_markdown()
    assert "### Model-Basis Grounding" in md
    assert "step 0:" in md
    assert "supported" in md


def test_post_markdown_no_grounding_section_for_corpus_run():
    # a corpus tool trace states no basis -> no grounding sub-section
    md = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1).to_markdown()
    assert "### Model-Basis Grounding" not in md


def test_post_markdown_empty_run_renders_without_crashing():
    md = analyze_run([], adapter=tau_bench_prior_db_reads_v1).to_markdown()
    assert "Auditable POST Report" in md
    assert "(none)" in md  # the empty ranked table degrades to a (none) line


# --- the top-level dispatcher ------------------------------------------------


def test_render_report_dispatches_pre_and_matches_method():
    rep = analyze_plan(_full_plan())
    assert render_report(rep) == rep.to_markdown()
    assert render_report(rep) == pre_to_markdown(rep)


def test_render_report_dispatches_post_and_matches_method():
    rep = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)
    assert render_report(rep) == rep.to_markdown()
    assert render_report(rep) == post_to_markdown(rep)


def test_render_report_rejects_other_types():
    with pytest.raises(TypeError):
        render_report({"not": "a report"})
    with pytest.raises(TypeError):
        render_report(42)


def test_render_report_level_offsets_headings():
    rep = analyze_plan(_full_plan())
    md = render_report(rep, level=2)
    assert md.startswith("## Auditable PRE Report")
    assert "### Keystone (Execution-Topology Chokepoint)" in md


# --- the additive surface does not disturb the plaintext summary -------------


def test_to_markdown_is_additive_pre_summary_stays_plaintext():
    rep = analyze_plan(_full_plan())
    plain = rep.summary()
    # the plaintext summary / __str__ stay terse: no Markdown headings leak in
    assert "#" not in plain
    assert str(rep) == plain
    # and it is NOT the markdown form
    assert rep.to_markdown() != plain


def test_to_markdown_is_additive_post_summary_stays_plaintext():
    rep = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1)
    plain = rep.summary()
    assert "#" not in plain
    assert str(rep) == plain
    assert rep.to_markdown() != plain


def test_pre_and_post_markdown_share_the_five_labeled_parts():
    pre_md = analyze_plan(_full_plan()).to_markdown()
    post_md = analyze_run(_tau_messages(), adapter=tau_bench_prior_db_reads_v1).to_markdown()
    for md, banner in (
        (pre_md, "Auditable PRE Report"),
        (post_md, "Auditable POST Report"),
    ):
        assert banner in md
        assert "What Is Risky on the Graph" in md
        assert "Keystone" in md
        assert "Recommended Action" in md
        assert "Notes" in md
