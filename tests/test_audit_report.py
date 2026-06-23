"""Tests for the human-facing audit-report charts in ``auditable._report_figures``.

The audit report has two consumers and the format serves the consumer: the
agent-facing Markdown carries no images, and the human-facing PDF carries the five
charts. This file covers chart generation only (the focus of this pass):

- the lazy-import contract: ``auditable._report_figures`` imports with the heavy
  ``report`` extra absent, and a figure call then raises a clear
  ``pip install auditable[report]`` error;
- each of the five figure functions returns non-empty PNG bytes (the ``%PNG``
  signature) from the STABLE upstream report objects (PRE ``PreReport``, POST
  ``AnalysisReport``, the LIVE ``Verdict`` sequence);
- the PNG bytes write to a real file that is produced and non-empty;
- the rendering is deterministic (the same inputs give byte-identical PNGs);
- a missing pillar renders an explicit "no data" panel rather than failing or
  drawing a fabricated zero, mirroring the upstream no-score honesty boundary.

The chart functions need matplotlib + networkx (the ``report`` extra), so the
figure tests are guarded with ``importorskip``. The import-contract test runs
without the extra by blocking the heavy modules.
"""
import builtins
import importlib
import json
from dataclasses import dataclass
from typing import List, Optional

import pytest


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --- the lazy-import contract (runs WITHOUT the report extra) -----------------


def test_report_figures_imports_without_heavy_deps(monkeypatch):
    """``_report_figures`` must import even when matplotlib / networkx / fpdf are absent.

    The auditable core stays dependency-free: only calling a figure function (or
    ``to_pdf``) may require the extra. We block the heavy roots at import time, reload
    the module, and assert the import succeeds and the public functions are present.
    """
    real_import = builtins.__import__
    blocked = {"matplotlib", "networkx", "fpdf"}

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.reload(importlib.import_module("auditable._report_figures"))
    try:
        for fn in (
            "blast_share_figure",
            "coverage_gauge_figure",
            "decision_graph_figure",
            "recovery_figure",
            "findings_by_severity_figure",
        ):
            assert hasattr(module, fn), f"{fn} missing after import without heavy deps"

        # Calling a figure function under blocked deps raises the guarded extra error.
        with pytest.raises(ImportError) as excinfo:
            module.recovery_figure([object()])
        assert "auditable[report]" in str(excinfo.value)
    finally:
        # Restore a clean, normally-imported module for any later test in the session.
        monkeypatch.undo()
        importlib.reload(importlib.import_module("auditable._report_figures"))


# --- the AuditReport agent-facing contract (runs WITHOUT the report extra) ----
#
# to_markdown is agent-facing: it embeds no image, so it must work with matplotlib
# and fpdf2 absent. These tests build the aggregate from tiny duck-typed pillar
# stand-ins (no networkx, no graph extra needed), block the heavy roots, and assert
# the Markdown is structured, parseable, image-free, and deterministic, and that
# to_pdf still raises the clear extra error.


@dataclass
class _StubLint:
    """A minimal stand-in for a PRE LintFinding (the fields the aggregator reads)."""

    lint: str
    node_idx: Optional[int]
    resource_id: Optional[str]
    detail: str
    severity: str = "warning"


@dataclass
class _StubPre:
    """A minimal stand-in for a PreReport (only the fields the aggregator reads)."""

    adapter: str
    n_steps: int
    keystone_idx: Optional[int]
    keystone_followers: int
    findings: List[_StubLint]
    preflight_coverage: Optional[object] = None
    resource_touch_completeness: Optional[object] = None


@dataclass
class _StubDecisionRisk:
    idx: int
    kind: str
    agent: str
    score: Optional[float]
    label: str


@dataclass
class _StubEdgeCoverage:
    n_dep_edges: int
    by_grade: dict
    rho: float
    observed_fraction: float


@dataclass
class _StubPost:
    """A minimal stand-in for an AnalysisReport (only the fields the aggregator reads)."""

    state: str
    adapter: str
    n_steps: int
    ranked: List[_StubDecisionRisk]
    keystone: Optional[_StubDecisionRisk]
    coverage: _StubEdgeCoverage


@dataclass
class _StubFixAction:
    value: str


@dataclass
class _StubVerdict:
    action: _StubFixAction
    justified: bool
    reason: str
    record_id: str = ""


def _stub_pre():
    return _StubPre(
        adapter="declared_plan_v1",
        n_steps=3,
        keystone_idx=0,
        keystone_followers=2,
        findings=[
            _StubLint("write_with_no_prior_read", 0, "kyc.tier", "writes kyc.tier blindly"),
            _StubLint("scope_vs_snapshot", 1, "ledger.balance", "scope exceeds snapshot"),
        ],
    )


def _stub_post():
    ranked = [
        _StubDecisionRisk(2, "tool_call", "a", 0.5, "tool_call x"),
        _StubDecisionRisk(1, "decision", "a", 0.25, "decision y"),
        _StubDecisionRisk(0, "tool_call", "a", 0.0, "tool_call z"),
    ]
    return _StubPost(
        state="scored",
        adapter="tau_bench_prior_db_reads_v1",
        n_steps=3,
        ranked=ranked,
        keystone=ranked[0],
        coverage=_StubEdgeCoverage(n_dep_edges=2, by_grade={}, rho=0.667, observed_fraction=1.0),
    )


def _stub_verdicts():
    return [
        _StubVerdict(_StubFixAction("rollback"), False, "relied on stale state", "abc123def456ghi"),
        _StubVerdict(_StubFixAction("allow"), True, "ok under live state", "zzz999"),
    ]


def test_audit_report_imports_and_renders_markdown_without_heavy_deps(monkeypatch):
    """``audit_report`` imports and ``to_markdown`` renders with matplotlib / fpdf absent.

    The agent-facing Markdown embeds no chart, so it must not need the report extra.
    We block matplotlib and fpdf at import time, build the aggregate from duck-typed
    pillar stand-ins (no networkx), and assert the Markdown carries the stable
    anchors, parses, is image-free, and that ``to_pdf`` raises the clear extra error.
    """
    real_import = builtins.__import__
    blocked = {"matplotlib", "fpdf"}

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.reload(importlib.import_module("auditable.audit_report"))
    try:
        AuditReport = module.AuditReport
        report = AuditReport.from_run(
            pre=_stub_pre(),
            post=_stub_post(),
            verdicts=_stub_verdicts(),
            title="No-Extra Report",
        )
        md = report.to_markdown()

        # Stable anchors an agent splits on.
        for anchor in ("## Verdict", "## Keystone", "## Findings", "## Recommended Actions", "## Coverage", "## JSON"):
            assert anchor in md, f"missing stable anchor {anchor!r}"

        # No raster image in the agent-facing Markdown.
        assert "![" not in md, "the agent Markdown must embed no image"
        assert b"\x89PNG".decode("latin-1") not in md
        assert "data:image" not in md

        # The JSON mirror parses and carries the headline fields.
        block = md.split("```json", 1)[1].split("```", 1)[0]
        payload = json.loads(block)
        assert payload["verdict"] in {"PASS", "REVIEW", "BLOCK"}
        assert payload["counts"]["findings"] == len(report.findings)
        assert payload["blast_keystone"]["idx"] == 2
        assert payload["pillars"]["live"] is True
        assert "real_time" not in payload["pillars"]

        # Calling to_pdf under blocked deps raises the guarded extra error.
        with pytest.raises(ImportError) as excinfo:
            report.to_pdf("unused.pdf")
        assert "auditable[report]" in str(excinfo.value)
    finally:
        monkeypatch.undo()
        importlib.reload(importlib.import_module("auditable.audit_report"))


def test_audit_report_markdown_is_deterministic_without_heavy_deps():
    """Two ``to_markdown`` renders over the same inputs give byte-identical strings."""
    from auditable.audit_report import AuditReport

    report = AuditReport.from_run(pre=_stub_pre(), post=_stub_post(), verdicts=_stub_verdicts())
    assert report.to_markdown() == report.to_markdown()


def test_audit_report_findings_sorted_severity_then_source(monkeypatch):
    """The unified findings table sorts severity desc, then source, then locus."""
    from auditable.audit_report import AuditReport, SOURCE_PRE, SOURCE_LIVE

    report = AuditReport.from_run(pre=_stub_pre(), post=_stub_post(), verdicts=_stub_verdicts())
    severities = [f.severity for f in report.findings]
    # rollback (a LIVE verdict) outranks the PRE warnings.
    assert severities[0] == "rollback"
    assert severities[1:] == ["warning", "warning"]
    # within the warnings, the PRE source rows are ordered by ascending step idx.
    pre_rows = [f for f in report.findings if f.source == SOURCE_PRE]
    assert [f.sort_idx for f in pre_rows] == sorted(f.sort_idx for f in pre_rows)
    # the ALLOW verdict is not a finding.
    assert all(f.source != SOURCE_LIVE or f.severity != "allow" for f in report.findings)


def test_audit_report_rollup_verdict_logic():
    """BLOCK if any block; REVIEW if any rollback/human-review or PRE lint; else PASS."""
    from auditable.audit_report import AuditReport, VERDICT_BLOCK, VERDICT_PASS, VERDICT_REVIEW

    block_v = [_StubVerdict(_StubFixAction("block"), False, "blocked", "r1")]
    rollback_v = [_StubVerdict(_StubFixAction("rollback"), False, "stale", "r2")]
    review_v = [_StubVerdict(_StubFixAction("human_review"), False, "undecidable", "r3")]
    allow_v = [_StubVerdict(_StubFixAction("allow"), True, "ok", "r4")]

    # A block anywhere dominates, even with only allows otherwise.
    assert AuditReport.from_run(verdicts=block_v + allow_v).headline_verdict == VERDICT_BLOCK
    # A rollback or human-review raises REVIEW.
    assert AuditReport.from_run(verdicts=rollback_v).headline_verdict == VERDICT_REVIEW
    assert AuditReport.from_run(verdicts=review_v).headline_verdict == VERDICT_REVIEW
    # A PRE lint alone raises REVIEW.
    assert AuditReport.from_run(pre=_stub_pre()).headline_verdict == VERDICT_REVIEW
    # Only allows, no PRE lint: PASS.
    assert AuditReport.from_run(verdicts=allow_v).headline_verdict == VERDICT_PASS
    # No pillars at all: PASS (nothing to act on).
    assert AuditReport.from_run().headline_verdict == VERDICT_PASS


def test_audit_report_single_pillar_and_missing_pillars_render():
    """PRE-only and POST-only reports render explicit missing-pillar lines, not zeros."""
    from auditable.audit_report import AuditReport

    pre_only = AuditReport.from_pre(_stub_pre())
    pre_md = pre_only.to_markdown()
    assert "no POST data" in pre_md  # the absent blast keystone is explicit
    assert "## Verdict" in pre_md

    post_only = AuditReport.from_analysis(_stub_post())
    post_md = post_only.to_markdown()
    assert "no PRE data" in post_md  # the absent structural keystone is explicit


def test_audit_report_withheld_score_renders_not_zero():
    """A no-score POST renders 'withheld', never a misleading 0.000 blast share."""
    from auditable.audit_report import AuditReport

    ranked = [_StubDecisionRisk(0, "decision", "a", None, "decision only")]
    post = _StubPost(
        state="no_score:single_decision",
        adapter="own_record_v1",
        n_steps=1,
        ranked=ranked,
        keystone=None,
        coverage=_StubEdgeCoverage(0, {}, 0.0, 0.0),
    )
    report = AuditReport.from_analysis(post)
    assert report.blast_keystone is None
    keystone_section = report.to_markdown().split("## Keystone", 1)[1].split("## Findings", 1)[0]
    assert "withheld" in keystone_section
    assert "0.000" not in keystone_section
    # the exact no-score reason reaches the Coverage section.
    coverage_section = report.to_markdown().split("## Coverage", 1)[1].split("## JSON", 1)[0]
    assert "no_score:single_decision" in coverage_section


def test_audit_report_writes_markdown_file(tmp_path):
    """``to_markdown(path)`` writes the same string it returns to a real file."""
    from auditable.audit_report import AuditReport

    report = AuditReport.from_run(pre=_stub_pre(), post=_stub_post(), verdicts=_stub_verdicts())
    out = tmp_path / "report.md"
    returned = report.to_markdown(str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == returned


# --- fixtures: the three STABLE upstream pillars ------------------------------

# The chart functions need the report extra (matplotlib + networkx). Skip the whole
# block below if it is unavailable, exactly as the rest of the suite skips on the
# graph extra.
pytest.importorskip("matplotlib")
pytest.importorskip("networkx")

from auditable import (  # noqa: E402
    Action,
    DependencySnapshot,
    MemorySink,
    analyze_run,
    audit,
    replay,
)
from auditable._report_figures import (  # noqa: E402
    blast_share_figure,
    coverage_gauge_figure,
    decision_graph_figure,
    findings_by_severity_figure,
    recovery_figure,
)
from auditable.graph.adapters import (  # noqa: E402
    declared_plan_v1,
    tau_bench_prior_db_reads_v1,
)
from auditable.graph.pre import analyze_plan  # noqa: E402
from auditable.graph.session import SessionGraph  # noqa: E402


def _payment_plan():
    """A declared payment-approver plan that trips every PRE lint (see analyze_plan demo)."""
    return {
        "plan_id": "payment-approver-v1",
        "framework": "declared",
        "nodes": [
            {"idx": 0, "agent": "kyc_tool", "kind": "tool_call", "writes": ["kyc.tier"]},
            {
                "idx": 1,
                "agent": "approver",
                "kind": "decision",
                "reads": [{"id": "kyc.tier", "producer": 0, "volatile": True}],
                "scope": ["kyc.tier", "ledger.balance"],
            },
            {
                "idx": 2,
                "agent": "ledger_tool",
                "kind": "tool_call",
                "reads": [{"id": "kyc.tier", "producer": 0}],
                "writes": ["ledger.entry"],
            },
        ],
    }


def _airline_run():
    """A tau-bench-style airline trajectory: one read, two writes rest on it."""
    return [
        {"role": "system", "content": "you are an airline agent; follow the policy"},
        {"role": "user", "content": "On reservation ZFA04Y, move me to the morning flights and add one checked bag."},
        {"role": "assistant", "content": "Pulling up the reservation.",
         "tool_calls": [{"function": {"name": "get_reservation_details"}}]},
        {"role": "tool", "name": "get_reservation_details",
         "content": '{"reservation_id": "ZFA04Y", "cabin": "economy", "origin": "SFO", "destination": "JFK", "baggages": 0}'},
        {"role": "assistant", "content": "Rebooking onto the morning flights.",
         "tool_calls": [{"function": {"name": "update_reservation_flights"}}]},
        {"role": "tool", "name": "update_reservation_flights", "content": "ok"},
        {"role": "assistant", "content": "Adding the checked bag.",
         "tool_calls": [{"function": {"name": "update_reservation_baggages"}}]},
        {"role": "tool", "name": "update_reservation_baggages", "content": "ok"},
    ]


def _budget_policy(state, action):
    bal = state.get("budget_remaining", 0)
    if bal >= action.cost:
        return True, "within budget"
    return False, f"over budget: need {action.cost}, have {bal}"


def _drift_verdicts():
    """Two recorded charges replayed under a drifted-down budget, so both rollback."""
    sink = MemorySink()
    with audit("charge", snapshot=DependencySnapshot(state={"budget_remaining": 5000}), sink=sink) as d:
        d.read(amount=1200).model("m1", "ok").act(Action(type="charge", cost=1200))
    with audit("charge", snapshot=DependencySnapshot(state={"budget_remaining": 5000}), sink=sink) as d:
        d.read(amount=400).model("m1", "ok").act(Action(type="charge", cost=400))
    live = {"budget_remaining": 300}
    return [replay(r, live_state=live, policy=_budget_policy) for r in sink.records]


@dataclass
class _Finding:
    """A minimal finding row for the by-severity chart (source + severity only)."""

    source: str
    severity: str


@pytest.fixture(scope="module")
def pre_report():
    return analyze_plan(_payment_plan(), adapter=declared_plan_v1)


@pytest.fixture(scope="module")
def post_report():
    return analyze_run(_airline_run(), adapter=tau_bench_prior_db_reads_v1)


@pytest.fixture(scope="module")
def session_graph():
    steps = list(tau_bench_prior_db_reads_v1.to_steps(_airline_run()))
    return SessionGraph.from_steps(steps)


@pytest.fixture(scope="module")
def verdicts():
    return _drift_verdicts()


@pytest.fixture(scope="module")
def findings(pre_report, verdicts):
    rows: List[_Finding] = [_Finding("PRE", f.severity) for f in pre_report.findings]
    rows += [_Finding("LIVE", v.action.value) for v in verdicts if v.action.value != "allow"]
    return rows


def _assert_png(data) -> None:
    assert isinstance(data, (bytes, bytearray)), "a figure must return bytes"
    assert len(data) > 0, "the PNG bytes must be non-empty"
    assert bytes(data[:8]) == PNG_MAGIC, "the bytes must carry the PNG signature"


# --- chart 1: blast-share ranking --------------------------------------------


def test_blast_share_figure_produces_png(post_report):
    _assert_png(blast_share_figure(post_report))


def test_blast_share_figure_writes_nonempty_file(post_report, tmp_path):
    out = tmp_path / "blast_share.png"
    out.write_bytes(blast_share_figure(post_report))
    assert out.exists() and out.stat().st_size > 0


def test_blast_share_none_renders_no_data_panel():
    # A missing POST pillar is an explicit panel, never a fabricated zero chart.
    _assert_png(blast_share_figure(None))


# --- chart 2: preflight / coverage gauge -------------------------------------


def test_coverage_gauge_pre_produces_png(pre_report):
    data = coverage_gauge_figure(
        pre_report.preflight_coverage,
        touch=pre_report.resource_touch_completeness,
    )
    _assert_png(data)


def test_coverage_gauge_post_produces_png(post_report):
    _assert_png(coverage_gauge_figure(post_report.coverage))


def test_coverage_gauge_none_renders_no_data_panel():
    _assert_png(coverage_gauge_figure(None))


# --- chart 3: two-layer decision graph ---------------------------------------


def test_decision_graph_figure_produces_png(session_graph, pre_report, post_report):
    data = decision_graph_figure(
        session_graph,
        structural_keystone_idx=pre_report.keystone_idx,
        blast_keystone_idx=post_report.keystone.idx,
    )
    _assert_png(data)


def test_decision_graph_figure_writes_nonempty_file(session_graph, tmp_path):
    out = tmp_path / "decision_graph.png"
    out.write_bytes(decision_graph_figure(session_graph))
    assert out.exists() and out.stat().st_size > 0


def test_decision_graph_none_renders_no_data_panel():
    _assert_png(decision_graph_figure(None))


def test_decision_graph_rejects_wrong_type():
    with pytest.raises(TypeError):
        decision_graph_figure(object())


# --- chart 4: snapshot-vs-live drift / recovery ------------------------------


def test_recovery_figure_produces_png(verdicts):
    _assert_png(recovery_figure(verdicts))


def test_recovery_figure_writes_nonempty_file(verdicts, tmp_path):
    out = tmp_path / "recovery.png"
    out.write_bytes(recovery_figure(verdicts))
    assert out.exists() and out.stat().st_size > 0


def test_recovery_empty_renders_no_data_panel():
    _assert_png(recovery_figure([]))


# --- chart 5: findings by severity and source --------------------------------


def test_findings_by_severity_figure_produces_png(findings):
    assert findings, "the fixture should produce at least one finding"
    _assert_png(findings_by_severity_figure(findings))


def test_findings_by_severity_empty_renders_no_data_panel():
    _assert_png(findings_by_severity_figure([]))


def test_findings_by_severity_accepts_dict_rows():
    # The unified findings table may reach the chart as mapping rows; both shapes work.
    rows = [
        {"source": "PRE", "severity": "warning"},
        {"source": "LIVE", "severity": "rollback"},
        {"source": "POST", "severity": "warning"},
    ]
    _assert_png(findings_by_severity_figure(rows))


# --- all five together, plus determinism -------------------------------------


def test_all_five_charts_produce_nonempty_files(
    pre_report, post_report, session_graph, verdicts, findings, tmp_path
):
    """Render the whole chart set the PDF embeds, and assert each file is produced."""
    charts = {
        "blast_share.png": blast_share_figure(post_report),
        "coverage_gauge.png": coverage_gauge_figure(
            pre_report.preflight_coverage, touch=pre_report.resource_touch_completeness
        ),
        "decision_graph.png": decision_graph_figure(
            session_graph,
            structural_keystone_idx=pre_report.keystone_idx,
            blast_keystone_idx=post_report.keystone.idx,
        ),
        "recovery.png": recovery_figure(verdicts),
        "findings.png": findings_by_severity_figure(findings),
    }
    assert len(charts) == 5
    for name, data in charts.items():
        _assert_png(data)
        out = tmp_path / name
        out.write_bytes(data)
        assert out.stat().st_size > 0, f"{name} should be a non-empty file"


def test_blast_share_figure_is_deterministic(post_report):
    # Two renders over the same input give byte-identical PNGs (a seeded, pure render).
    first = blast_share_figure(post_report)
    second = blast_share_figure(post_report)
    assert first == second


def test_decision_graph_figure_is_deterministic(session_graph):
    # The seeded spring layout makes the graph render reproducible.
    first = decision_graph_figure(session_graph, seed=11)
    second = decision_graph_figure(session_graph, seed=11)
    assert first == second


# --- the AuditReport over the REAL pillars: aggregation + PDF -----------------
#
# These tests use the real PRE / POST / LIVE fixtures above (built from
# analyze_plan, analyze_run, and replay), so they exercise the aggregator against
# the actual upstream report objects, not stand-ins. The PDF path also needs fpdf2.
#
# Reference the module object, not captured class symbols: an earlier no-extra test
# reloads ``auditable.audit_report``, which rebinds its class attributes to fresh
# objects. Dereferencing through the (stable) module object keeps the fixture and
# the ``isinstance`` checks on the same class.

import auditable.audit_report as _ar  # noqa: E402


def test_session_graph_carries_dependency_edges_for_scored_post(post_report):
    """A scored POST report's session graph keeps the depends_on edges for the chart.

    Regression: the PDF two-layer-graph chart dropped every dependency edge because
    AuditReport rebuilt the graph from ranked rows, which carry no per-edge
    dependency layer. The real SessionGraph is now carried on AnalysisReport and the
    reconstruction prefers it, so the count of depends_on edges in the graph handed
    to decision_graph_figure matches the report's coverage (n_dep_edges), not zero.
    """
    assert post_report.state == "scored"
    assert post_report.coverage.n_dep_edges > 0  # the run has real dependency edges

    report = _ar.AuditReport.from_analysis(post_report)
    graph = report._session_graph()
    assert graph is not None

    G = graph.to_networkx()
    dep_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("etype") == "depends_on"
    ]
    assert len(dep_edges) == post_report.coverage.n_dep_edges
    assert dep_edges, "the two-layer chart must receive the depends_on edges, not drop them"


def test_session_graph_fallback_when_post_carries_no_graph():
    """A POST stand-in without session_graph still reconstructs the control-flow backbone."""
    post = _stub_post()  # _StubPost has no session_graph attribute
    assert not hasattr(post, "session_graph")
    report = _ar.AuditReport.from_analysis(post)
    graph = report._session_graph()  # falls through to synthetic reconstruction
    assert graph is not None
    G = graph.to_networkx()
    step_nodes = [
        n for n, d in G.nodes(data=True) if d.get("ntype") in ("decision", "tool_call")
    ]
    assert len(step_nodes) == 3  # the three ranked stand-in steps


@pytest.fixture(scope="module")
def full_report(pre_report, post_report, verdicts):
    return _ar.AuditReport.from_run(
        pre=pre_report,
        post=post_report,
        verdicts=verdicts,
        title="Full Pillar Report",
    )


def test_audit_report_aggregates_real_pillars(full_report, pre_report, verdicts):
    """The aggregate folds the real PRE lints and LIVE non-allow verdicts."""
    n_pre = len(pre_report.findings)
    n_nonallow = sum(1 for v in verdicts if v.action.value != "allow")
    assert len(full_report.findings) == n_pre + n_nonallow
    # The blast keystone comes from POST and carries the downstream count.
    assert full_report.blast_keystone is not None
    assert full_report.blast_keystone.kind == "blast"
    # The structural keystone comes from PRE and is a distinct concept.
    assert full_report.structural_keystone is not None
    assert full_report.structural_keystone.kind == "structural"
    # One recommended action per finding.
    assert len(full_report.recommended_actions) == len(full_report.findings)


def test_audit_report_findings_are_finding_rows(full_report):
    """Every unified finding is a frozen Finding the chart layer can read."""
    assert full_report.findings
    for f in full_report.findings:
        assert isinstance(f, _ar.Finding)
        assert f.source in {"PRE", "LIVE", "POST"}
        assert isinstance(f.to_dict(), dict)


def test_audit_report_markdown_json_block_parses(full_report):
    """The fenced json block in the agent Markdown parses and mirrors the headline."""
    md = full_report.to_markdown()
    block = md.split("```json", 1)[1].split("```", 1)[0]
    payload = json.loads(block)
    assert payload["verdict"] == full_report.headline_verdict
    assert payload["counts"]["findings"] == len(full_report.findings)
    assert len(payload["findings"]) == len(full_report.findings)


def test_audit_report_findings_chart_consumes_unified_rows(full_report):
    """The findings rows feed the chart directly (the aggregator and chart compose)."""
    # The chart reads .source / .severity off each row; a non-empty PNG confirms it.
    _assert_png(findings_by_severity_figure(full_report.findings))


# The PDF path additionally needs fpdf2 (the rest of the report extra).
fpdf = pytest.importorskip("fpdf")


def test_audit_report_to_pdf_writes_nonempty_pdf(full_report, tmp_path):
    """``to_pdf`` writes a non-empty ``%PDF`` byte stream with the charts embedded."""
    out = tmp_path / "audit.pdf"
    data = full_report.to_pdf(str(out))
    assert isinstance(data, (bytes, bytearray))
    assert bytes(data[:4]) == b"%PDF", "the output must be a PDF stream"
    assert out.exists() and out.stat().st_size > 0
    # The five charts make the PDF substantial; a bare text PDF would be far smaller.
    assert out.stat().st_size > 10_000, "the embedded charts should make the PDF non-trivial"


def test_audit_report_to_pdf_single_pillar(pre_report, post_report, tmp_path):
    """A PRE-only and a POST-only report each still emit a valid PDF."""
    pre_pdf = _ar.AuditReport.from_pre(pre_report).to_pdf(str(tmp_path / "pre.pdf"))
    post_pdf = _ar.AuditReport.from_analysis(post_report).to_pdf(str(tmp_path / "post.pdf"))
    assert bytes(pre_pdf[:4]) == b"%PDF"
    assert bytes(post_pdf[:4]) == b"%PDF"


def test_audit_report_to_pdf_empty_report(tmp_path):
    """An empty report (no pillars) still emits a valid PDF with no-data panels."""
    data = _ar.AuditReport.from_run(title="Empty").to_pdf(str(tmp_path / "empty.pdf"))
    assert bytes(data[:4]) == b"%PDF"


def test_audit_report_to_pdf_returns_same_bytes_it_writes(full_report, tmp_path):
    """The bytes ``to_pdf`` returns match the bytes written to disk."""
    out = tmp_path / "audit.pdf"
    returned = full_report.to_pdf(str(out))
    assert bytes(returned) == out.read_bytes()
