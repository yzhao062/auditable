"""Pin the public surface importable from ``auditable`` (PyOD-style API guard).

The top level is the small stable surface: the capture / recovery flow, the
auditors, and the v0.3 offline-analysis entry (``analyze_run`` / ``AnalysisReport``)
plus the ``Adapter`` extension point. The heavy graph internals (``SessionGraph``,
``Step``, ``structural_risk``, ``build_graph`` ...) stay under ``auditable.graph.*``
and must NOT leak to the top level. Changes to this surface are additive only:
adding a name updates the pinned set deliberately; a name silently disappearing or
a graph internal leaking up fails this test.
"""
import auditable

# The pinned public set. Add to it deliberately; never remove (additive-only).
EXPECTED_PUBLIC = {
    "__version__",
    # capture, replay, executed recovery
    "audit",
    "replay",
    "DecisionRecord",
    "Report",
    "CompoundReport",
    "Action",
    "DependencySnapshot",
    "Verdict",
    "FixAction",
    "ReplayUndecidable",
    "ActionGate",
    "GateOutcome",
    "Rail",
    "ReferenceLedger",
    "MemorySink",
    "FileSink",
    "default_sink",
    "Decision",
    "Policy",
    "DataSpan",
    "ModelSpan",
    "HarnessSpan",
    # standalone auditors
    "Auditor",
    "DataAuditor",
    "ModelAuditor",
    "HarnessAuditor",
    # v0.3 offline analysis + the ingestion extension point
    "analyze_run",
    "AnalysisReport",
    "render_report",
    "Adapter",
}


def test_all_matches_pinned_public_set():
    assert set(auditable.__all__) == EXPECTED_PUBLIC


def test_every_public_name_is_importable():
    for name in EXPECTED_PUBLIC:
        assert hasattr(auditable, name), f"{name} listed in __all__ but not importable"


def test_no_duplicate_names_in_all():
    assert len(auditable.__all__) == len(set(auditable.__all__))


def test_v03_names_present():
    # the three names this round adds to the stable surface
    assert callable(auditable.analyze_run)
    assert isinstance(auditable.AnalysisReport, type)
    assert auditable.Adapter is not None


def test_graph_internals_do_not_leak_to_top_level():
    # these stay under auditable.graph.* on purpose (no decade-grade promise)
    for name in (
        "SessionGraph",
        "Step",
        "DependencyEdge",
        "structural_risk",
        "RiskResult",
        "build_graph",
        "layered_features",
        "ground_record",
        "ground_basis",
    ):
        assert not hasattr(auditable, name), f"{name} should not be a top-level export"


def test_graph_internals_reachable_under_submodule():
    # advanced users can still reach them, just not from the top level
    from auditable.graph import build_graph  # noqa: F401  (the kernel)
    from auditable.graph.risk import structural_risk  # noqa: F401
    from auditable.graph.session import SessionGraph, Step  # noqa: F401
