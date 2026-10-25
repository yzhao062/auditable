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
    # method library registry
    "list_methods",
    "get_method",
    "MethodCard",
    # agentpile records in, method inputs and labels out
    "from_agentpile",
    "configs_from_agentpile",
    "step_targets",
    "run_targets",
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
    # live (streaming) analysis: the same kernel over a growing prefix graph
    "LiveSession",
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


def test_live_session_is_a_class():
    # the live (streaming) companion to analyze_run
    assert isinstance(auditable.LiveSession, type)


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


def test_graph_all_lists_the_kernel_names_the_docs_render():
    import auditable.graph as graph

    # docs/api.md renders ``::: auditable.graph``, which shows only the names in __all__
    assert set(graph.__all__) == {"NODE_TYPES", "EXECUTION_EDGES", "DEPENDENCY_EDGES", "build_graph",
                                  "dependency_dag", "characterize", "downstream_reach", "execution_reach",
                                  "layered_features", "feature_vector"}
    assert all(hasattr(graph, name) for name in graph.__all__)


# Constructor keywords a benchmark passes (CatchBench's board adapters); docs/methods-contract.md
# lists them. Renaming one breaks a pinned consumer, so the names are fixed here.
CONSTRUCTOR_KEYWORDS = {
    "step.random": {"random_state"},
    "online.random": {"random_state"},
    "run.random": {"random_state"},
    "run.gsafeguard_lite": {"seed"},
    "run.guardian_lite": {"seed"},
    "step.llm_all_at_once": {"backend", "model"},
    "step.llm_step_by_step": {"backend", "model"},
    "step.llm_binary_search": {"backend", "model"},
    "config.llm_needed": {"backend", "model", "needed"},
}


def test_constructor_keywords_a_benchmark_passes():
    import inspect

    from auditable.methods import get_method

    for method_id, names in CONSTRUCTOR_KEYWORDS.items():
        # binding by keyword fails for a renamed, removed or positional-only parameter
        inspect.signature(get_method(method_id)).bind_partial(**{name: None for name in names})
