"""auditable: auditing methods for AI agents.

The method library (``auditable.methods``) collects configuration audits (PRE), run-level
failure detectors and step-level fault localizers (POST), and online monitors (LIVE),
each family behind one interface; ``list_methods`` and ``get_method`` read the registry.
``auditable.featurize`` turns traces into the features the methods read.

The decision-record modules (``audit``, ``replay``, the record and sink types) and the
graph analyses (``analyze_run``, ``LiveSession``) remain available. ``ActionGate``,
``ReferenceLedger`` and ``DataAuditor`` are deprecated.
"""
from .record import (
    Action,
    Auditor,
    DataSpan,
    DecisionRecord,
    DependencySnapshot,
    HarnessSpan,
    ModelSpan,
    Report,
)
from .compound import CompoundReport
from .chain import (
    Decision,
    FileSink,
    FixAction,
    MemorySink,
    Policy,
    ReplayUndecidable,
    Verdict,
    audit,
    default_sink,
    replay,
)
from .data import DataAuditor
from .model import ModelAuditor
from .harness import ActionGate, GateOutcome, HarnessAuditor, Rail, ReferenceLedger
from .analysis import AnalysisReport, LiveSession, analyze_run
from .report import render_report
from .graph.adapters import Adapter
from .base import MethodCard
from .methods import get_method, list_methods

__version__ = "0.3.0.dev0"

__all__ = [
    "__version__",
    # Method library: the registry and the card every method carries.
    "list_methods",
    "get_method",
    "MethodCard",
    # Composition surface (lead here): capture, replay, executed recovery.
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
    # Module API (standalone scorers; inputs to the record, not the headline product).
    "Auditor",
    "DataAuditor",
    "ModelAuditor",
    "HarnessAuditor",
    # v0.3 offline analysis: the unified session graph + structural risk, read here.
    "analyze_run",
    "AnalysisReport",
    # Live (streaming) analysis: the same kernel over a growing prefix graph.
    "LiveSession",
    # Markdown rendering for the PRE / POST analysis reports.
    "render_report",
    "Adapter",
]
