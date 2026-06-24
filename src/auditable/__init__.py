"""auditable: capture, replay, and recover AI agent decisions.

Open-source SDK for live-state replay and recovery of consequential agent decisions.
Capture a signed record of each decision with the dependency state it relied on, replay
it under the state that is live now, and route and execute a fix (allow, block,
human-review, rollback). One decision binds a data, model, and harness report in a single
record (the full chain).
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
from .analysis import AnalysisReport, analyze_run
from .report import render_report
from .graph.adapters import Adapter

__version__ = "0.1.1"

__all__ = [
    "__version__",
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
    # Markdown rendering for the PRE / POST analysis reports.
    "render_report",
    "Adapter",
]
