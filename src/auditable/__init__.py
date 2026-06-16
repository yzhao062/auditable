"""auditable: capture, replay, and audit AI agent decisions.

Open-source SDK for auditable AI agents. Capture each agent decision with the
dependency snapshot it relied on, replay it under the live state, and route a fix.

By Yue Zhao (creator of PyOD). Built on the Auditable Agents framework
(arXiv:2604.05485).
"""
from .core import (
    Action,
    Decision,
    FixAction,
    Verdict,
    audit,
    default_sink,
    replay,
)
from .record import (
    DataSpan,
    DecisionRecord,
    DependencySnapshot,
    HarnessSpan,
    ModelSpan,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "audit",
    "replay",
    "Action",
    "Decision",
    "Verdict",
    "FixAction",
    "default_sink",
    "DecisionRecord",
    "DataSpan",
    "ModelSpan",
    "HarnessSpan",
    "DependencySnapshot",
]
