"""The model module: audit the deciding model.

v0.1 ships ``ModelAuditor`` with a thin trust flag derived from whether the model stated
a decision basis. v0.3 replaces the body with TrustLLM trust signals; the interface and
the ``Report`` stay the same.
"""
from __future__ import annotations

from ..record import Auditor, ModelSpan, Report


class ModelAuditor(Auditor):
    """A thin trust flag on the deciding model (v0.1).

    Heuristic: a stated decision basis raises trust, its absence lowers it. The score is
    the risk (``1 - trust``); the trust value is kept in evidence. v0.3 replaces this with
    TrustLLM trust signals on the model and its output.
    """

    stage = "model"

    def __init__(self, name: str = "decision-basis-trust"):
        self.name = name

    def assess(self, subject: ModelSpan) -> Report:
        if subject.decision_basis.strip():
            return Report(
                self.stage, self.name, 0.1, "ok",
                "Model stated a decision basis.", {"trust": 0.9},
            )
        return Report(
            self.stage, self.name, 0.5, "low_trust",
            "Model stated no decision basis.", {"trust": 0.5},
        )
