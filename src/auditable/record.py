"""Decision records and the report leaf: the signed, replayable evidence unit.

One agent decision spans three layers, and auditable records all three in one record:

- data:    what the agent read, plus the dependency snapshot it relied on.
- model:   which model produced the output, and its stated decision basis.
- harness: the action the agent executed, and its cost.

Each layer attaches a normalized ``Report`` (the leaf an ``Auditor`` produces). The
record is content-addressed (SHA-256) and chained to the previous record, so the log is
tamper-evident; the record digest is taken over the bound reports, so it commits to all
three leaves.
"""
from __future__ import annotations

import abc
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .compound import CompoundReport


def _canonical_json(obj: Any) -> str:
    """Canonical JSON for hashing and persistence: sorted keys, compact, UTF-8, and no
    lenient stringify. Non-JSON content raises, so the digest never silently commits to an
    unstable ``str()`` of an arbitrary object. Record content must be JSON-serializable.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass
class Report:
    """The normalized leaf an ``Auditor`` returns. Uniform across stages.

    ``score`` is normalized to [0, 1] (0 normal, 1 maximal risk); it is the field that
    lets the compound combine reports across stages. ``flag`` is a short label such as
    ``"ok"``, ``"stale"``, ``"low_trust"``, or ``"over_cap"``. ``evidence`` holds the
    detector-specific detail behind the score.
    """

    stage: str
    name: str
    score: float = 0.0
    flag: str = "ok"
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    def digest(self) -> str:
        """Content hash of this leaf, independently verifiable."""
        return _sha256(asdict(self))


class Auditor(abc.ABC):
    """Detect-face base class, the analog of pyod's ``BaseDetector``.

    A concrete auditor sets ``stage`` and ``name`` and implements ``assess``, returning a
    normalized ``Report``. The ``subject`` type is stage-specific (a snapshot, a model
    span, an action); the uniform part is the ``Report`` return, not the input.
    """

    stage: str = ""
    name: str = ""

    @abc.abstractmethod
    def assess(self, subject: Any) -> Report:
        raise NotImplementedError


@dataclass
class Action:
    """What the agent is about to do (or did)."""

    type: str
    arguments: dict = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class DependencySnapshot:
    """The dependency state the agent relied on at decision time.

    ``state`` holds the versioned dependencies, for example
    ``{"budget_remaining": 5000, "allow_list_version": 7, "policy_id": "kyc-2026-03"}``.
    ``captured_at`` is the snapshot's own timestamp, which may lag the decision; a stale
    snapshot is the failure mode auditable is built to surface.
    """

    state: dict = field(default_factory=dict)
    captured_at: Optional[float] = None


def _empty_report(stage: str) -> "Report":
    return Report(stage=stage, name="")


@dataclass
class DataSpan:
    inputs: dict = field(default_factory=dict)
    retrieved: list = field(default_factory=list)
    snapshot: DependencySnapshot = field(default_factory=DependencySnapshot)
    report: Report = field(default_factory=lambda: _empty_report("data"))


@dataclass
class ModelSpan:
    model_id: str = ""
    decision_basis: str = ""
    output: Any = None
    report: Report = field(default_factory=lambda: _empty_report("model"))


@dataclass
class HarnessSpan:
    action_type: str = ""
    arguments: dict = field(default_factory=dict)
    cost: float = 0.0
    report: Report = field(default_factory=lambda: _empty_report("harness"))


@dataclass
class DecisionRecord:
    action_type: str
    data: DataSpan = field(default_factory=DataSpan)
    model: ModelSpan = field(default_factory=ModelSpan)
    harness: HarnessSpan = field(default_factory=HarnessSpan)
    compound: "Optional[CompoundReport]" = None
    decided_at: float = field(default_factory=time.time)
    prev_digest: Optional[str] = None
    record_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        """Content hash over the record body (minus ``record_id``), chained via prev.

        The body includes the three leaf reports and the compound, so the record digest
        transitively commits to all leaves.
        """
        body = self.to_dict()
        body.pop("record_id", None)
        return _sha256(body)
