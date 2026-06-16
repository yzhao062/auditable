"""Decision records: the signed, replayable evidence unit auditable captures.

One agent decision spans three layers, and auditable records all three in a single
record:

- data:    what the agent read (inputs, retrieved context, and the dependency
           snapshot: policy / budget / allow-list / config versions that were live).
- model:   which model produced the output, and the stated decision basis.
- harness: the action the agent executed, and its cost.

The record is content-addressed (SHA-256) and chained to the previous record so the
log is tamper-evident.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class DependencySnapshot:
    """The dependency state the agent relied on at decision time.

    ``state`` holds the versioned dependencies, for example
    ``{"budget_remaining": 5000, "allow_list_version": 7, "policy_id": "kyc-2026-03"}``.
    ``captured_at`` is the snapshot's own timestamp, which may lag the decision; a stale
    snapshot is the failure mode auditable is built to surface.
    """

    state: dict[str, Any] = field(default_factory=dict)
    captured_at: Optional[float] = None


@dataclass
class DataSpan:
    inputs: dict[str, Any] = field(default_factory=dict)
    retrieved: list[Any] = field(default_factory=list)
    snapshot: DependencySnapshot = field(default_factory=DependencySnapshot)


@dataclass
class ModelSpan:
    model_id: str = ""
    decision_basis: str = ""
    output: Any = None


@dataclass
class HarnessSpan:
    action_type: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class DecisionRecord:
    action_type: str
    data: DataSpan = field(default_factory=DataSpan)
    model: ModelSpan = field(default_factory=ModelSpan)
    harness: HarnessSpan = field(default_factory=HarnessSpan)
    decided_at: float = field(default_factory=time.time)
    prev_digest: Optional[str] = None
    record_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        """Content hash over the record body plus the previous digest (hash chain)."""
        body = self.to_dict()
        body.pop("record_id", None)
        encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
