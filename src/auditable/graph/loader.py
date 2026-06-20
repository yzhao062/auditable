"""Offline log loader: read a persisted FileSink JSONL run back into records.

The signed, chained records already persist (:class:`auditable.FileSink` in
``chain.py`` writes one canonical-JSON record per line), but there is no read
path yet, so post-hoc analysis cannot get the records back. ``load_log`` is that
read path, the offline entry the v0.3 SessionGraph build and
:func:`auditable.graph.grounding.ground_record` consume.

It is the strict inverse of the write path and it fails closed. Loading verifies
the whole hash chain, not only the tail that ``FileSink._read_last_digest`` reads
on resume:

- each line is a JSON object with a non-empty ``record_id`` string, mirroring the
  corrupt / non-object / id-less strictness of ``_read_last_digest``;
- each ``record_id`` equals the content digest recomputed over the rest of the
  body, the same hash :meth:`auditable.DecisionRecord.digest` commits to on write,
  so any in-place edit to any field is caught;
- each record's ``prev_digest`` links to the immediately preceding record's
  ``record_id``, and the first record is the genesis (``prev_digest is None``), so
  a dropped, reordered, duplicated, or forked record breaks the chain;
- each verified body is the canonical ``DecisionRecord`` shape: the rebuilt record
  re-serializes to the signed body, so a hash-consistent blob that is not a record
  (extra, missing, or wrong-typed fields) is rejected rather than coerced into a
  default-filled record whose own digest would then disagree with its id.

A corrupt, tampered, broken, or forked chain raises ``ValueError`` rather than
returning a partial or silently re-rooted list: a tamper-evident log must not load
a damaged run as if it were intact. An empty or all-blank file loads as an empty
run. Pure Python (``json`` plus the record dataclasses); no networkx, no torch,
and no network, so it runs offline.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

from ..compound import CompoundReport
from ..record import (
    DataSpan,
    DecisionRecord,
    DependencySnapshot,
    HarnessSpan,
    ModelSpan,
    Report,
    _sha256,
)

__all__ = ["load_log"]


def _report_from_dict(d: Optional[dict]) -> Report:
    d = d or {}
    return Report(
        stage=d.get("stage", ""),
        name=d.get("name", ""),
        score=d.get("score", 0.0),
        flag=d.get("flag", "ok"),
        reason=d.get("reason", ""),
        evidence=dict(d.get("evidence") or {}),
    )


def _compound_from_dict(d: Optional[dict]) -> Optional[CompoundReport]:
    if d is None:
        return None
    reports = [_report_from_dict(r) for r in (d.get("reports") or [])]
    return CompoundReport(reports=reports, uncalibrated_score=d.get("uncalibrated_score"))


def _record_from_dict(row: dict) -> DecisionRecord:
    """Rebuild the typed ``DecisionRecord`` from one body dict.

    The ``.get`` defaults are defensive: ``load_log`` calls this after the digest
    and chain checks and then enforces faithfulness (``rebuilt.to_dict()`` must
    round-trip to ``row`` and re-digest to ``record_id``), so a non-canonical row is
    rejected there rather than silently coerced here. For an accepted row the
    reconstruction is exact and the returned record's own
    :meth:`DecisionRecord.digest` equals its ``record_id``.
    """
    data = row.get("data") or {}
    snapshot = data.get("snapshot") or {}
    model = row.get("model") or {}
    harness = row.get("harness") or {}
    return DecisionRecord(
        action_type=row.get("action_type", ""),
        data=DataSpan(
            inputs=dict(data.get("inputs") or {}),
            retrieved=list(data.get("retrieved") or []),
            snapshot=DependencySnapshot(
                state=dict(snapshot.get("state") or {}),
                captured_at=snapshot.get("captured_at"),
            ),
            report=_report_from_dict(data.get("report")),
        ),
        model=ModelSpan(
            model_id=model.get("model_id", ""),
            decision_basis=model.get("decision_basis", ""),
            output=model.get("output"),
            report=_report_from_dict(model.get("report")),
        ),
        harness=HarnessSpan(
            action_type=harness.get("action_type", ""),
            arguments=dict(harness.get("arguments") or {}),
            cost=harness.get("cost", 0.0),
            report=_report_from_dict(harness.get("report")),
        ),
        compound=_compound_from_dict(row.get("compound")),
        decided_at=row.get("decided_at"),
        prev_digest=row.get("prev_digest"),
        record_id=row.get("record_id", ""),
    )


def _recompute_record_id(row: dict) -> str:
    """Content digest over the record body, the inverse of ``DecisionRecord.digest``:
    hash everything except ``record_id`` (the exact body FileSink commits to on write).
    Recomputing from the on-disk bytes, not from the reconstructed record, keeps the
    tamper check byte-exact and independent of reconstruction.
    """
    body = {k: v for k, v in row.items() if k != "record_id"}
    return _sha256(body)


def _is_number(x: Any) -> bool:
    """A real number, not a bool (``True`` is an ``int`` subclass)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_scalar_types(record: DecisionRecord, lineno: int, path: str) -> None:
    """Reject a re-signed record whose numeric scalar fields have the wrong type.

    The shape check catches extra / missing fields, but a dataclass does not enforce
    its annotations, so a value such as ``harness.cost = "x"`` round-trips through
    ``to_dict()`` and the digest unchanged and would otherwise load and poison the
    numeric replay / policy / combiner code downstream. Require the numeric fields those
    consumers do arithmetic on to be real numbers (bools excluded). ``None`` is allowed
    only where the schema default is ``None`` (``captured_at`` and the compound score);
    ``decided_at`` is required numeric, matching its timestamp default.
    """
    def fail(what: str) -> None:
        raise ValueError(
            f"Cannot load FileSink log: line {lineno} in {path} {what}; "
            "the body is not a well-typed DecisionRecord."
        )

    if not _is_number(record.harness.cost):
        fail("has a non-numeric harness.cost")
    for stage, rep in (
        ("data", record.data.report),
        ("model", record.model.report),
        ("harness", record.harness.report),
    ):
        if not _is_number(rep.score):
            fail(f"has a non-numeric {stage}.report.score")
    if not _is_number(record.decided_at):
        fail("has a non-numeric decided_at")
    if record.data.snapshot.captured_at is not None and not _is_number(
        record.data.snapshot.captured_at
    ):
        fail("has a non-numeric data.snapshot.captured_at")
    compound = record.compound
    if compound is not None:
        if compound.uncalibrated_score is not None and not _is_number(
            compound.uncalibrated_score
        ):
            fail("has a non-numeric compound.uncalibrated_score")
        for i, rep in enumerate(compound.reports or ()):
            if not _is_number(rep.score):
                fail(f"has a non-numeric compound.reports[{i}].score")


def load_log(path: str) -> List[DecisionRecord]:
    """Load a persisted ``FileSink`` JSONL run, verifying the full hash chain.

    Returns the records in log order. Raises ``ValueError`` and reads nothing
    further on the first sign of damage: a corrupt or non-object line, a line
    without a ``record_id``, a ``record_id`` that does not match the recomputed
    digest of its body, a ``prev_digest`` that does not link to the prior record
    (a dropped, reordered, duplicated, or forked run), or a hash-consistent body
    that is not the canonical ``DecisionRecord`` shape (extra, missing, or
    wrong-typed fields). An empty or all-blank file returns ``[]`` (a valid empty
    run). See the module docstring for the contract.
    """
    records: List[DecisionRecord] = []
    expected_prev: Optional[str] = None  # the genesis record has prev_digest=None
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue  # blank lines and an empty file are fine, as in FileSink

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot load FileSink log: corrupt JSONL line {lineno} in {path}."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Cannot load FileSink log: non-object JSONL line {lineno} in {path}."
                )
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"Cannot load FileSink log: line {lineno} without record_id in {path}."
                )

            # Tamper check: the id must be the content digest of the rest of the body.
            recomputed = _recompute_record_id(row)
            if recomputed != record_id:
                raise ValueError(
                    f"Cannot load FileSink log: record_id mismatch on line {lineno} in "
                    f"{path} (recorded {record_id[:12]}..., recomputed {recomputed[:12]}...); "
                    "the record body was altered after it was signed."
                )

            # Chain check: link to the immediately preceding record; genesis prev is None.
            prev_digest = row.get("prev_digest")
            if prev_digest != expected_prev:
                raise ValueError(
                    f"Cannot load FileSink log: broken or forked chain at line {lineno} in "
                    f"{path} (prev_digest {prev_digest!r} does not link to the prior record "
                    f"{expected_prev!r}); the run is incomplete, reordered, or tampered."
                )

            # Schema check: the row must be the canonical DecisionRecord shape, not
            # merely a hash-consistent blob. Rebuild it, then require the rebuilt
            # record to re-serialize to the same signed body and re-digest to the same
            # id. A row with extra, missing, or wrong-typed fields fails here even when
            # its raw bytes are internally hash-consistent, so the .get coercion in
            # _record_from_dict can never silently load a degenerate record. The
            # rebuild itself can raise on a wrong-typed field (a list where a dict is
            # expected); convert that to the same fail-closed ValueError.
            try:
                record = _record_from_dict(row)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot load FileSink log: line {lineno} in {path} is not the "
                    "canonical DecisionRecord shape."
                ) from exc
            if record.to_dict() != row or record.digest() != record_id:
                raise ValueError(
                    f"Cannot load FileSink log: line {lineno} in {path} is not the "
                    "canonical DecisionRecord shape (a hash-consistent body that is "
                    "not a DecisionRecord: extra, missing, or wrong-typed fields)."
                )
            _check_scalar_types(record, lineno, path)
            records.append(record)
            expected_prev = record_id
    return records
