"""Tests for the offline FileSink JSONL log loader (auditable.graph.loader).

Two halves, matching the loader's contract:

- **round-trip**: write signed, chained records via ``FileSink``, load them back,
  and verify identity (record ids, chain links, every persisted field, and that
  each loaded record re-digests to its own id);
- **fail closed**: a tampered body, a dropped / reordered / forked record, and a
  corrupt / non-object / id-less line all raise, mirroring the strictness of
  ``FileSink._read_last_digest`` plus full-chain verification.
"""
import json

import pytest

from auditable import (
    Action,
    CompoundReport,
    DependencySnapshot,
    FileSink,
    Report,
    audit,
)
from auditable.graph.loader import load_log


def _write_run(path, n=3):
    """Write ``n`` chained records via FileSink; return their record ids in order."""
    snap = DependencySnapshot(state={"budget_remaining": 5000, "policy_id": "kyc-2026-03"})
    sink = FileSink(path)
    ids = []
    for i in range(n):
        with audit(f"act-{i}", snapshot=snap, sink=sink) as d:
            d.read(invoice=f"INV-{100 + i}")
            d.model("gpt-x", decision_basis=f"invoice INV-{100 + i} matches an approved PO")
            d.act(Action(f"act-{i}", {"amount": 100 + i}, cost=1.0 + i))
        ids.append(d.record.record_id)
    return ids


def _read_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [ln for ln in handle.read().splitlines() if ln.strip()]


def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        for ln in lines:
            handle.write(ln + "\n")


# --- round-trip --------------------------------------------------------------


def test_round_trip_identity_and_chain(tmp_path):
    path = str(tmp_path / "log.jsonl")
    ids = _write_run(path, n=3)
    records = load_log(path)

    assert [r.record_id for r in records] == ids
    # the chain links: genesis prev is None, each later record links to the prior id
    assert records[0].prev_digest is None
    assert records[1].prev_digest == records[0].record_id
    assert records[2].prev_digest == records[1].record_id
    # reconstruction is faithful: each loaded record re-digests to its stored id
    for r in records:
        assert r.digest() == r.record_id


def test_round_trip_preserves_all_fields(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=3)
    records = load_log(path)

    assert records[0].action_type == "act-0"
    assert records[0].data.inputs["invoice"] == "INV-100"
    assert records[0].data.snapshot.state["budget_remaining"] == 5000
    assert records[0].data.snapshot.state["policy_id"] == "kyc-2026-03"
    assert records[1].model.model_id == "gpt-x"
    assert records[1].model.decision_basis == "invoice INV-101 matches an approved PO"
    assert records[2].harness.action_type == "act-2"
    assert records[2].harness.arguments["amount"] == 102
    assert records[2].harness.cost == 3.0


def test_round_trip_preserves_reports_and_compound(tmp_path):
    path = str(tmp_path / "log.jsonl")
    snap = DependencySnapshot(state={"x": 1})
    sink = FileSink(path)
    with audit("a", snapshot=snap, sink=sink) as d:
        d.act(Action("a", {}, cost=2.0))
        report = Report(stage="harness", name="cap", score=0.7, flag="over_cap", reason="too big")
        d.attach(report)
        d.record.compound = CompoundReport.of([report])

    records = load_log(path)
    assert len(records) == 1
    rec = records[0]
    assert rec.harness.report.name == "cap"
    assert rec.harness.report.score == 0.7
    assert rec.harness.report.flag == "over_cap"
    assert rec.compound is not None
    assert rec.compound.uncalibrated_score == 0.7
    assert rec.compound.reports[0].name == "cap"
    # the report and compound are inside the signed body, so the digest still matches
    assert rec.digest() == rec.record_id


def test_single_genesis_record_loads(tmp_path):
    path = str(tmp_path / "log.jsonl")
    ids = _write_run(path, n=1)
    records = load_log(path)
    assert len(records) == 1
    assert records[0].prev_digest is None
    assert records[0].record_id == ids[0]


def test_empty_file_loads_as_empty_run(tmp_path):
    path = str(tmp_path / "log.jsonl")
    open(path, "w", encoding="utf-8").close()
    assert load_log(path) == []


def test_blank_lines_are_ignored(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=2)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n   \n")  # blank trailing lines must not break the load
    records = load_log(path)
    assert len(records) == 2


# --- fail closed -------------------------------------------------------------


def test_tampered_body_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=2)
    lines = _read_lines(path)
    row = json.loads(lines[0])
    row["harness"]["cost"] = 999.0  # edit the body but keep the recorded record_id
    lines[0] = json.dumps(row)
    _write_lines(path, lines)
    with pytest.raises(ValueError):
        load_log(path)


def test_broken_chain_dropped_record_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=3)
    lines = _read_lines(path)
    del lines[1]  # drop the middle record; record 3's prev_digest now dangles
    _write_lines(path, lines)
    with pytest.raises(ValueError):
        load_log(path)


def test_reordered_head_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=2)
    lines = _read_lines(path)
    lines.reverse()  # the first line is no longer the genesis (prev_digest != None)
    _write_lines(path, lines)
    with pytest.raises(ValueError):
        load_log(path)


def test_forked_chain_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=3)
    lines = _read_lines(path)
    lines.insert(2, lines[1])  # duplicate record 2: two records claim the same parent
    _write_lines(path, lines)
    with pytest.raises(ValueError):
        load_log(path)


def test_corrupt_json_line_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=1)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{bad\n")  # a corrupt partial line
    with pytest.raises(ValueError):
        load_log(path)


def test_non_object_line_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=1)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("[1, 2, 3]\n")  # valid JSON, but not a record object
    with pytest.raises(ValueError):
        load_log(path)


def test_missing_record_id_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=1)
    lines = _read_lines(path)
    row = json.loads(lines[0])
    del row["record_id"]
    lines[0] = json.dumps(row)
    _write_lines(path, lines)
    with pytest.raises(ValueError):
        load_log(path)


def test_self_consistent_non_canonical_row_raises(tmp_path):
    # A genesis row that is internally hash-consistent (record_id == digest of the
    # rest of the body) and chain-valid (prev_digest is None), but is NOT the
    # canonical DecisionRecord shape: an unexpected field, none of the record's
    # spans. The raw-byte hash and chain both pass, so fail-closed must reject it on
    # schema rather than coerce it via .get into a default-filled record whose own
    # digest no longer matches the recorded id.
    from auditable.record import _sha256

    path = str(tmp_path / "log.jsonl")
    body = {"prev_digest": None, "foo": "bar"}
    row = dict(body, record_id=_sha256(body))
    _write_lines(path, [json.dumps(row)])
    with pytest.raises(ValueError):
        load_log(path)


def test_extra_field_on_resigned_record_raises(tmp_path):
    # Take a real genesis record, inject an extra field, and RE-SIGN record_id over
    # the altered body so the raw-byte tamper check passes. The schema check must
    # still reject it: the rebuilt record drops the unknown field, so it no longer
    # round-trips to the signed body.
    from auditable.record import _sha256

    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=1)
    lines = _read_lines(path)
    row = json.loads(lines[0])
    row.pop("record_id", None)
    row["surprise"] = "x"
    row["record_id"] = _sha256(row)
    _write_lines(path, [json.dumps(row)])
    with pytest.raises(ValueError):
        load_log(path)


def test_resigned_record_with_wrong_typed_cost_raises(tmp_path):
    # A re-signed row whose harness.cost is a string round-trips through the shape and
    # digest checks (dataclasses do not enforce annotations), so fail-closed must
    # reject it on scalar type rather than load a record that poisons numeric replay.
    from auditable.record import _sha256

    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=1)
    lines = _read_lines(path)
    row = json.loads(lines[0])
    row.pop("record_id", None)
    row["harness"]["cost"] = "not-a-number"
    row["record_id"] = _sha256(row)
    _write_lines(path, [json.dumps(row)])
    with pytest.raises(ValueError):
        load_log(path)


def test_resigned_record_with_wrong_typed_score_raises(tmp_path):
    # The same scalar-type drift on a report score (numeric field a downstream combiner
    # reads) must also fail closed.
    from auditable.record import _sha256

    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=1)
    lines = _read_lines(path)
    row = json.loads(lines[0])
    row.pop("record_id", None)
    row["data"]["report"]["score"] = "high"
    row["record_id"] = _sha256(row)
    _write_lines(path, [json.dumps(row)])
    with pytest.raises(ValueError):
        load_log(path)


def test_resigned_record_with_null_decided_at_raises(tmp_path):
    # decided_at's schema default is a numeric timestamp, not None, so a re-signed row
    # with decided_at=null must fail closed even though it round-trips cleanly.
    from auditable.record import _sha256

    path = str(tmp_path / "log.jsonl")
    _write_run(path, n=1)
    lines = _read_lines(path)
    row = json.loads(lines[0])
    row.pop("record_id", None)
    row["decided_at"] = None
    row["record_id"] = _sha256(row)
    _write_lines(path, [json.dumps(row)])
    with pytest.raises(ValueError):
        load_log(path)
