import json

import pytest

from auditable import Action, DependencySnapshot, FileSink, MemorySink, audit


def test_memory_sink_chains_records():
    sink = MemorySink()
    snap = DependencySnapshot(state={"b": 1})
    with audit("a", snapshot=snap, sink=sink) as d:
        d.act(Action("a", {}, cost=1))
    with audit("b", snapshot=snap, sink=sink) as d:
        d.act(Action("b", {}, cost=1))
    assert len(sink.records) == 2
    assert sink.records[0].prev_digest is None
    assert sink.records[1].prev_digest == sink.records[0].record_id


def test_file_sink_persists_and_chains(tmp_path):
    path = str(tmp_path / "log.jsonl")
    sink = FileSink(path)
    snap = DependencySnapshot(state={"b": 1})
    with audit("a", snapshot=snap, sink=sink) as d:
        d.act(Action("a", {}, cost=1))
    first_id = d.record.record_id
    with audit("b", snapshot=snap, sink=sink) as d:
        d.act(Action("b", {}, cost=1))

    with open(path, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert len(lines) == 2
    assert lines[0]["record_id"] == first_id
    assert lines[1]["prev_digest"] == first_id


def test_file_sink_resumes_chain_from_existing_file(tmp_path):
    path = str(tmp_path / "log.jsonl")
    snap = DependencySnapshot(state={"b": 1})
    with audit("a", snapshot=snap, sink=FileSink(path)) as d:
        d.act(Action("a", {}, cost=1))
    first_id = d.record.record_id
    # A fresh sink over the same file should chain onto the last record.
    with audit("b", snapshot=snap, sink=FileSink(path)) as d:
        d.act(Action("b", {}, cost=1))
    assert d.record.prev_digest == first_id


def test_file_sink_corrupt_tail_raises(tmp_path):
    path = str(tmp_path / "log.jsonl")
    snap = DependencySnapshot(state={"b": 1})
    with audit("a", snapshot=snap, sink=FileSink(path)) as d:
        d.act(Action("a", {}, cost=1))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{bad\n")  # a corrupt partial line
    # Fail closed rather than silently start a new chain in the same log.
    with pytest.raises(ValueError):
        FileSink(path)


def test_file_sink_empty_file_is_ok(tmp_path):
    path = str(tmp_path / "log.jsonl")
    open(path, "w", encoding="utf-8").close()  # empty file
    snap = DependencySnapshot(state={"b": 1})
    with audit("a", snapshot=snap, sink=FileSink(path)) as d:
        d.act(Action("a", {}, cost=1))
    assert d.record.prev_digest is None


def test_file_sink_blank_trailing_lines_are_ok(tmp_path):
    path = str(tmp_path / "log.jsonl")
    snap = DependencySnapshot(state={"b": 1})
    with audit("a", snapshot=snap, sink=FileSink(path)) as d:
        d.act(Action("a", {}, cost=1))
    first_id = d.record.record_id
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n   \n")  # blank trailing lines must not reset the chain
    with audit("b", snapshot=snap, sink=FileSink(path)) as d:
        d.act(Action("b", {}, cost=1))
    assert d.record.prev_digest == first_id
