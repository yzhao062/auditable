"""An extra counts as installed only when the library can name and find all of its modules."""
import importlib.util
import re
from pathlib import Path

from auditable.bench import _core


def test_every_pyproject_extra_has_a_probe():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    extras = set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", block, flags=re.M)) - {"dev", "docs", "all"}
    assert extras <= set(_core._EXTRA_MODULES), sorted(extras - set(_core._EXTRA_MODULES))


def test_unknown_extra_is_unavailable():
    assert _core._extra_available("no-such-extra") is False


def test_graph_ad_needs_a_sampling_backend(monkeypatch):
    present = {"torch", "torch_geometric", "pygod"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in present else None)
    assert _core._extra_available("graph-ad") is False
    present.add("torch_sparse")
    assert _core._extra_available("graph-ad") is True
