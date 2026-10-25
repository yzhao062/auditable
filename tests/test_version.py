"""The version in pyproject.toml and ``auditable.__version__`` must agree (a release sets both)."""
import pathlib
import re

import auditable


def test_package_version_matches_pyproject():
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = re.search(r'(?m)^version = "([^"]+)"$', pyproject.read_text(encoding="utf-8")).group(1)
    assert auditable.__version__ == declared
