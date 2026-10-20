"""Environment checks and imports for the CatchBench parity harness.

The harness compares the library against CatchBench 0.1.2 and GRADE at the commit the
CatchBench golden board pins. It refuses to run when either reference is missing or at
another version, rather than skipping: a parity claim against the wrong reference is
worse than none.
"""
from __future__ import annotations

import os
import subprocess
import sys

GRADE_PIN = "3839a57ac165d58a807fce0a3ff38346732ee936"
CATCHBENCH_VERSION = "0.1.2"


class ParityEnvironmentError(RuntimeError):
    pass


def check_environment() -> dict:
    """Validate the reference stack and return its description. Call before importing
    ``catchbench``; it also sets the variables CatchBench's board runner sets."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ParityEnvironmentError(
            "run the harness with PYTHONHASHSEED=0 (the CatchBench board is scored that way)")
    grade_dir = os.environ.get("GRADE_DIR")
    if not grade_dir or not os.path.isdir(grade_dir):
        raise ParityEnvironmentError("set GRADE_DIR to a GRADE checkout at commit " + GRADE_PIN)
    head = subprocess.run(["git", "-C", grade_dir, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()
    if head != GRADE_PIN:
        raise ParityEnvironmentError(f"GRADE_DIR is at {head or 'an unknown commit'}, expected {GRADE_PIN}")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import catchbench
    except ImportError as exc:
        raise ParityEnvironmentError("install CatchBench 0.1.2 (pip install -e <catchbench>[full])") from exc
    version = getattr(catchbench, "__version__", None)
    if version is None:
        from importlib.metadata import version as _v

        version = _v("catchbench")
    if version != CATCHBENCH_VERSION:
        raise ParityEnvironmentError(f"catchbench {version} is installed, expected {CATCHBENCH_VERSION}")
    import catchbench._reuse  # noqa: F401  puts GRADE's src/ and experiment/ on sys.path
    import auditable

    return {"grade": head, "catchbench": version, "auditable": auditable.__version__,
            "auditable_path": os.path.dirname(auditable.__file__), "python": sys.version.split()[0]}
