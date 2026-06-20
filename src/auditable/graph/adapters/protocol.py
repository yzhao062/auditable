"""The Adapter extension protocol and a small shared base.

``Adapter`` is the public extension point: a source-agnostic ``to_steps(source)``
that returns the typed :class:`~auditable.graph.session.Step` list the
``SessionGraph`` consumes, plus a stable ``name`` and ``version`` so the
ingestion source and version travel with the steps it produced. It is the
analog of subclassing pyod's ``BaseDetector``: user code conforms to this protocol
to teach auditable a new corpus or framework. The concrete adapters in this package
are versioned reference adapters, pinned by ``name`` / ``version`` so a given
version's behavior is stable to call by that versioned name (a later ``v2`` ships
alongside, never in place of, ``v1``); they are used directly in the examples, while
the protocol, not any one concrete class, is the contract third-party code
implements. The same ``Step`` target keeps every source on one representation.
"""
from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from ..session import Step

__all__ = ["Adapter"]


@runtime_checkable
class Adapter(Protocol):
    """Map one source into typed steps. The stable public ingestion contract.

    An adapter carries a ``name`` and a ``version`` (so a produced graph records
    which adapter built it) and implements ``to_steps``, which turns a source
    (a public-corpus trajectory, an auditable run's own records, or a live
    framework stream) into the ``Step`` list ``SessionGraph.from_steps`` reads.
    The protocol is ``runtime_checkable``, so an instance with these three
    members satisfies ``isinstance(obj, Adapter)`` without subclassing.
    """

    name: str
    version: str

    def to_steps(self, source: Any) -> List[Step]:
        ...


class _BaseAdapter:
    """Internal convenience base: a combined ``id`` and a callable shorthand.

    A concrete adapter sets ``name`` and ``version`` and implements ``to_steps``.
    ``id`` joins them (``<name>_<version>``), the label written into edge evidence
    so a produced dependency edge names the adapter and version that built it.
    ``__call__`` forwards to ``to_steps``, so an adapter instance is callable on
    its source (``adapter(source)`` equals ``adapter.to_steps(source)``).
    """

    name: str = ""
    version: str = ""

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"

    def to_steps(self, source: Any) -> List[Step]:
        raise NotImplementedError

    def __call__(self, source: Any) -> List[Step]:
        return self.to_steps(source)
