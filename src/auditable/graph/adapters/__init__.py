"""Source-agnostic ingestion adapters: map a source into typed Steps.

The :class:`Adapter` protocol is the public extension point (the analog of
subclassing pyod's ``BaseDetector``). The concrete adapters here are versioned
reference adapters: pinned by name and version, stable to call by that versioned
name, and used in the examples; the protocol, not any concrete class, is the
contract user code implements.
Each adapter turns one source into the typed ``Step`` list that
``auditable.graph.SessionGraph`` and the structural scorer consume, so a public
corpus, an auditable run's own records, and (in v0.3b) a live framework stream
converge on one representation.

Adapters shipped this round:

- :data:`tau_bench_prior_db_reads_v1` (corpus): a tau-bench-style trajectory to
  steps, with each consequential write depending on every prior DB read, graded
  ``OBSERVED`` but marked ``modeled`` in evidence (a conservative prior-read
  upper bound, not a causal label). Pure: messages in, steps out, no download.
- :data:`own_record_v1` (own records): a chain of signed ``DecisionRecord``s to
  steps, execution edges from the ``prev_digest`` backbone and model attributes
  on each node, with sparse ``DECLARED`` dependency edges (no fabricated observed
  edges) until the v0.3b resource-touch contract lands.
"""
from __future__ import annotations

from .protocol import Adapter
from .own_record import OwnRecordAdapter, own_record_v1
from .tau_bench import TauBenchPriorDBReadsAdapter, tau_bench_prior_db_reads_v1

__all__ = [
    "Adapter",
    "TauBenchPriorDBReadsAdapter",
    "tau_bench_prior_db_reads_v1",
    "OwnRecordAdapter",
    "own_record_v1",
]
