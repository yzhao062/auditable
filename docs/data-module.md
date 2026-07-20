# Data Module: Dependency-State Anomaly

`DataAuditor` scores the dependency snapshot a decision relied on. The input is
`DependencySnapshot.state` together with `DependencySnapshot.captured_at`, and the
output is one normalized `Report` leaf. When attached to a decision, that leaf is
stored as `DecisionRecord.data.report` and bound into the signed decision record with
the snapshot it describes.

This is a shipped module API, not a promoted headline capability. The fitted path can
help order work before replay, but benchmark-grade evidence is still pending. See
[Honesty and Promotion Status](#honesty-and-promotion-status) for that boundary.

## Install

Freshness mode is available in the dependency-free core install. Fitting and learned
scoring need the `anomaly` extra:

```bash
pip install "auditable[anomaly]"
```

PyOD brings NumPy, SciPy, and scikit-learn. Without the extra, `DataAuditor` still
works in freshness mode.

## Learned Mode

Fit the auditor on snapshots that represent normal dependency state, then assess a
new snapshot:

```python
from auditable import DataAuditor, DependencySnapshot

normal = [
    DependencySnapshot(
        state={
            "budget_remaining": 8_000,
            "allow_list_version": 7,
            "config_version": "cfg-12",
            "allow_list": ["acme", "globex"],
        },
        captured_at=1_700_000_000.0,
    ),
]

auditor = DataAuditor().fit(normal, now=1_700_000_300.0)
report = auditor.assess(normal[0], now=1_700_000_300.0)
```

ECOD is the default detector. In learned mode, `Report.score` is in `[0, 1]` and
comes from the detector's predicted outlier probability. `Report.evidence` includes
`mode: learned`, the detector name, and the most unusual feature names under
`top_features`. The attribution ranks standardized distance from the training mean;
it identifies features to inspect, not causes.

Any PyOD detector can be injected with `DataAuditor(detector=...)`. The auditor fits
and queries the injected detector through the PyOD detector interface. This is the
PyOD-swap property, so callers can change the detector without changing the snapshot
schema or `Report` contract.

## Feature Schema

The auditor builds one fixed feature schema from the training snapshots:

| Snapshot Value | Encoded Feature |
|---|---|
| Numeric key | The numeric value. |
| List value | The list length. |
| Categorical string | Rarity, computed as `1 - training frequency`; an unseen category encodes to `1.0`. |
| `captured_at` | Snapshot age at the supplied `now`, as one feature among several. |

The same schema and feature order are used for assessment. Snapshot age participates
in the learned state vector, but it is not the whole vector.

## Fallback Semantics

Fallback is explicit and remains useful without PyOD:

| Condition | `Report.evidence["mode"]` | `Report.evidence["reason_code"]` |
|---|---|---|
| Auditor has not been fitted | `freshness_fallback` | `freshness_rule` |
| Learned assessment raises an error | `freshness_fallback` | `learned_error:<type>` |

The unfitted path uses the v0.1 freshness rule: it compares snapshot age with
`max_age_seconds`, caps the score at `1.0`, and flags a snapshot at or beyond the
budget as `stale`. If learned assessment fails, the same rule produces the report and
`<type>` names the exception class. The mode always lands in `Report.evidence`, so a
consumer can distinguish a learned score from either fallback case.

## Fitting From Captured History

Two helpers fit the same schema from captured decision history:

```python
DataAuditor.fit_records(records, *, now=None)
DataAuditor.fit_from_log(path, *, now=None)
```

`fit_records` accepts an iterable of `DecisionRecord`s and uses only records whose
dependency snapshots are non-empty. It raises `ValueError` when none are available.
The optional `now` fixes the reference time used to compute snapshot age.

`fit_from_log` reads a `FileSink` JSONL path through the fail-closed loader, then calls
`fit_records`. The loader verifies record digests, chain links, and record shape before
any snapshot is used for fitting; damaged or altered logs raise instead of yielding a
partial training corpus.

## Evaluation Tooling

`auditable.data.inject` builds labeled snapshot faults for evaluation:

```python
from auditable.data.inject import (
    Injection,
    InjectionResult,
    CorpusRow,
    inject_stale,
    inject_drift,
    inject_ood,
    build_injected_corpus,
)
```

The public contracts are:

```python
inject_stale(snapshot, *, min_age_seconds=7*86400.0,
             max_age_seconds=30*86400.0, now=None, rng=None)
inject_drift(snapshot, *, keys=None, fraction=(0.3, 0.9),
             direction="down", rng=None)
inject_ood(snapshot, *, keys=None, rng=None)
build_injected_corpus(snapshots, *, kinds=("stale", "drift", "ood"),
                      rate=0.5, seed=0, now=None)
```

`Injection` is a frozen dataclass with `kind`, `key`, `original`, `injected`, and
`detail`. `InjectionResult` carries a faulted deep copy in `snapshot` and a list in
`injections`. `CorpusRow` carries `snapshot`, a `label` of `"clean"`, `"stale"`,
`"drift"`, or `"ood"`, and its `injections`.

Injection-site labels are fixed when the corpus is constructed and are independent of
any detector, so the labels are non-circular. Signed records are out of scope because
changing a signed record would break its digest; these helpers operate on dependency
snapshots and return faulted deep copies.

## Replay and Detector Roles

Replay re-reads the live value deterministically; the detector triages before replay.
Guessing staleness from behavior alone, without re-reading the source, is hard, and
our internal measurements on streaming agent runs bear that out. The detector
therefore feeds triage, and replay remains the decision mechanism.

## Honesty and Promotion Status

The fitted `DataAuditor` is a shipped module API and remains unpromoted. The concept
demo in `experiment/data_state_anomaly.py` uses synthetic drift over real ULB
transaction amounts. It also keeps detector inputs separate from replay-derived
labels: the detector sees dependency state, while replay produces labels from the
action cost and live budget.

That experiment is concept-demo evidence, not benchmark-grade validation. The pending
benchmark-grade evidence target is AuditableBench-Gold on a named-value substrate.
Promotion waits for that evidence.
