# POST Examples

## Rank a Finished Run

[`examples/analyze_run.py`](https://github.com/yzhao062/auditable/blob/main/examples/analyze_run.py) ranks a finished airline-reservation trajectory and names the keystone: the one reservation read that both later writes rest on.

```python
from auditable import analyze_run
from auditable.graph.adapters import tau_bench_prior_db_reads_v1

report = analyze_run(run, adapter=tau_bench_prior_db_reads_v1)
print(report.keystone.idx, report.keystone.node_attrs["tool"])   # 2  get_reservation_details
```

It prints the ranked `AnalysisReport`: the adapter and scored state, the keystone (step 2 `get_reservation_details`, structural risk 0.333, two of six steps rest on it), the blast-share list, and the honesty notes (dependency edges modeled as a conservative prior-read bound, ranking uncalibrated, grounding empty for a corpus trace). Run it with `python examples/analyze_run.py` (needs the `graph` extra).

See the [POST Overview](post-analysis.md) for the ranked signal, the no-score gates, and the modeled-edge caveat.
