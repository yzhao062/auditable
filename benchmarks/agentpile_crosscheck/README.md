# agentpile Cross-Check

`run_crosscheck.py` checks that `auditable.from_agentpile(run, deps="grade", convention="grade")` reproduces the step lists, dependencies, failure labels and method inputs that GRADE's converters (commit `3839a57`) produce for the same runs. It is a development harness and is not part of the wheel.

It needs four inputs:

- the GRADE expected outputs, one JSONL file per corpus, made from GRADE `3839a57` independently of agentpile. `baseline.json` pins the SHA-256 of each decompressed file and its run, step and edge counts, and the harness refuses a file that differs;
- agentpile's exception list, `tests/fixtures/grade/exceptions.json` in an agentpile checkout;
- the pinned dataset files, one folder per dataset and revision, as `~/.cache/agentpile` holds them after `agentpile.load`. `baseline.json` also pins the source files each comparison reads, the run files the expected runs name plus the files a reader joins into every run, AgentRewardBench's `data/annotations.csv` and SWE-rebench's `tools.json` (their number and one SHA-256 over the per-file hashes), and the harness refuses inputs that differ;
- GRADE checked out at `3839a57` (`GRADE_DIR`), whose `build_graph` builds the reference graphs, and CatchBench 0.1.2 for the PRE comparison.

```bash
export GRADE_DIR=/path/to/grade   # checked out at 3839a57
export PYTHONHASHSEED=0
python benchmarks/agentpile_crosscheck/run_crosscheck.py --expected /path/to/expected \
    --exceptions /path/to/agentpile/tests/fixtures/grade/exceptions.json --data ~/.cache/agentpile --out report/
```

For every run an expected file names, the harness compares the steps (`idx`, `agent`, `kind`, `deps`). A difference passes only when the exception list holds the same run, step, field and values, and an unused entry fails. On each run without a step difference it then compares `who_when_step_matrix`, the six GRADE layers, and `to_pyg` of the graph, read through the methods' own entry points on the adapter side and from GRADE's graph on the reference side. AgentRewardBench runs whose only difference is the listed E17 tail (two steps GRADE makes for the page after the last action) are compared after those steps are removed from the reference, and are counted apart. Two controls (the other shared-resource setting, and the dependencies removed) must each make the comparison fail on some sampled run. The Who&When step labels are compared on GRADE's 126-run localization population, and CatchBench PRE rows on the fields the config methods read and on the findings of the seven rule methods. The report records the per-file hashes (`inputs-<corpus>.tsv`), the keys of loaded runs GRADE's selection left out (`not-expected-<corpus>.txt`; the harness does not port GRADE's filters, caps and balancing, so it records no reason per run), and the feature column names.
