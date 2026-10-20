# auditable

`auditable` is a library of auditing methods for AI agents. It covers an agent configuration before a run (PRE), a run while it streams (LIVE), and a finished run (POST), with four method families and one interface per family:

| Family | When | Interface | Output |
|---|---|---|---|
| `config` | PRE | `audit(config)` | a finding: flag or abstain, excess capabilities, rules fired |
| `run` | POST | `fit`, `decision_function`, `fit_score` | one failure score per run |
| `step` | POST | `fit`, `step_scores_batch`, `step_scores`, `rank`, `predict_step` | one score per step, for locating the decisive mistake |
| `online` | LIVE | `step_alarm_scores`, `prefix_score` | one causal alarm per arriving step |

The library holds baselines from the literature (PyOD and PyGOD detectors, GUARDIAN, G-Safeguard, the Who&When LLM-judge protocols, OWASP-style configuration rules) beside the methods introduced with GRADE and CatchBench. [CatchBench](https://github.com/yzhao062/catchbench) is the benchmark: tasks, corpora, protocols, and metrics live there, and the methods here reproduce their CatchBench rows sample by sample.

```python
from auditable.methods import get_method, list_methods

trace = [
    {"idx": 0, "agent": "planner", "kind": "decision",  "deps": []},
    {"idx": 1, "agent": "coder",   "kind": "tool_call", "deps": [0]},
    {"idx": 2, "agent": "coder",   "kind": "decision",  "deps": [0]},
]
print(get_method("online.span_z")().step_alarm_scores(trace))
print([card.method_id for card in list_methods(family="run")])
```

## Where to Start

- [Method Contract](methods-contract.md): the families, fit scopes, input forms, outputs, the method card, and how to add a method.
- The [README](https://github.com/yzhao062/auditable#methods) lists every registered method with its source and CatchBench row.
- [`examples/example_methods.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_methods.py) runs one method from each family on synthetic traces.

## Records and Graph Analysis

The package also contains the modules it was first built around: decision records with `audit` and `replay`, the structural graph analyses `analyze_run` and `analyze_plan`, `LiveSession`, and capture integrations for LangGraph, LangChain, MCP, and any loop through `TouchRecorder`. The pages under [Lifecycle](lifecycle.md), [Quickstart](quickstart.md), PRE, LIVE, POST, and Reference describe them.

Records are content-addressed and hash-chained, and that property is limited: a digest authenticates only the records that were written, a chain exposes a later edit only when its head is anchored with an independent party, and neither reveals a decision that was never recorded.

`ActionGate`, `ReferenceLedger`, and `DataAuditor` are deprecated as of 0.3 and will be removed in a later release.

## Install

The method library is not on PyPI yet: the current release there (0.2.1) predates it. Until 0.3.0 is released, install from a clone of the repository:

```bash
git clone https://github.com/yzhao062/auditable.git
cd auditable
pip install .                  # core: the registry and the configuration rules
pip install ".[tabular-ad]"    # run, step, and online methods on trace features
pip install ".[graph-ad]"      # graph methods; also install torch and a matching pyg-lib wheel (see the README)
pip install ".[graph]"         # the structural graph analyses (analyze_run, analyze_plan)
```
