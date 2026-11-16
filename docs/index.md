# auditable

`auditable` is a library of auditing methods for AI agents, organized by when they audit: an agent configuration before a run (PRE), a run while it streams (LIVE), and a finished run (POST). Methods fall into four families, each with one interface, and every task has a leaderboard.

| Stage | Family | Question | Interface | Output | Leaderboards |
|---|---|---|---|---|---|
| PRE | `config` | Is the agent configuration over-privileged before it runs? | `audit(config)` | a finding: flag or abstain, excess capabilities, rules fired | `pre/over_privilege` |
| LIVE | `online` | Can the failure be caught while the run streams? | `step_alarm_scores`, `prefix_score` | one causal alarm per arriving step | `live/swegym`, `live/tau` |
| POST | `run` | Did the run fail? | `fit`, `decision_function`, `fit_score` | one failure score per run | `detect/swegym`, `detect/tau` |
| POST | `step` | Which step was the decisive mistake? | `fit`, `step_scores_batch`, `step_scores`, `rank`, `fit_step_scores`, `predict_step` | step scores, a ranking, or one predicted step, depending on the method | `localize/whoandwhen` |

The library holds baselines from the literature (PyOD and PyGOD detectors, GUARDIAN, G-Safeguard, the Who&When LLM-judge protocols, OWASP-style configuration rules) beside the methods introduced with GRADE and CatchBench. [CatchBench](https://github.com/yzhao062/catchbench) is the benchmark: tasks, corpora, protocols, and metrics live there, and the methods here reproduce their CatchBench rows sample by sample. The [leaderboards](leaderboard.md) rerun its boards with `python -m auditable.bench <task>` and rank a method you write on the same population and protocol.

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

- [Leaderboards](leaderboard.md): the six tasks, how to rerun a board, and how to place your own method on it.
- [Method Contract](methods-contract.md): the families, fit scopes, input forms, outputs, the method card, and how to add a method.
- The [README](https://github.com/yzhao062/auditable#implemented-algorithms) lists every registered method by stage, with its type, year, and main reference; `method_table(detailed=True)` adds the fit scope, fidelity, extras, and CatchBench row.
- [Agent Traces](agent-traces.md) reads public agent-trace datasets and trace files into the inputs the methods consume.
- `auditable.datasets.make_traces` generates toy traces with planted mistakes, and [`examples/example_methods.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_methods.py) runs one method from each family on them.

## Records and Graph Analysis

The package also contains the modules it was first built around: decision records with `audit` and `replay`, the structural graph analyses `analyze_run` and `analyze_plan`, `LiveSession`, and capture integrations for LangGraph, LangChain, MCP, and any loop through `TouchRecorder`. The pages under [Lifecycle](lifecycle.md), [Quickstart](quickstart.md), PRE, LIVE, POST, and Reference describe them.

Records are content-addressed and hash-chained, and that property is limited: a digest authenticates only the records that were written, a chain exposes a later edit only when its head is anchored with an independent party, and neither reveals a decision that was never recorded.

`ActionGate`, `ReferenceLedger`, and `DataAuditor` are deprecated as of 0.3 and will be removed in a later release.

## Install

```bash
pip install auditable                  # core: the registry and the configuration rules
pip install "auditable[tabular-ad]"    # run, step, and online methods on trace features
pip install "auditable[bench]"         # the leaderboards
pip install "auditable[agentpile]"     # read public datasets, trace files and captured runs through agentpile
pip install "auditable[graph-ad]"      # graph methods; also install torch and a matching pyg-lib wheel (see the install guide)
pip install "auditable[graph]"         # the structural graph analyses (analyze_run, analyze_plan)
```

The [install guide](install.md) lists the extras and covers the graph methods on Colab and on torch patch releases.
