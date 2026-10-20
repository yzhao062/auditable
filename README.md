<div align="center">

# auditable

**A library of auditing methods for AI agents: configuration audits before a run, online monitors while it runs, and failure detection and fault localization after it, with one interface per family.**

[![PyPI](https://img.shields.io/pypi/v/auditable.svg)](https://pypi.org/project/auditable/)
[![Python](https://img.shields.io/pypi/pyversions/auditable.svg)](https://pypi.org/project/auditable/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/yzhao062/auditable/actions/workflows/ci.yml/badge.svg)](https://github.com/yzhao062/auditable/actions/workflows/ci.yml)

[Quick Start](#quick-start) · [Methods](#methods) · [Install](#install) · [Reproducing CatchBench](#reproducing-catchbench) · [Adding a Method](#adding-a-method) · [Citation](#citation)

</div>

`auditable` collects methods that audit LLM agents at three points: an agent configuration before it runs (PRE), a run while it streams (LIVE), and a finished run (POST). Methods are grouped into four families, and each family has one interface, in the way PyOD gives outlier detectors one interface. The library holds baselines from the literature (PyOD and PyGOD detectors, GUARDIAN, G-Safeguard, the Who&When LLM-judge protocols, OWASP-style configuration rules) beside the methods introduced with GRADE and CatchBench.

[CatchBench](https://github.com/yzhao062/catchbench) is the benchmark for these methods: the tasks, corpora, evaluation protocols, and metrics live there. Every method here that appears on the CatchBench board reproduces its row sample by sample, and the full board reruns byte-identical with those rows computed by this library ([Reproducing CatchBench](#reproducing-catchbench)).

## Quick Start

A trace is a list of steps, each naming the acting agent, the kind of step, and the earlier steps it relied on. The same trace feeds every family.

```python
from auditable.methods import get_method, list_methods

trace = [
    {"idx": 0, "agent": "planner", "kind": "decision",  "deps": []},
    {"idx": 1, "agent": "coder",   "kind": "tool_call", "deps": [0]},
    {"idx": 2, "agent": "coder",   "kind": "decision",  "deps": [1]},
    {"idx": 3, "agent": "tester",  "kind": "tool_call", "deps": [2]},
    {"idx": 4, "agent": "coder",   "kind": "decision",  "deps": [0]},   # reads state four steps old
]

# LIVE: a causal alarm per arriving step (step t sees steps 0..t only)
monitor = get_method("online.span_z")()
print(monitor.step_alarm_scores(trace))           # the stale read at step 4 is the peak

# POST: which step is the decisive mistake, most suspected first
locator = get_method("step.dep_anomaly")()
print(locator.rank(trace))

# POST: run-level failure detection, fitted on labelled runs (1 = failed)
detector = get_method("run.lr_flatdep")()
# detector.fit(train_traces, train_labels).decision_function(new_traces)

print([card.method_id for card in list_methods(family="step")])
```

[`examples/example_methods.py`](examples/example_methods.py) runs one method from each family on synthetic traces, with no benchmark data.

## Method Families

| Family | When | Input | Interface | Output |
|---|---|---|---|---|
| `config` | PRE | an agent configuration (`AgentConfig`) | `audit(config)` | a finding: flag or abstain, the excess capabilities, the rules that fired |
| `run` | POST | a list of traces | `fit`, `decision_function`, `fit_score` | one score per run; higher means more likely failed |
| `step` | POST | a list of failed traces | `fit`, `step_scores_batch`, `step_scores`, `rank`; `predict_step` for single-step methods | one score per step; higher means more likely the decisive mistake |
| `online` | LIVE | one streaming trace | `step_alarm_scores`, `prefix_score` | one alarm per step, causal |

Every method also declares how it is fitted: `stateless` (rules), `inductive` (fit once, then score unseen inputs), or `transductive` (fit and score one batch together, as PyGOD's autoencoders do). Its `MethodCard` records the fit scope, the trace fields it reads, the extras it needs, where it comes from, and the CatchBench row it reproduces. The [method contract](docs/methods-contract.md) defines all of this on one page.

## Methods

The table is generated from the registry (`auditable.methods.method_table()`); a test keeps it current.

<!-- methods:start -->
| Method | Family | Fit | Fidelity | Extras | CatchBench row |
|---|---|---|---|---|---|
| `config.flag_risky_perms` | config (PRE) | stateless | original | none | flag_risky_perms |
| `config.llm_needed` | config (PRE) | stateless | faithful | none | llm_judge_needed(llama-3.3-70b) |
| `config.owasp_asi_combined` | config (PRE) | stateless | original | none | owasp_asi_combined |
| `config.owasp_excess_functionality` | config (PRE) | stateless | original | none | owasp_excess_functionality |
| `config.owasp_excess_permissions` | config (PRE) | stateless | original | none | owasp_excess_permissions |
| `config.owasp_privilege_escalation` | config (PRE) | stateless | original | none | owasp_privilege_escalation |
| `config.sensitive_access` | config (PRE) | stateless | original | none | sensitive_access |
| `config.unrequested_high_impact` | config (PRE) | stateless | original | none | unrequested_high_impact |
| `run.gsafeguard_lite` | run (POST) | inductive, supervised | simplified | graph-ad | g-safeguard (sup GNN) |
| `run.guardian_lite` | run (POST) | transductive | simplified | graph-ad | guardian (recon-AE) |
| `run.lr_flat` | run (POST) | inductive, supervised | original | tabular-ad | size (flat) |
| `run.lr_flatdep` | run (POST) | inductive, supervised | original | tabular-ad | auditable (size+deps) |
| `run.lr_full` | run (POST) | inductive, supervised | original | tabular-ad | full |
| `run.pygod_anomalydae` | run (POST) | transductive | wrapper | graph-ad | pygod-anomalydae |
| `run.pygod_conad` | run (POST) | transductive | wrapper | graph-ad | pygod-conad |
| `run.pygod_dominant` | run (POST) | transductive | wrapper | graph-ad | pygod (graph AD) |
| `run.pygod_gaan` | run (POST) | transductive | wrapper | graph-ad | pygod-gaan |
| `run.pyod_copod` | run (POST) | inductive | wrapper | tabular-ad | pyod-copod |
| `run.pyod_ecod` | run (POST) | inductive | wrapper | tabular-ad | pyod-flatten (ECOD); pyod (ECOD) |
| `run.pyod_hbos` | run (POST) | inductive | wrapper | tabular-ad | pyod-hbos |
| `run.pyod_iforest` | run (POST) | inductive | wrapper | tabular-ad | pyod-iforest |
| `run.pyod_knn` | run (POST) | inductive | wrapper | tabular-ad | pyod-knn |
| `run.pyod_lof` | run (POST) | inductive | wrapper | tabular-ad | pyod-lof |
| `run.random` | run (POST) | stateless | original | tabular-ad | random |
| `step.blast` | step (POST) | stateless | original | tabular-ad | auditable (blast) |
| `step.degree` | step (POST) | stateless | original | tabular-ad | degree |
| `step.dep_anomaly` | step (POST) | stateless | original | tabular-ad | auditable (dep-anomaly) |
| `step.exec_rank` | step (POST) | inductive, supervised | original | tabular-ad | exec-rank (sup.) |
| `step.has_dep` | step (POST) | stateless | original | tabular-ad | has-dep (control) |
| `step.llm_all_at_once` | step (POST) | stateless | faithful | tabular-ad | llm-judge all-at-once |
| `step.llm_binary_search` | step (POST) | stateless | faithful | none | llm-judge binary-search |
| `step.llm_step_by_step` | step (POST) | stateless | faithful | none | llm-judge step-by-step |
| `step.max_span` | step (POST) | stateless | original | tabular-ad | max-span (control) |
| `step.position_early_rule` | step (POST) | stateless | original | tabular-ad | position |
| `step.position_supervised` | step (POST) | inductive, supervised | original | tabular-ad | position |
| `step.pygod_dominant_sg4` | step (POST) | transductive | wrapper | graph-ad | pygod (graph AD) |
| `step.pygod_dominant_ww8` | step (POST) | transductive | wrapper | graph-ad | pygod (graph AD) |
| `step.random` | step (POST) | stateless | original | tabular-ad | random |
| `online.dep_count_z` | online (LIVE) | stateless | original | tabular-ad | dep-count (control) |
| `online.dep_span` | online (LIVE) | stateless | original | tabular-ad | dep-span (online) |
| `online.random` | online (LIVE) | stateless | original | tabular-ad | random |
| `online.raw_span` | online (LIVE) | stateless | original | tabular-ad | raw-span |
| `online.span_z` | online (LIVE) | stateless | original | tabular-ad | auditable (span z-score) |
<!-- methods:end -->

Fidelity: `wrapper` calls the third-party library, `faithful` is a verbatim port, `simplified` (classes named `...Lite`) is a reduced port whose docstring lists what was simplified, and `original` was introduced with GRADE or CatchBench. The CatchBench row is the board label; a `control` or `oracle` there describes the board's use of the method, not its value here.

## Install

The method library is not on PyPI yet: the current release there (0.2.1) predates it. Until 0.3.0 is released, install from a clone of this repository:

```bash
git clone https://github.com/yzhao062/auditable.git
cd auditable
pip install .                    # core: the registry and the configuration rules, no third-party dependencies
pip install ".[tabular-ad]"      # run, step, and online methods on trace features (NumPy, scikit-learn, PyOD, NetworkX)
```

The LLM judges take any `prompt -> reply` function as their backend; `OpenAICompatibleBackend` reaches an OpenAI-compatible endpoint with the standard library alone.

The graph methods (PyGOD detectors, GUARDIAN-Lite, G-Safeguard-Lite) need torch, PyTorch Geometric, PyGOD, and a sampling backend built for the installed torch, which pip cannot resolve from the extra. Install torch and the matching `pyg-lib` wheel first:

```bash
pip install "torch==2.12.1" --index-url https://download.pytorch.org/whl/cpu
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.12.1+cpu.html
pip install ".[graph-ad]"
python -c "import torch_geometric.typing as t; assert t.WITH_PYG_LIB"
```

## Traces and Inputs

Methods read traces through `auditable.featurize.normalize_trace`, which accepts a list of step dicts, a run dict (`{"steps": ..., "texts": ...}`), a typed NetworkX graph from `auditable.graph.build_graph`, or a `SessionGraph`. Steps carry `idx`, `agent`, `kind` (`"decision"` or `"tool_call"`), and optionally `deps` and `text`; a `SessionGraph` also keeps parallel branches and merges of the execution order. A trace without dependency annotations can be read with `dependency="full_context"`, the assumption that each step relied on every earlier step. A method that needs a field the trace lacks (step text for an LLM judge, for example) raises an error naming it.

`auditable.featurize` exposes the features the methods compute, so they can be inspected or reused: the GRADE structural layers (`grade_layer`, `grade_layer_matrix`), the Who&When step matrix (`who_when_step_matrix`), the step dependency graph and its statistics (`step_graph`, `span_series`, `run_max_span`), and graph encodings for PyGOD (`to_pyg`).

## Reproducing CatchBench

[`benchmarks/catchbench_parity`](benchmarks/catchbench_parity) checks the library against CatchBench 0.1.2 and GRADE at commit `3839a57`, the versions the CatchBench board pins. It is a development harness and is not part of the wheel.

```bash
export GRADE_DIR=/path/to/grade   # checked out at 3839a57
export PYTHONHASHSEED=0
python benchmarks/catchbench_parity/run_parity.py checks                     # per-sample comparisons
python benchmarks/catchbench_parity/run_parity.py board --out board.txt      # the full board through the library
python benchmarks/catchbench_parity/run_parity.py board --perturb --out perturbed.txt
python benchmarks/catchbench_parity/run_parity.py consumption board.txt perturbed.txt
```

The checks compare every migrated method with its CatchBench or GRADE original on the real corpora, sample by sample and bit for bit: per-run scores, per-step scores, out-of-fold predictions, configuration findings, and the features underneath them. The board run replaces CatchBench's method factories with library adapters, so the main tables and every breakdown and robustness table are computed by the library, while the original scoring functions of migrated methods are replaced by functions that raise. Its output must match the board computed before migration byte for byte. The consumption run perturbs every library output and confirms that each migrated row changes and no other row does.

## Adding a Method

Subclass the family base class in `auditable.base`, give it a `MethodCard`, and register it with `@register`. The [method contract](docs/methods-contract.md) lists the conventions: input forms, label alignment for supervised step methods, output order, optional dependencies, the tests a method needs, and how to add a parity check when the method reproduces a CatchBench row.

## Records and Graph Analysis

The package also contains the modules it was first built around. `audit` and `replay` capture an agent decision with the dependency state it relied on and re-decide it against later state; `analyze_run` ranks the steps of a finished run by structural blast share; `analyze_plan` lints a declared plan; `LiveSession` scores a growing run; and `auditable.integrations` captures observed dependency edges from LangGraph, LangChain, MCP servers, or any loop through `TouchRecorder`. See [`examples/README.md`](examples/README.md) and the [documentation](https://auditable-ai.readthedocs.io/).

Records are content-addressed and hash-chained, and that property is limited. A signature or digest authenticates only the entries that were submitted, a hash chain exposes a later edit only when its head is anchored with an independent party, and neither reveals an event that was never written. The chain therefore does not establish that a log is complete.

### Legacy

`ActionGate`, `ReferenceLedger` (the in-process reference rail), and `DataAuditor` are deprecated as of 0.3 and emit a `DeprecationWarning` when constructed. They remain importable and will be removed in a later release. Tabular anomaly detection on dependency snapshots belongs in [PyOD](https://github.com/yzhao062/pyod).

## Related Projects

- [CatchBench](https://github.com/yzhao062/catchbench): the benchmark for these methods (arXiv:2608.22808).
- [GRADE](https://arxiv.org/abs/2606.22741): the typed two-layer graph of agent execution and dependency that the structural features read.
- [awesome-auditable-ai](https://github.com/yzhao062/awesome-auditable-ai): a reading list on AI agent reliability and auditing.

## Citation

If you use `auditable`, please cite the work behind the methods you use (each `MethodCard.cite`) and, for the structural features and the benchmark, GRADE and CatchBench:

```bibtex
@article{zhao2026grade,
  title   = {GRADE: Graph Representation of LLM Agent Dependency and Execution},
  author  = {Zhao, Yue},
  journal = {arXiv preprint arXiv:2606.22741},
  year    = {2026}
}

@misc{zhao2026catchbench,
  title         = {{CatchBench}: When Can an Agent Failure Be Caught?},
  author        = {Yue Zhao},
  year          = {2026},
  eprint        = {2608.22808},
  archivePrefix = {arXiv}
}
```

## License

[Apache-2.0](LICENSE). Ported code keeps the notice of its source (GRADE and CatchBench, both MIT License) at the top of each module.
