<div align="center">

<!-- logo-family -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yzhao062/auditable/main/docs/assets/brand/logo-dark.svg">
  <img src="https://raw.githubusercontent.com/yzhao062/auditable/main/docs/assets/brand/logo.svg" alt="auditable" width="350">
</picture>


# auditable

**A library of auditing methods for AI agents: configuration audits before a run, online monitors while it runs, and failure detection and fault localization after it, with one interface per family.**

[![PyPI](https://img.shields.io/pypi/v/auditable.svg)](https://pypi.org/project/auditable/)
[![Python](https://img.shields.io/pypi/pyversions/auditable.svg)](https://pypi.org/project/auditable/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/yzhao062/auditable/blob/main/LICENSE)
[![CI](https://github.com/yzhao062/auditable/actions/workflows/ci.yml/badge.svg)](https://github.com/yzhao062/auditable/actions/workflows/ci.yml)

[Quick Start](#quick-start) · [Real Agent Traces](#real-agent-traces) · [Methods](#methods) · [Install](#install) · [Reproducing CatchBench](#reproducing-catchbench) · [Adding a Method](#adding-a-method) · [Citation](#citation)

</div>

`auditable` collects methods that audit LLM agents at three points: an agent configuration before it runs (PRE), a run while it streams (LIVE), and a finished run (POST). Methods are grouped into four families, and each family has one interface, in the way PyOD gives outlier detectors one interface. The library holds baselines from the literature (PyOD and PyGOD detectors, GUARDIAN, G-Safeguard, the Who&When LLM-judge protocols, OWASP-style configuration rules) beside the methods introduced with GRADE and CatchBench.

[CatchBench](https://github.com/yzhao062/catchbench) is the benchmark for these methods: the tasks, corpora, evaluation protocols, and metrics live there. Every method here that appears on the CatchBench board reproduces its row sample by sample, and the full board reruns byte-identical with those rows computed by this library ([Reproducing CatchBench](#reproducing-catchbench)).

## Quick Start

Install with `pip install "auditable[tabular-ad]"` (see [Install](#install)), then:

```python
from sklearn.metrics import roc_auc_score

from auditable.datasets import make_traces
from auditable.methods import get_method

# 100 toy agent runs; each failed run holds one stale read (a step relying on old state)
traces, y, mistakes = make_traces(n_runs=100, contamination=0.5, random_state=0)
train, test = slice(0, 50), slice(50, 100)

# Did the run fail? Fit on labelled runs, then score new ones, as with a PyOD detector.
detector = get_method("run.lr_flatdep")()
detector.fit(traces[train], y[train])
scores = detector.decision_function(traces[test])   # one score per run, higher = more likely failed
print(f"run-level ROC-AUC: {roc_auc_score(y[test], scores):.3f}")

# Which step was the mistake? Rank the steps of each failed run, most suspected first.
locator = get_method("step.dep_anomaly")()
failed = [i for i in range(100) if y[i] == 1]
hits = sum(locator.rank(traces[i])[0] == mistakes[i] for i in failed)
print(f"mistake ranked first in {hits} of {len(failed)} failed runs")

# Can it be caught live? One causal alarm per arriving step (step t sees steps 0..t only).
monitor = get_method("online.span_z")()
alarms = monitor.step_alarm_scores(traces[failed[0]])
print(f"first alarm peak at step {alarms.argmax()}, mistake at step {mistakes[failed[0]]}")
```

Every method is created from its ID with `get_method(id)()`, and the methods of one family share one interface, so trying another detector or locator changes one string. [`examples/example_methods.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_methods.py) adds a configuration audit before the run.

## Real Agent Traces

[agentpile](https://github.com/yzhao062/agentpile) loads public agent-trace datasets (Who&When, TRAIL, MAST, tau-bench, SWE-Gym and others), trace files (OpenAI-style messages, OTLP JSON, Claude Code sessions, MCP tool lists) and captured runs into one record. `from_agentpile` turns those records into the traces every method reads, and `configs_from_agentpile` turns them into the configurations the config methods read. `pip install "auditable[agentpile]"` installs the conversion; the example below also runs tabular methods, so it needs `pip install "auditable[agentpile,tabular-ad]"`.

```python
import agentpile as ap
import auditable as au
from auditable.datasets import load_traces
from auditable.methods import get_method

# Any run: name the dependency rules, since agentpile infers none when it loads.
runs = list(ap.load("tau_bench", limit=100))
traces = au.from_agentpile(runs, deps=["tool_call_id_v1", "value_flow_v1"])
scores = get_method("run.pyod_ecod")().fit_score(traces)   # one score per run
y = au.run_targets(traces)                                 # 1 failed, 0 succeeded, -1 unknown

# The same shapes as make_traces, on a public dataset; mistakes holds the labeled step (-1 if none).
traces, y, mistakes = load_traces("who_and_when", deps="full_context", subset="Algorithm-Generated")
labeled = [t for t, m in zip(traces, mistakes) if m >= 0]
print(get_method("step.dep_anomaly")().rank(labeled[0])[:3])

# Before a run: an MCP tools/list result plus the task the agent will get.
configs = au.configs_from_agentpile(next(ap.load("tools.json")), task="Summarize the open issues.")
print(get_method("config.owasp_asi_combined")().audit(configs[0]))
```

`deps` has no default. It is a list of agentpile rules, `"recorded"` for the dependencies the record already holds, or `"full_context"` for every earlier step, and each trace's `meta` keeps the rules that ran. The default `convention="generic"` makes one step per message and one per tool call with its result. `convention="grade"` with `deps="grade"` instead reproduces the steps, dependencies and graphs of the GRADE converters that CatchBench and the GRADE paper used, for the seven corpora below. [`benchmarks/agentpile_crosscheck`](https://github.com/yzhao062/auditable/tree/main/benchmarks/agentpile_crosscheck) compares that output with GRADE's run by run.

| Corpus | `load_traces` arguments | Runs in GRADE's output | Result |
|---|---|---|---|
| Who&When | `"who_and_when", convention="grade", deps="grade"` | 184 | equal; the 126 localization labels match |
| tau-bench | `"tau_bench", convention="grade", deps="grade", model=[...]` | 660 | equal |
| tau2-bench | `"tau2_bench", convention="grade", deps="grade", model=...` | 278 | equal to the frozen rule `tau2-frozen-v1`, since GRADE's own tau2 output changes with the hash seed |
| SWE-Gym | `"swe_gym_openhands", convention="grade", deps="grade"` | 376 | equal |
| AgentRewardBench | `"agent_reward_bench", convention="grade", deps="grade"` | 600 | equal on 3 runs; on the other 597, GRADE adds two steps for the page after the last action, which agentpile does not create, and the steps before them are equal |
| SWE-rebench OpenHands | `"swe_rebench_openhands", convention="grade", deps="grade"` | 600 | equal |
| SWE-agent | `"swe_agent_nebius", convention="grade", deps="grade"` | 600 | equal |

These results hold on the dataset revisions agentpile 0.3.0 loads by default. On any other revision, `convention="grade"` raises an error rather than return steps nobody has compared with GRADE's, and `convention="generic"` reads the run without that guarantee.

In the generic convention, a `user` step is a step the record attributes to the user role, which is not always text a person typed (a session's compact summary or a task notification has that role too). Content given as a list of parts (dicts with a string `type`, possibly mixed with bare strings) keeps its text; an image, a document or any other non-text part becomes a placeholder such as `[image]` and its other fields are dropped, while any other list stays JSON. `meta["report"]` counts the replaced parts and accounts for every step of the record.

A run with subagent trajectories (a Claude Code session with subagents, an Agents SDK handoff) raises an error, since this version does not flatten them into one trace. `from_agentpile(run.subagents, ...)` reads each subagent as a trace of its own, and `subagents="skip"` reads the top-level trajectory only; on the Claude Code sessions with subagents we measured, skipping left out about half of the steps.

## Methods

Configuration rules and the step-by-step and binary-search LLM judges run on the core install, graph detectors need the `graph-ad` extra, and every other method needs `tabular-ad`. The tables are generated from the registry: `list_methods(family=...)` returns each method's card, and `method_table(detailed=True)` prints the fit scope, fidelity to the source, extras, and the CatchBench row each method reproduces. The [method contract](https://github.com/yzhao062/auditable/blob/main/docs/methods-contract.md) defines the four interfaces on one page.

### Config: Is the Agent Over-Privileged Before It Runs?

`auditor.audit(config)` reads an `AgentConfig` (the declared capabilities and the task text) and returns a finding: `flag` (True, False, or None to abstain), the `excess` capabilities, and the `rules` that fired.

<!-- methods:config:start -->
| Type | Method | Algorithm | Reference |
|---|---|---|---|
| Rule | `config.flag_risky_perms` | Flag risky permission levels | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `config.owasp_asi_combined` | Combined OWASP and CWE scanner | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `config.owasp_excess_functionality` | Excess functionality | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `config.owasp_excess_permissions` | Excess permission level | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `config.owasp_privilege_escalation` | Unrequested privilege management | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `config.sensitive_access` | Unrequested sensitive access | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `config.unrequested_high_impact` | Unrequested high-impact action | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| LLM judge | `config.llm_needed` | LLM judge, needed capabilities | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
<!-- methods:config:end -->

### Online: Can the Failure Be Caught While the Run Streams?

`monitor.step_alarm_scores(trace)` returns one alarm per step, and the alarm at step t reads steps 0 to t only.

<!-- methods:online:start -->
| Type | Method | Algorithm | Reference |
|---|---|---|---|
| Rule | `online.dep_count_z` | Causal z-score of dependency count | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `online.dep_span` | Mean dependency span (running) | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `online.raw_span` | Raw dependency span | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `online.span_z` | Causal z-score of dependency span | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Baseline | `online.random` | Random alarm scores | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
<!-- methods:online:end -->

### Run: Did the Run Fail?

`detector.fit(traces, y)`, then `detector.decision_function(traces)`: one score per run, higher meaning more likely failed. Unsupervised detectors ignore `y`. Transductive detectors (the PyGOD detectors and GUARDIAN-Lite) fit and score one batch together with `detector.fit_score(traces)`.

<!-- methods:run:start -->
| Type | Method | Algorithm | Reference |
|---|---|---|---|
| Supervised | `run.lr_flat` | Logistic regression on run size (GRADE flat) | [GRADE (Zhao, 2026)](https://arxiv.org/abs/2606.22741) |
| Supervised | `run.lr_flatdep` | Logistic regression on size and dependency (GRADE flatdep) | [GRADE (Zhao, 2026)](https://arxiv.org/abs/2606.22741) |
| Supervised | `run.lr_full` | Logistic regression on all raw layers (GRADE full) | [GRADE (Zhao, 2026)](https://arxiv.org/abs/2606.22741) |
| PyOD detector | `run.pyod_copod` | COPOD on run size | [COPOD (Li et al., 2020)](https://arxiv.org/abs/2009.09463) |
| PyOD detector | `run.pyod_ecod` | ECOD on run size | [ECOD (Li et al., 2022)](https://arxiv.org/abs/2201.00382) |
| PyOD detector | `run.pyod_hbos` | HBOS on run size | HBOS (Goldstein and Dengel, 2012) |
| PyOD detector | `run.pyod_iforest` | Isolation Forest on run size | [Isolation Forest (Liu et al., 2008)](https://doi.org/10.1109/ICDM.2008.17) |
| PyOD detector | `run.pyod_knn` | KNN distance on run size | KNN outlier score (Ramaswamy et al., 2000) |
| PyOD detector | `run.pyod_lof` | Local outlier factor on run size | [LOF (Breunig et al., 2000)](https://doi.org/10.1145/335191.335388) |
| Graph detector | `run.gsafeguard_lite` | G-Safeguard-Lite (supervised graph classifier) | [G-Safeguard (Wang et al., 2025)](https://arxiv.org/abs/2502.11127) |
| Graph detector | `run.guardian_lite` | GUARDIAN-Lite (attribute-reconstruction autoencoder) | [GUARDIAN (Zhou et al., 2025)](https://arxiv.org/abs/2505.19234) |
| Graph detector | `run.pygod_anomalydae` | AnomalyDAE dual autoencoder (PyGOD) | AnomalyDAE (Fan et al., 2020) |
| Graph detector | `run.pygod_conad` | CONAD contrastive graph autoencoder (PyGOD) | CONAD (Xu et al., 2022) |
| Graph detector | `run.pygod_dominant` | DOMINANT graph autoencoder (PyGOD) | [DOMINANT (Ding et al., 2019)](https://doi.org/10.1137/1.9781611975673.67) |
| Graph detector | `run.pygod_gaan` | GAAN adversarial detector (PyGOD) | GAAN (Chen et al., 2020) |
| Baseline | `run.random` | Random run scores | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
<!-- methods:run:end -->

### Step: Which Step Was the Decisive Mistake?

`locator.rank(trace)` orders the steps of a failed run, most suspected first, and `locator.step_scores(trace)` gives their scores. Supervised locators are first fitted with `fit(traces, y)`, where `y` holds the `idx` of each run's mistake. The step-by-step and binary-search LLM judges name one step with `predict_step(trace)`.

<!-- methods:step:start -->
| Type | Method | Algorithm | Reference |
|---|---|---|---|
| Rule | `step.blast` | Blast radius (downstream reach) | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `step.degree` | Dependency degree | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `step.dep_anomaly` | Dependency anomaly (within-run z-score) | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `step.has_dep` | Has a dependency | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `step.max_span` | Maximum dependency span | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Rule | `step.position_early_rule` | Early-position prior | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
| Supervised | `step.exec_rank` | Execution-feature step ranker | [GRADE (Zhao, 2026)](https://arxiv.org/abs/2606.22741) |
| Supervised | `step.position_supervised` | Early-fault position prior | [GRADE (Zhao, 2026)](https://arxiv.org/abs/2606.22741) |
| Graph detector | `step.pygod_dominant_sg4` | DOMINANT on the step dependency graph | [DOMINANT (Ding et al., 2019)](https://doi.org/10.1137/1.9781611975673.67) |
| Graph detector | `step.pygod_dominant_ww8` | DOMINANT on the Who&When step matrix | [DOMINANT (Ding et al., 2019)](https://doi.org/10.1137/1.9781611975673.67) |
| LLM judge | `step.llm_all_at_once` | LLM judge, all-at-once | [Who&When (Zhang et al., 2025)](https://arxiv.org/abs/2505.00212) |
| LLM judge | `step.llm_binary_search` | LLM judge, binary-search | [Who&When (Zhang et al., 2025)](https://arxiv.org/abs/2505.00212) |
| LLM judge | `step.llm_step_by_step` | LLM judge, step-by-step | [Who&When (Zhang et al., 2025)](https://arxiv.org/abs/2505.00212) |
| Baseline | `step.random` | Random step scores | [CatchBench (2026)](https://arxiv.org/abs/2608.22808) |
<!-- methods:step:end -->

## Install

```bash
pip install auditable                  # core: the registry and the configuration rules, no third-party dependencies
pip install "auditable[tabular-ad]"    # run, step, and online methods on trace features (NumPy, scikit-learn, PyOD, NetworkX)
pip install "auditable[agentpile]"     # read public datasets, trace files and captured runs through agentpile
```

The examples, tests and benchmark harnesses are in the repository, not the wheel; to run them, install from a clone with `pip install -e ".[dev,tabular-ad]"`.

The LLM judges take any `prompt -> reply` function as their backend; `OpenAICompatibleBackend` reaches an OpenAI-compatible endpoint with the standard library alone.

The graph methods (PyGOD detectors, GUARDIAN-Lite, G-Safeguard-Lite) need torch, PyTorch Geometric, PyGOD, and a sampling backend built for the installed torch, which pip cannot resolve from the extra. Install torch and the matching `pyg-lib` wheel first:

```bash
pip install "torch==2.12.1" --index-url https://download.pytorch.org/whl/cpu
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.12.1+cpu.html
pip install "auditable[graph-ad]"
python -c "import torch_geometric.typing as t; assert t.WITH_PYG_LIB"
```

Where torch is already installed, as on Colab, keep it: skip the first line, and in the `pyg-lib` URL replace `2.12.1+cpu` with the installed version, which `python -c "import torch; print(torch.__version__)"` prints (for example `2.11.0+cu130`); the pinned CPU build would replace Colab's torch and break its preinstalled torchvision. The graph-ad install was checked on Colab CPU and T4 runtimes with Colab's torch, and on Linux aarch64 (DGX Spark) with the pinned one. The PyGOD methods run on the CPU (`gpu=-1`) even when a GPU is present.

## Traces and Inputs

Methods read traces through `auditable.featurize.normalize_trace`, which accepts a list of step dicts, a run dict (`{"steps": ..., "texts": ...}`), a typed NetworkX graph from `auditable.graph.build_graph`, a `SessionGraph`, or the `TraceView` that `from_agentpile` returns. Steps carry `idx`, `agent`, `kind` (`"decision"` or `"tool_call"`), and optionally `deps` and `text`; a `SessionGraph` also keeps parallel branches and merges of the execution order. A trace without dependency annotations can be read with `dependency="full_context"`, the assumption that each step relied on every earlier step. A method that needs a field the trace lacks (step text for an LLM judge, for example) raises an error naming it.

`auditable.featurize` exposes the features the methods compute, so they can be inspected or reused: the GRADE structural layers (`grade_layer`, `grade_layer_matrix`), the Who&When step matrix (`who_when_step_matrix`), the step dependency graph and its statistics (`step_graph`, `span_series`, `run_max_span`), and graph encodings for PyGOD (`to_pyg`).

## Reproducing CatchBench

[`benchmarks/catchbench_parity`](https://github.com/yzhao062/auditable/tree/main/benchmarks/catchbench_parity) checks the library against CatchBench 0.1.2 and GRADE at commit `3839a57`, the versions the CatchBench board pins. It is a development harness and is not part of the wheel.

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

Subclass the family base class in `auditable.base`, give it a `MethodCard`, and register it with `@register`. The [method contract](https://github.com/yzhao062/auditable/blob/main/docs/methods-contract.md) lists the conventions: input forms, label alignment for supervised step methods, output order, optional dependencies, the tests a method needs, and how to add a parity check when the method reproduces a CatchBench row.

## Records and Graph Analysis

The package also contains the modules it was first built around. `audit` and `replay` capture an agent decision with the dependency state it relied on and re-decide it against later state; `analyze_run` ranks the steps of a finished run by structural blast share; `analyze_plan` lints a declared plan; `LiveSession` scores a growing run; and `auditable.integrations` captures observed dependency edges from LangGraph, LangChain, MCP servers, or any loop through `TouchRecorder`. See [`examples/README.md`](https://github.com/yzhao062/auditable/blob/main/examples/README.md) and the [documentation](https://auditable-ai.readthedocs.io/).

Records are content-addressed and hash-chained, and that property is limited. A signature or digest authenticates only the entries that were submitted, a hash chain exposes a later edit only when its head is anchored with an independent party, and neither reveals an event that was never written. The chain therefore does not establish that a log is complete.

### Legacy

`ActionGate`, `ReferenceLedger` (the in-process reference rail), and `DataAuditor` are deprecated as of 0.3 and emit a `DeprecationWarning` when constructed. They remain importable and will be removed in a later release. Tabular anomaly detection on dependency snapshots belongs in [PyOD](https://github.com/yzhao062/pyod).

## Related Projects

- [CatchBench](https://github.com/yzhao062/catchbench): the benchmark for these methods (arXiv:2608.22808).
- [agentpile](https://github.com/yzhao062/agentpile): loads public agent-trace datasets, trace files and captured runs into one record, which `from_agentpile` reads.
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

[Apache-2.0](https://github.com/yzhao062/auditable/blob/main/LICENSE). Ported code keeps the notice of its source (GRADE and CatchBench, both MIT License) at the top of each module.
