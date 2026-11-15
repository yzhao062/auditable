# Changelog

## 0.4.0 (2026-11-14)

0.4.0 adds leaderboards: `auditable.bench` reruns the tables of the CatchBench board on the same populations and protocols, and places any new method on them.

### Added

- `auditable.bench`: `leaderboard(task_id, methods=None, extra=())` scores methods on a task and returns a ranked `Board` (`print`, `to_markdown()`, `to_dict()`, `compare_golden()`, `rank_of()`); `extra=[MyMethod]` adds a method that is not in the registry. A method whose extra is missing is listed as skipped with the extra named, and a method that raises is listed as an error while the rest of the board is still scored. Oracle rows, which read the labels, are shown as references and never ranked; control rows are ranked and marked.
- Six tasks, each reproducing a table of the CatchBench 0.1.3 board row by row: `detect/swegym` (376 runs) and `detect/tau` (660 runs), run-level failure detection by ROC-AUC; `live/swegym` and `live/tau`, early warning on 25, 50, 75 and 100 percent prefixes; `localize/whoandwhen` (126 failed runs), decisive-step localization by top-1, top-3 and MRR, including 31 LLM-judge rows replayed from CatchBench's committed prediction caches without an API call; and `pre/over_privilege` (1,187 configurations), configuration audit by F1, precision and recall, overall and per source. The populations are read through agentpile at the dataset revisions CatchBench pins, with the selection rules of GRADE commit 3839a57; the protocols are ported from CatchBench's adapters.
- `run.lr_spline_flat`, the flexible size control: a cubic spline of the four GRADE `flat` counts, then the board's logistic regression, ported from the `size (spline)` arm of CatchBench's size audits. It joins the `detect` and `live` boards as a control. On `detect/swegym` it ranks second, above both GRADE dependency regressions; on `detect/tau` it ranks below `run.lr_flatdep`. The library now holds 44 methods.
- Golden files with the board values of every row and the SHA-256 of the board they come from; the spline control's value comes from CatchBench's committed size audits instead, with a 0.001 tolerance. `python -m auditable.bench <task> --golden` compares a run with them and fails on a mismatch, on an error row, on a golden row left unchecked (skipped, missing a metric, absent from the task's default rows, or in a golden file that is missing), and on a run that compared no golden value; `--allow-skipped` accepts rows skipped for a missing extra. Three rows differ on macOS arm64 by a known platform effect and are reported as platform values; `--exact` counts them as mismatches, and the weekly Linux run uses it.
- `python -m auditable.bench`, the command-line entry: lists the tasks, prints a board, or emits it as Markdown or JSON.
- The `bench` extra (`pyod`, `scikit-learn`, `numpy`, `networkx`, `agentpile[hf]`); the graph rows also need `graph-ad`.
- CI: a `bench-core` job runs the bench unit tests, which download nothing, a weekly `bench golden` workflow reruns every task on Linux against its golden file, and a weekly `bench platforms` workflow installs the package from a clean checkout on Linux, macOS, and Windows (Python 3.10 and 3.13) with the README's graph-ad steps and checks three tasks against their golden files.

## 0.3.0 (2026-10-25)

0.3.0 turns `auditable` into a library of auditing methods for AI agents, with one interface per family and the ability to read public agent traces through agentpile.

### Added

- `auditable.methods`: 43 methods in four families, each with one interface. The `config` family (8 methods) audits an agent configuration before a run, `online` (5) raises causal alarms while a run streams, and `run` (16) and `step` (14) detect a failed run and localize its decisive step afterwards. A registry (`list_methods`, `get_method`) returns each method's `MethodCard`, which records its fit scope, fidelity to its source, required fields, extras and citations. The library holds baselines from the literature (PyOD and PyGOD detectors, GUARDIAN-Lite, G-Safeguard-Lite, the Who&When LLM-judge protocols, OWASP-style configuration rules) beside the methods introduced with GRADE and CatchBench; every method that reproduces a CatchBench 0.1.2 board row matches it sample by sample.
- `auditable.featurize`: trace normalization for step lists, run dicts, typed graphs, `SessionGraph` and `TraceView` inputs, and the features the methods compute (the GRADE structural layers, the Who&When step matrix, the step dependency graph, PyG encodings).
- `auditable.llm`: a `prompt -> reply` backend protocol, `FixedResponseBackend` for tests and replays, `OpenAICompatibleBackend` (standard library only), and a prediction cache.
- `auditable.from_agentpile`, `configs_from_agentpile`, `step_targets`, `run_targets` and `auditable.datasets.load_traces` read agentpile records (public datasets, trace files, captured runs) into the methods' inputs and labels. `convention="grade"` with `deps="grade"` reproduces the steps, dependencies and graphs of GRADE's converters (commit 3839a57) on seven corpora; `convention="generic"` (the default) makes one step per message and one per tool call.
- Extras `tabular-ad`, `graph-ad` and `agentpile`; the core install still has no third-party dependency.
- `docs/methods-contract.md`, the contract a method follows and the behaviours a pinned consumer such as CatchBench relies on.

### Behaviour of the Conversion

- In `convention="generic"`, content given as a list of parts (dicts with a string `type`, possibly mixed with bare strings) keeps its text and writes every other part (an image, a document) as `[<type>]`, dropping that part's other fields; `meta["report"]` counts those parts and says how each step of the record maps into the trace.
- `convention="grade"` raises `ValueError` for a run from a dataset revision other than the one its projection was checked on.
- A run with subagent trajectories is refused by default; the error names the two ways to read it, `from_agentpile(run.subagents, ...)` and `subagents="skip"`.

### Deprecated

- `ActionGate`, `ReferenceLedger` and `DataAuditor` emit a `DeprecationWarning` when constructed and will be removed in a later release.

## Earlier Releases

Releases up to 0.2.1 are listed on [PyPI](https://pypi.org/project/auditable/#history) and as [GitHub tags](https://github.com/yzhao062/auditable/tags).
