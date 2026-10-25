# Changelog

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
