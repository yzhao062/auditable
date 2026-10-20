# Method Contract

This page is the contract a method in `auditable.methods` follows. It is short on purpose: a new method should need nothing beyond it, the base classes in `auditable/base.py`, and one existing method of the same family to copy from.

## Families

A method belongs to exactly one family. The family fixes the input, the entry points, and what the output means.

| Family | When | Input | Entry points | Output |
|---|---|---|---|---|
| `config` | PRE, before a run | one agent configuration (`AgentConfig` or its dict form) | `audit(config)`, `audit_many(configs)` | a `ConfigFinding`: `flag` (True, False, or None to abstain), `excess` capability names, `rules` that fired |
| `run` | POST, after a run | a list of traces, or a feature matrix where the card says so | inductive: `fit(X, y=None)`, `decision_function(X)`, `fit_score(X, y=None)`; transductive: `fit_score(X, y=None)` | one score per run; higher means more likely failed |
| `step` | POST, after a failed run | a list of traces | `fit(traces, y=None)`, `step_scores_batch(traces)`, `step_scores(trace)`; transductive: `fit_step_scores(traces)`; single-step methods: `predict_step(trace)` | one score per step, in step order; higher means more likely the decisive mistake |
| `online` | LIVE, while a run streams | one trace (its prefix so far) | `step_alarm_scores(trace)`, `prefix_score(trace, k)` | one alarm score per step; the score at step `t` uses steps `0..t` only |

There is no cross-family `assess()`. A method that does not fit one family needs a new family, which enters only with a defined input and output, a reference implementation, a runnable case, and a correctness test specific to that family.

## Fit Scope

Every card declares how the method is fitted:

- `stateless`: rules; no fitting. `fit` is a no-op that returns the method.
- `inductive`: `fit` learns parameters (scalers, weights) and the method then scores unseen inputs without refitting. `fit_score` returns `decision_scores_`, the scores of the fitted batch at training time. These can differ from `decision_function` on the same data: PyOD's KNN and LOF exclude each point from its own neighbourhood during fitting.
- `transductive`: the method fits and scores one batch together (PyGOD's graph autoencoders, GUARDIAN-Lite). Only `fit_score` and `fit_step_scores` exist; calling `fit` or `decision_function` raises `NotSupported`.

Cross-validation folds, prefix truncation, and metrics belong to the benchmark, not to the method. A supervised method exposes `fit` and a scoring entry point; the evaluation protocol calls them once per fold.

## Inputs

Methods read traces through `auditable.featurize.normalize_trace`, which accepts:

- a `SessionGraph` from `auditable.graph.session`;
- a list of step dicts with `idx` (a unique int), `agent`, `kind` (`"decision"` or `"tool_call"`), and optionally `deps` (indices of the steps it relied on) and `text`;
- a run dict `{"steps": [...], "texts": [...], "G": typed_graph}`, where `texts` and `G` are optional;
- a typed NetworkX graph from `auditable.graph.build_graph`, which is read as it is.

Two differences between these forms matter. `SessionGraph.to_networkx()` has no `resource::shared_state` node, which `build_graph(..., shared_resource=True)` adds by default, so a method that reads every node sees different node sets. And a trace without `deps` has no dependency edges unless the caller passes `dependency="full_context"`, which makes every step depend on all earlier steps (the assumption of corpora such as Who&When) and replaces any recorded `deps`.

A card's `requires` names the fields the method reads, such as `deps`, `step_text`, `exec_graph`, or `spec_tokens`. A method that needs a field the trace lacks raises a `ValueError` naming it, rather than guessing.

A step's `idx` identifies it (dependencies and labels refer to it) and may be any unique int; position features and LLM prompts use the step's position in the list instead, so renumbering the steps leaves every score unchanged.

Supervised step methods take `y` with one entry per trace: the `idx` of the decisive mistake step. Batch outputs (`step_scores_batch`, `fit_step_scores`, `decision_function`) follow input order. A supervised step method predicts a whole batch in one call, so its output matches a reference that predicts a whole fold at once; `step_scores(trace)` on one trace may differ from the batched value in the last floating-point place.

Random baselines take `random_state`, which may be a shared `numpy.random.RandomState`; draws then follow call order, so a benchmark can feed several methods one stream.

## Outputs

Scores are higher-is-more-suspicious and uncalibrated unless the card says otherwise. A method with `output_kind="single"` names one step per trace through `predict_step`, as its 0-based position, and raises `NotSupported` from `step_scores`: padding a one-step answer into a full ranking would credit guesses the method never made. `output_kind="ranking"` methods return scores whose order is the ranking. Config findings abstain with `flag=None`; an abstention is never read as "not over-privileged".

## The Method Card

Each registered class carries a `MethodCard`:

| Field | Meaning |
|---|---|
| `method_id` | globally unique, starts with the family (`step.position_supervised`) |
| `name`, `summary` | a short name and one sentence on what the score measures |
| `family`, `information_state` | the family; PRE, LIVE, or POST (derived from the family by default) |
| `fit_scope`, `supervised` | as above |
| `output_kind` | `finding`, `score`, `ranking`, or `single` |
| `fidelity` | `original` (first published here or in CatchBench), `faithful` (a verbatim port), `wrapper` (calls a third-party library), `simplified` (a reduced port; the class name ends in `Lite` and its docstring lists what was simplified) |
| `requires`, `extras` | trace or config fields read; pip extras needed beyond the core install |
| `cite` | `Citation` entries for the method and any library it wraps |
| `board_aliases`, `benchmark_role` | the CatchBench row labels the method reproduces, and the role it plays there (`baseline`, `control`, `oracle`); the role describes the board and does not decide whether a method belongs here |

`list_methods(family=..., information_state=..., fit_scope=...)` returns cards, and `get_method(method_id)` returns the class. The method table in the README is generated from the registry.

## Adding a Method

1. Subclass the family base class in `auditable.base`, set `card`, and decorate the class with `@register` from `auditable.methods`. Put it in the family module that fits (`methods/<family>/<group>.py`); a new module is added to `_METHOD_MODULES` in `methods/__init__.py`.
2. Import numpy, scikit-learn, PyOD, torch, or PyGOD inside the functions that use them. The core install has no third-party dependencies, and the registry must load without them.
3. Ported code states its source at the top of the module: repository, license, release or commit, and the original function names. Keep the numerical order of operations, seeds, iteration order, and tie handling of the source; parity is checked bit for bit.
4. Tests: card fields and registry lookup, output shape and order, determinism under a fixed seed, a missing required field, a small case whose expected values can be checked by hand, causality for online methods (one prefix followed by different suffixes gives the same scores on the prefix), and a fixed-response backend (`auditable.llm.FixedResponseBackend`) for LLM methods. Optional dependencies are skipped with `pytest.importorskip`.
5. If the method reproduces a CatchBench row, add an adapter, a per-sample check, and any retired original scoring core (a sentinel) to `benchmarks/catchbench_parity`. The check compares the library output with the original sample by sample; the board run then routes that row through the library.

## Parity With CatchBench

`benchmarks/catchbench_parity` is a development harness, not part of the wheel. With CatchBench 0.1.2 and GRADE at commit `3839a57` installed, it:

- compares every migrated method with its CatchBench original on the real corpora, sample by sample and bit for bit (`run_parity.py checks`);
- reruns the full CatchBench board with every migrated row, breakdown, and robustness table routed through the library, and with the retired original scoring cores replaced by functions that raise (`run_parity.py board`); the result must be byte-identical to the board before migration;
- reruns the board with the library outputs perturbed and confirms that every migrated row changes and no other row does (`run_parity.py consumption`).
