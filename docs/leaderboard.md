# Leaderboards

`auditable.bench` scores the methods in `auditable` on fixed tasks and prints a ranked board. One command reruns the CatchBench board on your machine, and one argument gives your own method a rank on the same population and protocol. The module is new in 0.4.0.

Three projects share the work. `auditable` is the library of auditing methods and owns the leaderboards. [agentpile](https://github.com/yzhao062/agentpile) reads public agent-trace datasets into one record, which the tasks load. [CatchBench](https://github.com/yzhao062/catchbench) (arXiv:2608.22808) is the paper and benchmark whose board these leaderboards reproduce.

## Run a Board

### Install

```bash
pip install "auditable[bench]"
```

The `bench` extra installs PyOD, scikit-learn, NumPy, NetworkX, and agentpile with its Hugging Face loaders. The first run downloads and converts the dataset through agentpile; later runs read agentpile's cache, from which the SWE-Gym population loads in about 3 seconds.

The graph rows (PyGOD, GUARDIAN-Lite, and G-Safeguard-Lite) also need `auditable[graph-ad]` and the PyG sampling backend, installed as in the [README Install section](https://github.com/yzhao062/auditable/blob/main/README.md#install). Without them those rows are listed as skipped with the missing extra named, and the rest of the board is still scored.

### Rerun the Paper's Board

```bash
python -m auditable.bench detect/swegym --golden
```

The command scores all 17 run detectors on the SWE-Gym population and prints the board with the CatchBench value in brackets beside each row that has one. With `--golden` it exits with status 1 if a row differs from the CatchBench board or a golden row was not scored. On a laptop CPU the full board takes about 70 seconds, of which PyGOD GAAN, the slowest row, takes about 37.

```text
detect/swegym: SWE-Gym OpenHands trajectories (one scaffold, balanced), 376 runs (188 failed, 188 succeeded)
ranked by roc_auc, higher is better; the CatchBench reference roc_auc in brackets

rank  method                         roc_auc
   1  run.pygod_gaan                   0.850 [0.850]
   2  run.lr_spline_flat (control)     0.832 [0.832]
   3  run.gsafeguard_lite              0.828 [0.828]
   4  run.lr_full                      0.819 [0.819]
   5  run.lr_flatdep                   0.804 [0.804]
   6  run.guardian_lite                0.767 [0.767]
   7  run.pyod_ecod                    0.765 [0.765]
   8  run.pygod_conad                  0.750 [0.750]
   9  run.lr_flat                      0.663 [0.663]
  10  run.pyod_copod                   0.625 [0.625]
  11  run.pygod_anomalydae             0.592 [0.592]
  12  run.pyod_lof                     0.582 [0.584]
  13  run.pyod_iforest                 0.571 [0.571]
  14  run.pygod_dominant               0.547 [0.547]
  15  run.random (control)             0.483 [0.483]
  16  run.pyod_knn                     0.446 [0.446]
  17  run.pyod_hbos                    0.319 [0.319]
```

### Read the Board

This is the `detect/swegym` board as computed on 2026-11-14. Its 16 CatchBench rows equal the CatchBench board, which was computed on Linux; macOS prints `run.pyod_lof` as 0.582 ([Golden Values](#golden-values)). The seventeenth row, the flexible size control `run.lr_spline_flat`, is not a CatchBench board row: its reference value comes from CatchBench's committed size audit.

| Rank | Method | What it is | ROC-AUC |
|---|---|---|---|
| 1 | `run.pygod_gaan` | GAAN adversarial detector (PyGOD) | 0.850 |
| 2 | `run.lr_spline_flat` | Logistic regression on a cubic spline of run size, the flexible size control | 0.832 |
| 3 | `run.gsafeguard_lite` | G-Safeguard-Lite, a supervised graph classifier | 0.828 |
| 4 | `run.lr_full` | Logistic regression on all raw GRADE layers | 0.819 |
| 5 | `run.lr_flatdep` | Logistic regression on size and dependency (GRADE flatdep) | 0.804 |
| 6 | `run.guardian_lite` | GUARDIAN-Lite, an attribute-reconstruction autoencoder | 0.767 |
| 7 | `run.pyod_ecod` | ECOD on run size features | 0.765 |
| 8 | `run.pygod_conad` | CONAD contrastive graph autoencoder (PyGOD) | 0.750 |
| 9 | `run.lr_flat` | Logistic regression on run size (GRADE flat), the size baseline | 0.663 |
| 10 | `run.pyod_copod` | COPOD on run size features | 0.625 |
| 11 | `run.pygod_anomalydae` | AnomalyDAE dual autoencoder (PyGOD) | 0.592 |
| 12 | `run.pyod_lof` | Local outlier factor on run size features | 0.584 |
| 13 | `run.pyod_iforest` | Isolation Forest on run size features | 0.571 |
| 14 | `run.pygod_dominant` | DOMINANT graph autoencoder (PyGOD) | 0.547 |
| 15 | `run.random` | Uniform random scores, a control | 0.483 |
| 16 | `run.pyod_knn` | KNN distance on run size features | 0.446 |
| 17 | `run.pyod_hbos` | HBOS on run size features | 0.319 |

A board ranks by the first metric of its task, higher first, and rows that tie on the three-decimal score share the better rank. The printed board marks `run.random` and `run.lr_spline_flat` as controls. The value for `run.pyod_lof` is the Linux one; macOS prints 0.582 (see [Golden Values](#golden-values)).

GAAN from PyGOD, a method from the literature, leads this board. Second is the flexible size control, which fits a cubic spline to the same four counts the linear size baseline `run.lr_flat` reads: run size alone, fitted without a linear restriction, ranks above G-Safeguard-Lite and above both GRADE dependency regressions (`run.lr_full` and `run.lr_flatdep`). The board therefore does not show that dependency structure predicts failure beyond run size, the reading GRADE retired on 2026-09-08, and a new row that ranks below the spline control has not shown signal beyond size either. Both size rows stay on the board so that every row can be read against them. The order is not the same on every corpus: on `detect/tau`, `run.lr_flatdep` (0.665) ranks above the spline control (0.631).

Two further notes apply. The population is one scaffold with 376 runs, so the board ranks methods on that sample and says nothing about other scaffolds. A ROC-AUC below 0.5 (`run.pyod_knn`, `run.pyod_hbos`) means the method scores succeeded runs as more anomalous than failed ones; the board reports the value as computed and does not flip the sign.

## What a Task Fixes

A task fixes a population and a protocol. The population is which runs, steps, or configurations are scored, and how they are selected and labeled. The protocol is how each method is fit and scored on that population. Both are frozen.

The reason is comparison. A score is comparable with the CatchBench row, and with any other method on the same task, only when the data, the folds, the seeds, and the score call are the same. The tasks are ported from CatchBench 0.1.3 and GRADE at commit `3839a57`. The folds, seeds, and score calls are kept as they are there, and only the estimator inside each fold is the method being scored.

On a board, the method and its constructor arguments are the only things that vary. The golden check turns the freeze into a test: a change to the population or the protocol shows up as a row that no longer matches the CatchBench board.

## The Six Tasks

| Task | Question | Population | Method family | Metrics |
|---|---|---|---|---|
| `detect/swegym` | Did the run fail? | SWE-Gym OpenHands trajectories, one scaffold: 376 runs, 188 failed | `run` | ROC-AUC |
| `detect/tau` | Did the run fail? | tau-bench: 660 runs, 363 failed | `run` | ROC-AUC |
| `localize/whoandwhen` | Which step was the decisive mistake? | Who&When: 126 failed multi-agent runs, each with a human `mistake_step` label | `step` | top-1, top-3, MRR |
| `live/swegym` | Can the failure be caught early? | SWE-Gym runs, scored on 25, 50, 75, and 100 percent prefixes of their steps | `online` and `run` | ROC-AUC per prefix, and the mean over prefixes |
| `live/tau` | Can the failure be caught early? | tau-bench runs, scored on the same four prefixes | `online` and `run` | ROC-AUC per prefix, and the mean over prefixes |
| `pre/over_privilege` | Is the agent configuration over-privileged before it runs? | 1,187 configurations from six sources | `config` | F1, precision, recall |

`python -m auditable.bench` prints this list from the installed package, with each task's metrics in ranking order. The first metric ranks the board: ROC-AUC for the detection tasks, top-1 for localization, mean prefix ROC-AUC for the live tasks, and F1 for PRE. The PRE board also reports coverage and F1 per source. The LLM judges on the localization and PRE boards are replayed from committed caches, so those boards make no API call and need no key.

The method family matters when you add a row. A method scored on a task must belong to the task's family, and a method of another family is listed as skipped with both families named. The live tasks take two families: `online` monitors, and `run` detectors scored on the prefixes. The family list is in the [Method Contract](methods-contract.md#families).

A task defines its default rows. A detection board scores every registered `run` method. The other boards name their rows: the localization board a fixed set of step methods plus the cached LLM judges, a live board a fixed set that mixes `run` and `online` methods, and the PRE board the registered `config` auditors, the controls `config.flag_all` and `config.flag_none`, an oracle, and the cached LLM judge. The oracle row (`config.oracle_privilege_diff`) reads the labels, so the board prints it apart as an upper bound and does not rank it.

### What the CatchBench Boards Show

The numbers below come from the CatchBench board that the leaderboards reproduce. They give context for reading a new row and make no ranking claim for any method in this library.

- **`localize/whoandwhen`.** The best LLM judges (the all-at-once prompt) reach top-1 of about 0.42 to 0.45, against 0.119 for random. The structural blast rule ties the position prior at 0.159 top-1.
- **`detect/tau`.** The best rows are `run.lr_flatdep` and `run.lr_full`, tied at 0.665, against 0.619 for the size baseline.
- **`pre/over_privilege`.** The best rule scanner, `config.owasp_asi_combined`, reaches F1 0.654. Flagging every configuration already reaches 0.601, and an LLM judge reaches 0.695, so the F1 column is best read beside precision and recall.

## Protocols for Detection Tasks

Each run reaches a method as the typed graph that `auditable.graph.build_graph(steps, dependency="explicit", shared_resource=False)` builds from its steps: recorded dependencies only, and no shared-resource node. The label is 1 for a failed run and 0 for a succeeded one, and a higher score means more likely failed. The method card and the method id choose the protocol:

| Method kind | Protocol | Score |
|---|---|---|
| Supervised (`card.supervised`) | `StratifiedKFold(5, shuffle=True, random_state=s)` for each seed `s` in 0 to 4. A fresh estimator per fold is fit on the training folds and scores the held-out fold with `decision_function`. | Mean ROC-AUC over the 25 held-out folds |
| Unsupervised | One `fit_score` call over all runs, with no labels | ROC-AUC of those scores |
| `run.random` | One detector built with `RandomState(0)` draws five successive `fit_score` results | Mean ROC-AUC of the five draws |
| `run.gsafeguard_lite` | Its own loop: three seeds of 5 folds with `random_state=seed + s`, a network seeded per fold, and folds with a single class on either side skipped | Mean held-out ROC-AUC |

A supervised method that has a `features` method gets its features computed once over all runs, and the folds index that matrix, as the board does for the GRADE-layer regressions. Any other method is fit on the runs themselves. An unsupervised method never sees the labels: the protocol calls `fit_score(runs)` and nothing else.

The other tasks fix their protocols in the same way, in their task classes, and the next section sketches them.

### The SWE-Gym Population

`detect/swegym` reads the `swe_gym_openhands` dataset through agentpile at the dataset revision CatchBench pins (`baf3a4e4`). Shards are read in order and a run with fewer than two steps is dropped. Reading stops at the first shard boundary where the balanced counts per `run_id` (the smaller of the resolved and unresolved counts) add up to at least 400. The `run_id` with the most runs in its rarer class is then kept, which fixes one model, one iteration cap, and one temperature. Its first `k` resolved and first `k` unresolved runs form the population, which at the pinned revision is 376 runs with 188 failed.

GRADE keeps one `run_id` because pooling several would let step count separate scaffolds rather than outcomes.

## Protocols for the Other Tasks

**Localization.** `localize/whoandwhen` pools the steps of the failed runs. A method scores every step, the steps of each run are ranked by score, and the board reports the mean over runs of top-1, top-3, and MRR (mean reciprocal rank) for the human-labeled mistake step. A supervised locator is fit on training runs with five seeds of 5-fold cross-validation that keeps each run whole in one fold. An unsupervised method that learns from the runs is fit on all of them in one call, a rule scores each run on its own, and `step.random` averages five seeds.

**Live.** `live/swegym` and `live/tau` keep the runs with at least four steps. At each prefix of 25, 50, 75, and 100 percent of the steps, a `run` detector is scored with the detection protocol on typed graphs rebuilt from the prefix. An `online` monitor scores each run from its own prefix, with no labels and no other run. The board ranks by the mean of the four ROC-AUC values and also reports each one.

**PRE.** `pre/over_privilege` asks each auditor for a finding per configuration. Precision, recall, and F1 are micro-averaged over declared capabilities, against the labeled excess capabilities of each configuration, and F1 is also reported per source. A configuration on which an auditor abstains (`flag` is `None`) leaves the denominator, and coverage reports the share of configurations judged.

## Python API

```python
from auditable.bench import leaderboard, list_tasks, get_task

for task in list_tasks():
    print(task.task_id, task.family, task.metrics)

board = leaderboard("detect/swegym")
print(board)
```

`list_tasks()` returns the registered tasks sorted by id, and `get_task(task_id)` returns one, or raises a `KeyError` that lists the available ids. A task has `task_id`, `family`, `metrics`, and `title`.

### `leaderboard`

`leaderboard(task, methods=None, *, extra=(), progress=None)` scores methods on a task and returns a `Board`.

- `task` is a task id or a `Task`.
- `methods` names the rows: method ids, method classes, or `Entry` objects. The default is the task's default rows (see [The Six Tasks](#the-six-tasks)).
- `extra` adds rows to whichever set `methods` gives, which is how a new method is placed on the board.
- `progress` is called with each `Row` as soon as it is scored.

A method whose pip extra is missing is listed as skipped, with the extra named (for example `needs auditable[graph-ad]`). A method that raises is listed as an error with the exception, and the rest of the board is still scored. The board lists ranked rows first, by the task's first metric, then any oracle rows (an upper bound that reads the labels, never ranked), then the skipped and error rows.

### Board Methods

| Call | Returns |
|---|---|
| `print(board)` | The text board, with the CatchBench value in brackets and the reason for each skipped or error row |
| `board.to_markdown(golden=False)` | A Markdown table of the ranked rows, then the oracle rows with rank `ref`; `golden=True` adds a CatchBench column |
| `board.to_dict()` | A JSON-ready dict: task id, description, metrics, and for each row its rank, key, label, method id, role, scores, seconds, status, note, and golden values |
| `board.compare_golden(tol=0.0005)` | One dict per scored metric that has a golden value, with `key`, `metric`, `got`, `golden`, `ok`, `platform`, and `note` |
| `board.rank_of(key)` | The 1-based rank of a row by key or label, with ties on the three-decimal score sharing the better rank, or `None` if the row was not scored |
| `board[key]` | The `Row` for a key or label, or a `KeyError` |

A `Row` holds `key`, `label`, `method_id`, `family`, `scores` (a dict from metric to value), `seconds`, `status` (`"ok"`, `"skipped"`, or `"error"`), `note`, `golden`, and `role` (`"baseline"`, `"control"`, `"oracle"`, or `None`, from the method card).

### Place Your Own Method

A method is a class with a `MethodCard` that follows the [Method Contract](methods-contract.md). The card's family must match the task, and `supervised` selects the protocol above. A supervised method implements `fit(X, y)` and `decision_function(X)`; an unsupervised one implements `fit_score(X)`. Here `X` is the list of typed graphs, one per run.

```python
import numpy as np
from auditable.base import BaseRunDetector, MethodCard
from auditable.bench import leaderboard


class StepCount(BaseRunDetector):
    """Score a run by its number of graph nodes; higher means more likely failed."""

    card = MethodCard(
        method_id="run.my_step_count", name="Step count", family="run",
        fit_scope="transductive", supervised=False, output_kind="score", fidelity="original",
    )

    def fit_score(self, X, y=None):
        return np.array([len(g) for g in X], dtype=float)


# The full board plus your row:
board = leaderboard("detect/swegym", extra=[StepCount])
print(board.rank_of("run.my_step_count"))

# Only your row and two baseline rows, for a fast loop:
board = leaderboard("detect/swegym", methods=["run.lr_flat", "run.random", StepCount])
```

The class needs `@register` only to be reachable by id, as the command line needs. Passing the class through `methods` or `extra` does not require it.

### Variants and Constructor Arguments

An `Entry` is one board row: a method, the constructor arguments to build it with, a label, and a key. Use it to score a variant of a method, or a method that takes arguments.

```python
from auditable.bench import Entry, leaderboard
from auditable.methods import get_method

board = leaderboard("detect/swegym", methods=[
    "run.lr_flat",
    Entry(get_method("run.lr_flat"), {"C": 0.1}, label="lr_flat C=0.1", key="run.lr_flat:C=0.1"),
])
print(board["run.lr_flat:C=0.1"].scores)
```

`Entry(method, kwargs={}, label=None, key=None)` builds a fresh instance with `kwargs` each time the protocol needs one. The `key` names the row for `board[key]`, `rank_of`, and the golden lookup, and defaults to the method id. The `label` is the name printed and defaults to the key.

Give every variant its own key. A variant that keeps the default key is compared with the CatchBench row of its base method, and `compare_golden` reports a mismatch. Two rows with one key are both scored, but `board[key]` returns the one listed first.

The protocol supplies `random_state` to `run.random` and `seed` to `run.gsafeguard_lite` itself, so those two arguments in an `Entry` are overridden.

## Command Line

```bash
python -m auditable.bench [task_id ...] [--methods ID ...] [--golden [--allow-skipped] [--exact]] [--markdown] [--json]
```

| Argument | Effect |
|---|---|
| none | Lists the tasks with their family, metrics, and title |
| `task_id ...` | Scores each task in order and prints its board |
| `--methods ID ...` | Scores these method ids instead of the task's default rows; variants need the Python API |
| `--golden` | Compares with the CatchBench reference values and exits with status 1 on a mismatch or an unchecked value |
| `--allow-skipped` | With `--golden`, accepts golden rows skipped for a missing extra |
| `--exact` | With `--golden`, counts a known platform value as a mismatch |
| `--markdown` | Prints a Markdown table; with `--golden` it adds the CatchBench column |
| `--json` | Prints the board as the JSON of `board.to_dict()`; it takes precedence over `--markdown` |

```bash
python -m auditable.bench                                              # list the tasks
python -m auditable.bench detect/swegym                                # the full board
python -m auditable.bench detect/swegym --methods run.lr_flat run.lr_flatdep run.random
python -m auditable.bench detect/swegym --golden --markdown            # table with the CatchBench column
python -m auditable.bench detect/swegym --json > board.json
```

With `--golden`, a mismatch prints `golden mismatch: <key> <metric> <got> != <golden>` on stderr and sets the exit status to 1. A known platform value prints `golden platform value: ...` on stderr and does not fail.

The check also fails when a value it should check was never there to check, printing `golden unchecked: <key> (<reason>)`: a golden row that was skipped, an error row (golden or not), a golden row missing one of its metrics, a golden file the task expects but cannot find, a run that compared no golden value at all (for example `--methods` naming only rows without one), and, when the task chose the rows (no `--methods`), a golden row absent from the board, such as an LLM-judge row whose cache was not found. `--allow-skipped` accepts golden rows skipped for a missing extra, for a run without `graph-ad`. `--exact` counts a known platform value as a mismatch; the weekly Linux reference run uses it, since the board was computed on Linux. In Python, `board.golden_coverage()` returns the same list.

## Golden Values

Each task ships a golden file in `src/auditable/bench/golden/`, named by the task id with `/` replaced by `__` (`detect__swegym.json` for `detect/swegym`; `localize/whoandwhen` keeps a second file for its LLM judges). The file records its source, which is CatchBench 0.1.3, the file `tests/golden/board.txt`, and the SHA-256 of that file, and then one entry per row key with the CatchBench value and the board label.

A row matches when its score, rounded to three decimals, is within 0.0005 of the golden value, since the CatchBench board prints three decimals. The rows that train a neural network (PyGOD, GUARDIAN-Lite, and G-Safeguard-Lite) carry a tolerance of 0.005 of their own. The flexible size control on the `detect` and `live` tasks is the one golden row that is not a board row: its value is the `size (spline)` arm of CatchBench's committed size audits (`tools/detection_audit_results.json` and `tools/live_size_control_audit_results.json`, with their SHA-256 in the row), and it carries a tolerance of 0.001, because those audits ran on Windows and the spline moves in the fourth decimal across platforms. A method you add has no golden value.

The CatchBench board was computed on Linux x86-64. Three rows have accepted macOS arm64 values: `run.pyod_lof` on `detect/swegym` at 0.582, `run.pyod_knn` on `detect/tau` at 0.574, and `run.pygod_anomalydae` on `detect/tau` at 0.529. `--golden` reports these as platform values and accepts them. Any other difference beyond the applicable tolerance is a mismatch.

## Submit a Method

1. Add the method to `auditable.methods` for its family, with a `MethodCard`, `@register`, and the tests the [Method Contract](methods-contract.md) lists.
2. Run the full board with `python -m auditable.bench <task> --golden`. If every existing row matches and none is skipped, your environment reproduces the board and your rank is comparable. Then score the method with `leaderboard("<task>", extra=[YourMethod])`.
3. Save the result with `json.dump(board.to_dict(), fh, indent=1)`.
4. Open a pull request against the [auditable repository](https://github.com/yzhao062/auditable) that adds the method and the JSON.
5. Maintainers rerun the method with the same command before it is listed.

Read a submitted row with its protocol in mind. A supervised method sees the labels of the training folds only, and an unsupervised method sees no labels. The board has no split held out from the folds, so a method tuned by watching the board is tuned on its own test data. Submit the configuration you fixed before the run, and list every variant you tried.
