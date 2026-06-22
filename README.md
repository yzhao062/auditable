<div align="center">

# auditable

**Audit AI agent decisions across their lifecycle: lint the plan before deploy (PRE), replay and recover live (REAL-TIME), and rank a finished run by structural risk (POST), all over one typed decision graph.**

[![PyPI](https://img.shields.io/pypi/v/auditable.svg)](https://pypi.org/project/auditable/)
[![Python](https://img.shields.io/pypi/pyversions/auditable.svg)](https://pypi.org/project/auditable/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yzhao062/auditable.svg?style=social)](https://github.com/yzhao062/auditable)

[Quickstart](#quickstart) · [Lifecycle](#the-lifecycle) · [How It Works](#how-it-works) · [PRE Rules](#pre-lint-the-plan-before-deploy) · [Roadmap](#roadmap)

</div>

`auditable` is an open-source SDK for auditing AI agent decisions across the agent's lifecycle. It attaches at three points: before deployment, it lints a declared plan for structural risk (PRE); while the agent runs, it captures each consequential decision, replays it against the state that is live now, and executes a fix through a rail (REAL-TIME); after a run completes, it ranks every step by how much of the run rests on it and names the keystone (POST). The same typed two-layer decision graph is scored and reported at all three points, so a single graph kernel carries the analysis from design to live operation to review.

## The Problem

Most agent tools log what happened. They do not record the dependency state a decision relied on, so when a payment, an approval, or a tool call later looks wrong, the budget, the policy, and the allow-list that were live at that moment are already gone. The action was reasoned; the dependency it trusted had drifted.

Flagging that after the fact is observability. Re-deciding under the state that is live now, and reversing the action when it no longer holds, is recovery. Recovery is the gap `auditable` fills, and it spans the lifecycle: the same graph that catches a stale decision at run time can lint the plan before deploy and rank a finished run for review.

## The Lifecycle

`auditable` runs the same detection-and-report pass over one typed decision graph at three points in an agent's life. Each pillar has a public entry symbol and one honest job.

| Pillar | Public entry | What it does |
|---|---|---|
| **PRE** (before deploy) | `analyze_plan` (from `auditable.graph.pre`) | Read-only structural lints over a declared plan: four reachability checks, the execution-topology chokepoint, and a preflight coverage report. No value is executed; dependency-state risk is withheld because the plan is declared-only. |
| **REAL-TIME** (while running) | `audit` + `replay` + `ActionGate` | Capture a decision with the dependency state it relied on, replay it under the live state, and execute a routed fix (allow, block, human-review, rollback) through a rail. |
| **POST** (after a run) | `analyze_run` | Read a recorded run, build one session graph, rank every step by transitive structural blast share, and name the keystone the rest of the run rests on. |

The differentiator is the same typed two-layer graph (execution edges observed from the trace, dependency edges declared or inferred) scored and reported across all three pillars. Capture stays the ergonomic entry; the graph is built for you.

## Install

```bash
pip install auditable
```

The core is dependency-free and torch-free. Structural-graph analysis needs the optional graph extra (NetworkX):

```bash
pip install "auditable[graph]"
```

The graph extra is required by `analyze_plan` (PRE), `analyze_run` (POST), and any graph projection. The REAL-TIME quickstart below runs on the core install alone.

## Quickstart

The sharpest demo is REAL-TIME: capture a decision, replay it under the live state, and reverse the action when it no longer holds.

```python
from auditable import Action, ActionGate, DependencySnapshot, ReferenceLedger, audit, replay

def policy(state, action):
    ok = action.cost <= state["budget"]
    return ok, "within budget" if ok else "over budget"

ledger = ReferenceLedger(balance=10_000)
gate = ActionGate(ledger)
payment = Action("payment", {"to": "acme"}, cost=4_200)

# The agent pays $4,200 against a budget snapshot that read $10,000.
with audit("payment", snapshot=DependencySnapshot(state={"budget": 10_000})) as decision:
    decision.act(payment)
receipt = gate.commit(payment)                       # paid; balance is now 5,800

# The live budget is now $3,000. Replay re-decides; the gate reverses the payment.
verdict = replay(decision.record, live_state={"budget": 3_000}, policy=policy)
gate.enforce_post_commit(verdict, receipt=receipt)
print(verdict.action.value, "->", ledger.balance)   # rollback -> 10000
```

`replay` is pure: it deep-copies the live state and the action, so a policy can never alter the signed record. See [`examples/payment_audit.py`](examples/payment_audit.py) for the full demo that binds all three spans (data, model, harness) into one signed record, and the [REAL-TIME reference](docs/realtime-replay.md) for the full replay and recovery semantics.

## PRE: Lint the Plan Before Deploy

Before a single step runs, point `analyze_plan` at a declared plan (a plain plan dict, the neutral target a LangGraph, CrewAI, or AutoGen front-end would lower into) and it runs read-only structural lints over the plan graph. It reports four reachability findings, the execution-topology keystone, and a preflight coverage report. Every check is a pure NetworkX query: no value is executed, and every finding is a structural design warning, not a validated failure prediction.

```python
from auditable.graph.pre import analyze_plan
from auditable.graph.adapters import declared_plan_v1

plan = {
    "nodes": [
        # 0: read a volatile price, but grant scope far beyond what it read.
        {"idx": 0, "agent": "planner", "kind": "tool_call",
         "reads": [{"id": "price", "volatile": True}],
         "scope": ["price", "ledger", "vendor_db"]},
        # 1: a decision that rests on the unpinned, un-revalidated price.
        {"idx": 1, "agent": "planner", "kind": "decision",
         "reads": [{"id": "price", "producer": 0, "volatile": True}],
         "control_preds": [0]},
        # 2: a consequential write of 'order', with no prior read of 'order'
        #    and no re-read of 'price' between the volatile read and the action.
        {"idx": 2, "agent": "executor", "kind": "tool_call",
         "reads": [{"id": "price", "producer": 0, "volatile": True}],
         "writes": ["order"], "control_preds": [1]},
    ]
}

report = analyze_plan(plan, adapter=declared_plan_v1)
print(report)
```

The four shipping lints, all read-only queries at `severity='warning'`:

| Lint | Fires when |
|---|---|
| `write_with_no_prior_read` | A node writes a resource that nothing in its backward slice ever read. |
| `flippable_dependency_annotations` | An unpinned, non-revalidated volatile dependency feeds a decision. This is an annotation; the would-it-flip question needs runtime values and is out of scope at PRE. |
| `scope_vs_snapshot` | Granted tool scope strictly exceeds the snapshot the node read. |
| `missing_revalidation_barrier` | A volatile read reaches a consequential action with no intervening re-read. Drift confirmation needs runtime values and is out of scope at PRE. |

The report also names the **execution-topology keystone**: the structural chokepoint of the declared plan, the node that the most other nodes transitively follow in control flow (the argmax of `execution_reach` over the `handoff_to` projection). This is a structural design lint. It is a separate concept from the POST blast-radius keystone and does not predict failure.

Alongside the lints, the **Preflight Coverage Report** is a descriptive coverage-readiness view, explicitly not a risk score. It reports the dependency-edge grade mix, the observed fraction, the saturation ratio, the exact no-score reason the runtime scorer would apply, which declared reads, writes, and edges still lack a resource identity, and the declared revalidation barriers per resource. It tells the runtime and POST scorer what it will need before it can score.

Two boundaries stated plainly. Dependency-state blast-share risk is **withheld** at PRE: the declared dependency layer is declared-only (observed fraction zero), so `analyze_plan` returns `state_b_risk=None` with `state_b_withheld=True` and a reason string, and it raises rather than emit a number if a scored verdict ever came back. A table-stakes OWASP-Agentic and CWE rule floor is **planned**, not shipping; `auditable` does not run OWASP or CWE checks today.

See [`examples/analyze_plan.py`](examples/analyze_plan.py) and the [PRE rules reference](docs/pre-rules.md).

## POST: Rank a Finished Run, Find the Keystone

`analyze_run` reads a recorded agent run, builds one decision graph, and ranks every step by how much of the run transitively rests on it, so you review the keystone first. On a tau-bench airline trajectory, the one reservation read that both later writes depend on is the keystone.

![auditable analyze_run ranks a recorded tau-bench run by structural blast share and names the keystone decision](assets/analyze_run.png)

```python
from auditable import analyze_run
from auditable.graph.adapters import tau_bench_prior_db_reads_v1

report = analyze_run(run, adapter=tau_bench_prior_db_reads_v1)
k = report.keystone
print(k.idx, k.node_attrs["tool"])   # 2  get_reservation_details
```

The score is an uncalibrated triage ranking, not a calibrated probability. In a no-score state (`no_score:single_decision`, `no_score:low_coverage`) the scores are `None`, so a withheld score never reads as zero risk. The corpus write-to-read edges are modeled (a conservative prior-read upper bound, not a causal label); the report carries these caveats in `report.notes`. The trajectory is modeled on [tau-bench](https://github.com/sierra-research/tau-bench) (Sierra Research, MIT). See [`examples/analyze_run.py`](examples/analyze_run.py) (needs the `graph` extra) and the [POST analysis reference](docs/post-analysis.md).

## How It Works

![auditable models one decision as a two-layer graph: an execution layer (control flow, observed from the trace) over a dependency layer (what each step relied on); replay catches a step that rested on a value that has since gone stale](assets/twolayer.png)

*auditable links a run into one graph with two edge layers: execution (control flow, observed from the trace) over dependency (what each step relied on). When a step rested on a value that has since gone stale, like `price`, `replay` catches it. The record itself binds three spans per decision (data, model, harness), detailed below.*

The graph kernel has two edge layers, and the distinction is load-bearing. Execution edges (`emits`, `handoff_to`) are observed from the trace. Dependency edges (`depends_on`) are declared or inferred, never read off the trace. PRE and POST both run over this same typed graph; `audit()` is the ergonomic capture entry, so you are never asked to build the graph by hand.

One agent decision crosses three spans, and `auditable` binds all three in a single signed, hash-chained record, so a decision is judged as a unit rather than as three disconnected logs:

| Span | What the record binds | Signal in v0.1 |
|---|---|---|
| **Data** | What the agent read and the dependency snapshot it relied on | Snapshot freshness |
| **Model** | Which model produced the output, and its stated basis | Decision-basis trust flag |
| **Harness** | The action executed and its cost | A static cost-cap rule, plus the replay verdict |

`replay()` re-derives whether the action still holds under the live dependency state versus the snapshot the agent used, and returns one of four routed verdicts: `ALLOW`, `ROLLBACK` (justified on the snapshot but not on live state, the stale-state case), `BLOCK` (justified on neither), or `HUMAN_REVIEW` (the policy raised `ReplayUndecidable`). A `Policy` is any callable `(state, action) -> (justified, reason)`. The `ActionGate` then executes that verdict through a `Rail`: a post-commit `ROLLBACK` or `BLOCK` calls `rail.compensate(receipt)` to reverse a committed action, rather than printing a recommendation. The shipped `ReferenceLedger` is an in-process reference rail for demos and tests (commit spends, compensate refunds); it is not a production payment rail.

Records are signed and hash-chained: each record carries a `prev_digest` and a content-addressed `record_id`. Two sinks ship today, `MemorySink` (in-process) and `FileSink` (append-only JSONL, durable across process exit, fails closed on a corrupt tail). See the [architecture reference](docs/architecture.md) for the full kernel, adapters, and sinks.

Ingestion is source-agnostic through the public `Adapter` protocol (the extension point, analogous to subclassing a base detector). Three adapters ship: `tau_bench_prior_db_reads_v1` (public-corpus trajectory, POST), `own_record_v1` (auditable's own signed records, POST), and `declared_plan_v1` (a framework-agnostic declared plan dict, PRE). The declared-plan adapter is the neutral seam a LangGraph, CrewAI, or AutoGen front-end would lower into; it is not a parser for any framework.

## Using a Single Layer

Each span's check is a standalone `Auditor` that runs on its own, with no agent and no chain, and returns a signed `Report`. `DataAuditor` scores snapshot freshness, `ModelAuditor` produces a decision-basis trust flag, and `HarnessAuditor` applies one static cost-cap rule.

```python
import time
from auditable import DataAuditor, DependencySnapshot

snapshot = DependencySnapshot(state={"budget_remaining": 1000}, captured_at=time.time() - 7 * 86400)
report = DataAuditor(max_age_seconds=86400).assess(snapshot)
print(report.flag, report.score)   # stale 1.0
```

These standalone auditors are inputs to the record, not the headline. The composition (capture, replay, recovery, and the lifecycle pillars) is the main line; the modules feed it. See [`examples/standalone_report.py`](examples/standalone_report.py).

## Scope, Stated Honestly

> [!IMPORTANT]
> **What ships today.** The full signed chain, replay under live state, executed recovery through a rail-neutral gate, two sinks (in-memory and append-only JSONL), the POST `analyze_run` ranking, and the PRE `analyze_plan` lints plus preflight coverage report. The release does **not** yet claim a learned data-anomaly method, a calibrated model-trust score, calibrated cross-layer risk, or live incremental scoring. The POST structural score is an uncalibrated ranking; the compound report is a transparent, explicitly uncalibrated debug bundle, not a calibrated risk. Everything beyond this list is on the [roadmap](#roadmap).

## Roadmap

Every item below is planned, not shipping.

- [ ] **v0.2 Data** a fitted anomaly score on the dependency state (PyOD backend), with snapshot freshness as a fallback
- [ ] **v0.3 Model and compound** a calibrated cross-layer compound, and model as a first-class graph node attribute with grounding beyond the current deterministic basis check
- [ ] **v0.3b Live** live and incremental scoring, plus the runtime resource-touch contract that fills observed dependency edges
- [ ] **v0.4 Control** data refresh or quarantine, and model fallback or sign-off control faces
- [ ] **PRE rule floor** a table-stakes OWASP-Agentic and CWE rule floor for CI legibility, consumed rather than forked
- [ ] **v1.0** pluggable sinks (OpenTelemetry, LangSmith), exportable evidence bundles, and a stable public API
- [ ] Framework integrations (LangChain, LangGraph, CrewAI) and an MCP server

## Citation

If you use `auditable` in your work, please cite the software:

```bibtex
@software{auditable,
  title  = {auditable: Capture, Replay, and Recover AI Agent Decisions},
  author = {Zhao, Yue},
  year   = {2026},
  url    = {https://github.com/yzhao062/auditable}
}
```

## License

[Apache-2.0](LICENSE).
