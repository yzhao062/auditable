<div align="center">

# auditable

**Audit any agent decision across its past, present, and future, on one typed graph.**

*Your logs show what the agent did. `auditable` shows what it relied on, replays it under live state, and rolls it back when it no longer holds.*

[![PyPI](https://img.shields.io/pypi/v/auditable.svg)](https://pypi.org/project/auditable/)
[![Python](https://img.shields.io/pypi/pyversions/auditable.svg)](https://pypi.org/project/auditable/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yzhao062/auditable.svg?style=social)](https://github.com/yzhao062/auditable)

[Compare](#auditable-vs-the-tools-you-already-use) · [Flagship Demo](#the-flagship-moment) · [Plug In Your Agent](#plug-in-your-agent) · [Lifecycle](#the-lifecycle) · [Install](#install) · [Docs](https://auditable-ai.readthedocs.io/) · [Roadmap](#roadmap)

![One typed two-layer decision graph at the center, read by three lifecycle attach points: PRE lints a declared plan before deploy, LIVE replays and recovers while the agent runs, and POST ranks a finished run](https://raw.githubusercontent.com/yzhao062/auditable/main/assets/lifecycle.png)

</div>

### auditable vs. the Tools You Already Use

| Capability | **auditable** | Tracing and Observability<br><sub>[LangSmith](#ref-1), [Langfuse](#ref-2), [Phoenix](#ref-3)</sub> | Eval Harnesses<br><sub>[DeepEval](#ref-5), [Ragas](#ref-6), [promptfoo](#ref-7)</sub> | Guardrails<br><sub>[NeMo](#ref-9), [Guardrails AI](#ref-10), [Lakera](#ref-11)</sub> |
|---|:---:|:---:|:---:|:---:|
| Captures the dependency state a decision relied on | ✅ | 🟡 | ❌ | ❌ |
| Re-decides under live state (replay) | ✅ | 🟡 | ❌ | ❌ |
| Reverses a committed action (rollback) | ✅ | ❌ | ❌ | ❌ |
| One graph across plan, run, and review | ✅ | 🟡 | ❌ | ❌ |
| Framework-agnostic, dependency-free core | ✅ | 🟡 | 🟡 | 🟡 |

<sub>✅ yes · 🟡 partial · ❌ no, per each tool's public docs (2026). These categories are complementary to `auditable`, not competitors: tracing captures execution, evals score quality, guardrails block at runtime. `auditable` is the audit-and-recovery layer that plugs into them. See [References](#references).</sub>

**`auditable` is the only column with the full set.** What makes it new:

- **A unified graph representation for agentic AI.** Every agent run becomes one typed graph that PRE, LIVE, and POST all read. One representation from plan to live operation to review, instead of three disconnected tools.
- **Recover, do not just observe.** `auditable` captures the dependency state a decision relied on, replays it under the state that is live now, and reverses the committed action through a compensating rail when it no longer holds. Logging tells you what broke; `auditable` undoes it.
- **One decision, three spans, judged together.** Data, model, and harness are bound in a single signed, hash-chained record, so a decision is audited as one unit, not three disconnected logs.

### Proven on Public Agent Benchmarks

The graph is not just structure. Across six public agent corpora (GRADE, [arXiv:2606.22741](https://arxiv.org/abs/2606.22741)), the dependency layer predicts which runs fail at **ROC-AUC 0.805** where run length carries no signal, and the execution layer localizes the faulting step at **Top-3 0.614**. See [the benchmark detail](docs/post-analysis.md) for the full numbers and corpora.

<p align="center">
  <img src="https://raw.githubusercontent.com/yzhao062/auditable/main/assets/grade-transfer.png" alt="Leave-one-corpus-out transfer ROC-AUC: the size-normalized dependency signal clears chance on all six held-out agent corpora, while run size drops below chance on tau-bench and SWE-Gym" width="49%">
  <img src="https://raw.githubusercontent.com/yzhao062/auditable/main/assets/grade-localization.png" alt="Step-level fault localization on Who and When: ranking steps by execution-graph structure beats an early-fault position prior on top-1, top-3, and MRR" width="49%">
</p>
<p align="center"><sub>Figures from GRADE (<a href="https://arxiv.org/abs/2606.22741">arXiv:2606.22741</a>).</sub></p>

Agents act on dependency state that quietly drifts. A budget read minutes ago can fall below the amount already committed; a price pinned at plan time can move before the action lands. Most tools log what happened, yet they cannot re-decide under the state that is live now, so a stale decision stands until a human notices. `auditable` closes that recovery gap across the lifecycle on one graph: it captures the decision, replays it against live state, and reverses the committed action when it no longer holds.

## The Flagship Moment

One payment, walked through the whole lifecycle in 18 `auditable` calls. The agent approves a $2,083.20 vendor payment against a budget snapshot that covered it. Six days later the live budget has dropped below the amount. `replay` re-decides on the live state, and the gate reverses the committed payment. This is recovery, not a log line.

```bash
pip install "auditable[graph]"
python examples/example_end_to_end.py
```

The run prints a single audit report: a REVIEW verdict, the keystone with its coverage reason, six findings with severity tags and recommended actions, and the LIVE recovery that rolled the payment back. Paste it into a pull request or an issue.

![Audit report page for a payment-approver run. A REVIEW verdict banner sits above a roll-up of zero blocks, one rollback, and five PRE lints. Sections below list the keystone decision with its coverage reason, six numbered findings each carrying a severity tag and a recommended action, and a LIVE recovery tally showing the rolled-back payment record.](https://raw.githubusercontent.com/yzhao062/auditable/main/assets/audit-report-sample.png)

## Plug In Your Agent

`auditable` works with the agent you already run. Wrap a LangGraph `StateGraph` of plain sync or async function nodes over TypedDict state, and every node's reads and writes over the state channels become **observed** dependency edges, matched across the superstep barrier, with no change to your node logic:

```python
from langgraph.graph import StateGraph
from auditable import analyze_run
from auditable.integrations.langgraph import instrument

builder = instrument(StateGraph(State))          # 1) wrap once, then build / compile / invoke as usual
...
graph = builder.compile()
graph.invoke(initial_state)
report = analyze_run(builder, adapter=builder)   # 2) the observed dependency graph for that run
print(report.keystone)                           # the step the rest of the run rests on
```

```bash
pip install "auditable[langgraph]"
python examples/example_langgraph_capture.py     # a real LangGraph run -> observed=100%, keystone named
```

The same wrapped builder also yields **replayable** records, so the LIVE pillar runs on the captured run, not a hand-built dict. `builder.to_records(decisions={"approve": "vendor_payment"})` lowers a marked decision node into a `DecisionRecord` carrying the state it relied on, and `replay(record, live_state=...)` re-decides it under state that is live now and routes a rollback. See [`example_langgraph_live_replay.py`](examples/example_langgraph_live_replay.py): one real LangGraph run, captured, ranked (POST), then replayed and reversed under a drifted budget (LIVE).

Want a real model in the loop? [`example_langgraph_llm_agent.py`](examples/example_langgraph_llm_agent.py) runs the same capture with live LLM nodes against any OpenAI-compatible endpoint (a plain OpenAI key, a gateway, or a local vLLM or Ollama server). The captured edges are identical, because the capture sees the state channels a node read and wrote, not the model call inside it.

Not on LangGraph? The framework-agnostic `TouchRecorder` captures the same observed edges from any loop (a raw OpenAI or Anthropic agent, your own scheduler) by declaring each step's `reads()` and `writes()`. See [`examples/example_touch_capture.py`](examples/example_touch_capture.py). **Roadmap:** LangChain, CrewAI, MCP, and OpenTelemetry.

## Examples and Integrations

Point `auditable` at a scenario and it builds the same typed graph. Each row links a runnable example; browse them all in [`examples/README.md`](examples/README.md).

| Scenario | Pillar | Builds the graph from | Run |
|---|---|---|---|
| Capture a real LangGraph agent | LIVE → POST | a live `StateGraph` run (`instrument`) | [`example_langgraph_capture.py`](examples/example_langgraph_capture.py) |
| Capture a real LLM agent (live model) | LIVE → POST | a model-driven `StateGraph` run (`instrument` + OpenAI-compatible) | [`example_langgraph_llm_agent.py`](examples/example_langgraph_llm_agent.py) |
| Capture any tool loop by hand | LIVE → POST | declared resource touches (`TouchRecorder`) | [`example_touch_capture.py`](examples/example_touch_capture.py) |
| Capture a LangGraph agent, then replay and reverse it | LIVE → POST | a live `StateGraph` run lowered to replayable records (`instrument` + `to_records`) | [`example_langgraph_live_replay.py`](examples/example_langgraph_live_replay.py) |
| Lint a declared plan before deploy | PRE | a framework-agnostic plan dict (`declared_plan_v1`) | [`example_pre_lint_plan.py`](examples/example_pre_lint_plan.py) |
| Recover a payment as the budget drifts | LIVE | a decision captured live (`audit` + `replay`) | [`example_live_replay.py`](examples/example_live_replay.py) |
| Monitor a run as it streams, name the keystone live | LIVE | a run scored prefix by prefix (`LiveSession`) | [`example_live_monitor.py`](examples/example_live_monitor.py) |
| Rank a tau-bench run, name the keystone | POST | a tau-bench trajectory (`tau_bench_prior_db_reads_v1`) | [`example_post_rank_run.py`](examples/example_post_rank_run.py) |
| Walk one payment through every pillar | PRE, LIVE, POST | the full lifecycle (`own_record_v1` for POST) | [`example_end_to_end.py`](examples/example_end_to_end.py) |

The first four rows capture a real run; see [Plug In Your Agent](#plug-in-your-agent) for the two-line setup. **Roadmap:** LangChain, CrewAI, MCP, and OpenTelemetry.

## The Lifecycle

`auditable` runs the same detection-and-report pass over one typed decision graph at three points in an agent's life. The graph kernel stays constant; only the pillar changes (when it fires, what it scores).

| Pillar | When It Fires | Public Entry | Focus |
|---|---|---|---|
| **PRE** | Before deploy | `analyze_plan` | Read-only structural lints on a declared plan. Names the control-flow chokepoint. Dependency-state risk withheld. |
| **LIVE** | While running | `audit` + `replay` + `ActionGate` | Capture a decision, re-decide under live state, route a fix (allow, block, review, rollback) through a rail. The sharpest pillar. |
| **POST** | After a run | `analyze_run` | Rank a finished run by structural blast share. Name the keystone the run rests on, so you review that step first. |

## Install

```bash
pip install auditable
```

The core is dependency-free and torch-free. Structural-graph analysis (`analyze_plan` for PRE, `analyze_run` for POST) needs the optional graph extra (NetworkX):

```bash
pip install "auditable[graph]"
```

The LIVE example runs on the core install alone.

Docs: https://auditable-ai.readthedocs.io/

## The Three Pillars, in Detail

<details>
<summary><b>PRE: lint the plan before deploy</b> (four read-only lints, the chokepoint, a coverage report)</summary>

Point `analyze_plan` at a declared plan (a plain dict, the neutral target a LangGraph, CrewAI, or AutoGen front-end would lower into) and it runs read-only structural lints over the plan graph. Every check is a pure NetworkX query: no value is executed, and every finding is a structural design warning, not a validated failure prediction.

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

| Lint | Fires When |
|---|---|
| `write_with_no_prior_read` | A node writes a resource that nothing in its backward slice ever read. |
| `flippable_dependency_annotation` | An unpinned, non-revalidated volatile dependency feeds a decision. This is an annotation; the would-it-flip question needs runtime values and is out of scope at PRE. |
| `scope_vs_snapshot` | Granted tool scope strictly exceeds the snapshot the node read. |
| `missing_revalidation_barrier` | A volatile read reaches a consequential action with no intervening re-read. Drift confirmation needs runtime values and is out of scope at PRE. |

The report also names the **execution-topology keystone**: the structural chokepoint of the declared plan, the node that the most other nodes transitively follow in control flow (the argmax of `execution_reach` over the `handoff_to` projection). This is a structural design lint, a separate concept from the POST blast-radius keystone, and it does not predict failure.

Alongside the lints, the **Preflight Coverage Report** is a descriptive coverage-readiness view, explicitly not a risk score. It reports the dependency-edge grade mix, the observed fraction, the saturation ratio, the exact no-score reason the runtime scorer would apply, which declared reads, writes, and edges still lack a resource identity, and the declared revalidation barriers per resource.

Two boundaries, stated plainly. Dependency-state blast-share risk is **withheld** at PRE: the declared dependency layer is declared-only (observed fraction zero), so `analyze_plan` returns `state_b_risk=None` with `state_b_withheld=True` and a reason string, and it raises rather than emit a number if a scored verdict ever came back. A table-stakes OWASP-Agentic and CWE rule floor is **planned**, not shipping.

See [`examples/example_pre_lint_plan.py`](examples/example_pre_lint_plan.py) and the [PRE rules reference](docs/pre-rules.md).

</details>

<details>
<summary><b>POST: rank a finished run, find the keystone</b> (`analyze_run` over a recorded trajectory)</summary>

`analyze_run` reads a recorded agent run, builds one decision graph, and ranks every step by how much of the run transitively rests on it, so you review the keystone first. On a tau-bench airline trajectory, the one reservation read that both later writes depend on is the keystone.

![auditable analyze_run ranks a recorded tau-bench run by structural blast share and names the keystone decision](https://raw.githubusercontent.com/yzhao062/auditable/main/assets/analyze_run.png)

```python
from auditable import analyze_run
from auditable.graph.adapters import tau_bench_prior_db_reads_v1

report = analyze_run(run, adapter=tau_bench_prior_db_reads_v1)
k = report.keystone
print(k.idx, k.node_attrs["tool"])   # 2  get_reservation_details
```

The score is an uncalibrated triage ranking, not a calibrated probability. In a no-score state (`no_score:single_decision`, `no_score:low_coverage`) the scores are `None`, so a withheld score never reads as zero risk. The corpus write-to-read edges are modeled (a conservative prior-read upper bound, not a causal label); the report carries these caveats in `report.notes`. The trajectory is modeled on [tau-bench](https://github.com/sierra-research/tau-bench) (Sierra Research, MIT). See [`examples/example_post_rank_run.py`](examples/example_post_rank_run.py) and the [POST analysis reference](docs/post-analysis.md).

</details>

<details>
<summary><b>How It Works: the two-layer graph, the signed record, the rail</b> (kernel internals)</summary>

![auditable models one decision as a two-layer graph: an execution layer (control flow, observed from the trace) over a dependency layer (what each step relied on); replay catches a step that rested on a value that has since gone stale](https://raw.githubusercontent.com/yzhao062/auditable/main/assets/twolayer.png)

*auditable links a run into one graph with two edge layers: execution (control flow, observed from the trace) over dependency (what each step relied on). When a step rested on a value that has since gone stale, like `price`, `replay` catches it. The record itself binds three spans per decision (data, model, harness).*

The graph kernel has two edge layers, and the distinction is load-bearing. Execution edges (`emits`, `handoff_to`) are observed from the trace. Dependency edges (`depends_on`) are declared, inferred, or observed, and every edge records how it is known: the corpus and plan adapters declare or infer them, while the LangGraph capture path and the `TouchRecorder` read them off a real run as observed channel touches. PRE and POST both run over this same typed graph; `audit()` is the ergonomic capture entry, so you are never asked to build the graph by hand.

One agent decision crosses three spans, and `auditable` binds all three in a single signed, hash-chained record:

| Span | What the Record Binds | Signal in v0.1 |
|---|---|---|
| **Data** | What the agent read and the dependency snapshot it relied on | Snapshot freshness |
| **Model** | Which model produced the output, and its stated basis | Decision-basis trust flag |
| **Harness** | The action executed and its cost | A static cost-cap rule, plus the replay verdict |

`replay()` re-derives whether the action still holds under the live dependency state versus the snapshot the agent used, and returns one of four routed verdicts: `ALLOW`, `ROLLBACK` (justified on the snapshot but not on live state, the stale-state case), `BLOCK` (justified on neither), or `HUMAN_REVIEW` (the policy raised `ReplayUndecidable`). A `Policy` is any callable `(state, action) -> (justified, reason)`. The `ActionGate` then executes that verdict through a `Rail`: a post-commit `ROLLBACK` or `BLOCK` calls `rail.compensate(receipt)` to reverse a committed action, rather than printing a recommendation. The shipped `ReferenceLedger` is an in-process reference rail for demos and tests; it is not a production payment rail.

Records are signed and hash-chained: each record carries a `prev_digest` and a content-addressed `record_id`. Two sinks ship today, `MemorySink` (in-process) and `FileSink` (append-only JSONL, durable across process exit, fails closed on a corrupt tail).

Ingestion is source-agnostic through the public `Adapter` protocol. The corpus and plan adapters ship `tau_bench_prior_db_reads_v1` (public-corpus trajectory, POST), `own_record_v1` (auditable's own signed records, POST), and `declared_plan_v1` (a framework-agnostic declared plan dict, PRE). For a real run, the LangGraph capture path (`auditable.integrations.langgraph.instrument`) and the generic `TouchRecorder` produce observed dependency edges with a structured `ResourceRef`, matched superstep-aware and reducer-aware. The declared-plan adapter stays the neutral PRE seam a LangGraph or CrewAI front-end would lower a plan into; it is not a parser for any framework.

See the [architecture reference](docs/architecture.md) for the full kernel, adapters, and sinks.

</details>

<details>
<summary><b>Using a single layer</b> (standalone auditors as inputs to the record)</summary>

Each span's check is a standalone `Auditor` that runs on its own, with no agent and no chain, and returns a signed `Report`. `DataAuditor` scores snapshot freshness, `ModelAuditor` produces a decision-basis trust flag, and `HarnessAuditor` applies one static cost-cap rule.

```python
import time
from auditable import DataAuditor, DependencySnapshot

snapshot = DependencySnapshot(state={"budget_remaining": 1000}, captured_at=time.time() - 7 * 86400)
report = DataAuditor(max_age_seconds=86400).assess(snapshot)
print(report.flag, report.score)   # stale 1.0
```

These standalone auditors are inputs to the record, not the headline. The composition (capture, replay, recovery, and the lifecycle pillars) is the main line; the modules feed it. See [`examples/example_standalone_report.py`](examples/example_standalone_report.py).

</details>

<details>
<summary><b>Scope, stated honestly</b> (what ships today vs. what is planned)</summary>

**What ships today.** The full signed chain, replay under live state, executed recovery through a rail-neutral gate, two sinks (in-memory and append-only JSONL), the POST `analyze_run` ranking, the PRE `analyze_plan` lints plus preflight coverage report, and real-run capture into observed dependency edges (LangGraph via `instrument`, any loop via `TouchRecorder`).

The release does **not** yet claim a learned data-anomaly method, a calibrated model-trust score, calibrated cross-layer risk, or live incremental scoring. The POST structural score is an uncalibrated ranking; the compound report is a transparent, explicitly uncalibrated debug bundle. Everything beyond this list is on the [roadmap](#roadmap).

</details>

## Roadmap

<details>
<summary>What ships today, and what is next</summary>

<br>

Shipping today: the full capture, replay, and recover chain; PRE plan lints; POST run ranking; signed, hash-chained records; two sinks. Next:

- **Smarter drift detection.** Learn what healthy dependency state looks like and flag anomalies, not just staleness (PyOD backend, with freshness as the fallback).
- **One calibrated risk score.** Combine the data, model, and action signals into a single calibrated score, with the model treated as a first-class node in the graph.
- **Live scoring.** Score decisions on the prefix graph as the agent runs, not only after the run completes. (Capturing what each step read and wrote already ships for LangGraph and the `TouchRecorder`.)
- **Automatic fixes.** Refresh stale data, quarantine bad inputs, fall back to a safer model, or require human sign-off.
- **Built-in CI checks.** A table-stakes OWASP-Agentic and CWE rule floor for continuous integration.
- **Integrations (1.0).** LangGraph capture ships now (`instrument`); next are one-line hooks for LangChain and CrewAI, an MCP server, export to OpenTelemetry and LangSmith, downloadable evidence bundles, and a stable public API.

</details>

## Related Projects

`auditable` is the SDK in a small family of open agent-reliability projects:

- **[awesome-auditable-ai](https://github.com/yzhao062/awesome-auditable-ai)** — a curated reading list on AI agent reliability and auditing.
- **[GRADE](https://arxiv.org/abs/2606.22741)** — the research behind the typed two-layer graph this library is built on (also in [References](#references)).

## References

<details>
<summary>Compared tools, benchmarks, and method</summary>

<br>

The capability marks in the table above are per each project's public documentation as of 2026.

**Compared tools**

<a id="ref-1"></a>`[1]` [LangSmith](https://docs.langchain.com/langsmith)<br>
<a id="ref-2"></a>`[2]` [Langfuse](https://github.com/langfuse/langfuse)<br>
<a id="ref-3"></a>`[3]` [Arize Phoenix](https://github.com/Arize-ai/phoenix)<br>
<a id="ref-4"></a>`[4]` [Weights & Biases Weave](https://github.com/wandb/weave)<br>
<a id="ref-5"></a>`[5]` [DeepEval](https://github.com/confident-ai/deepeval)<br>
<a id="ref-6"></a>`[6]` [Ragas](https://github.com/explodinggradients/ragas)<br>
<a id="ref-7"></a>`[7]` [promptfoo](https://github.com/promptfoo/promptfoo)<br>
<a id="ref-8"></a>`[8]` [OpenAI Evals](https://github.com/openai/evals)<br>
<a id="ref-9"></a>`[9]` [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)<br>
<a id="ref-10"></a>`[10]` [Guardrails AI](https://github.com/guardrails-ai/guardrails)<br>
<a id="ref-11"></a>`[11]` [Lakera Guard](https://www.lakera.ai/lakera-guard)

**Benchmarks and datasets**

<a id="ref-12"></a>`[12]` [tau-bench](https://github.com/sierra-research/tau-bench) (arXiv:2406.12045)<br>
<a id="ref-13"></a>`[13]` [Who&When: Which Agent Causes Task Failures and When?](https://github.com/ag2ai/Agents_Failure_Attribution) (arXiv:2505.00212)

**Method**

<a id="ref-14"></a>`[14]` [GRADE: Graph Representation of LLM Agent Dependency and Execution](https://arxiv.org/abs/2606.22741)

</details>

## Citation

If you use `auditable` in your work, please cite the GRADE paper, the typed two-layer graph model the library is built on:

```bibtex
@article{zhao2026grade,
  title   = {GRADE: Graph Representation of LLM Agent Dependency and Execution},
  author  = {Zhao, Yue},
  journal = {arXiv preprint arXiv:2606.22741},
  year    = {2026}
}
```

## License

[Apache-2.0](LICENSE).
