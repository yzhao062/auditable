# auditable Examples

## The Method Library

`example_methods.py` runs one method from each family of `auditable.methods` (a
configuration rule, an online monitor, a step localizer, and two run detectors) on
synthetic traces it generates, with no benchmark data and no API key. The method
library is not on PyPI yet (the current release there, 0.2.1, predates it), so
install it from the root of a repository checkout:

```
pip install ".[tabular-ad]"
python examples/example_methods.py
```

`example_agentpile.py` runs the same families on records read through
[agentpile](https://github.com/yzhao062/agentpile): an OpenAI-style conversation and an
MCP tool list, built in the script, and with `--dataset` the first 20 tau-bench runs
from Hugging Face.

```
pip install ".[agentpile,tabular-ad]"
python examples/example_agentpile.py
```

## Running the Lifecycle

`auditable` attaches to an agent at three points in its lifecycle, and these
examples run one pillar each. All three operate on the same typed two-layer
decision graph (an execution layer for control flow and a dependency layer for
the state each step relied on). Every example here runs offline with no API key,
with one exception: `example_langgraph_llm_agent.py` drives a real model through
an OpenAI-compatible endpoint, and skips cleanly when no key is set.

The framing is deliberate. PRE and LIVE are capability demos: they show what the
SDK does on a worked case, and they carry no benchmark percentage. POST is the
pillar where the benchmark results live, because the graph's two layers were
measured on a six-corpus benchmark in the GRADE paper (arXiv:2606.22741). Where a
number appears below, it is a GRADE corpus result cited to that paper, not an
output of the shipped example.

Every command below runs from the root of a repository checkout
(`git clone https://github.com/yzhao062/auditable.git`). The published wheel
carries the library only, so a `pip install auditable` alone leaves these scripts
absent.

The graph examples need the graph extra (and the LangGraph capture example needs
the langgraph extra):

```
pip install "auditable[graph]"
pip install "auditable[langgraph]"       # for example_langgraph_capture.py
pip install "auditable[langgraph,llm]"   # for example_langgraph_llm_agent.py (a real model; needs an OpenAI-compatible key)
pip install "auditable[graph,langchain]" # for example_langchain_capture.py
pip install "auditable[graph,mcp]"       # for example_mcp_capture.py (Python >= 3.10)
```

| Pillar | Example | When it runs | Benchmark number |
|--------|---------|--------------|------------------|
| PRE | `example_pre_lint_plan.py` | Design time, before any step executes | None (capability demo) |
| LIVE | `example_live_replay.py` | At decision time, on a committed action | None (capability demo) |
| LIVE | `example_live_monitor.py` | While a run streams, before it finishes | None (capability demo) |
| POST | `example_post_rank_run.py` | After a run finishes | GRADE corpus results (see below) |
| Capture | `example_langgraph_capture.py` | Capture a real LangGraph run, then rank it | None (capability demo) |
| Capture | `example_langgraph_llm_agent.py` | Capture a real LLM-driven LangGraph run, then rank it | None (capability demo) |
| Capture | `example_langchain_capture.py` | Capture a real LangChain runnable, then rank it | None (capability demo) |
| Capture + LIVE | `example_langgraph_live_replay.py` | Capture a real LangGraph run, then replay and reverse its decision under drift | None (capability demo) |
| Capture + Data | `example_state_anomaly.py` | Capture clean state history, fit from it, then score synthetic faults | None (concept demo) |
| Capture | `example_touch_capture.py` | Capture any tool loop by hand, then rank it | None (capability demo) |
| Capture | `example_mcp_capture.py` | Capture MCP tool traffic at the server boundary (ops-side mount) into an honest zero-coverage report | None (capability demo) |

## PRE: Lint a Declared Plan Before Deploy

`example_pre_lint_plan.py` takes a declared agent plan (a framework-agnostic dict
that a LangGraph, CrewAI, or AutoGen front-end would lower into) and runs read-only
structural lints over it at design time, before a single step executes. The
worked plan is a small payment approver shaped to trip every shipping lint:
a write of a resource nothing read first, a volatile read feeding a decision,
and a granted scope wider than the snapshot the step actually read.

The example reports the execution-topology keystone (the structural chokepoint of
the plan), four reachability lints, and a coverage-readiness view. It withholds
the dependency-state risk, because a declared-only plan has no observed reads to
score. This pillar is structural lints on a declared plan. It has NO benchmark
percentage, by design.

Run it:

```
python examples/example_pre_lint_plan.py
```

## LIVE: Capture, Replay, and Reverse a Payment

This example uses the legacy recovery flow: `ActionGate` and `ReferenceLedger` are
deprecated as of 0.3 and will be removed in a later release. Its scenario is money
moving under a budget that drifted. `example_live_replay.py` captures one consequential
decision with the dependency state it relied on, replays it under the state that
is live now, and executes a fix. An agent approves a vendor payment against a
budget snapshot captured six days earlier, under which the payment was in policy,
and commits it through a `ReferenceLedger` (a demo rail, not a production payment
rail). The live budget has since dropped below the amount. `replay()` re-decides
on the live state and routes a rollback, and the `ActionGate` executes it through
the ledger, restoring the balance.

The example prints the replay verdict (`ROLLBACK`) and the restored balance. The
point is recovery: the gate reverses a committed action rather than flagging it
and stopping. This is a capability demo on a worked payment. It has NO benchmark
percentage; the recovery line is an executed reversal, not a measured failure
reduction.

`example_end_to_end.py` runs the same payment through all three pillars on one
dataset and one state, so PRE, LIVE, and POST share a single narrative; it is the
fuller companion to this LIVE demo.

Run it:

```
python examples/example_live_replay.py
```

## LIVE: Monitor a Run as It Streams

`example_live_monitor.py` is the other LIVE capability: a running structural score while
the run is still going. Where `example_live_replay.py` re-decides one committed action,
`LiveSession` watches the whole run grow. Feed each step as it happens
(`live.observe(step)`) and the running report names the keystone and its blast share over
the run seen so far, marked `completeness=prefix`, so a caller can flag, gate, or
checkpoint a high-blast step before the run ends. It is the same structural kernel as
POST, run on a growing graph.

The example replays the retail trajectory one step at a time so the score change is
visible: the first read carries no blast when it happens, becomes the keystone once the
first write lands on it, and its blast share then moves as later steps arrive. That
movement is the honest part. A prefix score is a live triage and gating signal, not a
validated early-warning probability; the blast structure can resolve late, so the keystone
may shift. This is a capability demo with NO benchmark percentage.

Run it:

```
python examples/example_live_monitor.py
```

## POST: Rank a Finished Run and Name the Keystone

`example_post_rank_run.py` reads a run after it completes, builds one typed
decision graph, scores every step by how much of the rest of the run transitively
rests on it, and names the keystone that heads the structural triage order. The shipped
trajectory is a small illustrative tau-bench-style airline run: the agent reads one
reservation, then makes two writes that both rest on that read, so the read is
the keystone. The example prints a keystone ranking on this small trajectory, not
a benchmark score.

CatchBench evaluates methods built from the two graph layers:

- **Run-level detection does not separate dependency structure from run size.**
  A linear size baseline cannot distinguish dependency shape from a nonlinear
  function of size, and GRADE retired the beyond-size reading on 2026-09-08.
- **Top-3 0.614 belongs to GRADE's supervised `exec-rank (sup.)`.** On
  Who&When (126 failed runs), the shipping `auditable (blast share)` method
  scores 0.516 and ties the position prior at displayed precision.

The six GRADE corpora are tau-bench and tau2-bench (tool use), SWE-agent,
SWE-Gym, and OpenHands (coding), and AgentRewardBench (web). Who&When supplies the
126 failed runs used for step localization. The coding corpora (SWE-Gym,
SWE-agent) appear here only as GRADE evaluation corpora; that corpus data does not
ship with this SDK, so no runnable SWE-Gym or coding example exists here.

These numbers are NOT outputs of the shipped `example_post_rank_run.py` example. The shipped example
runs on a small illustrative trajectory and prints a keystone ranking; the
full-corpus evaluation lives in CatchBench.

Run it:

```
python examples/example_post_rank_run.py
```

## Capture a Real Run: Plug In Your Agent

The PRE, LIVE, and POST examples above run on a declared plan, a hand-built
decision, or a corpus trajectory. These two capture a *real* run and lower it into
the same typed graph, with the dependency layer recorded as OBSERVED edges over the
resources each step read and wrote.

`example_langgraph_capture.py` instruments a real LangGraph `StateGraph` (a small
payment approver of pure-function nodes, no LLM or network). Wrapping the builder
with `instrument(...)` records, per node, the state channels it read and the
channels its returned update wrote; `analyze_run` then ranks the run and names the
keystone. The dependency edges are observed channel-level read-after-write touches,
matched across LangGraph's superstep barrier and reducer-aware, so the report shows
`observed=100%` on this run. It needs `pip install "auditable[langgraph]"`.

`example_langgraph_llm_agent.py` is the same capture path with a real model in the
loop: every node makes an OpenAI-compatible chat call instead of running a pure
function. The captured edges are identical, because `instrument(...)` records the
state channels each node read and wrote, not what the node did to produce the
write. So the dependency layer comes out `observed=100%` whether a write came from
a pure function or a frontier model, which is the point the example makes. It reads
`OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`, `OPENAI_MODEL`), works against a
plain OpenAI key or any compatible gateway or local server, and prints a notice and
exits cleanly when no key is set. It needs `pip install "auditable[langgraph,llm]"`.

`example_langgraph_live_replay.py` carries one real LangGraph run through both
pillars. The same wrapped builder that yields the POST ranking also lowers the
marked decision node into a replayable `DecisionRecord`. `to_records` takes the
budget the approval relied on into the snapshot, and `action_args` / `action_costs`
lift the recipient and the real payment cost onto the action itself, so the lowered
record is the executable payment rather than a state-only stub:

```python
record = builder.to_records(
    decisions={"approve": "vendor_payment"},
    action_args={"approve": {"recipient": "recipient"}},
    action_costs={"approve": "amount"},
)[0]
```

After capture, the example lowers the live budget below the approved amount (fault
injection, a controlled post-capture change, not drift the corpus supplied) and
`replay` re-decides on the live state with a cost-based policy: the action's cost
exceeds the budget that is live now, so it routes a `ROLLBACK` that the `ActionGate`
executes through a reference ledger. It needs `pip install "auditable[langgraph]"`.
It is a capability demo on one run, not the comparative replay-versus-baseline
experiment.

`example_touch_capture.py` does the same for a plain tool loop with no framework,
using the generic `TouchRecorder`: each step declares its `reads()` and `writes()`
and the same matcher produces the observed edges. The recorder also carries the
relied-on values (`reads(..., value=...)`) and a `decides(...)` action, so
`rec.to_records()` lowers a consequential step into the same replayable record. This
is the framework-agnostic path for a raw OpenAI or Anthropic loop, or any scheduler.

Both are capability demos: the keystone ranking is a triage signal, not a
calibrated probability, and the OBSERVED grade is an observed touch match, not a
precise causal claim. Run them:

```
python examples/example_langgraph_capture.py
python examples/example_touch_capture.py
```

## Other Files Here

- `example_end_to_end.py` hands a single payment down all three pillars (PRE lints
  the plan, LIVE replays and reverses the payment, POST ranks the signed record the
  run produced) so the pillars share one narrative.
- `example_payment_audit.py` is the fuller LIVE walkthrough that also attaches the
  data, model, and harness report leaves to the signed record.
- `example_audit_report.py` aggregates the three pillars into one `AuditReport` and
  renders it two ways: the agent-facing Markdown and the human-facing PDF.
- `example_standalone_report.py` scores one stage (the data a decision relied on) on
  its own, with no chain and no replay.
