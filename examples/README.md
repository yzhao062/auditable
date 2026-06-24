# auditable Examples: Running the Lifecycle

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

The graph examples need the graph extra (and the LangGraph capture example needs
the langgraph extra):

```
pip install "auditable[graph]"
pip install "auditable[langgraph]"       # for example_langgraph_capture.py
pip install "auditable[langgraph,llm]"   # for example_langgraph_llm_agent.py (a real model; needs an OpenAI-compatible key)
```

| Pillar | Example | When it runs | Benchmark number |
|--------|---------|--------------|------------------|
| PRE | `example_pre_lint_plan.py` | Design time, before any step executes | None (capability demo) |
| LIVE | `example_live_replay.py` | At decision time, on a committed action | None (capability demo) |
| POST | `example_post_rank_run.py` | After a run finishes | GRADE corpus results (see below) |
| Capture | `example_langgraph_capture.py` | Capture a real LangGraph run, then rank it | None (capability demo) |
| Capture | `example_langgraph_llm_agent.py` | Capture a real LLM-driven LangGraph run, then rank it | None (capability demo) |
| Capture + LIVE | `example_langgraph_live_replay.py` | Capture a real LangGraph run, then replay and reverse its decision under drift | None (capability demo) |
| Capture | `example_touch_capture.py` | Capture any tool loop by hand, then rank it | None (capability demo) |

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

LIVE is the flagship pillar, and its scenario is the most relatable: money moving
under a budget that drifted. `example_live_replay.py` captures one consequential
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

## POST: Rank a Finished Run and Name the Keystone

`example_post_rank_run.py` reads a run after it completes, builds one typed
decision graph, scores every step by how much of the rest of the run transitively
rests on it, and names the single step to review first (the keystone). The shipped
trajectory is a small illustrative tau-bench-style airline run: the agent reads one
reservation, then makes two writes that both rest on that read, so the read is
the keystone. The example prints a keystone ranking on this small trajectory, not
a benchmark score.

This is the pillar that surfaces the GRADE graph results. The two graph layers
were measured on a six-corpus benchmark in GRADE (arXiv:2606.22741):

- **Dependency layer PREDICTS run failure.** ROC-AUC 0.805 on SWE-Gym, +0.142
  over the run-length baseline (0.663). Under leave-one-corpus-out transfer, the
  dependency signal stays 0.551 to 0.662 above chance on all six corpora, while
  the run-length baseline inverts below 0.5 (0.468 on tau-bench, 0.350 on SWE-Gym).
- **Execution layer LOCALIZES the faulting step.** On Who&When (126 failed runs),
  Top-3 0.614, MRR 0.454, Top-1 0.211; against a position prior at
  0.516 / 0.407 / 0.159 and a random floor at 0.346 / 0.324 / 0.119.

The six GRADE corpora are tau-bench and tau2-bench (tool use), SWE-agent,
SWE-Gym, and OpenHands (coding), and AgentRewardBench (web). Who&When supplies the
126 failed runs used for step localization. The coding corpora (SWE-Gym,
SWE-agent) appear here only as GRADE evaluation corpora; that corpus data does not
ship with this SDK, so no runnable SWE-Gym or coding example exists here.

These numbers are GRADE full-corpus results, cited to arXiv:2606.22741. They are
NOT outputs of the shipped `example_post_rank_run.py` example. The shipped example
runs on a small illustrative trajectory and prints a keystone ranking; the
full-corpus evaluation lives in GRADE.

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
