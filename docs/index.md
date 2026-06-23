# auditable

Capture, replay, and audit AI agent decisions across the agent lifecycle, all over one typed decision graph.

`auditable` audits AI agent decisions at three points in an agent's life. The same typed two-layer decision graph is scored and reported before deploy, while the agent runs, and after a run finishes. Detection and report generation run on one graph kernel, so the same construction serves every pillar.

## One Graph Consolidates Everything

The differentiator is a single typed decision graph that three orthogonal views project onto. You do not assemble three disconnected tools; one structure carries the analysis.

- **Lifecycle (when a check fires):** PRE before deploy, LIVE while running, POST after a run. One graph, three attach points.
- **Signal (what each decision binds):** data, model, and harness, the three orthogonal spans bound per decision.
- **Coverage (which standard a finding maps to):** OWASP and CWE threats expressed as graph-structural predicates rather than a flat checklist. See [PRE Coverage](pre-coverage.md).

![One typed two-layer decision graph at the center, read by three lifecycle attach points: PRE lints a declared plan before deploy, LIVE replays and recovers while the agent runs, and POST ranks a finished run](assets/lifecycle.png)

Because every view lands on one graph, detection and report run on one structure instead of three. That is the honest answer to "is this enough": not more rules, one consolidated substrate. Threats with no structural signature, such as prompt injection or content poisoning, route to LIVE or stay out of scope; the [Coverage](pre-coverage.md) page states that boundary.

## The Lifecycle

| Pillar | When it fires | Focus | Public entry |
|---|---|---|---|
| PRE | Design time, before any step runs | Lint a declared plan, name the control-flow chokepoint, withhold dependency-state risk | `analyze_plan` (from `auditable.graph.pre`) |
| LIVE | While the agent runs | Capture a decision, re-decide under live state, route a fix through a rail | `replay` plus `ActionGate` |
| POST | After a run completes | Rank a finished run by structural blast share, name the keystone | `analyze_run` |

See [Lifecycle](lifecycle.md) for each pillar in detail, and [`examples/end_to_end.py`](https://github.com/yzhao062/auditable/blob/main/examples/end_to_end.py) for one payment carried through all three.

## Three Crown Jewels

**The full lifecycle chain.** Capture a consequential decision with the dependency state it relied on, replay it under the state that is live now, recover by reversing the action through a rail, and rank the finished run by structural blast share. One chain from design to live operation to review.

**The graph model.** One typed two-layer graph: execution edges (`emits`, `handoff_to`) observed from the trace, over dependency edges (`depends_on`) declared or inferred. PRE, LIVE, and POST all read this one structure. The model is introduced in GRADE ([arXiv:2606.22741](https://arxiv.org/abs/2606.22741)).

**The orthogonal data, model, harness decomposition.** One agent decision crosses three spans, and `auditable` binds all three in a single signed, hash-chained record, so a decision is judged as a unit rather than as three disconnected logs.

| Span | What the record binds | Signal in v0.1 |
|---|---|---|
| **Data** | What the agent read and the dependency snapshot it relied on | Snapshot freshness |
| **Model** | Which model produced the output, and its stated basis | Decision-basis trust flag |
| **Harness** | The action executed and its cost | A static cost-cap rule, plus the replay verdict |

## Install

```bash
pip install auditable            # core: capture, replay, recovery
pip install "auditable[graph]"   # adds the graph analyses (PRE lints, POST analyze_run)
```

The graph extra pulls in NetworkX, which the PRE and POST graph entries need.

## Where to Start

- [Quickstart](quickstart.md): the smallest runnable snippet for each pillar, plus the Markdown report renderer.
- [Lifecycle](lifecycle.md): the map across PRE, LIVE, and POST.
- [PRE Coverage](pre-coverage.md): how the lints map to OWASP and CWE.
- [Audit Report](audit-report.md): the Markdown and PDF report a run produces.
- [API Reference](api.md): the full public surface.

The fastest way to see the whole lifecycle on one dataset is [`examples/end_to_end.py`](https://github.com/yzhao062/auditable/blob/main/examples/end_to_end.py): one vendor payment walked through PRE, LIVE, and POST with a single `python examples/end_to_end.py`.
