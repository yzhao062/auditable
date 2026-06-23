# Lifecycle: PRE, LIVE, POST

`auditable` attaches at three points in an agent's life: before it is deployed (PRE), while it runs (LIVE), and after a run completes (POST). Each pillar has one public entry point and answers one honest question. The three share a single foundation: the same typed two-layer decision graph (execution edges observed from the trace, dependency edges declared or inferred) is what each pillar scores and reports. The pillar changes; the graph kernel does not. See [Architecture](architecture.md) for the graph itself.

![One typed two-layer decision graph at the center, read by three lifecycle attach points: PRE lints a declared plan before deploy, LIVE replays and recovers while the agent runs, and POST ranks a finished run](assets/lifecycle.png)

| Pillar | When it fires | Focus | Public entry |
|---|---|---|---|
| PRE | Design time, before any step runs | Lint a declared plan, name the control-flow chokepoint, withhold dependency-state risk | `analyze_plan` (from `auditable.graph.pre`) |
| LIVE | While the agent runs | Capture a decision, re-decide under live state, route a fix through a rail | `replay` plus `ActionGate` |
| POST | After a run completes | Rank a finished run by structural blast share, name the keystone | `analyze_run` |

The three rows are the whole map. Each pillar below expands into one collapsible block: open the one you need. For a single payment carried through all three at once, see [One Payment Across All Three](#one-payment-across-all-three).

## The Three Pillars in Detail

??? note "PRE: Lint the Plan Before Deploy"

    PRE runs at design time over a declared plan, before a single step executes. A declared plan carries control flow and declared data reads and writes, but no observed values, so PRE computes only the parts that are honest before the run: read-only structural lints and a coverage-readiness report.

    The entry point is `analyze_plan(plan, *, adapter=declared_plan_v1)`, imported from `auditable.graph.pre`. It is not a top-level `auditable` export, and it needs the `graph` extra. It returns a `PreReport` carrying four reachability lints, the execution-topology keystone (the structural chokepoint of the plan), and a Preflight Coverage Report.

    PRE withholds one number on purpose. Dependency-state blast-share risk is the runtime and POST job, not a design-time one. The declared dependency layer is declared-only, so `analyze_plan` reports `state_b_risk=None` with `state_b_withheld=True` rather than emit a number the declared evidence cannot support. The [PRE Rules](pre-rules.md) page covers each lint, the keystone, the coverage report, and this withhold boundary.

??? note "LIVE: Replay and Recover Live"

    LIVE is the sharpest pillar. The story is capture, then replay, then recover.

    You capture one consequential decision with the dependency state it relied on using the `audit(action_type, *, snapshot, sink=None)` context manager. Later, when the state has moved, `replay(record, *, live_state, policy)` re-derives whether the same action still holds under the state that is live now. Replay is pure: it deep-copies the state and the action so a policy cannot alter the signed record, and it returns a verdict without executing anything.

    The verdict routes one of four fixes. A fix is then executed through `ActionGate` over a `Rail`, so a routed verdict reverses a committed action rather than printing a recommendation. The [LIVE Replay and Recovery](realtime-replay.md) page covers the capture API, replay semantics, the four verdicts, the policy contract, and gate execution.

??? note "POST: Rank a Finished Run, Find the Keystone"

    POST reads a recorded run after it completes. The entry point is `analyze_run(source, *, adapter, ground=True)`, a top-level export. It maps the source through an adapter into one typed session graph, scores every step by its transitive structural blast share, names the keystone (the step that most of the run rests on), and returns a ranked `AnalysisReport`.

    The POST score is an uncalibrated triage ranking, not a calibrated probability. In a no-score state, scores are `None` rather than zero, so a withheld score never reads as no risk. The [POST Analysis](post-analysis.md) page covers the ranked signal, the keystone, the no-score gates, and the modeled-edge caveat.

## One Kernel Across Three Pillars

The reason the three pillars line up is that they read the same typed graph. PRE projects a declared plan into it and runs read-only queries (`execution_reach` for the chokepoint, the four lints over the dependency and control-flow projections). POST projects a recorded run into it and scores the dependency blast structure (`downstream_reach`). LIVE captures a decision with the dependency snapshot the same graph would record, then replays it. The two PRE and POST keystones are distinct concepts computed over different projections, and they must not be conflated: the PRE keystone is a structural control-flow chokepoint, while the POST keystone is the highest dependency blast share. Both pages state this distinction where each keystone appears.

## One Payment Across All Three

[`examples/end_to_end.py`](https://github.com/yzhao062/auditable/blob/main/examples/end_to_end.py) carries a single vendor payment down all three pillars on one dataset and one state, so the pillars share one narrative. PRE lints the declared payment-approver plan and withholds State B. LIVE captures the approver's decision against a six-day-old budget snapshot, then replays it under a live budget that has dropped below the amount, and the gate rolls the payment back. POST ranks the signed record the run produced and grounds its basis. The payment amount is one real value sampled from the ULB credit-card dataset; the only constructed dimension is the temporal budget drift. Run it with `python examples/end_to_end.py` (needs the `graph` extra). The script closes by printing the aggregate audit report (verdict, keystone, findings, and the LIVE recovery) through `AuditReport.to_markdown`; see [Audit Report](audit-report.md) for that output.
