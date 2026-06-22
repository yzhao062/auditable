# Lifecycle: PRE, REAL-TIME, POST

`auditable` attaches at three points in an agent's life: before it is deployed (PRE), while it runs (REAL-TIME), and after a run completes (POST). Each pillar has one public entry point and answers one honest question. The three share a single foundation: the same typed two-layer decision graph (execution edges observed from the trace, dependency edges declared or inferred) is what each pillar scores and reports. The pillar changes; the graph kernel does not. See [Architecture](architecture.md) for the graph itself.

| Pillar | When it fires | Public entry | What it does |
|---|---|---|---|
| PRE | Design time, before any step runs | `analyze_plan` (from `auditable.graph.pre`) | Runs read-only graph lints over a declared plan and reports the structural chokepoint and a coverage-readiness view. |
| REAL-TIME | While the agent runs | `replay` plus `ActionGate` | Re-derives whether a captured decision still holds under live state and executes a routed fix through a rail. |
| POST | After a run completes | `analyze_run` | Builds one typed session graph, ranks every step by structural blast share, and names the keystone. |

## PRE: Lint the Plan Before Deploy

PRE runs at design time over a declared plan, before a single step executes. A declared plan carries control flow and declared data reads and writes, but no observed values, so PRE computes only the parts that are honest before the run: read-only structural lints and a coverage-readiness report.

The entry point is `analyze_plan(plan, *, adapter=declared_plan_v1)`, imported from `auditable.graph.pre`. It is not a top-level `auditable` export, and it needs the `graph` extra. It returns a `PreReport` carrying four reachability lints, the execution-topology keystone (the structural chokepoint of the plan), and a Preflight Coverage Report.

PRE withholds one number on purpose. Dependency-state blast-share risk is the runtime and POST job, not a design-time one. The declared dependency layer is declared-only, so `analyze_plan` reports `state_b_risk=None` with `state_b_withheld=True` rather than emit a number the declared evidence cannot support. The [PRE Rules](pre-rules.md) page covers each lint, the keystone, the coverage report, and this withhold boundary.

## REAL-TIME: Replay and Recover Live

REAL-TIME is the sharpest pillar. The story is capture, then replay, then recover.

You capture one consequential decision with the dependency state it relied on using the `audit(action_type, *, snapshot, sink=None)` context manager. Later, when the state has moved, `replay(record, *, live_state, policy)` re-derives whether the same action still holds under the state that is live now. Replay is pure: it deep-copies the state and the action so a policy cannot alter the signed record, and it returns a verdict without executing anything.

The verdict routes one of four fixes. A fix is then executed through `ActionGate` over a `Rail`, so a routed verdict reverses a committed action rather than printing a recommendation. The [REAL-TIME Replay and Recovery](realtime-replay.md) page covers the capture API, replay semantics, the four verdicts, the policy contract, and gate execution.

## POST: Rank a Finished Run, Find the Keystone

POST reads a recorded run after it completes. The entry point is `analyze_run(source, *, adapter, ground=True)`, a top-level export. It maps the source through an adapter into one typed session graph, scores every step by its transitive structural blast share, names the keystone (the step that most of the run rests on), and returns a ranked `AnalysisReport`.

The POST score is an uncalibrated triage ranking, not a calibrated probability. In a no-score state, scores are `None` rather than zero, so a withheld score never reads as no risk. The [POST Analysis](post-analysis.md) page covers the ranked signal, the keystone, the no-score gates, and the modeled-edge caveat.

## One Kernel Across Three Pillars

The reason the three pillars line up is that they read the same typed graph. PRE projects a declared plan into it and runs read-only queries (`execution_reach` for the chokepoint, the four lints over the dependency and control-flow projections). POST projects a recorded run into it and scores the dependency blast structure (`downstream_reach`). REAL-TIME captures a decision with the dependency snapshot the same graph would record, then replays it. The two PRE and POST keystones are distinct concepts computed over different projections, and they must not be conflated: the PRE keystone is a structural control-flow chokepoint, while the POST keystone is the highest dependency blast share. Both pages state this distinction where each keystone appears.
