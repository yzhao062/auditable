# Architecture

This page describes the public, product-level architecture of `auditable`: the typed two-layer decision graph at the center of the library, the per-decision record that binds three spans, the ingestion adapters that feed the graph, and the two sinks that persist signed records. The same graph kernel is what the PRE, REAL-TIME, and POST pillars each run; see [Lifecycle](lifecycle.md) for how the three pillars use it.

## The Typed Two-Layer Decision Graph

An agent run is projected into one typed graph (a NetworkX `MultiDiGraph`). The graph has two edge layers, and the distinction between them is the load-bearing idea in `auditable`.

**Execution edges are observed from the trace.** These are the `emits` edges (an agent emits a step) and the `handoff_to` edges (one step hands control to the next). They record what actually happened in control flow, read directly off the run.

**Dependency edges are declared or inferred, never read off the trace.** These are the `depends_on` edges (a step relied on the state some earlier step produced or read). A `depends_on` edge points from the dependent step to the step it relied on. Because a trace does not say which earlier state a step actually consumed, this layer is either declared (a plan or a record states it) or inferred (a modeled prior-read assumption), and every edge records how it is known.

Each dependency edge carries a `grade` that records how it is known:

| Grade | Meaning |
|---|---|
| `observed` | A logged read/write match (the strongest grade). |
| `declared` | Stated by a plan or a record, but not observed in execution. |
| `inferred` | A full-history assumption (the weakest grade), used when nothing better is available. |

The default grade is `inferred`, so nothing is silently presented as observed. An edge also carries an `evidence` dict and an optional structured `resource` identity (a `ResourceRef` of `namespace`, `resource_id`, `key`). On the corpus adapters shipping today, `resource` is left `None` and the resource identity travels in `evidence`; filling the structured `ResourceRef` from a live read/write match is the planned runtime resource-touch contract (see [Roadmap](#roadmap)).

The graph kernel lives in `auditable.graph` and is queryable directly:

- `build_graph(steps)` builds the typed graph from a normalized trace.
- `dependency_dag(G)` projects the `depends_on` layer to a simple DiGraph.
- `downstream_reach(G, step_idx)` counts how many steps transitively depend on a step (its blast radius, used by the POST score).
- `execution_reach(G, step_idx)` counts how many steps transitively follow a step in control flow (the structural-chokepoint signal used by the PRE keystone).
- `characterize(G)` returns structural properties of one graph.

You do not build this graph by hand. `audit()` is the ergonomic capture entry for a single decision, and the ingestion adapters lower a whole run or plan into the typed `Step` list the graph reads. The graph is a first-class object you can query, and the capture path keeps it out of your way.

The library is torch-free. The graph layer depends only on NetworkX, installed through the `graph` extra. Any heavier graph stack is optional and lives outside this package.

## The Per-Decision Record: Three Spans, One Signed Unit

One agent decision spans three layers, and `auditable` records all three in a single `DecisionRecord` so the decision is judged as one unit rather than three disconnected logs.

| Span | What it captures | Standalone auditor | v0.1 signal |
|---|---|---|---|
| Data | What the agent read, plus the dependency snapshot it relied on | `DataAuditor` | Snapshot freshness rule (`flag` / `score` / `reason`) |
| Model | Which model produced the output, and its stated decision basis | `ModelAuditor` | Decision-basis trust flag |
| Harness | The action the agent executed, and its cost | `HarnessAuditor` | One static cost-cap rule |

Each span attaches a normalized `Report` (the leaf an `Auditor` returns), with a uniform shape across stages: `stage`, `name`, `score` in `[0, 1]`, `flag`, `reason`, and an `evidence` dict. The record is content-addressed with SHA-256 and chained to the previous record through a `prev_digest` field, so the log is tamper-evident; the record digest is taken over the bound spans, so it commits to all three leaves at once.

The three standalone auditors each subclass the `Auditor` base (the detect-face base class, analogous to a base detector in a detector library: set `stage` and `name`, implement `assess`, return a `Report`). Each is usable on its own with no agent and no chain. They are inputs to the record, and the composed full-chain record is the main line; see [Using a Single Layer](quickstart.md#using-a-single-layer) for the standalone path.

The `CompoundReport` is a transparent v0.1 bundle over the three leaves. It preserves the per-stage breakdown and exposes an explicitly named `uncalibrated_score` (the maximum of the per-stage scores) for debugging. This score is uncalibrated by design, it is not decision-grade, and it does not drive recovery; recovery is driven by the replay verdict (see [REAL-TIME Replay and Recovery](realtime-replay.md)). A calibrated cross-layer combiner is on the roadmap.

## Ingestion Adapters: One Representation, Many Sources

An adapter maps one source into the typed `Step` list the `SessionGraph` consumes, so a public corpus, `auditable`'s own signed records, and a declared plan all converge on one representation. The `Adapter` protocol is the public extension point (the analog of subclassing a base detector): user code conforms to `to_steps(source) -> List[Step]` plus a `name` and a `version`, and the protocol, rather than any one concrete class, is the contract third-party code implements. The protocol is `runtime_checkable`, so an object with these three members satisfies `isinstance(obj, Adapter)` without subclassing.

Three reference adapters ship today, each pinned by name and version so a later `v2` ships alongside, never in place of, a `v1`:

| Adapter | Pillar | Source it lowers |
|---|---|---|
| `tau_bench_prior_db_reads_v1` | POST | A public-corpus trajectory. Each consequential write depends on every prior read, graded `observed` but marked `modeled` in evidence (a conservative prior-read upper bound, not a causal label). |
| `own_record_v1` | POST | A chain of `auditable`'s own signed `DecisionRecord`s, with execution edges from the `prev_digest` backbone and sparse `declared` dependency edges. |
| `declared_plan_v1` | PRE | A framework-agnostic declared plan dict, with control flow lowered to execution edges and declared reads to `declared` dependency edges. |

The declared-plan adapter is the neutral seam a LangGraph, CrewAI, or AutoGen front-end would lower into. It is the target a real framework integration would write to, and it is explicitly not a parser for any of those frameworks. The plan dict has the shape `{"nodes": [...]}`, where each node carries `idx`, `agent`, `kind`, optional `control_preds`, and declared `reads` / `writes` / `scope`. See [PRE Rules](pre-rules.md) for the fields each lint reads.

## Sinks: Persisting the Signed Record

A sink signs each record and chains it to the previous one. Two concrete sinks ship today:

- `MemorySink` is the default in-process sink. It holds records in a list, signing and chaining each one as it is appended.
- `FileSink` is an append-only JSONL sink: one signed record per line, durable across process exit. It fails closed on a corrupt tail. A blank line or an empty file is fine, but a corrupt, non-object, or `record_id`-less line raises rather than silently restarting the chain, which would otherwise let a damaged tail fork an append-only tamper-evident log.

Pluggable sinks for external destinations (for example, OpenTelemetry or LangSmith) and exportable evidence bundles are on the roadmap.

## Roadmap

The shipping v0.1 architecture is the typed two-layer graph, the three-span signed record, the three reference adapters, and the two sinks. Planned additions, all labeled as roadmap and not shipping today:

- A fitted data-anomaly score (with a PyOD backend) on the data span, with the freshness rule as a fallback (v0.2).
- A calibrated cross-layer compound and model-as-first-class-node grounding beyond the current deterministic check (v0.3).
- Live / incremental scoring and the runtime resource-touch contract that fills the structured `ResourceRef` from a live read/write match (v0.3b).
- Pluggable sinks (OpenTelemetry, LangSmith), exportable evidence bundles, and a stable public API (v1.0), plus framework integrations (LangChain, LangGraph, CrewAI) and an MCP server.
