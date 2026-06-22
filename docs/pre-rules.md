# PRE Rules Reference

PRE runs at design time over a declared agent plan, before a single step executes. The entry point is `analyze_plan(plan, *, adapter=declared_plan_v1)`, imported from `auditable.graph.pre`. It is not a top-level `auditable` export, and it needs the `graph` extra. It returns a `PreReport` carrying four reachability lints, the execution-topology keystone, and a Preflight Coverage Report.

```python
from auditable.graph.pre import analyze_plan
from auditable.graph.adapters import declared_plan_v1

report = analyze_plan(plan, adapter=declared_plan_v1)
print(report)
```

A declared plan carries control flow and declared data reads and writes, but no observed values. So everything on this page is structural and read-only. Every lint is a pure NetworkX query over the projected declared graph: no mutation, no side effects, and no runtime or value execution. Two of the checks have a confirmation half (whether a value would actually flip, whether state actually drifted) that needs runtime values and is therefore out of scope at PRE; each such finding says so in its detail.

PRE applies only where a declared plan exists. A free-form agent with no declared plan falls outside this surface.

## The Declared-Plan Input

`analyze_plan` lowers the plan through the `declared_plan_v1` adapter into the typed session graph. The plan dict has the shape `{"nodes": [...]}`, and each node carries the fields the lints read:

| Field | Meaning |
|---|---|
| `idx` | A unique plain int. The step's identity. |
| `agent` | The owning agent name. |
| `kind` | `"decision"` or `"tool_call"`. |
| `control_preds` | The control-flow predecessors (the execution edges). Omitted means the linear default (the immediately prior node). |
| `reads` | The resources the node read. A read may be a bare resource id string, or a dict with `id` and optional `producer`, `volatile`, `pinned`, `revalidates`. |
| `writes` | The resources the node mutated (a consequential action). |
| `scope` | The granted tool-scope resource ids. |

A read whose `producer` names a prior in-plan node becomes a `declared`-graded `depends_on` edge carrying the resource id and the `volatile` / `pinned` / `revalidates` flags in its evidence. A read with no in-plan producer is a free read: its resource id is kept in the node attributes so the lints still see it, but it wires no dependency edge.

## The Four Shipping Lints

Each lint returns `LintFinding`s. A finding carries the lint name, the offending step `node_idx`, the `resource_id` it is about (or `None`), a one-line `detail`, and a `severity`. All PRE findings are `severity='warning'`: they are structural design warnings, not validated failure predictions. A warning marks a shape worth a second look at design time, and it does not assert that the run will fail.

### `write_with_no_prior_read`

Fires when a node writes a resource that no node in its backward slice (nor the writer itself) declares reading first. The backward slice is the set of steps the writer transitively rests on, computed as the descendants of the write node in the dependency DAG (`depends_on` points dependent to dependency, so the dependencies are the descendants). The primitive is `nx.descendants` over `dependency_dag(G)`, cross-referenced against the `reads` resource sets of the slice nodes. The finding names a write that lands without the read that would justify it.

### `flippable_dependency_annotations`

Annotates an unpinned, non-revalidated volatile dependency that feeds a decision. The primitive is `nx.descendants` over `dependency_dag(G)` from each decision node (its dependency set), intersected with the per-edge evidence flags on the declared `depends_on` edges. For each volatile edge on a decision's backward slice that is neither pinned nor revalidated, it records one annotation.

This is an annotation, not a value-flip proof. PRE flags that a volatile value flows into a decision unpinned; whether the value would actually flip is runtime work and out of scope here, and the finding's detail says so. The function name is plural (`flippable_dependency_annotations`); the label on each emitted finding reads `flippable_dependency_annotation` (singular).

### `scope_vs_snapshot`

Fires when a node's granted tool scope strictly exceeds the snapshot it read, so the node can act on state it never validated. The primitive is a set comparison: the declared `scope` versus the read-resource set pulled into the node's snapshot (the union of `reads` over the node and its backward slice). The lint fires only when `scope` is a strict superset of that read set, and it reports one finding per resource in the difference.

The strict-superset condition is deliberate. If the backward slice contributes reads that the scope does not also list, the scope is not a superset of the read set, and the lint does not fire. The check targets the case where every read is also in scope and the scope grants at least one resource beyond what was read.

### `missing_revalidation_barrier`

Fires when a volatile read reaches a consequential action with no intervening re-read on the control path between them. This is a two-projection query. First, `nx.descendants` over `dependency_dag(G)` locates a volatile read upstream of an action (the backward slice). Then `nx.descendants` over the `handoff_to` execution projection checks the control path from the read to the action for any barrier (a node that re-reads, that is, revalidates, the resource, recorded in its `barriers` set). When no such barrier sits between them, the lint fires.

As with the volatile annotation, drift confirmation is runtime work and out of scope at PRE; the finding's detail says so. PRE flags the missing barrier in the plan structure; it does not assert that the value drifted.

## The Execution-Topology Keystone

`analyze_plan` reports the structural chokepoint of the declared plan: the node that the most other nodes transitively follow in control flow. It is the argmax of `execution_reach` over the `handoff_to` projection, surfaced as `PreReport.keystone_idx` with `keystone_followers` (the count of transitive control-flow followers). Every step's follower count is also available in `execution_reach_by_idx`.

This is a structural design chokepoint, and it must be kept distinct from the POST keystone. The PRE keystone is computed over the control-flow projection and names where control concentrates in the plan. The POST keystone (see [POST Analysis](post-analysis.md)) is computed over the dependency DAG and names the step with the highest transitive blast share in a finished run. They are separate named concepts over different projections, and the PRE keystone does not predict failure.

## The Preflight Coverage Report

Alongside the lints and the keystone, `analyze_plan` attaches a Preflight Coverage Report. It is a descriptive, coverage-readiness view, and it is not a risk score. It tells the runtime and POST scorer what it will need before it can score, computed from three views:

- `PreflightCoverage` reports the grade mix (observed, declared, inferred counts), the `observed_fraction`, the saturation ratio `rho`, and the exact `no_score` reason the runtime scorer would apply. At PRE the `observed_fraction` is `0.0` on any non-empty declared layer, and `would_score` is always `False`, so the contract reads plainly as "the runtime scorer cannot score this declared layer yet."
- `ResourceTouchCompleteness` reports which declared reads, writes, and dependency edges carry a resource identity, and which do not. It also counts edges that carry an evidence resource id but no structured `ResourceRef`, since that `ResourceRef` is exactly what the planned runtime resource-touch contract fills.
- `BarrierInventory` lists, per resource, the declared revalidation barriers (the re-read nodes). It reports structure only: a resource that appears as a volatile read but is absent from the inventory has no declared barrier, which a consuming view can surface without claiming any drift occurred.

These three views are surfaced on `PreReport.preflight_coverage`, `.resource_touch_completeness`, and `.barrier_inventory`.

## The State-B Withhold Boundary

PRE withholds dependency-state (State B) blast-share risk on purpose. The declared dependency layer is declared-only (`observed_fraction=0`), so the structural scorer returns a `no_score` state on a declared plan. `PreReport.state_b_risk` is always `None`, `state_b_withheld` is always `True`, and `state_b_withheld_reason` carries the explanation. `analyze_plan` asserts this boundary: it expects a `no_score` verdict on a declared plan, and if a scored verdict ever came back, it raises rather than emit a number the declared evidence cannot support.

This is a deliberate honesty boundary. Dependency-state risk is the runtime and POST job, and PRE declines to guess it from declared-only structure.

## Planned: An OWASP-Agentic / CWE Rule Floor

A table-stakes rule floor of OWASP-Agentic and CWE checks for PRE is designed but not built. It is named in the harness module as a future direction: later versions add agent-audit-style OWASP-Agentic and CWE checks, consumed rather than reimplemented. No such rule floor ships today. `auditable` does not run OWASP or CWE checks now; the four lints above, the execution keystone, and the Preflight Coverage Report are the entire shipping PRE surface. Treat the rule floor as planned, for CI legibility, and nothing more.
