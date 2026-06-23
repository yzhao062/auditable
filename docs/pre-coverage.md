# PRE Coverage: External Standards on the Graph

The four shipping PRE lints are read-only structural queries, and this page maps each to its public-standard analog. `auditable` does not ship a flat OWASP or CWE checklist. It associates each external threat that has a structural signature with a graph predicate, and states the boundary for the threats a static plan cannot show. The map below is documentation: the lints do not yet emit a standard identifier on each finding, and that tagging is on the roadmap.

This is the consolidation principle applied to security standards. The same declared-plan graph that carries the lifecycle and the data, model, harness spans also carries the standards coverage. One structure, queried three ways.

## The Coverage Map

| External item | Graph-structural predicate | Lint | Status |
|---|---|---|---|
| Excessive Agency (OWASP LLM06); Privilege Compromise (OWASP agentic); Confused Deputy (CWE-441) | Granted tool scope strictly exceeds the snapshot the node read | `scope_vs_snapshot` | Shipped |
| Time-of-check-to-time-of-use, re-cast to benign drift (CWE-367) | A volatile read reaches a consequential action with no intervening re-read | `missing_revalidation_barrier` | Shipped |
| Data and Model Poisoning (OWASP LLM04), re-cast to benign drift | An unpinned, non-revalidated volatile dependency feeds a decision | `flippable_dependency_annotation` | Shipped |
| Output and action grounding | A write whose target resource was never read in its backward slice | `write_with_no_prior_read` | Shipped |
| Overwhelming Human-in-the-Loop (OWASP agentic) | A consequential write whose control-predecessor chain holds no human-review node | `missing_human_gate` | Planned |
| Privilege Compromise (OWASP agentic); least privilege; Confused Deputy (CWE-441) | Tool scope grows along a `handoff_to` edge | `scope_escalation_across_handoff` | Planned |
| Unbounded Consumption (OWASP LLM10); Resource Overload (OWASP agentic) | Unbounded fan-out or a cycle in the execution topology | `unbounded_execution_topology` | Planned |
| Repudiation & Untraceability (OWASP agentic) | Every decision is a signed, hash-chained record | Architecture, not a lint | Shipped |

The four shipped lints are covered on the [PRE Overview](pre-rules.md). The three planned lints are graph-native control checks a flat rule list cannot express, because they need the control-flow and handoff topology the graph already carries; they are tracked on the [Roadmap](https://github.com/yzhao062/auditable#roadmap).

## The Boundary

PRE fires only on what the declared-plan graph makes visible: reads, writes, scope, control flow, and dependency topology. Threats with no static-structural signature stay out of PRE. Prompt injection (OWASP LLM01), content-level poisoning, hallucination and misinformation (OWASP LLM09), and deceptive behavior are runtime, content, or model-behavior threats. They route to LIVE or are marked out of scope. PRE never claims a structural answer to a non-structural threat.

The anchors are the OWASP LLM Top 10, the OWASP agentic threat taxonomy, and CWE. The identifiers above are tracked against the current published specifications.
