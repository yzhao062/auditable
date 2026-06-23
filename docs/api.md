# API Reference

The stable public surface is the flat top-level API. The graph analyses (PRE and the
graph kernel) live under `auditable.graph.*`.

## Top-Level API

The capture, replay, and recovery flow, the standalone auditors, the POST
offline-analysis entry, and `render_report` (the dependency-free Markdown
renderer for both the PRE and POST reports; see the [Quickstart](quickstart.md#render-a-report-as-markdown)).

::: auditable

## Report Rendering

The Markdown renderer for the typed reports: `render_report` (the top-level
dispatcher), the per-pillar `pre_to_markdown` and `post_to_markdown`, and the
`to_markdown` method on each report. It formats the fields the report already
carries and is standard-library only.

::: auditable.report

## PRE: Declared-Plan Analysis

`analyze_plan`, the four reachability lints, the execution-topology keystone, the
preflight coverage report, and the `PreReport` it returns.

::: auditable.graph.pre

## Graph Adapters

The ingestion extension point and the shipped adapters, including the declared-plan
adapter the PRE pillar consumes.

::: auditable.graph.adapters

## Graph Kernel

The typed decision-graph construction and the structural queries the analyses read.

::: auditable.graph
