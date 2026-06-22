# API Reference

The stable public surface is the flat top-level API. The graph analyses (PRE and the
graph kernel) live under `auditable.graph.*`.

## Top-Level API

The capture, replay, and recovery flow, the standalone auditors, and the POST
offline-analysis entry.

::: auditable

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
