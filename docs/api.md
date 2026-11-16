# API Reference

The method library lives in `auditable.methods` (the registry and the method classes),
`auditable.base` (the family base classes and the method card), `auditable.featurize`
(trace normalization and features), `auditable.llm` (backends and the prediction
cache), `auditable.featurize.agentpile` (agentpile records as method inputs), and
`auditable.datasets` (synthetic traces and agentpile datasets). The record and graph modules keep
their flat top-level API, and the graph analyses (PRE and the graph kernel) live under
`auditable.graph.*`.

## Method Registry

`method_table(family=None, detailed=False)` renders the registry as the Markdown tables of the README, in the order of the audit: PRE (`config`), LIVE (`online`), then POST (`run`, `step`). The default columns are Type, Method, Algorithm, Year, and Reference. `detailed=True` lists Method, Family, Fit, Fidelity, Extras, and CatchBench row instead. The [Method Contract](methods-contract.md#the-method-card) says where each column comes from.

The leaderboards in `auditable.bench` print the same Type for each row. `Board.to_markdown()` has the columns Rank, Type, Method, Algorithm, and one per metric. A board whose shown rows audit at more than one stage adds a Stage column after Rank; the two LIVE boards do, because they score LIVE monitors and POST run detectors together. `Board.to_dict()` rows carry `type` and `algorithm`. The [Leaderboards](leaderboard.md#board-methods) page documents them.

::: auditable.methods

## Family Base Classes and the Method Card

::: auditable.base

## Features

::: auditable.featurize

## agentpile Records

::: auditable.featurize.agentpile

## LLM Backends and Cache

::: auditable.llm

## Synthetic Traces and Datasets

::: auditable.datasets

## Top-Level API

The capture, replay, and recovery flow, the standalone auditors, the POST
offline-analysis entry, and `render_report` (the dependency-free Markdown
renderer for both the PRE and POST reports; see the [Quickstart](quickstart.md#render-a-report-as-markdown)).
`ActionGate` and `ReferenceLedger` are deprecated as of 0.3 and will be removed in a
later release.

::: auditable

## Data Module

`DataAuditor` provides the freshness fallback and the optional fitted dependency-state
path. It is deprecated as of 0.3 and will be removed in a later release. See the
[Data Module](data-module.md) for its feature schema, history-fitting helpers,
fallback evidence, and detector-independent injection tools.

::: auditable.data.DataAuditor

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
