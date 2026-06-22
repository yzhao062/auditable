# auditable

Capture, replay, and audit AI agent decisions across the agent lifecycle.

`auditable` attaches at three points in an agent's life, and one typed two-layer decision graph carries all three:

- **PRE** (before deployment): lint a declared plan for dependency-state reliability issues, before a single step runs. See [PRE Rules](pre-rules.md).
- **REAL-TIME** (while the agent runs): capture each consequential decision with the dependency state it relied on, replay it under the state that is live now, and route a fix (allow, block, human-review, rollback). See [REAL-TIME Replay](realtime-replay.md).
- **POST** (after a run): rank a finished run by structural risk and name the keystone step that the most of the run rests on. See [POST Analysis](post-analysis.md).

The [Lifecycle](lifecycle.md) page is the map across the three pillars; [Architecture](architecture.md) is how the pieces fit. The detection and report generation run on one graph kernel, so the same construction serves every pillar.

## Install

```bash
pip install auditable            # core: capture, replay, recovery
pip install "auditable[graph]"   # adds the graph analyses (PRE lints, POST analyze_run)
```

The graph extra pulls in NetworkX, which the PRE and POST graph entries need.

## Where to Start

- [Quickstart](quickstart.md): the smallest runnable snippet for each pillar.
- [PRE Rules](pre-rules.md): the four declared-plan lints and the preflight coverage report.
- [API Reference](api.md): the full public surface.
