# Audit Report

An audit report turns a finished analysis into a deliverable. `auditable` writes two formats from one report, because a report has two consumers.

- **Markdown, for an agent.** Lean, structured, and parseable, with no embedded images. An agent reads it inside a loop and acts on the findings, the keystone, and the recommended actions. It works on the core install, with no heavy dependency.
- **PDF, for a person.** The polished, visual artifact, with the charts and the layout a reviewer or a stakeholder reads.

```python
from auditable.audit_report import AuditReport

# pre, post, and verdicts come from analyze_plan, analyze_run, and replay over one run.
report = AuditReport.from_run(pre=pre, post=post, verdicts=verdicts)
report.to_markdown("audit.md")   # agent-facing: lean, structured, parseable
report.to_pdf("audit.pdf")       # human-facing: charts and layout
```

Every pillar is optional, so `AuditReport.from_pre(pre)` or `AuditReport.from_analysis(post)` renders a single-pillar report. The agent-facing Markdown carries stable section anchors (`## Verdict`, `## Keystone`, `## Findings`, `## Recommended Actions`, `## Coverage`) plus a fenced JSON block, so a loop can split and act on it. See [`examples/example_audit_report.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_audit_report.py) for the full run.

The PDF visuals (the blast-share ranking with the keystone highlighted, the preflight coverage, the two-layer graph, and the budget drift) ship in the optional report extra. The Markdown path has no heavy dependency.

```bash
pip install "auditable[report]"
```

The report aggregates the same three pillars over one graph: the PRE lints and preflight coverage, the LIVE replay verdicts and recoveries, and the POST keystone and blast-share ranking. One run, one report, two readers.
