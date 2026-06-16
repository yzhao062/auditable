<div align="center">

# auditable

**Capture, replay, and audit the decisions your AI agents make.**

*AI Risk Audit and Control across Agents, Foundation Models, and Data.*

[![PyPI](https://img.shields.io/pypi/v/auditable.svg)](https://pypi.org/project/auditable/)
[![Python](https://img.shields.io/pypi/pyversions/auditable.svg)](https://pypi.org/project/auditable/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yzhao062/auditable.svg?style=social)](https://github.com/yzhao062/auditable)

</div>

`auditable` is an open-source SDK that makes an AI agent's decisions reconstructable after the fact. For every consequential action your agent takes, it captures a signed record of what the agent read, which model decided, and what the agent did, then lets you **replay** that decision under the state that was actually live and route a fix: allow, block, hand off, or roll back.

> By [Yue Zhao](https://github.com/yzhao062), creator of [PyOD](https://github.com/yzhao062/pyod) (42M+ downloads). Built on the Auditable Agents framework ([arXiv:2604.05485](https://arxiv.org/abs/2604.05485)).

## The Problem

Most agent tools log *what happened*. They do not record *what the agent relied on when it decided*, so when a payment, an approval, or a tool call later looks wrong, the budget, the policy, and the allow-list that were live at that moment are already gone. The action was reasoned; the dependency it trusted had drifted. By the time anyone asks, the decision can no longer be reconstructed.

## Install

```bash
pip install auditable
```

## 60-Second Quickstart

```python
from auditable import Action, DependencySnapshot, audit, replay

def policy(state, action):
    ok = action.cost <= state.get("budget_remaining", 0)
    return ok, ("within budget" if ok else f"${action.cost:,.0f} over budget")

# Capture the decision with the dependency snapshot it relied on.
snapshot = DependencySnapshot(state={"budget_remaining": 10000})
with audit("vendor_payment", snapshot=snapshot) as d:
    d.model("gpt-x", decision_basis="invoice matches approved PO")
    d.act(Action("vendor_payment", {"recipient": "acme"}, cost=4200))

# Later, replay it against the live state. The budget has since dropped.
verdict = replay(d.record, live_state={"budget_remaining": 3000}, policy=policy)
print(verdict.action, verdict.reason)
# FixAction.ROLLBACK  Decision relied on stale dependency state: $4,200 over budget
```

See [`examples/payment_audit.py`](examples/payment_audit.py) for the full flagship demo.

## How It Works

One agent decision crosses three layers, and `auditable` records all three in a single signed, hash-chained record:

| Layer | What It Captures |
|---|---|
| **Data** | What the agent read: inputs, retrieved context, and the dependency snapshot (budget, policy, allow-list, config versions) that was live |
| **Model** | Which model produced the output, and the stated decision basis |
| **Harness** | The action the agent executed, and its cost |

`replay()` re-derives whether the action is still justified under the live dependency state versus the snapshot the agent actually used. A decision that passed on a stale snapshot but fails on live state is exactly the failure `auditable` exists to catch.

## The Full Chain: AI Risk Audit and Control across Agents, Foundation Models, and Data

A single agent decision crosses the whole stack, so one `auditable` record already spans all three layers, and the library is built to connect a signal source at each:

| Layer | What the record captures | Signal source it connects |
|---|---|---|
| **Agent (harness)** | The action executed, its cost, and the replayable verdict | Native (shipping now) |
| **Foundation model** | Which model produced the output, and the stated basis | TrustLLM trust and behavior signals (roadmap) |
| **Data** | What the agent read, and the dependency snapshot it relied on | [PyOD](https://github.com/yzhao062/pyod) anomaly scores on inputs and retrieved context (roadmap) |

The agent decision is the spine. `auditable` starts there, captures the full chain in one signed record, and brings the data-layer and model-layer signals in as plug-in inputs. That is the main line, delivered through one replayable decision record that threads all three layers.

**v0 scope (honest):** the decision record spans all three layers, and the agent-decision capture and replay ship today. The PyOD and TrustLLM signal integrations that fill the data and model spans are on the [roadmap](#roadmap); `auditable` does not yet score data anomalies or model trust itself.

## Roadmap

The path to the full chain, one signal source at a time:

- [ ] **Agent layer**: LangChain callback integration (`auditable.integrations.langchain`)
- [ ] **Data layer**: anomaly-triggered surfacing via [PyOD](https://github.com/yzhao062/pyod), flag which decisions to replay from anomalies in what the agent read
- [ ] **Model layer**: TrustLLM trust and behavior signals attached to the model span
- [ ] Pluggable sinks (file, OpenTelemetry, LangSmith, Datadog)
- [ ] Signed, exportable evidence bundles

## Citation

If you use `auditable` in research, please cite the Auditable Agents framework:

```bibtex
@inproceedings{auditable-agents-2026,
  title  = {Auditable Agents},
  author = {Nian, Yi and Yuan, Aojie and Zhao, Yue},
  year   = {2026},
  note   = {arXiv:2604.05485, ACL 2026 KnowFM Workshop}
}
```

## License

[Apache-2.0](LICENSE).
