# PRE Examples

## Lint a Declared Plan

[`examples/example_pre_lint_plan.py`](https://github.com/yzhao062/auditable/blob/main/examples/example_pre_lint_plan.py) lints a declared three-step payment-approver plan before any step runs. The plan is a framework-agnostic dict: a tool writes `kyc.tier`, an approver reads the volatile `kyc.tier` with scope over `kyc.tier` and `ledger.balance`, and a ledger tool writes `ledger.entry`. The plan is built in the example file and shaped to trip every shipping lint.

```python
from auditable.graph.pre import analyze_plan
from auditable.graph.adapters import declared_plan_v1

report = analyze_plan(plan, adapter=declared_plan_v1)
print(report)
```

It prints a `PreReport`: the execution keystone (step 0, the structural chokepoint that the other two steps follow), five lint findings (two `write_with_no_prior_read`, one `flippable_dependency_annotation`, one `scope_vs_snapshot`, one `missing_revalidation_barrier`), the withheld State-B risk (`state_b_risk=None`, `state_b_withheld=True`), and a Preflight Coverage Report. Run it with `python examples/example_pre_lint_plan.py` (needs the `graph` extra).

See the [PRE Overview](pre-rules.md) for each lint and the keystone, and [PRE Coverage](pre-coverage.md) for the OWASP and CWE map.
