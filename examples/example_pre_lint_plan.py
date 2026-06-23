"""PRE pillar demo: lint a DECLARED agent plan before any step runs.

What this shows, in plain terms: given a plan an agent intends to run, auditable
points at the design-time risks (a write nothing read first, a volatile read
feeding a decision, scope wider than the snapshot) before any step executes. It
is a read-only, design-time capability demo and carries NO benchmark percentage.
GRADE relation: PRE operates on the SAME typed two-layer decision graph that
GRADE (arXiv:2606.22741) measures, but the GRADE ROC-AUC / localization numbers
are scored on FINISHED runs and surface at POST (example_post_rank_run.py); a
declared-only plan has no observed dependency reads to score, so none of those
numbers attach here. PRE's job is structural lints and a coverage-readiness view,
not a score.

auditable attaches at three points in an agent's lifecycle. This is the PRE
pillar: design-time, read-only lints over a DECLARED plan, before a single step
executes. The other two pillars are example_live_replay.py (LIVE: replay under
live state and execute a fix) and example_post_rank_run.py (POST: rank a finished
run). All three run over the same typed two-layer decision graph.

The plan below is a small payment approver, written as a framework-agnostic
declared plan dict (the neutral target a LangGraph, CrewAI, or AutoGen front-end
would lower into; the adapter is not a parser for any of those). It is shaped to
trip every shipping PRE lint at once:

  step 0  fetch_kyc_tier   writes kyc.tier, reads nothing
  step 1  approve_payment  reads kyc.tier (volatile), granted scope over
                           kyc.tier AND ledger.balance, makes the decision
  step 2  post_payment     reads kyc.tier, writes ledger.entry

From that one plan, analyze_plan reports four things and withholds a fifth:

  1. the execution-topology keystone, the structural chokepoint the most other
     steps transitively follow in control flow (step 0 here). This is a
     STRUCTURAL design lint, not a failure predictor, and it is a different
     concept from the POST blast-radius keystone in example_post_rank_run.py.
  2. the four reachability lints, all pure read-only NetworkX queries over the
     declared graph at severity 'warning':
       - write_with_no_prior_read   (steps 0 and 2 write a resource their
                                      backward slice never read first)
       - flippable_dependency_annotation
                                     (step 1 reads volatile kyc.tier unpinned
                                      and un-revalidated feeding the decision;
                                      an annotation, NOT a value-flip proof)
       - scope_vs_snapshot          (step 1 is granted ledger.balance but never
                                      read it into its snapshot)
       - missing_revalidation_barrier
                                     (the volatile kyc.tier read reaches the
                                      decision with no intervening re-read;
                                      drift confirmation is runtime work)
  3. the Preflight Coverage Report, a descriptive coverage-readiness view (grade
     mix, the exact reason the runtime scorer will withhold a score, which
     declared touches still lack a resource identity, and the declared
     revalidation barriers per resource). It is NOT a risk score.

  Withheld: dependency-state (State B) blast-share risk. The declared dependency
  layer is declared-only (observed_fraction is 0), so analyze_plan returns
  state_b_risk=None with state_b_withheld=True. Putting a number there is the
  runtime and POST job, not PRE's; this is a deliberate honesty boundary.

A note on scope. The OWASP-Agentic / CWE table-stakes rule floor for PRE is
planned, not shipping; nothing in this example runs an OWASP or CWE check today.

Needs the graph extra:  pip install "auditable[graph]"
Run:  python examples/example_pre_lint_plan.py
"""
from auditable.graph.adapters import declared_plan_v1
from auditable.graph.pre import analyze_plan


def payment_plan():
    """A declared payment-approver plan that trips every shipping PRE lint.

    Three steps. A KYC tool writes the tier, an approver reads that tier (a
    volatile read) and decides the payment under a scope wider than what it read,
    and a posting tool writes a ledger entry. Resource ids and the volatile flag
    are declared on the plan; no value is observed, because PRE runs before the
    plan executes.
    """
    return {
        "plan_id": "payment-approver-v1",
        "framework": "declared",  # neutral plan dict, not parsed from any framework
        "nodes": [
            {
                "idx": 0,
                "agent": "kyc_tool",
                "kind": "tool_call",
                # writes kyc.tier with no prior read of it -> write_with_no_prior_read
                "writes": ["kyc.tier"],
            },
            {
                "idx": 1,
                "agent": "approver",
                "kind": "decision",
                # reads the tier as a VOLATILE dependency (unpinned, not revalidated):
                #   - flippable_dependency_annotation (volatile feeds a decision)
                #   - missing_revalidation_barrier   (no re-read before the decision)
                "reads": [{"id": "kyc.tier", "producer": 0, "volatile": True}],
                # granted scope strictly exceeds what it read into its snapshot:
                #   {kyc.tier, ledger.balance} > {kyc.tier} -> scope_vs_snapshot on
                #   ledger.balance (it can act on state it never validated)
                "scope": ["kyc.tier", "ledger.balance"],
            },
            {
                "idx": 2,
                "agent": "ledger_tool",
                "kind": "tool_call",
                "reads": [{"id": "kyc.tier", "producer": 0}],
                # writes a ledger entry its backward slice never read -> a second
                # write_with_no_prior_read
                "writes": ["ledger.entry"],
            },
        ],
    }


def main():
    plan = payment_plan()

    # One public call: declared plan -> typed session graph -> PRE-honest report.
    # analyze_plan is reached under auditable.graph.pre (it is intentionally NOT a
    # top-level auditable export); the adapter is declared_plan_v1.
    report = analyze_plan(plan, adapter=declared_plan_v1)
    print(report)

    # The execution keystone, read off the report. It is the structural chokepoint
    # of the declared plan (argmax of execution_reach over the control-flow
    # projection), distinct from the POST blast-radius keystone in example_post_rank_run.py.
    if report.keystone_idx is not None:
        print(
            f"\nExecution keystone: step {report.keystone_idx} "
            f"(structural chokepoint; {report.keystone_followers} of "
            f"{max(report.n_steps - 1, 0)} other steps follow it in control flow). "
            "This is a design lint, not a failure prediction."
        )

    # The withheld State-B number, stated plainly so a withheld score never reads
    # as zero risk.
    print(
        f"\nState B (dependency-state) blast-share risk: "
        f"{report.state_b_risk} (withheld={report.state_b_withheld}). "
        "PRE withholds it on a declared-only layer; scoring it is the runtime / POST job."
    )


if __name__ == "__main__":
    main()
