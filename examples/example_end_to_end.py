"""End-to-end: one payment, walked through the whole agent lifecycle.

auditable attaches at three points in an agent's lifecycle. The other examples
demo one pillar each (example_pre_lint_plan.py = PRE, example_payment_audit.py =
LIVE, example_post_rank_run.py = POST). This one hands a SINGLE payment down all
three, so the pillars share one narrative and one dataset:

  PRE        lint the declared payment-approver plan before any step runs.
  LIVE  capture the approver's decision, replay it under a drifted live
             budget, and roll the payment back through the gate.
  POST       rank the signed record the run just produced, and ground its basis.

One dataset, one state. The payment amount is a real value sampled from the ULB
credit-card dataset (OpenML), cached under experiment/.cache. The one constructed
dimension is temporal budget drift: a six-day-old snapshot budget the approver
read, versus a live budget that has since dropped below the amount.

Honesty holds at the seams. PRE withholds the dependency-state (State B) risk on
a declared-only plan. POST on auditable's OWN records reports low coverage rather
than a calibrated keystone, because the own-record adapter declares its
dependency edges instead of fabricating observed reads (the runtime resource-touch
contract that would upgrade them is planned, not shipping). Grounding still lights
up on the own record, because it carries a stated decision basis. The corpus path
(example_post_rank_run.py) is where a SCORED POST keystone shows today.

Needs the graph extra:  pip install "auditable[graph]"
Run:  python examples/example_end_to_end.py
"""
import csv
import os
import time

from auditable import (
    Action,
    ActionGate,
    DataAuditor,
    DependencySnapshot,
    MemorySink,
    ReferenceLedger,
    analyze_run,
    audit,
    replay,
)
from auditable.audit_report import AuditReport
from auditable.graph.adapters import declared_plan_v1, own_record_v1
from auditable.graph.pre import analyze_plan

SIX_DAYS = 6 * 24 * 60 * 60
_CACHE = os.path.join(
    os.path.dirname(__file__), os.pardir, "experiment", ".cache",
    "creditcard_amounts_10k.csv",
)


def one_real_amount(floor=1000.0):
    """One real, consequential transaction amount from the cached ULB sample.

    Amount distribution only, not fraud labels. Takes the first real amount above
    ``floor`` so the payment reads as a consequential vendor payment (the sample
    runs from cents to tens of thousands). Falls back to a fixed amount if the
    cache is absent, so the example still runs from a bare checkout."""
    try:
        with open(_CACHE) as f:
            for row in csv.DictReader(f):
                amt = float(row["Amount"])
                if amt >= floor:
                    return round(amt, 2)
    except FileNotFoundError:
        pass
    return 4200.0


def payment_plan():
    """The declared 3-step payment-approver plan (the PRE input).

    A KYC tool writes the tier; the approver reads that tier (a volatile read),
    is granted scope over the tier AND the ledger balance, and decides the spend;
    a posting tool writes the ledger entry. Framework-agnostic dict; PRE lints it
    before a single step runs. Shaped to trip every shipping PRE lint at once."""
    return {
        "plan_id": "payment-approver-v1",
        "framework": "declared",
        "nodes": [
            {"idx": 0, "agent": "kyc_tool", "kind": "tool_call",
             "writes": ["kyc.tier"]},  # write with no prior read
            {"idx": 1, "agent": "approver", "kind": "decision",
             "reads": [{"id": "kyc.tier", "producer": 0, "volatile": True}],
             "scope": ["kyc.tier", "ledger.balance"]},  # scope exceeds snapshot
            {"idx": 2, "agent": "ledger_tool", "kind": "tool_call",
             "reads": [{"id": "kyc.tier", "producer": 0}],
             "writes": ["ledger.entry"]},
        ],
    }


def budget_policy(state, action):
    """A payment is justified if the recipient is allow-listed and the amount fits
    the remaining budget, evaluated against whatever state is supplied."""
    if action.arguments["recipient"] not in state.get("allow_list", []):
        return False, f"Recipient {action.arguments['recipient']} not allow-listed."
    if action.cost > state.get("budget_remaining", 0):
        return False, (
            f"${action.cost:,.2f} exceeds the remaining budget "
            f"${state.get('budget_remaining', 0):,.2f}."
        )
    return True, "Within budget and allow-list."


def main():
    # One real ULB amount; the two budgets are the constructed temporal-drift
    # dimension (same stale-failure setup as experiment/replay_vs_baseline.py):
    # the snapshot budget covers the payment, the live budget has drifted below it.
    amount = one_real_amount()
    snapshot_budget = round(amount * 2, 2)   # in policy when the agent decided
    live_budget = round(amount * 0.5, 2)     # drifted below the amount, six days on
    now = time.time()
    print("=" * 72)
    print(f"auditable end-to-end: one ${amount:,.2f} vendor payment, three pillars")
    print("=" * 72)

    # ----------------------------------------------------------------------- PRE
    # Lint the declared approver plan before deploy. One call; the adapter lowers
    # the plan dict to the typed session graph and four read-only lints run.
    print("\n[1/3] PRE  -- lint the declared plan before any step runs\n")
    pre = analyze_plan(payment_plan(), adapter=declared_plan_v1)
    # Five lint findings; State B (dependency-state risk) is honestly WITHHELD.
    for f in pre.findings:
        rid = f"'{f.resource_id}'" if f.resource_id else "-"
        print(f"      lint: {f.lint} @ step {f.node_idx} ({rid})")
    print(f"      State B blast-share risk: {pre.state_b_risk} "
          f"(withheld={pre.state_b_withheld}); scoring it is the runtime / POST job")

    # ----------------------------------------------------------- LIVE (capture)
    # The approver read a six-day-old budget snapshot under which the payment was
    # in policy. One amount, one budget, one decision. The basis names the amount
    # and the budget, so POST grounding can later check it against what was read.
    print("\n[2/3] LIVE  -- capture the decision, replay live, recover\n")
    snapshot = DependencySnapshot(
        state={"budget_remaining": snapshot_budget, "allow_list": ["acme-supplies"]},
        captured_at=now - SIX_DAYS,
    )
    action = Action("vendor_payment", {"recipient": "acme-supplies"}, cost=amount)
    ledger = ReferenceLedger(balance=snapshot_budget)  # reference / demo rail
    gate = ActionGate(ledger)
    sink = MemorySink()                       # signs each record, chains by prev_digest

    with audit("vendor_payment", snapshot=snapshot, sink=sink) as d:
        d.read(invoice="INV-4471", amount=amount, budget_remaining=snapshot_budget)
        d.model("gpt-x", decision_basis=(
            f"Invoice INV-4471 for ${amount:,.2f} is within the "
            f"${snapshot_budget:,.2f} remaining budget; recipient is allow-listed."))
        d.act(action)
        # One signed data leaf: snapshot freshness, the signal that drives recovery.
        d.attach(DataAuditor(max_age_seconds=SIX_DAYS / 2).assess(snapshot, now=now))
    record = d.record

    receipt = gate.commit(action)             # the agent commits the payment
    print(f"      committed ${amount:,.2f}; ledger balance now "
          f"${ledger.balance:,.2f} (data leaf: {record.data.report.flag})")

    # Six days on, the live budget has dropped below the amount. Replay re-decides
    # on the live state and routes a fix; the gate EXECUTES the rollback.
    live = {"budget_remaining": live_budget, "allow_list": ["acme-supplies"]}
    verdict = replay(record, live_state=live, policy=budget_policy)
    outcome = gate.enforce_post_commit(verdict, receipt=receipt)
    print(f"      replay verdict: {verdict.action.value.upper()} -- {verdict.reason}")
    print(f"      gate executed: {outcome.executed}; balance restored to "
          f"${ledger.balance:,.2f} (recovery, not just a flag)")

    # ---------------------------------------------------------------------- POST
    # Rank the signed record the run just produced. own_record_v1 declares its
    # dependency edges (it does not fabricate observed reads), so a single-record
    # chain lands in a no-score state and the keystone is withheld, by design.
    # Grounding still lights up, because the record carries a stated basis.
    print("\n[3/3] POST  -- rank the signed record this run produced\n")
    post = analyze_run(sink.records, adapter=own_record_v1)
    if post.keystone is not None:
        print(f"      keystone: step {post.keystone.idx} [{post.keystone.label}]")
    else:
        print(f"      keystone: withheld (state={post.state}); own records are "
              "coverage-honest until the runtime touch contract fills observed edges")
    for idx, g in sorted(post.grounding.items()):
        print(f"      grounding @ step {idx}: {g.score} supported ({g.state})")

    # ------------------------------------------------------------ the full report
    # The aggregate over all three pillars, rendered as the agent-facing Markdown
    # (dependency-free). This is the single report the README flagship shows.
    report = AuditReport.from_run(
        pre=pre,
        post=post,
        verdicts=[verdict],
        records=sink.records,
        title="Auditable Audit Report: ULB Payment Approver Run",
    )
    print("\n" + "=" * 72)
    print("Full audit report (Markdown via AuditReport.to_markdown):")
    print("=" * 72)
    print(report.to_markdown())


if __name__ == "__main__":
    main()
