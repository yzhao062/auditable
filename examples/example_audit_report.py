"""The AuditReport: one ULB payment across three pillars, rendered for two consumers.

This example walks a SINGLE credit-card payment down the whole agent lifecycle and
aggregates the three pillars into one :class:`AuditReport`, then writes BOTH outputs,
one per consumer:

  PRE        lint the declared payment-approver plan before any step runs.
  LIVE  capture the approver's decision, replay it under a drifted-down live
             budget, and execute the rollback through the gate (recover, not flag).
  POST       rank the signed record the run just produced and ground its basis.

The point of the example is the consumer split. The same aggregated findings render
two ways, and the FORMAT serves the CONSUMER:

  to_markdown -> the AGENT-facing report. Lean, parseable, deterministic, no images.
                 An agent loop reads the verdict, the keystone, the findings, and the
                 recommended actions (as prose or from the fenced json block) and acts.
  to_pdf      -> the HUMAN-facing report. The same findings as narrative prose plus
                 the five embedded charts, a verdict banner, and a layout to share.

The Markdown needs no heavy dependency. The PDF needs the report extra:

  pip install "auditable[graph]"   # for analyze_plan / analyze_run (NetworkX)
  pip install "auditable[report]"  # for to_pdf (fpdf2 + matplotlib + networkx)

One dataset, one state. The payment amount is a real value sampled from the ULB
credit-card dataset (OpenML), cached under experiment/.cache, with a fixed fallback
so the example runs from a bare checkout. The one constructed dimension is temporal
budget drift: a six-day-old snapshot budget the approver read, versus a live budget
that has since dropped below the amount. No network and no API key.

POST runs on auditable's OWN records, so the keystone is honestly WITHHELD: the
own-record adapter declares its dependency edges instead of fabricating observed
reads, which lands a single-record chain in a no-score state. The report renders
that as "withheld", never as a zero. The agent Markdown's Coverage line carries the
exact no-score reason so an agent never reads a withheld score as zero risk.

Run:  python examples/example_audit_report.py
"""
import csv
import os
import tempfile
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
    ``floor`` so the payment reads as a consequential vendor payment. Falls back to a
    fixed amount when the cache is absent, so the example still runs from a bare
    checkout.
    """
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

    A KYC tool writes the tier; the approver reads that tier (a volatile read), is
    granted scope over the tier AND the ledger balance, and decides the spend; a
    posting tool writes the ledger entry. Framework-agnostic dict, shaped to trip
    every shipping PRE lint at once so the report has findings to show.
    """
    return {
        "plan_id": "payment-approver-v1",
        "framework": "declared",
        "nodes": [
            {"idx": 0, "agent": "kyc_tool", "kind": "tool_call", "writes": ["kyc.tier"]},
            {
                "idx": 1,
                "agent": "approver",
                "kind": "decision",
                "reads": [{"id": "kyc.tier", "producer": 0, "volatile": True}],
                "scope": ["kyc.tier", "ledger.balance"],
            },
            {
                "idx": 2,
                "agent": "ledger_tool",
                "kind": "tool_call",
                "reads": [{"id": "kyc.tier", "producer": 0}],
                "writes": ["ledger.entry"],
            },
        ],
    }


def budget_policy(state, action):
    """A payment is justified when the recipient is allow-listed and the amount fits
    the remaining budget, evaluated against whatever state is supplied."""
    if action.arguments["recipient"] not in state.get("allow_list", []):
        return False, f"Recipient {action.arguments['recipient']} not allow-listed."
    if action.cost > state.get("budget_remaining", 0):
        return False, (
            f"${action.cost:,.2f} exceeds the remaining budget "
            f"${state.get('budget_remaining', 0):,.2f}."
        )
    return True, "Within budget and allow-list."


def real_time(amount, snapshot_budget, live_budget, now):
    """Capture one payment decision, replay it live, and execute the rollback.

    The approver read a six-day-old budget snapshot under which the payment was in
    policy and committed the charge. Six days on, the live budget has dropped below
    the amount, so replay re-decides on the live state and the gate EXECUTES the
    rollback (a recovery, not just a flag). Returns the replay verdict and the signed
    records the run produced, which POST then analyzes.
    """
    snapshot = DependencySnapshot(
        state={"budget_remaining": snapshot_budget, "allow_list": ["acme-supplies"]},
        captured_at=now - SIX_DAYS,
    )
    action = Action("vendor_payment", {"recipient": "acme-supplies"}, cost=amount)
    ledger = ReferenceLedger(balance=snapshot_budget)
    gate = ActionGate(ledger)
    sink = MemorySink()  # signs each record, chains by prev_digest

    with audit("vendor_payment", snapshot=snapshot, sink=sink) as d:
        d.read(invoice="INV-4471", amount=amount, budget_remaining=snapshot_budget)
        d.model("gpt-x", decision_basis=(
            f"Invoice INV-4471 for ${amount:,.2f} is within the "
            f"${snapshot_budget:,.2f} remaining budget; recipient is allow-listed."))
        d.act(action)
        d.attach(DataAuditor(max_age_seconds=SIX_DAYS / 2).assess(snapshot, now=now))
    record = d.record

    receipt = gate.commit(action)  # the agent commits the payment
    committed_balance = ledger.balance

    live = {"budget_remaining": live_budget, "allow_list": ["acme-supplies"]}
    verdict = replay(record, live_state=live, policy=budget_policy)
    outcome = gate.enforce_post_commit(verdict, receipt=receipt)
    return verdict, outcome, ledger.balance, committed_balance, sink.records


def main():
    # A fresh temp directory so no build artifact lands in the repo tree.
    out_dir = tempfile.mkdtemp(prefix="auditable_report_")
    md_path = os.path.join(out_dir, "audit_report.md")
    pdf_path = os.path.join(out_dir, "audit_report.pdf")

    amount = one_real_amount()
    snapshot_budget = round(amount * 2, 2)   # in policy when the approver decided
    live_budget = round(amount * 0.5, 2)     # drifted below the amount, six days on
    now = time.time()

    print("=" * 72)
    print(f"auditable AuditReport: one ${amount:,.2f} ULB payment, three pillars, two consumers")
    print("=" * 72)

    # ----------------------------------------------------------------------- PRE
    pre = analyze_plan(payment_plan(), adapter=declared_plan_v1)
    print(f"\n[PRE]        {len(pre.findings)} lint finding(s); "
          f"structural keystone = step {pre.keystone_idx}")

    # ------------------------------------------------------------------ LIVE
    verdict, outcome, balance, committed_balance, records = real_time(
        amount, snapshot_budget, live_budget, now
    )
    print(f"[LIVE]  committed ${amount:,.2f} (balance ${committed_balance:,.2f}); "
          f"replay {verdict.action.value.upper()}; gate executed={outcome.executed}; "
          f"balance restored to ${balance:,.2f}")

    # ---------------------------------------------------------------------- POST
    # Rank the signed record the run above produced. own_record_v1 declares its
    # dependency edges (it does not fabricate observed reads), so the single-record
    # chain lands in a no-score state and the blast keystone is WITHHELD, by design.
    post = analyze_run(records, adapter=own_record_v1)
    ks = post.keystone
    if ks is not None:
        print(f"[POST]       state {post.state}; blast keystone = step {ks.idx} [{ks.label}]")
    else:
        print(f"[POST]       state {post.state}; blast keystone withheld (own records are "
              "coverage-honest)")

    # ------------------------------------------------------------- the aggregate
    report = AuditReport.from_run(
        pre=pre,
        post=post,
        verdicts=[verdict],
        records=records,
        title="Auditable Audit Report: ULB Payment Approver Run",
    )
    print(f"\nroll-up verdict: {report.headline_verdict}  "
          f"({len(report.findings)} finding(s) total)")

    # --------------------------------------------------- consumer 1: the AGENT
    # The agent-facing Markdown: lean, structured, parseable, no images. Written
    # whether or not the report extra is installed (it embeds no chart).
    markdown = report.to_markdown(md_path)
    print(f"\n[agent -> Markdown]  {md_path}")
    print(f"                     {len(markdown)} chars; stable anchors an agent splits on: "
          "## Verdict, ## Keystone, ## Findings, ## Recommended Actions, ## Coverage, ## JSON")

    # --------------------------------------------------- consumer 2: the HUMAN
    # The human-facing PDF: the same findings as prose plus the five charts. Needs
    # the report extra; the example degrades to a clear hint if it is absent.
    try:
        report.to_pdf(pdf_path)
        print(f"[human -> PDF]       {pdf_path}")
        print(f"                     {os.path.getsize(pdf_path)} bytes; verdict banner + "
              "5 charts embedded")
    except ImportError as exc:
        print(f"[human -> PDF]       skipped: {exc}")
        print('                     install the report extra: pip install "auditable[report]"')

    print("\nnext: open the PDF for the human review, or pipe the Markdown back to the agent:")
    print(f"      open  {pdf_path}")
    print(f"      cat   {md_path}   # feed to the agent loop")


if __name__ == "__main__":
    main()
