# LIVE Examples

## Capture, Replay, Recover a Payment

[`examples/payment_audit.py`](https://github.com/yzhao062/auditable/blob/main/examples/payment_audit.py) captures one $4,200 vendor-payment decision with the six-day-old budget snapshot it relied on, commits it through a reference ledger, then replays under a live budget dropped to $3,000 and executes a rollback. This is the full signed chain binding the data, model, and harness spans into one record.

```python
verdict = replay(decision.record, live_state={"budget": 3_000}, policy=policy)
gate.enforce_post_commit(verdict, receipt=receipt)
```

It prints the captured decision and model, the three signed leaf reports (data stale, model ok, harness ok), the uncalibrated compound debug score, the ledger balance after paying, the `ROLLBACK` verdict with its reason ($4,200 exceeds the live $3,000), and the restored balance after the gate reverses the committed payment. The closing line states that `auditable` re-decided on the live budget and reversed the payment: recovery, not just observability. Run it with `python examples/payment_audit.py`.

See the [LIVE Overview](realtime-replay.md) for the capture API, the four verdicts, the policy contract, and gate execution.
