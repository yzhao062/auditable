"""Tests for the deterministic grounding helper (auditable.graph.grounding).

The helper scores whether a stated ``decision_basis`` is supported by the context
the decision actually read. The context contract is pinned (Round 3 review):
context is drawn from a decision's ``data.inputs``, ``data.retrieved``, and
``data.snapshot.state``. These tests pin three things so later own-record / live
callers inherit a fixed contract:

- the flattening rule (dict keys + values, retrieved items, and state values all
  become comparable tokens / numbers / entities);
- the scoring direction (a grounded basis scores high, an ungrounded basis low);
- the empty / no-context behavior (a clearly marked "insufficient" result with no
  numeric score, never a false high).

The score is deterministic consistency evidence, not a calibrated probability.
"""
from auditable.graph.grounding import (
    STATE_INSUFFICIENT_BASIS,
    STATE_INSUFFICIENT_CONTEXT,
    STATE_SCORED,
    GroundingResult,
    ground_basis,
    ground_record,
)

# A small, realistic decision context reused across cases: an approved invoice
# paid within a budget. Anchors: the invoice id INV-100, status "approved", and
# the number 5000.
_INPUTS = {"invoice_id": "INV-100", "status": "approved"}
_STATE = {"budget_remaining": 5000, "policy_id": "kyc-2026-03"}
_RETRIEVED = ["purchase order PO-9 was approved by finance"]


def test_grounded_basis_scores_high():
    # every checkable anchor (entity INV-100, number 5000, words invoice/approved)
    # appears in the read context, so support is high
    r = ground_basis(
        "invoice INV-100 is approved and within the 5000 budget",
        inputs=_INPUTS,
        retrieved=_RETRIEVED,
        state=_STATE,
    )
    assert isinstance(r, GroundingResult)
    assert r.state == STATE_SCORED
    assert r.score is not None and r.score >= 0.6
    assert "5000" in r.matched
    assert "inv-100" in r.matched


def test_ungrounded_basis_scores_low():
    # the basis cites a different order, a different number, and unrelated words;
    # none appear in the same context, so support is near zero
    r = ground_basis(
        "refund order ORD-999 for 250 dollars overdue",
        inputs=_INPUTS,
        retrieved=_RETRIEVED,
        state=_STATE,
    )
    assert r.state == STATE_SCORED
    assert r.score is not None and r.score <= 0.2
    # the unsupported anchors are surfaced for the auditor
    assert "ord-999" in r.unmatched
    assert "999" in r.unmatched


def test_grounded_scores_strictly_above_ungrounded():
    ctx = dict(inputs=_INPUTS, retrieved=_RETRIEVED, state=_STATE)
    grounded = ground_basis("invoice INV-100 approved for 5000", **ctx)
    ungrounded = ground_basis("refund ORD-999 for 250", **ctx)
    assert grounded.score > ungrounded.score


def test_empty_context_returns_insufficient():
    # the named no-context case: a basis but nothing read. Marked insufficient with
    # no numeric score, never a false high.
    r = ground_basis("invoice INV-100 approved for 5000", inputs={}, retrieved=[], state={})
    assert r.state == STATE_INSUFFICIENT_CONTEXT
    assert r.score is None


def test_no_context_args_returns_insufficient():
    # all context sources omitted (None) behaves like empty context
    r = ground_basis("invoice INV-100 approved for 5000")
    assert r.state == STATE_INSUFFICIENT_CONTEXT
    assert r.score is None


def test_empty_basis_returns_insufficient():
    # the symmetric case: nothing to check. A contentless basis must not score high.
    for basis in ("", "   ", "the and of to"):  # blank, whitespace, stopwords only
        r = ground_basis(basis, inputs=_INPUTS, state=_STATE)
        assert r.state == STATE_INSUFFICIENT_BASIS
        assert r.score is None


def test_insufficient_is_never_a_false_high():
    # explicit guard: no insufficient state ever returns a high numeric score
    for r in (
        ground_basis("invoice INV-100", inputs={}, retrieved=[], state={}),
        ground_basis("", inputs=_INPUTS),
    ):
        assert r.score is None  # withheld, not a number that could read as "grounded"


def test_context_from_inputs_only():
    # an anchor present only in data.inputs still matches (contract: inputs count)
    r = ground_basis("invoice INV-100 was read", inputs={"invoice_id": "INV-100"})
    assert r.state == STATE_SCORED
    assert "inv-100" in r.matched


def test_context_from_retrieved_only():
    # an anchor present only in data.retrieved still matches (contract: retrieved counts)
    r = ground_basis(
        "purchase order PO-9 approved",
        retrieved=["finance note: PO-9 approved last week"],
    )
    assert r.state == STATE_SCORED
    assert "po-9" in r.matched


def test_context_from_snapshot_state_only():
    # an anchor present only in data.snapshot.state still matches (contract: state counts)
    r = ground_basis("budget_remaining is 5000", state={"budget_remaining": 5000})
    assert r.state == STATE_SCORED
    assert "5000" in r.matched


def test_number_normalization_matches_int_and_formatted():
    # a formatted currency amount in the basis matches a plain int in the state, and
    # a non-matching amount does not
    r = ground_basis("approved spend of $5,000", state={"budget_remaining": 5000})
    assert "5000" in r.matched
    miss = ground_basis("approved spend of $4,999", state={"budget_remaining": 5000})
    assert "4999" in miss.unmatched


def test_entity_exact_match_versus_mismatch():
    # an identifier matches exactly or not at all; a near-miss id is unmatched
    hit = ground_basis("relied on policy kyc-2026-03", state={"policy_id": "kyc-2026-03"})
    assert "kyc-2026-03" in hit.matched
    miss = ground_basis("relied on policy kyc-2026-04", state={"policy_id": "kyc-2026-03"})
    assert "kyc-2026-04" in miss.unmatched


def test_retrieved_dict_items_are_flattened():
    # retrieved items may be structured dicts, not just strings; keys and values
    # both become comparable tokens
    r = ground_basis(
        "vendor Acme shipped order 42",
        retrieved=[{"vendor": "Acme", "order_no": 42}],
    )
    assert r.state == STATE_SCORED
    assert "acme" in r.matched and "42" in r.matched


def test_evidence_breakdown_is_transparent():
    r = ground_basis(
        "invoice INV-100 approved for 5000",
        inputs=_INPUTS,
        state=_STATE,
    )
    ev = r.evidence
    assert "method" in ev and "calibrated" in ev["method"]  # method states it is NOT calibrated
    assert ev["n_context_tokens"] > 0
    # per-kind matched/total breakdown for word / number / entity overlap
    assert set(ev["by_kind"]) == {"word", "number", "entity"}
    assert ev["by_kind"]["number"]["matched"] == ev["by_kind"]["number"]["total"]  # 5000 matched


def test_matched_and_unmatched_partition_basis_anchors():
    r = ground_basis("invoice INV-100 approved for 9999", inputs=_INPUTS, state=_STATE)
    # 9999 is not in context; INV-100 / invoice / approved are
    assert "9999" in r.unmatched
    assert "inv-100" in r.matched
    # matched and unmatched are disjoint
    assert not (set(r.matched) & set(r.unmatched))


def test_deterministic_repeat_calls_identical():
    args = dict(inputs=_INPUTS, retrieved=_RETRIEVED, state=_STATE)
    a = ground_basis("invoice INV-100 approved for 5000", **args)
    b = ground_basis("invoice INV-100 approved for 5000", **args)
    assert a.state == b.state and a.score == b.score
    assert a.matched == b.matched and a.unmatched == b.unmatched


# ---- ground_record: the "runnable from the record" convenience ------------------


class _Snap:
    def __init__(self, state):
        self.state = state


class _Data:
    def __init__(self, inputs, retrieved, state):
        self.inputs = inputs
        self.retrieved = retrieved
        self.snapshot = _Snap(state)


class _Model:
    def __init__(self, decision_basis):
        self.decision_basis = decision_basis


class _Record:
    """A minimal DecisionRecord-shaped stub (duck-typed, no heavy import)."""

    def __init__(self, basis, inputs, retrieved, state):
        self.model = _Model(basis)
        self.data = _Data(inputs, retrieved, state)


def test_ground_record_pulls_basis_and_three_context_sources():
    rec = _Record(
        basis="invoice INV-100 approved for 5000",
        inputs=_INPUTS,
        retrieved=_RETRIEVED,
        state=_STATE,
    )
    r = ground_record(rec)
    assert r.state == STATE_SCORED
    assert r.score is not None and r.score >= 0.6
    assert "inv-100" in r.matched and "5000" in r.matched


def test_ground_record_on_real_decision_record():
    # exercise the actual DecisionRecord type end to end (offline, no network)
    from auditable.record import DataSpan, DecisionRecord, DependencySnapshot, ModelSpan

    rec = DecisionRecord(action_type="vendor_payment")
    rec.model = ModelSpan(model_id="gpt-x", decision_basis="invoice INV-100 approved for 5000")
    rec.data = DataSpan(
        inputs=_INPUTS,
        retrieved=_RETRIEVED,
        snapshot=DependencySnapshot(state=_STATE),
    )
    r = ground_record(rec)
    assert r.state == STATE_SCORED
    assert r.score is not None and r.score >= 0.6


def test_ground_record_empty_context_is_insufficient():
    rec = _Record(basis="invoice INV-100 approved", inputs={}, retrieved=[], state={})
    r = ground_record(rec)
    assert r.state == STATE_INSUFFICIENT_CONTEXT
    assert r.score is None
