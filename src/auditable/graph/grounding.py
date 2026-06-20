"""Deterministic grounding: is a stated ``decision_basis`` supported by what was read?

The model facet of the v0.3 graph asks a model-consistency question: does the
basis a model gave for a decision line up with the context the decision actually
relied on? This helper answers it deterministically, with no model call and no
NLI backend, so it runs straight from a record offline.

It is consistency evidence, NOT a calibrated probability. A high score means the
checkable anchors in the basis (its words, numbers, and identifiers) appear in the
read context; it does not mean the decision was correct. The score is a normalized
support / ranking signal until labeled data supports calibration, so this module
never claims a calibrated number.

Pinned context contract (Round 3 review), fixed in v0.3 so own-record and live
callers in v0.3b inherit it unchanged:

- **Accepted context** is drawn from exactly three places on a decision: its
  ``data.inputs`` (a dict), its ``data.retrieved`` (a list of items), and its
  ``data.snapshot.state`` (a dict). Nothing else is read.
- **Flattening rule.** Every accepted source is walked into one comparable bag of
  anchors. For a dict, both keys and values contribute (a key like
  ``budget_remaining`` carries meaning, not only its value). For a list, each item
  contributes; an item may itself be a string, a number, or a nested dict / list,
  and is walked the same way. Each scalar leaf becomes anchors by the same rule
  used on the basis text: numeric literals become canonical **numbers**
  (``5000`` == ``"$5,000"`` == ``5000.0``), identifier-like tokens become
  **entities** (``INV-100``, ``kyc-2026-03``, matched exactly), and the remaining
  content words become **words** (function words are dropped). Numbers and
  entities are weighted above plain words because they are the load-bearing,
  checkable parts of a basis.
- **Empty / no-context behavior.** If none of the three sources yields any anchor,
  the result is marked ``insufficient_context`` with no numeric score (``None``),
  never a false high. The symmetric case, a basis with no checkable anchor (blank
  or only function words), is marked ``insufficient_basis``, also with no score.

The score is weighted recall of the basis anchors against the context anchors: of
what the basis asserts, how much is present in what was read. A caller that wants a
model-facet *risk* uses ``1 - score`` on a ``scored`` result. The module is pure
Python (``re`` only): no model call, no NLI, no networkx, no torch.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; ground_record duck-types so there is no runtime import
    from ..record import DecisionRecord

__all__ = [
    "GroundingResult",
    "ground_basis",
    "ground_record",
    "STATE_SCORED",
    "STATE_INSUFFICIENT_CONTEXT",
    "STATE_INSUFFICIENT_BASIS",
]

STATE_SCORED = "scored"
STATE_INSUFFICIENT_CONTEXT = "insufficient_context"
STATE_INSUFFICIENT_BASIS = "insufficient_basis"

# Anchor weights. Numbers and identifiers are the checkable anchors of a basis, so
# they outweigh plain words: a basis that names a number or id absent from the read
# context is a stronger ungrounded signal than a missing common word.
W_WORD = 1.0
W_NUMBER = 2.0
W_ENTITY = 2.0

_METHOD = (
    "deterministic token / number / entity overlap; weighted recall of the basis "
    "anchors against the read context. Consistency evidence, not a calibrated probability."
)

# A number: optional currency sign, then digits with optional thousands separators
# and an optional decimal part. No leading +/- so a hyphen inside an identifier
# (INV-100) is a separator, not a minus; embedded numbers (the 100 in INV-100) are
# still captured, identically on both the basis and the context side.
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_SEP = "-_/:"
_SEP_RE = re.compile(r"[%s]+" % re.escape(_SEP))
_EDGE_PUNCT = "\"'`.,;:!?()[]{}<>"

# Function words only: dropped so they do not inflate overlap. Content words
# (including status verbs such as "approved" / "overdue") are kept.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
        "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
        "into", "is", "it", "its", "of", "on", "or", "our", "she", "so", "than",
        "that", "the", "their", "then", "there", "these", "they", "this", "to",
        "was", "we", "were", "what", "when", "which", "will", "with", "within",
        "you", "your", "not", "no", "do", "does", "did", "also", "per", "over",
    }
)


def _canon_number(value: Any) -> Optional[str]:
    """Canonicalize a numeric literal so int / float / formatted forms compare equal.

    ``5000``, ``5000.0``, ``"5,000"``, and ``"$5,000"`` all map to ``"5000"``.
    Returns ``None`` for non-finite or unparsable values (booleans are not numbers).
    """
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            num = float(value.replace(",", "").replace("$", ""))
        else:
            num = float(value)
        if not math.isfinite(num):
            return None
        if num == int(num):
            return str(int(num))
        return "%g" % num
    except (ValueError, OverflowError):
        return None


def _is_identifier(tok: str) -> bool:
    """An identifier-like token (entity): a letter+digit mix, or alphanumerics joined
    by an internal separator. ``INV-100`` / ``kyc-2026-03`` / ``policy_id`` qualify;
    a plain word or a bare number does not."""
    if not tok:
        return False
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    if has_alpha and has_digit:
        return True
    if has_alpha and any(c in _SEP for c in tok):
        core = _SEP_RE.sub("", tok)
        return core.isalnum() and len(core) >= 2
    return False


def _norm_id(tok: str) -> str:
    """Canonicalize an identifier so separator style does not block a match
    (``po_status`` and ``po-status`` both normalize to ``po-status``)."""
    return _SEP_RE.sub("-", tok).strip("-")


def _anchors_from_text(text: str) -> Dict[str, float]:
    """Extract weighted anchors from one text leaf: numbers (``#``), entities (``@``),
    and content words (bare). One namespace keyed by a typed prefix; the value is the
    anchor weight. Applied identically to the basis and to every context leaf."""
    anchors: Dict[str, float] = {}
    if not text:
        return anchors
    s = text.lower()
    for raw_num in _NUM_RE.findall(s):
        canon = _canon_number(raw_num)
        if canon is not None:
            _bump(anchors, "#" + canon, W_NUMBER)
    for raw_tok in s.split():
        tok = raw_tok.strip(_EDGE_PUNCT)
        if _is_identifier(tok):
            _bump(anchors, "@" + _norm_id(tok), W_ENTITY)
    for word in _WORD_SPLIT_RE.split(s):
        if word and not word.isdigit() and word not in _STOPWORDS:
            _bump(anchors, word, W_WORD)
    return anchors


def _bump(anchors: Dict[str, float], key: str, weight: float) -> None:
    if weight > anchors.get(key, 0.0):
        anchors[key] = weight


def _collect_context(obj: Any, acc: Dict[str, float]) -> None:
    """Flatten one accepted context source into ``acc`` per the pinned contract:
    dict keys and values, list / tuple / set items, and scalar leaves all become
    anchors. Booleans become the words ``true`` / ``false``; numbers canonicalize."""
    if obj is None:
        return
    if isinstance(obj, bool):
        for k, w in _anchors_from_text("true" if obj else "false").items():
            _bump(acc, k, w)
    elif isinstance(obj, (int, float)):
        canon = _canon_number(obj)
        if canon is not None:
            _bump(acc, "#" + canon, W_NUMBER)
    elif isinstance(obj, str):
        for k, w in _anchors_from_text(obj).items():
            _bump(acc, k, w)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _collect_context(key, acc)  # keys carry meaning, not only their values
            _collect_context(value, acc)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _collect_context(item, acc)
    else:
        for k, w in _anchors_from_text(str(obj)).items():
            _bump(acc, k, w)


def _render(key: str) -> str:
    """Human-readable anchor: strip the typed prefix (``#`` number, ``@`` entity)."""
    return key[1:] if key[:1] in "#@" else key


def _render_sorted(keys) -> List[str]:
    return sorted({_render(k) for k in keys})


def _by_kind(claim: Dict[str, float], ctx: Dict[str, float]) -> Dict[str, Dict[str, int]]:
    """Per-kind matched / total counts, so the word / number / entity overlap is
    visible behind the single score."""
    counts = {"word": [0, 0], "number": [0, 0], "entity": [0, 0]}
    for key in claim:
        kind = "number" if key.startswith("#") else "entity" if key.startswith("@") else "word"
        counts[kind][1] += 1
        if key in ctx:
            counts[kind][0] += 1
    return {kind: {"matched": m, "total": t} for kind, (m, t) in counts.items()}


@dataclass
class GroundingResult:
    """The grounding verdict for one ``decision_basis`` against its read context.

    - ``state``: ``scored`` / ``insufficient_context`` / ``insufficient_basis``.
    - ``score``: weighted recall of the basis anchors found in context, in [0, 1]
      when ``scored`` (1.0 = every checkable anchor is supported); ``None`` in either
      insufficient state, so a withheld result can never read as a false high. This
      is support, not risk: a model-facet risk is ``1 - score`` on a scored result.
    - ``matched`` / ``unmatched``: the basis anchors found in / absent from the
      context (human-readable). ``unmatched`` names the unsupported parts of the
      stated basis, which is the actionable signal for an auditor.
    - ``evidence``: the transparent breakdown (method, weights, context size, and
      the per-kind word / number / entity overlap).
    """

    state: str
    score: Optional[float]
    matched: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


def ground_basis(
    decision_basis: str,
    *,
    inputs: Optional[dict] = None,
    retrieved: Optional[list] = None,
    state: Optional[dict] = None,
) -> GroundingResult:
    """Score whether ``decision_basis`` is supported by the read context.

    Context is the three accepted sources from the pinned contract: ``inputs``
    (``data.inputs``), ``retrieved`` (``data.retrieved``), and ``state``
    (``data.snapshot.state``). Each is flattened into comparable anchors and the
    basis anchors are scored by weighted recall against them. See the module
    docstring for the flattening rule and the empty-context behavior. The result is
    deterministic consistency evidence, not a calibrated probability.
    """
    claim = _anchors_from_text(decision_basis or "")
    ctx: Dict[str, float] = {}
    _collect_context(inputs, ctx)
    _collect_context(retrieved, ctx)
    _collect_context(state, ctx)

    if not claim:
        # nothing checkable in the basis: cannot assess grounding, and must not score high
        return GroundingResult(
            state=STATE_INSUFFICIENT_BASIS,
            score=None,
            matched=[],
            unmatched=[],
            evidence={
                "method": _METHOD,
                "reason": "decision_basis has no checkable token, number, or entity anchor",
                "n_context_tokens": len(ctx),
            },
        )

    if not ctx:
        # the named no-context case: a basis but nothing was read. Every basis anchor
        # is unsupported; mark insufficient with no numeric score, never a false high.
        return GroundingResult(
            state=STATE_INSUFFICIENT_CONTEXT,
            score=None,
            matched=[],
            unmatched=_render_sorted(claim.keys()),
            evidence={
                "method": _METHOD,
                "reason": "no readable context in inputs / retrieved / snapshot.state",
                "n_context_tokens": 0,
            },
        )

    total_weight = sum(claim.values())
    matched_keys = [k for k in claim if k in ctx]
    unmatched_keys = [k for k in claim if k not in ctx]
    matched_weight = sum(claim[k] for k in matched_keys)
    score = round(matched_weight / total_weight, 4) if total_weight else 0.0

    return GroundingResult(
        state=STATE_SCORED,
        score=score,
        matched=_render_sorted(matched_keys),
        unmatched=_render_sorted(unmatched_keys),
        evidence={
            "method": _METHOD,
            "matched_weight": round(matched_weight, 4),
            "total_weight": round(total_weight, 4),
            "n_context_tokens": len(ctx),
            "by_kind": _by_kind(claim, ctx),
            "weights": {"word": W_WORD, "number": W_NUMBER, "entity": W_ENTITY},
        },
    )


def ground_record(record: "DecisionRecord") -> GroundingResult:
    """Grounding straight from a ``DecisionRecord`` (the "runnable from the record" path).

    Pulls the basis from ``record.model.decision_basis`` and the context from the
    three pinned sources ``record.data.inputs`` / ``record.data.retrieved`` /
    ``record.data.snapshot.state``. Duck-typed: any object with that shape works, so
    the offline loader, a live sink record, and a corpus-adapter record all share one
    entry without a hard import of the record type.
    """
    model = getattr(record, "model", None)
    data = getattr(record, "data", None)
    snapshot = getattr(data, "snapshot", None)
    basis = getattr(model, "decision_basis", None) or ""
    inputs = getattr(data, "inputs", None)
    retrieved = getattr(data, "retrieved", None)
    state = getattr(snapshot, "state", None)
    return ground_basis(basis, inputs=inputs, retrieved=retrieved, state=state)
