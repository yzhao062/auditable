"""The AuditReport: one aggregate over PRE, LIVE, and POST, two consumers.

The audit report has two consumers, and the FORMAT serves the CONSUMER. The same
aggregated data renders two ways:

- :meth:`AuditReport.to_markdown` is AGENT-facing. It is lean, deterministic,
  parseable, token-light, and ACTIONABLE: stable section anchors an agent can
  regex or split on (``## Verdict``, ``## Keystone``, ``## Findings``,
  ``## Recommended Actions``, ``## Coverage``) plus a fenced ``json`` block that
  mirrors the headline fields for callers that prefer structured data over prose.
  It embeds NO raster image, because an agent consumes structured text, not PNG
  charts. It works with ZERO heavy dependencies, so an agent loop can render it on
  a bare install.
- :meth:`AuditReport.to_pdf` is HUMAN-facing. It is the polished, shareable
  artifact: the same findings as narrative prose plus the five embedded charts, a
  cover with run identity and the verdict banner, headings, and a layout a
  stakeholder skims. The charts live ONLY here. ``to_pdf`` lazy-imports fpdf2,
  matplotlib, and networkx (the ``report`` extra) and raises a clear
  ``pip install auditable[report]`` error when any is absent, mirroring how
  :mod:`auditable.graph` guards the ``graph`` extra.

The report aggregates three STABLE upstream report objects, none from
``auditable.report``:

- PRE: :class:`auditable.graph.pre.PreReport` from :func:`analyze_plan`. It carries
  the STRUCTURAL execution-topology keystone (kept distinct from the blast
  keystone), the lint findings, and the Preflight Coverage Report.
- POST: :class:`auditable.analysis.AnalysisReport` from :func:`analyze_run`. It
  carries the blast-radius keystone, the ranked blast-share list, coverage, and
  the no-score state.
- LIVE: a sequence of :class:`auditable.chain.Verdict` from
  :func:`auditable.chain.replay`, each correlated to a
  :class:`auditable.record.DecisionRecord` by ``record_id`` for the recovery
  table.

Every pillar is optional. A missing pillar renders as an explicit "no PRE / no
LIVE / no POST data" line, never a fabricated zero, mirroring the upstream
no-score honesty boundary (a withheld value is not a zero value).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["AuditReport", "Finding"]


# The roll-up verdict the agent loop branches on. One of three values.
VERDICT_PASS = "PASS"
VERDICT_REVIEW = "REVIEW"
VERDICT_BLOCK = "BLOCK"

# The hint repeated wherever the heavy report extra is required.
REPORT_EXTRA_HINT = (
    "the human-facing PDF requires the 'report' extra: pip install auditable[report]"
)

# Source tags for a finding row. The three lifecycle pillars.
SOURCE_PRE = "PRE"
SOURCE_LIVE = "LIVE"
SOURCE_POST = "POST"

# A stable severity rank for the deterministic findings sort (most severe first).
# This mirrors the rank the chart layer uses, so the table and the chart agree.
_SEVERITY_RANK = {"block": 0, "rollback": 1, "human_review": 2, "warning": 3, "info": 4}

# A stable source rank so the secondary sort key is deterministic.
_SOURCE_RANK = {SOURCE_PRE: 0, SOURCE_LIVE: 1, SOURCE_POST: 2}


def _fmt_score(x: Optional[float]) -> str:
    """Render a blast-share score: ``n/a`` when withheld, else three decimals.

    A withheld (``None``) score must never read as ``0.000``: a zero score means
    "nothing rests on this step", a real and different claim. The same precision
    is used everywhere so two runs over the same inputs diff cleanly.
    """
    return "n/a" if x is None else f"{x:.3f}"


def _round3(x: Optional[float]) -> Optional[float]:
    """Round a float to three places for the JSON mirror, or pass ``None`` through."""
    return None if x is None else round(float(x), 3)


@dataclass(frozen=True)
class Finding:
    """One row in the unified findings table.

    A finding is either a PRE lint or a LIVE non-ALLOW verdict, normalized to
    one shape so the agent reads a single sorted table.

    - ``severity``: ``'warning'`` for a PRE lint, or the :class:`FixAction` value
      (``'block'`` / ``'rollback'`` / ``'human_review'``) for a LIVE verdict.
    - ``source``: one of ``'PRE'`` / ``'LIVE'`` / ``'POST'``.
    - ``locus``: the step idx (PRE) or the record id (LIVE) the finding is
      about, rendered as a string so the table column is uniform.
    - ``code``: the lint name (PRE) or the verdict action (LIVE).
    - ``detail``: a one-line human reason.
    - ``action``: the imperative, machine-pickable line an agent can act on.
    - ``sort_idx``: an integer ordering hint within a source (the step idx for
      PRE, the verdict position for LIVE), used only for a stable sort.
    """

    severity: str
    source: str
    locus: str
    code: str
    detail: str
    action: str
    sort_idx: int = 0

    def _sort_key(self) -> Tuple[int, int, int, str]:
        """Severity desc, then source, then the within-source ordering, then code."""
        return (
            _SEVERITY_RANK.get(self.severity, 9),
            _SOURCE_RANK.get(self.source, 9),
            self.sort_idx,
            self.code,
        )

    def to_dict(self) -> Dict[str, Any]:
        """The JSON-mirror shape: the columns an agent extracts, no ordering hint."""
        return {
            "severity": self.severity,
            "source": self.source,
            "locus": self.locus,
            "code": self.code,
            "detail": self.detail,
            "action": self.action,
        }


# --- PRE -> findings + actions -----------------------------------------------


def _pre_findings(pre: Any) -> List[Finding]:
    """Lower each PRE :class:`LintFinding` into a unified :class:`Finding` row.

    Every PRE lint is a structural design warning (``severity='warning'``). The
    imperative action is derived from the lint name so an agent gets a concrete,
    pickable instruction per finding.
    """
    out: List[Finding] = []
    for f in getattr(pre, "findings", None) or []:
        lint = getattr(f, "lint", "lint")
        node_idx = getattr(f, "node_idx", None)
        resource_id = getattr(f, "resource_id", None)
        severity = getattr(f, "severity", "warning") or "warning"
        detail = getattr(f, "detail", "") or ""
        out.append(
            Finding(
                severity=severity,
                source=SOURCE_PRE,
                locus="" if node_idx is None else f"step {node_idx}",
                code=lint,
                detail=detail,
                action=_pre_action(lint, node_idx, resource_id),
                sort_idx=node_idx if isinstance(node_idx, int) else 0,
            )
        )
    return out


def _pre_action(lint: str, node_idx: Optional[int], resource_id: Optional[str]) -> str:
    """One imperative line per PRE lint, derived from the lint name (no new analysis)."""
    res = f"'{resource_id}'" if resource_id else "the flagged resource"
    at = f"step {node_idx}" if node_idx is not None else "the flagged step"
    if lint == "write_with_no_prior_read":
        return f"read {res} before {at} writes it, or justify the blind write"
    if lint == "flippable_dependency_annotation":
        return f"pin or revalidate {res} before {at} decides on it"
    if lint == "scope_vs_snapshot":
        return f"narrow the granted scope at {at} to the state it read, dropping {res}"
    if lint == "missing_revalidation_barrier":
        return f"insert a revalidation re-read of {res} before {at} acts"
    return f"review the {lint} lint at {at}"


# --- LIVE -> findings + actions + recovery ------------------------------


@dataclass(frozen=True)
class Recovery:
    """One non-ALLOW LIVE verdict, the recovery row for the human PDF table.

    - ``action``: the :class:`FixAction` value (``'block'`` / ``'human_review'`` /
      ``'rollback'``).
    - ``record_id``: the correlated record digest, or ``''`` when the verdict
      carries none.
    - ``reason``: the verdict's reason string.
    """

    action: str
    record_id: str
    reason: str


def _verdict_action_value(verdict: Any) -> Optional[str]:
    """The FixAction value of a verdict (``'allow'`` / ``'block'`` / ...), or ``None``."""
    action = getattr(verdict, "action", None)
    if action is None:
        return None
    return getattr(action, "value", None) or str(action)


def _realtime_findings(verdicts: Sequence[Any]) -> List[Finding]:
    """Lower each non-ALLOW LIVE :class:`Verdict` into a :class:`Finding` row.

    An ALLOW verdict is not a finding (nothing to act on). Each non-ALLOW verdict
    becomes a row whose severity is the FixAction value, so a BLOCK sorts above a
    ROLLBACK above a HUMAN_REVIEW. The imperative action names the routed fix and
    the record it applies to.
    """
    out: List[Finding] = []
    for pos, v in enumerate(verdicts or ()):
        value = _verdict_action_value(v)
        if value is None or value == "allow":
            continue
        record_id = getattr(v, "record_id", "") or ""
        reason = getattr(v, "reason", "") or ""
        out.append(
            Finding(
                severity=value,
                source=SOURCE_LIVE,
                locus=f"record {_short_id(record_id)}" if record_id else f"verdict {pos}",
                code=value,
                detail=reason,
                action=_realtime_action(value, record_id),
                sort_idx=pos,
            )
        )
    return out


def _realtime_action(value: str, record_id: str) -> str:
    """One imperative line per non-ALLOW verdict, naming the routed fix."""
    rid = _short_id(record_id) if record_id else "the affected record"
    if value == "rollback":
        return f"rollback record {rid}: it relied on stale dependency state"
    if value == "block":
        return f"block record {rid}: it is unjustified under the live state"
    if value == "human_review":
        return f"route record {rid} to human review: replay could not decide"
    return f"review record {rid} (verdict {value})"


def _short_id(record_id: str, n: int = 12) -> str:
    """A short, stable record-id prefix for legibility (the full id stays in JSON)."""
    if not record_id:
        return ""
    return record_id if len(record_id) <= n else record_id[:n]


def _recoveries(verdicts: Sequence[Any]) -> List[Recovery]:
    """The non-ALLOW verdicts as recovery rows (the human PDF recovery table)."""
    out: List[Recovery] = []
    for v in verdicts or ():
        value = _verdict_action_value(v)
        if value is None or value == "allow":
            continue
        out.append(
            Recovery(
                action=value,
                record_id=getattr(v, "record_id", "") or "",
                reason=getattr(v, "reason", "") or "",
            )
        )
    return out


def _recovery_counts(verdicts: Sequence[Any]) -> Dict[str, int]:
    """Count the verdict sequence by FixAction value (the four outcomes)."""
    counts = {"allow": 0, "block": 0, "human_review": 0, "rollback": 0}
    for v in verdicts or ():
        value = _verdict_action_value(v)
        if value in counts:
            counts[value] += 1
    return counts


def _record_index(records: Optional[Sequence[Any]]) -> Dict[str, Any]:
    """Map ``record_id -> DecisionRecord`` for the optional recovery-table records."""
    out: Dict[str, Any] = {}
    for r in records or ():
        rid = getattr(r, "record_id", None)
        if isinstance(rid, str) and rid:
            out[rid] = r
    return out


# --- POST -> keystone view + ranked rows -------------------------------------


@dataclass(frozen=True)
class KeystoneView:
    """A normalized keystone row, shared by the markdown and the PDF.

    - ``idx`` / ``label``: the keystone step's identity.
    - ``score``: the blast share (POST), or ``None`` when withheld.
    - ``downstream`` / ``others``: the "N of M other steps rest on it" pair, or
      ``None`` when no score is available.
    - ``kind``: ``'blast'`` (POST blast-radius) or ``'structural'`` (PRE
      execution-topology chokepoint), kept as two distinct named concepts.
    """

    idx: int
    label: str
    score: Optional[float]
    downstream: Optional[int]
    others: Optional[int]
    kind: str


def _blast_keystone(post: Any) -> Optional[KeystoneView]:
    """The POST blast-radius keystone as a :class:`KeystoneView`, or ``None``."""
    k = getattr(post, "keystone", None)
    if k is None or getattr(k, "score", None) is None:
        return None
    n_steps = int(getattr(post, "n_steps", 0) or 0)
    others = max(n_steps - 1, 0)
    score = float(k.score)
    downstream = round(score * others) if others else 0
    return KeystoneView(
        idx=int(getattr(k, "idx", 0) or 0),
        label=getattr(k, "label", "") or "step",
        score=score,
        downstream=downstream,
        others=others,
        kind="blast",
    )


def _structural_keystone(pre: Any) -> Optional[KeystoneView]:
    """The PRE execution-topology chokepoint as a :class:`KeystoneView`, or ``None``."""
    idx = getattr(pre, "keystone_idx", None)
    if idx is None:
        return None
    followers = int(getattr(pre, "keystone_followers", 0) or 0)
    n_steps = int(getattr(pre, "n_steps", 0) or 0)
    others = max(n_steps - 1, 0)
    return KeystoneView(
        idx=int(idx),
        label="execution chokepoint",
        score=None,
        downstream=followers,
        others=others,
        kind="structural",
    )


# --- coverage view -----------------------------------------------------------


def _coverage_line(pre: Any, post: Any) -> str:
    """One coverage line: grade mix + rho + the exact no-score reason, POST-preferred.

    POST coverage is the runtime truth (observed edges may exist), so it is
    preferred when present; PRE preflight coverage is the design-time fallback. The
    line states the exact no-score reason so an agent knows when a score is withheld
    versus a real zero. Returns an explicit "no coverage data" line when neither
    pillar carries coverage.
    """
    if post is not None and getattr(post, "coverage", None) is not None:
        cov = post.coverage
        mix = _grade_mix_from_by_grade(getattr(cov, "by_grade", {}) or {})
        reason = getattr(post, "state", "") or ""
        return (
            f"{cov.n_dep_edges} dependency edge(s); grades {mix}; "
            f"rho={cov.rho:.3f}; observed_fraction={cov.observed_fraction:.0%}; "
            f"state {reason}"
        )
    pc = getattr(pre, "preflight_coverage", None) if pre is not None else None
    if pc is not None:
        mix = (
            f"observed={pc.observed}, declared={pc.declared}, inferred={pc.inferred}"
        )
        return (
            f"{pc.n_dep_edges} dependency edge(s); grades {mix}; "
            f"rho={pc.rho:.3f}; observed_fraction={pc.observed_fraction:.0%}; "
            f"runtime no-score reason: {pc.no_score_reason}"
        )
    return "no coverage data (no PRE preflight coverage and no POST coverage)"


def _grade_mix_from_by_grade(by_grade: Dict[Any, int]) -> str:
    """The observed/declared/inferred mix as a compact string from a Grade map."""
    parts: List[str] = []
    for grade, count in by_grade.items():
        if count:
            name = getattr(grade, "value", str(grade))
            parts.append(f"{name}={count}")
    return ", ".join(parts) if parts else "none"


# --- the roll-up verdict -----------------------------------------------------


def _headline_verdict(
    pre: Any, findings: Sequence[Finding], recovery_counts: Dict[str, int]
) -> str:
    """The single roll-up verdict the agent loop branches on.

    BLOCK if any LIVE verdict blocked; else REVIEW if any verdict routed a
    human review or a rollback, or any PRE lint fired; else PASS. A withheld POST
    score does not raise the verdict on its own (a missing number is not a failure).
    """
    if recovery_counts.get("block", 0) > 0:
        return VERDICT_BLOCK
    if recovery_counts.get("human_review", 0) > 0 or recovery_counts.get("rollback", 0) > 0:
        return VERDICT_REVIEW
    if pre is not None and (getattr(pre, "findings", None) or []):
        return VERDICT_REVIEW
    if any(f.source == SOURCE_PRE for f in findings):
        return VERDICT_REVIEW
    return VERDICT_PASS


# --- the aggregate -----------------------------------------------------------


@dataclass(frozen=True)
class AuditReport:
    """The aggregate over PRE, LIVE, and POST, rendered two ways.

    Holds the three upstream report objects plus a small set of derived,
    format-agnostic view fields computed once at construction, so
    :meth:`to_markdown` and :meth:`to_pdf` render identical numbers. Aggregation is
    pure and side-effect-free, and nothing here imports ``auditable.report``.

    Build with :meth:`from_run` (the primary constructor) or the single-pillar
    :meth:`from_pre` / :meth:`from_analysis` helpers. Every input is optional, so a
    PRE-only or POST-only report still renders.

    Derived view fields:

    - ``headline_verdict``: the roll-up the agent branches on (PASS / REVIEW /
      BLOCK).
    - ``blast_keystone`` (POST) and ``structural_keystone`` (PRE): the two distinct
      keystone concepts, each ``None`` when its pillar is absent or withheld.
    - ``findings``: the unified, sorted findings table (PRE lints + LIVE
      non-ALLOW verdicts).
    - ``recommended_actions``: one imperative line per finding, machine-pickable.
    - ``recoveries``: the non-ALLOW verdicts as recovery rows (the human table).
    - ``recovery_counts``: verdict counts by FixAction.
    - ``coverage_line``: the grade mix, rho, and the exact no-score reason.
    """

    title: str
    pre: Optional[Any]
    post: Optional[Any]
    verdicts: Tuple[Any, ...]
    records: Tuple[Any, ...]
    headline_verdict: str
    blast_keystone: Optional[KeystoneView]
    structural_keystone: Optional[KeystoneView]
    findings: Tuple[Finding, ...]
    recommended_actions: Tuple[str, ...]
    recoveries: Tuple[Recovery, ...]
    recovery_counts: Dict[str, int]
    coverage_line: str

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_run(
        cls,
        *,
        pre: Optional[Any] = None,
        post: Optional[Any] = None,
        verdicts: Optional[Sequence[Any]] = (),
        records: Optional[Sequence[Any]] = None,
        title: str = "Auditable Audit Report",
    ) -> "AuditReport":
        """Aggregate the three pillars into one report. The primary constructor.

        ``pre`` is a :class:`~auditable.graph.pre.PreReport`, ``post`` an
        :class:`~auditable.analysis.AnalysisReport`, ``verdicts`` a sequence of
        :class:`~auditable.chain.Verdict`, and ``records`` an optional sequence of
        :class:`~auditable.record.DecisionRecord` for the human recovery table. Each
        is optional; a missing pillar is rendered explicitly rather than zeroed.
        Aggregation is pure: it reads the passed objects and computes the derived
        view fields, importing nothing heavy.
        """
        verdict_tuple: Tuple[Any, ...] = tuple(verdicts or ())
        record_tuple: Tuple[Any, ...] = tuple(records or ())

        findings: List[Finding] = []
        findings.extend(_pre_findings(pre) if pre is not None else [])
        findings.extend(_realtime_findings(verdict_tuple))
        findings.sort(key=lambda f: f._sort_key())

        recovery_counts = _recovery_counts(verdict_tuple)
        headline = _headline_verdict(pre, findings, recovery_counts)
        recommended = tuple(f.action for f in findings)

        return cls(
            title=title,
            pre=pre,
            post=post,
            verdicts=verdict_tuple,
            records=record_tuple,
            headline_verdict=headline,
            blast_keystone=_blast_keystone(post) if post is not None else None,
            structural_keystone=_structural_keystone(pre) if pre is not None else None,
            findings=tuple(findings),
            recommended_actions=recommended,
            recoveries=tuple(_recoveries(verdict_tuple)),
            recovery_counts=recovery_counts,
            coverage_line=_coverage_line(pre, post),
        )

    @classmethod
    def from_pre(cls, pre: Any, *, title: str = "Auditable Audit Report (PRE)") -> "AuditReport":
        """A single-pillar PRE report (no LIVE, no POST)."""
        return cls.from_run(pre=pre, title=title)

    @classmethod
    def from_analysis(
        cls, post: Any, *, title: str = "Auditable Audit Report (POST)"
    ) -> "AuditReport":
        """A single-pillar POST report (no PRE, no LIVE)."""
        return cls.from_run(post=post, title=title)

    # -- the JSON mirror ------------------------------------------------------

    def _headline_dict(self) -> Dict[str, Any]:
        """The headline fields the ``json`` block mirrors, for structured callers.

        Deterministic: fixed keys, scores at three-decimal precision, the findings
        list in the same sorted order as the table. An agent that prefers structured
        data over prose parses this instead of the section bodies.
        """
        blast = self.blast_keystone
        structural = self.structural_keystone
        return {
            "verdict": self.headline_verdict,
            "title": self.title,
            "pillars": {
                "pre": self.pre is not None,
                "live": bool(self.verdicts),
                "post": self.post is not None,
            },
            "blast_keystone": (
                None
                if blast is None
                else {
                    "idx": blast.idx,
                    "label": blast.label,
                    "score": _round3(blast.score),
                    "downstream": blast.downstream,
                    "others": blast.others,
                }
            ),
            "structural_keystone": (
                None
                if structural is None
                else {
                    "idx": structural.idx,
                    "followers": structural.downstream,
                    "others": structural.others,
                }
            ),
            "counts": {
                "findings": len(self.findings),
                "by_severity": self._severity_counts(),
                "recovery": dict(sorted(self.recovery_counts.items())),
            },
            "findings": [f.to_dict() for f in self.findings],
            "recommended_actions": list(self.recommended_actions),
        }

    def _severity_counts(self) -> Dict[str, int]:
        """Finding counts by severity, sorted by key for a deterministic JSON block."""
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return dict(sorted(counts.items()))

    # -- the AGENT-facing Markdown -------------------------------------------

    def to_markdown(self, path: Optional[str] = None) -> str:
        """Render the AGENT-facing Markdown, and write it when ``path`` is given.

        Lean, deterministic, parseable, and ACTIONABLE. Stable section anchors an
        agent can split on: ``## Verdict``, ``## Keystone``, ``## Findings``,
        ``## Recommended Actions``, ``## Coverage``, plus a fenced ``json`` block
        mirroring the headline fields. NO embedded raster image: an agent consumes
        structured text, not charts. This method embeds no figure, so it works with
        ZERO heavy dependencies (it never imports matplotlib, fpdf2, or networkx).

        Determinism: fixed section order, fixed sort keys (severity desc, then
        source, then locus), floats at three-decimal precision. Two runs over the
        same inputs diff cleanly.
        """
        lines: List[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(
            "_Agent-facing report. Structured, parseable, no images. The human PDF "
            "(`to_pdf`) carries the charts._"
        )

        lines.extend(self._md_verdict())
        lines.extend(self._md_keystone())
        lines.extend(self._md_findings())
        lines.extend(self._md_recommended_actions())
        lines.extend(self._md_coverage())
        lines.extend(self._md_json_block())

        text = "\n".join(lines) + "\n"
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    def _md_verdict(self) -> List[str]:
        """The ``## Verdict`` section: the one-line roll-up the loop branches on."""
        pillars = ", ".join(
            name
            for name, present in (
                ("PRE", self.pre is not None),
                ("LIVE", bool(self.verdicts)),
                ("POST", self.post is not None),
            )
            if present
        ) or "none"
        n_block = self.recovery_counts.get("block", 0)
        n_rollback = self.recovery_counts.get("rollback", 0)
        n_review = self.recovery_counts.get("human_review", 0)
        n_pre = sum(1 for f in self.findings if f.source == SOURCE_PRE)
        rationale = (
            f"{n_block} block, {n_rollback} rollback, {n_review} human-review "
            f"verdict(s); {n_pre} PRE lint(s)"
        )
        return [
            "",
            "## Verdict",
            "",
            f"**{self.headline_verdict}** ({rationale})",
            "",
            f"- pillars present: {pillars}",
            f"- findings: {len(self.findings)}",
        ]

    def _md_keystone(self) -> List[str]:
        """The ``## Keystone`` section: the single step to review first.

        The POST blast-radius keystone leads (it is the runtime triage signal). The
        PRE execution-topology chokepoint is reported alongside as a distinct named
        concept, never conflated with the blast keystone.
        """
        out = ["", "## Keystone", ""]
        blast = self.blast_keystone
        if blast is not None:
            out.append(
                f"- blast-radius (POST): step {blast.idx} [{blast.label}], "
                f"blast share {_fmt_score(blast.score)} "
                f"({blast.downstream} of {blast.others} other steps rest on it)"
            )
        elif self.post is not None:
            out.append("- blast-radius (POST): withheld (no score; see Coverage)")
        else:
            out.append("- blast-radius (POST): no POST data")

        structural = self.structural_keystone
        if structural is not None:
            out.append(
                f"- execution chokepoint (PRE): step {structural.idx} "
                f"({structural.downstream} of {structural.others} other steps follow "
                "it in control flow; a structural design lint, not the blast keystone)"
            )
        elif self.pre is not None:
            out.append("- execution chokepoint (PRE): none (no control-flow followers)")
        else:
            out.append("- execution chokepoint (PRE): no PRE data")
        return out

    def _md_findings(self) -> List[str]:
        """The ``## Findings`` section: a flat, sorted, stable-ordered table.

        Columns: severity | source | locus | code | detail. Sorted severity desc,
        then source, then within-source locus, so two runs diff cleanly. A withheld
        POST score is not a finding; it surfaces in Coverage instead.
        """
        out = ["", "## Findings", ""]
        if not self.findings:
            out.append("(none): no PRE lints fired and no LIVE verdict routed a fix.")
            return out
        out.append("| severity | source | locus | code | detail |")
        out.append("| --- | --- | --- | --- | --- |")
        for f in self.findings:
            out.append(
                f"| {_cell(f.severity)} | {_cell(f.source)} | {_cell(f.locus)} | "
                f"{_cell(f.code)} | {_cell(f.detail)} |"
            )
        return out

    def _md_recommended_actions(self) -> List[str]:
        """The ``## Recommended Actions`` section: one imperative line per finding.

        Each line is machine-pickable (an agent can lift it into a loop). Rendered
        as a numbered list so a caller can reference an action by position.
        """
        out = ["", "## Recommended Actions", ""]
        if not self.recommended_actions:
            out.append("(none): no findings require action.")
            return out
        for i, action in enumerate(self.recommended_actions, 1):
            out.append(f"{i}. {action}")
        return out

    def _md_coverage(self) -> List[str]:
        """The ``## Coverage`` section: grade mix + rho + the exact no-score reason.

        The exact no-score reason tells the agent when a score is withheld versus a
        real zero, so it never reads a withheld blast share as zero risk.
        """
        return ["", "## Coverage", "", f"- {self.coverage_line}"]

    def _md_json_block(self) -> List[str]:
        """The fenced ``json`` block mirroring the headline fields.

        For callers that prefer structured data over prose. ``sort_keys`` keeps the
        block byte-stable across runs over the same inputs.
        """
        payload = json.dumps(self._headline_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        return ["", "## JSON", "", "```json", payload, "```"]

    # -- the HUMAN-facing PDF -------------------------------------------------

    def to_pdf(self, path: str) -> bytes:
        """Render the HUMAN-facing PDF to ``path`` and return its bytes.

        The polished, shareable artifact: a cover with the run identity and the
        verdict banner, per-stage finding sections, narrative prose, and the five
        embedded charts. The charts live ONLY here, never in the agent Markdown.

        Lazy-imports fpdf2 (and, through the figure functions, matplotlib +
        networkx). With the ``report`` extra absent it raises a clear
        ``pip install auditable[report]`` ImportError, mirroring the
        ``auditable.graph`` extra guard. Importing this module (and calling
        :meth:`to_markdown`) succeeds without the extra; only this method requires it.
        """
        try:
            from fpdf import FPDF
        except Exception as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(REPORT_EXTRA_HINT) from exc

        from . import _report_figures as figs

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_title(self.title)
        pdf.add_page()

        self._pdf_cover(pdf)
        self._pdf_verdict_banner(pdf)
        self._pdf_keystone(pdf)
        self._pdf_findings(pdf)
        self._pdf_recovery(pdf)
        self._pdf_charts(pdf, figs)

        out = pdf.output()
        data = bytes(out)
        with open(path, "wb") as handle:
            handle.write(data)
        return data

    # -- PDF section helpers (only reached from to_pdf, deps already imported) -

    def _pdf_cover(self, pdf: Any) -> None:
        """The cover header: title plus the run-identity meta lines."""
        pdf.set_font("Helvetica", "B", 18)
        _pdf_text(pdf, self.title, h=10)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(74, 85, 104)
        adapter = (
            getattr(self.post, "adapter", None)
            or getattr(self.pre, "adapter", None)
            or "n/a"
        )
        _pdf_text(pdf, f"adapter: {adapter}    pillars: {self._pillar_line()}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def _pillar_line(self) -> str:
        """The present-pillar list as a compact string, shared by cover and banner."""
        return ", ".join(
            name
            for name, present in (
                ("PRE", self.pre is not None),
                ("LIVE", bool(self.verdicts)),
                ("POST", self.post is not None),
            )
            if present
        ) or "none"

    def _pdf_verdict_banner(self, pdf: Any) -> None:
        """The verdict banner: a coloured bar with the roll-up verdict."""
        colors = {
            VERDICT_PASS: (47, 133, 90),
            VERDICT_REVIEW: (214, 158, 46),
            VERDICT_BLOCK: (197, 48, 48),
        }
        r, g, b = colors.get(self.headline_verdict, (74, 85, 104))
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_x(pdf.l_margin)
        pdf.cell(_epw(pdf), 12, _ascii(f"  Verdict: {self.headline_verdict}"), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
        n_block = self.recovery_counts.get("block", 0)
        n_rollback = self.recovery_counts.get("rollback", 0)
        n_review = self.recovery_counts.get("human_review", 0)
        n_pre = sum(1 for f in self.findings if f.source == SOURCE_PRE)
        pdf.set_font("Helvetica", "", 10)
        _pdf_text(
            pdf,
            f"{n_block} block, {n_rollback} rollback, {n_review} human-review "
            f"verdict(s); {n_pre} PRE lint(s); {len(self.findings)} finding(s) total.",
        )
        pdf.ln(2)

    def _pdf_keystone(self, pdf: Any) -> None:
        """The keystone section: the blast keystone and the structural chokepoint."""
        _pdf_heading(pdf, "Keystone")
        pdf.set_font("Helvetica", "", 10)
        blast = self.blast_keystone
        if blast is not None:
            _pdf_text(
                pdf,
                f"Blast-radius (POST): step {blast.idx} [{blast.label}], blast "
                f"share {_fmt_score(blast.score)} ({blast.downstream} of "
                f"{blast.others} other steps rest on it).",
            )
        elif self.post is not None:
            _pdf_text(pdf, "Blast-radius (POST): withheld (no score; see Coverage).")
        else:
            _pdf_text(pdf, "Blast-radius (POST): no POST data.")

        structural = self.structural_keystone
        if structural is not None:
            _pdf_text(
                pdf,
                f"Execution chokepoint (PRE): step {structural.idx} "
                f"({structural.downstream} of {structural.others} other steps "
                "follow it in control flow; a structural design lint, distinct "
                "from the blast keystone).",
            )
        elif self.pre is not None:
            _pdf_text(pdf, "Execution chokepoint (PRE): none (no control-flow followers).")
        else:
            _pdf_text(pdf, "Execution chokepoint (PRE): no PRE data.")
        pdf.ln(1)
        _pdf_text(pdf, f"Coverage: {self.coverage_line}")
        pdf.ln(2)

    def _pdf_findings(self, pdf: Any) -> None:
        """The findings section: the same unified table as the Markdown, as prose rows."""
        _pdf_heading(pdf, "Findings")
        if not self.findings:
            pdf.set_font("Helvetica", "", 10)
            _pdf_text(pdf, "No PRE lints fired and no LIVE verdict routed a fix.")
            pdf.ln(2)
            return
        for i, f in enumerate(self.findings, 1):
            pdf.set_font("Helvetica", "B", 10)
            locus = f" @ {f.locus}" if f.locus else ""
            _pdf_text(pdf, f"{i}. [{f.severity}] {f.source} {f.code}{locus}")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(74, 85, 104)
            if f.detail:
                _pdf_text(pdf, f"    {f.detail}", h=5)
            _pdf_text(pdf, f"    action: {f.action}", h=5)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def _pdf_recovery(self, pdf: Any) -> None:
        """The recovery section: the non-ALLOW verdicts routed to a fix."""
        if not self.verdicts:
            return
        _pdf_heading(pdf, "Recovery (LIVE)")
        pdf.set_font("Helvetica", "", 10)
        counts = self.recovery_counts
        _pdf_text(
            pdf,
            f"Verdicts: {counts.get('allow', 0)} allow, "
            f"{counts.get('human_review', 0)} human-review, "
            f"{counts.get('rollback', 0)} rollback, {counts.get('block', 0)} block.",
        )
        if not self.recoveries:
            _pdf_text(pdf, "All verdicts allowed: replay caught no stale-state drift.")
            pdf.ln(2)
            return
        for rec in self.recoveries:
            pdf.set_font("Helvetica", "B", 9)
            rid = _short_id(rec.record_id) if rec.record_id else "(no record id)"
            _pdf_text(pdf, f"  {rec.action.upper()} - record {rid}", h=5)
            if rec.reason:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(74, 85, 104)
                _pdf_text(pdf, f"    {rec.reason}", h=5)
                pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def _pdf_charts(self, pdf: Any, figs: Any) -> None:
        """Place the five charts, each on its own labeled block.

        The charts are the human-facing payload the Markdown omits. Each PNG is
        embedded from an in-memory stream via ``FPDF.image`` (no temp file). A
        missing pillar renders the figure function's explicit "no data" panel, so
        the section is honest rather than blank.
        """
        import io

        s_idx = getattr(self.structural_keystone, "idx", None)
        b_idx = getattr(self.blast_keystone, "idx", None)
        graph = self._session_graph()
        touch = getattr(self.pre, "resource_touch_completeness", None) if self.pre else None
        coverage_obj = self._coverage_object()

        charts = [
            ("Blast-Share Ranking", figs.blast_share_figure(self.post)),
            ("Preflight Coverage Gauge", figs.coverage_gauge_figure(coverage_obj, touch=touch)),
            (
                "Two-Layer Decision Graph",
                figs.decision_graph_figure(
                    graph, structural_keystone_idx=s_idx, blast_keystone_idx=b_idx
                ),
            ),
            ("Snapshot-vs-Live Drift and Recovery", figs.recovery_figure(self.verdicts)),
            ("Findings by Source and Severity", figs.findings_by_severity_figure(self.findings)),
        ]
        pdf.add_page()
        _pdf_heading(pdf, "Charts")
        img_w = min(180.0, _epw(pdf))
        for caption, png in charts:
            pdf.set_font("Helvetica", "B", 11)
            _pdf_text(pdf, caption, h=7)
            pdf.set_x(pdf.l_margin)
            pdf.image(io.BytesIO(png), w=img_w)
            pdf.ln(3)

    def _session_graph(self) -> Optional[Any]:
        """Return the session graph for the decision-graph chart, when possible.

        Prefer the real :class:`SessionGraph` the POST
        :class:`~auditable.analysis.AnalysisReport` carries on ``session_graph``:
        it retains the per-edge dependency layer, so the two-layer chart draws the
        actual ``depends_on`` edges. When the POST report carries no graph (a PRE-only
        or LIVE-only report, a no_score report built without a real
        ``SessionGraph``, or a duck-typed stand-in), rebuild a minimal control-flow
        backbone from the ranked rows' node attributes. The rebuild has no dependency
        layer (the ranked rows do not carry it); it draws a faithful, if conservative,
        handoff topology. Return ``None`` when neither path yields a graph, and the
        chart renders its explicit no-graph panel.
        """
        if self.post is None:
            return None
        # Prefer the real SessionGraph the POST report carries: it retains the
        # per-edge dependency layer (depends_on edges) that the ranked rows drop. A
        # graph-less, PRE-only, or stand-in POST report has none; fall through to the
        # synthetic control-flow reconstruction below in that case.
        real_graph = getattr(self.post, "session_graph", None)
        if real_graph is not None:
            return real_graph
        try:
            from .graph.session import DependencyEdge, Grade, SessionGraph, Step
        except Exception:  # pragma: no cover - graph extra absent
            return None

        ranked = list(getattr(self.post, "ranked", None) or [])
        if not ranked:
            return None
        # Rebuild a minimal step list from the ranked rows. The chart needs the step
        # nodes plus the handoff backbone and any dependency edges the rows imply; the
        # ranked rows do not carry the per-edge dependency layer, so this draws the
        # control-flow backbone (a faithful, if conservative, topology).
        by_idx = sorted(ranked, key=lambda d: getattr(d, "idx", 0))
        steps: List[Any] = []
        for d in by_idx:
            steps.append(
                Step(
                    idx=int(getattr(d, "idx", 0) or 0),
                    agent=getattr(d, "agent", "") or "agent",
                    kind=getattr(d, "kind", "decision") or "decision",
                    node_attrs=dict(getattr(d, "node_attrs", {}) or {}),
                )
            )
        try:
            return SessionGraph.from_steps(steps)
        except Exception:  # pragma: no cover - defensive
            return None

    def _coverage_object(self) -> Optional[Any]:
        """The coverage object the gauge chart reads: POST EdgeCoverage, else PRE preflight."""
        if self.post is not None and getattr(self.post, "coverage", None) is not None:
            return self.post.coverage
        if self.pre is not None:
            return getattr(self.pre, "preflight_coverage", None)
        return None


# --- module-private string helpers -------------------------------------------


def _cell(value: Any) -> str:
    """Make a value safe inside a Markdown pipe-table cell: ``-`` for empty, escape ``|``."""
    if value is None or value == "":
        return "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _ascii(text: str) -> str:
    """Down-map text to Latin-1 for fpdf2's core fonts, replacing any stray glyph.

    fpdf2's built-in Helvetica is Latin-1 only. The report content is ASCII by
    construction, but a record id or a reason string could carry a non-Latin-1
    character; this keeps ``to_pdf`` from raising on it.
    """
    return text.encode("latin-1", "replace").decode("latin-1")


def _epw(pdf: Any) -> float:
    """The effective printable width: page width minus both margins.

    fpdf2's ``multi_cell(w=0, ...)`` derives its width from the current cursor x,
    which is zero (and then raises) when the cursor sits at the right margin after a
    full-width ``cell``. Passing an explicit printable width sidesteps that, so every
    body write is robust regardless of where the prior write left the cursor.
    """
    return pdf.w - pdf.l_margin - pdf.r_margin


def _pdf_text(pdf: Any, text: str, *, h: float = 6.0) -> None:
    """Write one wrapped paragraph at the left margin with an explicit width.

    Resets x to the left margin first, then writes a ``multi_cell`` sized to the
    printable width. This is the single robust text-write path the PDF sections use,
    so a preceding full-width banner cell can never starve a later ``multi_cell`` of
    horizontal space.
    """
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(_epw(pdf), h, _ascii(text))


def _pdf_heading(pdf: Any, text: str) -> None:
    """A consistent section heading in the PDF: a bold blue line."""
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(43, 108, 176)
    _pdf_text(pdf, text, h=8)
    pdf.set_text_color(0, 0, 0)
