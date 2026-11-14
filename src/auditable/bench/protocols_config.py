"""Evaluation protocols for PRE configuration auditing on the CatchBench board.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.3,
``src/catchbench/pre.py`` (``pre_score`` and ``pre_source_breakdown``) and
``src/catchbench/via_auditable/config.py`` (``RuleEntry`` and ``JudgeEntry``).

Every configuration auditor returns a :class:`~auditable.base.ConfigFinding` per configuration.
An abstaining finding (``flag is None``) leaves the configuration out of the evaluable set
instead of counting it as flagging nothing, so ``coverage`` reports the fraction of
configurations actually judged. Micro-averaged precision, recall, and F1 are computed over the
evaluable configurations overall and per logical source.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

DEFAULT_SOURCES: Tuple[str, ...] = ("crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic")


def pre_score(
    flagged: Mapping[str, Iterable[str]],
    targets: Sequence[Mapping[str, Any]],
    evaluable: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Micro-averaged precision, recall, F1, and coverage over declared capabilities.

    ``targets`` is a sequence of dicts each holding ``instance_id`` and ``excess_set``.
    Configurations outside ``evaluable`` (when given) are abstentions and leave the
    denominator rather than counting as empty predictions.
    """
    scored = [t for t in targets if evaluable is None or t["instance_id"] in evaluable]
    tp = fp = fn = 0
    for target in scored:
        gt = set(target["excess_set"])
        pred = set(flagged.get(target["instance_id"], ()))
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    coverage = len(scored) / len(targets) if targets else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
    }


def score_config_auditor(
    auditor: Any,
    configs: Sequence[Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> Dict[str, float]:
    """Score one fitted or stateless configuration auditor overall and per logical source."""
    findings = auditor.audit_many(configs)
    flagged: Dict[str, Set[str]] = {}
    abstained = False
    for cfg, finding in zip(configs, findings):
        if finding.flag is None:
            abstained = True
            continue
        flagged[cfg.instance_id] = set(finding.excess)
    evaluable = set(flagged) if abstained else None
    scores = pre_score(flagged, targets, evaluable=evaluable)

    by_source: Dict[str, list] = {}
    for target in targets:
        by_source.setdefault(target["logical_source"], []).append(target)
    for source in sources:
        subset = by_source.get(source, ())
        scores[f"f1@{source}"] = pre_score(flagged, subset, evaluable=evaluable)["f1"] if subset else 0.0
    return scores
