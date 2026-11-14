"""PRE over-privilege configurations: the 1,187-configuration multi-corpus population.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.3,
``src/catchbench/pre.py`` (``_load_data_dir``, ``_pre_logical_source``, and
``_PRE_SOURCE_ORDER``). Configurations are loaded through ``agentpile`` (dataset
``catchbench_pre``) and projected with :func:`auditable.featurize.agentpile.configs_from_agentpile`
in the board's exact order.

The four synthetic buckets (``synth_crewai``, ``synth_mcp``, ``synth_flowise``,
``synth_sweagent``) are grouped under the single logical source ``"synthetic"`` for per-source
scoring, matching ``pre_source_breakdown``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

DATASET = "catchbench_pre"

SOURCE_ORDER: Tuple[str, ...] = ("crewai", "n8n", "mcp", "injecagent", "sweagent", "synthetic")

LABEL_SOURCES: Dict[str, str] = {
    "crewai": "llm_judge",
    "n8n": "llm_judge",
    "mcp": "llm_judge",
    "injecagent": "roster_relabel",
    "sweagent": "declared_minus_used",
    "synthetic": "synthetic_inject",
}


def logical_source(source: str) -> str:
    """Map raw corpus source names to the six logical columns of the F1-by-source table."""
    return "synthetic" if source.startswith("synth") else source


def load_configs(**load_kwargs: Any) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Return ``(configs, labels, info)`` for the CatchBench PRE population in board order.

    ``configs`` is a list of :class:`~auditable.featurize.spec.AgentConfig` objects, ``labels``
    carries ``instance_id``, ``source``, ``logical_source``, ``excess_set``, ``label_source``,
    and ``minimal_reference`` per configuration, and ``info`` records per-source counts.
    """
    import agentpile as ap

    from auditable.featurize.agentpile import configs_from_agentpile

    configs: List[Any] = []
    labels: List[Dict[str, Any]] = []
    for run in ap.load(DATASET, **load_kwargs):
        (cfg,) = configs_from_agentpile(run)
        ds = run.dataset or {}
        raw_source = ds["source"]
        configs.append(cfg)
        labels.append({
            "instance_id": cfg.instance_id,
            "source": raw_source,
            "logical_source": logical_source(raw_source),
            "excess_set": list(ds["labels"]["excess_set"]),
            "label_source": ds["labels"]["label_source"],
            "minimal_reference": list(ds["minimal_reference"]),
        })
    if not configs:
        raise RuntimeError(f"agentpile returned no {DATASET} configurations")
    by_source: Dict[str, int] = {}
    for rec in labels:
        group = rec["logical_source"]
        by_source[group] = by_source.get(group, 0) + 1
    ordered = {group: by_source[group] for group in SOURCE_ORDER if group in by_source}
    for group in sorted(by_source):
        if group not in ordered:
            ordered[group] = by_source[group]
    info = {"dataset": DATASET, "n_configs": len(configs), "by_source": ordered}
    return configs, labels, info
