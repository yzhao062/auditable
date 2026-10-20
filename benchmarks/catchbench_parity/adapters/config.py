"""PRE configuration audit through the library: the rule scanners and the held-out LLM judge.

Every ``pre_over_privilege`` row except the three that stay in CatchBench (``flag_all``,
``flag_none``, ``oracle_privilege_diff``) is scored from library findings. The adapter reads
the configurations through ``views.pre_view``, builds an ``AgentConfig`` from each, takes the
excess set of the library's finding, and hands the flagged sets to CatchBench's own
``pre_score``, the evaluation shell. The judge row replays the recorded judgments of
CatchBench's held-out cache through the library's ``needed`` mapping; a configuration absent
from the cache abstains and leaves the denominator, as in the original.

The released PRE corpus carries no task text, so ``derive_spec_features`` and the LLM prompt
are checked on the one source whose raw text CatchBench ships as code: the 56 authored
synthetic configurations ``tools/pre_harvest_synthetic.py`` builds. The other harvesters need
the network, and their raw text is not distributed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from catchbench_parity import perturb, views
from catchbench_parity.manifest import check, item, sentinel

_RULE_ROWS = (
    ("flag_risky_perms", "config.flag_risky_perms"),
    ("owasp_excess_permissions", "config.owasp_excess_permissions"),
    ("owasp_excess_functionality", "config.owasp_excess_functionality"),
    ("owasp_privilege_escalation", "config.owasp_privilege_escalation"),
    ("unrequested_high_impact", "config.unrequested_high_impact"),
    ("sensitive_access", "config.sensitive_access"),
    ("owasp_asi_combined", "config.owasp_asi_combined"),
)
_JUDGE = "config.llm_needed"


def _shells(view):
    """The records ``pre_score`` reads (instance id and labels), built from the view alone."""
    return [SimpleNamespace(instance_id=c["instance_id"], labels=lab) for c, lab in zip(view.configs, view.labels)]


class _RuleEntry:
    """A ``pre_over_privilege`` row whose flagged sets are the excess sets of a library auditor."""

    def __init__(self, original, library_method: str) -> None:
        self.method_id = original.method_id
        self.supports = set(original.supports)
        self.library_method = library_method

    def _method(self):
        from auditable.methods import get_method

        return perturb.wrap(get_method(self.library_method)())

    @staticmethod
    def _flagged(configs, findings):
        return {c.instance_id: set(f.excess) for c, f in zip(configs, findings)}, None

    def evaluate(self, task):
        from auditable.featurize import AgentConfig
        from catchbench.pre import pre_score

        view = views.pre_view(task)
        configs = [AgentConfig.from_view(c) for c in view.configs]
        flagged, evaluable = self._flagged(configs, self._method().audit_many(configs))
        return pre_score(flagged, _shells(view), evaluable)


class _JudgeEntry(_RuleEntry):
    """The held-out judge row: recorded judgments replayed by instance id. An abstaining
    finding (``flag is None``) is left out of ``evaluable``."""

    def __init__(self, original) -> None:
        super().__init__(original, _JUDGE)
        self.cache_path = original.cache_path

    def _method(self):
        from auditable.methods import get_method

        with open(self.cache_path, encoding="utf-8") as f:
            rows = json.load(f)
        return perturb.wrap(get_method(_JUDGE)(needed=rows))

    @staticmethod
    def _flagged(configs, findings):
        flagged = {c.instance_id: set(f.excess) for c, f in zip(configs, findings) if f.flag is not None}
        return flagged, set(flagged)


_JUDGE_ROWS = []  # one row per committed judge cache, as CatchBench builds them


def _register():
    from catchbench.pre_baselines import _llm_judge_methods

    for alias, method_id in _RULE_ROWS:
        item(task_id="pre_over_privilege", board_alias=alias, variant="", factory="pre_methods",
             method_id=method_id, build=lambda o, m=method_id: _RuleEntry(o, m))
    for m in _llm_judge_methods():
        _JUDGE_ROWS.append(m.method_id)
        item(task_id="pre_over_privilege", board_alias=m.method_id, variant="", factory="pre_methods",
             method_id=_JUDGE, build=lambda o: _JudgeEntry(o))


_register()
# Every board caller of these cores is a row above: the rule functions run only inside
# _RuleScanner.evaluate and rule_owasp_asi, and the judge cache is read only by its evaluate.
sentinel("catchbench.pre_static_scanner:_RuleScanner.evaluate",
         "catchbench.pre_static_scanner:rule_excess_permissions",
         "catchbench.pre_static_scanner:rule_excess_functionality",
         "catchbench.pre_static_scanner:rule_privilege_escalation",
         "catchbench.pre_static_scanner:rule_unrequested_high_impact",
         "catchbench.pre_static_scanner:rule_sensitive_access",
         "catchbench.pre_static_scanner:rule_owasp_asi",
         "catchbench.pre_baselines:FlagRiskyPermsMethod.evaluate",
         "catchbench.pre_baselines:LlmJudgeNeededMethod.evaluate")


# --- per-sample comparisons ------------------------------------------------------------------

_TOOLS = {}


def _tool(name: str):
    """A script from CatchBench's ``tools/`` imported as a module, without writing bytecode
    into that repository."""
    import importlib.util
    import sys
    from pathlib import Path

    import catchbench

    if name not in _TOOLS:
        tools = Path(catchbench.__file__).resolve().parents[2] / "tools"
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(tools))
        try:
            spec = importlib.util.spec_from_file_location(name, tools / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(tools))
            sys.dont_write_bytecode = old
        _TOOLS[name] = module
    return _TOOLS[name]


def _synthetic_rows():
    """The 56 raw synthetic configurations (task text, capabilities, instance id)."""
    return _tool("pre_harvest_synthetic").build_instances()


def _original(row_label: str):
    from catchbench.pre import pre_methods

    for m in pre_methods():
        if m.method_id == row_label:
            return m
    raise KeyError(f"no CatchBench PRE method labelled {row_label!r}")


def _run_original(original, task):
    """Run the original ``evaluate`` and capture what it hands to ``pre_score``:
    ``({"flagged": {id: set}, "evaluable": set or None}, metrics)``."""
    import importlib

    module = importlib.import_module(type(original).__module__)
    real, seen = module.pre_score, {}

    def capture(flagged, instances, evaluable=None):
        seen["flagged"] = {k: set(v) for k, v in flagged.items()}
        seen["evaluable"] = None if evaluable is None else set(evaluable)
        return real(flagged, instances, evaluable)

    module.pre_score = capture
    try:
        metrics = original.evaluate(task)
    finally:
        module.pre_score = real
    return seen, metrics


def _same_json(ctx, a, b) -> bool:
    return ctx.same(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


@check(*[("pre_over_privilege", alias, "") for alias, _ in _RULE_ROWS],
       *[("pre_over_privilege", alias, "") for alias in _JUDGE_ROWS])
def check_rows(ctx, it):
    """Per configuration (all 1187): the library finding's flag and excess set against what the
    original method hands to ``pre_score``, plus the configuration's tokens and capabilities after
    ``AgentConfig.from_view``. Then the adapter's ``evaluate`` against the original's on the pooled
    task and on every per-source sub-task."""
    from auditable.featurize import AgentConfig
    from auditable.methods import get_method
    from catchbench.pre import PreOverPrivilege, _pre_logical_source

    task = ctx.task("pre_over_privilege")
    view = views.pre_view(task)
    original = _original(it.row_label)
    seen, _ = _run_original(original, task)
    judge = it.method_id == _JUDGE
    if judge:
        with open(original.cache_path, encoding="utf-8") as f:
            method = get_method(_JUDGE)(needed=json.load(f))
    else:
        method = get_method(it.method_id)()

    configs = [AgentConfig.from_view(c) for c in view.configs]
    findings = method.audit_many(configs)
    n = bad = 0
    for raw, cfg, f in zip(view.configs, configs, findings):
        n += 1
        same_cfg = (ctx.same(list(cfg.spec_tokens), raw["spec_tokens"])
                    and _same_json(ctx, {k: list(v) for k, v in cfg.spec_token_overrides.items()},
                                   raw["spec_token_overrides"])
                    and _same_json(ctx, [dict(c) for c in cfg.declared_capabilities], raw["declared_capabilities"]))
        expected = seen["flagged"].get(raw["instance_id"])
        if expected is None:  # the judge abstained on this configuration
            same_finding = judge and f.flag is None and not f.excess
        else:
            same_finding = f.flag == bool(expected) and ctx.same(sorted(f.excess), sorted(expected))
        bad += not (same_cfg and same_finding)
    parts = {"findings": {"compared": n, "mismatches": bad}}
    if judge:
        abstained = {c.instance_id for c, f in zip(configs, findings) if f.flag is not None}
        parts["evaluable"] = {"compared": 1, "mismatches": int(not ctx.same(sorted(abstained),
                                                                           sorted(seen["evaluable"])))}

    adapter = it.build(original)
    groups = {}
    for o in task.instances:
        groups.setdefault(_pre_logical_source(o.source), []).append(o)
    tasks = [("pooled", task)] + [(g, PreOverPrivilege(instances=insts)) for g, insts in groups.items()]
    n = bad = 0
    for _, t in tasks:
        n += 1
        bad += not ctx.same(adapter.evaluate(t), original.evaluate(t))
    parts["metrics"] = {"compared": n, "mismatches": bad}
    out = ctx.merge(parts)
    out["unit"] = "configurations (flag and excess set), plus pooled and per-source metrics"
    return out


_CRAFTED_SPECS = (
    "Email alice.deploy@example.com the report and fetch https://example.com/delete/records "
    "then save it to /var/data/export.csv",
    "Contact @sudo_admin or +1 (555) 123-4567 and run C:\\Users\\root\\bin\\install.exe once",
    "Summarize the quarterly reports and the vault policies; do not send anything",
    "Translate the ancient runes. Do not run anything. Compile the notes, then queried the index",
    "Resume genere: rapport de synthese, \u62a5\u544a\u5904\u7406, read-only access to docs/guide.md",
    "IAMRole SSHPrivateKey grant revoke Deleting Transferred paying withdrew",
    "",
)


@check(("pre_over_privilege", "derive_spec_features", "raw spec text"))
def check_spec_features(ctx, it):
    """``derive_spec_features`` and the auto-derivation in ``AgentConfig`` against the original on
    the 56 synthetic configurations that have raw task text. Parts: the same call on the original
    roster; the tokens CatchBench released for those configurations; every spec against every
    roster of the board (66k pairs); crafted specs with identifiers; the auto-derived config's rule
    findings against the original rules reading the raw text."""
    from auditable.featurize import AgentConfig
    from auditable.featurize import spec as lib
    from auditable.methods import get_method
    from catchbench import pre_static_scanner as ref

    task = ctx.task("pre_over_privilege")
    released = {o["instance_id"]: o for o in task.method_view()}
    rows = _synthetic_rows()
    rosters = [o["declared_capabilities"] for o in task.method_view()]
    parts = {}

    n = bad = 0
    for r in rows:
        n += 1
        bad += not ctx.same(lib.derive_spec_features(r["task_or_role_spec"], r["declared_capabilities"]),
                            ref.derive_spec_features(r["task_or_role_spec"], r["declared_capabilities"]))
    parts["roster"] = {"compared": n, "mismatches": bad}

    n = bad = 0
    for r in rows:
        derived = lib.derive_spec_features(r["task_or_role_spec"], r["declared_capabilities"])
        o = released[r["instance_id"]]
        n += 1
        bad += not (ctx.same(derived["spec_tokens"], o["spec_tokens"])
                    and _same_json(ctx, derived["spec_token_overrides"], o["spec_token_overrides"]))
    parts["released tokens"] = {"compared": n, "mismatches": bad}

    n = bad = 0
    for r in rows:
        for caps in rosters:
            n += 1
            bad += not ctx.same(lib.derive_spec_features(r["task_or_role_spec"], caps),
                                ref.derive_spec_features(r["task_or_role_spec"], caps))
    parts["spec x roster"] = {"compared": n, "mismatches": bad}

    n = bad = 0
    for spec in _CRAFTED_SPECS:
        for caps in rosters[::40] + [max(rosters, key=len)]:
            n += 1
            bad += not ctx.same(lib.derive_spec_features(spec, caps), ref.derive_spec_features(spec, caps))
    parts["crafted specs"] = {"compared": n, "mismatches": bad}

    rules = (("config.owasp_excess_permissions", ref.rule_excess_permissions),
             ("config.owasp_excess_functionality", ref.rule_excess_functionality),
             ("config.owasp_privilege_escalation", ref.rule_privilege_escalation),
             ("config.unrequested_high_impact", ref.rule_unrequested_high_impact),
             ("config.sensitive_access", ref.rule_sensitive_access),
             ("config.owasp_asi_combined", ref.rule_owasp_asi))
    n = bad = 0
    for r in rows:
        cfg = AgentConfig(declared_capabilities=r["declared_capabilities"], task_or_role_spec=r["task_or_role_spec"],
                          instance_id=r["instance_id"])
        o = released[r["instance_id"]]
        n += 1
        bad += not (ctx.same(list(cfg.spec_tokens), o["spec_tokens"])
                    and _same_json(ctx, {k: list(v) for k, v in cfg.spec_token_overrides.items()},
                                   o["spec_token_overrides"]))
        for method_id, rule in rules:
            n += 1
            found = get_method(method_id)().audit(cfg)
            text_backed = rule(r["task_or_role_spec"], r["declared_capabilities"])
            bad += not ctx.same(sorted(found.excess), sorted(text_backed))
    parts["auto-derived config"] = {"compared": n, "mismatches": bad}

    out = ctx.merge(parts)
    out["unit"] = "feature dicts and findings; raw text exists for %d of %d board configurations" % (
        len(rows), len(released))
    return out


_CRAFTED_REPLIES = (
    "Reasoning.\nNEEDED: read_file, write_file",
    "NEEDED:",
    "NEEDED: none.",
    "NEEDED: []",
    'NEEDED: ["read_file", "write_file"]',
    'NEEDED: ["read_file", 3]',
    "NEEDED: [read_file, write_file]",
    "NEEDED: 1. read_file, 2. `write_file`",
    "NEEDED: read_file, read_file",
    "NEEDED: read_file, delete_everything",
    "needed : write_file.",
    "NEEDED: read_file\nlater thoughts\nNEEDED: write_file",
    "The needed capabilities are read_file.",
    "",
)


@check(("pre_over_privilege", "llm judge prompt and parse", "fixed inputs"))
def check_llm_text(ctx, it):
    """The prompt and the parser against the originals, and the generation path end to end. Parts:
    the prompt for the 56 configurations with raw text, and its hash against the one recorded with
    the held-out judge's replies; the parser on all 1187 recorded replies and on crafted replies;
    the library judge driven by a fixed-response backend that returns each recorded reply,
    against the original method's flagged set."""
    import hashlib
    from pathlib import Path

    from auditable.featurize import AgentConfig
    from auditable.llm import FixedResponseBackend
    from auditable.methods import get_method
    from auditable.methods.config import llm as lib

    ref = _tool("pre_label_llm_judge")
    task = ctx.task("pre_over_privilege")
    declared = {o["instance_id"]: {c["name"] for c in o["declared_capabilities"]} for o in task.method_view()}
    rows = _synthetic_rows()
    original = _original("llm_judge_needed(llama-3.3-70b)")
    with open(original.cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    # the raw replies and prompt hashes the judge's cache was built from (resume record, not read by the board)
    votes_path = Path(original.cache_path).parent.parent / "llm_judge_method_votes" / Path(original.cache_path).name
    with open(votes_path, encoding="utf-8") as f:
        votes = json.load(f)
    parts = {}

    def config(r):
        return AgentConfig(declared_capabilities=r["declared_capabilities"],
                           task_or_role_spec=r["task_or_role_spec"], instance_id=r["instance_id"])

    n = bad = 0
    for r in rows:
        prompt = lib.build_prompt(config(r))
        n += 1
        bad += not ctx.same(prompt, ref.build_prompt(r))
    parts["prompt"] = {"compared": n, "mismatches": bad}

    n = bad = 0
    for r in rows:
        n += 1
        bad += not ctx.same(hashlib.sha256(lib.build_prompt(config(r)).encode("utf-8")).hexdigest(),
                            votes[r["instance_id"]]["prompt_sha256"])
    parts["recorded prompt hash"] = {"compared": n, "mismatches": bad}

    n = bad = 0
    for iid, entry in votes.items():
        n += 1
        bad += not _same_json(ctx, lib.parse_needed(entry["raw_response"], declared[iid]),
                              ref.parse_needed(entry["raw_response"], declared[iid]))
    parts["parse recorded replies"] = {"compared": n, "mismatches": bad}

    names = {"read_file", "write_file", "delete_everything"}
    n = bad = 0
    for raw in _CRAFTED_REPLIES:
        n += 1
        bad += not _same_json(ctx, lib.parse_needed(raw, names), ref.parse_needed(raw, names))
    parts["parse crafted replies"] = {"compared": n, "mismatches": bad}

    n = bad = 0
    for r in rows:
        iid = r["instance_id"]
        backend = FixedResponseBackend([votes[iid]["raw_response"]] * 2)  # a second reply for the retry
        finding = get_method(lib.LLMNeeded.card.method_id)(backend=backend, model="llama-3.3-70b").audit(config(r))
        expected = None if iid not in cache else declared[iid] - set(cache[iid])
        n += 1
        if expected is None:  # an unparsable reply: the judge asks twice, then abstains
            bad += not (finding.flag is None and backend.prompts == [ref.build_prompt(r)] * 2)
        else:
            bad += not (finding.flag == bool(expected) and ctx.same(sorted(finding.excess), sorted(expected))
                        and ctx.same(backend.prompts, [ref.build_prompt(r)]))
    parts["generation path"] = {"compared": n, "mismatches": bad}

    out = ctx.merge(parts)
    out["unit"] = "prompts, replies and findings; raw text exists for %d of %d board configurations" % (
        len(rows), len(declared))
    return out
