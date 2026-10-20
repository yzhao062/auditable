"""Rule-based configuration auditors (PRE).

Each method reads the capabilities an agent is granted and the evidence of what its task
asks for (``spec_tokens`` and ``spec_token_overrides``, see :mod:`auditable.featurize.spec`),
and names the capabilities it judges excess. No trace and no model call is involved. The
rules are keyword based, so a task text in a language the keyword lists do not cover falls
back to the least-privilege floor and over-flags.

- ``config.flag_risky_perms`` flags every write, execute, network or admin capability, the
  naive floor;
- ``config.owasp_excess_permissions`` flags a permission level above what the task's verbs
  justify (OWASP LLM06:2025; CWE-272, CWE-250);
- ``config.owasp_excess_functionality`` flags a capability, at any permission level, whose
  subject the task never mentions (OWASP LLM06:2025);
- ``config.owasp_privilege_escalation`` flags a privilege-management capability the task does
  not ask for (CWE-269; OWASP ASI Privilege Compromise);
- ``config.unrequested_high_impact`` flags a destructive, execution or outbound action the
  task does not ask for, an approximation of excessive autonomy that is not an approval-gate
  check (OWASP LLM06:2025);
- ``config.sensitive_access`` flags credential or secret access the task does not ask for
  (OWASP LLM02:2025 exposure surface);
- ``config.owasp_asi_combined`` is the union of the five rules above.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.2 (``pre_static_scanner.rule_*`` and ``pre_baselines.FlagRiskyPermsMethod``).
"""
from __future__ import annotations

from typing import Any, Callable, FrozenSet, Set, Tuple

from auditable.base import BaseConfigAuditor, Citation, ConfigFinding, MethodCard
from auditable.featurize.spec import (
    AgentConfig,
    _cap_tokens,
    _GENERIC,
    _HIGH_IMPACT_F2B,
    _levels_for_tokens,
    _PRIVILEGE_F2B,
    _requested_bases,
    _SENSITIVE_F2B,
    _singular,
    as_agent_config,
)
from auditable.methods import register

_CATCHBENCH = Citation(key="CatchBench (2026)", title="CatchBench: When Can an Agent Failure Be Caught?",
                       arxiv="2608.22808")
_OWASP_LLM06 = Citation(key="OWASP LLM06:2025", title="OWASP Top 10 for LLM Applications 2025: LLM06 Excessive Agency",
                        url="https://genai.owasp.org/llmrisk/llm062025-excessive-agency/")
_OWASP_ASI = Citation(key="OWASP Agentic AI threats", title="Agentic AI: Threats and Mitigations (OWASP)",
                      url="https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/")
_CWE_250 = Citation(key="CWE-250", title="Execution with Unnecessary Privileges",
                    url="https://cwe.mitre.org/data/definitions/250.html")
_CWE_269 = Citation(key="CWE-269", title="Improper Privilege Management",
                    url="https://cwe.mitre.org/data/definitions/269.html")
_CWE_272 = Citation(key="CWE-272", title="Least Privilege Violation",
                    url="https://cwe.mitre.org/data/definitions/272.html")

_RISKY_PERMISSION_LEVELS = frozenset({"write", "execute", "network", "admin"})

Rule = Callable[[AgentConfig], Set[str]]


def _override_values(config: AgentConfig, key: str) -> Set[str]:
    return set(config.spec_token_overrides.get(key, ()))


def _allowed_levels(config: AgentConfig) -> Set[str]:
    allowed = _levels_for_tokens(set(config.spec_tokens))
    allowed.update(_override_values(config, "permission_levels"))
    return allowed


def risky_permissions(config: AgentConfig) -> Set[str]:
    """Capabilities whose permission level is write, execute, network or admin."""
    return {c["name"] for c in config.declared_capabilities
            if c.get("permission_level") in _RISKY_PERMISSION_LEVELS}


def excess_permissions(config: AgentConfig) -> Set[str]:
    """OWASP LLM06 (permissions) / CWE-272 / CWE-250: level above what the task's verbs justify."""
    allowed = _allowed_levels(config)
    return {c["name"] for c in config.declared_capabilities if c.get("permission_level") not in allowed}


def excess_functionality(config: AgentConfig) -> Set[str]:
    """OWASP LLM06 (functionality): a capability, at ANY permission level, whose subject is
    unrelated to the task. Overlap on a generic word (file, tool, create) does not count as
    related; a capability whose tokens are all generic is left unassessed rather than flagged.
    Relatedness uses singular-normalized content tokens (reports matches report) with no
    lossy stemming."""
    spec_content = {_singular(t) for t in set(config.spec_tokens)} - _GENERIC
    related_overrides = _override_values(config, "functionality_capabilities")
    flagged = set()
    for c in config.declared_capabilities:
        cap_content = {_singular(t) for t in _cap_tokens(c)} - _GENERIC
        if cap_content and c["name"] not in related_overrides and not (cap_content & spec_content):
            flagged.add(c["name"])
    return flagged


def _flag_unrequested(config: AgentConfig, forms_to_base: dict, override_key: str) -> Set[str]:
    """Flag a capability carrying a class token the task never asks for. A cap token counts if it
    is an inflected form of a class keyword; the task "requests" a keyword if any of its inflected
    forms appears in the spec. Flags the unrequested REMAINDER, so ``send_and_delete_email`` is
    flagged for the unrequested ``delete`` even when the task asks to ``send``."""
    requested = _requested_bases(set(config.spec_tokens), forms_to_base)
    requested.update(_override_values(config, override_key))
    flagged = set()
    for c in config.declared_capabilities:
        cap_bases = {forms_to_base[t] for t in _cap_tokens(c) if t in forms_to_base}
        if cap_bases - requested:
            flagged.add(c["name"])
    return flagged


def privilege_escalation(config: AgentConfig) -> Set[str]:
    """CWE-269 / OWASP ASI Privilege Compromise: a privilege-management capability not requested."""
    return _flag_unrequested(config, _PRIVILEGE_F2B, "privilege_bases")


def unrequested_high_impact(config: AgentConfig) -> Set[str]:
    """OWASP LLM06 autonomy driver, approximated: a high-impact action the task never asks for.
    NOT an approval-gate check (the schema has no approval field)."""
    return _flag_unrequested(config, _HIGH_IMPACT_F2B, "high_impact_bases")


def sensitive_access(config: AgentConfig) -> Set[str]:
    """OWASP LLM02 exposure surface: credential or secret access the task does not ask for."""
    return _flag_unrequested(config, _SENSITIVE_F2B, "sensitive_bases")


# Rule identifier -> rule, in the order the combined scanner takes their union.
_STANDARD_RULES: Tuple[Tuple[str, Rule], ...] = (
    ("owasp_excess_permissions", excess_permissions),
    ("owasp_excess_functionality", excess_functionality),
    ("owasp_privilege_escalation", privilege_escalation),
    ("unrequested_high_impact", unrequested_high_impact),
    ("sensitive_access", sensitive_access),
)

_SCANNER_FIELDS: FrozenSet[str] = frozenset({"declared_capabilities", "spec_tokens", "spec_token_overrides"})


def _card(method_id: str, name: str, role: str, summary: str, cite: Tuple[Citation, ...],
          requires: FrozenSet[str] = _SCANNER_FIELDS) -> MethodCard:
    return MethodCard(method_id=method_id, name=name, family="config", fit_scope="stateless",
                      supervised=False, output_kind="finding", fidelity="original", requires=requires,
                      cite=cite, board_aliases=(method_id.split(".", 1)[1],), benchmark_role=role,
                      summary=summary)


class _RuleAuditor(BaseConfigAuditor):
    """Runs its rules over one configuration. ``audit`` takes an :class:`AgentConfig` or a
    mapping with the same keys; ``excess`` is the union of the rules' flagged capabilities and
    ``rules`` lists the rules that flagged at least one."""

    _rules: Tuple[Tuple[str, Rule], ...] = ()

    def audit(self, config: Any) -> ConfigFinding:
        cfg = as_agent_config(config)
        excess: Set[str] = set()
        fired = []
        by_rule = {}
        for rule_id, rule in self._rules:
            flagged = rule(cfg)
            if flagged:
                fired.append(rule_id)
                by_rule[rule_id] = frozenset(flagged)
                excess |= flagged
        return ConfigFinding(flag=bool(excess), excess=frozenset(excess), rules=tuple(fired),
                             detail={"by_rule": by_rule})


@register
class FlagRiskyPerms(_RuleAuditor):
    """Flag every capability with write, execute, network or admin permission, whatever the
    task asks for. The naive floor the standards-based scanners are compared with."""

    _rules = (("flag_risky_perms", risky_permissions),)
    card = _card("config.flag_risky_perms", "Flag risky permission levels", "baseline",
                 "Every write, execute, network or admin capability is excess.", (_CATCHBENCH,),
                 requires=frozenset({"declared_capabilities"}))


@register
class ExcessPermissions(_RuleAuditor):
    """A capability whose permission level is above what the verbs in the task justify."""

    _rules = (("owasp_excess_permissions", excess_permissions),)
    card = _card("config.owasp_excess_permissions", "Excess permission level", "baseline",
                 "Permission level above what the task's verbs imply (OWASP LLM06, least privilege).",
                 (_CATCHBENCH, _OWASP_LLM06, _CWE_272, _CWE_250))


@register
class ExcessFunctionality(_RuleAuditor):
    """A capability, at any permission level, whose subject the task never mentions."""

    _rules = (("owasp_excess_functionality", excess_functionality),)
    card = _card("config.owasp_excess_functionality", "Excess functionality", "baseline",
                 "Capability whose subject is unrelated to the task (OWASP LLM06, excessive functionality).",
                 (_CATCHBENCH, _OWASP_LLM06))


@register
class PrivilegeEscalation(_RuleAuditor):
    """A privilege-management capability (grant, role, policy, sudo) the task does not ask for."""

    _rules = (("owasp_privilege_escalation", privilege_escalation),)
    card = _card("config.owasp_privilege_escalation", "Unrequested privilege management", "baseline",
                 "Privilege-management capability the task does not request (CWE-269, OWASP ASI).",
                 (_CATCHBENCH, _OWASP_ASI, _CWE_269))


@register
class UnrequestedHighImpact(_RuleAuditor):
    """A destructive, execution or outbound action the task never asks for. It approximates
    excessive autonomy and does not check for an approval gate."""

    _rules = (("unrequested_high_impact", unrequested_high_impact),)
    card = _card("config.unrequested_high_impact", "Unrequested high-impact action", "baseline",
                 "Destructive, execution or outbound action the task never asks for (OWASP LLM06, autonomy).",
                 (_CATCHBENCH, _OWASP_LLM06))


@register
class SensitiveAccess(_RuleAuditor):
    """Credential or secret access the task does not ask for."""

    _rules = (("sensitive_access", sensitive_access),)
    card = _card("config.sensitive_access", "Unrequested sensitive access", "baseline",
                 "Credential or secret access the task does not ask for (OWASP LLM02, exposure surface).",
                 (_CATCHBENCH,))


@register
class OwaspAsiCombined(_RuleAuditor):
    """The union of the five standards-based rules, the comprehensive static scanner."""

    _rules = _STANDARD_RULES
    card = _card("config.owasp_asi_combined", "Combined OWASP and CWE scanner", "baseline",
                 "Union of the permission, functionality, privilege, high-impact and sensitive-access rules.",
                 (_CATCHBENCH, _OWASP_LLM06, _OWASP_ASI))
