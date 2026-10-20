"""Agent-configuration features: the declared capabilities, the task text, and the
distributable token evidence the PRE rules read.

A configuration audit reads two things, the capabilities an agent is granted and what the
task or role asks for. The task text can carry names, addresses and paths, so the rules do
not read it. They read ``spec_tokens`` and ``spec_token_overrides``, sufficient statistics
derived from the text by :func:`derive_spec_features` after direct identifiers are scrubbed.
:class:`AgentConfig` holds all of these, and derives the tokens itself when it is given
only the raw task text.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.2 (``pre_static_scanner.derive_spec_features`` and the tokenizer and keyword tables it
reads). The module uses the standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Set, Tuple

_LOW_RISK_LEVELS = {"read", "unknown"}

# Verb families -> the permission level a task implies when a matching verb appears in its spec.
# "read" and "unknown" are always allowed as levels (the least-privilege floor for the permissions
# rule); unnecessary read/unknown capabilities are still caught by the functionality rule.
_VERB_LEVELS: Dict[str, Tuple[str, ...]] = {
    "network": (
        "fetch", "http", "https", "url", "download", "request", "scrape", "browse", "crawl",
        "search", "query",
    ),
    "write": (
        "write", "save", "edit", "create", "update", "delete", "modify", "store", "insert",
        "append", "remove", "generate", "draft", "compose", "upload", "export", "publish",
    ),
    "execute": (
        "run", "execute", "shell", "compile", "build", "deploy", "install", "calculate",
        "compute", "simulate",
    ),
    "admin": (
        "administrate", "configure", "provision", "grant", "revoke", "sudo",
    ),
}

# High-impact / irreversible ACTION tokens (OWASP LLM06 autonomy driver, approximated): destructive,
# execution, and outbound actions. Without an approval-gate field this is an "unrequested high-impact
# action" heuristic, not a true autonomy check.
_HIGH_IMPACT_ACTION: FrozenSet[str] = frozenset({
    "delete", "drop", "remove", "destroy", "truncate", "purge", "wipe", "terminate",
    "shutdown", "reboot", "rm", "exec", "execute", "shell", "bash", "deploy", "publish",
    "release", "install", "migrate", "transfer", "pay", "payment", "wire", "withdraw",
    "send", "email", "post", "tweet", "message", "upload",
})

# Privilege-management tokens (CWE-269; OWASP ASI Privilege Compromise threat): capabilities that
# grant, manage, or escalate privilege.
_PRIVILEGE_MGMT: FrozenSet[str] = frozenset({
    "grant", "revoke", "admin", "administrate", "role", "policy", "iam", "permission",
    "provision", "root", "sudo", "escalate", "privilege",
})

# Sensitive-resource tokens (OWASP LLM02 exposure surface): credential and secret access.
_SENSITIVE: FrozenSet[str] = frozenset({
    "credential", "secret", "password", "passwd", "apikey",
    "token", "ssh", "privatekey", "keypair", "vault", "keychain", "key",
})

# Generic tokens that do NOT establish that a capability is related to a task's subject. Overlap on
# one of these must not clear the functionality rule (create, file, tool, data, and so on).
_GENERIC: FrozenSet[str] = frozenset({
    "tool", "tools", "api", "apis", "service", "services", "function", "functions", "agent",
    "helper", "util", "utils", "utility", "data", "info", "information", "get", "set", "use",
    "run", "the", "and", "for", "with", "create", "read", "write", "list", "file", "files",
    "manager", "client", "handler", "system", "task", "action", "execute", "process", "request",
    "call", "access", "value", "input", "output", "result", "item", "object", "content",
})

# Short security-relevant tokens kept even under the length filter.
_KEEP_SHORT: FrozenSet[str] = frozenset({"rm", "db", "fs", "s3", "ls", "cp", "mv", "iam", "ssh"})

# Known irregular inflections, keyword base -> extra forms, added forward (never merged backward).
_IRREGULAR_FORMS: Dict[str, Set[str]] = {
    "run": {"ran"}, "write": {"wrote", "written"}, "send": {"sent"}, "build": {"built"},
    "pay": {"paid"}, "withdraw": {"withdrew", "withdrawn"},
}

# Keywords that take verb inflection (-ed / -ing). Everything else in a class is treated as a noun
# and takes only a plural, so a noun token like ``secret`` or ``vault`` never generates ``secretes``
# / ``vaulting`` and cannot be cleared by an unrelated gland or gymnastics spec.
_VERBS: FrozenSet[str] = frozenset(
    {v for kws in _VERB_LEVELS.values() for v in kws}
    | {
        "delete", "drop", "remove", "destroy", "truncate", "purge", "wipe", "terminate",
        "shutdown", "reboot", "exec", "deploy", "publish", "release", "migrate", "transfer",
        "pay", "wire", "withdraw", "send", "post", "tweet", "upload", "email", "message",
        "grant", "revoke", "administrate", "provision", "escalate",
    }
)
_SIBILANTS = ("s", "x", "z", "ch", "sh")


def _is_cvc(b: str) -> bool:
    """True for a consonant-vowel-consonant final (run, transfer): the final consonant doubles
    before -ed / -ing. Excludes final w/x/y and roots already ending in a double (install, shell)."""
    return len(b) >= 3 and b[-1] not in "aeiouwxy" and b[-2] in "aeiou" and b[-3] not in "aeiou"


def _plural(b: str) -> str:
    if b.endswith(_SIBILANTS):
        return b + "es"                       # search -> searches (not run -> runes)
    if b.endswith("y") and b[-2] not in "aeiou":
        return b[:-1] + "ies"                 # query -> queries (consonant + y)
    return b + "s"                            # run -> runs; deploy -> deploys (vowel + y)


def _inflect(base: str) -> Set[str]:
    """Forward inflected forms of one keyword. Verbs get the plural plus -ed / -ing (drop-e, CVC
    doubling, consonant-y -> -ied / -ying) and any known irregulars; nouns get only the plural. Only
    KNOWN keywords are expanded, so matching a spec token against these forms by exact membership
    cannot merge an unrelated word into a keyword."""
    b = base.lower()
    if len(b) < 3:
        return {b}                            # too short to inflect (rm, s3, iam, ssh): literal
    forms = {b, _plural(b)}
    if b in _VERBS:
        if b.endswith("e"):
            forms.update({b + "d", b[:-1] + "ing"})            # create -> created, creating
        elif b.endswith("y") and b[-2] not in "aeiou":
            forms.update({b[:-1] + "ied", b + "ing"})          # query -> queried, querying
        else:
            forms.update({b + "ed", b + "ing"})               # deploy -> deployed, deploying
            if _is_cvc(b):
                forms.update({b + b[-1] + "ed", b + b[-1] + "ing"})  # transfer -> transferred
        forms |= _IRREGULAR_FORMS.get(b, set())
    return forms


def _forms_to_base(words) -> Dict[str, str]:
    """Reverse map: every inflected form -> its base keyword. Bases are visited in a fixed (length,
    then lexicographic) order and the first base wins a shared form, so the map is deterministic
    across hash seeds."""
    out: Dict[str, str] = {}
    for base in sorted(words, key=lambda w: (len(w), w)):
        for form in _inflect(base):
            out.setdefault(form, base)
    return out


_VERB_LEVEL_FORMS: Dict[str, FrozenSet[str]] = {
    level: frozenset().union(*(_inflect(k) for k in kws)) for level, kws in _VERB_LEVELS.items()
}
_HIGH_IMPACT_F2B = _forms_to_base(_HIGH_IMPACT_ACTION)
_PRIVILEGE_F2B = _forms_to_base(_PRIVILEGE_MGMT)
_SENSITIVE_F2B = _forms_to_base(_SENSITIVE)

_FIXED_SPEC_FORMS: FrozenSet[str] = frozenset().union(
    *_VERB_LEVEL_FORMS.values(),
    _HIGH_IMPACT_F2B,
    _PRIVILEGE_F2B,
    _SENSITIVE_F2B,
)

_EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
)
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_.-]{1,38}")
_PHONE_RE = re.compile(r"(?<![\w])(?:\+?\d{1,3}[ .()-]*)?(?:\d[ .()-]*){7,14}\d(?![\w])")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![\w])(?:[A-Z]:\\|\\\\)[^\s<>\"'`]+")
_POSIX_PATH_RE = re.compile(
    r"(?<![\w])(?:~?/|(?:\.{1,2}/))[A-Za-z0-9._~+%-]+(?:/[A-Za-z0-9._~+%-]+)+/?"
)
_RELATIVE_PATH_RE = re.compile(
    r"(?<![\w])(?:[A-Za-z0-9._~+%-]+/)+(?:[A-Za-z0-9._~+%-]*\.[A-Za-z0-9._~-]+)(?![\w])"
)

SPEC_OVERRIDE_KEYS = frozenset({
    "permission_levels",
    "functionality_capabilities",
    "privilege_bases",
    "high_impact_bases",
    "sensitive_bases",
})


def _tokens(text: str) -> Set[str]:
    """Lowercased token set. Splits snake/kebab and camelCase INCLUDING acronym boundaries
    (``IAMRole`` -> ``iam``, ``role``; ``SSHPrivateKey`` -> ``ssh``, ``private``, ``key``). Keeps
    tokens of length 3+, plus a small set of short security tokens (``rm``, ``s3``, ...)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    out = set()
    for part in re.split(r"[^A-Za-z0-9]+", spaced):
        low = part.lower()
        if len(low) >= 3 or low in _KEEP_SHORT:
            out.add(low)
    return out


def scrub_spec_identifiers(text: str) -> str:
    """Remove direct identifiers (email addresses, URLs, file paths, handles, phone numbers)
    before the token evidence is built, so the tokens can be shared without the text."""
    replaced = text or ""
    for pattern in (
        _EMAIL_RE,
        _URL_RE,
        _WINDOWS_PATH_RE,
        _POSIX_PATH_RE,
        _RELATIVE_PATH_RE,
        _HANDLE_RE,
        _PHONE_RE,
    ):
        replaced = pattern.sub(" ", replaced)
    return replaced


def _cap_tokens(cap: Mapping[str, Any]) -> Set[str]:
    return _tokens(cap.get("name", "")) | _tokens(cap.get("type", ""))


def _singular(token: str) -> str:
    """Strip a single trailing plural -s (collision-safe): reports -> report, roles -> role. Not -es
    and not short words, so notes -> note (not "not") and news stays news."""
    if len(token) >= 5 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _requested_bases(tokens: Set[str], forms_to_base: Mapping[str, str]) -> Set[str]:
    return {forms_to_base[token] for token in tokens if token in forms_to_base}


def _levels_for_tokens(tokens: Set[str]) -> Set[str]:
    allowed = set(_LOW_RISK_LEVELS)
    for level, forms in _VERB_LEVEL_FORMS.items():
        if tokens & forms:
            allowed.add(level)
    return allowed


def derive_spec_features(spec: str, caps: Sequence[Mapping[str, Any]]) -> Dict[str, object]:
    """Build the distributable token evidence from the raw task text of one configuration.

    Tokens are taken only from scrubbed text and only when a rule can consult them for this
    capability roster. A direct identifier can contain a meaningful rule word. Its effect is
    kept as an enum, a fixed keyword base, or a declared capability name, never as an
    identifier token. This keeps scoring from the tokens equal to scoring from the text while
    removing the text. Returns ``{"spec_tokens": [...], "spec_token_overrides": {...}}``, both
    sorted.
    """
    raw_tokens = _tokens(spec)
    scrubbed_tokens = _tokens(scrub_spec_identifiers(spec))
    cap_content_by_name = {
        cap["name"]: {_singular(token) for token in _cap_tokens(cap)} - _GENERIC for cap in caps
    }
    reachable_content = set().union(*cap_content_by_name.values()) if cap_content_by_name else set()
    safe_tokens = {
        token
        for token in scrubbed_tokens
        if token in _FIXED_SPEC_FORMS or _singular(token) in reachable_content
    }

    raw_content = {_singular(token) for token in raw_tokens} - _GENERIC
    safe_content = {_singular(token) for token in safe_tokens} - _GENERIC
    raw_related = {
        name for name, content in cap_content_by_name.items() if content and content & raw_content
    }
    safe_related = {
        name for name, content in cap_content_by_name.items() if content and content & safe_content
    }

    cap_levels = {cap.get("permission_level") for cap in caps}
    cap_tokens = set().union(*(_cap_tokens(cap) for cap in caps)) if caps else set()
    cap_privilege_bases = _requested_bases(cap_tokens, _PRIVILEGE_F2B)
    cap_high_impact_bases = _requested_bases(cap_tokens, _HIGH_IMPACT_F2B)
    cap_sensitive_bases = _requested_bases(cap_tokens, _SENSITIVE_F2B)

    override_sets = {
        "permission_levels": (
            _levels_for_tokens(raw_tokens) - _levels_for_tokens(safe_tokens)
        ) & cap_levels,
        "functionality_capabilities": raw_related - safe_related,
        "privilege_bases": (
            _requested_bases(raw_tokens, _PRIVILEGE_F2B)
            - _requested_bases(safe_tokens, _PRIVILEGE_F2B)
        ) & cap_privilege_bases,
        "high_impact_bases": (
            _requested_bases(raw_tokens, _HIGH_IMPACT_F2B)
            - _requested_bases(safe_tokens, _HIGH_IMPACT_F2B)
        ) & cap_high_impact_bases,
        "sensitive_bases": (
            _requested_bases(raw_tokens, _SENSITIVE_F2B)
            - _requested_bases(safe_tokens, _SENSITIVE_F2B)
        ) & cap_sensitive_bases,
    }
    overrides = {key: sorted(values) for key, values in override_sets.items() if values}
    return {"spec_tokens": sorted(safe_tokens), "spec_token_overrides": overrides}


def _string_tuple(values: Any, what: str) -> Tuple[str, ...]:
    if isinstance(values, str) or not all(isinstance(v, str) for v in values):
        raise TypeError(f"{what} must be a list of strings")
    return tuple(values)


@dataclass(frozen=True)
class AgentConfig:
    """One agent configuration as the PRE methods read it.

    ``declared_capabilities`` is a sequence of mappings with ``name``, ``type`` and
    ``permission_level`` (``read``, ``write``, ``execute``, ``network``, ``admin`` or
    ``unknown``). ``spec_tokens`` and ``spec_token_overrides`` are the distributable evidence
    :func:`derive_spec_features` builds from the task text. Give ``task_or_role_spec`` (the
    raw text) without tokens and the tokens are derived on construction. Give tokens without
    the text to audit a configuration whose text was removed. ``instance_id`` identifies the
    configuration for replayed judgments. The text is also what an LLM judge reads.
    """

    declared_capabilities: Sequence[Mapping[str, Any]]
    spec_tokens: Optional[Sequence[str]] = None
    spec_token_overrides: Optional[Mapping[str, Sequence[str]]] = None
    task_or_role_spec: Optional[str] = None
    instance_id: Optional[str] = None

    def __post_init__(self) -> None:
        caps = []
        for cap in self.declared_capabilities:
            if not isinstance(cap, Mapping) or "name" not in cap:
                raise TypeError("each declared capability must be a mapping with a 'name'")
            caps.append(dict(cap))
        if self.task_or_role_spec is not None and not isinstance(self.task_or_role_spec, str):
            raise TypeError("task_or_role_spec must be a string")
        tokens, overrides = self.spec_tokens, self.spec_token_overrides
        if tokens is None and self.task_or_role_spec is not None:
            derived = derive_spec_features(self.task_or_role_spec, caps)
            tokens = derived["spec_tokens"]
            if overrides is None:
                overrides = derived["spec_token_overrides"]
        tokens = _string_tuple(tokens if tokens is not None else (), "spec_tokens")
        overrides = overrides if overrides is not None else {}
        if set(overrides) - SPEC_OVERRIDE_KEYS:
            raise TypeError("spec_token_overrides has invalid keys")
        frozen = {key: _string_tuple(values, f"spec_token_overrides.{key}") for key, values in overrides.items()}
        object.__setattr__(self, "declared_capabilities", tuple(caps))
        object.__setattr__(self, "spec_tokens", tokens)
        object.__setattr__(self, "spec_token_overrides", frozen)

    @classmethod
    def from_view(cls, view: Mapping[str, Any]) -> "AgentConfig":
        """Build from a CatchBench ``method_view()`` entry, or any mapping with the same keys.
        A key the config does not use (``source``) is ignored."""
        return cls(declared_capabilities=view["declared_capabilities"],
                   spec_tokens=view.get("spec_tokens"),
                   spec_token_overrides=view.get("spec_token_overrides"),
                   task_or_role_spec=view.get("task_or_role_spec"),
                   instance_id=view.get("instance_id"))

    @property
    def capability_names(self) -> Tuple[str, ...]:
        return tuple(cap["name"] for cap in self.declared_capabilities)


def as_agent_config(config: Any) -> AgentConfig:
    """``config`` itself when it is an :class:`AgentConfig`, or the one built from a mapping."""
    if isinstance(config, AgentConfig):
        return config
    if isinstance(config, Mapping):
        return AgentConfig.from_view(config)
    raise TypeError(f"expected an AgentConfig or a mapping, got {type(config).__name__}")


__all__ = ["AgentConfig", "SPEC_OVERRIDE_KEYS", "as_agent_config", "derive_spec_features",
           "scrub_spec_identifiers"]
