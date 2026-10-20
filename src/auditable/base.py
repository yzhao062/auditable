"""Base classes and method cards for the auditing-algorithm library.

The library groups auditing methods into four families by the question they answer and
the grain of their output:

- ``config`` (PRE): what is risky in a declared agent configuration, before a run;
- ``run`` (POST): did this finished run fail or violate, one score per run;
- ``step`` (POST): which step (or agent) is the decisive mistake, one score per step;
- ``online`` (LIVE): should a running agent raise an alarm now, one causal score per step.

How a method is fitted is a separate axis, declared on its :class:`MethodCard`:

- ``stateless``: a rule; ``fit`` is a no-op;
- ``inductive``: ``fit`` stores the training-time scaler and model, and later scoring of
  unseen inputs does not refit. Inductive methods also offer a whole-batch entry point that
  returns training-time scores (``fit_score`` for run detectors), because some reference
  implementations score the fitted data differently from new data (PyOD's KNN and LOF
  exclude a point's own neighbour in ``decision_scores_``);
- ``transductive``: the method fits and scores one batch together and makes no promise
  about unseen inputs (PyGOD's graph autoencoders, GUARDIAN). Only the whole-batch entry
  points are available.

There is deliberately no cross-family ``assess()``: a family fixes its inputs and outputs,
and the card says how the method was fitted, where it comes from, and what it needs.

This module uses the standard library only, so the registry and the card listing work on
the core install; numerical methods import their dependencies inside their own modules.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

FAMILIES = ("config", "run", "step", "online")
INFORMATION_STATES = ("PRE", "LIVE", "POST")
FIT_SCOPES = ("stateless", "inductive", "transductive")
FIDELITIES = ("faithful", "simplified", "wrapper", "original")
OUTPUT_KINDS = ("finding", "score", "ranking", "single")
BENCHMARK_ROLES = ("baseline", "control", "oracle")

_FAMILY_STATE = {"config": "PRE", "run": "POST", "step": "POST", "online": "LIVE"}


@dataclass(frozen=True)
class Citation:
    """Where a method comes from. ``key`` is a short human label such as
    ``"GUARDIAN (Zhou et al., 2025)"``; ``arxiv`` is the bare identifier."""

    key: str
    title: str = ""
    arxiv: str = ""
    url: str = ""


@dataclass(frozen=True)
class MethodCard:
    """What a method is, independent of any benchmark.

    ``method_id`` is globally unique and starts with the family name
    (``"step.position_supervised"``). ``board_aliases`` lists the CatchBench board row
    labels the method reproduces; ``benchmark_role`` records the role it plays on that
    board (``baseline``, ``control`` or ``oracle``). The role describes the board and does
    not decide whether a method belongs in the library.

    ``requires`` names the trace or configuration fields the method reads, for example
    ``"deps"``, ``"step_text"``, ``"exec_graph"``, ``"spec_tokens"``. ``extras`` names the
    pip extras the method needs beyond the core install.
    """

    method_id: str
    name: str
    family: str
    fit_scope: str
    supervised: bool
    output_kind: str
    fidelity: str
    requires: FrozenSet[str] = frozenset()
    information_state: str = ""
    cite: Tuple[Citation, ...] = ()
    extras: Tuple[str, ...] = ()
    board_aliases: Tuple[str, ...] = ()
    benchmark_role: Optional[str] = None
    summary: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if not self.method_id.startswith(self.family + "."):
            raise ValueError(f"method_id {self.method_id!r} must start with {self.family + '.'!r}")
        if not self.information_state:
            object.__setattr__(self, "information_state", _FAMILY_STATE[self.family])
        if self.information_state not in INFORMATION_STATES:
            raise ValueError(f"information_state must be one of {INFORMATION_STATES}")
        if self.fit_scope not in FIT_SCOPES:
            raise ValueError(f"fit_scope must be one of {FIT_SCOPES}, got {self.fit_scope!r}")
        if self.fidelity not in FIDELITIES:
            raise ValueError(f"fidelity must be one of {FIDELITIES}, got {self.fidelity!r}")
        if self.output_kind not in OUTPUT_KINDS:
            raise ValueError(f"output_kind must be one of {OUTPUT_KINDS}, got {self.output_kind!r}")
        if self.benchmark_role is not None and self.benchmark_role not in BENCHMARK_ROLES:
            raise ValueError(f"benchmark_role must be one of {BENCHMARK_ROLES} or None")
        object.__setattr__(self, "requires", frozenset(self.requires))


class NotSupported(TypeError):
    """Raised when a method is asked for an entry point its fit scope does not offer,
    such as ``decision_function`` on a transductive detector."""


class BaseAuditingMethod:
    """Parameter handling shared by every family, in the sklearn and PyOD style.

    A subclass stores its constructor arguments under the same names and sets the class
    attribute ``card``. ``get_params`` reads the constructor signature, so
    ``sklearn.base.clone`` works on any method without importing sklearn here.
    """

    card: MethodCard

    @classmethod
    def _param_names(cls) -> List[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        params = inspect.signature(init).parameters.values()
        return sorted(p.name for p in params
                      if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> "BaseAuditingMethod":
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"invalid parameter {name!r} for {type(self).__name__}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _require_scope(self, *allowed: str, entry: str) -> None:
        scope = type(self).card.fit_scope
        if scope not in allowed:
            raise NotSupported(
                f"{type(self).__name__} is {scope}; {entry} is available only for "
                f"{' or '.join(allowed)} methods"
            )


@dataclass
class ConfigFinding:
    """The result of auditing one configuration.

    ``flag`` is True when the method calls the configuration over-privileged, False when
    it does not, and None when it abstains (for example, an LLM judge with no parsable
    answer). ``excess`` holds the capabilities the method judges unnecessary, and
    ``rules`` the identifiers of the rules that fired.
    """

    flag: Optional[bool]
    excess: FrozenSet[str] = frozenset()
    rules: Tuple[str, ...] = ()
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.flag is None


class BaseConfigAuditor(BaseAuditingMethod):
    """PRE family: audit a declared configuration before a run."""

    def fit(self, configs: Sequence[Any] = (), y: Any = None) -> "BaseConfigAuditor":
        return self

    def audit(self, config: Any) -> ConfigFinding:
        raise NotImplementedError

    def audit_many(self, configs: Sequence[Any]) -> List[ConfigFinding]:
        return [self.audit(c) for c in configs]


class BaseRunDetector(BaseAuditingMethod):
    """POST family, run level: one score per finished run, higher means more likely
    failed.

    Inductive detectors implement ``fit`` and ``decision_function``; after ``fit`` they
    expose ``decision_scores_``, the training-time scores of the fitted batch, which
    ``fit_score`` returns. Transductive detectors implement ``fit_score`` only.
    ``X`` is a list of traces or, where the card says so, a feature matrix.
    """

    decision_scores_: Any

    def fit(self, X: Any, y: Any = None) -> "BaseRunDetector":
        self._require_scope("inductive", "stateless", entry="fit")
        raise NotImplementedError

    def decision_function(self, X: Any) -> Any:
        self._require_scope("inductive", "stateless", entry="decision_function")
        raise NotImplementedError

    def fit_score(self, X: Any, y: Any = None) -> Any:
        """Fit on ``X`` and return the scores of ``X`` itself, in input order."""
        self.fit(X, y)
        return self.decision_scores_


class BaseStepLocalizer(BaseAuditingMethod):
    """POST family, step level: one score per step, higher means more likely the
    decisive mistake.

    ``step_scores_batch`` scores a list of traces and returns one array per trace in step
    order. Supervised inductive methods score the batch in one call, so results match a
    reference implementation that predicts a whole fold at once; ``step_scores`` is the
    single-trace convenience and may differ from a batched call in the last floating-point
    place. Transductive methods offer ``fit_step_scores`` only. A method with
    ``output_kind == "single"`` predicts one step per trace through ``predict_step``.
    Supervised methods take ``y`` with one entry per trace: the ``idx`` of its decisive
    mistake step.
    """

    def fit(self, traces: Sequence[Any] = (), y: Any = None) -> "BaseStepLocalizer":
        self._require_scope("inductive", "stateless", entry="fit")
        return self

    def step_scores_batch(self, traces: Sequence[Any]) -> List[Any]:
        self._require_scope("inductive", "stateless", entry="step_scores_batch")
        return [self.step_scores(t) for t in traces]

    def step_scores(self, trace: Any) -> Any:
        self._require_scope("inductive", "stateless", entry="step_scores")
        if type(self).step_scores_batch is BaseStepLocalizer.step_scores_batch:
            raise NotImplementedError(f"{type(self).__name__} implements neither scoring entry point")
        return self.step_scores_batch([trace])[0]

    def fit_step_scores(self, traces: Sequence[Any], y: Any = None) -> List[Any]:
        """Fit on ``traces`` and return the step scores of the same traces, in order."""
        return self.fit(traces, y).step_scores_batch(traces)

    def rank(self, trace: Any) -> List[int]:
        """Step positions from most to least suspected; ties keep step order."""
        scores = list(self.step_scores(trace))
        return sorted(range(len(scores)), key=lambda i: -scores[i])


class BaseOnlineMonitor(BaseAuditingMethod):
    """LIVE family: a causal alarm score per arriving step.

    The score at position ``t`` uses only steps ``0..t``, so the same prefix followed by
    different suffixes yields the same scores on that prefix. ``prefix_score(trace, k)``
    is the run-level score after ``k`` steps, by default the alarm score at position
    ``k - 1``.
    """

    def fit(self, traces: Sequence[Any] = (), y: Any = None) -> "BaseOnlineMonitor":
        return self

    def step_alarm_scores(self, trace: Any) -> Any:
        raise NotImplementedError

    def prefix_score(self, trace: Any, k: int) -> float:
        if k < 1:
            raise ValueError("k must be at least 1")
        scores = self.step_alarm_scores(trace)
        if len(scores) == 0:
            return 0.0
        return float(scores[min(k, len(scores)) - 1])
