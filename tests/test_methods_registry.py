"""The method registry, the method card, and the family base classes.

Runs on the core install (standard library only): CI runs this file with no extras, so
nothing here may import numpy or another optional dependency.
"""
import pathlib

import pytest

import auditable
from auditable.base import (
    FAMILIES,
    BaseOnlineMonitor,
    BaseRunDetector,
    BaseStepLocalizer,
    Citation,
    ConfigFinding,
    MethodCard,
    NotSupported,
)
from auditable.methods import get_method, list_methods, method_table, register

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _card(**kw):
    base = dict(method_id="step.toy", name="toy", family="step", fit_scope="stateless", supervised=False,
                output_kind="score", fidelity="original")
    base.update(kw)
    return MethodCard(**base)


def test_registry_lists_every_family_and_unique_ids():
    cards = list_methods()
    ids = [c.method_id for c in cards]
    assert len(ids) == len(set(ids))
    assert {c.family for c in cards} == set(FAMILIES)
    for c in cards:
        assert c.method_id.startswith(c.family + ".")
        assert get_method(c.method_id).card is c
        assert c.name and c.summary, c.method_id
        assert c.cite, f"{c.method_id} has no citation"


def test_list_methods_filters():
    online = list_methods(family="online")
    assert online and all(c.family == "online" and c.information_state == "LIVE" for c in online)
    assert all(c.information_state == "PRE" for c in list_methods(information_state="PRE"))
    assert all(c.fit_scope == "transductive" for c in list_methods(fit_scope="transductive"))


def test_get_method_unknown_id_names_the_registry():
    with pytest.raises(KeyError, match="unknown method_id"):
        get_method("run.does_not_exist")


def test_top_level_exports_the_registry():
    assert auditable.list_methods is list_methods
    assert auditable.get_method is get_method
    assert auditable.MethodCard is MethodCard


@pytest.mark.parametrize("bad", [
    {"family": "trace"},
    {"method_id": "run.toy"},
    {"fit_scope": "online"},
    {"fidelity": "approximate"},
    {"output_kind": "probability"},
    {"benchmark_role": "reference"},
    {"information_state": "DURING"},
])
def test_card_rejects_invalid_fields(bad):
    with pytest.raises(ValueError):
        _card(**bad)


def test_card_defaults_information_state_from_family():
    assert _card().information_state == "POST"
    assert _card(method_id="online.toy", family="online").information_state == "LIVE"
    assert _card(method_id="config.toy", family="config", output_kind="finding").information_state == "PRE"


def test_register_rejects_duplicate_ids_and_missing_cards(monkeypatch):
    import auditable.methods as registry

    list_methods()  # load every module first, then register into a throwaway copy
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))

    class A(BaseStepLocalizer):
        card = _card(method_id="step.registry_test_dup")

    class B(BaseStepLocalizer):
        card = _card(method_id="step.registry_test_dup")

    register(A)
    assert register(A) is A  # re-registering the same class is a no-op
    with pytest.raises(ValueError, match="duplicate"):
        register(B)
    with pytest.raises(TypeError):
        register(type("NoCard", (BaseStepLocalizer,), {}))


def test_fit_scope_gates_entry_points():
    class Transductive(BaseRunDetector):
        card = _card(method_id="run.toy_transductive", family="run", fit_scope="transductive")

        def fit_score(self, X, y=None):
            return [float(len(x)) for x in X]

    det = Transductive()
    assert det.fit_score([[1], [1, 2]]) == [1.0, 2.0]
    with pytest.raises(NotSupported):
        det.fit([[1]])
    with pytest.raises(NotSupported):
        det.decision_function([[1]])


def test_stateless_step_localizer_batches_and_ranks():
    class Late(BaseStepLocalizer):
        card = _card(method_id="step.toy_late")

        def step_scores(self, trace):
            return [float(i) for i in range(len(trace))]

    loc = Late()
    assert loc.fit([]) is loc
    assert loc.step_scores_batch([[0, 0], [0, 0, 0]]) == [[0.0, 1.0], [0.0, 1.0, 2.0]]
    assert loc.rank([0, 0, 0]) == [2, 1, 0]


def test_step_localizer_without_scoring_entry_fails_loudly():
    class Empty(BaseStepLocalizer):
        card = _card(method_id="step.toy_empty")

    with pytest.raises(NotImplementedError):
        Empty().step_scores([0])


def test_online_prefix_score_reads_the_alarm_at_k():
    class Count(BaseOnlineMonitor):
        card = _card(method_id="online.toy_count", family="online")

        def step_alarm_scores(self, trace):
            return [float(i) for i in range(len(trace))]

    mon = Count()
    assert mon.prefix_score([0, 0, 0, 0], 2) == 1.0
    assert mon.prefix_score([0, 0], 5) == 1.0
    assert mon.prefix_score([], 3) == 0.0
    with pytest.raises(ValueError):
        mon.prefix_score([0], 0)


def test_params_round_trip_and_repr():
    cls = get_method("online.random")
    m = cls(random_state=3)
    assert m.get_params() == {"random_state": 3}
    assert m.set_params(random_state=4).random_state == 4
    assert repr(m) == "RandomOnline(random_state=4)"
    with pytest.raises(ValueError):
        m.set_params(seed=1)


def test_single_step_llm_judges_refuse_score_vectors():
    for mid in ("step.llm_step_by_step", "step.llm_binary_search"):
        cls = get_method(mid)
        assert cls.card.output_kind == "single"
        with pytest.raises(NotSupported):
            cls(model="m").step_scores([{"idx": 0, "agent": "a", "kind": "decision", "text": "x"}])


def test_methods_without_extras_run_on_the_core_install(monkeypatch):
    """A card with no extras promises that its entry points import no optional dependency,
    even in an environment where those dependencies happen to be installed."""
    import builtins

    from auditable.featurize import AgentConfig
    from auditable.llm import FixedResponseBackend

    optional = {"numpy", "scipy", "sklearn", "networkx", "pyod", "torch", "torch_geometric", "pygod", "openai"}
    real_import = builtins.__import__

    def core_only(name, *args, **kwargs):
        if name.split(".")[0] in optional:
            raise ModuleNotFoundError(f"{name} is not part of the core install")
        return real_import(name, *args, **kwargs)

    config = AgentConfig(declared_capabilities=[{"name": "shell", "type": "tool", "permission_level": "execute"}],
                         task_or_role_spec="Summarize a document.")
    trace = [{"idx": i, "agent": "a", "kind": "decision", "text": f"step {i}"} for i in range(3)]
    free = [c for c in list_methods() if not c.extras]
    assert free
    monkeypatch.setattr(builtins, "__import__", core_only)
    for card in free:
        method = get_method(card.method_id)
        kw = {"backend": FixedResponseBackend([], default="NEEDED: read_file")} if "backend" in method().get_params() else {}
        if card.family == "config":
            assert isinstance(method(**kw).audit(config), ConfigFinding)
        elif card.output_kind == "single":
            assert method(**kw).predict_step(trace) in (0, 1, 2, None)
        else:
            pytest.fail(f"{card.method_id} declares no extras; add a core-install probe for its family here")


def test_config_finding_abstention():
    assert ConfigFinding(flag=None).abstained
    assert not ConfigFinding(flag=False).abstained
    assert ConfigFinding(flag=True, excess=frozenset({"shell"})).excess == {"shell"}


def test_citation_fields():
    c = Citation(key="X (2026)", arxiv="2601.00001")
    assert c.title == "" and c.url == ""


def test_readme_method_table_matches_registry():
    text = README.read_text(encoding="utf-8")
    start, end = "<!-- methods:start -->", "<!-- methods:end -->"
    assert start in text and end in text, "README lacks the method-table markers"
    table = text.split(start, 1)[1].split(end, 1)[0].strip()
    assert table == method_table(), "README method table is stale; regenerate it from method_table()"
