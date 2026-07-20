"""Integration tests for the LangChain callback capture adapter.

The real runs use only ``langchain-core`` pure-Python runnables, tools, and its
fake LLM: no network, provider package, or API key.  The module skips cleanly when
the optional LangChain extra is absent.
"""
import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.tools import tool

from auditable import LiveSession, analyze_run
from auditable.graph.session import Grade, GraphCompleteness
from auditable.integrations.langchain import LangChainCallbackHandler


@tool
def double_value(value: int) -> int:
    """Double one integer."""
    return value * 2


def _tool_chain():
    return (
        RunnableLambda(lambda value: {"value": value}, name="prepare")
        | double_value
        | RunnableLambda(lambda value: {"answer": value}, name="finish")
    ).with_config(run_name="number_chain")


def _run_tool_chain(value=3):
    handler = LangChainCallbackHandler()
    output = _tool_chain().invoke(value, config={"callbacks": [handler]})
    return handler, output


def test_handler_is_a_versioned_langchain_callback_adapter():
    handler = LangChainCallbackHandler()

    assert isinstance(handler, BaseCallbackHandler)
    assert handler.name == "langchain_callback"
    assert handler.version == "v1"
    assert handler.id == "langchain_callback_v1"
    assert handler.steps() == []
    assert handler.to_steps(handler) == []


def test_real_tool_sequence_captures_order_kinds_attrs_and_outputs():
    handler, output = _run_tool_chain()
    steps = handler.steps()

    assert output == {"answer": 6}
    assert [step.agent for step in steps] == [
        "number_chain",
        "prepare",
        "double_value",
        "finish",
    ]
    assert [step.kind for step in steps] == [
        "decision",
        "decision",
        "tool_call",
        "decision",
    ]
    assert [step.idx for step in steps] == [0, 1, 2, 3]
    assert steps[2].node_attrs["tool"] == "double_value"
    assert steps[2].node_attrs["outputs"] == 6
    assert steps[3].node_attrs["outputs"] == {"answer": 6}
    assert steps[2].node_attrs["inputs_digest"].startswith("sha256:")
    assert len(steps[2].node_attrs["inputs_digest"]) == len("sha256:") + 64
    assert all(step.correlation_id == step.node_attrs["langchain_run_id"] for step in steps)


def test_exact_runnable_sequence_handoffs_are_observed_value_edges():
    handler, _ = _run_tool_chain()
    root, prepare, tool_step, finish = handler.steps()

    assert root.deps == []
    assert prepare.deps == []  # root/child input equality is not a prior output
    assert [(edge.src_idx, edge.grade) for edge in tool_step.deps] == [
        (prepare.idx, Grade.OBSERVED)
    ]
    assert [(edge.src_idx, edge.grade) for edge in finish.deps] == [
        (tool_step.idx, Grade.OBSERVED)
    ]
    edge = tool_step.deps[0]
    assert edge.resource.namespace == "langchain_callback_value"
    assert edge.resource.resource_id == tool_step.node_attrs["inputs_digest"]
    assert edge.evidence["relation"] == "callback_output_to_input"
    assert edge.evidence["topology"] == "runnable_sequence"
    assert edge.evidence["observed"] is True


def test_fake_llm_run_captures_model_id_and_normalized_value_flow():
    # Keep these imports local: the prompt/fake-model modules import the optional
    # huggingface client in current langchain-core, and collection must not pollute
    # unrelated adapter-purity tests that run before this test.
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.prompts import PromptTemplate

    handler = LangChainCallbackHandler()
    chain = (
        PromptTemplate.from_template("Say {word}")
        | FakeListLLM(responses=["accepted"])
        | RunnableLambda(lambda text: text.upper(), name="normalize")
    ).with_config(run_name="llm_chain")

    output = chain.invoke({"word": "yes"}, config={"callbacks": [handler]})
    steps = handler.steps()
    llm_step = next(step for step in steps if step.node_attrs["langchain_run_type"] == "llm")
    prompt_step = next(step for step in steps if step.agent == "PromptTemplate")
    normalize_step = next(step for step in steps if step.agent == "normalize")

    assert output == "ACCEPTED"
    assert llm_step.kind == "decision"
    assert llm_step.node_attrs["model_id"] == "FakeListLLM"
    assert llm_step.node_attrs["outputs"] == "accepted"
    # StringPromptValue -> prompt string and LLMResult -> returned text are
    # normalized to the values actually handed across the two callback seams.
    assert [(edge.src_idx, edge.grade) for edge in llm_step.deps] == [
        (prompt_step.idx, Grade.OBSERVED)
    ]
    assert [(edge.src_idx, edge.grade) for edge in normalize_step.deps] == [
        (llm_step.idx, Grade.OBSERVED)
    ]


def test_sequence_without_exact_payload_match_is_declared_not_observed():
    handler = LangChainCallbackHandler()
    root, first, second = uuid4(), uuid4(), uuid4()
    handler.on_chain_start(None, "root input", run_id=root, name="root")
    handler.on_chain_start(
        None,
        {"value": 1},
        run_id=first,
        parent_run_id=root,
        tags=["seq:step:1"],
        name="first",
    )
    handler.on_chain_end({"value": 2}, run_id=first, parent_run_id=root)
    handler.on_chain_start(
        None,
        {"value": 999},
        run_id=second,
        parent_run_id=root,
        tags=["seq:step:2"],
        name="second",
    )
    handler.on_chain_end("done", run_id=second, parent_run_id=root)
    handler.on_chain_end("done", run_id=root)

    second_step = next(step for step in handler.steps() if step.agent == "second")
    assert len(second_step.deps) == 1
    edge = second_step.deps[0]
    assert edge.grade is Grade.DECLARED
    assert edge.resource is None
    assert edge.evidence["declared"] is True
    assert edge.evidence["observed"] is False
    assert edge.evidence["relation"] == "runnable_sequence_adjacency"


def test_unstructured_payload_equality_is_inferred_not_observed():
    handler = LangChainCallbackHandler()
    root, first, second = uuid4(), uuid4(), uuid4()
    handler.on_chain_start(None, 0, run_id=root, name="root")
    handler.on_chain_start(None, "same", run_id=first, parent_run_id=root, name="first")
    handler.on_chain_end("same", run_id=first, parent_run_id=root)
    handler.on_chain_start(None, "same", run_id=second, parent_run_id=root, name="second")
    handler.on_chain_end("done", run_id=second, parent_run_id=root)
    handler.on_chain_end("done", run_id=root)

    second_step = next(step for step in handler.steps() if step.agent == "second")
    assert len(second_step.deps) == 1
    edge = second_step.deps[0]
    assert edge.grade is Grade.INFERRED
    assert edge.resource is None
    assert edge.evidence["inferred"] is True
    assert edge.evidence["observed"] is False
    assert edge.evidence["relation"] == "callback_payload_equality"


def test_parallel_runnable_branches_do_not_gain_dependency_edges():
    handler = LangChainCallbackHandler()
    parallel = RunnableParallel(
        left=RunnableLambda(lambda value: value, name="left"),
        right=RunnableLambda(lambda value: value, name="right"),
    ).with_config(run_name="parallel")

    assert parallel.invoke(7, config={"callbacks": [handler]}) == {"left": 7, "right": 7}
    by_agent = {step.agent: step for step in handler.steps()}

    assert by_agent["left"].deps == []
    assert by_agent["right"].deps == []
    assert by_agent["left"].exec_preds == [by_agent["parallel"].idx]
    assert by_agent["right"].exec_preds == [by_agent["parallel"].idx]


def test_captured_steps_drive_analyze_run_and_live_session():
    handler, _ = _run_tool_chain()

    report = analyze_run(handler, adapter=handler, ground=False)
    assert report.adapter == "langchain_callback_v1"
    assert report.n_steps == 4
    assert report.coverage.by_grade[Grade.OBSERVED] == 2
    assert report.session_graph.steps == handler.steps()

    live = LiveSession(adapter=handler)
    for step in handler.steps():
        prefix = live.observe(step)
    assert prefix.adapter == "langchain_callback_v1"
    assert prefix.completeness is GraphCompleteness.PREFIX
    assert prefix.n_steps == 4


def test_sequential_handler_reuse_keeps_only_latest_run_and_reset_clears_it():
    handler = LangChainCallbackHandler()
    first = RunnableLambda(lambda value: value + 1).with_config(run_name="first")
    second = RunnableLambda(lambda value: value * 3).with_config(run_name="second")

    assert first.invoke(2, config={"callbacks": [handler]}) == 3
    assert [step.agent for step in handler.steps()] == ["first"]
    assert second.invoke(4, config={"callbacks": [handler]}) == 12
    assert [step.agent for step in handler.steps()] == ["second"]
    assert handler.steps()[0].node_attrs["outputs"] == 12

    handler.reset()
    assert handler.steps() == []


def test_async_runnable_sequence_is_captured():
    async def prepare(value):
        return {"value": value}

    handler = LangChainCallbackHandler()
    chain = (
        RunnableLambda(prepare, name="async_prepare")
        | double_value
        | RunnableLambda(lambda value: value + 1, name="finish")
    ).with_config(run_name="async_chain")

    output = asyncio.run(chain.ainvoke(5, config={"callbacks": [handler]}))
    by_agent = {step.agent: step for step in handler.steps()}

    assert output == 11
    assert by_agent["double_value"].kind == "tool_call"
    assert by_agent["double_value"].deps[0].grade is Grade.OBSERVED
    assert by_agent["finish"].deps[0].src_idx == by_agent["double_value"].idx


def test_failed_runnable_is_closed_as_an_error_step_without_faking_output_flow():
    def fail(_value):
        raise ValueError("offline failure")

    handler = LangChainCallbackHandler()
    runnable = RunnableLambda(fail).with_config(run_name="failure")

    with pytest.raises(ValueError, match="offline failure"):
        runnable.invoke("input", config={"callbacks": [handler]})

    steps = handler.steps()
    assert len(steps) == 1
    assert steps[0].agent == "failure"
    assert steps[0].node_attrs["status"] == "error"
    assert steps[0].node_attrs["outputs"] is None
    assert steps[0].node_attrs["error_type"].endswith("ValueError")
    assert "outputs_digest" not in steps[0].node_attrs
    assert steps[0].deps == []


def test_error_capture_retains_type_and_digest_never_the_raw_message():
    sentinel = "PRIVATE-LANGCHAIN-ERROR-992"

    def fail(_value):
        raise ValueError(f"provider rejected token {sentinel}")

    handler = LangChainCallbackHandler()
    runnable = RunnableLambda(fail).with_config(run_name="leaky_failure")

    with pytest.raises(ValueError):
        runnable.invoke("input", config={"callbacks": [handler]})

    (step,) = handler.steps()
    assert sentinel not in repr(step.node_attrs)
    assert "error" not in step.node_attrs
    assert step.node_attrs["error_type"].endswith("ValueError")
    assert step.node_attrs["error_digest"].startswith("sha256:")
