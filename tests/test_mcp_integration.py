"""Integration tests for the MCP tool-boundary capture (auditable.integrations.mcp).

These build a real FastMCP server with pure-function tools (no LLM, no network,
no subprocess) and drive it through the SDK's in-memory client-server transport,
asserting that every tools/call lands as one typed ``tool_call`` step: ordered,
digested, timed, and with no fabricated dependency edges. They skip when ``mcp``
is not installed.
"""
import asyncio

import pytest

pytest.importorskip("mcp")

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from auditable.graph.session import SessionGraph
from auditable.integrations.mcp import instrument
from auditable.record import _sha256


def _build_mount(**kwargs):
    """A toy shop server (one read tool, one consequential write tool), instrumented."""
    server = FastMCP("shop")

    @server.tool()
    def get_order(order_id: str) -> dict:
        """Look up one order."""
        return {"order_id": order_id, "total": 42.0}

    @server.tool()
    def refund_order(order_id: str, amount: float) -> str:
        """Refund an order; fails over the cap."""
        if amount > 100:
            raise ValueError("refund cap exceeded")
        return f"refunded {amount} on {order_id}"

    return server, instrument(server, **kwargs)


def _drive(server, calls):
    """One in-memory client-server session; returns the CallToolResults in order."""

    async def _main():
        results = []
        async with create_connected_server_and_client_session(server) as client:
            for name, args in calls:
                results.append(await client.call_tool(name, args))
        return results

    return anyio.run(_main)


def test_wire_roundtrip_captures_ordered_tool_call_steps():
    server, mount = _build_mount()
    results = _drive(
        server,
        [("get_order", {"order_id": "A1"}), ("refund_order", {"order_id": "A1", "amount": 10.0})],
    )
    assert [r.isError for r in results] == [False, False]
    steps = mount.to_steps()
    assert [s.idx for s in steps] == [0, 1]
    assert [s.kind for s in steps] == ["tool_call", "tool_call"]
    assert [s.node_attrs["tool"] for s in steps] == ["get_order", "refund_order"]
    assert all(s.agent == "shop" for s in steps)  # default agent label: the server name
    assert [s.exec_preds for s in steps] == [[], []]  # ordering is not a client handoff


def test_step_node_attrs_carry_digests_and_timing():
    server, mount = _build_mount()
    args = {"order_id": "A1"}
    _drive(server, [("get_order", args)])
    (step,) = mount.to_steps()
    attrs = step.node_attrs
    assert attrs["arguments_digest"] == _sha256(args)  # pinned: canonical-JSON digest
    assert "arguments_digest_mode" not in attrs  # absent mode key means canonical JSON
    assert attrs["argument_keys"] == ["order_id"]  # names in the clear, values digested
    assert len(attrs["result_digest"]) == 64
    assert attrs["duration_s"] >= 0.0
    assert attrs["started_at"] > 0.0
    assert attrs["is_error"] is False
    assert attrs["mcp_server"] == "shop"
    assert attrs["mcp_request_id"]  # the wire path runs inside a request


def test_tool_error_is_captured_and_still_reported_to_client():
    server, mount = _build_mount()
    sentinel = "PRIVATE-REJECTED-AMOUNT"
    (result,) = _drive(server, [("refund_order", {"order_id": "A1", "amount": sentinel})])
    assert result.isError is True  # the client still sees the error result
    (step,) = mount.to_steps()
    attrs = step.node_attrs
    assert attrs["is_error"] is True
    assert attrs["error_type"] == "ToolError"
    assert len(attrs["error_digest"]) == 64
    assert sentinel not in repr(attrs)  # pydantic's rejected input value stays out of the graph
    assert "error" not in attrs
    assert "result_digest" not in attrs  # a failed call has no result to digest


def test_unknown_tool_call_is_captured_as_an_error_step():
    server, mount = _build_mount()
    (result,) = _drive(server, [("no_such_tool", {})])
    assert result.isError is True
    (step,) = mount.to_steps()
    assert step.node_attrs["tool"] == "no_such_tool"
    assert step.node_attrs["is_error"] is True


def test_cancelled_call_is_captured_and_propagates():
    server = FastMCP("cancellation")

    @server.tool()
    async def cancel_call() -> str:
        raise asyncio.CancelledError("cancel after possible side effect")

    mount = instrument(server)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.call_tool("cancel_call", {}))

    (step,) = mount.to_steps()
    assert step.node_attrs["is_error"] is True
    assert step.node_attrs["error_type"] == "CancelledError"
    assert len(step.node_attrs["error_digest"]) == 64


def test_reset_discards_a_call_begun_in_the_prior_window():
    server = FastMCP("windowed")
    entered = asyncio.Event()
    release = asyncio.Event()

    @server.tool()
    async def slow_call() -> str:
        entered.set()
        await release.wait()
        return "old-window"

    @server.tool()
    async def fresh_call() -> str:
        return "new-window"

    mount = instrument(server)

    async def overlap_reset():
        old_call = asyncio.create_task(server.call_tool("slow_call", {}))
        await entered.wait()  # slow_call has reserved idx 0 in the old window
        mount.reset()
        await server.call_tool("fresh_call", {})  # reserves idx 0 in the new window
        release.set()
        await old_call  # its stale append must be dropped

    asyncio.run(overlap_reset())
    steps = mount.to_steps()
    assert [(s.idx, s.node_attrs["tool"]) for s in steps] == [(0, "fresh_call")]
    assert len({s.idx for s in steps}) == len(steps)
    SessionGraph.from_steps(steps)  # the reset cannot create duplicate graph ids


def test_no_dependency_edges_are_fabricated():
    # the boundary observes no resource touches, so the dependency layer must be
    # empty: no edge is presented as OBSERVED without an observed read/write match
    server, mount = _build_mount()
    _drive(
        server,
        [
            ("get_order", {"order_id": "A1"}),
            ("get_order", {"order_id": "B2"}),
            ("refund_order", {"order_id": "A1", "amount": 5.0}),
        ],
    )
    steps = mount.to_steps()
    assert all(s.deps == [] for s in steps)
    coverage = SessionGraph.from_steps(steps).coverage()
    assert coverage.n_dep_edges == 0
    assert coverage.observed_fraction == 0.0


def test_direct_in_process_dispatch_is_captured():
    # Level 2 also covers a host calling the server object directly (no transport);
    # off the wire there is no request context, so no request id is recorded.
    server, mount = _build_mount()
    anyio.run(lambda: server.call_tool("get_order", {"order_id": "B2"}))
    (step,) = mount.to_steps()
    assert step.node_attrs["tool"] == "get_order"
    assert "mcp_request_id" not in step.node_attrs


def test_analyze_run_reports_no_score_at_zero_coverage():
    pytest.importorskip("networkx")
    from auditable import analyze_run
    from auditable.graph.adapters import Adapter

    server, mount = _build_mount()
    _drive(
        server,
        [("get_order", {"order_id": "A1"}), ("refund_order", {"order_id": "A1", "amount": 9.0})],
    )
    assert isinstance(mount, Adapter)  # name + version + to_steps
    assert mount.id == "mcp_capture_v1"
    report = analyze_run(mount, adapter=mount)
    assert report.state == "no_score:low_coverage"
    assert report.coverage.n_dep_edges == 0
    assert report.coverage.observed_fraction == 0.0
    assert report.keystone is None
    assert all(row.score is None for row in report.ranked)


def test_reset_clears_the_capture_window():
    server, mount = _build_mount()
    _drive(server, [("get_order", {"order_id": "A1"})])
    mount.reset()
    _drive(server, [("refund_order", {"order_id": "A1", "amount": 1.0})])
    steps = mount.to_steps()
    assert [(s.idx, s.node_attrs["tool"]) for s in steps] == [(0, "refund_order")]


def test_top_level_call_tool_result_uses_canonical_json_digest():
    server = FastMCP("direct-result")
    returned = CallToolResult(
        content=[TextContent(type="text", text="direct result")],
        structuredContent={"status": "ok"},
    )

    @server.tool()
    def direct_result() -> CallToolResult:
        return returned

    mount = instrument(server)
    actual = anyio.run(lambda: server.call_tool("direct_result", {}))
    assert actual is returned
    (step,) = mount.to_steps()
    assert step.node_attrs["result_digest"] == _sha256(returned.model_dump(mode="json"))
    assert "result_digest_mode" not in step.node_attrs


def test_double_instrument_raises():
    server, _mount = _build_mount()
    with pytest.raises(RuntimeError):
        instrument(server)


def test_lowlevel_server_is_rejected_with_a_clear_error():
    from mcp.server.lowlevel import Server

    with pytest.raises(TypeError):
        instrument(Server("lowlevel"))


def test_core_import_does_not_pull_mcp():
    # the framework-agnostic core (and the touch spine) must not import mcp; the
    # MCP adapter is the only place that does, behind the optional extra
    import subprocess
    import sys

    code = (
        "import sys, auditable, auditable.graph.touch\n"
        "pulled = sorted(m for m in sys.modules if m == 'mcp' or m.startswith('mcp.'))\n"
        "assert not pulled, pulled\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
