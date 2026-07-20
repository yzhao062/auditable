"""Capture MCP tool traffic at the server boundary (the ops-side mount).

What this shows, in plain terms: when the agent is a black box (a vendor app, a
closed harness, someone else's code) and the one surface your team controls is
the MCP server it calls, you can still get an auditable record. `instrument(...)`
patches the server's tool dispatch in place, so every tools/call through any
transport lands as one typed `tool_call` step: tool name, an arguments digest, a
result digest, and timing. No agent (client) code changes at all:

    server = FastMCP("support-desk")
    ... register tools as usual ...
    mount = instrument(server)                   # 1) wrap once, serve as usual
    ...
    report = analyze_run(mount, adapter=mount)   # 2) the calls captured so far

The scenario is a small support desk: an order lookup tool and a refund tool.
A client drives three calls through the SDK's own in-memory client-server
transport (a real MCP roundtrip: initialize, tools/call, typed results; no
subprocess, no network), including one refund that fails over the cap, so the
captured record shows an honest error step too.

Honesty holds in the output. The boundary sees tool traffic only (State A):
names, argument and result digests, timing, order. It cannot see which
dependency state a tool read or wrote, so the captured steps carry no dependency
edges; none are fabricated, and dependency-state (State B) providers still have
to be registered separately. Analysis therefore yields an honest no-score /
zero-coverage report until State-B providers supply dependency edges.

Needs:  pip install "auditable[mcp,graph]"
Run:    python examples/example_mcp_capture.py
"""


import logging


def main():
    try:
        import anyio  # ships with the mcp SDK
        from mcp.server.fastmcp import FastMCP
        from mcp.shared.memory import create_connected_server_and_client_session
    except ImportError:
        print('This example needs the mcp extra:  pip install "auditable[mcp]"')
        return

    # keep the demo output readable: the SDK logs one INFO line per request
    logging.getLogger("mcp").setLevel(logging.WARNING)

    from auditable import analyze_run
    from auditable.integrations.mcp import instrument

    # 1) a toy MCP server with two tools, exactly as an ops team would host it.
    server = FastMCP("support-desk")

    @server.tool()
    def get_order(order_id: str) -> dict:
        """Look up one order."""
        return {"order_id": order_id, "total": 42.0, "status": "delivered"}

    @server.tool()
    def refund_order(order_id: str, amount: float) -> str:
        """Refund an order; fails over the 100.0 cap."""
        if amount > 100.0:
            raise ValueError("refund cap exceeded")
        return f"refunded {amount} on {order_id}"

    # 2) wrap once. The server keeps working through any transport; the capture
    # sits at the tool dispatch every tools/call passes through.
    mount = instrument(server)

    # 3) drive a few calls over the SDK's in-memory client-server transport.
    async def drive():
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool("get_order", {"order_id": "A1"})
            await client.call_tool("refund_order", {"order_id": "A1", "amount": 12.5})
            await client.call_tool("refund_order", {"order_id": "A1", "amount": 500.0})

    anyio.run(drive)

    # 4) analyze the captured traffic. With no dependency edges at this boundary,
    # the honest outcome is zero coverage with structural scores withheld.
    try:
        report = analyze_run(mount, adapter=mount)
    except ImportError:
        print('Analysis needs the graph extra:  pip install "auditable[graph]"')
        return
    print(
        "Analysis report: "
        f"{report.state}; {report.coverage.n_dep_edges} dependency edges "
        "(zero coverage). Structural scores are withheld until State-B providers "
        "supply dependency edges."
    )

    print("\nTool calls captured at the MCP boundary:")
    for s in mount.to_steps():
        a = s.node_attrs
        outcome = (
            f"error {a['error_type']} sha256 {a['error_digest'][:12]}..."
            if a["is_error"]
            else f"result sha256 {a['result_digest'][:12]}..."
        )
        print(
            f"  step {s.idx}: {a['tool']}({', '.join(a['argument_keys'])}) "
            f"in {a['duration_s'] * 1000:.2f} ms -> {outcome}"
        )

    print(
        "\nNote the empty dependency layer: this boundary observes tool traffic "
        "only (State A), so no OBSERVED dependency edge is fabricated. What a "
        "tool's decision relied on (State B) needs its own registration; until "
        "then the analysis remains an honest no-score / zero-coverage report."
    )


if __name__ == "__main__":
    main()
