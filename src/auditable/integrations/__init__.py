"""Optional framework integrations.

Each integration is an optional extra and imports its framework lazily, so the
core package never depends on any framework. Importing this package pulls in no
framework. The LangGraph integration lives in
:mod:`auditable.integrations.langgraph` and needs ``pip install auditable[langgraph]``;
the LangChain callback capture lives in :mod:`auditable.integrations.langchain` and
needs ``pip install auditable[langchain]``; the MCP tool-boundary capture lives in
:mod:`auditable.integrations.mcp` and needs ``pip install auditable[mcp]``.
"""
