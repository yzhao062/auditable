"""Optional framework integrations.

Each integration is an optional extra and imports its framework lazily, so the
core package never depends on any framework. Importing this package pulls in no
framework. The LangGraph integration lives in
:mod:`auditable.integrations.langgraph` and needs ``pip install auditable[langgraph]``.
"""
