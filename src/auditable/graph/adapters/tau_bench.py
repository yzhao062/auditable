"""Pure tau-bench corpus adapter: a trajectory of role / tool messages to Steps.

tau-bench is a tool-using agent acting against a backend database (retail /
airline). The DB read and write events are in the trace, so the read/write
structure is observed rather than inferred, and some calls are consequential
writes (book / cancel / modify a reservation or order). This adapter stays pure:
messages in, typed steps out, with no ``huggingface_hub`` and no file download.
The dataset fetch stays out of core; an examples-only helper or the optional
``corpora`` extra owns it.

Each tool event becomes a ``tool_call`` step tagged with its tool name and
whether it reads or writes the DB. For each consequential write, every prior
DB-read step is attached as a dependency edge graded ``OBSERVED`` (the read and
write events are observed in the trace), but the edge ``evidence`` marks the
write-to-all-prior-reads set as ``modeled``: it is a conservative temporal upper
bound on the audit surface, not a causal label, because tau-bench does not record
the exact read subset a write used. Utility calls that touch no DB state
(``calculate``, ``think``, a human handoff, a static reference lookup) are tool
steps but are not reads and never enter any write's audit surface.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..session import DependencyEdge, Grade, Step
from .protocol import _BaseAdapter

__all__ = ["TauBenchPriorDBReadsAdapter", "tau_bench_prior_db_reads_v1"]

# Consequential DB mutations vs reads, by tool-name prefix (tau-bench retail +
# airline). These match the experiment's classification exactly.
_WRITE_PREFIXES = ("book_", "cancel_", "modify_", "update_", "return_", "exchange_", "send_")
# Tool calls that touch no DB state (local compute, no-op thought, human handoff,
# static reference lookup): tool steps, but not reads, so not part of any audit surface.
_NON_DB_TOOLS = frozenset({"calculate", "think", "transfer_to_human_agents", "list_all_airports"})


def _is_write(name: Any) -> bool:
    return bool(name) and name.startswith(_WRITE_PREFIXES)


def _is_db_read(name: Any) -> bool:
    return bool(name) and not _is_write(name) and name not in _NON_DB_TOOLS


class TauBenchPriorDBReadsAdapter(_BaseAdapter):
    """Map a tau-bench-style message trajectory to typed steps with modeled deps.

    ``assistant_agent`` is the agent label for assistant turns (the messages do
    not name the model, so it defaults to ``"assistant"``); ``model_id``, when
    set, is attached to each assistant decision node so the model attribute is
    populated without inventing an identity the trace does not carry.
    """

    name = "tau_bench_prior_db_reads"
    version = "v1"

    def __init__(self, *, assistant_agent: str = "assistant", model_id: str = "") -> None:
        self.assistant_agent = assistant_agent
        self.model_id = model_id

    def to_steps(self, messages: Any) -> List[Step]:
        """Normalize one tau-bench task run (a list of role / tool messages) into
        typed steps. Read and write tool events are observed; each write depends on
        every prior DB read as a conservative, modeled prior-read upper bound."""
        steps: List[Step] = []
        pending: List[Any] = []  # tool-call names the assistant requested, awaiting results
        reads_so_far: List[int] = []  # idx of every prior DB-read step
        idx = 0
        for m in messages or ():
            role = m.get("role")
            if role == "system":
                continue  # the policy text: context, not a step
            if role == "user":
                steps.append(Step(idx=idx, agent="user", kind="decision"))
                idx += 1
            elif role == "assistant":
                attrs: Dict[str, Any] = {"model_id": self.model_id} if self.model_id else {}
                steps.append(
                    Step(idx=idx, agent=self.assistant_agent, kind="decision", node_attrs=attrs)
                )
                idx += 1
                for tc in (m.get("tool_calls") or ()):
                    name = (tc.get("function") or {}).get("name") or tc.get("name")
                    pending.append(name)
            elif role == "tool":
                name = m.get("name")
                if name is None and pending:
                    name = pending.pop(0)
                elif name in pending:
                    pending.remove(name)
                write = _is_write(name)
                db_read = _is_db_read(name)
                deps = self._write_deps(name, reads_so_far) if write else []
                steps.append(
                    Step(
                        idx=idx,
                        agent="env",
                        kind="tool_call",
                        deps=deps,
                        node_attrs={"tool": name, "is_write": write, "is_db_read": db_read},
                    )
                )
                if db_read:
                    reads_so_far.append(idx)
                idx += 1
        return steps

    def _write_deps(self, write_tool: Any, reads_so_far: List[int]) -> List[DependencyEdge]:
        """One OBSERVED-but-modeled dependency edge per prior DB read.

        The read and write events are observed in the trace, so the grade is
        ``OBSERVED``; the evidence marks the write-to-all-prior-reads edge set as a
        conservative prior-read upper bound rather than a causal label, and leaves
        ``resource`` unset (the v0.3b own-record touch contract fills resource
        identity; a corpus trace does not expose it)."""
        return [
            DependencyEdge(
                src_idx=r,
                grade=Grade.OBSERVED,
                resource=None,
                evidence={
                    "modeled": True,
                    "relation": "write_to_prior_db_read",
                    "note": (
                        "conservative prior-read upper bound over the observed DB "
                        "reads; not a causal label"
                    ),
                    "adapter": self.id,
                    "write_tool": write_tool,
                },
            )
            for r in reads_so_far
        ]


# The versioned singleton, callable as tau_bench_prior_db_reads_v1(messages).
tau_bench_prior_db_reads_v1 = TauBenchPriorDBReadsAdapter()
