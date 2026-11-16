# Agent Traces

[agentpile](https://github.com/yzhao062/agentpile) loads public agent-trace datasets (Who&When, TRAIL, MAST, tau-bench, SWE-Gym and others), trace files (OpenAI-style messages, OTLP JSON, Claude Code sessions, MCP tool lists) and captured runs into one record. `from_agentpile` turns those records into the traces every method reads, and `configs_from_agentpile` turns them into the configurations the config methods read. `pip install "auditable[agentpile]"` installs the conversion, and an example that also runs tabular methods needs `pip install "auditable[agentpile,tabular-ad]"`. The [README](https://github.com/yzhao062/auditable/blob/main/README.md#real-agent-traces) has a runnable example, and the [method contract](methods-contract.md#inputs) lists the input forms a method accepts.

## Dependencies and Conventions

`deps` has no default. It is a list of agentpile rules, `"recorded"` for the dependencies the record already holds, or `"full_context"` for every earlier step, and each trace's `meta` keeps the rules that ran.

The default `convention="generic"` makes one step per message and one per tool call with its result. `convention="grade"` with `deps="grade"` instead reproduces the steps, dependencies and graphs of the GRADE converters that CatchBench and the GRADE paper used, for the seven corpora below. [`benchmarks/agentpile_crosscheck`](https://github.com/yzhao062/auditable/tree/main/benchmarks/agentpile_crosscheck) compares that output with GRADE's run by run.

| Corpus | `load_traces` arguments | Runs in GRADE's output | Result |
|---|---|---|---|
| Who&When | `"who_and_when", convention="grade", deps="grade"` | 184 | equal; the 126 localization labels match |
| tau-bench | `"tau_bench", convention="grade", deps="grade", model=[...]` | 660 | equal |
| tau2-bench | `"tau2_bench", convention="grade", deps="grade", model=...` | 278 | equal to the frozen rule `tau2-frozen-v1`, since GRADE's own tau2 output changes with the hash seed |
| SWE-Gym | `"swe_gym_openhands", convention="grade", deps="grade"` | 376 | equal |
| AgentRewardBench | `"agent_reward_bench", convention="grade", deps="grade"` | 600 | equal on 3 runs; on the other 597, GRADE adds two steps for the page after the last action, which agentpile does not create, and the steps before them are equal |
| SWE-rebench OpenHands | `"swe_rebench_openhands", convention="grade", deps="grade"` | 600 | equal |
| SWE-agent | `"swe_agent_nebius", convention="grade", deps="grade"` | 600 | equal |

These results hold on the dataset revisions agentpile 0.3.0 loads by default. On any other revision, `convention="grade"` raises an error instead of returning steps that nobody has compared with GRADE's. `convention="generic"` still reads the run, without that guarantee.

## Text and Roles

In the generic convention, a `user` step is a step the record attributes to the user role, which is not always text a person typed (a session's compact summary or a task notification has that role too). Content given as a list of parts (dicts with a string `type`, possibly mixed with bare strings) keeps its text; an image, a document or any other non-text part becomes a placeholder such as `[image]` and its other fields are dropped, while any other list stays JSON. `meta["report"]` counts the replaced parts and accounts for every step of the record.

## Subagents

A run with subagent trajectories (a Claude Code session with subagents, an Agents SDK handoff) raises an error, since this version does not flatten them into one trace. `from_agentpile(run.subagents, ...)` reads each subagent as a trace of its own, and `subagents="skip"` reads the top-level trajectory only; on the Claude Code sessions with subagents we measured, skipping left out about half of the steps.
