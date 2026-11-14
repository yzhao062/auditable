# Cached PRE judge votes for `auditable.bench`

This directory holds the recorded held-out LLM judge votes read by the
`pre/over_privilege` leaderboard task so that `config.llm_needed` replays
recorded judgments without making any network call.

- **Source repository**: CatchBench (`https://github.com/yzhao062/catchbench`)
- **License**: MIT License
- **Release**: `0.1.3`
- **Source path**: `data/pre/llm_judge_method/llama-3.3-70b.json`
- **Local path**: `llm_judge_method/llama-3.3-70b.json`
- **SHA-256**: `4245ad9e46487cb915e069e7a5268b3c5c7c8ea8b0f7e74a6397436470cfbe3e`

Each entry maps a configuration `instance_id` to the list of declared capability
names the `llama-3.3-70b` judge marked as needed. Configurations absent from the
file or holding a non-list value are treated as abstentions (`flag=None`) and
left out of the evaluable denominator, reducing `coverage` accordingly.
