"""The populations the board tasks score, read through agentpile.

Each module selects the same runs, in the same order, as the GRADE loader the CatchBench board
reads, so a task built on it scores the board's population. The selection rules are ported from
GRADE (https://github.com/yzhao062/grade, MIT License) at commit 3839a57; agentpile's
``convention="grade"`` projection supplies the steps and dependencies, which the agentpile
cross-check (``benchmarks/agentpile_crosscheck``) confirms equal GRADE's run by run.
"""
