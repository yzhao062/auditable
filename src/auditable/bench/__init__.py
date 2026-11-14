"""Leaderboards for agent-auditing methods, on the populations and protocols of the CatchBench board.

``leaderboard(task_id)`` scores every registered method of the task's family and returns a
ranked :class:`Board`; ``leaderboard(task_id, extra=[MyMethod])`` places a new method on it.
The data come through agentpile (``pip install "auditable[bench]"``), and each task's golden
file holds the CatchBench 0.1.3 board values the scores are checked against.

    >>> from auditable.bench import leaderboard
    >>> board = leaderboard("detect/swegym")       # doctest: +SKIP
    >>> print(board)                                # doctest: +SKIP

From a shell: ``python -m auditable.bench detect/swegym``.
"""
from auditable.bench._core import Board, Entry, Row, Task, get_task, leaderboard, list_tasks, register_task

__all__ = ["leaderboard", "list_tasks", "get_task", "Board", "Entry", "Row", "Task", "register_task"]
