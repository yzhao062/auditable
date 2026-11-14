"""Rerun the CatchBench detection board on SWE-Gym with a few lines of code.

Run from a checkout after installing the bench extra:

    pip install "auditable[bench]"
    python examples/example_leaderboard.py

The graph rows (PyGOD, GUARDIAN-Lite, G-Safeguard-Lite) also need ``auditable[graph-ad]``
and the PyG sampling backend described in the README Install section. Without them those
rows are listed as skipped with the extra named, and the rest of the board is still scored.

``detect/swegym`` asks whether a finished run failed, on 376 SWE-Gym OpenHands runs (188
failed) with the population and protocol of the CatchBench board, so each score can be
compared with the same row of that board (shown in brackets). The first call loads the runs
through agentpile, and the full 17-row board takes about 70 seconds on a laptop CPU. The
command-line form is ``python -m auditable.bench detect/swegym --golden``, which exits 1
when a row differs from the CatchBench board.
"""
from auditable.bench import leaderboard


def main():
    # progress fires as each row is scored, so the slow rows do not look like a hang.
    board = leaderboard("detect/swegym",
                        progress=lambda row: print(f"  scored {row.label} ({row.seconds:.0f} s)", flush=True))
    print()
    print(board)

    # Naming methods restricts the board to those rows. The population is cached on the
    # task, so this second board costs only the three fits. The size baseline run.lr_flat
    # belongs beside the dependency method: compare every row with it.
    print()
    print(leaderboard("detect/swegym", methods=["run.lr_flat", "run.lr_flatdep", "run.pyod_ecod"]))

    # The same board as a Markdown table, ready to paste into an issue or a README.
    print()
    print(board.to_markdown())


if __name__ == "__main__":
    main()
