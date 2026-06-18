"""Second experiment: does the graph localize the fault, not just detect failure?

The keystone (agent_failure_detection.py) is run-level: does a run's graph predict that
it failed? This is step-level: given a failed run, which step is the fault? That is the
auditing question the unified view is meant to answer, not only "this run went wrong" but
"here is the decision it went wrong at."

Corpus: Who&When (the failure-only corpus the keystone had to skip), the labeled-failed
multi-agent runs, each with a human-verified ground-truth ``mistake_step``. Run-level
detection is inapplicable here (every run failed); step-level localization is exactly what
its labels support, so the corpus that could not anchor detection anchors auditing instead.

Ablation, mirroring the keystone's nested feature sets:
  - RANDOM   : a floor (rank the steps at random).
  - POSITION : one feature, normalized step position. The faults-are-early prior the
               characterization found (median fault position 0.33, 57% in the first third).
  - STRUCTURE: position plus graph-structural step features read from the typed graph: is
               the step's agent the most active, its activity share and rank, decision vs
               tool step, how far into the collaboration, and the agent's in/out handoff
               degree in the execution graph. STRUCTURE - POSITION isolates the localization
               signal the graph structure carries beyond "faults are early."

Honest scope: under the full-context dependency assumption these debate traces use, every
step depends on all prior steps, so the dependency layer is position-equivalent here and
carries no localization signal beyond POSITION. The structural lift therefore comes from the
EXECUTION layer (agent activity and handoff centrality), not the inferred dependency layer.
That a corpus with observed dependencies AND step-level fault labels does not yet exist is
the gap; on this corpus the claim is execution-structure localization beyond position.

Method: a pointwise ranker. A standardized logistic regression (liblinear, balanced class
weights; liblinear rather than the default lbfgs, which segfaults under this Windows conda
BLAS stack, and the two agree elsewhere) scores P(mistake | step features); steps are ranked
within a run by that score. Cross-validation is grouped by run (no step from a test run is
ever in training). Per-run metrics: top-1 (is the top-ranked step the fault?), top-3, and
MRR (1 / rank of the true fault). Reported as the mean over runs with a seed-block 95% CI
over 5 seeds. POSITION is a single monotone feature, so its ranking is identical across
seeds and its CI is zero by construction; the STRUCTURE - POSITION CI comes from STRUCTURE's
split-to-split variation.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python experiment/agent_failure_localization.py
"""
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from auditable.graph import build_graph

import agent_graph_characterization as ww

SEEDS = range(5)
_T975_4 = 2.776  # t(0.975, df=4): seed-block 95% CI half-width over 5 seeds
# Feature columns, position first so POSITION is a strict prefix of STRUCTURE (nested ablation).
_FEATURES = ["norm_pos", "is_tool", "agent_share", "is_most_active",
             "agent_rank", "agents_so_far", "handoff_out", "handoff_in"]
_LAYER_COLS = {"position": 1, "structure": len(_FEATURES)}


# --- corpus: failed Who&When runs with a usable ground-truth mistake step ---------

def load_failed_runs():
    ww._ensure_corpus()
    runs = []
    for t in ww.load_tasks():
        if str(t.get("is_correct", "")).lower() != "false":
            continue  # localization is posed within failed runs (the audit setting)
        steps = ww.to_steps(t)
        if len(steps) < 3:
            continue  # need a few steps to make ranking meaningful
        try:
            ms = int(t.get("mistake_step"))
        except (TypeError, ValueError):
            continue
        if not (0 <= ms < len(steps)):
            continue  # mistake_step must index a real step
        runs.append({"steps": steps, "G": build_graph(steps, dependency="full_context"), "mistake": ms})
    return runs


# --- per-step structural features from the typed graph ----------------------------

def _agent_handoff_degrees(G):
    """Distinct-agent in/out handoff degree per agent, from the execution graph's
    ``handoff_to`` edges (step -> step), projected to the agents that own those steps."""
    agent_of = {n: d["agent"] for n, d in G.nodes(data=True) if d["ntype"] in ("decision", "tool_call")}
    out, inn = {}, {}
    for u, v, d in G.edges(data=True):
        if d["etype"] != "handoff_to":
            continue
        au, av = agent_of.get(u), agent_of.get(v)
        if au is None or av is None or au == av:
            continue  # self-handoff is the same agent acting again, not a delegation
        out.setdefault(au, set()).add(av)
        inn.setdefault(av, set()).add(au)
    return {a: len(s) for a, s in out.items()}, {a: len(s) for a, s in inn.items()}


def _step_matrix(run):
    """Feature rows aligned to ``run['steps']`` (same order), columns ``_FEATURES``."""
    steps, G = run["steps"], run["G"]
    n = len(steps)
    dec = Counter(s["agent"] for s in steps if s["kind"] == "decision")
    rank = {a: i for i, (a, _) in enumerate(dec.most_common())}  # 0 = most active decider
    n_ag = max(1, len({s["agent"] for s in steps}))
    n_dec = max(1, sum(dec.values()))
    hout, hin = _agent_handoff_degrees(G)
    seen, rows = set(), []
    for s in steps:
        a = s["agent"]
        seen.add(a)
        rows.append([
            s["idx"] / max(1, n - 1),                       # norm_pos          (POSITION)
            1.0 if s["kind"] == "tool_call" else 0.0,       # is_tool
            dec.get(a, 0) / n_dec,                          # agent activity share
            1.0 if rank.get(a, 10 ** 9) == 0 else 0.0,      # is most-active agent
            rank.get(a, n_ag) / n_ag,                       # agent activity rank (0 = top)
            len(seen) / n_ag,                               # distinct agents seen so far
            hout.get(a, 0) / n_ag,                          # agent handoff out-degree
            hin.get(a, 0) / n_ag,                           # agent handoff in-degree
        ])
    return np.array(rows, dtype=float)


# --- pooled design matrix with per-run grouping -----------------------------------

def _pool(runs):
    X_rows, y, groups, mistake_row = [], [], [], {}
    for ri, run in enumerate(runs):
        M = _step_matrix(run)
        base = len(y)
        for j, s in enumerate(run["steps"]):
            X_rows.append(M[j])
            hit = 1 if s["idx"] == run["mistake"] else 0
            y.append(hit)
            groups.append(ri)
            if hit:
                mistake_row[ri] = base + j
    return np.array(X_rows, dtype=float), np.array(y), np.array(groups), mistake_row


# --- grouped CV and ranking metrics -----------------------------------------------

def _grouped_folds(groups, n_splits, seed):
    """Test masks for ``n_splits`` folds, partitioning whole runs with a seeded shuffle."""
    uniq = np.array(sorted(set(groups.tolist())))
    np.random.RandomState(seed).shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    fold = np.array([fold_of[g] for g in groups])
    return [fold == k for k in range(n_splits)]


def _oof_scores(X, y, groups, ncols, seed):
    """Out-of-fold P(mistake) for every step; each step scored by a model never trained
    on its own run."""
    scores = np.full(len(y), np.nan)
    for test in _grouped_folds(groups, 5, seed):
        train = ~test
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(solver="liblinear", class_weight="balanced"))
        clf.fit(X[train][:, :ncols], y[train])
        scores[test] = clf.predict_proba(X[test][:, :ncols])[:, 1]
    return scores


def _rank_metrics(scores, groups, mistake_row):
    """Mean per-run top-1, top-3, and MRR of the true fault step given step scores."""
    top1, top3, rr = [], [], []
    for ri in sorted(set(groups.tolist())):
        rows = np.where(groups == ri)[0]
        order = rows[np.argsort(-scores[rows], kind="stable")]  # highest score first
        rank = int(np.where(order == mistake_row[ri])[0][0]) + 1
        top1.append(rank == 1)
        top3.append(rank <= 3)
        rr.append(1.0 / rank)
    return np.array([np.mean(top1), np.mean(top3), np.mean(rr)])


def _seed_metrics(X, y, groups, mistake_row, model):
    """(5 seeds, 3 metrics) array of per-run-averaged top1/top3/MRR for one model."""
    out = []
    for s in SEEDS:
        if model == "random":
            sc = np.random.RandomState(s).rand(len(y))
        else:
            sc = _oof_scores(X, y, groups, _LAYER_COLS[model], s)
        out.append(_rank_metrics(sc, groups, mistake_row))
    return np.array(out)


def _ci(col):
    """Mean and seed-block 95% CI half-width for a length-5 vector of per-seed values."""
    return col.mean(), _T975_4 * col.std(ddof=1) / np.sqrt(len(col))


def main():
    print("Localization: does the graph point to the fault step in a failed run?")
    runs = load_failed_runs()
    n = len(runs)
    n_steps = sum(len(r["steps"]) for r in runs)
    print(f"corpus: Who&When, {n} failed runs with a ground-truth mistake step "
          f"({n_steps} steps, {n / n_steps:.0%} are faults).\n")

    X, y, groups, mistake_row = _pool(runs)
    metrics = {m: _seed_metrics(X, y, groups, mistake_row, m)
               for m in ("random", "position", "structure")}

    print(f"  {'model':10s}  {'top-1':>14s}  {'top-3':>14s}  {'MRR':>14s}")
    for m in ("random", "position", "structure"):
        cells = []
        for k in range(3):
            mean, half = _ci(metrics[m][:, k])
            cells.append(f"{mean:.3f} +/-{half:.3f}")
        print(f"  {m:10s}  {cells[0]:>14s}  {cells[1]:>14s}  {cells[2]:>14s}")

    # the ablation: structure beyond the position prior, per-seed paired, seed-block CI
    gain = metrics["structure"] - metrics["position"]
    names = ["top-1", "top-3", "MRR"]
    print("\n  gain STRUCTURE - POSITION (the graph-structure lift beyond the early-fault prior):")
    for k in range(3):
        mean, half = _ci(gain[:, k])
        pos = int((gain[:, k] > 0).sum())
        print(f"    {names[k]:6s}: {mean:+.3f}  95% CI [{mean - half:+.3f}, {mean + half:+.3f}]  "
              f"({pos}/{len(SEEDS)} seeds +)")

    print("\nReading: POSITION already localizes well above RANDOM because faults cluster early, "
          "so the honest baseline is the early-fault prior, not chance. The question is whether the "
          "graph's execution structure (which agent is central, how control hands off) localizes "
          "the fault BEYOND knowing it tends to be early. The STRUCTURE - POSITION gain answers it; "
          "where its CI excludes zero, execution-graph structure carries localization signal the "
          "position prior does not. The dependency layer is position-equivalent on this full-context "
          "corpus and is excluded by construction, so this is an execution-layer result; a corpus "
          "with observed dependencies and step-level fault labels is what would test the dependency "
          "layer for localization.")


if __name__ == "__main__":
    main()
