"""Keystone experiment: does the two-layer graph predict agent failure better than
its execution-only or flat (size) projections?

For each labeled corpus we build the typed decision graph, extract nested feature
sets with ``auditable.graph.feature_vector`` (flat -> +execution topology ->
+dependency layer), and measure how well a simple detector separates failed from
successful runs. If the full features beat the flat baseline the typed structure is
informative (not merely expressible); and where the dependency layer beats
execution-only, that layer earns its place.

Corpora (label 1 = a FAILED run):
  - tau-bench : db_match,        dependency OBSERVED
  - SWE-agent : issue resolved,  dependency OBSERVED  (heavily imbalanced)
  - Who&When  : is_correct,      dependency INFERRED  -- a failure-only corpus, so
                run-level detection is inapplicable (reported as skipped)

Method: 5 seeds x 5-fold stratified CV, standardized logistic regression (liblinear;
the default lbfgs solver segfaults under this Windows conda BLAS stack), balanced
class weights, ROC-AUC. The flat / exec / full feature sets are nested, so EXEC and
FLAT are column prefixes of the FULL matrix and every fold split is shared across
layers. Adjacent-layer gains are reported as the mean per-seed gain with a seed-block
95% CI: the 5 seeds are the independent-ish unit, while the 5 folds within a seed
share runs and are not independent, so a fold-level test would overstate significance.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python experiment/agent_failure_detection.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from auditable.graph import build_graph, feature_vector

import agent_graph_characterization as ww
import agent_graph_swe_agent as swe
import agent_graph_tau_bench as tau

SEEDS = range(5)
_LAYER_COLS = {"flat": 4, "exec": 8, "full": 12}  # nested prefixes of the full vector
_T975_4 = 2.776  # t(0.975, df=4): seed-block 95% CI half-width over 5 seeds


# --- corpus loaders: each returns (graphs, labels) with label 1 = failed ---------

def load_whoandwhen():
    ww._ensure_corpus()
    graphs, labels = [], []
    for t in ww.load_tasks():
        steps = ww.to_steps(t)
        if len(steps) < 2:
            continue
        ic = str(t.get("is_correct", "")).lower()
        if ic not in ("true", "false"):
            continue  # unlabeled run: cannot use for detection
        graphs.append(build_graph(steps, dependency="full_context"))
        labels.append(1 if ic == "false" else 0)
    return graphs, labels


def load_tau():
    paths = tau._ensure_files()
    graphs, labels = [], []
    for rec in tau.load_runs(paths):
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 3:
            continue
        er = rec.get("eval_result") or {}
        if "db_match" not in er:
            continue  # no gold label for this run
        steps = tau.to_steps(msgs, rec.get("model_path", "model"))
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(0 if bool(er["db_match"]) else 1)
    return graphs, labels


def load_swe():
    graphs, labels = [], []
    for rec in swe.load_runs(swe.N_RUNS):
        steps = swe.to_steps(rec["traj"])
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(0 if rec["target"] else 1)  # target = issue resolved
    return graphs, labels


# --- evaluation ------------------------------------------------------------------

def _matrix(graphs):
    return np.array([feature_vector(G, layer="full")[1] for G in graphs], dtype=float)


def _cv_matrix(X, y, ncols):
    """ROC-AUC per (seed, fold); folds aligned across layers by shared seed splits."""
    Xs = X[:, :ncols]
    rows = []
    for s in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(solver="liblinear", class_weight="balanced"))
        rows.append(cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc"))
    return np.array(rows)  # shape (n_seeds, n_folds)


def evaluate(name, graphs, labels, observed_dep):
    y = np.array(labels)
    n, nf = len(y), int(y.sum())
    tag = "observed dependency" if observed_dep else "INFERRED dependency"
    print(f"\n=== {name}: {n} runs, {nf} failed ({nf / n:.0%} base rate), {tag} ===")
    if nf < 5 or n - nf < 5:
        print("  one class too small for a stable AUC; run-level detection inapplicable")
        return
    X = _matrix(graphs)
    mats = {layer: _cv_matrix(X, y, ncols) for layer, ncols in _LAYER_COLS.items()}
    for layer, ncols in _LAYER_COLS.items():
        a = mats[layer]
        seed_auc = a.mean(axis=1)  # per-seed mean AUC, then a seed-block CI (matches the gains)
        mean = seed_auc.mean()
        half = _T975_4 * seed_auc.std(ddof=1) / np.sqrt(len(seed_auc))
        print(f"  {layer:4s} ({ncols:2d} feat): ROC-AUC {mean:.3f} +/-{half:.3f}")

    def seed_gain(hi, lo):
        g = mats[hi].mean(axis=1) - mats[lo].mean(axis=1)  # per-seed mean gain (len 5)
        half = _T975_4 * g.std(ddof=1) / np.sqrt(len(g))
        return g.mean(), half, int((g > 0).sum()), len(g)

    for hi, lo, lab in (("exec", "flat", "exec - flat"), ("full", "exec", "full - exec")):
        m, half, pos, k = seed_gain(hi, lo)
        note = "   <- does the dependency layer earn its place?" if hi == "full" else ""
        print(f"  gain {lab:11s}: {m:+.3f}  95% CI [{m - half:+.3f}, {m + half:+.3f}]"
              f"  ({pos}/{k} seeds +){note}")


def main():
    print("Keystone: two-layer graph vs execution-only vs flat at predicting agent failure.")
    print("Feature sets are nested (flat subset of exec subset of full); label 1 = failed run.")
    evaluate("tau-bench", *load_tau(), observed_dep=True)
    evaluate("SWE-agent", *load_swe(), observed_dep=True)
    evaluate("Who&When", *load_whoandwhen(), observed_dep=False)
    print("\nReading: execution topology improves over the flat size baseline on "
          "tau-bench's multi-agent runs, while adding the dependency layer improves "
          "over execution-only on both observed-dependency corpora: by a small margin "
          "on tau-bench (a one-hop DB audit surface) and a large, seed-stable margin "
          "on SWE-agent, where the agent is effectively single and execution topology "
          "does not help. The dependency layer, the novel half of the representation, "
          "adds the most where it is richest. The absolute AUCs are modest (structure "
          "is one informative factor in failure, not a full predictor); the layered "
          "gains, with their seed-block CIs, are the result. Two honest limits: the "
          "gains are improvements over a size and count baseline, not a claim that "
          "the features are size-free; and whether INFERRED dependency adds nothing "
          "is not tested at run level here, because the one inferred-dependency corpus "
          "(Who&When) is failure-only.")


if __name__ == "__main__":
    main()
