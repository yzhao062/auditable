"""Graph anomaly detectors over a run's typed graph (POST, run level).

Six detectors read each run as the typed graph the library builds from its steps (decision,
tool-call, agent and resource nodes) and encode it with :func:`auditable.featurize.to_pyg`:
a row per node holding in and out degree and a decision / tool-call / other one-hot, plus the
edges. A run may be given as a typed NetworkX graph, a step list, a ``SessionGraph``, or an
already encoded ``(node_features, edge_index)`` pair.

- ``run.pygod_dominant``, ``run.pygod_conad``, ``run.pygod_anomalydae`` and ``run.pygod_gaan``
  fit one unsupervised PyGOD node detector over the whole batch (transductive); a run's score
  is the mean of its node scores.
- ``run.guardian_lite`` fits a graph-convolutional attribute autoencoder over the whole batch
  (transductive); a run's score is its mean node reconstruction error.
- ``run.gsafeguard_lite`` trains a graph-classification network on labelled runs (inductive,
  supervised) and scores unseen runs with the trained network.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.2
(``detection.PyGODDetection``, ``pygod_extra._PyGODExtraDetection``,
``agent_detectors.guardian_run_scores``, ``agent_detectors.gsafeguard_cv_auc``). The chunked
PyGOD fit is shared with the ``step.pygod_dominant_*`` localizers
(:mod:`auditable.methods._pygod`).
"""
from __future__ import annotations

from typing import Any, List, Sequence

from auditable.base import BaseRunDetector, Citation, MethodCard
from auditable.methods import register

_DOMINANT = Citation(
    key="DOMINANT (Ding et al., 2019)",
    title="Deep Anomaly Detection on Attributed Networks",
    url="https://doi.org/10.1137/1.9781611975673.67",
)
_CONAD = Citation(key="CONAD (Xu et al., 2022)",
                  title="Contrastive Attributed Network Anomaly Detection with Data Augmentation")
_ANOMALYDAE = Citation(key="AnomalyDAE (Fan et al., 2020)",
                       title="AnomalyDAE: Dual Autoencoder for Anomaly Detection on Attributed Networks")
_GAAN = Citation(key="GAAN (Chen et al., 2020)", title="Generative Adversarial Attributed Network Anomaly Detection")
_PYGOD = Citation(key="PyGOD (Liu et al., 2024)", title="PyGOD: A Python Library for Graph Outlier Detection",
                  arxiv="2204.12095")
_GUARDIAN = Citation(key="GUARDIAN (Zhou et al., 2025)",
                     title="GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling",
                     arxiv="2505.19234")
_GSAFEGUARD = Citation(key="G-Safeguard (Wang et al., 2025)",
                       title="G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based "
                             "Multi-agent Systems",
                       arxiv="2502.11127")

_REQUIRES = frozenset({"agent", "kind", "deps"})


def _run_graph(trace: Any):
    """``(node_features, edge_index)`` of one run: an encoded pair passes through, anything
    else is read through ``normalize_trace`` and encoded from its typed graph."""
    if isinstance(trace, tuple) and len(trace) == 2 and hasattr(trace[0], "shape"):
        return trace
    from auditable.featurize import normalize_trace, to_pyg

    return to_pyg(normalize_trace(trace).typed_graph())


def _with_isolated_loops(edges: Any, n: int):
    """``edges`` as a (2, m) int64 array with a self-loop added to each isolated node only,
    so message passing and PyG's samplers are defined on sparse graphs."""
    import numpy as np

    e = np.asarray(edges, dtype=np.int64).reshape(2, -1)
    touched = set(e.flatten().tolist()) if e.shape[1] else set()
    isolated = [v for v in range(n) if v not in touched]
    if isolated:
        loops = np.array(isolated, dtype=np.int64)
        e = np.concatenate([e, np.stack([loops, loops])], axis=1)
    return e


# --- PyGOD node detectors ---------------------------------------------------------------------

class _PyGODRunDetector(BaseRunDetector):
    """A PyGOD node detector fit over the batch of run graphs. Features are standardized over
    the whole batch, isolated nodes get a self-loop, and runs fill chunks of at most
    ``max_chunk_nodes`` nodes in input order (a run is never split), each chunk fit full-batch
    as one disconnected graph. Scores belong to the batch they were fit with."""

    _pygod: str = "DOMINANT"

    def __init__(self, hid_dim: int = 32, num_layers: int = 2, epoch: int = 40, max_chunk_nodes: int = 3000,
                 seed: int = 0) -> None:
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.epoch = epoch
        self.max_chunk_nodes = max_chunk_nodes
        self.seed = seed

    def fit_node_scores(self, X: Sequence[Any], y: Any = None) -> List[Any]:
        """Per-node anomaly scores of every run in ``X``, one array per run in node order."""
        from auditable.methods._pygod import chunked_node_scores

        graphs = [_run_graph(t) for t in X]
        return chunked_node_scores(self._pygod, graphs, hid_dim=self.hid_dim, num_layers=self.num_layers,
                                   epoch=self.epoch, max_chunk_nodes=self.max_chunk_nodes, seed=self.seed)

    def fit_score(self, X: Sequence[Any], y: Any = None):
        """One failure score per run: the mean of its node scores, which stay available as
        ``node_scores_``. ``y`` is ignored."""
        import numpy as np

        self.node_scores_ = self.fit_node_scores(X)
        self.decision_scores_ = np.array([scores.mean() for scores in self.node_scores_])
        return self.decision_scores_


def _pygod_card(method_id: str, name: str, alias: str, cite: tuple, summary: str) -> MethodCard:
    return MethodCard(
        method_id=method_id,
        name=name,
        family="run",
        fit_scope="transductive",
        supervised=False,
        output_kind="score",
        fidelity="wrapper",
        requires=_REQUIRES,
        cite=cite + (_PYGOD,),
        extras=("graph-ad",),
        board_aliases=(alias,),
        benchmark_role="baseline",
        summary=summary,
    )


@register
class PyGODDominant(_PyGODRunDetector):
    """DOMINANT graph autoencoder over the run graphs; a run's score is its mean node
    reconstruction error."""

    _pygod = "DOMINANT"
    card = _pygod_card("run.pygod_dominant", "DOMINANT graph autoencoder (PyGOD)", "pygod (graph AD)",
                       (_DOMINANT,), "Mean node reconstruction error of a DOMINANT autoencoder fit over the batch.")


@register
class PyGODCONAD(_PyGODRunDetector):
    """CONAD (contrastive graph autoencoder) over the run graphs; a run's score is its mean
    node anomaly score."""

    _pygod = "CONAD"
    card = _pygod_card("run.pygod_conad", "CONAD contrastive graph autoencoder (PyGOD)", "pygod-conad",
                       (_CONAD,), "Mean node anomaly score of a CONAD detector fit over the batch.")


@register
class PyGODAnomalyDAE(_PyGODRunDetector):
    """AnomalyDAE (dual structure and attribute autoencoder) over the run graphs; a run's
    score is its mean node anomaly score. PyGOD ignores ``num_layers`` for this detector."""

    _pygod = "AnomalyDAE"
    card = _pygod_card("run.pygod_anomalydae", "AnomalyDAE dual autoencoder (PyGOD)", "pygod-anomalydae",
                       (_ANOMALYDAE,), "Mean node anomaly score of an AnomalyDAE detector fit over the batch.")


@register
class PyGODGAAN(_PyGODRunDetector):
    """GAAN (generative adversarial attributed network detector) over the run graphs; a run's
    score is its mean node anomaly score."""

    _pygod = "GAAN"
    card = _pygod_card("run.pygod_gaan", "GAAN adversarial detector (PyGOD)", "pygod-gaan",
                       (_GAAN,), "Mean node anomaly score of a GAAN detector fit over the batch.")


# --- GUARDIAN-Lite ----------------------------------------------------------------------------

def _batched(graphs: Sequence[Any]):
    """Standardize node features over all graphs and pack them into one disconnected PyG
    ``Batch``, self-looping isolated nodes. Returns the batch and the number of runs."""
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Batch, Data

    sizes = [len(np.asarray(x)) for x, _ in graphs]
    stacked = StandardScaler().fit_transform(
        np.vstack([np.asarray(x, dtype=float) for x, _ in graphs]))
    datas, offset = [], 0
    for (x, edges), n in zip(graphs, sizes):
        node_x = torch.tensor(stacked[offset:offset + n], dtype=torch.float)
        offset += n
        e = _with_isolated_loops(edges, n)
        datas.append(Data(x=node_x, edge_index=torch.tensor(e, dtype=torch.long)))
    return Batch.from_data_list(datas), len(datas)


def _attribute_autoencoder(in_dim: int, hid: int):
    import torch
    from torch_geometric.nn import GCNConv

    class _AE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = GCNConv(in_dim, hid * 2)
            self.enc2 = GCNConv(hid * 2, hid)          # latent bottleneck
            self.dec1 = GCNConv(hid, hid * 2)
            self.dec2 = GCNConv(hid * 2, in_dim)       # reconstruct attributes

        def forward(self, x, edge_index):
            z = self.enc2(torch.relu(self.enc1(x, edge_index)), edge_index)
            x_hat = self.dec2(torch.relu(self.dec1(z, edge_index)), edge_index)
            return x_hat

    return _AE()


@register
class GuardianLite(BaseRunDetector):
    """GUARDIAN-Lite: a directed-GCN attribute-reconstruction autoencoder, a simplified
    stand-in for GUARDIAN (Zhou et al., 2025). It has no temporal graph model, drops the
    adjacency-reconstruction term and the information-bottleneck compression, and keeps only
    the attribute-reconstruction objective, which stays structure-aware through message
    passing. Unsupervised: it never sees a failure label.

    The autoencoder (two encoder and two decoder GCN layers) is fit full-batch with Adam on the
    standardized node features of all runs at once. A run's score is its mean node
    reconstruction error, averaged over ``n_inits`` initializations (seeds ``seed``,
    ``seed + 1``, ...) because one initialization is noisy at this graph size. Transductive:
    the scores belong to the fitted batch.
    """

    card = MethodCard(
        method_id="run.guardian_lite",
        name="GUARDIAN-Lite (attribute-reconstruction autoencoder)",
        family="run",
        fit_scope="transductive",
        supervised=False,
        output_kind="score",
        fidelity="simplified",
        requires=_REQUIRES,
        cite=(_GUARDIAN,),
        extras=("graph-ad",),
        board_aliases=("guardian (recon-AE)",),
        benchmark_role="baseline",
        summary="Mean node reconstruction error of a graph autoencoder fit over the batch, over three seeds.",
    )

    def __init__(self, hid: int = 16, epochs: int = 50, n_inits: int = 3, seed: int = 0) -> None:
        self.hid = hid
        self.epochs = epochs
        self.n_inits = n_inits
        self.seed = seed

    def fit_score(self, X: Sequence[Any], y: Any = None):
        """One failure score per run, in input order. ``y`` is ignored."""
        import numpy as np
        import torch

        batch, n_runs = _batched([_run_graph(t) for t in X])
        in_dim = batch.x.shape[1]

        per_seed = []
        for s in range(self.n_inits):
            torch.manual_seed(self.seed + s)
            model = _attribute_autoencoder(in_dim, self.hid)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                x_hat = model(batch.x, batch.edge_index)
                loss = ((batch.x - x_hat) ** 2).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                x_hat = model(batch.x, batch.edge_index)
                node_err = ((batch.x - x_hat) ** 2).mean(dim=1).numpy()
            per_seed.append(np.array([node_err[batch.batch.numpy() == k].mean() for k in range(n_runs)]))
        self.decision_scores_ = np.mean(per_seed, axis=0)
        return self.decision_scores_


# --- G-Safeguard-Lite -------------------------------------------------------------------------

def _to_data(graph: Any, scaler: Any):
    """One PyG ``Data`` from an encoded graph: node features through an already fitted
    scaler, isolated nodes self-looped."""
    import numpy as np
    import torch
    from torch_geometric.data import Data

    x, edges = graph
    node_x = torch.tensor(scaler.transform(np.asarray(x, dtype=float)), dtype=torch.float)
    e = _with_isolated_loops(edges, node_x.shape[0])
    return Data(x=node_x, edge_index=torch.tensor(e, dtype=torch.long))


def _graph_classifier(in_dim: int, hid: int):
    import torch
    from torch_geometric.nn import GCNConv, global_mean_pool

    class _GNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = GCNConv(in_dim, hid)
            self.c2 = GCNConv(hid, hid)
            self.lin = torch.nn.Linear(hid, 1)

        def forward(self, b):
            h = torch.relu(self.c1(b.x, b.edge_index))
            h = torch.relu(self.c2(h, b.edge_index))
            return self.lin(global_mean_pool(h, b.batch)).squeeze(-1)

    return _GNN()


@register
class GSafeguardLite(BaseRunDetector):
    """G-Safeguard-Lite: a supervised graph-classification network, a simplified stand-in for
    G-Safeguard (Wang et al., 2025). The published method detects anomalous agents on a
    multi-agent utterance graph and then remediates the topology. This port classifies whole
    runs from their dependency graph and has no remediation step, because the benchmark scores
    detection and not intervention.

    Two GCN layers, mean pooling and a linear head are trained full-batch with Adam on a
    binary cross-entropy loss (``epochs`` steps, ``y`` = 1 for a failed run). Node features are
    standardized with a scaler fit on the training graphs only. ``fit`` seeds torch with
    ``seed`` before it builds the network, so a fixed ``seed`` fixes the initialization;
    cross-validated seed averaging is the caller's job (see the CatchBench parity protocol).
    ``decision_function`` returns the failure probability, the sigmoid of the network output,
    which is what the CatchBench port ranks.
    """

    card = MethodCard(
        method_id="run.gsafeguard_lite",
        name="G-Safeguard-Lite (supervised graph classifier)",
        family="run",
        fit_scope="inductive",
        supervised=True,
        output_kind="score",
        fidelity="simplified",
        requires=_REQUIRES,
        cite=(_GSAFEGUARD,),
        extras=("graph-ad",),
        board_aliases=("g-safeguard (sup GNN)",),
        benchmark_role="baseline",
        summary="Two-layer GCN with mean pooling trained on labelled run graphs; scores are failure probabilities.",
    )

    def __init__(self, hid: int = 32, epochs: int = 60, seed: int = 0) -> None:
        self.hid = hid
        self.epochs = epochs
        self.seed = seed

    def features(self, X: Sequence[Any]) -> List[Any]:
        """The encoded ``(node_features, edge_index)`` pair of each run; encode once and pass
        the pairs to ``fit`` and ``decision_function`` when the same runs are scored often
        (cross-validation)."""
        return [_run_graph(t) for t in X]

    def fit(self, X: Sequence[Any], y: Any = None) -> "GSafeguardLite":
        import numpy as np
        import torch
        from sklearn.preprocessing import StandardScaler
        from torch_geometric.data import Batch

        if y is None:
            raise ValueError(f"{type(self).__name__} is supervised; pass run labels y (1 = failed)")
        graphs = self.features(X)
        y = np.asarray(y)
        in_dim = np.asarray(graphs[0][0]).shape[1]
        self.scaler_ = StandardScaler().fit(np.vstack([np.asarray(x, dtype=float) for x, _ in graphs]))
        batch = Batch.from_data_list([_to_data(g, self.scaler_) for g in graphs])
        target = torch.tensor(y, dtype=torch.float)

        torch.manual_seed(self.seed)  # after the batch is built, immediately before the network
        model = _graph_classifier(in_dim, self.hid)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            opt.step()
        model.eval()
        self.model_ = model
        self.decision_scores_ = self._probability(batch)
        return self

    def _probability(self, batch: Any):
        import torch

        with torch.no_grad():
            return torch.sigmoid(self.model_(batch)).numpy().astype(float)

    def _fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, y) first")

    def decision_function(self, X: Sequence[Any]):
        """The failure probability of each run in ``X``, in input order."""
        from torch_geometric.data import Batch

        self._fitted()
        batch = Batch.from_data_list([_to_data(g, self.scaler_) for g in self.features(X)])
        return self._probability(batch)

    def predict_proba(self, X: Sequence[Any]):
        """``(n, 2)`` class probabilities, columns ordered resolved then failed."""
        import numpy as np

        p = self.decision_function(X)
        return np.column_stack([1.0 - p, p])
