"""Chunked full-batch scoring with PyGOD graph anomaly detectors.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.2: ``graph_ad.flat_disconnected``, ``graph_ad.pygod_node_scores`` and
``pygod_extra._pygod_extra_node_scores``. The order of operations is kept exactly, because
the scores depend on it: seed torch and NumPy globally, standardize node features over the
whole batch, add self-loops to isolated nodes only, then fit one detector per chunk of at
most ``max_chunk_nodes`` nodes, chunks filled in input order with a run's nodes never split.
"""
from __future__ import annotations

import os
from typing import Any, List, Sequence

_DETECTORS = ("DOMINANT", "CONAD", "AnomalyDAE", "GAAN")


def flat_disconnected(datas: Sequence[Any], members: Sequence[int]):
    """Join several per-run graphs into one disconnected PyG ``Data`` and return it with an
    integer membership vector (each node's index into ``members``).

    A PyG ``Batch`` is avoided on purpose: PyGOD's ``DeepDetector`` reads ``batch_size`` off
    the object its loader yields and scores only that many nodes, so a ``Batch`` of g graphs
    would score the chunk's first g nodes and leave the rest at zero. A plain ``Data`` has
    no such attribute, and every node is scored.
    """
    import numpy as np
    import torch
    from torch_geometric.data import Data

    xs, edges, membership, offset = [], [], [], 0
    for local_k, gi in enumerate(members):
        d = datas[gi]
        n = int(d.x.shape[0])
        xs.append(d.x)
        edges.append(d.edge_index + offset)
        membership.append(np.full(n, local_k, dtype=np.int64))
        offset += n
    joined = Data(x=torch.cat(xs, dim=0), edge_index=torch.cat(edges, dim=1))
    return joined, np.concatenate(membership)


def chunked_node_scores(detector: str, graphs: Sequence[Any], *, hid_dim: int = 32,
                        num_layers: int = 2, epoch: int = 40, max_chunk_nodes: int = 3000,
                        seed: int = 0, **detector_kwargs: Any) -> List[Any]:
    """Fit PyGOD ``detector`` (one of DOMINANT, CONAD, AnomalyDAE, GAAN) over a batch of
    per-run graphs ``(node_features, edge_index)`` and return per-node anomaly scores split
    back per run, each array in the graph's node order. Transductive: the scores belong to
    this batch and change if the batch changes."""
    if detector not in _DETECTORS:
        raise ValueError(f"detector must be one of {_DETECTORS}, got {detector!r}")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Data
    import pygod.detector as pd

    detector_cls = getattr(pd, detector)

    torch.manual_seed(seed)
    np.random.seed(seed)

    sizes = [len(np.asarray(x)) for x, _ in graphs]
    stacked = StandardScaler().fit_transform(
        np.vstack([np.asarray(x, dtype=float) for x, _ in graphs]))

    datas, offset = [], 0
    for (x, edges), n in zip(graphs, sizes):
        node_x = torch.tensor(stacked[offset:offset + n], dtype=torch.float)
        offset += n
        e = np.asarray(edges, dtype=np.int64).reshape(2, -1)
        touched = set(e.flatten().tolist()) if e.shape[1] else set()
        isolated = [v for v in range(n) if v not in touched]  # only these trip PyG's sampler
        if isolated:
            loops = np.array(isolated, dtype=np.int64)
            e = np.concatenate([e, np.stack([loops, loops])], axis=1)
        datas.append(Data(x=node_x, edge_index=torch.tensor(e, dtype=torch.long)))

    results: List[Any] = [np.zeros(0)] * len(datas)

    def _fit_chunk(members: list) -> None:
        if not members:
            return
        joined, membership = flat_disconnected(datas, members)
        model = detector_cls(hid_dim=hid_dim, num_layers=num_layers, epoch=epoch, batch_size=0,
                             gpu=-1, verbose=0, **detector_kwargs)
        model.fit(joined)
        scores = np.asarray(model.decision_score_, dtype=float)
        for local_k, gi in enumerate(members):
            results[gi] = scores[membership == local_k]

    members, chunk_nodes = [], 0
    for gi, n in enumerate(sizes):
        if members and chunk_nodes + n > max_chunk_nodes:
            _fit_chunk(members)
            members, chunk_nodes = [], 0
        members.append(gi)
        chunk_nodes += n
    _fit_chunk(members)
    return results
