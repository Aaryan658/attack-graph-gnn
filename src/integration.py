"""
integration.py -- PART 3 bridge between the GNN and the classical algorithms.

The GNN emits ONE propensity score per existing directed edge. The five
hand-written algorithms in :mod:`src.algorithms` each want a different plain
data structure:

    Warshall's         -- V x V boolean reachability matrix
    Floyd-Warshall     -- V x V float weight matrix (inf = no edge, 0 diagonal)
    Dijkstra's         -- adjacency list  {u: [(v, weight), ...]}
    Kruskal's / Prim's -- undirected weighted edge list  [(u, v, weight), ...]

This module:
  1. runs the trained model on one :class:`AttackGraphSample`
     (:class:`GNNInference`),
  2. scatters the per-edge scores back into a dense ``(V, V)`` propensity
     matrix,
  3. produces every structure above from that matrix
     (:func:`build_algorithm_inputs`),
  4. offers :func:`largest_connected_component` so the two MST algorithms can be
     compared on a guaranteed-connected sub-graph.

Weight convention (matches the algorithm modules): ``cost = 1 - propensity``.
A high-propensity edge is cheap, so "cheapest path" == "most plausible attack
path".
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .data_loader import AttackGraphSample, load_sample
from .model import build_model_from_config, GraphSAGEEdgeClassifier
from .algorithms.warshall import reachability_from_propensity
from .algorithms.floyd_warshall import weight_matrix_from_propensity
from .algorithms.dijkstra import adjacency_from_propensity, Adjacency
from .algorithms.kruskal import undirected_edges_from_directed, DisjointSet, Edge

INF = math.inf


# ========================================================================== #
# GNN inference
# ========================================================================== #
class GNNInference:
    """
    Wraps a trained checkpoint and produces per-edge propensities for a sample.

    Usage
    -----
        inf = GNNInference.from_checkpoint("checkpoints/latest.pt", device)
        propensity_edges = inf.predict(sample)     # (M,) aligned to
                                                   # sample.edge_index
    """

    def __init__(self, model: GraphSAGEEdgeClassifier, device: torch.device,
                 meta: Optional[dict] = None) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.meta = meta or {}

    @staticmethod
    def from_checkpoint(path: str, device: torch.device) -> "GNNInference":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No checkpoint at {path!r}. Train first: "
                f"`python -m src.train`, or pass --train-if-missing to main.py."
            )
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_model_from_config(ckpt["model_config"])
        model.load_state_dict(ckpt["model_state"])
        return GNNInference(model, device, meta={
            "checkpoint": path,
            "epoch": ckpt.get("epoch"),
            "best_test_auc": ckpt.get("best_test_auc"),
        })

    @torch.no_grad()
    def predict(self, sample: AttackGraphSample) -> torch.Tensor:
        """Return a length-M CPU tensor of propensities in [0, 1]."""
        x = sample.x.to(self.device)
        edge_index = sample.edge_index.to(self.device)
        edge_attr = sample.edge_type_multihot.to(self.device)
        probs = self.model.predict_propensity(x, edge_index, edge_attr)
        return probs.detach().cpu()


# ========================================================================== #
# Per-edge scores -> dense (V, V) propensity matrix
# ========================================================================== #
def scatter_propensity_matrix(
    sample: AttackGraphSample,
    edge_propensity: torch.Tensor,
) -> Tuple[List[List[float]], List[List[bool]]]:
    """
    Place each edge's propensity at ``matrix[u][v]``.

    Returns
    -------
    propensity : V x V list of floats  -- 0.0 where no edge exists.
    present    : V x V list of bools   -- True where a directed edge exists.
    """
    V = sample.num_nodes
    ei = sample.edge_index
    M = ei.shape[1]
    if edge_propensity.numel() != M:
        raise ValueError(
            f"edge_propensity has {edge_propensity.numel()} entries but the "
            f"sample has {M} edges."
        )

    propensity = [[0.0] * V for _ in range(V)]
    present = [[False] * V for _ in range(V)]
    src = ei[0].tolist()
    dst = ei[1].tolist()
    scores = edge_propensity.tolist()
    for u, v, p in zip(src, dst, scores):
        propensity[u][v] = float(p)
        present[u][v] = True
    return propensity, present


# ========================================================================== #
# Build every algorithm's input structure
# ========================================================================== #
@dataclass
class AlgorithmInputs:
    """All the plain structures the five algorithms need, for one graph."""

    num_nodes: int
    propensity: List[List[float]]          # (V, V) dense, 0 where no edge
    present: List[List[bool]]              # (V, V) edge mask
    threshold: float

    # Warshall's
    reachability: List[List[bool]]         # propensity >= threshold

    # Floyd-Warshall
    weight_matrix: List[List[float]]       # 1 - propensity, inf off-edges

    # Dijkstra's
    adjacency: Adjacency                   # {u: [(v, 1 - p + eps), ...]}

    # Kruskal's / Prim's  (undirected view)
    undirected_edges: List[Edge]           # [(u, v, 1 - p_min), ...]

    directed_edges: List[Edge]             # [(u, v, 1 - p), ...]  (for reference)


def build_algorithm_inputs(
    sample: AttackGraphSample,
    edge_propensity: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-9,
) -> AlgorithmInputs:
    """
    Turn one graph + its per-edge propensities into :class:`AlgorithmInputs`.
    """
    V = sample.num_nodes
    propensity, present = scatter_propensity_matrix(sample, edge_propensity)

    reachability = reachability_from_propensity(propensity, threshold)
    weight_matrix = weight_matrix_from_propensity(propensity, present, epsilon)

    edge_list = sample.edge_list()
    adjacency = adjacency_from_propensity(propensity, edge_list, epsilon)

    directed_edges: List[Edge] = [
        (u, v, (1.0 - propensity[u][v]) + epsilon) for (u, v) in edge_list
    ]
    undirected_edges = undirected_edges_from_directed(directed_edges)

    return AlgorithmInputs(
        num_nodes=V,
        propensity=propensity,
        present=present,
        threshold=threshold,
        reachability=reachability,
        weight_matrix=weight_matrix,
        adjacency=adjacency,
        undirected_edges=undirected_edges,
        directed_edges=directed_edges,
    )


# ========================================================================== #
# Connected-component helper (uses our own Union-Find)
# ========================================================================== #
def largest_connected_component(
    num_nodes: int,
    undirected_edges: Sequence[Edge],
) -> Tuple[List[int], List[Edge], Dict[int, int]]:
    """
    Find the largest connected component of an undirected graph and return it
    relabelled to a dense ``0 .. k-1`` range.

    Returns
    -------
    members      : original vertex ids in the component (sorted).
    local_edges  : edges of the component, endpoints relabelled to 0..k-1.
    relabel      : {original_id: local_id}.
    """
    dsu = DisjointSet(num_nodes)
    for u, v, _w in undirected_edges:
        dsu.union(u, v)

    # Bucket vertices by their component root.
    buckets: Dict[int, List[int]] = {}
    for node in range(num_nodes):
        buckets.setdefault(dsu.find(node), []).append(node)
    if not buckets:
        return [], [], {}

    biggest_root = max(buckets, key=lambda r: len(buckets[r]))
    members = sorted(buckets[biggest_root])
    relabel = {orig: i for i, orig in enumerate(members)}
    member_set = set(members)

    local_edges: List[Edge] = [
        (relabel[u], relabel[v], w)
        for (u, v, w) in undirected_edges
        if u in member_set and v in member_set
    ]
    return members, local_edges, relabel


# ========================================================================== #
# Demo -- synthetic propensities, no checkpoint required.
# ========================================================================== #
def _demo() -> None:
    import glob

    found = sorted(glob.glob(os.path.join("data", "_data_", "*.pt")))
    if not found:
        raise SystemExit("no dataset; extract data/_data_.zip first")
    sample = load_sample(found[0])

    # Fake "GNN output": propensity = fraction of edge types present on the edge,
    # nudged into (0.05, 0.95). Deterministic, no model needed.
    et = sample.edge_type_multihot
    fake_prop = (et.sum(dim=1) / et.shape[1]).clamp(0.05, 0.95)

    inp = build_algorithm_inputs(sample, fake_prop, threshold=0.15)
    print(f"graph {os.path.basename(sample.source_path)}  V={inp.num_nodes}")
    print(f"  directed edges         : {len(inp.directed_edges)}")
    print(f"  undirected edges       : {len(inp.undirected_edges)}")
    n_reach = sum(c for row in inp.reachability for c in row)
    print(f"  reachable pairs (>=thr): {n_reach}")
    members, local_edges, _ = largest_connected_component(
        inp.num_nodes, inp.undirected_edges)
    print(f"  largest component      : {len(members)} nodes, "
          f"{len(local_edges)} edges")
    finite = sum(1 for row in inp.weight_matrix for c in row if c != INF)
    print(f"  finite weight entries  : {finite} "
          f"(== directed edges + V diagonal = "
          f"{len(inp.directed_edges) + inp.num_nodes})")
    print("integration self-check OK")


if __name__ == "__main__":
    _demo()
