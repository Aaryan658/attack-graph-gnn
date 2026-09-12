"""
test_algorithms.py -- pytest wrapper around the correctness checks already
built into each algorithm module's __main__ demo (src/algorithms/*.py).

These tests need no GPU and no dataset -- every graph here is a small
hard-coded toy instance, so `pytest tests/` runs anywhere, including CI.
"""

import math

import pytest

from src.algorithms.warshall import transitive_closure, reachable_set
from src.algorithms.floyd_warshall import (floyd_warshall, reconstruct_path,
                                           weight_matrix_from_propensity, INF)
from src.algorithms.dijkstra import dijkstra, adjacency_from_propensity
from src.algorithms.kruskal import (kruskal, DisjointSet,
                                    undirected_edges_from_directed)
from src.algorithms.prim import prim, adjacency_from_edges


# --------------------------------------------------------------------------- #
# Warshall
# --------------------------------------------------------------------------- #
def _cycle_and_tail_graph():
    labels = ["A", "B", "C", "D", "E", "F"]
    idx = {c: i for i, c in enumerate(labels)}
    edges = [("A", "B"), ("B", "C"), ("C", "A"),
             ("C", "D"), ("D", "E"), ("F", "D")]
    n = len(labels)
    reach = [[False] * n for _ in range(n)]
    for u, v in edges:
        reach[idx[u]][idx[v]] = True
    return labels, idx, reach


def test_warshall_transitive_closure():
    labels, idx, reach = _cycle_and_tail_graph()
    transitive_closure(reach)

    cycle = {idx[c] for c in "ABC"}
    tail = {idx[c] for c in "DE"}
    for start in cycle:
        # A->B->C->A is a real cycle, so each cycle member reaches ITSELF too
        # (via the loop) -- the expected set includes `start`, not excludes it.
        assert set(reachable_set(reach, start)) == cycle | tail, (
            "every node in the A-B-C cycle should reach the whole cycle "
            "(itself included, via the loop) plus the whole D-E tail"
        )
    assert reachable_set(reach, idx["E"]) == [], "E is a sink"
    assert reachable_set(reach, idx["F"]) == sorted(idx[c] for c in "DE"), (
        "F only feeds into the tail"
    )


def test_warshall_rejects_ragged_matrix():
    bad = [[False, False], [False]]
    with pytest.raises(ValueError):
        transitive_closure(bad)


# --------------------------------------------------------------------------- #
# Floyd-Warshall
# --------------------------------------------------------------------------- #
def test_floyd_warshall_shortest_paths():
    n = 5
    edges = [(0, 1, 4.0), (0, 2, 1.0), (2, 1, 2.0),
             (1, 3, 1.0), (2, 3, 5.0), (3, 4, 3.0)]
    weight = [[INF] * n for _ in range(n)]
    for u, v, w in edges:
        weight[u][v] = w

    dist, nxt = floyd_warshall(weight)

    assert dist[0][3] == 4.0, "cheapest 0->3 is 0->2->1->3 (1+2+1=4)"
    assert reconstruct_path(nxt, 0, 3) == [0, 2, 1, 3]
    assert dist[0][4] == 7.0
    assert dist[4][0] == INF, "node 4 is a sink"
    assert dist[0][0] == 0.0


def test_floyd_warshall_negative_cycle_detected():
    n = 2
    weight = [[INF] * n for _ in range(n)]
    weight[0][1] = -1.0
    weight[1][0] = -1.0
    with pytest.raises(ValueError):
        floyd_warshall(weight)


def test_weight_matrix_from_propensity_is_nonnegative():
    prop = [[0.0, 1.0], [0.5, 0.0]]
    present = [[False, True], [True, False]]
    w = weight_matrix_from_propensity(prop, present)
    assert w[0][1] > 0, "propensity 1.0 must still map to a strictly positive cost"
    assert abs(w[1][0] - 0.5) < 1e-6
    assert w[0][0] == 0.0 and w[1][1] == 0.0


# --------------------------------------------------------------------------- #
# Dijkstra
# --------------------------------------------------------------------------- #
def test_dijkstra_classic_instance():
    adj = {
        0: [(1, 7.0), (2, 9.0), (5, 14.0)],
        1: [(2, 10.0), (3, 15.0)],
        2: [(3, 11.0), (5, 2.0)],
        3: [(4, 6.0)],
        5: [(4, 9.0)],
    }
    result = dijkstra(adj, 0, target=4)
    assert result.dist[4] == 20.0
    assert result.path_to(4) == [0, 2, 5, 4]


def test_dijkstra_rejects_negative_weight():
    with pytest.raises(ValueError):
        dijkstra({0: [(1, -1.0)]}, 0)


def test_dijkstra_unreachable_target():
    adj = {0: [(1, 1.0)], 2: [(3, 1.0)]}   # {0,1} and {2,3} disconnected
    result = dijkstra(adj, 0, target=3)
    assert result.path_to(3) == []
    assert result.dist[3] == math.inf


def test_adjacency_from_propensity_weights_are_positive():
    prop = [[0.0, 1.0], [0.0, 0.0]]
    adj = adjacency_from_propensity(prop, [(0, 1)])
    assert adj[0][0][0] == 1        # edge target
    assert adj[0][0][1] > 0         # weight strictly positive even at propensity=1


# --------------------------------------------------------------------------- #
# Kruskal + DisjointSet
# --------------------------------------------------------------------------- #
_CLRS_N = 7
_CLRS_EDGES = [
    (0, 1, 7.0), (0, 3, 5.0), (1, 2, 8.0), (1, 3, 9.0), (1, 4, 7.0),
    (2, 4, 5.0), (3, 4, 15.0), (3, 5, 6.0), (4, 5, 8.0), (4, 6, 9.0),
    (5, 6, 11.0),
]


def test_kruskal_clrs_mst():
    result = kruskal(_CLRS_N, _CLRS_EDGES)
    assert result.total_weight == 39.0
    assert len(result.mst_edges) == _CLRS_N - 1
    assert result.num_components == 1


def test_disjoint_set_union_find():
    dsu = DisjointSet(5)
    assert dsu.union(0, 1) is True
    assert dsu.union(0, 1) is False, "already connected -- second union is a no-op"
    assert dsu.connected(0, 1) is True
    assert dsu.connected(0, 2) is False
    dsu.union(2, 3)
    dsu.union(1, 2)
    assert dsu.connected(0, 3) is True, "unions should chain by transitivity"
    assert dsu.num_sets == 2  # {0,1,2,3} and {4}


def test_undirected_edges_from_directed_keeps_min_weight():
    directed = [(0, 1, 3.0), (1, 0, 1.0), (0, 0, 5.0)]
    und = undirected_edges_from_directed(directed)
    assert und == [(0, 1, 1.0)], "cheaper direction wins; self-loop dropped"


# --------------------------------------------------------------------------- #
# Prim -- must agree with Kruskal on the identical graph
# --------------------------------------------------------------------------- #
def test_prim_matches_kruskal():
    adj = adjacency_from_edges(_CLRS_N, _CLRS_EDGES)
    prim_result = prim(_CLRS_N, adj, start=0)
    kruskal_result = kruskal(_CLRS_N, _CLRS_EDGES)

    assert prim_result.total_weight == 39.0
    assert prim_result.total_weight == kruskal_result.total_weight
    assert len(prim_result.mst_edges) == _CLRS_N - 1


def test_prim_reports_disconnected_graph():
    # Two disjoint triangles: Prim from node 0 can only span its own component.
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0),
             (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0)]
    adj = adjacency_from_edges(6, edges)
    result = prim(6, adj, start=0)
    assert len(result.mst_edges) == 2          # only the {0,1,2} triangle
    assert result.num_components == 4          # 1 spanned + 3 unreached singles
