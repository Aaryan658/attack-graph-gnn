"""
floyd_warshall.py -- Floyd-Warshall all-pairs shortest paths on a weighted
directed graph, with path reconstruction.

PROBLEM
-------
Given a V x V weight matrix ``weight`` where ``weight[i][j]`` is the cost of the
direct edge i -> j (``math.inf`` if there is no such edge, ``0`` on the
diagonal), compute ``dist[i][j]`` = the cost of the cheapest path from i to j
for *every* ordered pair (i, j), plus enough bookkeeping to rebuild the actual
paths.

In this project the edge costs come from the GNN: ``cost = 1 - propensity``.
A high-propensity edge (the model thinks it is part of an attack path) is
*cheap*; a low-propensity edge is *expensive*. The cheapest i -> j path is then
the most plausible full attack route from i to j.

ALGORITHM (Floyd 1962; Warshall 1962)
-------------------------------------
Same "admit one intermediate vertex at a time" idea as Warshall's, but on
numeric path costs instead of booleans::

    dist[i][j] = min( dist[i][j],
                      dist[i][k] + dist[k][j] )

After the k-loop finishes for a given ``k``, ``dist[i][j]`` is the shortest
path using intermediate vertices only from {0..k}. When ``k`` has swept the
whole vertex set, ``dist`` holds the true shortest-path costs.

Path reconstruction: ``nxt[i][j]`` stores "from i, heading toward j, the first
hop to take". It starts as ``j`` for every real edge i -> j, and whenever we
relax (i, j) through k we copy ``nxt[i][j] = nxt[i][k]``. Following ``nxt``
from i rebuilds the path.

COMPLEXITY
----------
Time  : O(V^3)   -- three nested loops.
Space : O(V^2)   -- ``dist`` and ``nxt`` matrices.

NEGATIVE CYCLES
---------------
If any ``dist[i][i]`` becomes negative, the graph has a negative-weight cycle
and "shortest path" is undefined for pairs routed through it. With
``cost = 1 - propensity`` in [0, 1] this cannot happen, but the check is cheap
and we surface it rather than return nonsense.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

INF = math.inf
FloatMatrix = List[List[float]]
IntMatrix = List[List[Optional[int]]]


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
def floyd_warshall(
    weight: FloatMatrix,
    trace: Optional[list] = None,
    snapshot_matrices: bool = True,
) -> Tuple[FloatMatrix, IntMatrix]:
    """
    Run Floyd-Warshall on ``weight`` and return ``(dist, nxt)``.

    Parameters
    ----------
    weight
        V x V matrix of direct edge costs. ``math.inf`` marks "no edge". The
        diagonal is treated as 0 regardless of what is passed in.
    trace
        Optional list; when given, a dict is appended for the initial state and
        after each intermediate vertex ``k``, listing the (i, j) pairs whose
        distance improved on that pass.
    snapshot_matrices
        When True (default, good for toy graphs) each trace entry also carries a
        full copy of the ``dist`` matrix (``inf`` rendered as ``"inf"`` for
        JSON). Set False for large graphs: the ``improved_pairs`` deltas plus
        the ``init`` edge costs still allow an animation to replay every
        intermediate matrix without writing a V*V grid V times.

    Returns
    -------
    dist
        V x V matrix of shortest-path costs (``math.inf`` if unreachable).
    nxt
        V x V "next hop" matrix for path reconstruction; ``nxt[i][j]`` is None
        when j is unreachable from i.
    """
    n = len(weight)
    for r, row in enumerate(weight):
        if len(row) != n:
            raise ValueError(
                f"floyd_warshall expects a square matrix; row {r} has length "
                f"{len(row)}, expected {n}."
            )

    # --- Initialise dist and nxt from the direct edges ------------------- #
    dist: FloatMatrix = [[INF] * n for _ in range(n)]
    nxt: IntMatrix = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0.0          # zero-cost self path
                nxt[i][j] = i
            elif weight[i][j] != INF:
                dist[i][j] = float(weight[i][j])
                nxt[i][j] = j             # first hop of i -> j is j itself

    if trace is not None:
        init_entry = {
            "step": "init",
            "k": None,
            "improved_pairs": [],
            # Compact seed for an animation: the direct edge costs.
            "initial_edges": [[i, j, round(float(weight[i][j]), 6)]
                              for i in range(n) for j in range(n)
                              if i != j and weight[i][j] != INF],
        }
        if snapshot_matrices:
            init_entry["dist"] = _jsonable(dist)
        trace.append(init_entry)

    # --- Triple loop ---------------------------------------------------------- #
    for k in range(n):
        improved_pairs: List[List[int]] = []
        dist_k = dist[k]

        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                # i cannot reach k, so no path i -> k -> j can help.
                continue
            row_i = dist[i]
            nxt_i = nxt[i]

            for j in range(n):
                through_k = dik + dist_k[j]
                if through_k < row_i[j]:
                    row_i[j] = through_k
                    # To go from i toward j, first move the same way you would
                    # to head toward k.
                    nxt_i[j] = nxt[i][k]
                    improved_pairs.append([i, j])

        if trace is not None:
            entry = {
                "step": f"k={k}",
                "k": k,
                "improved_pairs": improved_pairs,
            }
            if snapshot_matrices:
                entry["dist"] = _jsonable(dist)
            trace.append(entry)

    _check_negative_cycle(dist)
    return dist, nxt


# --------------------------------------------------------------------------- #
# Path reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_path(nxt: IntMatrix, source: int, target: int) -> List[int]:
    """
    Rebuild the shortest path ``source -> ... -> target`` from the ``nxt``
    matrix produced by :func:`floyd_warshall`.

    Returns the list of vertices (inclusive of both ends), or an empty list if
    ``target`` is unreachable from ``source``.
    """
    if nxt[source][target] is None:
        return []
    path = [source]
    node = source
    # Walk the "next hop" pointers until we arrive at target. The guard on
    # path length is a safety net against a malformed nxt matrix.
    while node != target:
        node = nxt[node][target]
        if node is None or len(path) > len(nxt):
            raise ValueError("reconstruct_path: corrupt 'nxt' matrix.")
        path.append(node)
    return path


# --------------------------------------------------------------------------- #
# GNN output -> weight matrix
# --------------------------------------------------------------------------- #
def weight_matrix_from_propensity(
    propensity: List[List[float]],
    edge_present: Optional[List[List[bool]]] = None,
    epsilon: float = 1e-9,
) -> FloatMatrix:
    """
    Build the Floyd-Warshall weight matrix from GNN propensities.

    ``cost[i][j] = 1 - propensity[i][j]`` for edges that exist, ``math.inf``
    otherwise, ``0`` on the diagonal.

    Parameters
    ----------
    propensity
        V x V predicted attack-propensity scores in [0, 1].
    edge_present
        Optional V x V boolean mask of which edges physically exist in the
        graph. If omitted, every off-diagonal pair is treated as a candidate
        edge (dense graph).
    epsilon
        Tiny floor added to each cost so that a propensity of exactly 1.0 does
        not create a genuinely zero-cost edge (which would make path lengths
        ambiguous).
    """
    n = len(propensity)
    cost: FloatMatrix = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                cost[i][j] = 0.0
            elif edge_present is None or edge_present[i][j]:
                cost[i][j] = (1.0 - float(propensity[i][j])) + epsilon
    return cost


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _check_negative_cycle(dist: FloatMatrix) -> None:
    for i, row in enumerate(dist):
        if row[i] < 0:
            raise ValueError(
                f"Negative-weight cycle detected (dist[{i}][{i}] = {row[i]}). "
                f"Shortest paths are undefined."
            )


def _jsonable(dist: FloatMatrix) -> List[List[object]]:
    """Copy a distance matrix with inf -> 'inf' so json.dump won't choke."""
    return [["inf" if x == INF else round(x, 6) for x in row] for row in dist]


# --------------------------------------------------------------------------- #
# Stand-alone demonstration
# --------------------------------------------------------------------------- #
def _demo() -> None:
    # Toy weighted directed graph, 5 nodes (0..4).
    #   0 -> 1 (4)    0 -> 2 (1)
    #   2 -> 1 (2)    1 -> 3 (1)
    #   2 -> 3 (5)    3 -> 4 (3)
    # Cheapest 0 -> 3 is 0->2->1->3 with cost 1 + 2 + 1 = 4 (not the direct-ish
    # 0->1->3 = 5, nor 0->2->3 = 6).
    n = 5
    edges = [(0, 1, 4.0), (0, 2, 1.0), (2, 1, 2.0),
             (1, 3, 1.0), (2, 3, 5.0), (3, 4, 3.0)]

    weight: FloatMatrix = [[INF] * n for _ in range(n)]
    for u, v, w in edges:
        weight[u][v] = w

    print("=" * 60)
    print("Floyd-Warshall -- all-pairs shortest paths demo")
    print("=" * 60)
    print(f"Edges (u -> v : cost): {edges}")

    trace: list = []
    dist, nxt = floyd_warshall(weight, trace=trace)

    for snapshot in trace:
        if snapshot["step"] == "init":
            continue
        improved = [f"{i}->{j}" for i, j in snapshot["improved_pairs"]]
        print(f"\nAfter admitting intermediate vertex {snapshot['k']}: "
              f"improved pairs = {improved if improved else 'none'}")

    print("\nFinal shortest-path cost matrix (. = unreachable):")
    print("     " + " ".join(f"{j:>5}" for j in range(n)))
    for i in range(n):
        cells = " ".join(
            "    ." if dist[i][j] == INF else f"{dist[i][j]:5.1f}"
            for j in range(n)
        )
        print(f"{i:>3} |{cells}")

    path = reconstruct_path(nxt, 0, 4)
    total = dist[0][4]
    print(f"\nShortest path 0 -> 4 : {' -> '.join(map(str, path))}  "
          f"(cost {total})")

    assert dist[0][3] == 4.0, "expected cheapest 0->3 cost of 4"
    assert reconstruct_path(nxt, 0, 3) == [0, 2, 1, 3]
    assert dist[4][0] == INF, "node 4 is a sink, cannot reach 0"
    print("All self-checks passed.")


if __name__ == "__main__":
    _demo()
