"""
prim.py -- Prim's algorithm for a Minimum Spanning Tree, grown from a start
vertex using a binary-heap priority queue.

PROBLEM
-------
Same goal as Kruskal's (see :mod:`kruskal`): the cheapest cycle-free set of
edges that connects every vertex of a connected, undirected, weighted graph.
This module exists so the project can run *two* independent correct MST
algorithms on the same graph and confirm they agree on the total weight -- a
cross-check that the graph construction and both implementations are right.

ALGORITHM (Jarnik 1930; Prim 1957; Dijkstra 1959)
-------------------------------------------------
Keep a growing tree ``T`` that starts as the single vertex ``start``.
Repeatedly add the **cheapest edge that has exactly one endpoint in ``T``**
(a "crossing edge"), which pulls one new vertex into ``T``. After V - 1 such
additions every vertex is in ``T`` and the chosen edges form the MST.

A min-priority-queue makes "cheapest crossing edge" fast: whenever a vertex
``u`` joins the tree we push all of its incident edges ``(weight, u, v)``. To
pop the cheapest crossing edge we take the heap minimum and skip it if its
other endpoint ``v`` is already in the tree (lazy deletion -- same trick as in
:mod:`dijkstra`).

COMPLEXITY
----------
Time  : O(E log E) ~ O(E log V)   -- every edge is pushed and popped at most
                                     once; each heap op is O(log E).
Space : O(E)                      -- the heap can hold up to one entry per edge.

Requires a **connected** graph to produce a spanning *tree*. If the graph is
disconnected this routine spans only the component containing ``start`` and
reports the shortfall as extra components.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

# Reuse the shared Edge alias and MSTResult type from kruskal.py. Support both
# "python -m src.algorithms.prim" (package context -> relative import) and
# "python src/algorithms/prim.py" (script context -> flat import).
try:
    from .kruskal import Edge, MSTResult
except ImportError:  # run as a loose script, no parent package
    from kruskal import Edge, MSTResult

Adjacency = Dict[int, List[Tuple[int, float]]]  # u -> [(v, weight), ...]


# --------------------------------------------------------------------------- #
# Build an undirected adjacency list from an undirected edge list
# --------------------------------------------------------------------------- #
def adjacency_from_edges(num_nodes: int, edges: Sequence[Edge]) -> Adjacency:
    """
    Turn an undirected ``(u, v, w)`` edge list into a symmetric adjacency list
    (each edge stored under both endpoints), which is what Prim's inner loop
    walks.
    """
    adj: Adjacency = {i: [] for i in range(num_nodes)}
    for u, v, w in edges:
        if u == v:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
def prim(
    num_nodes: int,
    adj: Adjacency,
    start: int = 0,
    trace: Optional[list] = None,
) -> MSTResult:
    """
    Grow an MST from ``start`` with Prim's algorithm.

    Parameters
    ----------
    num_nodes
        Vertex count; vertices are ``0 .. num_nodes - 1``.
    adj
        Symmetric adjacency list (see :func:`adjacency_from_edges`).
    start
        Vertex the tree grows from. For a connected graph the choice does not
        affect the total weight, only the order edges are added.
    trace
        Optional list; when given, one dict is appended each time a vertex is
        pulled into the tree, recording the crossing edge used, its weight, the
        running total, and the current heap size (the "fringe").

    Returns
    -------
    MSTResult
    """
    if not (0 <= start < num_nodes):
        raise ValueError(f"start vertex {start} outside 0..{num_nodes - 1}")

    in_tree: List[bool] = [False] * num_nodes
    mst_edges: List[Edge] = []
    total_weight = 0.0

    # Bootstrap: put `start` in the tree and push its incident edges.
    in_tree[start] = True
    heap: List[Tuple[float, int, int]] = []
    for nbr, w in adj.get(start, ()):
        heapq.heappush(heap, (w, start, nbr))
    if trace is not None:
        trace.append({
            "added_vertex": start,
            "via_edge": None,          # the root, added for free
            "edge_weight": 0.0,
            "edges_in_tree": 0,
            "running_weight": 0.0,
            "fringe_size": len(heap),
            "candidates_pushed": len(heap),
        })

    # Grow until the tree spans every vertex (V - 1 edges) or we run out of
    # crossing edges (disconnected graph).
    while heap and len(mst_edges) < num_nodes - 1:
        weight, frm, to = heapq.heappop(heap)

        # Lazy deletion: this crossing edge is stale if `to` was pulled in by a
        # cheaper edge after this entry was pushed.
        if in_tree[to]:
            continue

        in_tree[to] = True
        mst_edges.append((frm, to, weight))
        total_weight += weight

        # Newly reachable crossing edges: every edge from `to` to a vertex not
        # yet in the tree.
        pushed = 0
        for nbr, w in adj.get(to, ()):
            if not in_tree[nbr]:
                heapq.heappush(heap, (w, to, nbr))
                pushed += 1

        if trace is not None:
            trace.append({
                "added_vertex": to,
                "via_edge": [frm, to],
                "edge_weight": round(weight, 6),
                "edges_in_tree": len(mst_edges),
                "running_weight": round(total_weight, 6),
                "fringe_size": len(heap),
                "candidates_pushed": pushed,
            })

    num_reached = sum(in_tree)
    # Disconnected graph: `num_reached < num_nodes`. Count each unreached vertex
    # as its own component so the number lines up with Kruskal's convention.
    num_components = 1 + (num_nodes - num_reached)
    return MSTResult(mst_edges, total_weight, num_components)


# --------------------------------------------------------------------------- #
# Stand-alone demonstration
# --------------------------------------------------------------------------- #
def _demo() -> None:
    # SAME toy graph as kruskal.py's demo, so the totals can be compared.
    num_nodes = 7
    edges: List[Edge] = [
        (0, 1, 7.0), (0, 3, 5.0), (1, 2, 8.0), (1, 3, 9.0), (1, 4, 7.0),
        (2, 4, 5.0), (3, 4, 15.0), (3, 5, 6.0), (4, 5, 8.0), (4, 6, 9.0),
        (5, 6, 11.0),
    ]
    adj = adjacency_from_edges(num_nodes, edges)

    print("=" * 60)
    print("Prim's algorithm -- minimum spanning tree demo")
    print("=" * 60)
    print(f"{num_nodes} vertices, {len(edges)} undirected edges, start = 0")

    trace: list = []
    result = prim(num_nodes, adj, start=0, trace=trace)

    print("\nTree growth (one row per vertex pulled in):")
    print("  vertex  via edge   w     tree  fringe  running w")
    for d in trace:
        via = "  (seed) " if d["via_edge"] is None \
              else f"{d['via_edge'][0]}--{d['via_edge'][1]:<3}"
        print(f"  {d['added_vertex']:>5}   {via:<9} {d['edge_weight']:>4}  "
              f"{d['edges_in_tree']:>4}  {d['fringe_size']:>5}   "
              f"{d['running_weight']}")

    print("\nMST edges (in the order Prim added them):")
    for u, v, w in result.mst_edges:
        print(f"    {u} -- {v}   weight {w}")
    print(f"\nTotal MST weight : {result.total_weight}")

    # Cross-check against Kruskal's on the identical graph.
    try:
        from .kruskal import kruskal
    except ImportError:
        from kruskal import kruskal
    kruskal_result = kruskal(num_nodes, edges)
    agree = kruskal_result.total_weight == result.total_weight
    print(f"Kruskal's total  : {kruskal_result.total_weight}   "
          f"--> {'AGREE' if agree else 'MISMATCH'}")

    assert result.total_weight == 39.0, "CLRS MST total is 39"
    assert result.total_weight == kruskal_result.total_weight
    assert len(result.mst_edges) == num_nodes - 1
    print("\nAll self-checks passed.")


if __name__ == "__main__":
    _demo()
