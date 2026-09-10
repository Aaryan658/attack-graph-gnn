"""
dijkstra.py -- Dijkstra's single-source shortest-path algorithm on a weighted
directed graph with non-negative edge costs, using a binary-heap priority
queue, returning both the distances and the reconstructed path.

PROBLEM
-------
Given a graph as an adjacency list ``adj`` (``adj[u]`` is a list of
``(v, weight)`` pairs) and a ``source`` vertex, find the minimum-cost path from
``source`` to every other vertex -- and in particular the full vertex sequence
of the cheapest path from ``source`` to a specified ``target``.

In this project ``source`` is a "compromised" host and ``target`` is the asset
the attacker wants. Edge cost is ``1 - propensity`` from the GNN, so the
cheapest path is the chain of edges the model is most confident an attacker
would traverse.

ALGORITHM (Dijkstra, 1959)
--------------------------
Grow a set of vertices whose shortest distance from ``source`` is final
("settled"). Repeatedly:

    1. pop the unsettled vertex ``u`` with the smallest tentative distance
       (this is what the min-priority-queue gives us in O(log n));
    2. settle ``u`` -- its distance is now known to be optimal, because every
       edge cost is non-negative so no not-yet-seen route could be cheaper;
    3. "relax" each outgoing edge (u, v, w): if ``dist[u] + w < dist[v]`` we
       found a better route to v, so update ``dist[v]`` and record
       ``prev[v] = u`` for path reconstruction, and push v with its new key.

We use *lazy deletion*: instead of decreasing a key inside the heap (which
Python's ``heapq`` does not support) we push a fresh ``(distance, vertex)``
entry and ignore any popped entry whose distance is stale.

COMPLEXITY
----------
With a binary heap and lazy deletion:
    Time  : O((V + E) log V)      -- each edge causes at most one push.
    Space : O(V + E)              -- adjacency list, dist/prev maps, heap.

Correctness depends on **non-negative edge weights**; the constructor helper
:func:`adjacency_from_propensity` guarantees that, and we assert it on entry.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Tuple

INF = math.inf
Adjacency = Dict[int, List[Tuple[int, float]]]


class DijkstraResult:
    """Bundle of everything one Dijkstra run produces, for readable call sites."""

    def __init__(
        self,
        source: int,
        dist: Dict[int, float],
        prev: Dict[int, Optional[int]],
        order_settled: List[int],
    ) -> None:
        self.source = source
        self.dist = dist                    # vertex -> best known distance
        self.prev = prev                    # vertex -> predecessor on best path
        self.order_settled = order_settled  # vertices in the order they settled

    def path_to(self, target: int) -> List[int]:
        """
        Walk ``prev`` pointers backward from ``target`` to ``source`` and return
        the path as a forward list of vertices. Empty list if unreachable.
        """
        if self.dist.get(target, INF) == INF:
            return []
        path = []
        node: Optional[int] = target
        while node is not None:
            path.append(node)
            if node == self.source:
                break
            node = self.prev[node]
        path.reverse()
        return path


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
def dijkstra(
    adj: Adjacency,
    source: int,
    target: Optional[int] = None,
    trace: Optional[list] = None,
) -> DijkstraResult:
    """
    Run Dijkstra's algorithm from ``source``.

    Parameters
    ----------
    adj
        Adjacency list: ``adj[u]`` -> iterable of ``(v, weight)``. Vertices with
        no outgoing edges may be omitted as keys. Weights must be >= 0.
    source
        Start vertex.
    target
        Optional goal vertex. When given, the search stops as soon as
        ``target`` is settled (its distance is then final).
    trace
        Optional list; when given, one event dict is appended per heap pop,
        recording the settled vertex, its distance, the current frontier
        (unsettled vertices with a finite tentative distance), and every edge
        relaxation performed from it.

    Returns
    -------
    DijkstraResult
    """
    # Collect the full vertex set: keys plus every endpoint that appears in a
    # neighbour list (so isolated targets still get a dist entry of INF).
    vertices = set(adj.keys())
    for u, neighbours in adj.items():
        for v, w in neighbours:
            vertices.add(v)
            if w < 0:
                raise ValueError(
                    f"Dijkstra requires non-negative weights; edge "
                    f"({u} -> {v}) has weight {w}."
                )
    vertices.add(source)

    dist: Dict[int, float] = {v: INF for v in vertices}
    prev: Dict[int, Optional[int]] = {v: None for v in vertices}
    settled: Dict[int, bool] = {v: False for v in vertices}
    order_settled: List[int] = []

    dist[source] = 0.0

    # The priority queue holds (tentative_distance, vertex). Python's heapq is a
    # min-heap keyed on the tuple's first element -- exactly the "extract-min"
    # operation Dijkstra needs.
    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        d_u, u = heapq.heappop(heap)

        # Lazy deletion: this entry is stale if we already settled u, or if a
        # better distance was recorded after this entry was pushed.
        if settled[u] or d_u > dist[u]:
            continue

        settled[u] = True
        order_settled.append(u)

        relaxations: List[dict] = []
        for v, w in adj.get(u, ()):  # tolerate missing keys for sink vertices
            if settled[v]:
                continue
            candidate = d_u + w
            if candidate < dist[v]:
                dist[v] = candidate
                prev[v] = u
                heapq.heappush(heap, (candidate, v))
                relaxations.append({
                    "edge": [u, v],
                    "weight": round(w, 6),
                    "new_dist": round(candidate, 6),
                })

        if trace is not None:
            frontier = sorted(
                (vtx for vtx in vertices
                 if not settled[vtx] and dist[vtx] < INF),
                key=lambda vtx: (dist[vtx], vtx),
            )
            trace.append({
                "pop_order": len(order_settled),
                "settled_vertex": u,
                "settled_distance": round(d_u, 6),
                "frontier": [[vtx, round(dist[vtx], 6)] for vtx in frontier],
                "relaxations": relaxations,
            })

        # Early exit: once target is settled its distance cannot improve.
        if target is not None and u == target:
            break

    return DijkstraResult(source, dist, prev, order_settled)


# --------------------------------------------------------------------------- #
# GNN output -> adjacency list
# --------------------------------------------------------------------------- #
def adjacency_from_propensity(
    propensity: List[List[float]],
    edges: Iterable[Tuple[int, int]],
    epsilon: float = 1e-9,
) -> Adjacency:
    """
    Build a non-negative-weighted adjacency list from GNN propensities.

    Only the pairs in ``edges`` (the graph's real directed edges) become arcs;
    the weight of arc (u, v) is ``(1 - propensity[u][v]) + epsilon`` which lies
    in ``(0, 1]`` and is therefore always >= 0, satisfying Dijkstra's
    precondition.
    """
    adj: Adjacency = {}
    for u, v in edges:
        w = (1.0 - float(propensity[u][v])) + epsilon
        adj.setdefault(u, []).append((v, w))
    return adj


# --------------------------------------------------------------------------- #
# Stand-alone demonstration
# --------------------------------------------------------------------------- #
def _demo() -> None:
    # Toy weighted directed graph, 6 nodes (0..5).
    #   0->1 (7)   0->2 (9)   0->5 (14)
    #   1->2 (10)  1->3 (15)
    #   2->3 (11)  2->5 (2)
    #   3->4 (6)   5->4 (9)
    # Classic example: cheapest 0 -> 4 is 0->2->5->4 = 9 + 2 + 9 = 20.
    adj: Adjacency = {
        0: [(1, 7.0), (2, 9.0), (5, 14.0)],
        1: [(2, 10.0), (3, 15.0)],
        2: [(3, 11.0), (5, 2.0)],
        3: [(4, 6.0)],
        5: [(4, 9.0)],
    }

    print("=" * 60)
    print("Dijkstra's algorithm -- single-source shortest path demo")
    print("=" * 60)
    print("Graph (u -> v : cost):")
    for u in sorted(adj):
        for v, w in adj[u]:
            print(f"    {u} -> {v} : {w}")

    source, target = 0, 4
    trace: list = []
    result = dijkstra(adj, source, target=target, trace=trace)

    print(f"\nFrontier expansion order (source = {source}):")
    for event in trace:
        relax = ", ".join(
            f"{r['edge'][0]}->{r['edge'][1]} (dist now {r['new_dist']})"
            for r in event["relaxations"]
        )
        print(f"  #{event['pop_order']:>2}  settle vertex {event['settled_vertex']} "
              f"at distance {event['settled_distance']}")
        print(f"        frontier now: {event['frontier']}")
        print(f"        relaxed: {relax if relax else '(none)'}")

    path = result.path_to(target)
    print(f"\nShortest path {source} -> {target}: "
          f"{' -> '.join(map(str, path))}")
    print(f"Total cost: {result.dist[target]}")

    assert result.dist[target] == 20.0, "expected cheapest 0->4 cost of 20"
    assert path == [0, 2, 5, 4], f"unexpected path {path}"
    print("\nAll self-checks passed.")


if __name__ == "__main__":
    _demo()
