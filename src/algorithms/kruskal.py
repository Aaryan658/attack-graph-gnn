"""
kruskal.py -- Kruskal's algorithm for a Minimum Spanning Tree / Forest, with a
from-scratch Union-Find (disjoint-set) structure.

PROBLEM
-------
Given a connected, undirected, weighted graph, find a spanning tree (a
cycle-free subset of edges that touches every vertex) whose total edge weight
is as small as possible. If the graph is not connected, the same procedure
yields a minimum spanning *forest* -- one MST per connected component.

In this project we first make the GNN-weighted attack graph undirected (keep,
for each unordered pair, the cheaper of the two directed costs). The MST is
then the cheapest sub-network of edges that still connects every host the
model considers reachable -- a compact "backbone" view of the attack surface.

ALGORITHM (Kruskal, 1956)
-------------------------
    1. Sort all edges by non-decreasing weight.
    2. Walk the sorted edges. For edge (u, v): if u and v are currently in
       different components, adding (u, v) cannot create a cycle, so take it
       and merge the two components. Otherwise u and v are already connected,
       so (u, v) would close a cycle -- reject it.
    3. Stop once the tree has V - 1 edges (a spanning tree of a V-vertex
       connected graph), or once the edges run out (forest case).

The "same component?" test and the merge are done with **Union-Find**:
    * ``find(x)``  -- return the representative ("root") of x's set, compressing
      the path so future queries are near-O(1);
    * ``union(a, b)`` -- attach the shorter tree under the taller one
      ("union by rank") to keep trees shallow.
With both optimisations, m operations cost O(m * alpha(n)), where alpha is the
inverse Ackermann function (<= 4 for any n that fits in the universe).

COMPLEXITY
----------
Time  : O(E log E)  -- dominated by the initial sort; the union-find sweep is
                       effectively O(E).
Space : O(V + E)    -- parent/rank arrays plus the edge list.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Edge = Tuple[int, int, float]  # (u, v, weight)


# --------------------------------------------------------------------------- #
# Disjoint Set Union (Union-Find) -- built from scratch
# --------------------------------------------------------------------------- #
class DisjointSet:
    """
    Union-Find over the integer vertices ``0 .. n - 1`` with path compression
    and union by rank.

    ``parent[x]``  points one step toward the root of x's tree; a root points
                   at itself.
    ``rank[x]``    is an upper bound on the height of the tree rooted at x; only
                   meaningful when x is a root.
    """

    def __init__(self, n: int) -> None:
        self.parent: List[int] = list(range(n))  # every vertex starts alone
        self.rank: List[int] = [0] * n
        self.num_sets: int = n

    def find(self, x: int) -> int:
        """Return the representative of x's set, compressing the path."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Second pass: point every node on the path straight at the root.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """
        Merge the sets containing ``a`` and ``b``.

        Returns True if a merge happened, False if they were already in the same
        set (which, in Kruskal's, means "this edge would form a cycle").
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False

        # Union by rank: hang the lower-rank root under the higher-rank root.
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        self.num_sets -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


# --------------------------------------------------------------------------- #
# Result bundle
# --------------------------------------------------------------------------- #
class MSTResult:
    """The output of an MST run, shared by :mod:`kruskal` and :mod:`prim`."""

    def __init__(
        self,
        mst_edges: List[Edge],
        total_weight: float,
        num_components: int,
    ) -> None:
        self.mst_edges = mst_edges           # edges kept, in the order chosen
        self.total_weight = total_weight     # sum of kept edge weights
        self.num_components = num_components  # 1 => the graph was connected

    def __repr__(self) -> str:  # helps when printed from main.py
        return (f"MSTResult(|E|={len(self.mst_edges)}, "
                f"total_weight={self.total_weight:.6f}, "
                f"components={self.num_components})")


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
def kruskal(
    num_nodes: int,
    edges: Sequence[Edge],
    trace: Optional[list] = None,
) -> MSTResult:
    """
    Compute an MST (or MSF) of an undirected weighted graph with Kruskal's
    algorithm.

    Parameters
    ----------
    num_nodes
        Number of vertices; they are assumed to be labelled ``0 .. num_nodes-1``.
    edges
        Iterable of ``(u, v, weight)`` undirected edges. Duplicates and both
        orientations are harmless (the second copy is simply rejected as a
        cycle).
    trace
        Optional list; when given, one decision dict is appended per edge
        examined, in sorted order, recording whether it was accepted or
        rejected and the running total.

    Returns
    -------
    MSTResult
    """
    # Step 1: sort a *copy* of the edges by weight (ascending). Ties broken by
    # endpoints just to make the output deterministic / reproducible.
    sorted_edges = sorted(edges, key=lambda e: (e[2], e[0], e[1]))

    dsu = DisjointSet(num_nodes)
    mst_edges: List[Edge] = []
    total_weight = 0.0

    # Step 2: greedily take the lightest edge that does not close a cycle.
    for rank_index, (u, v, w) in enumerate(sorted_edges):
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(
                f"Edge ({u}, {v}, {w}) references a vertex outside "
                f"0..{num_nodes - 1}."
            )

        accepted = dsu.union(u, v)  # False => u,v already connected => cycle
        if accepted:
            mst_edges.append((u, v, w))
            total_weight += w

        if trace is not None:
            trace.append({
                "examined_rank": rank_index,      # position in the sorted list
                "edge": [u, v],
                "weight": round(w, 6),
                "decision": "accept" if accepted else "reject (cycle)",
                "edges_in_tree": len(mst_edges),
                "running_weight": round(total_weight, 6),
                "components_remaining": dsu.num_sets,
            })

        # Step 3: early stop once we have a spanning tree.
        if len(mst_edges) == num_nodes - 1:
            break

    return MSTResult(mst_edges, total_weight, dsu.num_sets)


# --------------------------------------------------------------------------- #
# Helper: directed weighted graph -> undirected edge list
# --------------------------------------------------------------------------- #
def undirected_edges_from_directed(
    directed_edges: Sequence[Edge],
) -> List[Edge]:
    """
    Collapse a directed weighted edge list into an undirected one, keeping the
    minimum weight seen for each unordered pair ``{u, v}``.

    MST algorithms are defined on undirected graphs; the attack graph is
    directed, so we need this bridge. Using the *min* of the two directions is
    the natural choice: it is the cheapest way to "connect" the pair.
    """
    best: dict = {}
    for u, v, w in directed_edges:
        if u == v:
            continue  # self-loops never belong in a tree
        key = (u, v) if u < v else (v, u)
        if key not in best or w < best[key]:
            best[key] = w
    return [(u, v, w) for (u, v), w in best.items()]


# --------------------------------------------------------------------------- #
# Stand-alone demonstration
# --------------------------------------------------------------------------- #
def _demo() -> None:
    # Toy undirected weighted graph, 7 nodes (0..6) -- the classic CLRS MST
    # example (relabelled a..g -> 0..6).
    #   (0,1,7) (0,3,5) (1,2,8) (1,3,9) (1,4,7) (2,4,5)
    #   (3,4,15) (3,5,6) (4,5,8) (4,6,9) (5,6,11)
    num_nodes = 7
    edges: List[Edge] = [
        (0, 1, 7.0), (0, 3, 5.0), (1, 2, 8.0), (1, 3, 9.0), (1, 4, 7.0),
        (2, 4, 5.0), (3, 4, 15.0), (3, 5, 6.0), (4, 5, 8.0), (4, 6, 9.0),
        (5, 6, 11.0),
    ]

    print("=" * 60)
    print("Kruskal's algorithm -- minimum spanning tree demo")
    print("=" * 60)
    print(f"{num_nodes} vertices, {len(edges)} edges")

    trace: list = []
    result = kruskal(num_nodes, edges, trace=trace)

    print("\nEdges considered in sorted order (accept / reject):")
    print("  rank  edge     weight  decision            tree|comp  running w")
    for d in trace:
        print(f"  {d['examined_rank']:>4}  "
              f"{d['edge'][0]}--{d['edge'][1]:<3} "
              f"{d['weight']:>6}  {d['decision']:<18} "
              f"{d['edges_in_tree']:>4}|{d['components_remaining']:<4} "
              f"{d['running_weight']}")

    print("\nMST edges (in the order Kruskal accepted them):")
    for u, v, w in result.mst_edges:
        print(f"    {u} -- {v}   weight {w}")
    print(f"\nTotal MST weight : {result.total_weight}")
    print(f"Components       : {result.num_components} "
          f"({'connected' if result.num_components == 1 else 'forest'})")

    # Known answer for this classic instance.
    assert result.total_weight == 39.0, "CLRS MST total is 39"
    assert len(result.mst_edges) == num_nodes - 1
    print("\nAll self-checks passed.")


if __name__ == "__main__":
    _demo()
