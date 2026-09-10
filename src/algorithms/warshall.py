"""
warshall.py -- Warshall's algorithm for the transitive closure of a directed
graph (boolean "can node i reach node j?" reachability).

PROBLEM
-------
Given a directed graph as a V x V boolean adjacency matrix ``reach`` where
``reach[i][j]`` is True iff there is a *direct* edge i -> j, compute the
*transitive closure*: the matrix ``R`` where ``R[i][j]`` is True iff there is a
path (of any length >= 1) from i to j.

In this project the direct edges come from the GNN: we threshold the predicted
per-edge attack propensities, keep the edges whose score is >= the threshold,
and ask "which machines can an attacker ultimately reach from a compromised
host, following only high-propensity edges?".

ALGORITHM (Warshall, 1962)
--------------------------
Consider intermediate vertices one at a time. After processing intermediate
vertex ``k``, ``reach[i][j]`` is True iff there is a path from i to j whose
intermediate vertices are all drawn from {0, 1, ..., k}. The recurrence is::

    reach[i][j] = reach[i][j]  OR  (reach[i][k] AND reach[k][j])

Running ``k`` from 0 to V-1 lifts the restriction entirely, leaving the full
transitive closure.

COMPLEXITY
----------
Time  : O(V^3)  -- three nested loops over the vertex set.
Space : O(V^2)  -- the reachability matrix is updated in place.

The update is safe to do in place: row ``k`` and column ``k`` are unchanged
during iteration ``k`` (``reach[k][k]`` only ever turns True, which does not
change the value of ``reach[i][k] AND reach[k][j]``).
"""

from __future__ import annotations

from typing import List, Optional

BoolMatrix = List[List[bool]]


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
def transitive_closure(
    reach: BoolMatrix,
    trace: Optional[list] = None,
    snapshot_matrices: bool = True,
) -> BoolMatrix:
    """
    Compute the transitive closure of ``reach`` in place and return it.

    Parameters
    ----------
    reach
        V x V boolean adjacency matrix. Modified in place; also returned for
        convenience.
    trace
        Optional list. When supplied, one snapshot dict is appended per
        intermediate-vertex iteration ``k`` (plus one for the initial state),
        recording which cells flipped False -> True. Used to build the
        step-by-step explainability artifact.
    snapshot_matrices
        When True (default, good for small toy graphs) each trace entry also
        carries a full 0/1 copy of the matrix at that step. Set False for large
        graphs: the per-step ``changed_cells`` deltas plus the ``init`` edge
        list still fully determine every intermediate matrix for an animation,
        without writing a V*V grid V times.

    Returns
    -------
    The same list object as ``reach``, now holding the transitive closure.
    """
    n = len(reach)

    # Defensive shape check -- a ragged matrix almost certainly means a bug
    # upstream, and would otherwise fail with a confusing IndexError deep in
    # the loop.
    for row_index, row in enumerate(reach):
        if len(row) != n:
            raise ValueError(
                f"transitive_closure expects a square matrix; row {row_index} "
                f"has length {len(row)}, expected {n}."
            )

    if trace is not None:
        init_entry = {
            "step": "init",
            "k": None,
            "changed_cells": [],
            # Always record the starting edges as a compact list so an animation
            # has a seed even when full matrices are not snapshotted.
            "initial_edges": [[i, j] for i in range(n) for j in range(n)
                              if reach[i][j]],
        }
        if snapshot_matrices:
            init_entry["matrix"] = _copy_as_int(reach)
        trace.append(init_entry)

    # --- Warshall's triple loop ------------------------------------------- #
    # k = the vertex we now allow to appear as an intermediate hop.
    for k in range(n):
        changed_cells: List[List[int]] = []

        # i = path start vertex.
        for i in range(n):
            # Micro-optimisation that is also a readability win: if i cannot
            # reach k, then no j gets a new path *through* k from i, so the
            # whole inner loop can be skipped.
            if not reach[i][k]:
                continue

            row_i = reach[i]
            row_k = reach[k]

            # j = path end vertex.
            for j in range(n):
                # New path i -> ... -> k -> ... -> j discovered?
                if not row_i[j] and row_k[j]:
                    row_i[j] = True
                    changed_cells.append([i, j])

        if trace is not None:
            entry = {
                "step": f"k={k}",
                "k": k,
                "changed_cells": changed_cells,
            }
            if snapshot_matrices:
                entry["matrix"] = _copy_as_int(reach)
            trace.append(entry)

    return reach


# --------------------------------------------------------------------------- #
# Helpers to go from GNN output -> boolean reachability matrix
# --------------------------------------------------------------------------- #
def reachability_from_propensity(
    propensity: List[List[float]],
    threshold: float,
    keep_self_loops: bool = False,
) -> BoolMatrix:
    """
    Turn a V x V matrix of predicted edge propensities into the boolean
    adjacency matrix Warshall's algorithm consumes.

    ``reach[i][j]`` becomes True iff ``propensity[i][j] >= threshold``. The
    diagonal is forced False unless ``keep_self_loops`` is set, because "node i
    reaches itself in zero hops" is trivially true and only clutters the
    reachable set.
    """
    n = len(propensity)
    reach: BoolMatrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j and not keep_self_loops:
                continue
            reach[i][j] = propensity[i][j] >= threshold
    return reach


def reachable_set(closure: BoolMatrix, source: int) -> List[int]:
    """Return the sorted list of vertices reachable from ``source``."""
    return [j for j, can_reach in enumerate(closure[source]) if can_reach]


def _copy_as_int(matrix: BoolMatrix) -> List[List[int]]:
    """Snapshot a boolean matrix as 0/1 ints (JSON-friendly, compact to read)."""
    return [[1 if cell else 0 for cell in row] for row in matrix]


# --------------------------------------------------------------------------- #
# Stand-alone demonstration
# --------------------------------------------------------------------------- #
def _print_matrix(matrix: BoolMatrix, labels: List[str], title: str) -> None:
    print(f"\n{title}")
    header = "     " + " ".join(f"{lab:>3}" for lab in labels)
    print(header)
    for lab, row in zip(labels, matrix):
        cells = " ".join("  1" if c else "  ." for c in row)
        print(f"{lab:>3} |{cells}")


def _demo() -> None:
    # Toy directed graph, 6 nodes (A..F).
    #   A -> B      B -> C      C -> A        (a 3-cycle A,B,C)
    #   C -> D      D -> E                    (a tail D,E hanging off C)
    #   F -> D                                (F feeds into the tail, nothing
    #                                          points back at F)
    labels = ["A", "B", "C", "D", "E", "F"]
    index = {name: i for i, name in enumerate(labels)}
    edges = [("A", "B"), ("B", "C"), ("C", "A"),
             ("C", "D"), ("D", "E"), ("F", "D")]

    n = len(labels)
    reach: BoolMatrix = [[False] * n for _ in range(n)]
    for u, v in edges:
        reach[index[u]][index[v]] = True

    print("=" * 60)
    print("Warshall's algorithm -- transitive closure demo")
    print("=" * 60)
    print(f"Vertices : {labels}")
    print(f"Edges    : {edges}")
    _print_matrix(reach, labels, "Direct adjacency (before):")

    trace: list = []
    transitive_closure(reach, trace=trace)

    # Show how the matrix grows as each intermediate vertex is admitted.
    for snapshot in trace:
        if snapshot["step"] == "init":
            continue
        flipped = snapshot["changed_cells"]
        pretty = [f"{labels[i]}->{labels[j]}" for i, j in flipped]
        print(f"\nAfter admitting intermediate vertex "
              f"{labels[snapshot['k']]}: new reachable pairs = "
              f"{pretty if pretty else 'none'}")

    _print_matrix(reach, labels, "Transitive closure (after):")

    for name in labels:
        rs = reachable_set(reach, index[name])
        print(f"  {name} can reach: {[labels[j] for j in rs]}")

    # Sanity checks a reader can verify by eye from the edge list.
    assert reachable_set(reach, index["A"]) == sorted(
        index[x] for x in ["A", "B", "C", "D", "E"]
    ), "A should reach the whole cycle plus the tail"
    assert reachable_set(reach, index["E"]) == [], "E is a sink"
    assert reachable_set(reach, index["F"]) == sorted(
        index[x] for x in ["D", "E"]
    ), "F only reaches the tail"
    print("\nAll self-checks passed.")


if __name__ == "__main__":
    _demo()
