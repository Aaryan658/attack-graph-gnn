"""
Classical graph algorithms, implemented from scratch (textbook style).

Every algorithm in this package is written to be *read*, not just run:
    * module docstring states the problem and the time/space complexity,
    * variable names spell out what they hold,
    * inline comments explain each step,
    * each module has a ``__main__`` block that runs the algorithm on a small
      hard-coded toy graph and prints the intermediate state, so it can be
      demonstrated in isolation without the GNN pipeline.

None of these use networkx's built-in algorithms. ``heapq`` is used for the
binary-heap priority queue in Dijkstra's and Prim's -- that is a standard data
structure, not a shortcut around the algorithm logic.

Modules
-------
warshall         -- transitive closure (boolean reachability), O(V^3)
floyd_warshall   -- all-pairs shortest paths, O(V^3)
dijkstra         -- single-source shortest path with a min-heap, O(E log V)
kruskal          -- minimum spanning tree via Union-Find, O(E log E)
prim             -- minimum spanning tree via a min-heap, O(E log V)
"""
