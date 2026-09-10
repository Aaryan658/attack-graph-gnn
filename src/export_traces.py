"""
export_traces.py -- emit a single JSON bundle that drives src/animation.html.

It runs each of the five classical algorithms twice:

  * on its small **toy graph** (the same textbook instance used in each module's
    ``__main__`` demo) with full per-step matrix snapshots -- this is the
    teaching animation;
  * on the **real 361-node attack graph** (one test-split sample, weighted by
    the trained GNN) for Dijkstra and Prim -- this is the "here it is at scale"
    animation.

Output: ``logs/animation_data.json``  (embedded verbatim into animation.html).

Run:  python -m src.export_traces
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import torch

from .algorithms.warshall import transitive_closure
from .algorithms.floyd_warshall import floyd_warshall, reconstruct_path, INF
from .algorithms.dijkstra import dijkstra, Adjacency
from .algorithms.kruskal import kruskal
from .algorithms.prim import prim, adjacency_from_edges

OUT = os.path.join("logs", "animation_data.json")
TEMPLATE = os.path.join("src", "animation.template.html")
BUILT = os.path.join("src", "animation.html")
REAL_GRAPH = os.path.join("data", "_data_", "graph_3DIqLl2D.pt")
CHECKPOINT = os.path.join("checkpoints", "latest.pt")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _ring(n: int, r: float = 1.0, cx: float = 0.0, cy: float = 0.0,
          rot: float = -math.pi / 2) -> List[List[float]]:
    """n points evenly on a circle -- tidy default layout for a toy graph."""
    return [[round(cx + r * math.cos(rot + 2 * math.pi * i / n), 4),
             round(cy + r * math.sin(rot + 2 * math.pi * i / n), 4)]
            for i in range(n)]


def _fmt(w: float) -> float:
    return round(float(w), 4)


# --------------------------------------------------------------------------- #
# TOY: Warshall
# --------------------------------------------------------------------------- #
def toy_warshall() -> dict:
    labels = ["A", "B", "C", "D", "E", "F"]
    idx = {c: i for i, c in enumerate(labels)}
    edges = [("A", "B"), ("B", "C"), ("C", "A"),
             ("C", "D"), ("D", "E"), ("F", "D")]
    n = len(labels)
    reach = [[False] * n for _ in range(n)]
    for u, v in edges:
        reach[idx[u]][idx[v]] = True

    trace: list = []
    transitive_closure(reach, trace=trace, snapshot_matrices=True)

    steps = []
    for snap in trace:
        if snap["step"] == "init":
            steps.append({
                "kind": "init", "k": None,
                "matrix": snap["matrix"],
                "changed": [],
                "desc": "Start from the direct-edge adjacency matrix. "
                        "reach[i][j] = 1 means a one-hop edge i -> j.",
            })
        else:
            k = snap["k"]
            pretty = ", ".join(f"{labels[i]}->{labels[j]}"
                               for i, j in snap["changed_cells"]) \
                or "nothing new"
            steps.append({
                "kind": "k", "k": k,
                "matrix": snap["matrix"],
                "changed": snap["changed_cells"],
                "desc": f"Allow {labels[k]} as an intermediate hop. "
                        f"reach[i][j] |= reach[i][{labels[k]}] & "
                        f"reach[{labels[k]}][j].  New: {pretty}.",
            })
    final = trace[-1]["matrix"]
    reach_from_a = [labels[j] for j in range(n) if final[0][j]]
    return {
        "graph": {"nodes": [{"id": i, "label": labels[i]} for i in range(n)],
                  "edges": [[idx[u], idx[v]] for u, v in edges],
                  "directed": True, "layout": _ring(n)},
        "steps": steps,
        "result": {"reachable_from_A": reach_from_a,
                   "note": "A reaches the whole cycle plus the tail; "
                           "E is a sink; F only reaches the tail."},
    }


# --------------------------------------------------------------------------- #
# TOY: Floyd-Warshall
# --------------------------------------------------------------------------- #
def toy_floyd() -> dict:
    n = 5
    edges = [(0, 1, 4.0), (0, 2, 1.0), (2, 1, 2.0),
             (1, 3, 1.0), (2, 3, 5.0), (3, 4, 3.0)]
    weight = [[INF] * n for _ in range(n)]
    for u, v, w in edges:
        weight[u][v] = w

    trace: list = []
    dist, nxt = floyd_warshall(weight, trace=trace, snapshot_matrices=True)

    steps = []
    for snap in trace:
        grid = [[None if c == "inf" else c for c in row]
                for row in snap["dist"]]
        if snap["step"] == "init":
            steps.append({
                "kind": "init", "k": None, "dist": grid, "improved": [],
                "desc": "Initialise: 0 on the diagonal, the edge cost where an "
                        "edge exists, infinity everywhere else.",
            })
        else:
            k = snap["k"]
            imp = ", ".join(f"{i}->{j}" for i, j in snap["improved_pairs"]) \
                or "no pair improved"
            steps.append({
                "kind": "k", "k": k, "dist": grid,
                "improved": snap["improved_pairs"],
                "desc": f"Route through vertex {k}: "
                        f"dist[i][j] = min(dist[i][j], dist[i][{k}] + "
                        f"dist[{k}][j]).  Improved: {imp}.",
            })
    path = reconstruct_path(nxt, 0, 4)
    return {
        "graph": {"nodes": [{"id": i, "label": str(i)} for i in range(n)],
                  "edges": [[u, v, _fmt(w)] for u, v, w in edges],
                  "directed": True, "layout": _ring(n)},
        "steps": steps,
        "result": {"path_0_to_4": path, "cost_0_to_4": dist[0][4]},
    }


# --------------------------------------------------------------------------- #
# TOY: Dijkstra
# --------------------------------------------------------------------------- #
def toy_dijkstra() -> dict:
    adj: Adjacency = {
        0: [(1, 7.0), (2, 9.0), (5, 14.0)],
        1: [(2, 10.0), (3, 15.0)],
        2: [(3, 11.0), (5, 2.0)],
        3: [(4, 6.0)],
        5: [(4, 9.0)],
    }
    n = 6
    edges = [(u, v, w) for u in adj for v, w in adj[u]]
    trace: list = []
    res = dijkstra(adj, 0, target=4, trace=trace)

    settled: list = []
    steps = [{
        "kind": "init", "vertex": 0, "dist": 0.0, "settled": [],
        "frontier": [[0, 0.0]], "relax": [],
        "desc": "Source = 0, distance 0. Every other vertex starts at infinity.",
    }]
    for ev in trace:
        settled.append(ev["settled_vertex"])
        relax = [[r["edge"][0], r["edge"][1], r["new_dist"]]
                 for r in ev["relaxations"]]
        rtxt = ", ".join(f"{a}->{b} = {d}" for a, b, d in relax) or "none"
        steps.append({
            "kind": "pop", "vertex": ev["settled_vertex"],
            "dist": ev["settled_distance"],
            "settled": list(settled),
            "frontier": [[v, d] for v, d in ev["frontier"]],
            "relax": relax,
            "desc": f"Pop the cheapest unsettled vertex: "
                    f"{ev['settled_vertex']} at distance "
                    f"{ev['settled_distance']}. Settle it, then relax its "
                    f"out-edges ({rtxt}).",
        })
    return {
        "graph": {"nodes": [{"id": i, "label": str(i)} for i in range(n)],
                  "edges": [[u, v, _fmt(w)] for u, v, w in edges],
                  "directed": True, "layout": _ring(n)},
        "source": 0, "target": 4,
        "steps": steps,
        "result": {"path_0_to_4": res.path_to(4), "cost": res.dist[4]},
    }


# --------------------------------------------------------------------------- #
# TOY: Kruskal + Prim (shared CLRS graph)
# --------------------------------------------------------------------------- #
_CLRS_N = 7
_CLRS_EDGES = [
    (0, 1, 7.0), (0, 3, 5.0), (1, 2, 8.0), (1, 3, 9.0), (1, 4, 7.0),
    (2, 4, 5.0), (3, 4, 15.0), (3, 5, 6.0), (4, 5, 8.0), (4, 6, 9.0),
    (5, 6, 11.0),
]
_CLRS_LAYOUT = [
    [-1.0, 0.9], [-0.1, 1.0], [0.9, 0.7], [-1.0, -0.2],
    [0.15, 0.0], [-0.6, -1.0], [0.9, -0.7],
]


def toy_kruskal() -> dict:
    trace: list = []
    res = kruskal(_CLRS_N, _CLRS_EDGES, trace=trace)
    parent = list(range(_CLRS_N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    steps = [{
        "kind": "init", "edge": None, "weight": None, "decision": None,
        "forest": list(range(_CLRS_N)), "running": 0.0, "mst": [],
        "desc": "Sort every edge by weight. Each vertex starts in its own "
                "component (disjoint-set).",
    }]
    mst: list = []
    for d in trace:
        u, v = d["edge"]
        if d["decision"].startswith("accept"):
            ru, rv = find(u), find(v)
            parent[ru] = rv
            mst.append([u, v, d["weight"]])
        forest = [find(i) for i in range(_CLRS_N)]
        steps.append({
            "kind": "consider", "edge": [u, v], "weight": d["weight"],
            "decision": d["decision"],
            "forest": forest, "running": d["running_weight"],
            "mst": [e[:] for e in mst],
            "desc": f"Edge {u}-{v} (w={d['weight']}): "
                    + ("endpoints are in different components -> ACCEPT, "
                       "merge them."
                       if d["decision"].startswith("accept")
                       else "endpoints already connected -> REJECT (would "
                            "make a cycle)."),
        })
    return {
        "graph": {"nodes": [{"id": i, "label": str(i)}
                            for i in range(_CLRS_N)],
                  "edges": [[u, v, _fmt(w)] for u, v, w in _CLRS_EDGES],
                  "directed": False, "layout": _CLRS_LAYOUT},
        "steps": steps,
        "result": {"mst_edges": res.mst_edges,
                   "total_weight": res.total_weight},
    }


def toy_prim() -> dict:
    adj = adjacency_from_edges(_CLRS_N, _CLRS_EDGES)
    trace: list = []
    res = prim(_CLRS_N, adj, start=0, trace=trace)

    in_tree: list = []
    mst: list = []
    steps = []
    for d in trace:
        in_tree.append(d["added_vertex"])
        via = d["via_edge"]
        if via is not None:
            mst.append([via[0], via[1], d["edge_weight"]])
        steps.append({
            "kind": "add", "vertex": d["added_vertex"], "via": via,
            "weight": d["edge_weight"],
            "in_tree": list(in_tree), "mst": [e[:] for e in mst],
            "fringe_size": d["fringe_size"], "running": d["running_weight"],
            "desc": ("Seed: put vertex 0 in the tree for free, push its "
                     "incident edges onto the heap."
                     if via is None else
                     f"Pop the cheapest crossing edge {via[0]}-{via[1]} "
                     f"(w={d['edge_weight']}); it pulls vertex "
                     f"{d['added_vertex']} into the tree."),
        })
    return {
        "graph": {"nodes": [{"id": i, "label": str(i)}
                            for i in range(_CLRS_N)],
                  "edges": [[u, v, _fmt(w)] for u, v, w in _CLRS_EDGES],
                  "directed": False, "layout": _CLRS_LAYOUT},
        "start": 0,
        "steps": steps,
        "result": {"mst_edges": res.mst_edges,
                   "total_weight": res.total_weight},
    }


# --------------------------------------------------------------------------- #
# REAL graph: Dijkstra + Prim on the GNN-weighted 361-node attack graph
# --------------------------------------------------------------------------- #
def _real_inputs():
    from .data_loader import load_sample
    from .integration import (GNNInference, build_algorithm_inputs,
                              largest_connected_component)
    from .main import attack_path_endpoints

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = load_sample(REAL_GRAPH)
    infer = GNNInference.from_checkpoint(CHECKPOINT, device)
    prop = infer.predict(sample)
    inp = build_algorithm_inputs(sample, prop, threshold=0.5)
    src, tgt = attack_path_endpoints(sample)
    return sample, inp, src, tgt, largest_connected_component


def _spring_layout(n: int, edges, seed: int = 42):
    """Positions for the big graph. networkx is fine for LAYOUT (not for the
    algorithms). Falls back to a deterministic circle if networkx is absent."""
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((u, v) for u, v, *_ in edges)
        pos = nx.spring_layout(g, seed=seed, k=1.6 / math.sqrt(max(n, 1)),
                               iterations=90)
        return [[round(float(pos[i][0]), 4), round(float(pos[i][1]), 4)]
                for i in range(n)]
    except Exception:
        return _ring(n, r=1.0)


def real_traces() -> dict:
    sample, inp, src, tgt, lcc = _real_inputs()
    V = inp.num_nodes
    layout = _spring_layout(V, inp.directed_edges)

    graph = {
        "nodes": [{"id": i, "label": str(i)} for i in range(V)],
        "edges": [[u, v, _fmt(w)] for u, v, w in inp.directed_edges],
        "directed": True, "layout": layout,
        "meta": {"graph_id": os.path.basename(REAL_GRAPH), "V": V,
                 "num_edges": len(inp.directed_edges),
                 "checkpoint": CHECKPOINT, "threshold": 0.5,
                 "source": src, "target": tgt},
    }

    # ----- Dijkstra (real) ----- #
    d_trace: list = []
    d_res = dijkstra(inp.adjacency, src, target=tgt, trace=d_trace)
    settled: list = []
    d_steps = [{"kind": "init", "vertex": src, "dist": 0.0, "settled": [],
                "frontier_size": 1, "frontier_head": [[src, 0.0]], "relax": [],
                "desc": f"Source = compromised host {src}. Target = {tgt}."}]
    for ev in d_trace:
        settled.append(ev["settled_vertex"])
        d_steps.append({
            "kind": "pop", "vertex": ev["settled_vertex"],
            "dist": ev["settled_distance"],
            "settled_new": ev["settled_vertex"],
            "frontier_size": len(ev["frontier"]),
            "frontier_head": [[v, d] for v, d in ev["frontier"][:8]],
            "relax": [[r["edge"][0], r["edge"][1], r["new_dist"]]
                      for r in ev["relaxations"]],
            "desc": f"Settle {ev['settled_vertex']} at cost "
                    f"{ev['settled_distance']}; {len(ev['relaxations'])} "
                    f"edge(s) relaxed.",
        })
    d_path = d_res.path_to(tgt)

    # ----- Prim (real, on the largest connected component) ----- #
    members, local_edges, relabel = lcc(V, inp.undirected_edges)
    inv = {v: k for k, v in relabel.items()}
    p_adj = adjacency_from_edges(len(members), local_edges)
    p_trace: list = []
    p_res = prim(len(members), p_adj, start=0, trace=p_trace)
    p_steps = []
    running_edges = 0
    for d in p_trace:
        via = d["via_edge"]
        p_steps.append({
            "kind": "add",
            "vertex": inv.get(d["added_vertex"], d["added_vertex"]),
            "via": ([inv[via[0]], inv[via[1]]] if via else None),
            "weight": d["edge_weight"],
            "edges_in_tree": d["edges_in_tree"],
            "running": d["running_weight"],
            "fringe_size": d["fringe_size"],
            "desc": ("Seed vertex." if via is None
                     else f"Add edge {inv[via[0]]}-{inv[via[1]]} "
                          f"(w={d['edge_weight']:.4f}); tree now "
                          f"{d['edges_in_tree']} edges."),
        })
        running_edges = d["edges_in_tree"]

    return {
        "graph": graph,
        "dijkstra": {"source": src, "target": tgt, "steps": d_steps,
                     "result": {"path": d_path, "cost": d_res.dist.get(tgt),
                                "settled": len(d_res.order_settled)}},
        "prim": {"component_nodes": len(members), "steps": p_steps,
                 "result": {"mst_edges": running_edges,
                            "total_weight": p_res.total_weight}},
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    bundle: Dict[str, dict] = {
        "toy": {
            "warshall": toy_warshall(),
            "floyd_warshall": toy_floyd(),
            "dijkstra": toy_dijkstra(),
            "kruskal": toy_kruskal(),
            "prim": toy_prim(),
        },
    }
    if os.path.isfile(REAL_GRAPH) and os.path.isfile(CHECKPOINT):
        print("building real-graph traces (this loads the GNN checkpoint) ...")
        bundle["real"] = real_traces()
    else:
        print(f"[skip real] missing {REAL_GRAPH} or {CHECKPOINT}; "
              f"toy traces only.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = json.dumps(bundle, separators=(",", ":"))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(f"wrote {OUT}  ({os.path.getsize(OUT):,} bytes)")

    # Build the self-contained animation page by inlining the JSON into the
    # template's __DATA__ marker.
    if os.path.isfile(TEMPLATE):
        html = open(TEMPLATE, encoding="utf-8").read()
        html = html.replace("__DATA__", payload)
        with open(BUILT, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"wrote {BUILT}  ({os.path.getsize(BUILT):,} bytes)")
    else:
        print(f"[skip build] {TEMPLATE} not found; "
              f"{OUT} written but animation.html not rebuilt.")


if __name__ == "__main__":
    main()
