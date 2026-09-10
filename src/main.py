"""
main.py -- PART 3 end-to-end pipeline.

    load data
      -> train the GNN or load the latest checkpoint
      -> pick one test-split graph
      -> run the GNN to get per-edge attack propensities
      -> convert those into the plain structures the classical algorithms want
      -> run all five hand-written algorithms on that graph
      -> print a clear per-algorithm results summary
      -> dump step-by-step execution traces (PART 4 artifacts)

USAGE
-----
    python -m src.main                        # uses checkpoints/latest.pt
    python -m src.main --train-if-missing --epochs 20
    python -m src.main --graph 3 --threshold 0.6
    python -m src.main --graph data/_data_/graph_00hliAZI.pt --source 10 --target 200
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import List, Optional, Tuple

import torch

from .env_check import require_cuda
from .data_loader import (AttackGraphDataset, load_sample, AttackGraphSample)
from .integration import (GNNInference, build_algorithm_inputs,
                          largest_connected_component)
from .algorithms.warshall import transitive_closure, reachable_set
from .algorithms.floyd_warshall import floyd_warshall, reconstruct_path, INF
from .algorithms.dijkstra import dijkstra
from .algorithms.kruskal import kruskal
from .algorithms.prim import prim, adjacency_from_edges
from .explain import ExplainRun, save_explainability_artifacts

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "latest.pt")


# ========================================================================== #
# Helpers
# ========================================================================== #
def resolve_graph(dataset: AttackGraphDataset, spec: Optional[str],
                  test_indices: List[int]) -> str:
    """
    Turn ``--graph`` into a concrete ``.pt`` path.

    Accepts: None (-> first test-split graph), an integer (-> that position in
    the test split), a bare filename, or a full path.
    """
    if spec is None:
        return dataset.paths[test_indices[0]]
    if os.path.isfile(spec):
        return spec
    candidate = os.path.join(dataset.data_dir, spec)
    if os.path.isfile(candidate):
        return candidate
    try:
        pos = int(spec)
    except ValueError:
        raise SystemExit(f"--graph {spec!r}: not a file, not an index.")
    if not (0 <= pos < len(test_indices)):
        raise SystemExit(f"--graph index {pos} out of range "
                         f"(test split has {len(test_indices)} graphs).")
    return dataset.paths[test_indices[pos]]


def attack_path_endpoints(sample: AttackGraphSample) -> Tuple[int, int]:
    """
    Derive a (source, target) pair from the labelled attack path itself.

    The attack path is a chain in ``y_matrix``: its start has out-edges but no
    in-edge, its end has an in-edge but no out-edge. Falls back to the highest
    out-/in-degree node if the chain is not clean.
    """
    y = sample.y_matrix
    out_deg = y.sum(dim=1)
    in_deg = y.sum(dim=0)
    starts = [(out_deg[i] > 0) and (in_deg[i] == 0)
              for i in range(sample.num_nodes)]
    ends = [(in_deg[i] > 0) and (out_deg[i] == 0)
            for i in range(sample.num_nodes)]
    source = next((i for i, s in enumerate(starts) if s),
                  int(out_deg.argmax()))
    target = next((i for i, e in enumerate(ends) if e), int(in_deg.argmax()))
    return source, target


def summarise_finite(matrix: List[List[float]]) -> dict:
    """min / mean / max / count of the finite entries of a distance matrix."""
    vals = [c for row in matrix for c in row if c != INF]
    if not vals:
        return {"finite": 0}
    return {
        "finite": len(vals),
        "min": min(vals),
        "mean": sum(vals) / len(vals),
        "max": max(vals),
    }


# ========================================================================== #
# Pipeline
# ========================================================================== #
def run_pipeline(args: argparse.Namespace) -> None:
    device = require_cuda(verbose=True)

    # ---- checkpoint: load or (optionally) train --------------------- #
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        if args.train_if_missing:
            print(f"\n[main] no checkpoint at {ckpt_path} -- training "
                  f"{args.epochs} epochs first ...\n")
            from . import train as train_module
            train_module.main([
                "--data-dir", args.data_dir,
                "--epochs", str(args.epochs),
                "--checkpoint-dir", os.path.dirname(ckpt_path) or "checkpoints",
            ])
        else:
            raise SystemExit(
                f"[main] no checkpoint at {ckpt_path}. Run "
                f"`python -m src.train` or pass --train-if-missing."
            )

    # ---- data + deterministic split (same seed/scheme as train.py) --- #
    dataset = AttackGraphDataset(args.data_dir, as_pyg=False, limit=args.limit)
    n = len(dataset)
    n_train = int(round(n * 0.8))
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    test_indices = perm[n_train:]
    print(f"\ndataset {n} graphs -> train {n_train} | test {len(test_indices)}")

    graph_path = resolve_graph(dataset, args.graph, test_indices)
    sample = load_sample(graph_path)
    gid = os.path.basename(graph_path)
    print(f"selected test graph: {gid}  (V={sample.num_nodes}, "
          f"{sample.meta['num_edges']} edges, "
          f"{sample.meta['num_attack_edges']} on the labelled attack path)")

    # ---- GNN inference --------------------------------------------- #
    infer = GNNInference.from_checkpoint(ckpt_path, device)
    print(f"model: {infer.meta}")
    edge_prop = infer.predict(sample)
    print(f"predicted propensity: min={edge_prop.min():.3f} "
          f"mean={edge_prop.mean():.3f} max={edge_prop.max():.3f}")

    inp = build_algorithm_inputs(sample, edge_prop, threshold=args.threshold)

    source, target = args.source, args.target
    if source is None or target is None:
        s, t = attack_path_endpoints(sample)
        source = s if source is None else source
        target = t if target is None else target
    print(f"Dijkstra / path queries: source = {source}, target = {target}")

    run = ExplainRun.new(
        gid, checkpoint=ckpt_path, threshold=args.threshold,
        source=source, target=target,
        predicted_propensity={"min": float(edge_prop.min()),
                              "mean": float(edge_prop.mean()),
                              "max": float(edge_prop.max())},
    )

    print("\n" + "#" * 78)
    print("# CLASSICAL ALGORITHMS ON GNN-WEIGHTED GRAPH")
    print("#" * 78)

    # ===== 1. Warshall -- transitive closure / reachability ========== #
    t0 = time.time()
    reach = copy.deepcopy(inp.reachability)
    w_trace: list = []
    # V=361 here: record per-k deltas only, not a 361x361 grid x362 (keeps the
    # JSON small; the deltas + initial edges still fully replay the animation).
    transitive_closure(reach, trace=w_trace, snapshot_matrices=False)
    reachable_from_source = reachable_set(reach, source)
    can_reach_target = target in set(reachable_from_source)
    total_reach_pairs = sum(c for row in reach for c in row)
    dt_w = time.time() - t0
    print(f"\n[1] WARSHALL  (transitive closure, threshold "
          f"propensity >= {args.threshold})   [{dt_w:.1f}s]")
    print(f"    edges kept above threshold : "
          f"{sum(c for row in inp.reachability for c in row)}")
    print(f"    reachable ordered pairs    : {total_reach_pairs}")
    print(f"    |reachable(source={source})|   : {len(reachable_from_source)}")
    print(f"    source can reach target?   : {can_reach_target}")
    run.add("warshall", w_trace, result={
        "threshold": args.threshold,
        "reachable_pairs_total": total_reach_pairs,
        "reachable_from_source_count": len(reachable_from_source),
        "reachable_from_source_sample": reachable_from_source[:25],
        "source_can_reach_target": can_reach_target,
    })

    # ===== 2. Floyd-Warshall -- all-pairs shortest paths ============= #
    t0 = time.time()
    fw_trace: list = []
    dist, nxt = floyd_warshall(inp.weight_matrix, trace=fw_trace,
                               snapshot_matrices=False)
    fw_path = reconstruct_path(nxt, source, target)
    fw_cost = dist[source][target]
    dt_fw = time.time() - t0
    stats = summarise_finite(dist)
    print(f"\n[2] FLOYD-WARSHALL  (all-pairs shortest paths, "
          f"cost = 1 - propensity)   [{dt_fw:.1f}s]")
    print(f"    finite dist entries : {stats['finite']}")
    if stats["finite"]:
        print(f"    finite cost min/mean/max : {stats['min']:.4f} / "
              f"{stats['mean']:.4f} / {stats['max']:.4f}")
    print(f"    dist[{source}][{target}] : "
          f"{'unreachable' if fw_cost == INF else f'{fw_cost:.4f}'}")
    if fw_path:
        print(f"    path ({len(fw_path)} nodes): "
              f"{' -> '.join(map(str, fw_path[:20]))}"
              f"{' ...' if len(fw_path) > 20 else ''}")
    run.add("floyd_warshall", fw_trace, result={
        "finite_summary": stats,
        "source_target_cost": None if fw_cost == INF else fw_cost,
        "source_target_path": fw_path,
    })

    # ===== 3. Dijkstra -- single-source shortest path =============== #
    t0 = time.time()
    d_trace: list = []
    d_res = dijkstra(inp.adjacency, source, target=target, trace=d_trace)
    d_path = d_res.path_to(target)
    d_cost = d_res.dist.get(target, INF)
    dt_d = time.time() - t0
    print(f"\n[3] DIJKSTRA  (source {source} -> target {target}, "
          f"manual min-heap)   [{dt_d:.2f}s]")
    print(f"    settled vertices : {len(d_res.order_settled)}")
    print(f"    cost : {'unreachable' if d_cost == INF else f'{d_cost:.4f}'}")
    if d_path:
        print(f"    path ({len(d_path)} nodes): "
              f"{' -> '.join(map(str, d_path[:20]))}"
              f"{' ...' if len(d_path) > 20 else ''}")
    if d_cost != INF and fw_cost != INF:
        agree = abs(d_cost - fw_cost) < 1e-6
        print(f"    == Floyd-Warshall cost? {agree} "
              f"(FW {fw_cost:.4f} vs Dijkstra {d_cost:.4f})")
    run.add("dijkstra", d_trace, result={
        "settled_count": len(d_res.order_settled),
        "cost": None if d_cost == INF else d_cost,
        "path": d_path,
        "matches_floyd_warshall": (d_cost != INF and fw_cost != INF
                                   and abs(d_cost - fw_cost) < 1e-6),
    })

    # ===== 4 & 5. Kruskal + Prim -- MST on the largest component ==== #
    members, local_edges, relabel = largest_connected_component(
        inp.num_nodes, inp.undirected_edges)
    k_local = len(members)
    print(f"\n[4/5] MST  (undirected, weight = 1 - max propensity per pair)")
    print(f"    largest connected component: {k_local} nodes, "
          f"{len(local_edges)} edges")

    t0 = time.time()
    k_trace: list = []
    k_res = kruskal(k_local, local_edges, trace=k_trace)
    dt_k = time.time() - t0

    t0 = time.time()
    p_trace: list = []
    p_adj = adjacency_from_edges(k_local, local_edges)
    p_res = prim(k_local, p_adj, start=0, trace=p_trace)
    dt_p = time.time() - t0

    print(f"\n[4] KRUSKAL  (Union-Find, sort edges)   [{dt_k:.2f}s]")
    print(f"    MST edges : {len(k_res.mst_edges)}   "
          f"total weight : {k_res.total_weight:.6f}   "
          f"components : {k_res.num_components}")
    print(f"\n[5] PRIM  (manual min-heap, grow from node 0)   [{dt_p:.2f}s]")
    print(f"    MST edges : {len(p_res.mst_edges)}   "
          f"total weight : {p_res.total_weight:.6f}   "
          f"components : {p_res.num_components}")

    mst_agree = abs(k_res.total_weight - p_res.total_weight) < 1e-6
    print(f"\n    Kruskal total == Prim total ? {mst_agree}  "
          f"(|diff| = {abs(k_res.total_weight - p_res.total_weight):.2e})")
    if not mst_agree:
        print("    NOTE: a mismatch usually means the component was not fully "
              "connected for Prim's start node.")

    run.add("kruskal", k_trace, result={
        "component_nodes": k_local,
        "mst_edge_count": len(k_res.mst_edges),
        "total_weight": k_res.total_weight,
        "mst_edges_sample": [[u, v, round(w, 6)]
                             for u, v, w in k_res.mst_edges[:25]],
    })
    run.add("prim", p_trace, result={
        "component_nodes": k_local,
        "mst_edge_count": len(p_res.mst_edges),
        "total_weight": p_res.total_weight,
        "agrees_with_kruskal": mst_agree,
    })

    # ---- PART 4 artifacts ---------------------------------------------- #
    written = save_explainability_artifacts(run, out_dir=args.out_dir)
    print("\n" + "#" * 78)
    print("# EXPLAINABILITY ARTIFACTS (PART 4)")
    print("#" * 78)
    for name, path in written.items():
        print(f"    {name:20s} {path}  ({os.path.getsize(path):,} bytes)")

    print("\ndone.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-dir", default=os.path.join("data", "_data_"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--graph", default=None,
                   help="test-split index, a filename, or a path "
                        "(default: first test graph)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="propensity cut-off for Warshall's reachability")
    p.add_argument("--source", type=int, default=None)
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--train-if-missing", action="store_true")
    p.add_argument("--epochs", type=int, default=20,
                   help="epochs to train when --train-if-missing fires")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=os.path.join("logs", "explain"))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    run_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()
