"""
evaluate.py -- does the GNN actually help? Three linked evaluations, all on the
untouched 80/20 TEST split (same seed/scheme as train.py and main.py):

  (A) ABLATION -- pool predicted-edge scores over the whole test split for
      three weighting schemes and score them as rankings against the true
      attack-path label:
        * gnn      -- the trained GraphSAGE model's propensity
        * heuristic-- a fixed, non-learned "AD security analyst" severity per
                      edge type (DCSync/GenericAll/WriteDacl highest, GpLink/
                      Contains lowest), max over the types active on an edge.
                      This is domain knowledge, not a second model.
        * uniform  -- every edge scored 0.5 (the "no model at all" floor)
      Metrics: ROC-AUC, Average Precision, precision@10, precision@50.

  (B) PATH-QUALITY ABLATION -- for a sample of test graphs, run Dijkstra under
      each scheme from the true attack-path start to its end, and measure the
      Jaccard overlap between the predicted path's edges and the TRUE labelled
      attack-path edges. This is the concrete "does the GNN find the real
      attack path better than a fixed rule?" number.

  (C) MULTI-GRAPH ALGORITHM CROSS-CHECK -- extends the single-graph agreement
      check in main.py to a sample of test graphs: does Dijkstra's cost equal
      Floyd-Warshall's for the same pair, and does Kruskal's MST total equal
      Prim's, on every graph (not just one)?

  (D) THRESHOLD SWEEP -- Warshall's reachable-set size as the propensity
      threshold varies, averaged over a small sample of graphs.

USAGE
-----
    python -m src.evaluate                       # everything, default sample sizes
    python -m src.evaluate --path-sample 20 --crosscheck-sample 15 --quick

Outputs (all under logs/):
    evaluation_report.json     -- every number below, machine-readable
    ablation_metrics.csv       -- (A) per-scheme pooled metrics
    path_quality.csv           -- (B) per-graph Jaccard per scheme
    multigraph_crosscheck.csv  -- (C) per-graph agreement results
    threshold_sweep.csv        -- (D) per-threshold reachable-set sizes
    ablation_metrics.png       -- (A) bar chart
    threshold_sweep.png        -- (D) line chart
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Tuple

import torch

from .env_check import require_cuda
from .data_loader import AttackGraphDataset, load_sample, AttackGraphSample
from .integration import (GNNInference, build_algorithm_inputs,
                          largest_connected_component)
from .train import roc_auc
from .algorithms.warshall import transitive_closure, reachable_set
from .algorithms.floyd_warshall import floyd_warshall, INF
from .algorithms.dijkstra import dijkstra, adjacency_from_propensity
from .algorithms.kruskal import kruskal
from .algorithms.prim import prim, adjacency_from_edges
from .main import attack_path_endpoints

LOG_DIR = "logs"

# --------------------------------------------------------------------------- #
# Edge-type severity heuristic -- REAL channel order, recovered from the
# upstream preprocessing notebook (_graphPreprocess_.ipynb), not guessed:
#   ["AdminTo","AllowedToDelegate","CanRDP","Contains","DCSync","ExecuteDCOM",
#    "GenericAll","GetChanges","GetChangesAll","GpLink","HasSession",
#    "MemberOf","Open","Owns","WriteDacl","WriteOwner"]
# Severities below are standard AD attack-path tradecraft (BloodHound-style
# edge weighting), not derived from any model -- this is the "expert rule"
# baseline the GNN has to beat.
# --------------------------------------------------------------------------- #
EDGE_TYPES = ["AdminTo", "AllowedToDelegate", "CanRDP", "Contains", "DCSync",
              "ExecuteDCOM", "GenericAll", "GetChanges", "GetChangesAll",
              "GpLink", "HasSession", "MemberOf", "Open", "Owns", "WriteDacl",
              "WriteOwner"]
EDGE_SEVERITY = torch.tensor([
    0.80,  # AdminTo            -- local admin -> credential dumping
    0.65,  # AllowedToDelegate  -- constrained delegation abuse
    0.55,  # CanRDP             -- interactive access, not admin
    0.10,  # Contains           -- structural (OU hierarchy)
    1.00,  # DCSync             -- full domain credential replication
    0.70,  # ExecuteDCOM        -- remote code execution
    0.95,  # GenericAll         -- full control over the target object
    0.50,  # GetChanges         -- partial replication right
    0.90,  # GetChangesAll      -- paired with GetChanges = DCSync capability
    0.10,  # GpLink             -- structural (GPO application)
    0.45,  # HasSession         -- credential harvesting from a live session
    0.25,  # MemberOf           -- indirect, via group membership
    0.60,  # Open               -- RCE if a CVE is present
    0.85,  # Owns               -- already controls the object
    0.90,  # WriteDacl          -- can grant itself any permission
    0.90,  # WriteOwner         -- can take ownership, then grant permissions
])


def heuristic_propensity(edge_type_multihot: torch.Tensor) -> torch.Tensor:
    """(M, 16) multi-hot -> (M,) score = max severity over active edge types."""
    scored = edge_type_multihot * EDGE_SEVERITY.unsqueeze(0)
    return scored.max(dim=1).values


def uniform_propensity(edge_type_multihot: torch.Tensor) -> torch.Tensor:
    return torch.full((edge_type_multihot.shape[0],), 0.5)


# --------------------------------------------------------------------------- #
# Ranking metrics (manual -- no sklearn)
# --------------------------------------------------------------------------- #
def average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    Area under the precision-recall curve, rank-based (matches the standard
    definition used by sklearn.metrics.average_precision_score):

        AP = sum_i( precision@i * is_positive(i) ) / num_positives

    where i ranges over items sorted by score, descending.
    """
    y_true = y_true.detach().flatten().float()
    y_score = y_score.detach().flatten().float()
    n_pos = y_true.sum().item()
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(y_score, descending=True)
    y_sorted = y_true[order]
    cum_tp = torch.cumsum(y_sorted, dim=0)
    ranks = torch.arange(1, len(y_sorted) + 1, dtype=torch.float32)
    precision_at_i = cum_tp / ranks
    return float((precision_at_i * y_sorted).sum() / n_pos)


def precision_at_k(y_true: torch.Tensor, y_score: torch.Tensor, k: int) -> float:
    """Fraction of the top-k scored items that are true positives."""
    y_true = y_true.detach().flatten().float()
    y_score = y_score.detach().flatten().float()
    k = min(k, y_score.numel())
    if k == 0:
        return float("nan")
    top = torch.topk(y_score, k).indices
    return float(y_true[top].mean())


def jaccard(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- #
# (A) Pooled ranking-metric ablation, over the WHOLE test split
# --------------------------------------------------------------------------- #
def part_a_pooled_metrics(samples: List[AttackGraphSample],
                          infer: GNNInference) -> Dict[str, dict]:
    pooled_scores = {"gnn": [], "heuristic": [], "uniform": []}
    pooled_labels = []
    for s in samples:
        pooled_scores["gnn"].append(infer.predict(s))
        pooled_scores["heuristic"].append(heuristic_propensity(s.edge_type_multihot))
        pooled_scores["uniform"].append(uniform_propensity(s.edge_type_multihot))
        pooled_labels.append(s.edge_label)
    labels = torch.cat(pooled_labels)

    out = {}
    for scheme, chunks in pooled_scores.items():
        scores = torch.cat(chunks)
        out[scheme] = {
            "roc_auc": roc_auc(labels, scores),
            "average_precision": average_precision(labels, scores),
            "precision_at_10": precision_at_k(labels, scores, 10),
            "precision_at_50": precision_at_k(labels, scores, 50),
            "num_edges": int(scores.numel()),
            "num_positive": int(labels.sum().item()),
        }
    return out


# --------------------------------------------------------------------------- #
# (B) Path-quality ablation: predicted path vs the TRUE attack path
# --------------------------------------------------------------------------- #
def part_b_path_quality(samples: List[AttackGraphSample],
                        infer: GNNInference) -> List[dict]:
    rows = []
    for s in samples:
        src, tgt = attack_path_endpoints(s)
        true_edges = [(int(u), int(v)) for u, v in
                      (s.y_matrix == 1).nonzero(as_tuple=False).tolist()]
        edge_list = s.edge_list()
        row = {"graph": os.path.basename(s.source_path), "source": src,
               "target": tgt, "true_path_edges": len(true_edges)}
        gnn = infer.predict(s)
        for scheme, scores in (("gnn", gnn),
                               ("heuristic", heuristic_propensity(s.edge_type_multihot)),
                               ("uniform", uniform_propensity(s.edge_type_multihot))):
            prop_matrix = [[0.0] * s.num_nodes for _ in range(s.num_nodes)]
            for (u, v), p in zip(edge_list, scores.tolist()):
                prop_matrix[u][v] = p
            adj = adjacency_from_propensity(prop_matrix, edge_list)
            res = dijkstra(adj, src, target=tgt)
            path = res.path_to(tgt)
            pred_edges = list(zip(path[:-1], path[1:]))
            row[f"{scheme}_jaccard"] = jaccard(pred_edges, true_edges)
            row[f"{scheme}_reached_target"] = res.dist.get(tgt, INF) != INF
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# (C) Multi-graph classical-algorithm cross-check (GNN weights)
# --------------------------------------------------------------------------- #
def part_c_crosscheck(samples: List[AttackGraphSample],
                      infer: GNNInference) -> List[dict]:
    rows = []
    for s in samples:
        prop = infer.predict(s)
        inp = build_algorithm_inputs(s, prop, threshold=0.5)
        src, tgt = attack_path_endpoints(s)

        dist, _ = floyd_warshall(inp.weight_matrix)
        fw_cost = dist[src][tgt]
        d_res = dijkstra(inp.adjacency, src, target=tgt)
        d_cost = d_res.dist.get(tgt, INF)
        dist_agree = (fw_cost == INF and d_cost == INF) or \
            (fw_cost != INF and d_cost != INF and abs(fw_cost - d_cost) < 1e-6)

        members, local_edges, _ = largest_connected_component(
            inp.num_nodes, inp.undirected_edges)
        k_res = kruskal(len(members), local_edges)
        p_adj = adjacency_from_edges(len(members), local_edges)
        p_res = prim(len(members), p_adj, start=0)
        mst_agree = abs(k_res.total_weight - p_res.total_weight) < 1e-6

        rows.append({
            "graph": os.path.basename(s.source_path),
            "dijkstra_eq_floyd_warshall": dist_agree,
            "fw_cost": None if fw_cost == INF else fw_cost,
            "dijkstra_cost": None if d_cost == INF else d_cost,
            "kruskal_eq_prim": mst_agree,
            "kruskal_weight": k_res.total_weight,
            "prim_weight": p_res.total_weight,
            "component_nodes": len(members),
        })
    return rows


# --------------------------------------------------------------------------- #
# (D) Threshold sweep for Warshall's reachability, averaged over a few graphs
# --------------------------------------------------------------------------- #
def part_d_threshold_sweep(samples: List[AttackGraphSample], infer: GNNInference,
                           thresholds: List[float]) -> List[dict]:
    rows = []
    for thr in thresholds:
        pairs, sizes = [], []
        for s in samples:
            prop = infer.predict(s)
            inp = build_algorithm_inputs(s, prop, threshold=thr)
            reach = copy.deepcopy(inp.reachability)
            transitive_closure(reach, snapshot_matrices=False)
            src, _ = attack_path_endpoints(s)
            pairs.append(sum(c for row in reach for c in row))
            sizes.append(len(reachable_set(reach, src)))
        rows.append({
            "threshold": thr,
            "mean_reachable_pairs": sum(pairs) / len(pairs),
            "mean_reach_from_source": sum(sizes) / len(sizes),
        })
    return rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_ablation(metrics: Dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = ["uniform", "heuristic", "gnn"]
    metric_names = ["roc_auc", "average_precision", "precision_at_10",
                    "precision_at_50"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(metric_names))
    width = 0.25
    for i, scheme in enumerate(schemes):
        vals = [metrics[scheme][m] for m in metric_names]
        ax.bar([xi + i * width for xi in x], vals, width, label=scheme)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(["ROC-AUC", "Avg Precision", "P@10", "P@50"])
    ax.set_ylim(0, 1.02)
    ax.set_title("Edge-ranking ablation: GNN vs. non-learned baselines "
                 "(pooled over the test split)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


def plot_threshold_sweep(rows: List[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    thr = [r["threshold"] for r in rows]
    ax.plot(thr, [r["mean_reachable_pairs"] for r in rows], "-o",
            label="mean reachable pairs (whole graph)")
    ax2 = ax.twinx()
    ax2.plot(thr, [r["mean_reach_from_source"] for r in rows], "-o",
            color="darkorange", label="mean |reach(source)|")
    ax.set_xlabel("propensity threshold")
    ax.set_ylabel("mean reachable pairs")
    ax2.set_ylabel("mean |reach(source)|", color="darkorange")
    ax.set_title("Warshall reachability vs. propensity threshold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


# --------------------------------------------------------------------------- #
# CSV helper
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", default=os.path.join("data", "_data_"))
    p.add_argument("--checkpoint", default=os.path.join("checkpoints", "latest.pt"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--path-sample", type=int, default=50,
                  help="graphs used for part (B) path-quality ablation")
    p.add_argument("--crosscheck-sample", type=int, default=30,
                  help="graphs used for part (C) algorithm cross-check")
    p.add_argument("--sweep-sample", type=int, default=15,
                  help="graphs used for part (D) threshold sweep")
    p.add_argument("--quick", action="store_true",
                  help="tiny sample sizes, for a fast smoke test")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.quick:
        args.path_sample, args.crosscheck_sample, args.sweep_sample = 5, 5, 3

    device = require_cuda(verbose=True)
    infer = GNNInference.from_checkpoint(args.checkpoint, device)

    dataset = AttackGraphDataset(args.data_dir, as_pyg=False)
    n = len(dataset)
    n_train = int(round(n * 0.8))
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    test_paths = [dataset.paths[i] for i in perm[n_train:]]
    print(f"test split: {len(test_paths)} graphs")

    print(f"\n[A] loading + scoring all {len(test_paths)} test graphs for "
         f"pooled ranking metrics ...")
    all_samples = [load_sample(p) for p in test_paths]
    metrics_a = part_a_pooled_metrics(all_samples, infer)
    print("    scheme      ROC-AUC   AvgPrec   P@10    P@50")
    for scheme in ("uniform", "heuristic", "gnn"):
        m = metrics_a[scheme]
        print(f"    {scheme:<10}  {m['roc_auc']:.4f}    "
             f"{m['average_precision']:.4f}    "
             f"{m['precision_at_10']:.4f}  {m['precision_at_50']:.4f}")

    print(f"\n[B] path-quality ablation on {args.path_sample} sampled graphs ...")
    rows_b = part_b_path_quality(all_samples[:args.path_sample], infer)
    for scheme in ("uniform", "heuristic", "gnn"):
        mean_j = sum(r[f"{scheme}_jaccard"] for r in rows_b) / len(rows_b)
        reach_rate = sum(r[f"{scheme}_reached_target"] for r in rows_b) / len(rows_b)
        print(f"    {scheme:<10}  mean Jaccard(pred path, true path) = "
             f"{mean_j:.4f}   reached target {reach_rate*100:.1f}% of graphs")

    print(f"\n[C] classical-algorithm cross-check on {args.crosscheck_sample} "
         f"sampled graphs ...")
    rows_c = part_c_crosscheck(all_samples[:args.crosscheck_sample], infer)
    dj_rate = sum(r["dijkstra_eq_floyd_warshall"] for r in rows_c) / len(rows_c)
    mst_rate = sum(r["kruskal_eq_prim"] for r in rows_c) / len(rows_c)
    print(f"    Dijkstra == Floyd-Warshall on {dj_rate*100:.1f}% of graphs")
    print(f"    Kruskal  == Prim            on {mst_rate*100:.1f}% of graphs")

    print(f"\n[D] threshold sweep on {args.sweep_sample} sampled graphs ...")
    thresholds = [round(0.1 * i, 1) for i in range(1, 10)]
    rows_d = part_d_threshold_sweep(all_samples[:args.sweep_sample], infer,
                                    thresholds)
    for r in rows_d:
        print(f"    thr={r['threshold']:.1f}  mean reachable pairs="
             f"{r['mean_reachable_pairs']:.1f}  mean |reach(source)|="
             f"{r['mean_reach_from_source']:.1f}")

    os.makedirs(LOG_DIR, exist_ok=True)
    write_csv(os.path.join(LOG_DIR, "path_quality.csv"), rows_b)
    write_csv(os.path.join(LOG_DIR, "multigraph_crosscheck.csv"), rows_c)
    write_csv(os.path.join(LOG_DIR, "threshold_sweep.csv"), rows_d)
    with open(os.path.join(LOG_DIR, "ablation_metrics.csv"), "w", newline="",
             encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scheme", "roc_auc", "average_precision", "precision_at_10",
                   "precision_at_50", "num_edges", "num_positive"])
        for scheme, m in metrics_a.items():
            w.writerow([scheme, m["roc_auc"], m["average_precision"],
                       m["precision_at_10"], m["precision_at_50"],
                       m["num_edges"], m["num_positive"]])

    plot_ablation(metrics_a, os.path.join(LOG_DIR, "ablation_metrics.png"))
    plot_threshold_sweep(rows_d, os.path.join(LOG_DIR, "threshold_sweep.png"))

    report = {
        "test_split_size": len(test_paths),
        "pooled_ranking_metrics": metrics_a,
        "path_quality_sample_size": len(rows_b),
        "path_quality_mean_jaccard": {
            scheme: sum(r[f"{scheme}_jaccard"] for r in rows_b) / len(rows_b)
            for scheme in ("uniform", "heuristic", "gnn")
        },
        "crosscheck_sample_size": len(rows_c),
        "crosscheck_agreement_rate": {"dijkstra_eq_floyd_warshall": dj_rate,
                                      "kruskal_eq_prim": mst_rate},
        "threshold_sweep": rows_d,
    }
    with open(os.path.join(LOG_DIR, "evaluation_report.json"), "w",
             encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nwrote logs/evaluation_report.json, ablation_metrics.{{csv,png}}, "
         f"path_quality.csv, multigraph_crosscheck.csv, "
         f"threshold_sweep.{{csv,png}}")


if __name__ == "__main__":
    main()
