"""
logger.py -- serialise algorithm execution traces to JSON + CSV (PART 4).

The classical algorithms each accept a ``trace`` list and append one dict per
step. This module collects those lists for a whole pipeline run and writes:

    logs/explain/<run_id>.json            -- everything, nested, animation-ready
    logs/explain/<run_id>__warshall.csv   -- one row per k iteration
    logs/explain/<run_id>__floyd_warshall.csv -- one row per k iteration
    logs/explain/<run_id>__dijkstra.csv   -- one row per settled vertex
    logs/explain/<run_id>__kruskal.csv    -- one row per edge examined
    logs/explain/<run_id>__prim.csv       -- one row per vertex added

JSON keeps full matrices (needed to animate the reachability / distance grid
filling in). The CSVs deliberately DROP the big matrices and keep only scalar
per-step facts, so they stay small and skimmable when shown to a human.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Per-algorithm CSV flatteners
# --------------------------------------------------------------------------- #
def _flatten_matrix_trace(trace: List[dict], value_key: str) -> List[dict]:
    """
    Shared flattener for Warshall's ``matrix`` trace and Floyd-Warshall's
    ``dist`` trace: keep step id + k, replace the matrix with cheap summaries.
    """
    rows = []
    for snap in trace:
        grid = snap.get(value_key)
        if grid is not None:
            n_cells = sum(len(r) for r in grid)
            if value_key == "matrix":  # boolean 0/1 reachability
                filled = sum(int(c) for r in grid for c in r)
            else:                      # distance grid; "filled" = finite entries
                filled = sum(1 for r in grid for c in r if c != "inf")
        else:
            n_cells = filled = None
        changed = (snap.get("changed_cells")
                   or snap.get("improved_pairs") or [])
        rows.append({
            "step": snap.get("step"),
            "k": snap.get("k"),
            "cells_filled": filled,
            "cells_total": n_cells,
            "num_changed_this_step": len(changed),
        })
    return rows


def _flatten_dijkstra(trace: List[dict]) -> List[dict]:
    rows = []
    for ev in trace:
        rows.append({
            "pop_order": ev.get("pop_order"),
            "settled_vertex": ev.get("settled_vertex"),
            "settled_distance": ev.get("settled_distance"),
            "frontier_size": len(ev.get("frontier", [])),
            "frontier": ";".join(f"{v}:{d}" for v, d in ev.get("frontier", [])),
            "num_relaxations": len(ev.get("relaxations", [])),
            "relaxations": ";".join(
                f"{r['edge'][0]}->{r['edge'][1]}={r['new_dist']}"
                for r in ev.get("relaxations", [])
            ),
        })
    return rows


def _flatten_kruskal(trace: List[dict]) -> List[dict]:
    return [{
        "examined_rank": d.get("examined_rank"),
        "edge": f"{d['edge'][0]}--{d['edge'][1]}" if d.get("edge") else None,
        "weight": d.get("weight"),
        "decision": d.get("decision"),
        "edges_in_tree": d.get("edges_in_tree"),
        "running_weight": d.get("running_weight"),
        "components_remaining": d.get("components_remaining"),
    } for d in trace]


def _flatten_prim(trace: List[dict]) -> List[dict]:
    rows = []
    for d in trace:
        via = d.get("via_edge")
        rows.append({
            "added_vertex": d.get("added_vertex"),
            "via_edge": f"{via[0]}--{via[1]}" if via else "(seed)",
            "edge_weight": d.get("edge_weight"),
            "edges_in_tree": d.get("edges_in_tree"),
            "running_weight": d.get("running_weight"),
            "fringe_size": d.get("fringe_size"),
            "candidates_pushed": d.get("candidates_pushed"),
        })
    return rows


# Map an algorithm key -> (flattener, note about the row granularity).
_FLATTENERS = {
    "warshall": (lambda t: _flatten_matrix_trace(t, "matrix"),
                 "one row per intermediate vertex k"),
    "floyd_warshall": (lambda t: _flatten_matrix_trace(t, "dist"),
                       "one row per intermediate vertex k"),
    "dijkstra": (_flatten_dijkstra, "one row per settled vertex"),
    "kruskal": (_flatten_kruskal, "one row per edge examined (sorted order)"),
    "prim": (_flatten_prim, "one row per vertex pulled into the tree"),
}


# --------------------------------------------------------------------------- #
# Run container
# --------------------------------------------------------------------------- #
@dataclass
class ExplainRun:
    """
    Collects the traces + summary results of running the classical algorithms
    on one graph, then serialises them.

    Attributes
    ----------
    run_id
        Unique, filesystem-safe identifier for this run.
    metadata
        Free-form context: graph id, propensity threshold, source/target nodes,
        model checkpoint, timestamps, etc.
    traces
        ``{algorithm_key: trace_list}``.
    results
        ``{algorithm_key: summary_dict}`` -- the final answer of each algorithm
        (reachable set size, MST total weight, Dijkstra path + cost, ...).
    """

    run_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    traces: Dict[str, List[dict]] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(graph_id: str, **metadata: Any) -> "ExplainRun":
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_graph = "".join(c if c.isalnum() or c in "-_" else "_"
                             for c in str(graph_id))
        run = ExplainRun(run_id=f"run_{stamp}_{safe_graph}")
        run.metadata = {
            "graph_id": graph_id,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            **metadata,
        }
        return run

    def add(self, algorithm_key: str, trace: List[dict],
            result: Any = None) -> None:
        self.traces[algorithm_key] = trace
        if result is not None:
            self.results[algorithm_key] = result


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def _json_default(obj: Any):
    """Fallback encoder for stray tensor / set / numpy types."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def save_explainability_artifacts(
    run: ExplainRun,
    out_dir: str = os.path.join("logs", "explain"),
) -> Dict[str, str]:
    """
    Write ``<out_dir>/<run_id>.json`` plus one
    ``<out_dir>/<run_id>__<algo>.csv`` per algorithm that has a trace.

    Returns a ``{artifact_name: path}`` map of everything written.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}

    # ---- full JSON ------------------------------------------------------- #
    json_path = os.path.join(out_dir, f"{run.run_id}.json")
    payload = {
        "run_id": run.run_id,
        "metadata": run.metadata,
        "results": run.results,
        "traces": run.traces,
        "csv_granularity": {k: note for k, (_, note) in _FLATTENERS.items()
                            if k in run.traces},
    }
    # Compact separators: this file is machine input for a future animation, and
    # on a 361-node graph the per-step deltas pretty-print to tens of MB. The
    # CSVs below are the human-readable view.
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), default=_json_default)
    written["json"] = json_path

    # ---- one CSV per algorithm ---------------------------------------- #
    for algo_key, trace in run.traces.items():
        flatten, _note = _FLATTENERS.get(
            algo_key, (lambda t: [{"step_index": i, **_scalars_only(d)}
                                  for i, d in enumerate(t)], "")
        )
        rows = flatten(trace)
        if not rows:
            continue
        csv_path = os.path.join(out_dir, f"{run.run_id}__{algo_key}.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        written[f"csv:{algo_key}"] = csv_path

    return written


def _scalars_only(d: dict) -> dict:
    """Keep only JSON-scalar values from a step dict (drop nested lists/grids)."""
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float, str, bool)) or v is None}


# --------------------------------------------------------------------------- #
# Demo -- builds a tiny run from the algorithm demos and writes artifacts.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import sys

    # Make "src" importable when run as a loose script.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(here)))

    from src.algorithms.warshall import transitive_closure
    from src.algorithms.dijkstra import dijkstra, Adjacency

    run = ExplainRun.new("demo_graph", note="logger self-test")

    # Warshall on a 4-node chain 0->1->2->3.
    reach = [[False] * 4 for _ in range(4)]
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        reach[u][v] = True
    w_trace: list = []
    transitive_closure(reach, trace=w_trace)
    run.add("warshall", w_trace,
            result={"reachable_from_0": [j for j in range(4) if reach[0][j]]})

    # Dijkstra on a tiny weighted graph.
    adj: Adjacency = {0: [(1, 1.0), (2, 4.0)], 1: [(2, 1.0), (3, 5.0)],
                      2: [(3, 1.0)]}
    d_trace: list = []
    res = dijkstra(adj, 0, target=3, trace=d_trace)
    run.add("dijkstra", d_trace,
            result={"path_0_to_3": res.path_to(3), "cost": res.dist[3]})

    written = save_explainability_artifacts(
        run, out_dir=os.path.join("logs", "explain")
    )
    print("wrote:")
    for name, path in written.items():
        print(f"  {name:16s} {path}  ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    _demo()
