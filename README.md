# Attack-Graph GNN + Classical Graph Algorithms

[![tests](https://github.com/Aaryan658/attack-graph-gnn/actions/workflows/tests.yml/badge.svg)](https://github.com/Aaryan658/attack-graph-gnn/actions/workflows/tests.yml)

A research-style pipeline that combines a **Graph Neural Network** with **five
from-scratch classical graph algorithms** to analyse Active Directory (AD)
**attack graphs**.

The GNN learns, per edge, how likely that edge is to lie on a real attack path
("attack propensity"). Those learned scores are then fed as **edge weights**
into hand-implemented Warshall, Floyd-Warshall, Dijkstra, Kruskal and Prim, each
answering a different security question about the graph.

---

## 1. The problem

An AD environment is a directed graph:

* **nodes** = users, computers, groups, OUs, GPOs, domains (361 per graph here);
* **edges** = privilege / relationship edges of **16 types** (`AdminTo`,
  `DCSync`, `GenericAll`, `HasSession`, `MemberOf`, ...);
* a **labelled attack path** (`Y`) marks the specific edges an attacker would
  traverse from a compromised host (`owned`) to a high-value target
  (`highvalue` / `target`).

Dataset: 1033 pre-processed samples from the
[PhD_Replication_Package](https://github.com/mbdlrocks/PhD_Replication_Package)
("Physics-Informed-GNN (PIGNN)" / `_data_.zip`). Each `.pt` file is a plain
`dict`:

| key          | shape           | dtype   | meaning                                   |
|--------------|-----------------|---------|-------------------------------------------|
| `adj_tensor` | `(361, 361, 16)`| float32 | binary adjacency, one slice per edge type |
| `X_matrix`   | `(361, 19)`     | float32 | binary node features                      |
| `Y_matrix`   | `(361, 361)`    | int64   | `1` where edge *i to j* is on the attack path |

Only ~0.17 % of edges are on the attack path, so this is a **heavily imbalanced
edge-classification** task.

---

## 2. Part 1 - the GNN (edge propensity)

`src/model.py` - **GraphSAGE** node encoder (3 x `SAGEConv`, mean aggregator) +
an MLP edge head:

```
h            = GraphSAGE(x, edge_index)                 # node embeddings (V, H)
logit(u->v)  = MLP( concat[ h[u], h[v], edge_type_multihot[u,v] ] )
propensity   = sigmoid(logit)   in [0, 1]
```

* Concatenation `h[u] || h[v]` keeps the prediction **direction-sensitive**
  (`u->v` on the path is not the same as `v->u` on the path).
* The 16-d edge-type multi-hot is appended because some AD relations are far
  more attack-relevant than others.

`src/train.py`:

* hard **CUDA precondition check** first (`src/env_check.py`) - aborts loudly if
  the GPU/CUDA is unavailable, never silently runs on CPU;
* deterministic **80 / 20 train/test split** (seeded);
* `BCEWithLogitsLoss(pos_weight = #neg / #pos)` to counter the imbalance;
* per epoch: **loss, accuracy@0.5, ROC-AUC** (Mann-Whitney U, hand-rolled) on
  both splits -> printed **and** appended to `logs/training_log.csv`;
* **checkpoint every 5 epochs** to `checkpoints/` (+ `checkpoints/latest.pt`);
  on start-up it **resumes** from the newest checkpoint automatically.

`logs/training_log.csv` is the source for the report's training-curve chart
(`python -m src.plot_training` renders `logs/training_curve.png`).

---

## 3. Part 2 - the five classical algorithms (from scratch)

All in `src/algorithms/`, **no `networkx` algorithm calls**. Each file has a
docstring stating the time complexity, inline step comments, and a `__main__`
demo on a small toy graph with printed step-by-step output:

```
python src/algorithms/warshall.py
python src/algorithms/floyd_warshall.py
python src/algorithms/dijkstra.py
python src/algorithms/kruskal.py
python src/algorithms/prim.py
```

| # | Algorithm | File | Complexity | Sub-task it answers here |
|---|-----------|------|-----------|--------------------------|
| 1 | **Warshall** (transitive closure) | `warshall.py` | O(V^3) triple loop | Threshold the propensities into a boolean "high-risk edge" graph, then compute **reachability**: from a compromised host, which machines can the attacker ultimately reach following only high-propensity edges? |
| 2 | **Floyd-Warshall** (all-pairs shortest path) | `floyd_warshall.py` | O(V^3) triple loop + `nxt` matrix | With `cost = 1 - propensity`, get the **cheapest path between every pair** of nodes = the most plausible full attack route for every (start, target) simultaneously. |
| 3 | **Dijkstra** (single-source shortest path) | `dijkstra.py` | O((V+E) log V), manual binary-heap PQ + predecessor array | The focused version of (2): the single **most plausible attack path from one compromised node to one target**, returning the actual node sequence, not just the cost. |
| 4 | **Kruskal** (MST) | `kruskal.py` | O(E log E), from-scratch Union-Find (path compression + union by rank) | On the undirected GNN-weighted graph, the **cheapest sub-network that still connects every reachable host** - a compact "attack-surface backbone". |
| 5 | **Prim** (MST) | `prim.py` | O(E log V), manual binary-heap PQ | Same MST, grown from a start node by a different method - run alongside Kruskal as an **independent cross-check** that both correct algorithms report the same total weight. |

`heapq` is used as the binary-heap priority queue in Dijkstra & Prim - that is a
standard data structure, not a shortcut around the algorithm logic (unlike, say,
`nx.shortest_path`).

---

## 4. Part 3 - integration

`src/integration.py` converts the GNN's **per-edge propensity vector** into the
exact structure each algorithm wants:

| Consumer | Structure built | Rule |
|----------|-----------------|------|
| Warshall | `V x V` boolean matrix | `reach[i][j] = propensity[i][j] >= threshold` |
| Floyd-Warshall | `V x V` float matrix | `1 - propensity` on edges, `inf` off-edges, `0` diagonal |
| Dijkstra | adjacency list `{u: [(v, w)]}` | `w = (1 - propensity) + eps >= 0` (Dijkstra needs non-negative weights) |
| Kruskal / Prim | undirected edge list `[(u, v, w)]` | collapse each `{u,v}` pair to its **min** directed cost; MST run on the largest connected component so Kruskal and Prim are comparable |

`src/main.py` runs the whole thing:

```
load data -> load checkpoints/latest.pt (or train) -> pick one test graph
   -> GNN inference -> build structures -> run all 5 algorithms
   -> print a per-algorithm summary -> write Part 4 trace artifacts
```

The summary prints: Warshall's reachable-set size and source-reaches-target?,
Floyd-Warshall's distance-matrix summary + reconstructed source->target path,
Dijkstra's path + cost (and a check that it equals Floyd-Warshall's for that
pair), and both MST totals with a **Kruskal == Prim** agreement check.

---

## 5. Part 4 - explainability artifacts

Every algorithm takes an optional `trace` list and records its intermediate
state (Warshall's matrix after each `k`; Floyd-Warshall's improved pairs per
`k`; Dijkstra's frontier-expansion order + relaxations; Kruskal's sorted edges
with accept/reject; Prim's tree growth). `src/explain/logger.py` writes per run:

* `logs/explain/<run_id>.json` - full nested trace (matrices included),
  intended as input for a future step-by-step animation;
* `logs/explain/<run_id>__<algo>.csv` - one flat, skimmable row per step, to
  demonstrate the algorithm is doing the right thing rather than just emitting a
  final answer.

---

## 6. Does the GNN actually help? (evaluation & ablation)

`src/evaluate.py` compares the trained GNN against two **non-learned**
baselines on the untouched test split (207 graphs):

* `uniform` -- every edge scored 0.5 (the "no model at all" floor).
* `heuristic` -- a fixed AD-security-analyst severity per edge type (DCSync /
  GenericAll / WriteDacl highest, GpLink / Contains lowest; the 16-channel
  order was recovered from the upstream `_graphPreprocess_.ipynb`, not
  guessed). This is domain expertise, not a second model.

**Pooled edge-ranking metrics, all 207 test graphs (~600k edges, 0.17% positive):**

| scheme | ROC-AUC | Average Precision | P@10 | P@50 |
|---|---|---|---|---|
| uniform | 0.500 | 0.001 | 0.000 | 0.000 |
| heuristic | 0.333 | 0.001 | 0.000 | 0.000 |
| **gnn** | **0.993** | **0.316** | **1.000** | **0.700** |

The heuristic baseline scoring *below* random (0.333) is a real, non-cherry-
picked finding: the sampled attack paths in this dataset don't preferentially
use "high-severity" edge types, so an expert rule alone actively misleads.
The GNN is not just better than nothing -- it is the only scheme that beats
chance at all.

**Path-quality ablation** (Dijkstra from the true attack-path start to its
end, Jaccard overlap between the predicted path's edges and the true path,
averaged over 50 sampled test graphs):

| scheme | mean Jaccard vs. true path |
|---|---|
| uniform | 0.580 |
| heuristic | 0.564 |
| **gnn** | **0.638** |

**Multi-graph algorithm cross-check** (30 sampled test graphs, GNN weights):
Dijkstra's cost equalled Floyd-Warshall's, and Kruskal's MST total equalled
Prim's, on **100%** of sampled graphs -- the single-graph agreement shown by
`main.py` holds up at scale, not just on one lucky example.

**Threshold sweep** (Warshall's reachable-set size vs. propensity threshold,
averaged over 15 graphs) shows a smooth, monotonic collapse from ~4,900 mean
reachable pairs at threshold 0.1 down to ~84 at threshold 0.9 -- see
`logs/threshold_sweep.png`.

```bash
python -m src.evaluate                 # full run, writes logs/evaluation_report.json
python -m src.evaluate --quick         # tiny sample sizes, fast smoke test
```

Outputs: `logs/evaluation_report.json`, `ablation_metrics.{csv,png}`,
`path_quality.csv`, `multigraph_crosscheck.csv`, `threshold_sweep.{csv,png}`.

### Honest limitations

* Average Precision (0.32) is far from 1.0 -- with 0.17% positive edges this
  is a hard ranking problem; ROC-AUC alone would have overstated how good the
  model is.
* The path-quality gain (0.64 vs 0.58 Jaccard) is real but modest; the GNN's
  advantage is much sharper at the edge-ranking level (P@10 = 1.0) than at the
  whole-path level.
* Everything above is evaluated on one dataset/source. No cross-dataset or
  live-environment generalisation test exists yet: the AD graphs are
  synthetic topology from a published dataset (François et al., 2025), not a
  live-collected BloodHound/SharpHound export from a real or lab AD domain.
  A live-collection validation pass would need a real AD environment (e.g. a
  GOAD lab) to gather data from, which was scoped out of this project.
* The classical algorithms are O(V^3) (Warshall, Floyd-Warshall); fine for the
  361-node graphs here, would need a sparse-graph rework at real-world scale.

### Interactive step-through animation

`src/animation.html` (self-contained, open directly in a browser, or see the
published Artifact) lets you step through all five algorithms on toy graphs,
plus Dijkstra and Prim on the real 361-node graph. Rebuild it after changing
a trace with:

```bash
python -m src.export_traces   # regenerates logs/animation_data.json
                              # and rebuilds src/animation.html from
                              # src/animation.template.html
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

21 tests cover the five algorithms (including edge cases: ragged matrices,
negative weights, disconnected graphs), the shape-based loader, and the
model's forward/backward pass -- all synthetic, no dataset or GPU required, so
they also run in CI (`.github/workflows/tests.yml`).

---

## 7. Setup & run

**Environment** (already provisioned on this machine): Python 3.14,
`torch 2.13.0+cu126`, `torch_geometric 2.8.0.post1`, CUDA 12.6, NVIDIA
GeForce RTX 4060. `torch.cuda.is_available()` must be `True` or the training /
pipeline scripts abort by design.

```bash
# 1. dependencies  (torch must be the CUDA build -- see requirements.txt)
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install -r requirements.txt

# 2. dataset  (clone + unzip the 1033 .pt samples into data/_data_/)
git clone https://github.com/mbdlrocks/PhD_Replication_Package.git external/PhD_Replication_Package
python - <<'PY'
import zipfile
zipfile.ZipFile(r"external/PhD_Replication_Package/Physics-Informed-GNN (PIGNN)/_Preprocessing_/_data_.zip").extractall("data")
PY

# 3. sanity checks
python src/env_check.py                 # prints the detected RTX 4060
python src/diagnostic.py --n 3          # prints the real .pt schema
python -m src.data_loader               # parses one sample, shows shapes

# 4. Part 2 demos (no dataset / GPU needed)
python src/algorithms/warshall.py
python src/algorithms/floyd_warshall.py
python src/algorithms/dijkstra.py
python src/algorithms/kruskal.py
python src/algorithms/prim.py

# 5. train the GNN  (resumes from the latest checkpoint if present)
python -m src.train --epochs 40 --batch-size 32
python -m src.plot_training             # logs/training_curve.png

# 6. full pipeline on one test graph
python -m src.main
python -m src.main --graph 5 --threshold 0.6      # pick a different test graph
python -m src.main --train-if-missing --epochs 20 # train then run in one go

# 7. does the GNN help? (ablation, multi-graph cross-check, threshold sweep)
python -m src.evaluate

# 8. tests
pip install -r requirements-dev.txt && pytest tests/ -v
```

### Layout

```
requirements.txt / requirements-dev.txt
.github/workflows/tests.yml        # CI: pytest on every push (CPU-only)
data/_data_/*.pt                   # extracted dataset (git-ignored)
external/PhD_Replication_Package/  # cloned upstream (git-ignored)
checkpoints/                       # ckpt_epoch_XXXX.pt, latest.pt (git-ignored)
logs/
  diagnostic_report.txt
  training_log.csv                 # per-epoch metrics -> report chart
  training_curve.png
  evaluation_report.json           # ablation + cross-check + sweep, all in one
  ablation_metrics.{csv,png}
  path_quality.csv
  multigraph_crosscheck.csv
  threshold_sweep.{csv,png}
  explain/<run_id>.json + <run_id>__<algo>.csv  (git-ignored, regenerable)
src/
  env_check.py                     # hard CUDA gate
  diagnostic.py                    # dataset schema discovery
  data_loader.py       (Part 1)    # shape-auto-detecting loader + Dataset/split
  model.py             (Part 1)    # GraphSAGE edge classifier
  train.py             (Part 1)    # train / checkpoint / resume / CSV log
  plot_training.py                 # training-curve figure
  integration.py       (Part 3)    # GNN output -> algorithm input structures
  main.py              (Part 3)    # end-to-end pipeline
  algorithms/          (Part 2)    # warshall / floyd_warshall / dijkstra / kruskal / prim
  explain/             (Part 4)    # trace -> JSON + CSV
  export_traces.py                 # trace data -> animation.html
  animation.html / animation.template.html   # interactive step-through player
  evaluate.py                      # ablation, multi-graph cross-check, sweep
tests/
  test_algorithms.py               # the 5 algorithms, edge cases, DisjointSet
  test_pipeline.py                 # shape-detection loader + model, synthetic data
```

### Citation

Dataset / task from: Francois, Arduin, Merad, *"Physics Informed Graph Neural
Networks for Attack Path Prediction"* (2025).
