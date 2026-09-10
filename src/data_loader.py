"""
data_loader.py -- PART 1 loader for the Active-Directory attack-graph .pt
samples.

WHAT ONE SAMPLE LOOKS LIKE (verified by src/diagnostic.py)
---------------------------------------------------------
Each ``.pt`` file is a plain ``dict`` with three tensors::

    'adj_tensor' : float32,  shape (V, V, 16)   -- 16 binary edge-type slices
    'X_matrix'   : float32,  shape (V, 19)      -- binary node features
    'Y_matrix'   : int64,    shape (V, V)       -- 1 where edge i->j is on the
                                                  labelled attack path

with ``V == 361`` for every retained graph.

DESIGN CHOICE: DETECT BY SHAPE, NOT BY KEY NAME
----------------------------------------------
The brief says not to trust the key strings. So :func:`classify_fields` looks at
each tensor's rank / squareness / dtype and decides which role it plays:

    * rank-3 with a square (V, V) face          -> adjacency tensor
    * rank-2 square (V, V)                       -> attack-path target matrix
    * rank-2 rectangular (V, p), p != V          -> node feature matrix

The same classifier is reused whether the container is a ``dict``, a
``list``/``tuple``, or a ``torch_geometric.data.Data`` object.

EVERYTHING IS VALIDATED
-----------------------
:func:`load_sample` raises :class:`MalformedSampleError` with a precise message
on: missing role, inconsistent ``V`` across the three tensors, a non-square
target, a target that is not binary, an all-zero adjacency, etc. Nothing is
silently coerced.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[data_loader] could not import torch: {exc!r}")


DEFAULT_DATA_DIR = os.path.join("data", "_data_")


class MalformedSampleError(ValueError):
    """A .pt sample did not match the expected attack-graph structure."""


# ========================================================================== #
# In-memory representation
# ========================================================================== #
@dataclass
class AttackGraphSample:
    """
    One parsed attack-graph, in a form both the GNN and the classical
    algorithms can consume.

    Attributes
    ----------
    adj_tensor   : (V, V, E) float32     -- per-edge-type binary adjacency.
    x            : (V, F)   float32      -- node features.
    y_matrix     : (V, V)   int64        -- attack-path target adjacency.
    edge_index   : (2, M)   int64        -- COO connectivity, one column per
                                           directed edge that exists in ANY
                                           edge-type slice.
    edge_type_multihot : (M, E) float32  -- for edge k, which of the E edge
                                           types connect its endpoints.
    edge_label   : (M,)     float32      -- y_matrix gathered at each edge;
                                           1.0 => that edge is on the attack
                                           path. This is the GNN's target.
    num_nodes    : int
    source_path  : str
    """

    adj_tensor: "torch.Tensor"
    x: "torch.Tensor"
    y_matrix: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_type_multihot: "torch.Tensor"
    edge_label: "torch.Tensor"
    num_nodes: int
    source_path: str = ""
    meta: dict = field(default_factory=dict)

    # -- convenience views used by PART 3 -------------------------------- #
    def dense_binary_adjacency(self) -> "torch.Tensor":
        """(V, V) uint8 -- 1 where a directed edge exists in any edge type."""
        return (self.adj_tensor.sum(dim=-1) > 0).to(torch.uint8)

    def edge_list(self) -> List[Tuple[int, int]]:
        """The existing directed edges as a list of ``(u, v)`` int pairs."""
        ei = self.edge_index
        return [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])]


# ========================================================================== #
# Robust torch.load
# ========================================================================== #
def robust_torch_load(path: str):
    """
    Load ``path`` coping with the torch >=2.6 ``weights_only=True`` default.

    A pure dict-of-tensors loads fine under the safe path. ``torch_geometric``
    ``Data`` objects and numpy-backed payloads do not, so we retry with
    ``weights_only=False`` -- acceptable because this is our own local dataset,
    not an untrusted download.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such sample file: {path!r}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise MalformedSampleError(
                f"{os.path.basename(path)}: torch.load failed under both "
                f"weights_only=True and weights_only=False -- file is corrupt "
                f"or not a torch archive. Underlying error: {exc!r}"
            ) from exc


# ========================================================================== #
# Shape-based field classification
# ========================================================================== #
def _is_tensor(obj) -> bool:
    return isinstance(obj, torch.Tensor)


def classify_fields(
    tensors: Dict[str, "torch.Tensor"],
) -> Dict[str, str]:
    """
    Map each provided tensor to a role in {"adjacency", "target", "features"}
    using shape and dtype only.

    Parameters
    ----------
    tensors
        ``{label: tensor}``. ``label`` is just for error messages -- it may be a
        dict key, a positional index like ``"[0]"``, or a Data attribute name.

    Returns
    -------
    ``{label: role}`` for every tensor that could be classified. Labels that do
    not look like any known role are omitted (the caller decides whether that
    is fatal).
    """
    roles: Dict[str, str] = {}
    for label, t in tensors.items():
        if not _is_tensor(t):
            continue
        if t.dim() == 3:
            a, b, _c = t.shape
            if a == b:
                roles[label] = "adjacency"
        elif t.dim() == 2:
            r, c = t.shape
            if r == c:
                roles[label] = "target"
            else:
                roles[label] = "features"
    return roles


def _pick_single(roles: Dict[str, str], wanted: str, container_desc: str) -> str:
    """Return the one label mapped to ``wanted``; error if 0 or >1."""
    hits = [lbl for lbl, role in roles.items() if role == wanted]
    if not hits:
        raise MalformedSampleError(
            f"{container_desc}: no tensor looks like the '{wanted}' field "
            f"(classified roles: {roles or 'none'})."
        )
    if len(hits) > 1:
        raise MalformedSampleError(
            f"{container_desc}: {len(hits)} tensors classified as '{wanted}' "
            f"({hits}); cannot disambiguate automatically."
        )
    return hits[0]


# ========================================================================== #
# Derive edge_index / edge features / edge labels from a dense adjacency tensor
# ========================================================================== #
def build_edge_structures(
    adj_tensor: "torch.Tensor",
    y_matrix: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Collapse ``(V, V, E)`` into a directed edge list plus per-edge features and
    per-edge attack-path labels.

    An edge ``u -> v`` is included iff at least one of the E edge-type slices is
    non-zero at ``(u, v)``. Self-loops are dropped.

    Returns
    -------
    edge_index          : (2, M) int64
    edge_type_multihot  : (M, E) float32
    edge_label          : (M,)   float32   -- ``y_matrix[u, v]`` per edge
    """
    if adj_tensor.dim() != 3:
        raise MalformedSampleError(
            f"build_edge_structures expects a rank-3 adjacency tensor, got "
            f"shape {tuple(adj_tensor.shape)}."
        )
    V, V2, E = adj_tensor.shape
    if V != V2:
        raise MalformedSampleError(
            f"adjacency tensor faces are not square: {tuple(adj_tensor.shape)}."
        )

    any_edge = adj_tensor.sum(dim=-1) > 0          # (V, V) bool
    any_edge.fill_diagonal_(False)                 # no self-loops
    src, dst = any_edge.nonzero(as_tuple=True)     # each (M,)
    edge_index = torch.stack([src, dst], dim=0).to(torch.long)

    if edge_index.shape[1] == 0:
        raise MalformedSampleError(
            "adjacency tensor has no off-diagonal edges in any edge type "
            "(all-zero graph)."
        )

    edge_type_multihot = adj_tensor[src, dst, :].to(torch.float32)  # (M, E)
    edge_label = y_matrix[src, dst].to(torch.float32)               # (M,)
    return edge_index, edge_type_multihot, edge_label


# ========================================================================== #
# torch_geometric.data.Data path
# ========================================================================== #
def _from_pyg_data(obj, source_path: str) -> AttackGraphSample:
    """Build an AttackGraphSample from a torch_geometric Data-like object."""
    x = getattr(obj, "x", None)
    edge_index = getattr(obj, "edge_index", None)
    edge_attr = getattr(obj, "edge_attr", None)
    y = getattr(obj, "y", None)

    if not _is_tensor(x) or x.dim() != 2:
        raise MalformedSampleError(
            f"{os.path.basename(source_path)}: torch_geometric Data.x missing "
            f"or not a rank-2 tensor."
        )
    if (not _is_tensor(edge_index) or edge_index.dim() != 2
            or edge_index.shape[0] != 2):
        raise MalformedSampleError(
            f"{os.path.basename(source_path)}: Data.edge_index missing or not "
            f"shaped (2, M)."
        )
    num_nodes = x.shape[0]

    # Reconstruct a dense (V, V, E) adjacency so PART 3 has something uniform to
    # work with. If edge_attr is absent, treat every edge as a single type.
    if _is_tensor(edge_attr) and edge_attr.dim() == 2:
        E = edge_attr.shape[1]
        edge_type_multihot = edge_attr.to(torch.float32)
    else:
        E = 1
        edge_type_multihot = torch.ones(
            edge_index.shape[1], 1, dtype=torch.float32
        )

    adj_tensor = torch.zeros(num_nodes, num_nodes, E, dtype=torch.float32)
    adj_tensor[edge_index[0], edge_index[1], :] = edge_type_multihot

    # Target: prefer a (V, V) y; otherwise fall back to a per-edge y vector.
    if (_is_tensor(y) and y.dim() == 2
            and y.shape[0] == y.shape[1] == num_nodes):
        y_matrix = y.to(torch.long)
        edge_label = y_matrix[edge_index[0], edge_index[1]].to(torch.float32)
    elif _is_tensor(y) and y.dim() == 1 and y.shape[0] == edge_index.shape[1]:
        edge_label = y.to(torch.float32)
        y_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
        y_matrix[edge_index[0], edge_index[1]] = edge_label.to(torch.long)
    else:
        raise MalformedSampleError(
            f"{os.path.basename(source_path)}: torch_geometric Data.y is "
            f"neither a (V,V) matrix nor a per-edge vector; cannot derive "
            f"edge labels."
        )

    _validate(adj_tensor, x, y_matrix, source_path)
    return AttackGraphSample(
        adj_tensor=adj_tensor,
        x=x.to(torch.float32),
        y_matrix=y_matrix,
        edge_index=edge_index.to(torch.long),
        edge_type_multihot=edge_type_multihot,
        edge_label=edge_label,
        num_nodes=num_nodes,
        source_path=source_path,
        meta={"container": type(obj).__name__},
    )


# ========================================================================== #
# Validation
# ========================================================================== #
def _validate(adj_tensor, x, y_matrix, source_path: str) -> None:
    """Raise MalformedSampleError unless the three core tensors are coherent."""
    name = os.path.basename(source_path) or "<sample>"

    if adj_tensor.dim() != 3 or adj_tensor.shape[0] != adj_tensor.shape[1]:
        raise MalformedSampleError(
            f"{name}: adjacency tensor must be (V, V, E); got "
            f"{tuple(adj_tensor.shape)}."
        )
    V = adj_tensor.shape[0]

    if x.dim() != 2 or x.shape[0] != V:
        raise MalformedSampleError(
            f"{name}: feature matrix must be (V={V}, p); got {tuple(x.shape)}."
        )
    if x.shape[1] == V:
        raise MalformedSampleError(
            f"{name}: feature matrix is square (V, V) -- it was probably "
            f"mis-classified as features when it is really a second matrix."
        )

    if y_matrix.dim() != 2 or y_matrix.shape != (V, V):
        raise MalformedSampleError(
            f"{name}: target matrix must be (V={V}, V={V}); got "
            f"{tuple(y_matrix.shape)}."
        )

    y_vals = torch.unique(y_matrix)
    allowed = torch.tensor([0, 1], device=y_vals.device, dtype=y_vals.dtype)
    if not torch.isin(y_vals, allowed).all():
        raise MalformedSampleError(
            f"{name}: target matrix is not binary; distinct values = "
            f"{y_vals.tolist()[:10]}."
        )

    if torch.isnan(x).any() or torch.isnan(adj_tensor).any():
        raise MalformedSampleError(
            f"{name}: NaN present in features or adjacency."
        )

    if int(adj_tensor.sum().item()) == 0:
        raise MalformedSampleError(f"{name}: adjacency tensor is entirely zero.")


# ========================================================================== #
# Public entry point
# ========================================================================== #
def load_sample(path: str) -> AttackGraphSample:
    """
    Parse one ``.pt`` file into an :class:`AttackGraphSample`.

    Accepts a ``dict`` of tensors, a ``list``/``tuple`` of tensors, or a
    ``torch_geometric.data.Data`` object. Raises :class:`MalformedSampleError`
    (or :class:`FileNotFoundError`) with a specific message on anything it
    cannot make sense of.
    """
    raw = robust_torch_load(path)
    name = os.path.basename(path)

    # --- torch_geometric Data-like? (duck-typed to avoid a hard import) --- #
    if type(raw).__module__.startswith("torch_geometric"):
        return _from_pyg_data(raw, path)
    if (hasattr(raw, "edge_index") and hasattr(raw, "x")
            and not isinstance(raw, dict)):
        return _from_pyg_data(raw, path)

    # --- dict / list / tuple of tensors -------------------------------- #
    if isinstance(raw, dict):
        tensors = {str(k): v for k, v in raw.items() if _is_tensor(v)}
        if not tensors:
            raise MalformedSampleError(
                f"{name}: loaded a dict with no tensor values "
                f"(keys = {list(raw.keys())})."
            )
        container_desc = f"{name} (dict keys {list(tensors.keys())})"
    elif isinstance(raw, (list, tuple)):
        tensors = {f"[{i}]": v for i, v in enumerate(raw) if _is_tensor(v)}
        if not tensors:
            raise MalformedSampleError(
                f"{name}: loaded a {type(raw).__name__} with no tensor entries."
            )
        container_desc = (
            f"{name} ({type(raw).__name__} of {len(tensors)} tensors)"
        )
    elif _is_tensor(raw):
        raise MalformedSampleError(
            f"{name}: sample is a bare tensor of shape {tuple(raw.shape)} -- "
            f"expected a container with adjacency + features + target."
        )
    else:
        raise MalformedSampleError(
            f"{name}: unsupported top-level type {type(raw)!r}."
        )

    roles = classify_fields(tensors)
    adj_label = _pick_single(roles, "adjacency", container_desc)
    tgt_label = _pick_single(roles, "target", container_desc)
    feat_label = _pick_single(roles, "features", container_desc)

    adj_tensor = tensors[adj_label].to(torch.float32)
    y_matrix = tensors[tgt_label].to(torch.long)
    x = tensors[feat_label].to(torch.float32)

    _validate(adj_tensor, x, y_matrix, path)
    edge_index, edge_type_multihot, edge_label = build_edge_structures(
        adj_tensor, y_matrix
    )

    return AttackGraphSample(
        adj_tensor=adj_tensor,
        x=x,
        y_matrix=y_matrix,
        edge_index=edge_index,
        edge_type_multihot=edge_type_multihot,
        edge_label=edge_label,
        num_nodes=adj_tensor.shape[0],
        source_path=path,
        meta={
            "container": "dict" if isinstance(raw, dict) else type(raw).__name__,
            "field_map": {adj_label: "adjacency", tgt_label: "target",
                          feat_label: "features"},
            "num_edges": int(edge_index.shape[1]),
            "num_attack_edges": int(edge_label.sum().item()),
            "num_edge_types": int(adj_tensor.shape[2]),
        },
    )


# ========================================================================== #
# torch_geometric bridge for the model (PART 1)
# ========================================================================== #
def to_pyg_data(sample: AttackGraphSample):
    """
    Convert an :class:`AttackGraphSample` into a ``torch_geometric.data.Data``
    with per-edge supervision, ready for ``torch_geometric.loader.DataLoader``.

        data.x          : (V, F)  node features
        data.edge_index : (2, M)  connectivity
        data.edge_attr  : (M, E)  multi-hot edge types  (also an edge feature)
        data.edge_label : (M,)    1.0 if the edge is on the attack path
    """
    from torch_geometric.data import Data

    data = Data(
        x=sample.x,
        edge_index=sample.edge_index,
        edge_attr=sample.edge_type_multihot,
    )
    data.edge_label = sample.edge_label
    data.num_nodes = sample.num_nodes
    return data


# ========================================================================== #
# Dataset + split
# ========================================================================== #
class AttackGraphDataset(Dataset):
    """
    Lazy dataset over a directory of ``.pt`` attack-graph samples.

    Parameters
    ----------
    data_dir
        Directory containing ``*.pt`` files (default ``data/_data_``).
    as_pyg
        If True, ``__getitem__`` returns a ``torch_geometric.data.Data``
        (for the GNN). If False, it returns the raw :class:`AttackGraphSample`
        (for PART 3 / inspection).
    cache
        Keep parsed samples in memory after first access. Default False so a
        full pass does not pin GBs.
    limit
        Optional cap on the number of files (handy for smoke tests).
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        as_pyg: bool = True,
        cache: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self.data_dir = data_dir
        self.as_pyg = as_pyg
        self.cache = cache
        self._cache: Dict[int, object] = {}

        self.paths: List[str] = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(
                f"No .pt files under {data_dir!r}. Did _data_.zip get "
                f"extracted? See README 'Setup'."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self.cache and idx in self._cache:
            return self._cache[idx]
        sample = load_sample(self.paths[idx])
        out = to_pyg_data(sample) if self.as_pyg else sample
        if self.cache:
            self._cache[idx] = out
        return out


def split_dataset(
    dataset: AttackGraphDataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple["torch.utils.data.Subset", "torch.utils.data.Subset"]:
    """
    Deterministic train/test split by index (default 80/20).

    Uses a fixed-seed permutation so the same split is reproduced on every run
    -- important because the training log / checkpoints must line up across
    resumes.
    """
    from torch.utils.data import Subset

    n = len(dataset)
    n_train = int(round(n * train_fraction))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    return Subset(dataset, perm[:n_train]), Subset(dataset, perm[n_train:])


# ========================================================================== #
# Smoke test / manual inspection
# ========================================================================== #
def _demo() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="inspect the parsed loader output"
    )
    parser.add_argument("path", nargs="?", help="a specific .pt file")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    target = args.path
    if target is None:
        found = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
        if not found:
            raise SystemExit(f"no .pt files under {args.data_dir!r}")
        target = found[0]

    print(f"Loading {target}")
    s = load_sample(target)
    print(f"  container         : {s.meta['container']}")
    print(f"  field_map         : {s.meta.get('field_map')}")
    print(f"  num_nodes  (V)    : {s.num_nodes}")
    print(f"  features   x      : {tuple(s.x.shape)}  dtype={s.x.dtype}")
    print(f"  adj_tensor        : {tuple(s.adj_tensor.shape)}  "
          f"edge_types={s.adj_tensor.shape[2]}")
    print(f"  y_matrix          : {tuple(s.y_matrix.shape)}  "
          f"dtype={s.y_matrix.dtype}")
    print(f"  edge_index        : {tuple(s.edge_index.shape)}  "
          f"(M={s.edge_index.shape[1]} directed edges)")
    print(f"  edge_type_multihot: {tuple(s.edge_type_multihot.shape)}")
    print(f"  edge_label        : {tuple(s.edge_label.shape)}  "
          f"positives={int(s.edge_label.sum())} "
          f"({100 * s.edge_label.mean():.2f}% of edges)")

    # How does the attack-path target relate to the physical edges?
    dense = s.dense_binary_adjacency().bool()
    y_bool = s.y_matrix.bool()
    y_on_edges = (y_bool & dense).sum().item()
    y_total = y_bool.sum().item()
    print(f"\n  attack-path edges total          : {y_total}")
    print(f"  ...that coincide with a real edge : {y_on_edges}")
    print(f"  ...NOT backed by any adj edge     : {y_total - y_on_edges}")
    if y_total:
        covered = "a subset of physical edges" if y_on_edges == y_total \
            else "NOT fully covered by physical edges"
        print(f"  => attack path is {covered}")

    print("\nParsed OK.")


if __name__ == "__main__":
    _demo()
