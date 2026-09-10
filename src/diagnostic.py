"""
diagnostic.py -- Dataset schema discovery for the AD attack-graph .pt samples.

WHY THIS EXISTS
---------------
We were handed 1033 pre-processed ``.pt`` files but *not* a guaranteed schema.
Before writing a real loader we must find out, empirically, what one sample
actually contains: is it a ``dict``? a ``tuple``/``list``? a raw
``torch.Tensor``? a ``torch_geometric.data.Data`` object? What are the key
names, tensor shapes and dtypes?

This script answers exactly that and nothing else. It does not transform, train,
or hardcode key names. Run it, read the output, THEN design the loader.

USAGE
-----
    python src/diagnostic.py                                # first sample found
    python src/diagnostic.py data/_data_/graph_00hliAZI.pt  # a specific file
    python src/diagnostic.py --n 3                          # first 3 samples

Output is echoed to the console and saved to ``logs/diagnostic_report.txt`` so
it can be pasted into the project write-up.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

try:
    import torch
except Exception as exc:  # environment problem, not a logic bug
    sys.exit(f"[diagnostic] Could not import torch: {exc!r}")

DATA_DIR = os.path.join("data", "_data_")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def robust_torch_load(path: str):
    """
    Load a ``.pt`` file, coping with the PyTorch 2.6+ ``weights_only=True``
    default which refuses to unpickle non-tensor container types
    (torch_geometric ``Data``, numpy arrays, ...).

    Strategy: try the safe path first; if it raises, retry permissively because
    this is our own trusted local dataset file.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_exc:
        print(
            f"  [note] weights_only=True failed "
            f"({type(safe_exc).__name__}: {safe_exc}); "
            f"retrying weights_only=False on trusted local file."
        )
        return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# Shape-based semantic guess (the same heuristic the loader will use)
# --------------------------------------------------------------------------- #
def classify_by_shape(t: "torch.Tensor") -> str:
    """
    Guess what a tensor represents purely from its shape.

    * 3-D  (V, V, d) or (d, V, V) with a square face -> adjacency tensor
      (d edge types stacked).
    * 2-D  square (V, V)                             -> attack-path target matrix.
    * 2-D  rectangular (V, p)                        -> node feature matrix.
    * 1-D                                            -> per-node vector / label.
    Anything else -> "unknown".
    """
    shape = tuple(t.shape)
    if t.dim() == 3:
        a, b, c = shape
        if a == b:
            return f"adjacency tensor  (V={a}, edge_types={c})"
        if b == c:
            return f"adjacency tensor  (edge_types={a}, V={b})"
        return "3-D tensor (non-square faces) -- inspect manually"
    if t.dim() == 2:
        r, c = shape
        if r == c:
            return f"square matrix (V={r}) -- likely attack-path target/label"
        return f"rectangular matrix -- likely node features (V={r}, p={c})"
    if t.dim() == 1:
        return f"1-D vector (len={shape[0]}) -- per-node attribute or label"
    if t.dim() == 0:
        return "scalar"
    return "unknown"


def describe_tensor(t: "torch.Tensor", name: str = "") -> str:
    """One dense line: shape, dtype, device, value range, #unique, shape-guess."""
    label = f"{name}: " if name else ""
    shape = tuple(t.shape)

    try:
        if t.numel() == 0:
            stats = "empty"
        elif t.is_floating_point():
            stats = (f"min={t.min().item():.4g} max={t.max().item():.4g} "
                     f"mean={t.float().mean().item():.4g}")
        else:
            stats = f"min={t.min().item()} max={t.max().item()}"
    except Exception as exc:
        stats = f"<stats unavailable: {exc}>"

    try:
        uq = torch.unique(t)
        n_unique = int(uq.numel())
        uniq = f" unique={n_unique}"
        if n_unique <= 6:
            vals = ", ".join(
                f"{v:.4g}" if t.is_floating_point() else str(int(v))
                for v in uq.tolist()
            )
            uniq += f" [{vals}]"
    except Exception:
        uniq = ""

    guess = classify_by_shape(t)
    guess_str = f"   --> {guess}" if guess else ""
    return (f"{label}Tensor shape={shape} dtype={t.dtype} device={t.device} "
            f"{stats}{uniq}{guess_str}")


# --------------------------------------------------------------------------- #
# Recursive container walk
# --------------------------------------------------------------------------- #
def describe_any(obj, indent: int = 0, name: str = "") -> None:
    """Print a structured description of an arbitrary loaded object."""
    pad = "  " * indent
    label = f"{name} = " if name else ""

    # torch_geometric Data / HeteroData -- detect by duck typing so we don't
    # need torch_geometric imported unless it is actually used.
    cls = type(obj)
    cls_path = f"{cls.__module__}.{cls.__qualname__}"

    if isinstance(obj, torch.Tensor):
        print(pad + label + describe_tensor(obj))
        return

    if isinstance(obj, dict):
        print(f"{pad}{label}dict  ({len(obj)} keys)  [{cls_path}]")
        for k, v in obj.items():
            describe_any(v, indent + 1, name=repr(k))
        return

    if isinstance(obj, (list, tuple)):
        print(f"{pad}{label}{cls.__name__}  (len {len(obj)})")
        for i, v in enumerate(obj):
            describe_any(v, indent + 1, name=f"[{i}]")
        return

    if "torch_geometric" in cls.__module__:
        print(f"{pad}{label}{cls_path}")
        # Common torch_geometric attributes.
        for attr in ("x", "edge_index", "edge_attr", "y", "pos", "adj", "adj_t",
                     "num_nodes", "num_edges"):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    print("  " * (indent + 1) + describe_tensor(val, name=attr))
                else:
                    print("  " * (indent + 1) + f"{attr}: {val!r}")
        # Anything else the object chooses to expose.
        try:
            keys = list(obj.keys()) if hasattr(obj, "keys") else []
            extra = [k for k in keys if k not in
                     {"x", "edge_index", "edge_attr", "y", "pos"}]
            if extra:
                print("  " * (indent + 1) + f"other keys: {extra}")
        except Exception:
            pass
        return

    # Fallback: primitives, numpy arrays, etc.
    rep = repr(obj)
    if len(rep) > 200:
        rep = rep[:200] + "..."
    print(f"{pad}{label}{cls_path}  ->  {rep}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def inspect_file(path: str) -> None:
    print("=" * 78)
    print(f"FILE: {path}")
    print(f"size: {os.path.getsize(path) / 1e6:.2f} MB")
    print("-" * 78)

    obj = robust_torch_load(path)
    print(f"top-level type: {type(obj).__module__}.{type(obj).__name__}")
    print("-" * 78)
    describe_any(obj)
    print("=" * 78)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*",
                        help="specific .pt files to inspect (default: scan "
                             f"{DATA_DIR})")
    parser.add_argument("--n", type=int, default=1,
                        help="how many samples to inspect when scanning")
    args = parser.parse_args()

    if args.paths:
        targets = args.paths
    else:
        found = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
        if not found:
            sys.exit(f"[diagnostic] No .pt files under {DATA_DIR!r}. "
                     f"Did _data_.zip get extracted?")
        targets = found[: args.n]

    # Tee output to a report file as well as the console.
    os.makedirs("logs", exist_ok=True)
    report_path = os.path.join("logs", "diagnostic_report.txt")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(report_path, "w", encoding="utf-8") as fh:
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, fh)
        try:
            print(f"torch {torch.__version__}  |  scanning {len(targets)} "
                  f"sample(s)\n")
            for p in targets:
                if not os.path.isfile(p):
                    print(f"[skip] not a file: {p}")
                    continue
                inspect_file(p)
        finally:
            sys.stdout = real_stdout

    print(f"[diagnostic] report written to {report_path}")


if __name__ == "__main__":
    main()
