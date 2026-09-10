"""
train.py -- PART 1 training loop for the GraphSAGE attack-propensity model.

WHAT IT DOES (exactly what the brief asks)
------------------------------------------
* Hard CUDA precondition check first (``env_check.require_cuda``) -- the run
  aborts loudly if the RTX 4060 / CUDA is not usable, never falls back to CPU.
* Deterministic 80 / 20 train / test split of the 1033 samples.
* Trains with ``BCEWithLogitsLoss(pos_weight=...)`` -- the positive class
  (edges on the attack path) is ~0.17 % of edges, so the loss is re-weighted.
* Every epoch: computes loss, accuracy (@0.5) and ROC-AUC on BOTH splits and
  writes one row to ``logs/training_log.csv`` (for the report's training-curve
  chart) as well as printing to the console.
* Checkpoints to ``checkpoints/ckpt_epoch_XXXX.pt`` every 5 epochs (plus
  ``checkpoints/latest.pt``); on start-up it resumes from the newest checkpoint
  if one exists.

ROC-AUC is implemented by hand (:func:`roc_auc` -- the Mann-Whitney U form) so
we do not add scikit-learn just for one metric.

USAGE
-----
    python -m src.train                       # full run, resumes if possible
    python -m src.train --epochs 50 --batch-size 16
    python -m src.train --limit 64 --epochs 3 # quick smoke test
    python -m src.train --no-resume           # ignore existing checkpoints
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .env_check import require_cuda
from .data_loader import AttackGraphDataset, split_dataset
from .model import GraphSAGEEdgeClassifier, build_model_from_config

CHECKPOINT_DIR = "checkpoints"
LOG_CSV = os.path.join("logs", "training_log.csv")
CSV_COLUMNS = [
    "epoch", "train_loss", "train_acc", "train_auc",
    "test_loss", "test_acc", "test_auc", "lr", "seconds", "timestamp",
]


# ========================================================================== #
# Metrics
# ========================================================================== #
def roc_auc(y_true: torch.Tensor, y_score: torch.Tensor,
            neg_chunk: int = 4096) -> float:
    """
    ROC-AUC via the Mann-Whitney U statistic:

        AUC = P(score(random positive) > score(random negative))

    computed exactly (ties count as 0.5). Chunked over the negatives so the
    pairwise comparison never materialises a huge matrix.

    Returns ``float('nan')`` if either class is absent (AUC undefined).
    """
    y_true = y_true.detach().flatten()
    y_score = y_score.detach().flatten().float()
    pos = y_score[y_true > 0.5]
    neg = y_score[y_true <= 0.5]
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pos_col = pos.unsqueeze(1)  # (n_pos, 1)
    wins = 0.0
    for j in range(0, n_neg, neg_chunk):
        n = neg[j:j + neg_chunk].unsqueeze(0)          # (1, c)
        wins += (pos_col > n).sum().item()
        wins += 0.5 * (pos_col == n).sum().item()
    return wins / (n_pos * n_neg)


def accuracy_at(y_true: torch.Tensor, y_score: torch.Tensor,
                threshold: float = 0.5) -> float:
    """Plain 0/1 accuracy of ``score >= threshold`` vs ``y_true``."""
    pred = (y_score >= threshold).float()
    return (pred == y_true).float().mean().item()


# ========================================================================== #
# One epoch
# ========================================================================== #
def train_one_epoch(model, loader, optimizer, loss_fn, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_edges = 0
    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        target = batch.edge_label.float()
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()

        bs = target.numel()
        total_loss += loss.item() * bs
        n_edges += bs
        all_scores.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(target.detach().cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    return {
        "loss": total_loss / max(n_edges, 1),
        "acc": accuracy_at(labels, scores),
        "auc": roc_auc(labels, scores),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_edges = 0
    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        target = batch.edge_label.float()
        loss = loss_fn(logits, target)

        bs = target.numel()
        total_loss += loss.item() * bs
        n_edges += bs
        all_scores.append(torch.sigmoid(logits).cpu())
        all_labels.append(target.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    return {
        "loss": total_loss / max(n_edges, 1),
        "acc": accuracy_at(labels, scores),
        "auc": roc_auc(labels, scores),
    }


# ========================================================================== #
# Checkpointing
# ========================================================================== #
_CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pt$")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path of the highest-epoch ``ckpt_epoch_XXXX.pt`` or None."""
    best_epoch, best_path = -1, None
    for path in glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")):
        m = _CKPT_RE.search(os.path.basename(path))
        if m and int(m.group(1)) > best_epoch:
            best_epoch, best_path = int(m.group(1)), path
    return best_path


def save_checkpoint(path: str, *, epoch: int, model, optimizer,
                    pos_weight: float, best_test_auc: float, args: dict) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "model_config": model.config(),
        "pos_weight": pos_weight,
        "best_test_auc": best_test_auc,
        "args": args,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def load_checkpoint(path: str, device) -> dict:
    # weights_only=False: this is our own file and it stores RNG state / config
    # dicts that the safe loader rejects.
    return torch.load(path, map_location=device, weights_only=False)


def _restore_rng_state(ckpt: dict) -> None:
    """
    Best-effort restore of CPU + CUDA RNG state from a checkpoint.

    ``set_rng_state`` requires a CPU ``ByteTensor``; if the checkpoint was
    loaded with ``map_location='cuda'`` the state tensors come back on the GPU
    and/or with the wrong dtype, so coerce them and swallow any failure.
    """
    cpu_rng = ckpt.get("torch_rng_state")
    if cpu_rng is not None:
        try:
            torch.set_rng_state(cpu_rng.detach().to("cpu", torch.uint8))
        except Exception as exc:  # pragma: no cover - reproducibility only
            print(f"        [warn] could not restore CPU RNG state: {exc}")

    cuda_rng = ckpt.get("cuda_rng_state")
    if cuda_rng is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state(cuda_rng.detach().to("cpu", torch.uint8))
        except Exception as exc:  # pragma: no cover - reproducibility only
            print(f"        [warn] could not restore CUDA RNG state: {exc}")


# ========================================================================== #
# CSV logging
# ========================================================================== #
def append_csv_row(csv_path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ========================================================================== #
# pos_weight estimation
# ========================================================================== #
def estimate_pos_weight(loader, cap: float = 1000.0) -> float:
    """
    ``pos_weight`` for BCEWithLogitsLoss = (#negative / #positive) over the
    training edges, so the rare positive class is up-weighted to parity.
    Capped so a split with very few positives cannot produce a runaway weight.
    """
    pos = neg = 0
    for batch in loader:
        y = batch.edge_label
        p = int((y > 0.5).sum())
        pos += p
        neg += y.numel() - p
    if pos == 0:
        return cap
    return min(neg / pos, cap)


# ========================================================================== #
# Main
# ========================================================================== #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", default=os.path.join("data", "_data_"))
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N samples (smoke test)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--pos-weight", type=float, default=None,
                   help="override the auto-estimated BCE pos_weight")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    p.add_argument("--log-csv", default=LOG_CSV)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--no-resume", action="store_true",
                   help="start fresh even if a checkpoint exists")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # (1) HARD CUDA CHECK -- before anything touches a device.
    device = require_cuda(verbose=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from torch_geometric.loader import DataLoader as GeoDataLoader

    # (2) Data + deterministic 80/20 split.
    dataset = AttackGraphDataset(args.data_dir, as_pyg=True, limit=args.limit)
    train_set, test_set = split_dataset(dataset, train_fraction=0.8,
                                        seed=args.seed)
    print(f"\nsamples: {len(dataset)}  ->  train {len(train_set)} | "
          f"test {len(test_set)}")

    train_loader = GeoDataLoader(train_set, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
    test_loader = GeoDataLoader(test_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    # (3) Model / optimiser / loss.
    model = GraphSAGEEdgeClassifier(
        in_channels=19, edge_dim=16,
        hidden_channels=args.hidden, num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    pos_weight = (args.pos_weight if args.pos_weight is not None
                  else estimate_pos_weight(train_loader))
    print(f"BCE pos_weight: {pos_weight:.1f}")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )

    # (4) Resume?
    start_epoch = 1
    best_test_auc = float("-inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest = (None if args.no_resume
              else find_latest_checkpoint(args.checkpoint_dir))
    if latest:
        ckpt = load_checkpoint(latest, device)
        # Rebuild in case architecture args differ from the checkpoint's.
        model = build_model_from_config(ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_test_auc = float(ckpt.get("best_test_auc", best_test_auc))
        # Restoring RNG state makes a resumed run bit-reproducible, but it is a
        # convenience, not a correctness requirement -- never let a quirk here
        # (e.g. the state tensor coming back on the GPU, or a torch version
        # change) abort the resume.
        _restore_rng_state(ckpt)
        print(f"resumed from {latest} -> continuing at epoch {start_epoch}")
    else:
        print("no checkpoint found -- training from scratch")

    if start_epoch > args.epochs:
        print(f"nothing to do: checkpoint epoch {start_epoch - 1} >= "
              f"--epochs {args.epochs}")
        return

    # (5) Epoch loop.
    print("\n" + "=" * 78)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        te = evaluate(model, test_loader, loss_fn, device)
        secs = time.time() - t0

        if te["auc"] == te["auc"]:  # not NaN
            best_test_auc = max(best_test_auc, te["auc"])

        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} "
              f"auc {tr['auc']:.4f} | "
              f"test loss {te['loss']:.4f} acc {te['acc']:.4f} "
              f"auc {te['auc']:.4f} | {secs:5.1f}s")

        append_csv_row(args.log_csv, {
            "epoch": epoch,
            "train_loss": round(tr["loss"], 6),
            "train_acc": round(tr["acc"], 6),
            "train_auc": round(tr["auc"], 6),
            "test_loss": round(te["loss"], 6),
            "test_acc": round(te["acc"], 6),
            "test_auc": round(te["auc"], 6),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(secs, 2),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        })

        # (6) Periodic + latest checkpoint.
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir,
                                     f"ckpt_epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch=epoch, model=model,
                            optimizer=optimizer, pos_weight=pos_weight,
                            best_test_auc=best_test_auc, args=vars(args))
            save_checkpoint(os.path.join(args.checkpoint_dir, "latest.pt"),
                            epoch=epoch, model=model, optimizer=optimizer,
                            pos_weight=pos_weight, best_test_auc=best_test_auc,
                            args=vars(args))
            print(f"        checkpoint -> {ckpt_path}")

    print("=" * 78)
    print(f"done. best test ROC-AUC = {best_test_auc:.4f}")
    print(f"training log: {args.log_csv}")


if __name__ == "__main__":
    main()
