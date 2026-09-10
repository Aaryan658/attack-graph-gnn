"""
plot_training.py -- render the training-curve figure for the report from
``logs/training_log.csv``.

Produces a 2-panel PNG:
    left  : train/test loss vs epoch
    right : train/test ROC-AUC (and accuracy) vs epoch

USAGE
-----
    python -m src.plot_training
    python -m src.plot_training --csv logs/training_log.csv --out logs/training_curve.png
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: just write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join("logs", "training_log.csv"))
    ap.add_argument("--out", default=os.path.join("logs", "training_curve.png"))
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(
            f"no training log at {args.csv}; run `python -m src.train` first."
        )

    df = pd.read_csv(args.csv).drop_duplicates("epoch", keep="last")
    df = df.sort_values("epoch")
    if df.empty:
        raise SystemExit(f"{args.csv} has no rows.")

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(df["epoch"], df["train_loss"], "-o", ms=3, label="train")
    ax_loss.plot(df["epoch"], df["test_loss"], "-o", ms=3, label="test")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("BCE loss (pos-weighted)")
    ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_auc.plot(df["epoch"], df["train_auc"], "-o", ms=3, label="train ROC-AUC")
    ax_auc.plot(df["epoch"], df["test_auc"], "-o", ms=3, label="test ROC-AUC")
    ax_auc.plot(df["epoch"], df["test_acc"], "--", alpha=0.6,
                label="test acc@0.5")
    ax_auc.set_xlabel("epoch")
    ax_auc.set_ylabel("score")
    ax_auc.set_ylim(0, 1.02)
    ax_auc.set_title("Ranking quality")
    ax_auc.grid(alpha=0.3)
    ax_auc.legend()

    best = df.loc[df["test_auc"].idxmax()]
    fig.suptitle(
        f"attack-propensity GraphSAGE -- {len(df)} epochs -- "
        f"best test ROC-AUC {best['test_auc']:.3f} @ epoch {int(best['epoch'])}"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
