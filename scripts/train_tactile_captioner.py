#!/usr/bin/env python3
"""Train the tactile captioner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.captioner.training import CaptionerTrainConfig
from tactile_vla.captioner.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "tactile_captioner_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tactile_captioner")
    parser.add_argument("--run-name", default="tcn_v1_balanced")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-balanced-train", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--frame-feature-dim", type=int, default=128)
    parser.add_argument("--temporal-hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CaptionerTrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        balanced_train=not args.no_balanced_train,
        normalize=not args.no_normalize,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        patience=args.patience,
        frame_feature_dim=args.frame_feature_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        dropout=args.dropout,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
    )
    summary = train(config)
    print(f"best_checkpoint={summary['best_checkpoint']}")
    print(f"best_val_macro_f1={summary['best_val_macro_f1']:.6f}")
    print(f"test_macro_f1={summary['test']['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
