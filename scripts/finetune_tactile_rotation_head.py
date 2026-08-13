#!/usr/bin/env python3
"""Fine-tune only the tactile Captioner's rotation head on robot demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.captioner.rotation_finetuning import RotationFineTuneConfig  # noqa: E402
from tactile_vla.captioner.rotation_finetuning import fine_tune_rotation_head  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--demo-data-dir", type=Path, default=Path("/data1/tac_data/raw_data"))
    parser.add_argument("--pure-dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/data1/outputs/tactile_captioner"))
    parser.add_argument("--run-name")
    parser.add_argument("--episode-start", type=int, default=41)
    parser.add_argument("--episode-end", type=int, default=120)
    parser.add_argument("--forced-test-episode", type=int, action="append", dest="forced_test_episodes")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RotationFineTuneConfig(
        base_checkpoint=args.base_checkpoint,
        demo_data_dir=args.demo_data_dir,
        pure_dataset_dir=args.pure_dataset_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        forced_test_episodes=tuple(args.forced_test_episodes or (75,)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
    )
    summary = fine_tune_rotation_head(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
