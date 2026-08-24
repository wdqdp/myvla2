#!/usr/bin/env python3
"""Aggregate paired correct/wrong phase-prompt losses and enforce V5 gates.

The input JSONL is emitted by an OpenPI checkpoint evaluator using the same
validation observation, 60x7 history, H30 target and folded noise for all
three prompts.  Each row must contain ``true_phase``, ``phase_losses`` and
``v4_loss``; ``chunk_phase_pure`` separates the primary protocol from chunks
whose unchanged V4 H30 target crosses a phase boundary.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tactile_vla.vla.prompts import ACTION_PHASES


def summarize_counterfactual_losses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Counterfactual loss stream is empty")
    phase_values: dict[str, list[dict[str, float]]] = defaultdict(list)
    crossing_count = 0
    for row_number, row in enumerate(rows):
        phase = str(row.get("true_phase", ""))
        if phase not in ACTION_PHASES:
            raise ValueError(f"row {row_number} has invalid true_phase={phase!r}")
        losses = row.get("phase_losses")
        if not isinstance(losses, dict) or set(losses) != set(ACTION_PHASES):
            raise ValueError(f"row {row_number} must contain exactly three phase_losses")
        numeric = {name: float(losses[name]) for name in ACTION_PHASES}
        v4_loss = float(row.get("v4_loss", math.nan))
        if not all(math.isfinite(value) and value >= 0 for value in [*numeric.values(), v4_loss]):
            raise ValueError(f"row {row_number} contains an invalid action loss")
        if not isinstance(row.get("chunk_phase_pure"), bool):
            raise ValueError(f"row {row_number} lacks boolean chunk_phase_pure")
        if not row["chunk_phase_pure"]:
            crossing_count += 1
            continue
        correct = numeric[phase]
        wrong = min(value for name, value in numeric.items() if name != phase)
        phase_values[phase].append(
            {
                "correct": correct,
                "wrong_best": wrong,
                "margin": wrong - correct,
                "win": float(correct < wrong),
                "v4": v4_loss,
            }
        )

    phase_summary: dict[str, Any] = {}
    correct_all: list[float] = []
    v4_all: list[float] = []
    gates: list[bool] = []
    for phase in ACTION_PHASES:
        values = phase_values.get(phase, [])
        if not values:
            raise ValueError(f"Primary pure-chunk evaluation has no {phase} support")
        average_margin = sum(value["margin"] for value in values) / len(values)
        win_rate = sum(value["win"] for value in values) / len(values)
        phase_summary[phase] = {
            "count": len(values),
            "mean_correct_loss": sum(value["correct"] for value in values) / len(values),
            "mean_best_wrong_loss": sum(value["wrong_best"] for value in values) / len(values),
            "mean_prompt_margin": average_margin,
            "correct_prompt_win_rate": win_rate,
            "margin_gate_passed": average_margin > 0.0,
            "win_rate_gate_passed": win_rate >= 0.60,
        }
        gates.extend((average_margin > 0.0, win_rate >= 0.60))
        correct_all.extend(value["correct"] for value in values)
        v4_all.extend(value["v4"] for value in values)
    correct_mean = sum(correct_all) / len(correct_all)
    v4_mean = sum(v4_all) / len(v4_all)
    degradation = correct_mean / v4_mean - 1.0 if v4_mean > 0 else math.inf
    degradation_gate = degradation <= 0.10
    gates.append(degradation_gate)
    return {
        "schema_version": "tactile_vla_v5_phase_prompt_counterfactual_eval_v1",
        "primary_chunk_phase_pure_count": len(correct_all),
        "excluded_cross_phase_chunk_count": crossing_count,
        "phases": phase_summary,
        "overall": {
            "correct_prompt_action_loss": correct_mean,
            "v4_action_loss": v4_mean,
            "action_loss_degradation": degradation,
            "action_loss_degradation_limit": 0.10,
            "degradation_gate_passed": degradation_gate,
        },
        "all_acceptance_gates_passed": all(gates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.loss_jsonl.read_text().splitlines() if line.strip()]
    summary = summarize_counterfactual_losses(rows)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    if args.require_pass and not summary["all_acceptance_gates_passed"]:
        raise SystemExit("V5 phase-prompt offline acceptance gates failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
