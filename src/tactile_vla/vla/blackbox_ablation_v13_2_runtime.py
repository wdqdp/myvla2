"""Same model/normalization/LoRA loop as V13.1; V13.2 diagnostic metrics only."""
from tactile_vla.vla.blackbox_ablation_v13_1_runtime import evaluate as evaluate_common
from tactile_vla.vla.blackbox_ablation_v13_2_data import metrics, prediction_record


def evaluate(state, loader, grammar):
    return evaluate_common(state, loader, grammar, record_fn=prediction_record, metrics_fn=metrics)
