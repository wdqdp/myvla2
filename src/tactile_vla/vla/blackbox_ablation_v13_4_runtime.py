"""Three-group token audit; qpos masking is opt-in here, never a main-method default."""
import numpy as np

from tactile_vla.vla.blackbox_ablation_v13_2_runtime import evaluate  # noqa: F401
from tactile_vla.vla.blackbox_ablation_v13_3_runtime import make_loader  # noqa: F401
from tactile_vla.vla.blackbox_ablation_v13_4_data import GROUPS, QPOS_PROTOCOL


def validate_tokens(rows, *, tokenizer, grammar, norm, max_len):
    del norm
    state = np.zeros(7, dtype=np.float32)
    report = {g: {'count': 0, 'max_tokens': 0, 'limit': max_len, 'qpos_input_protocol': QPOS_PROTOCOL} for g in GROUPS}
    for row in rows:
        target = grammar.sequence_for_text(row['target_recovery_plan'])
        for group in GROUPS:
            _, mask, _, _, _ = tokenizer.tokenize_structured_response(row['prompts'][group], state, target, max_len=max_len)
            report[group]['count'] += 1
            report[group]['max_tokens'] = max(report[group]['max_tokens'], int(mask.sum()))
    return report
