"""Auxiliary VLA heads trained on top of the pi05 prefix encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


@dataclass(frozen=True)
class AuxiliaryHeadConfig:
    hidden_dim: int = 512
    dropout: float = 0.1
    num_failure_reasons: int = 4
    num_recovery_plans: int = 4


class TactileVLAAuxiliaryHeads(nn.Module):
    """Frozen or finetuned pi05 prefix encoder plus three small classification heads."""

    def __init__(self, backbone: nn.Module, config: AuxiliaryHeadConfig) -> None:
        super().__init__()
        self.backbone = backbone
        width = backbone.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.pool_norm = nn.LayerNorm(width)
        self.shared = nn.Sequential(
            nn.Linear(width, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.need_recovery_head = nn.Linear(config.hidden_dim, 2)
        self.failure_reason_head = nn.Linear(config.hidden_dim, config.num_failure_reasons)
        self.recovery_plan_head = nn.Linear(config.hidden_dim, config.num_recovery_plans)

    def encode_prefix(self, observation, *, train: bool) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, _ = self.backbone._preprocess_observation(observation, train=train)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.backbone.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        target_dtype = self.backbone.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if target_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        att_2d_masks_4d = self.backbone._prepare_attention_masks_4d(att_2d_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_out, _), _ = self.backbone.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        prefix_out = prefix_out.to(dtype=torch.float32)
        mask = prefix_pad_masks.unsqueeze(-1).to(dtype=prefix_out.dtype)
        return (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, observation, *, train_backbone: bool = False) -> dict[str, torch.Tensor]:
        if train_backbone:
            features = self.encode_prefix(observation, train=True)
        else:
            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                features = self.encode_prefix(observation, train=False)
            if was_training:
                self.backbone.train()
        return self.forward_from_features(features)

    def forward_from_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.shared(self.pool_norm(features))
        return {
            "need_recovery": self.need_recovery_head(pooled),
            "failure_reason": self.failure_reason_head(pooled),
            "recovery_plan": self.recovery_plan_head(pooled),
        }
