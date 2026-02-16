from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import nn

from config.model_configs.IA_MPPN_configs import IA_MPPNConfig
from models.IA_MPPN.IA_Transformer import IA_Transformer
from models.IA_MPPN.mppn_IA_ViT import MPPN_IA_ViT


@dataclass
class AttentionWeights:
    """Container for attention weights and matrices."""
    feature_name: Optional[str] = None
    attn_weights: Optional[List] = None
    new_atts: Optional[List] = None
    cls_atts: Optional[List] = None


@dataclass
class CollectedAttentions:
    """Collected attention weights from all model components."""
    perspective_attentions: List[AttentionWeights] = field(default_factory=list)
    relation_attention: Optional[AttentionWeights] = None


class FlattenedTokenEmbedding(nn.Module):
    """Add perspective and patch embeddings to flattened tokens."""

    def __init__(self, P: int, N: int, D: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.persp_emb = nn.Embedding(P, D)
        self.patch_emb = nn.Embedding(N, D)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "persp_ids", torch.arange(P, dtype=torch.long).view(1, P, 1), persistent=False
        )
        self.register_buffer(
            "patch_ids", torch.arange(N, dtype=torch.long).view(1, 1, N), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add perspective and patch embeddings, then flatten."""
        B, P, N, D = x.shape
        x = (
            x
            + self.persp_emb(self.persp_ids[:, :P, :]) # type: ignore
            + self.patch_emb(self.patch_ids[:, :, :N]) # type: ignore
        )
        return self.dropout(x).reshape(B, P * N, D)


class IA_MPPN(nn.Module):
    """Multi-Perspective Vision Transformer with Relation Attention."""

    def __init__(self, config: IA_MPPNConfig, input_shape: Tuple[int, int, int], vis: bool = False) -> None:
        super(IA_MPPN, self).__init__()
        self.config = config
        self.input_shape = input_shape
        self.vis = vis
        self.ablation_mode = config.ablation_mode
        self.num_classes = 1  # regression task
        assert self.ablation_mode in [0, 1, 2, 3], "ablation_mode must be 0, 1, 2, or 3."

        H, W, C = input_shape
        assert H == W, "Currently only square inputs are supported"
        P = C  # number of perspectives
        ph, pw = self.config.perspective_model_config.patches["size"]
        grid_h = H // ph
        grid_w = W // pw
        self.num_patches = grid_h * grid_w

        self._build_models(P)
        self._build_heads(P)
        self._validate_config()

    def _build_models(self, P: int) -> None:
        """Build models based on ablation mode."""
        self.perspectives_model = MPPN_IA_ViT(
            config=self.config.perspective_model_config,
            input_shape=self.input_shape,
            num_patches=self.num_patches,
        )

        if self.ablation_mode == 0:
            print("Using full IA-MPPN model")
            self.flattener = FlattenedTokenEmbedding(
                P=P,
                N=self.num_patches,
                D=self.config.perspective_model_config.hidden_size,
                dropout=0.0,
            )
            self.relation_model = IA_Transformer(
                config=self.config.relation_model_config,
                input_shape=(P * self.num_patches, self.config.perspective_model_config.hidden_size),
            )
        elif self.ablation_mode == 1:
            print("Using only perspective models with averaging (ablation mode 1)")
        elif self.ablation_mode == 2:
            print("Using perspective and relation model without flattener (ablation mode 2)")
            self.relation_model = IA_Transformer(
                config=self.config.relation_model_config,
                input_shape=(P * self.num_patches, self.config.perspective_model_config.hidden_size),
            )

    def _build_heads(self, P: int) -> None:
        """Build output heads."""
        D = self.config.relation_model_config.hidden_size
        self.logits_head = nn.Linear(D, 1)
        L = P * self.num_patches
        self.inter_logits_head = nn.Linear(L, 1)

    def _validate_config(self) -> None:
        """Validate configuration consistency."""
        assert (
            self.config.relation_model_config.hidden_size
            == self.config.perspective_model_config.hidden_size
        ), "Hidden sizes of perspective and relation model must match."

    def forward(self, x: torch.Tensor, evaluation: bool = False) -> Tuple:
        """Forward pass processing perspectives and relations."""
        B, C, _, _ = x.shape
        perspective_features: List[torch.Tensor] = []
        inter_feat_list: List[torch.Tensor] = []
        cls_tokens: List[torch.Tensor] = []
        ca = CollectedAttentions() if evaluation else None

        for c in range(C):
            inter_feat, p_attn_weights, p_new_atts, forward_x = self._forward_perspective(x, c)
            patch_tokens = forward_x[:, 1:]  # Exclude CLS token
            cls_token = forward_x[:, 0]  # CLS token
            perspective_features.append(patch_tokens)
            inter_feat_list.append(inter_feat)
            cls_tokens.append(cls_token)
            pers_w = AttentionWeights(
                feature_name=f"perspective_{c}",
                attn_weights=p_attn_weights,
                new_atts=p_new_atts,
                cls_atts=None,
            )
            if evaluation:
                ca.perspective_attentions.append(pers_w) # type: ignore

        patch_tokens = torch.stack(perspective_features, dim=1)  # [B,P,N,D]
        return self._forward_by_ablation_mode(B, C, patch_tokens, cls_tokens, inter_feat_list, ca, evaluation)

    def _forward_perspective(self, x: torch.Tensor, perspective_idx: int):
        """Process a single perspective through perspective model."""
        perspective = x[:, perspective_idx : perspective_idx + 1, :, :]
        return self.perspectives_model(perspective)

    def _forward_by_ablation_mode(
        self,
        B: int,
        C: int,
        patch_tokens: torch.Tensor,
        cls_tokens: List[torch.Tensor],
        inter_feat_list: List[torch.Tensor],
        ca: Optional[CollectedAttentions],
        evaluation: bool,
    ) -> Tuple: # type: ignore
        """Route forward pass based on ablation mode."""
        if self.ablation_mode == 0:
            return self._forward_ablation_0(patch_tokens, inter_feat_list, ca, evaluation)
        if self.ablation_mode == 1:
            return self._forward_ablation_1(cls_tokens, inter_feat_list, ca, evaluation)
        if self.ablation_mode == 2:
            return self._forward_ablation_2(B, C, patch_tokens, ca, evaluation)

    def _forward_ablation_0(
        self,
        patch_tokens: torch.Tensor,
        inter_feat_list: List[torch.Tensor],
        ca: Optional[CollectedAttentions],
        evaluation: bool,
    ) -> Tuple:
        """Full IA-MPPN model."""
        features = self.flattener(patch_tokens)
        cls_token, attn_weights, inter_feat, new_atts, _ = self.relation_model(features)
        logits = self.logits_head(cls_token)
        inter_logits = self.inter_logits_head(inter_feat)
        out = (logits, inter_logits, inter_feat_list)

        if evaluation:
            rel_w = AttentionWeights(
                feature_name="relation",
                attn_weights=attn_weights,
                new_atts=new_atts,
                cls_atts=None,
            )
            ca.relation_attention = rel_w # type: ignore
            out = out + (ca,)
        return out

    def _forward_ablation_1(
        self,
        cls_tokens: List[torch.Tensor],
        inter_feat_list: List[torch.Tensor],
        ca: Optional[CollectedAttentions],
        evaluation: bool,
    ) -> Tuple:
        """Perspective models with averaging."""
        cls_tokens_stacked = torch.stack(cls_tokens, dim=1)
        pooled = cls_tokens_stacked.mean(dim=1)
        logits = self.logits_head(pooled)

        inter_feat = torch.cat(inter_feat_list, dim=1)
        inter_logits = self.inter_logits_head(inter_feat)

        out = (logits, inter_logits, None)
        if evaluation:
            out = out + (ca,)
        return out

    def _forward_ablation_2(
        self,
        B: int,
        C: int,
        patch_tokens: torch.Tensor,
        ca: Optional[CollectedAttentions],
        evaluation: bool,
    ) -> Tuple:
        """Perspective and relation model without flattener."""
        features = patch_tokens.reshape(B, C * self.num_patches, -1)
        cls_token, attn_weights, inter_feat, new_atts, _ = self.relation_model(features)
        logits = self.logits_head(cls_token)
        inter_logits = self.inter_logits_head(inter_feat)
        out = (logits, inter_logits, None)

        if evaluation:
            rel_w = AttentionWeights(
                feature_name="relation",
                attn_weights=attn_weights,
                new_atts=new_atts,
                cls_atts=None,
            )
            ca.relation_attention = rel_w # type: ignore
            out = out + (ca,)
        return out