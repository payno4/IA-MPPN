"""Multi-Perspective ViT with attention-based patch scoring."""

from typing import Tuple

import torch
from torch import nn

from config.model_configs.IA_MPPN_configs import PerspectiveConfigs
from models.IA_ViT.IA_ViT import Transformer

class MPPN_IA_ViT(nn.Module):
    """Multi-Perspective Vision Transformer with attention-based interpretability."""

    def __init__(
        self,
        config: PerspectiveConfigs,
        input_shape: Tuple[int, int, int],
        num_patches: int,
        vis: bool = True,
    ) -> None:
        """Initialize the model.
        
        Args:
            config: Perspective model configuration.
            input_shape: (height, width, channels) of input.
            num_patches: Number of patches (unused but kept for interface compatibility).
            vis: Whether to collect visualization/attention weights.
        """
        super().__init__()
        self.config = config
        self.num_patches = num_patches
        
        # Adjust input shape to single channel per perspective
        H, W, _ = input_shape
        input_shape = (H, W, 1)

        # Transformer backbone
        self.transformer = Transformer(config, vis, input_shape=input_shape)  # type: ignore
        feature_dim = config.hidden_size

        # Attention mechanism for patch scoring
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc1 = nn.Linear(feature_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize linear layer weights."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
    
    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor [B, N, D].
            labels: Optional labels (unused, kept for interface compatibility).
            
        Returns:
            Tuple of:
                - Interpretability features [B, N].
                - Raw attention weights from transformer layers.
                - Patch attention probabilities [B, N, N].
                - Encoded output [B, N+1, D] (includes CLS token).
        """
        # Transformer encoding
        x, raw_attn = self.transformer(x)  # [B, N+1, D], raw attention weights

        # Self-attention over patches (excluding CLS token)
        h = x[:, 1:]  # [B, N, D]
        Q = self.query(h)
        K = self.key(h)
        V = self.value(h)

        # Attention scoring
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_probs = self.softmax(attn_scores)  # [B, N, N]

        # Context and interpretability features
        context_layer = torch.matmul(attn_probs, V)
        inter_feat = self.fc1(context_layer).squeeze(dim=2)  # [B, N]

        return inter_feat, raw_attn, attn_probs, x