from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from config.model_configs.IA_MPPN_configs import RelationsConfig


class EncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention and feedforward."""

    def __init__(self, config: RelationsConfig, vis: bool) -> None:
        super().__init__()
        self.config = config
        self.vis = vis
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout_rate,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_dim, config.hidden_size),
            nn.Dropout(config.dropout_rate),
        )

    def _init_weights(self) -> None:
        """Initialize weights of linear layers."""
        nn.init.xavier_uniform_(self.ffn[0].weight) # type: ignore
        nn.init.xavier_uniform_(self.ffn[3].weight) # type: ignore
        nn.init.normal_(self.ffn[0].bias, std=1e-6) # type: ignore
        nn.init.normal_(self.ffn[3].bias, std=1e-6) # type: ignore

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention and feedforward with residual connections.
        
        Args:
            x: Input tensor [B, N, D].
            
        Returns:
            Tuple of:
                - Output tensor [B, N, D].
                - Attention weights [B, num_heads, N, N].
        """
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, config: RelationsConfig, vis: bool) -> None:
        super().__init__()
        self.config = config
        self.vis = vis
        self.encoder_norm = nn.LayerNorm(config.hidden_size)
        self.layer = nn.ModuleList(
            [EncoderLayer(config, vis) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Process input through all encoder layers.
        
        Args:
            x: Input tensor [B, N, D].
            
        Returns:
            Tuple of:
                - Output tensor [B, N, D].
                - List of attention weight tensors from each layer.
        """
        attention_weights: List[torch.Tensor] = []
        for layer_module in self.layer:
            x, weights = layer_module(x)
            if self.vis:
                attention_weights.append(weights)
        x = self.encoder_norm(x)
        return x, attention_weights



class Embeddings(nn.Module):
    """Token embeddings with positional encoding and CLS token."""

    def __init__(self, config: RelationsConfig, input_shape: Tuple[int, int]) -> None:
        """Initialize embeddings.
        
        Args:
            config: Configuration.
            input_shape: (num_sequences, feature_dim).
        """
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        self.token_embeddings = nn.Linear(input_shape[1], config.hidden_size)
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, input_shape[0] + 1, config.hidden_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed and add positional information.
        
        Args:
            x: Input tensor [B, N, feature_dim].
            
        Returns:
            Embedded tensor [B, N+1, hidden_size] (N+1 includes CLS token).
        """
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.token_embeddings(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embeddings
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    """Complete transformer with embeddings and encoder stack."""

    def __init__(self, config: RelationsConfig, input_shape: Tuple[int, int], vis: bool) -> None:
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        self.embeddings = Embeddings(config, input_shape)
        self.encoder = Encoder(config, vis)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Process input through embeddings and encoder.
        
        Args:
            x: Input tensor [B, N, feature_dim].
            
        Returns:
            Tuple of:
                - Encoded tensor [B, N+1, hidden_size].
                - List of attention weight tensors.
        """
        x = self.embeddings(x)
        x, attn_weights = self.encoder(x)
        return x, attn_weights


class IA_Transformer(nn.Module):
    """Interpretable transformer with attention-based feature scoring."""

    def __init__(self, config: RelationsConfig, input_shape: Tuple[int, int], vis: bool = True) -> None:
        super().__init__()
        self.config = config
        self.num_classes = 1
        self.transformer = Transformer(config, input_shape, vis)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute interpretable attention over sequence.
        
        Args:
            x: Input tensor [B, N, feature_dim].
            
        Returns:
            Tuple of:
                - CLS token [B, hidden_size].
                - List of raw attention weights from transformer layers.
                - Interpretability features [B, N].
                - Attention probabilities [B, N, N].
                - None (placeholder for future use).
        """
        x, raw_atts = self.transformer(x)
        cls_token = x[:, 0]  # [B, D]
        h = x[:, 1:]  # [B, N, D]

        Q = self.query(h)
        K = self.key(h)
        V = self.value(h)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_probs = self.softmax(attn_scores)  # [B, N, N]

        context_layer = torch.matmul(attn_probs, V)  # [B, N, D]
        inter_feat = self.fc1(context_layer).squeeze(dim=2)  # [B, N]

        return cls_token, raw_atts, inter_feat, attn_probs, None