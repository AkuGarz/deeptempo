"""PatchTST implementation adapted for fine-tuning.

Based on the paper "A Time Series is Worth 64 Words" (Nie et al., 2023).
This is a simplified reimplementation — not the official code.

Known issues:
- Channel independence is assumed (each variable treated separately).
- The pretrained weight loading path is hardcoded. Needs a proper hub integration.
- Positional encoding is fixed sinusoidal, not learned.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from deeptempo.models.registry import register_model


class PatchEmbedding(nn.Module):
    """Break the input sequence into patches and project them.

    Args:
        seq_len: Length of input sequence.
        patch_len: Length of each patch.
        stride: Stride between patches. Default = patch_len (non-overlapping).
        d_model: Embedding dimension.
    """

    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Linear projection for each patch
        self.projection = nn.Linear(patch_len, d_model)

        # TODO: make positional encoding learnable (currently fixed sinusoidal)
        self.positional_encoding = self._sinusoidal_encoding(self.n_patches, d_model)

    def _sinusoidal_encoding(self, n_positions: int, d_model: int) -> nn.Parameter:
        """Standard sinusoidal position encoding."""
        pe = torch.zeros(n_positions, d_model)
        position = torch.arange(0, n_positions).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) — univariate for now"""
        # Unfold into patches: (batch, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Project patches
        x = self.projection(x)  # (batch, n_patches, d_model)

        # Add positional encoding
        x = x + self.positional_encoding[: x.size(1), :]

        return x


class PatchTSTEncoder(nn.Module):
    """Transformer encoder for PatchTST.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm is more stable for fine-tuning
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_patches, d_model)"""
        x = self.encoder(x)
        x = self.norm(x)
        return x


class PatchTST(nn.Module):
    """PatchTST for univariate time-series forecasting.

    Args:
        seq_len: Input sequence length.
        pred_len: Forecast horizon.
        patch_len: Patch length.
        stride: Stride between patches.
        d_model: Hidden dimension.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.patch_embed = PatchEmbedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
        )

        self.encoder = PatchTSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Flatten & project to prediction length
        self.head = nn.Linear(d_model * self.patch_embed.n_patches, pred_len)

        # TODO: add dropout before head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len) univariate input.

        Returns:
            (batch, pred_len) forecast.
        """
        # Patch + embed
        x = self.patch_embed(x)  # (batch, n_patches, d_model)

        # Encode
        x = self.encoder(x)  # (batch, n_patches, d_model)

        # Flatten and predict
        x = rearrange(x, "b n d -> b (n d)")
        x = self.head(x)  # (batch, pred_len)

        return x


@register_model("patchtst_tiny")
def patchtst_tiny(pretrained: bool = False, **kwargs) -> PatchTST:
    """Tiny PatchTST variant (~200k params). Good for quick experiments."""
    model = PatchTST(
        d_model=64,
        n_heads=2,
        n_layers=2,
        **kwargs,
    )

    if pretrained:
        # TODO: download from HuggingFace Hub instead of hardcoded path
        # TODO: actually train and upload pretrained weights somewhere
        raise NotImplementedError(
            "Pretrained weights not yet available. Training from scratch for now."
        )

    return model


@register_model("patchtst_base")
def patchtst_base(pretrained: bool = False, **kwargs) -> PatchTST:
    """Base PatchTST variant (~1M params)."""
    model = PatchTST(
        d_model=128,
        n_heads=4,
        n_layers=3,
        **kwargs,
    )

    if pretrained:
        raise NotImplementedError(
            "Pretrained weights not yet available. Training from scratch for now."
        )

    return model
