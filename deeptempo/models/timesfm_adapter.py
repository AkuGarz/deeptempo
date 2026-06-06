"""Experimental adapter for Google's TimesFM model.

⚠️ EXPERIMENTAL — This code is incomplete and untested on real workloads.
The TimesFM checkpoint loading relies on the unofficial `timesfm` package,
which itself is unstable. Use at your own risk.

If you get this working on a real dataset, please open an issue or PR.
I haven't had time to debug the full pipeline.
"""

from typing import Optional

import torch
import torch.nn as nn

from deeptempo.utils.logging import get_logger

logger = get_logger(__name__)


class TimesFMAdapter(nn.Module):
    """Wraps TimesFM for use with DeepTempo's training loop.

    This is a thin wrapper that converts between DeepTempo's expected
    input/output shapes and TimesFM's internal format.

    Status: EXPERIMENTAL. Known issues:
        - TimesFM checkpoint loading is fragile (version-dependent)
        - Output format doesn't always match what the paper describes
        - Only tested on a Colab T4 with a toy dataset
    """

    def __init__(
        self,
        checkpoint_path: str,
        seq_len: int = 96,
        pred_len: int = 24,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # TODO: import timesfm properly
        # The package name might be `timesfm` or `timesfm-pytorch` depending on version
        try:
            import timesfm  # type: ignore[import-untyped]

            self.timesfm = timesfm.TimesFm(
                context_len=seq_len,
                horizon_len=pred_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend="gpu",
                per_core_batch_size=32,
                checkpoint_path=checkpoint_path,
            )
        except ImportError:
            logger.error(
                "TimesFM package not found. Install it with:\n"
                "  pip install timesfm  # or the appropriate package\n"
                "This adapter will not work without it."
            )
            raise

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.timesfm.parameters():
                param.requires_grad = False
            logger.info("TimesFM backbone frozen.")

        # Simple prediction head on top
        self.head = nn.Linear(pred_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TimesFM.

        Args:
            x: (batch, seq_len) univariate input.

        Returns:
            (batch, pred_len) forecast.

        NOTE: TimesFM expects specific input formatting. The conversion
        between DeepTempo's format and TimesFM's is janky. Expect shape errors.
        """
        # TimesFM expects (batch, seq_len) with specific frequency metadata
        # This is a best-effort conversion — may need adjustment
        # TODO: handle frequency encoding properly
        try:
            # The TimesFM API is not stable between versions
            # This code was written against timesfm==0.1.0
            output = self.timesfm.forecast(x, horizon=self.pred_len)
            output = self.head(output)
            return output
        except Exception as e:
            logger.error(f"TimesFM forward pass failed: {e}")
            logger.error(
                "This is expected — the TimesFM adapter is experimental. "
                "If you're seeing this, please open a GitHub issue with your "
                "TimesFM version and input shape."
            )
            raise
