"""Fine-tuning configuration. Pretty standard stuff."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FineTuneConfig:
    """Configuration for a fine-tuning run.

    Attributes:
        epochs: Number of training epochs.
        learning_rate: Peak learning rate (used with cosine schedule).
        batch_size: Per-GPU batch size. Reduce if you OOM.
        seq_len: Input sequence length (must match model's expected context).
        pred_len: Forecast horizon.
        use_gpu: Whether to use GPU. Falls back to CPU if no GPU available.
        gradient_accumulation_steps: Simulate larger batch sizes.
            NOTE: partially implemented, may not interact correctly with LR schedule.
        max_grad_norm: Gradient clipping threshold.
        warmup_steps: Linear warmup steps. 0 = no warmup.
        output_dir: Where to save checkpoints and logs.
        seed: Random seed. Not fully deterministic yet (known issue with dropout).
        fp16: Use mixed-precision training. Only works on GPU.
        val_every_n_epochs: Run validation every N epochs. Set to 1 for per-epoch.
        early_stopping_patience: Patience for early stopping. None = disabled.
        save_total_limit: Max checkpoints to keep. Deletes oldest.
    """

    # Training
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Data
    seq_len: int = 96
    pred_len: int = 24

    # Hardware
    use_gpu: bool = True
    fp16: bool = False  # TODO: test on non-NVIDIA GPUs (ROCm, MPS)

    # Regularization
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Saving
    output_dir: str = "./checkpoints"
    save_total_limit: int = 3
    val_every_n_epochs: int = 1
    early_stopping_patience: Optional[int] = None

    # Reproducibility
    seed: int = 42
    # NOTE: full determinism is hard. CuDNN benchmarking is enabled by default
    # for performance. Set torch.backends.cudnn.deterministic = True if you
    # need bit-for-bit reproducibility (it'll be slower).

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
