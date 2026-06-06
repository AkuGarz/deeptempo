#!/usr/bin/env python3
"""Fine-tune PatchTST on ETTm1.

This is a working example that should run on any machine with a GPU.
Expected runtime: ~5-8 minutes on an A10G, ~45 minutes on CPU.

Usage:
    python examples/02_finetune_ettm1.py

If you're on CPU, it'll work but be painfully slow. You've been warned.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from deeptempo.core.config import FineTuneConfig
from deeptempo.core.trainer import Trainer
from deeptempo.data.loader import load_ett_dataset
from deeptempo.models.registry import get_model


def main():
    print("=" * 60)
    print("DeepTempo — ETTm1 Fine-Tuning Example")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("No GPU detected. This will be slow. Go make coffee.")

    # Load data
    print("\n[1/3] Loading ETTm1 dataset...")
    train_ds, val_ds = load_ett_dataset(
        name="ettm1",
        seq_len=96,
        pred_len=24,
    )
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")

    # Create model
    print("\n[2/3] Creating model (patchtst_base)...")
    model = get_model("patchtst_base", seq_len=96, pred_len=24)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Configure training
    config = FineTuneConfig(
        epochs=10,
        learning_rate=1e-4,
        batch_size=64,  # reduce to 16 if you OOM
        seq_len=96,
        pred_len=24,
        use_gpu=True,
        output_dir="./checkpoints/ettm1_example",
        val_every_n_epochs=1,
    )

    # Train
    print("\n[3/3] Training...")
    trainer = Trainer(config, model, train_ds, val_ds)
    history = trainer.fit()

    # Results
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Checkpoints saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
