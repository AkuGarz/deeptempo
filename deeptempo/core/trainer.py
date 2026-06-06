"""Main training loop. The workhorse of DeepTempo.

This is where most of the GPU time is spent. Currently single-GPU only.
DDP support is the #1 priority for v0.3.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR  # TODO: switch to cosine with warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeptempo.core.config import FineTuneConfig
from deeptempo.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Single-GPU training loop for time-series models.

    This is deliberately simple. No callbacks, no plugins, no distributed.
    Those come later.

    Args:
        config: FineTuneConfig dataclass.
        model: A PyTorch model (should accept (batch, seq_len, n_vars) and
               return (batch, pred_len, n_vars)).
        train_dataset: Training dataset (returns (x, y) tuples).
        val_dataset: Optional validation dataset.
    """

    def __init__(
        self,
        config: FineTuneConfig,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Device setup
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            self.device = torch.device("cpu")
            if config.use_gpu:
                logger.warning("use_gpu=True but no GPU found. Falling back to CPU.")
            else:
                logger.info("Using CPU (this will be slow)")

        self.model = self.model.to(self.device)

        # TODO: add option to freeze specific layers
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # TODO: replace with CosineAnnealingWarmRestarts or a proper scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )

        self.criterion = nn.MSELoss()

        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def fit(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with training history (train_losses, val_losses, best_val_loss).
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,  # TODO: make configurable
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device.type == "cuda" else False,
            )

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            log_msg = f"Epoch {epoch+1}/{self.config.epochs} | train_loss: {train_loss:.4f}"

            if val_loader is not None and (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)
                log_msg += f" | val_loss: {val_loss:.4f}"

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")
                    log_msg += " ✓"

            logger.info(log_msg)
            self.scheduler.step()

        self._save_checkpoint("last.pt")
        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch. Standard PyTorch stuff."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # TODO: add gradient accumulation
        for batch in tqdm(loader, desc=f"Train epoch {self.current_epoch+1}", leave=False):
            x, y = batch
            x, y = x.to(self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else float("inf")

    def _validate(self, loader: DataLoader) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x, y = x.to(self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else float("inf")

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.output_dir) / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        # TODO: implement save_total_limit cleanup (delete old checkpoints)
        logger.debug(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load a saved checkpoint. Useful for resuming.

        NOTE: The optimizer state loading assumes the same model architecture.
        If you changed the model, this will fail silently or crash.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch+1}")
