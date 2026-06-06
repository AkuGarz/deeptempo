"""Benchmark suite for model evaluation.

TODO: Implement full benchmark with multiple datasets and standardized metrics.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from deeptempo.eval.metrics import compute_metrics
from deeptempo.utils.logging import get_logger

logger = get_logger(__name__)


def run_benchmark(
    model: torch.nn.Module,
    test_dataset,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> dict:
    """Run a benchmark evaluation on a test dataset.

    Args:
        model: Trained model.
        test_dataset: Test dataset (returns (x, y) tuples).
        batch_size: Batch size for evaluation.
        device: Device to use. If None, auto-detects GPU.

    Returns:
        Dict with aggregate metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(preds, targets)
    logger.info(f"Benchmark results: {metrics}")

    return metrics
