"""GPU utility functions for diagnostics and memory management.

These are helpful when you're iterating on a single GPU and need to
track memory usage or debug OOM issues.
"""

import gc
from typing import Optional

import torch


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get current GPU memory usage.

    Returns:
        Dict with 'allocated_gb', 'reserved_gb', 'free_gb', 'total_gb'.
        Returns empty dict if no GPU available.
    """
    if not torch.cuda.is_available():
        return {}

    total = torch.cuda.get_device_properties(device_id).total_mem
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    free = total - reserved

    return {
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "free_gb": free / 1e9,
        "total_gb": total / 1e9,
    }


def estimate_batch_size(
    model: torch.nn.Module,
    seq_len: int,
    pred_len: int,
    target_vram_gb: Optional[float] = None,
    safety_factor: float = 0.7,
) -> int:
    """Estimate maximum batch size that fits in GPU memory.

    This is a rough heuristic. It runs a forward pass with increasing
    batch sizes until it approaches the VRAM limit. Not exact, but
    usually close enough.

    Args:
        model: The model to test.
        seq_len: Input sequence length.
        pred_len: Prediction length.
        target_vram_gb: Target VRAM usage. If None, uses total GPU memory.
        safety_factor: Fraction of VRAM to use (0.7 = leave 30% headroom).

    Returns:
        Estimated safe batch size.

    NOTE: This is slow because it runs forward passes. Only call it once
    per model/config combination, not every run.
    """
    if not torch.cuda.is_available():
        print("No GPU found. Returning batch_size=1.")
        return 1

    if target_vram_gb is None:
        target_vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

    target_bytes = int(target_vram_gb * safety_factor * 1e9)

    model = model.cuda()
    model.train()

    batch_size = 1
    while True:
        try:
            x = torch.randn(batch_size, seq_len, device="cuda")
            y = torch.randn(batch_size, pred_len, device="cuda")
            y_pred = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            torch.cuda.empty_cache()
            gc.collect()

            if torch.cuda.max_memory_allocated() > target_bytes:
                return max(1, batch_size - 1)

            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                return max(1, batch_size // 2)
            raise

    return batch_size


def clear_gpu_cache():
    """Clear PyTorch GPU cache. Call between experiments.

    NOTE: This won't free memory held by live tensors.
    Make sure you delete or let go of your models/data first.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
