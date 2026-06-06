"""Model registry. Maps string names to model constructors.

Currently barebones. Will grow as more architectures are added.
"""

from typing import Callable

import torch.nn as nn

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model constructor."""
    def decorator(fn: Callable[..., nn.Module]):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """Get a model by name.

    Args:
        name: Model identifier (e.g., "patchtst_tiny").
        **kwargs: Passed to the model constructor.

    Returns:
        Initialized nn.Module.

    Raises:
        KeyError: If model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Unknown model '{name}'. Available: {available or '(none registered)'}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())
