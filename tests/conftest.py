"""Shared fixtures for tests."""

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_timeseries():
    """A simple sine wave for testing."""
    t = np.linspace(0, 10 * np.pi, 1000)
    return np.sin(t).astype(np.float32)


@pytest.fixture
def device():
    """Get the available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
