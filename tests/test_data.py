"""Tests for data loading. Minimal but useful."""

import pytest
import numpy as np

from deeptempo.data.loader import TimeSeriesDataset


class TestTimeSeriesDataset:
    def test_basic(self):
        data = np.sin(np.linspace(0, 4 * np.pi, 500))
        ds = TimeSeriesDataset(data, seq_len=50, pred_len=10)

        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (50,)
        assert y.shape == (10,)

    def test_stride(self):
        data = np.arange(200, dtype=np.float32)
        ds1 = TimeSeriesDataset(data, seq_len=10, pred_len=5, stride=1)
        ds2 = TimeSeriesDataset(data, seq_len=10, pred_len=5, stride=5)

        assert len(ds2) < len(ds1)
        assert len(ds2) == (200 - 10 - 5) // 5 + 1

    def test_too_small_dataset(self):
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Dataset too small"):
            TimeSeriesDataset(data, seq_len=5, pred_len=3)

    def test_deterministic(self):
        data = np.random.randn(300).astype(np.float32)
        ds = TimeSeriesDataset(data, seq_len=30, pred_len=10)

        x0, y0 = ds[0]
        x0_again, y0_again = ds[0]
        assert np.allclose(x0.numpy(), x0_again.numpy())
        assert np.allclose(y0.numpy(), y0_again.numpy())
