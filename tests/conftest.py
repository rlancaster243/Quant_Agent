"""Shared fixtures. All tests are offline: no market-data or LLM network calls."""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless chart rendering in CI

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_ohlcv(n: int = 250, drift: float = 0.001, vol: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Geometric random walk with realistic OHLC structure (High >= max(O,C), Low <= min(O,C))."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 10_000_000, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def uptrend() -> pd.DataFrame:
    return make_ohlcv(drift=0.01, vol=0.003, seed=1)


@pytest.fixture
def downtrend() -> pd.DataFrame:
    return make_ohlcv(drift=-0.01, vol=0.003, seed=2)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Never let a developer's real .env / API key leak into tests; write charts to tmp."""
    for var in ("GROQ_API_KEY", "LLM_MODEL", "OPENBB_PAT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("utils.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.chdir(tmp_path)
