from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


TICKERS = [f"Asset{i:02d}" for i in range(22)]


@pytest.fixture(scope="session")
def returns_panel() -> pd.DataFrame:
    """Create deterministic, correlated returns without external market data."""
    generator = np.random.default_rng(20260917)
    dates = pd.date_range("2020-01-02", periods=800, freq="B", name="date")
    market = generator.normal(0.0002, 0.007, size=(len(dates), 1))
    idiosyncratic = generator.normal(0.0, 0.009, size=(len(dates), len(TICKERS)))
    loadings = np.linspace(0.55, 1.15, len(TICKERS))
    return pd.DataFrame(market * loadings + idiosyncratic, index=dates, columns=TICKERS)


@pytest.fixture(scope="session")
def sector_map() -> pd.Series:
    sectors = [f"Sector {index:02d}" for index in range(11) for _ in range(2)]
    return pd.Series(sectors, index=TICKERS, name="sector")
