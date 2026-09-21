"""Transparent daily covariance estimators used by the research stages."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def validate_returns_window(returns_window: pd.DataFrame) -> None:
    """Check the finite, numeric return panel required by every estimator."""
    if not isinstance(returns_window, pd.DataFrame):
        raise TypeError("returns_window must be a pandas DataFrame.")
    if returns_window.shape[0] < 2 or returns_window.shape[1] < 1:
        raise ValueError("returns_window needs at least two rows and one asset.")
    if returns_window.columns.duplicated().any():
        raise ValueError("returns_window columns must be unique.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in returns_window.dtypes):
        raise TypeError("returns_window must contain only numeric columns.")
    values = returns_window.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns_window must not contain missing or non-finite values.")


def sample_covariance(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Return the conventional daily sample covariance, preserving ticker labels."""
    validate_returns_window(returns_window)
    return returns_window.cov()


def ewma_weights(observations: int, lambda_: float = 0.94) -> np.ndarray:
    """Return normalized oldest-to-newest EWMA weights for a return window."""
    if observations < 1:
        raise ValueError("observations must be positive.")
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be strictly between 0 and 1.")

    # Rows are chronological: the first is oldest (age T-1), the last newest (age 0).
    ages = np.arange(observations - 1, -1, -1, dtype=float)
    raw_weights = np.power(lambda_, ages)
    return raw_weights / raw_weights.sum()


def ewma_covariance(returns_window: pd.DataFrame, lambda_: float = 0.94) -> pd.DataFrame:
    """Return explicitly weighted, demeaned daily EWMA covariance without ddof correction."""
    validate_returns_window(returns_window)
    values = returns_window.to_numpy(dtype=float)
    weights = ewma_weights(len(returns_window), lambda_)

    weighted_mean = weights @ values
    centered = values - weighted_mean
    covariance = (centered * weights[:, None]).T @ centered
    return pd.DataFrame(covariance, index=returns_window.columns, columns=returns_window.columns)


def ledoit_wolf_covariance(returns_window: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Return daily Ledoit-Wolf covariance and its fitted shrinkage coefficient."""
    validate_returns_window(returns_window)
    estimator = LedoitWolf().fit(returns_window.to_numpy(dtype=float))
    covariance = pd.DataFrame(
        estimator.covariance_,
        index=returns_window.columns,
        columns=returns_window.columns,
    )
    return covariance, float(estimator.shrinkage_)


def validate_covariance_matrix(matrix: pd.DataFrame, ticker_order: Sequence[str], *, tolerance: float = 1e-12) -> None:
    """Validate the labeled numerical properties required of a Stage 2 covariance matrix."""
    tickers = list(ticker_order)
    if matrix.shape != (len(tickers), len(tickers)):
        raise ValueError(f"Covariance shape must be {(len(tickers), len(tickers))}; found {matrix.shape}.")
    if list(matrix.index) != tickers or list(matrix.columns) != tickers:
        raise ValueError("Covariance labels must equal the input ticker order.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise TypeError("Covariance matrix must contain only numeric values.")

    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix contains missing or infinite values.")
    if not np.allclose(values, values.T, rtol=1e-10, atol=tolerance):
        raise ValueError("Covariance matrix is not symmetric within tolerance.")
    if (np.diag(values) < 0.0).any():
        raise ValueError("Covariance matrix contains a negative diagonal variance.")

    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())
    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            f"Covariance matrix has materially negative minimum eigenvalue: {minimum_eigenvalue:.6e}."
        )
