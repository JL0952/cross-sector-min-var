"""Constrained global minimum-variance portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cross_sector_min_var.covariance import validate_covariance_matrix


DEFAULT_STOCK_CAP = 0.15
DEFAULT_SECTOR_CAP = 0.25
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class OptimizationResult:
    """Small, explicit record of one SLSQP optimization result."""

    weights: pd.Series
    objective_value: float
    success: bool
    message: str


SectorMap = Union[Mapping[str, str], pd.Series]


def _validated_sector_map(sector_map: SectorMap, tickers: Sequence[str]) -> dict[str, str]:
    """Return an exact ticker-aligned sector mapping or raise a clear error."""
    mapping = sector_map.to_dict() if isinstance(sector_map, pd.Series) else dict(sector_map)
    expected = set(tickers)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(f"Sector map ticker mismatch; missing={missing}, extra={extra}.")
    if any(not isinstance(mapping[ticker], str) or not mapping[ticker] for ticker in tickers):
        raise ValueError("Every ticker must map to a non-empty sector string.")
    return mapping


def sector_weights(weights: pd.Series, sector_map: SectorMap) -> pd.Series:
    """Aggregate ordered portfolio weights by the sectors supplied in metadata."""
    mapping = _validated_sector_map(sector_map, weights.index.tolist())
    sectors = list(dict.fromkeys(mapping[ticker] for ticker in weights.index))
    return pd.Series(
        {sector: float(weights[[ticker for ticker in weights.index if mapping[ticker] == sector]].sum()) for sector in sectors},
        name="sector_weight",
        dtype=float,
    )


def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Calculate the daily portfolio variance for exactly aligned labeled inputs."""
    if list(weights.index) != list(covariance.index):
        raise ValueError("Weight ticker order must exactly match covariance ticker order.")
    values = weights.to_numpy(dtype=float)
    matrix = covariance.to_numpy(dtype=float)
    return float(values @ matrix @ values)


def predicted_risk_metrics(weights: pd.Series, covariance: pd.DataFrame) -> dict[str, float]:
    """Return daily variance, daily volatility, and annualized predicted volatility."""
    daily_variance = portfolio_variance(weights, covariance)
    if daily_variance < 0.0:
        raise ValueError(f"Predicted portfolio variance is negative: {daily_variance:.6e}.")
    daily_volatility = float(np.sqrt(daily_variance))
    return {
        "daily_predicted_variance": daily_variance,
        "daily_predicted_volatility": daily_volatility,
        "annualized_predicted_volatility": float(np.sqrt(252.0 * daily_variance)),
    }


def equal_weight_portfolio(tickers: Sequence[str]) -> pd.Series:
    """Return a labeled equal-weight portfolio in the supplied ticker order."""
    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("At least one ticker is required.")
    return pd.Series(1.0 / len(ticker_list), index=ticker_list, name="equal_weight", dtype=float)


def minimum_variance_weights(
    covariance: pd.DataFrame,
    sector_map: SectorMap,
    max_stock_weight: float = DEFAULT_STOCK_CAP,
    max_sector_weight: float = DEFAULT_SECTOR_CAP,
) -> OptimizationResult:
    """Solve the long-only, stock-capped, sector-capped global minimum-variance problem."""
    tickers = covariance.index.tolist()
    validate_covariance_matrix(covariance, tickers)
    mapping = _validated_sector_map(sector_map, tickers)
    if max_stock_weight <= 0.0 or max_sector_weight <= 0.0:
        raise ValueError("Stock and sector caps must be positive.")

    initial_weights = equal_weight_portfolio(tickers)
    initial_sector_weights = sector_weights(initial_weights, mapping)
    if (initial_weights > max_stock_weight + DEFAULT_TOLERANCE).any() or (
        initial_sector_weights > max_sector_weight + DEFAULT_TOLERANCE
    ).any():
        raise ValueError("Equal-weight initial guess is not feasible under the requested caps.")

    matrix = covariance.to_numpy(dtype=float)
    sector_indices = {
        sector: np.array([position for position, ticker in enumerate(tickers) if mapping[ticker] == sector], dtype=int)
        for sector in dict.fromkeys(mapping[ticker] for ticker in tickers)
    }

    def objective(weights: np.ndarray) -> float:
        return float(weights @ matrix @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * matrix @ weights

    constraints = [{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0), "jac": lambda weights: np.ones_like(weights)}]
    constraints.extend(
        {
            "type": "ineq",
            "fun": lambda weights, indices=indices: float(max_sector_weight - weights[indices].sum()),
            "jac": lambda weights, indices=indices: np.where(
                np.arange(len(weights))[:, None] == indices, -1.0, 0.0
            ).sum(axis=1),
        }
        for indices in sector_indices.values()
    )

    result = minimize(
        objective,
        initial_weights.to_numpy(),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, max_stock_weight)] * len(tickers),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
    )
    weights = pd.Series(result.x, index=tickers, name="weight", dtype=float)
    return OptimizationResult(
        weights=weights,
        objective_value=float(objective(result.x)),
        success=bool(result.success),
        message=str(result.message),
    )


def validate_optimization_result(
    result: OptimizationResult,
    covariance: pd.DataFrame,
    sector_map: SectorMap,
    max_stock_weight: float = DEFAULT_STOCK_CAP,
    max_sector_weight: float = DEFAULT_SECTOR_CAP,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """Validate all Stage 3 feasibility and numerical requirements without clipping."""
    tickers = covariance.index.tolist()
    if not result.success:
        raise RuntimeError(f"SLSQP optimization failed: {result.message}")
    if list(result.weights.index) != tickers:
        raise ValueError("Optimized weight labels/order do not match covariance labels/order.")
    values = result.weights.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(result.objective_value):
        raise ValueError("Optimization returned non-finite weights or objective value.")
    if abs(values.sum() - 1.0) > tolerance:
        raise ValueError(f"Optimized weights do not sum to one: {values.sum():.12f}.")
    if (values < -tolerance).any():
        raise ValueError("Optimization returned materially negative weights.")
    if (values > max_stock_weight + tolerance).any():
        raise ValueError("Optimization exceeded the individual-stock cap.")

    aggregated_sectors = sector_weights(result.weights, sector_map)
    if (aggregated_sectors > max_sector_weight + tolerance).any():
        raise ValueError("Optimization exceeded a sector cap.")

    predicted_variance = portfolio_variance(result.weights, covariance)
    if predicted_variance < -tolerance:
        raise ValueError(f"Optimization returned materially negative variance: {predicted_variance:.6e}.")
    if not np.isclose(result.objective_value, predicted_variance, rtol=1e-9, atol=tolerance):
        raise ValueError("Reported optimizer objective does not equal recomputed portfolio variance.")

