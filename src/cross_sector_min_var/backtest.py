"""Transparent monthly holding-period simulation and turnover mechanics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from cross_sector_min_var.covariance import validate_returns_window


DEFAULT_LOOKBACK = 504
MODEL_ORDER = ["sample", "ewma", "ledoit_wolf", "equal_weight"]


def build_rebalance_schedule(returns: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """Build complete monthly out-of-sample periods with a fixed historical lookback."""
    validate_returns_window(returns)
    if not isinstance(returns.index, pd.DatetimeIndex) or not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must have a sorted DatetimeIndex.")
    if returns.index.has_duplicates:
        raise ValueError("Return dates must be unique.")
    if len(returns) < lookback + 2:
        raise ValueError("Not enough returns to form a lookback and a future holding period.")

    month_ends = (
        returns.index.to_series()
        .groupby(returns.index.to_period("M"), sort=True)
        .last()
        .tolist()
    )
    positions = {date: position for position, date in enumerate(returns.index)}
    eligible = [date for date in month_ends if positions[date] + 1 >= lookback]
    if len(eligible) < 2:
        raise ValueError("Need two eligible month-end dates to create one complete holding period.")

    records = []
    for rebalance_date, next_rebalance_date in zip(eligible[:-1], eligible[1:]):
        rebalance_position = positions[rebalance_date]
        next_position = positions[next_rebalance_date]
        estimation = returns.iloc[rebalance_position - lookback + 1 : rebalance_position + 1]
        holding = returns.iloc[rebalance_position + 1 : next_position + 1]
        if len(estimation) != lookback:
            raise ValueError("A scheduled estimation window does not contain the requested lookback.")
        if holding.empty:
            raise ValueError("A scheduled holding period is empty.")
        records.append(
            {
                "rebalance_date": rebalance_date,
                "estimation_start": estimation.index[0],
                "estimation_end": estimation.index[-1],
                "holding_start": holding.index[0],
                "holding_end": holding.index[-1],
                "holding_days": len(holding),
                "estimation_observations": len(estimation),
            }
        )

    return pd.DataFrame(records)


def simulate_drifted_holding(
    target_weights: pd.Series, holding_returns: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Generate daily buy-and-hold returns and end-of-day drifted weights."""
    if not isinstance(holding_returns, pd.DataFrame) or holding_returns.empty:
        raise ValueError("Holding returns must be a non-empty DataFrame.")
    if holding_returns.columns.duplicated().any() or not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in holding_returns.dtypes
    ):
        raise ValueError("Holding returns must have unique numeric ticker columns.")
    if not np.isfinite(holding_returns.to_numpy(dtype=float)).all():
        raise ValueError("Holding returns must be finite.")
    if list(target_weights.index) != list(holding_returns.columns):
        raise ValueError("Target weights must exactly match the holding-return ticker order.")
    weights = target_weights.astype(float).copy()
    if not np.isfinite(weights.to_numpy()).all() or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("Target weights must be finite and sum to one.")

    portfolio_returns = []
    for date, asset_returns in holding_returns.iterrows():
        daily_return = float(np.dot(weights.to_numpy(), asset_returns.to_numpy(dtype=float)))
        gross_portfolio_return = 1.0 + daily_return
        if gross_portfolio_return <= 0.0:
            raise ValueError(f"Portfolio gross return is non-positive on {date.date()}.")
        weights = weights * (1.0 + asset_returns) / gross_portfolio_return
        portfolio_returns.append((date, daily_return))

    daily_series = pd.Series(
        [value for _, value in portfolio_returns],
        index=pd.DatetimeIndex([date for date, _ in portfolio_returns], name="date"),
        name="portfolio_return",
        dtype=float,
    )
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("Drifted weights do not sum to one.")
    return daily_series, weights.rename("pretrade_weight")


def one_sided_turnover(target_weights: pd.Series, pretrade_weights: pd.Series) -> float:
    """Calculate turnover against post-holding, pre-trade weights in identical ticker order."""
    if list(target_weights.index) != list(pretrade_weights.index):
        raise ValueError("Target and pre-trade weights must have identical ticker order.")
    return float(0.5 * np.abs(target_weights.to_numpy() - pretrade_weights.to_numpy()).sum())


def forecast_error_metrics(predicted_vol: float, realized_vol: float) -> dict[str, float]:
    """Keep the project-wide forecast-error sign convention explicit."""
    forecast_error = float(predicted_vol - realized_vol)
    return {
        "forecast_error": forecast_error,
        "absolute_error": float(abs(forecast_error)),
        "squared_error": float(forecast_error**2),
    }


def realized_annualized_volatility(portfolio_returns: pd.Series) -> float:
    """Calculate holding-period annualized realized volatility from daily portfolio returns."""
    if len(portfolio_returns) < 2:
        raise ValueError("At least two holding-period returns are needed for sample volatility.")
    return float(portfolio_returns.std(ddof=1) * np.sqrt(252.0))


def validate_backtest_outputs(
    schedule: pd.DataFrame,
    forecasts: pd.DataFrame,
    target_weights: pd.DataFrame,
    daily_portfolio_returns: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
) -> None:
    """Validate look-ahead controls, panel uniqueness, and fixed-window counts."""
    required_schedule = {
        "rebalance_date",
        "estimation_start",
        "estimation_end",
        "holding_start",
        "holding_end",
        "holding_days",
        "estimation_observations",
    }
    if not required_schedule.issubset(schedule.columns):
        raise ValueError("Schedule is missing required audit fields.")
    if not (schedule["estimation_observations"] == lookback).all():
        raise ValueError("Every covariance estimation window must contain exactly the requested lookback.")
    if not (schedule["estimation_end"] <= schedule["rebalance_date"]).all():
        raise ValueError("An estimation window extends beyond its rebalance date.")
    if not (schedule["holding_start"] > schedule["rebalance_date"]).all():
        raise ValueError("A holding period starts on or before its rebalance date.")
    if not (schedule["holding_end"] > schedule["holding_start"]).all():
        raise ValueError("A holding period ends before it starts.")

    if forecasts.duplicated(["rebalance_date", "model"]).any():
        raise ValueError("Forecast output has duplicate model/rebalance observations.")
    if target_weights.duplicated(["rebalance_date", "model", "ticker"]).any():
        raise ValueError("Target-weight output has duplicate model/rebalance/ticker observations.")
    if daily_portfolio_returns.duplicated(["date", "model"]).any():
        raise ValueError("Daily portfolio returns contain overlapping duplicate observations.")
    if not np.isfinite(daily_portfolio_returns["portfolio_return"].to_numpy(dtype=float)).all():
        raise ValueError("Daily portfolio return output contains non-finite values.")

    expected_forecasts = len(schedule) * len(MODEL_ORDER)
    if len(forecasts) != expected_forecasts:
        raise ValueError("Forecast output does not have one row per model and rebalance.")
    expected_weights = len(schedule) * len(MODEL_ORDER) * target_weights["ticker"].nunique()
    if len(target_weights) != expected_weights:
        raise ValueError("Target-weight output does not have one row per model/rebalance/ticker.")


def reindex_weights(weights: pd.Series, ticker_order: Sequence[str]) -> pd.Series:
    """Require and return weights in the return-panel ticker order."""
    if set(weights.index) != set(ticker_order):
        raise ValueError("Weight tickers do not match the return-panel ticker universe.")
    return weights.reindex(list(ticker_order))
