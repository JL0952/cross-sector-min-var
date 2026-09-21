"""Historical tail-risk calculations for saved out-of-sample portfolio returns."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_RISK_LOOKBACK = 252
DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)

DAILY_VAR_COLUMNS = [
    "date",
    "model",
    "confidence_level",
    "estimation_start",
    "estimation_end",
    "estimation_observations",
    "historical_var",
    "historical_es",
    "realized_return",
    "realized_loss",
    "var_exceedance",
]

VAR_BACKTEST_SUMMARY_COLUMNS = [
    "model",
    "confidence_level",
    "var_forecasts",
    "var_exceedances",
    "empirical_exceedance_rate",
    "expected_exceedance_rate",
    "average_historical_var",
    "average_historical_es",
    "maximum_realized_loss",
    "kupiec_lr_uc",
    "kupiec_p_value",
]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_confidence_level(confidence_level: float) -> float:
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be finite and strictly between zero and one.")
    return confidence


def historical_var_es(portfolio_returns: Sequence[float], confidence_level: float) -> dict[str, float]:
    """Calculate unfloored historical VaR and ES from one return history.

    Loss is defined as ``-portfolio_return``. VaR uses NumPy's ``higher``
    quantile method, so it is an empirical order statistic rather than an
    interpolated loss. No zero floor is applied: an unusual all-positive
    return history can therefore yield a negative VaR or ES.
    """
    confidence = _validate_confidence_level(confidence_level)
    returns = np.asarray(portfolio_returns, dtype=float)
    if returns.ndim != 1 or len(returns) == 0 or not np.isfinite(returns).all():
        raise ValueError("portfolio_returns must be a non-empty finite one-dimensional sequence.")

    losses = -returns
    value_at_risk = float(np.quantile(losses, confidence, method="higher"))
    tail_losses = losses[losses >= value_at_risk]
    return {
        "historical_var": value_at_risk,
        "historical_es": float(tail_losses.mean()),
    }


def var_exceedance(realized_return: float, historical_var: float) -> bool:
    """Classify a VaR exception when the realized loss strictly exceeds VaR."""
    realized_loss = -float(realized_return)
    value_at_risk = float(historical_var)
    if not np.isfinite(realized_loss) or not np.isfinite(value_at_risk):
        raise ValueError("realized_return and historical_var must be finite.")
    return bool(realized_loss > value_at_risk)


def validate_daily_portfolio_returns(
    daily_returns: pd.DataFrame, expected_models: Sequence[str] | None = None
) -> None:
    """Validate the tidy daily-return contract consumed by Stage 7."""
    _require_columns(daily_returns, ["date", "model", "portfolio_return"], "Daily portfolio returns")
    if daily_returns.empty:
        raise ValueError("Daily portfolio returns must not be empty.")
    if not pd.api.types.is_datetime64_any_dtype(daily_returns["date"]):
        raise TypeError("Daily portfolio return dates must be datetime values.")
    if daily_returns["date"].isna().any() or daily_returns["model"].isna().any():
        raise ValueError("Daily portfolio return dates and models must be present.")
    if daily_returns.duplicated(["date", "model"]).any():
        raise ValueError("Daily portfolio returns contain duplicate model/date rows.")
    values = pd.to_numeric(daily_returns["portfolio_return"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Daily portfolio returns must be finite.")

    models = daily_returns["model"].drop_duplicates().tolist()
    if expected_models is not None:
        expected = list(expected_models)
        if set(models) != set(expected) or len(models) != len(expected):
            raise ValueError(f"Daily portfolio models must be exactly {expected}.")
        reference_dates: pd.DatetimeIndex | None = None
        for model in expected:
            model_dates = pd.DatetimeIndex(
                daily_returns.loc[daily_returns["model"] == model, "date"].sort_values()
            )
            if reference_dates is None:
                reference_dates = model_dates
            elif not model_dates.equals(reference_dates):
                raise ValueError("Daily portfolio models must have identical date coverage.")


def generate_historical_var_forecasts(
    daily_returns: pd.DataFrame,
    *,
    lookback: int = DEFAULT_RISK_LOOKBACK,
    confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
) -> pd.DataFrame:
    """Forecast one-day historical VaR/ES using only the preceding observations."""
    if lookback < 1:
        raise ValueError("lookback must be positive.")
    confidences = tuple(_validate_confidence_level(level) for level in confidence_levels)
    if not confidences or len(set(confidences)) != len(confidences):
        raise ValueError("confidence_levels must contain unique values.")
    validate_daily_portfolio_returns(daily_returns)

    records: list[dict[str, object]] = []
    for model, group in daily_returns.groupby("model", sort=False):
        ordered = group.sort_values("date", kind="stable").reset_index(drop=True)
        dates = ordered["date"]
        returns = ordered["portfolio_return"].to_numpy(dtype=float)
        for position in range(lookback, len(ordered)):
            historical_returns = returns[position - lookback : position]
            realized_return = float(returns[position])
            for confidence in confidences:
                estimates = historical_var_es(historical_returns, confidence)
                records.append(
                    {
                        "date": dates.iloc[position],
                        "model": model,
                        "confidence_level": confidence,
                        "estimation_start": dates.iloc[position - lookback],
                        "estimation_end": dates.iloc[position - 1],
                        "estimation_observations": lookback,
                        **estimates,
                        "realized_return": realized_return,
                        "realized_loss": -realized_return,
                        "var_exceedance": var_exceedance(realized_return, estimates["historical_var"]),
                    }
                )

    forecasts = pd.DataFrame(records, columns=DAILY_VAR_COLUMNS)
    validate_daily_var_forecasts(forecasts, lookback=lookback, allow_empty=True)
    return forecasts


def validate_daily_var_forecasts(
    forecasts: pd.DataFrame, *, lookback: int | None = None, allow_empty: bool = False
) -> None:
    """Validate Stage 7 forecast rows, including their no-look-ahead audit fields."""
    _require_columns(forecasts, DAILY_VAR_COLUMNS, "Daily VaR forecasts")
    if forecasts.empty:
        if allow_empty:
            return
        raise ValueError("Daily VaR forecasts must not be empty.")
    for column in ["date", "estimation_start", "estimation_end"]:
        if not pd.api.types.is_datetime64_any_dtype(forecasts[column]):
            raise TypeError(f"Daily VaR forecast {column} values must be datetime values.")
    if forecasts.duplicated(["date", "model", "confidence_level"]).any():
        raise ValueError("Daily VaR forecasts contain duplicate model/date/confidence rows.")
    if not (forecasts["estimation_start"] <= forecasts["estimation_end"]).all():
        raise ValueError("A VaR estimation start occurs after its estimation end.")
    if not (forecasts["estimation_end"] < forecasts["date"]).all():
        raise ValueError("A VaR estimation window includes its forecast-date return.")
    if lookback is not None and not (forecasts["estimation_observations"] == lookback).all():
        raise ValueError("VaR forecast lookback counts differ from the requested value.")

    numeric_columns = [
        "confidence_level",
        "estimation_observations",
        "historical_var",
        "historical_es",
        "realized_return",
        "realized_loss",
    ]
    values = forecasts[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Daily VaR forecasts contain missing or non-finite numeric values.")
    if not np.allclose(
        forecasts["realized_loss"].to_numpy(dtype=float),
        -forecasts["realized_return"].to_numpy(dtype=float),
    ):
        raise ValueError("Daily VaR realized losses must equal negative realized returns.")
    expected_exceedances = forecasts["realized_loss"] > forecasts["historical_var"]
    if not forecasts["var_exceedance"].astype(bool).equals(expected_exceedances):
        raise ValueError("Daily VaR exceedance flags do not match the loss convention.")


def kupiec_unconditional_coverage(
    exceedances: int, forecasts: int, expected_exceedance_rate: float
) -> tuple[float, float]:
    """Return the Kupiec LR statistic and one-degree-of-freedom chi-square p-value."""
    if forecasts < 1:
        raise ValueError("Kupiec coverage requires at least one forecast.")
    if exceedances < 0 or exceedances > forecasts:
        raise ValueError("exceedances must be between zero and forecasts.")
    expected_rate = float(expected_exceedance_rate)
    if not np.isfinite(expected_rate) or not 0.0 < expected_rate < 1.0:
        raise ValueError("expected_exceedance_rate must be strictly between zero and one.")

    non_exceedances = forecasts - exceedances

    def _x_log_ratio(count: int, expected_count: float) -> float:
        return 0.0 if count == 0 else float(count * math.log(count / expected_count))

    statistic = 2.0 * (
        _x_log_ratio(exceedances, forecasts * expected_rate)
        + _x_log_ratio(non_exceedances, forecasts * (1.0 - expected_rate))
    )
    statistic = max(0.0, float(statistic))
    # For chi-square with one degree of freedom, sf(x) = erfc(sqrt(x / 2)).
    return statistic, float(math.erfc(math.sqrt(statistic / 2.0)))


def summarize_var_backtest(
    forecasts: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
) -> pd.DataFrame:
    """Summarize historical-VaR coverage and tail estimates by model and confidence level."""
    validate_daily_var_forecasts(forecasts, allow_empty=True)
    confidences = tuple(_validate_confidence_level(level) for level in confidence_levels)
    if not confidences or len(set(confidences)) != len(confidences):
        raise ValueError("confidence_levels must contain unique values.")
    summary_models = list(models) if models is not None else forecasts["model"].drop_duplicates().tolist()
    if not summary_models:
        raise ValueError("models must be supplied when summarizing an empty forecast table.")

    records: list[dict[str, object]] = []
    for model in summary_models:
        for confidence in confidences:
            subset = forecasts.loc[
                (forecasts["model"] == model) & np.isclose(forecasts["confidence_level"], confidence)
            ]
            count = len(subset)
            expected_rate = 1.0 - confidence
            if count:
                exceedances = int(subset["var_exceedance"].sum())
                kupiec_statistic, kupiec_p_value = kupiec_unconditional_coverage(exceedances, count, expected_rate)
                empirical_rate = float(exceedances / count)
                average_var = float(subset["historical_var"].mean())
                average_es = float(subset["historical_es"].mean())
                maximum_loss = float(subset["realized_loss"].max())
            else:
                exceedances = 0
                kupiec_statistic = np.nan
                kupiec_p_value = np.nan
                empirical_rate = np.nan
                average_var = np.nan
                average_es = np.nan
                maximum_loss = np.nan
            records.append(
                {
                    "model": model,
                    "confidence_level": confidence,
                    "var_forecasts": count,
                    "var_exceedances": exceedances,
                    "empirical_exceedance_rate": empirical_rate,
                    "expected_exceedance_rate": expected_rate,
                    "average_historical_var": average_var,
                    "average_historical_es": average_es,
                    "maximum_realized_loss": maximum_loss,
                    "kupiec_lr_uc": kupiec_statistic,
                    "kupiec_p_value": kupiec_p_value,
                }
            )
    return pd.DataFrame(records, columns=VAR_BACKTEST_SUMMARY_COLUMNS)
