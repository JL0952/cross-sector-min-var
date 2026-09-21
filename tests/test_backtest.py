import numpy as np
import pandas as pd
import pytest

from cross_sector_min_var.backtest import (
    DEFAULT_LOOKBACK,
    MODEL_ORDER,
    build_rebalance_schedule,
    forecast_error_metrics,
    one_sided_turnover,
    simulate_drifted_holding,
    validate_backtest_outputs,
)
from cross_sector_min_var.stage4 import full_period_metrics, rolling_backtest


def test_drifted_weights_and_one_day_portfolio_return() -> None:
    target = pd.Series([0.6, 0.4], index=["AAA", "BBB"])
    date = pd.Timestamp("2024-01-02")
    holding_returns = pd.DataFrame({"AAA": [0.10], "BBB": [-0.05]}, index=[date])

    daily_returns, end_weights = simulate_drifted_holding(target, holding_returns)
    expected_portfolio_return = 0.6 * 0.10 + 0.4 * -0.05
    expected_end_weights = pd.Series(
        {
            "AAA": 0.6 * 1.10 / (1.0 + expected_portfolio_return),
            "BBB": 0.4 * 0.95 / (1.0 + expected_portfolio_return),
        }
    )

    assert daily_returns.loc[date] == pytest.approx(expected_portfolio_return)
    pd.testing.assert_series_equal(end_weights, expected_end_weights.rename("pretrade_weight"))


def test_turnover_uses_drifted_pretrade_weights_not_prior_target() -> None:
    target = pd.Series([0.6, 0.4], index=["AAA", "BBB"])
    next_target = pd.Series([0.5, 0.5], index=["AAA", "BBB"])
    holding_returns = pd.DataFrame({"AAA": [0.10], "BBB": [-0.05]}, index=[pd.Timestamp("2024-01-02")])
    _, drifted_pretrade = simulate_drifted_holding(target, holding_returns)

    actual = one_sided_turnover(next_target, drifted_pretrade)
    expected = 0.5 * np.abs(next_target - drifted_pretrade).sum()
    target_to_target = 0.5 * np.abs(next_target - target).sum()

    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(target_to_target)


def test_forecast_error_sign_convention() -> None:
    overprediction = forecast_error_metrics(0.12, 0.10)
    underprediction = forecast_error_metrics(0.08, 0.10)

    assert overprediction["forecast_error"] == pytest.approx(0.02)
    assert overprediction["absolute_error"] == pytest.approx(0.02)
    assert overprediction["squared_error"] == pytest.approx(0.0004)
    assert underprediction["forecast_error"] == pytest.approx(-0.02)


@pytest.fixture(scope="module")
def returns(returns_panel: pd.DataFrame) -> pd.DataFrame:
    return returns_panel


@pytest.fixture(scope="module")
def schedule(returns) -> pd.DataFrame:
    return build_rebalance_schedule(returns)


def test_schedule_has_exact_causal_windows_and_complete_months(schedule) -> None:
    assert (schedule["estimation_observations"] == DEFAULT_LOOKBACK).all()
    assert (schedule["estimation_end"] <= schedule["rebalance_date"]).all()
    assert (schedule["holding_start"] > schedule["rebalance_date"]).all()
    assert (schedule["holding_end"] > schedule["holding_start"]).all()
    assert (schedule["holding_start"] > schedule["estimation_end"]).all()
    assert len(schedule) >= 3
    assert schedule["rebalance_date"].is_monotonic_increasing


def test_short_rolling_backtest_has_unique_oos_returns_and_valid_outputs(returns, schedule, sector_map) -> None:
    short_schedule = schedule.iloc[:3].copy()
    generated_schedule, forecasts, weights, daily_returns = rolling_backtest(
        returns, sector_map, schedule=short_schedule
    )
    validate_backtest_outputs(generated_schedule, forecasts, weights, daily_returns)

    assert forecasts.shape[0] == len(short_schedule) * len(MODEL_ORDER)
    assert not daily_returns.duplicated(["date", "model"]).any()
    assert daily_returns.groupby("model")["date"].min().eq(short_schedule.iloc[0]["holding_start"]).all()
    assert set(full_period_metrics(forecasts, daily_returns)["model"]) == set(MODEL_ORDER)
