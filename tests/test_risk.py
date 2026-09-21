import numpy as np
import pandas as pd
import pytest

from cross_sector_min_var.risk import (
    DEFAULT_CONFIDENCE_LEVELS,
    generate_historical_var_forecasts,
    historical_var_es,
    kupiec_unconditional_coverage,
    summarize_var_backtest,
    validate_daily_portfolio_returns,
    validate_daily_var_forecasts,
    var_exceedance,
)


def test_historical_var_uses_higher_empirical_order_statistic() -> None:
    returns = pd.Series([0.00, -0.01, -0.02, -0.03])

    estimates = historical_var_es(returns, 0.75)

    assert estimates["historical_var"] == pytest.approx(0.03)
    assert estimates["historical_var"] != pytest.approx(np.quantile(-returns, 0.75))


def test_expected_shortfall_is_mean_of_losses_at_or_above_var() -> None:
    estimates = historical_var_es([0.00, -0.01, -0.02, -0.03], 0.50)

    assert estimates["historical_var"] == pytest.approx(0.02)
    assert estimates["historical_es"] == pytest.approx(0.025)


def test_historical_var_is_not_artificially_floored_at_zero() -> None:
    estimates = historical_var_es([0.01, 0.02, 0.03, 0.04], 0.95)

    assert estimates["historical_var"] == pytest.approx(-0.01)
    assert estimates["historical_es"] == pytest.approx(-0.01)


def test_forecasts_use_only_trailing_observations_before_each_forecast_date() -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    daily = pd.DataFrame(
        {"date": dates, "model": "sample", "portfolio_return": [-0.01, 0.02, -0.03, -0.90, 0.01]}
    )

    forecasts = generate_historical_var_forecasts(daily, lookback=3, confidence_levels=(0.95,))
    first = forecasts.iloc[0]

    assert first["date"] == dates[3]
    assert first["estimation_start"] == dates[0]
    assert first["estimation_end"] == dates[2]
    assert first["historical_var"] == pytest.approx(0.03)
    assert first["realized_loss"] == pytest.approx(0.90)
    assert first["var_exceedance"]


def test_changing_a_forecast_date_return_does_not_change_its_var() -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    original = pd.DataFrame(
        {"date": dates, "model": "sample", "portfolio_return": [-0.01, 0.02, -0.03, 0.01, 0.02]}
    )
    changed = original.copy()
    changed.loc[3, "portfolio_return"] = -0.90

    original_first = generate_historical_var_forecasts(original, lookback=3, confidence_levels=(0.95,)).iloc[0]
    changed_first = generate_historical_var_forecasts(changed, lookback=3, confidence_levels=(0.95,)).iloc[0]

    assert changed_first["historical_var"] == pytest.approx(original_first["historical_var"])
    assert changed_first["realized_loss"] != pytest.approx(original_first["realized_loss"])


def test_var_exceedance_uses_strict_loss_comparison() -> None:
    assert var_exceedance(-0.03, 0.02)
    assert not var_exceedance(-0.02, 0.02)


def test_kupiec_handles_expected_zero_and_all_exceedance_cases() -> None:
    statistic, p_value = kupiec_unconditional_coverage(5, 100, 0.05)
    zero_statistic, zero_p_value = kupiec_unconditional_coverage(0, 100, 0.05)
    all_statistic, all_p_value = kupiec_unconditional_coverage(100, 100, 0.05)

    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)
    assert np.isfinite([zero_statistic, zero_p_value, all_statistic, all_p_value]).all()
    assert zero_statistic > 0.0
    assert all_statistic > zero_statistic


def test_insufficient_lookback_returns_empty_forecasts_and_zero_count_summary() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "model": "sample",
            "portfolio_return": [0.01, -0.01, 0.02],
        }
    )

    forecasts = generate_historical_var_forecasts(daily, lookback=3)
    summary = summarize_var_backtest(forecasts, models=["sample"])

    assert forecasts.empty
    assert summary["var_forecasts"].tolist() == [0, 0]
    assert summary["var_exceedances"].tolist() == [0, 0]
    assert summary["kupiec_lr_uc"].isna().all()


def test_forecasts_have_unique_model_date_confidence_rows_and_duplicate_input_is_rejected() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    daily = pd.DataFrame(
        [
            {"date": date, "model": model, "portfolio_return": 0.01 * (index - 1)}
            for model in ["sample", "ewma"]
            for index, date in enumerate(dates)
        ]
    )

    forecasts = generate_historical_var_forecasts(daily, lookback=3, confidence_levels=DEFAULT_CONFIDENCE_LEVELS)
    assert not forecasts.duplicated(["date", "model", "confidence_level"]).any()
    validate_daily_var_forecasts(forecasts, lookback=3)

    duplicate = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate model/date"):
        validate_daily_portfolio_returns(duplicate)
