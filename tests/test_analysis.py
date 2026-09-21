import numpy as np
import pandas as pd
import pytest

from cross_sector_min_var.analysis import (
    apply_monthly_transaction_costs,
    consecutive_target_weight_changes,
    equal_weight_high_volatility_months,
    effective_holdings,
    hhi,
    high_volatility_months,
    l1_weight_distance,
    sector_allocations,
    transaction_cost_analysis,
    transaction_cost_fraction,
)


def test_hhi_and_effective_holdings() -> None:
    weights = pd.Series([0.5, 0.5], index=["AAA", "BBB"])

    assert hhi(weights) == pytest.approx(0.5)
    assert effective_holdings(weights) == pytest.approx(2.0)


def test_pairwise_l1_distance_requires_aligned_labels() -> None:
    left = pd.Series([0.6, 0.4], index=["AAA", "BBB"])
    right = pd.Series([0.5, 0.5], index=["AAA", "BBB"])

    assert l1_weight_distance(left, right) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="identical ticker labels"):
        l1_weight_distance(left, right.reindex(["BBB", "AAA"]))


def test_common_high_volatility_classification_uses_75th_percentile() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    forecasts = pd.DataFrame(
        [
            {"rebalance_date": date, "model": model, "realized_vol": value}
            for date, value in zip(dates, [0.10, 0.20, 0.30, 0.40])
            for model in ["sample", "ewma", "ledoit_wolf"]
        ]
    )

    regimes, threshold = high_volatility_months(forecasts)

    assert threshold == pytest.approx(0.325)
    assert regimes["high_volatility"].tolist() == [False, False, False, True]


def test_equal_weight_high_volatility_classification_uses_equal_weight_only() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    forecasts = pd.DataFrame(
        [
            {"rebalance_date": date, "model": model, "realized_vol": value}
            for model in ["sample", "ewma", "ledoit_wolf"]
            for date, value in zip(dates, [0.40, 0.30, 0.20, 0.10])
        ]
        + [
            {"rebalance_date": date, "model": "equal_weight", "realized_vol": value}
            for date, value in zip(dates, [0.10, 0.20, 0.30, 0.40])
        ]
    )

    regimes, threshold = equal_weight_high_volatility_months(forecasts)

    assert threshold == pytest.approx(0.325)
    assert regimes["high_volatility"].tolist() == [False, False, False, True]


def test_transaction_cost_is_charged_once_on_holding_start() -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["sample", "sample"],
            "holding_start": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "turnover": [np.nan, 0.20],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-03-01"]),
            "model": ["sample", "sample", "sample"],
            "portfolio_return": [0.01, 0.02, 0.03],
        }
    )

    net = apply_monthly_transaction_costs(forecasts, daily, cost_bps=10.0)

    assert transaction_cost_fraction(pd.Series([0.20]), 10.0).iloc[0] == pytest.approx(0.0004)
    assert net["transaction_cost_fraction"].tolist() == pytest.approx([0.0, 0.0, 0.0004])
    assert net.loc[2, "net_portfolio_return"] == pytest.approx((1.03 * (1.0 - 0.0004)) - 1.0)


def test_transaction_cost_analysis_reports_each_requested_cost() -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["sample", "sample"],
            "holding_start": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "turnover": [np.nan, 0.10],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "model": ["sample", "sample"],
            "portfolio_return": [0.01, 0.01],
        }
    )

    metrics = transaction_cost_analysis(forecasts, daily, cost_bps_values=(5.0, 20.0))

    assert metrics["cost_bps_per_traded_dollar"].tolist() == [5.0, 20.0]
    assert metrics["cumulative_total_transaction_cost"].tolist() == pytest.approx([0.0001, 0.0004])
    assert (metrics["net_annualized_return"] < metrics["gross_annualized_return"]).all()


def test_consecutive_target_weight_change_uses_l1_difference() -> None:
    dates = [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    records = []
    for model in ["sample", "ewma", "ledoit_wolf", "equal_weight"]:
        for date, aaa_weight in zip(dates, [0.5, 0.6]):
            records.extend(
                [
                    {"rebalance_date": date, "model": model, "ticker": "AAA", "target_weight": aaa_weight},
                    {"rebalance_date": date, "model": model, "ticker": "BBB", "target_weight": 1.0 - aaa_weight},
                ]
            )
    changes = consecutive_target_weight_changes(pd.DataFrame(records))
    sample_changes = changes.loc[changes["model"] == "sample", "l1_target_weight_change"].tolist()

    assert np.isnan(sample_changes[0])
    assert sample_changes[1] == pytest.approx(0.2)


def test_sector_weight_aggregation_sums_target_weights() -> None:
    weights = pd.DataFrame(
        [
            {"rebalance_date": pd.Timestamp("2024-01-31"), "model": model, "sector": "Tech", "target_weight": 0.30},
            {"rebalance_date": pd.Timestamp("2024-01-31"), "model": model, "sector": "Tech", "target_weight": 0.20},
            {"rebalance_date": pd.Timestamp("2024-01-31"), "model": model, "sector": "Energy", "target_weight": 0.50},
        ]
        for model in ["sample", "ewma", "ledoit_wolf"]
    )
    flattened = pd.DataFrame([record for model_records in weights.to_numpy().tolist() for record in model_records])
    timeseries, summary = sector_allocations(flattened)

    assert timeseries.loc[(timeseries["model"] == "sample") & (timeseries["sector"] == "Tech"), "sector_weight"].iloc[0] == pytest.approx(0.5)
    assert set(summary["sector"]) == {"Tech", "Energy"}
