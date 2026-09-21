import numpy as np
import pandas as pd
import pytest

from cross_sector_min_var.optimizer import (
    DEFAULT_SECTOR_CAP,
    DEFAULT_STOCK_CAP,
    equal_weight_portfolio,
    minimum_variance_weights,
    portfolio_variance,
    sector_weights,
    validate_optimization_result,
)
from cross_sector_min_var.stage3 import (
    build_stage3_diagnostics,
    covariance_estimates,
)


@pytest.fixture(scope="module")
def returns_window(returns_panel: pd.DataFrame) -> pd.DataFrame:
    return returns_panel.tail(504)


@pytest.fixture(scope="module")
def sectors(sector_map: pd.Series) -> pd.Series:
    return sector_map


@pytest.mark.parametrize("model", ["sample", "ewma", "ledoit_wolf"])
def test_optimized_portfolios_are_feasible_and_improve_on_equal_weight(returns_window, sectors, model) -> None:
    covariance = covariance_estimates(returns_window)[model]
    result = minimum_variance_weights(covariance, sectors)
    validate_optimization_result(result, covariance, sectors)

    assert result.success
    assert list(result.weights.index) == list(returns_window.columns)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (result.weights >= -1e-8).all()
    assert (result.weights <= DEFAULT_STOCK_CAP + 1e-8).all()
    assert (sector_weights(result.weights, sectors) <= DEFAULT_SECTOR_CAP + 1e-8).all()
    assert np.isfinite(result.objective_value)

    equal_weight = equal_weight_portfolio(returns_window.columns)
    assert portfolio_variance(result.weights, covariance) <= portfolio_variance(equal_weight, covariance) + 1e-10


def test_malformed_ticker_sector_alignment_raises_clear_error(returns_window, sectors) -> None:
    covariance = covariance_estimates(returns_window)["sample"]
    malformed_sectors = sectors.drop(sectors.index[0])

    with pytest.raises(ValueError, match="Sector map ticker mismatch"):
        minimum_variance_weights(covariance, malformed_sectors)


def test_malformed_covariance_label_alignment_raises_clear_error(returns_window, sectors) -> None:
    covariance = covariance_estimates(returns_window)["sample"]
    malformed_covariance = covariance.copy()
    malformed_covariance.columns = list(reversed(malformed_covariance.columns))

    with pytest.raises(ValueError, match="Covariance labels"):
        minimum_variance_weights(malformed_covariance, sectors)


def test_stage3_diagnostics_have_requested_compact_outputs(returns_window, sectors) -> None:
    weights, diagnostics, report = build_stage3_diagnostics(returns_window, sectors)

    assert weights.columns.tolist() == ["ticker", "sector", "sample", "ewma", "ledoit_wolf", "equal_weight"]
    assert weights.shape == (22, 6)
    for model in ["sample", "ewma", "ledoit_wolf", "equal_weight"]:
        assert weights[model].sum() == pytest.approx(1.0, abs=1e-8)

    assert diagnostics["model"].tolist() == ["sample", "ewma", "ledoit_wolf"]
    assert (diagnostics["daily_predicted_variance"] <= diagnostics["equal_weight_daily_predicted_variance"] + 1e-10).all()
    assert (diagnostics["hhi"] > 0.0).all()
    assert (diagnostics["effective_holdings"] > 0.0).all()
    assert set(report["l1_weight_distances"]) == {
        "sample_vs_ewma",
        "sample_vs_ledoit_wolf",
        "ewma_vs_ledoit_wolf",
    }
