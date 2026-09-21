import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from cross_sector_min_var.covariance import (
    ewma_covariance,
    ewma_weights,
    ledoit_wolf_covariance,
    sample_covariance,
    validate_covariance_matrix,
)
from cross_sector_min_var.stage2 import (
    DEFAULT_WINDOW_SIZE,
    build_stage2_diagnostics,
    load_latest_return_window,
)


@pytest.fixture(scope="module")
def returns_window(returns_panel: pd.DataFrame) -> pd.DataFrame:
    return returns_panel.tail(DEFAULT_WINDOW_SIZE)


def assert_common_covariance_properties(matrix: pd.DataFrame, returns_window: pd.DataFrame) -> None:
    validate_covariance_matrix(matrix, returns_window.columns)
    values = matrix.to_numpy(dtype=float)
    assert matrix.shape == (22, 22)
    assert list(matrix.index) == list(returns_window.columns)
    assert list(matrix.columns) == list(returns_window.columns)
    np.testing.assert_allclose(values, values.T, rtol=1e-10, atol=1e-12)
    assert (np.diag(values) >= 0.0).all()
    assert np.isfinite(values).all()
    assert np.issubdtype(values.dtype, np.number)
    assert np.linalg.eigvalsh(values).min() >= -1e-12


@pytest.mark.parametrize(
    "estimator",
    [
        sample_covariance,
        ewma_covariance,
        lambda returns: ledoit_wolf_covariance(returns)[0],
    ],
)
def test_all_estimators_return_valid_labeled_daily_covariance(returns_window, estimator) -> None:
    assert_common_covariance_properties(estimator(returns_window), returns_window)


def test_sample_covariance_matches_pandas(returns_window) -> None:
    actual = sample_covariance(returns_window)
    expected = returns_window.cov()
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-15)


def test_ewma_weights_and_lambda_change_covariance(returns_window) -> None:
    weights = ewma_weights(len(returns_window), lambda_=0.94)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] > weights[0]

    covariance_94 = ewma_covariance(returns_window, lambda_=0.94)
    covariance_80 = ewma_covariance(returns_window, lambda_=0.80)
    assert not np.allclose(covariance_94.to_numpy(), covariance_80.to_numpy())
    assert_common_covariance_properties(covariance_94, returns_window)


def test_ledoit_wolf_matches_sklearn_and_exposes_shrinkage(returns_window) -> None:
    actual, shrinkage = ledoit_wolf_covariance(returns_window)
    reference = LedoitWolf().fit(returns_window.to_numpy(dtype=float))

    assert 0.0 <= shrinkage <= 1.0
    assert shrinkage == pytest.approx(reference.shrinkage_)
    np.testing.assert_allclose(actual.to_numpy(), reference.covariance_, rtol=1e-13, atol=1e-15)
    assert_common_covariance_properties(actual, returns_window)


def test_latest_window_and_diagnostic_use_exact_input(tmp_path, returns_panel: pd.DataFrame) -> None:
    path = tmp_path / "daily_returns.parquet"
    returns_panel.to_parquet(path)
    returns_window = load_latest_return_window(path)
    diagnostics, distances = build_stage2_diagnostics(returns_window)

    assert len(returns_window) == DEFAULT_WINDOW_SIZE
    assert returns_window.index[0] == returns_panel.index[-DEFAULT_WINDOW_SIZE]
    assert returns_window.index[-1] == returns_panel.index[-1]
    assert list(returns_window.columns) == list(returns_panel.columns)
    assert diagnostics["estimator"].tolist() == ["sample", "ewma", "ledoit_wolf"]
    assert (diagnostics["observations"] == DEFAULT_WINDOW_SIZE).all()
    assert (diagnostics["assets"] == 22).all()
    assert set(distances) == {
        "sample_vs_ewma",
        "sample_vs_ledoit_wolf",
        "ewma_vs_ledoit_wolf",
    }
    assert all(value > 0.0 for value in distances.values())
