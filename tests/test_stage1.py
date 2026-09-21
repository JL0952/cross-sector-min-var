from pathlib import Path

import pandas as pd
import pytest

from cross_sector_min_var.stage1 import build_adjusted_close_panel, build_complete_return_panel, load_universe


def test_return_panel_uses_no_forward_fill_and_removes_incomplete_dates() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    raw = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "adj_close": [100.0, 101.0, 102.0, 103.0, 200.0, 201.0, None, 203.0],
        }
    )
    prices = build_adjusted_close_panel(raw, ["AAA", "BBB"])
    final_prices, returns, rows_before, rows_removed = build_complete_return_panel(prices)

    assert rows_before == 4
    assert rows_removed == 3
    assert final_prices.index.tolist() == [dates[1]]
    assert returns.index.tolist() == [dates[1]]
    assert returns.loc[dates[1], "AAA"] == pytest.approx(0.01)
    assert returns.loc[dates[1], "BBB"] == pytest.approx(0.005)


def test_adjusted_close_panel_preserves_unavailable_ticker_as_missing() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "AAA"],
            "adj_close": [10.0, 11.0],
        }
    )
    prices = build_adjusted_close_panel(raw, ["AAA", "BBB"])

    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices["BBB"].isna().all()


def test_universe_requires_all_22_unique_tickers(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    pd.DataFrame({"ticker": ["AAA"], "company": ["Example"], "sector": ["Sector"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="exactly 22"):
        load_universe(path)


def test_committed_universe_has_two_stocks_per_sector() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "metadata" / "universe.csv"
    universe = load_universe(path)

    assert universe["sector"].nunique() == 11
    assert universe.groupby("sector")["ticker"].size().eq(2).all()
