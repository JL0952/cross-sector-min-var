"""Stage 1 data acquisition, processing, and quality checks.

The module deliberately contains no covariance estimation or portfolio logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_START = "2013-01-01"
# yfinance treats the end date as exclusive; this includes 2026-08-31.
DEFAULT_END_EXCLUSIVE = "2026-09-01"
RAW_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


@dataclass(frozen=True)
class Stage1Paths:
    """All local paths used by the Stage 1 pipeline."""

    root: Path
    universe: Path
    raw_prices: Path
    adjusted_close: Path
    daily_returns: Path

    @classmethod
    def from_root(cls, root: Path) -> "Stage1Paths":
        root = root.resolve()
        return cls(
            root=root,
            universe=root / "data" / "metadata" / "universe.csv",
            raw_prices=root / "data" / "raw" / "yahoo_prices.parquet",
            adjusted_close=root / "data" / "processed" / "adjusted_close.parquet",
            daily_returns=root / "data" / "processed" / "daily_returns.parquet",
        )


@dataclass(frozen=True)
class QualityReport:
    """Audit values printed after a successful Stage 1 run."""

    downloaded_ticker_count: int
    raw_date_range: dict[str, str]
    final_common_date_range: dict[str, str]
    return_rows_before_cleaning: int
    return_rows_removed: int
    missing_observations_per_ticker: dict[str, int]
    final_adjusted_close_shape: tuple[int, int]
    final_return_shape: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        report = asdict(self)
        # JSON has no tuple type; arrays make printed dataframe shapes explicit.
        report["final_adjusted_close_shape"] = list(self.final_adjusted_close_shape)
        report["final_return_shape"] = list(self.final_return_shape)
        return report


def load_universe(path: Path) -> pd.DataFrame:
    """Load and validate the study's fixed asset universe."""
    universe = pd.read_csv(path)
    required_columns = ["ticker", "company", "sector"]
    if list(universe.columns) != required_columns:
        raise ValueError(f"Universe columns must be exactly {required_columns}.")
    if len(universe) != 22:
        raise ValueError(f"Universe must contain exactly 22 stocks; found {len(universe)}.")
    if universe["ticker"].isna().any() or universe["ticker"].duplicated().any():
        raise ValueError("Universe tickers must be present and unique.")
    if universe[["company", "sector"]].isna().any().any():
        raise ValueError("Universe company and sector values must be present.")
    return universe


def download_yahoo_prices(
    tickers: Sequence[str], start: str = DEFAULT_START, end_exclusive: str = DEFAULT_END_EXCLUSIVE
) -> pd.DataFrame:
    """Download one raw Yahoo dataset and normalize it to date/ticker rows.

    `auto_adjust=False` preserves Yahoo's raw OHLC fields while retaining the
    separately supplied Adjusted Close column needed for return construction.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError("yfinance is required; install the project dependencies first.") from exc

    ticker_list = list(tickers)
    downloaded = yf.download(
        tickers=ticker_list,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")
    if not isinstance(downloaded.columns, pd.MultiIndex):
        raise RuntimeError("Expected yfinance multi-level columns for a multi-ticker download.")

    column_levels = [set(downloaded.columns.get_level_values(level)) for level in range(downloaded.columns.nlevels)]
    ticker_level = max(range(downloaded.columns.nlevels), key=lambda level: len(set(ticker_list) & column_levels[level]))
    if not set(ticker_list).issubset(column_levels[ticker_level]):
        missing = sorted(set(ticker_list).difference(column_levels[ticker_level]))
        raise RuntimeError(f"Yahoo Finance response omitted ticker columns: {missing}")

    pieces: list[pd.DataFrame] = []
    field_names = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    for ticker in ticker_list:
        ticker_frame = downloaded.xs(ticker, axis=1, level=ticker_level, drop_level=True)
        if isinstance(ticker_frame.columns, pd.MultiIndex):
            ticker_frame.columns = ticker_frame.columns.get_level_values(0)
        normalized = ticker_frame.rename(columns=field_names).reindex(columns=list(field_names.values()))
        normalized.index = pd.to_datetime(normalized.index).normalize()
        normalized.index.name = "date"
        normalized = normalized.reset_index()
        normalized.insert(1, "ticker", ticker)
        pieces.append(normalized)

    raw = pd.concat(pieces, ignore_index=True)[RAW_COLUMNS]
    raw = raw.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    if raw["adj_close"].notna().sum() == 0:
        raise RuntimeError("Yahoo Finance returned no Adjusted Close values.")
    return raw


def build_adjusted_close_panel(raw_prices: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Pivot normalized raw records into the unfilled adjusted-close panel."""
    required = {"date", "ticker", "adj_close"}
    missing = required.difference(raw_prices.columns)
    if missing:
        raise ValueError(f"Raw prices are missing required columns: {sorted(missing)}")
    if raw_prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Raw prices contain duplicate date/ticker observations.")

    prices = raw_prices.pivot(index="date", columns="ticker", values="adj_close")
    prices.index = pd.to_datetime(prices.index).normalize()
    prices.index.name = "date"
    prices = prices.reindex(columns=list(tickers)).sort_index()
    if prices.empty:
        raise ValueError("Adjusted-close panel is empty.")
    return prices


def build_complete_return_panel(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Calculate unfilled simple returns and remove incomplete common dates."""
    unfiltered_returns = prices.pct_change(fill_method=None)
    incomplete_rows = unfiltered_returns.isna().any(axis=1)
    complete_returns = unfiltered_returns.loc[~incomplete_rows].copy()
    complete_prices = prices.reindex(complete_returns.index).copy()
    if complete_returns.empty:
        raise ValueError("No complete common return observations remain after quality filtering.")
    return complete_prices, complete_returns, len(unfiltered_returns), int(incomplete_rows.sum())


def make_quality_report(
    raw_prices: pd.DataFrame,
    unfiltered_prices: pd.DataFrame,
    complete_prices: pd.DataFrame,
    complete_returns: pd.DataFrame,
    return_rows_before_cleaning: int,
    return_rows_removed: int,
) -> QualityReport:
    """Validate Stage 1 outputs and collect reproducible data-quality results."""
    if raw_prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Data-quality failure: duplicate raw date/ticker rows.")
    if raw_prices["date"].isna().any() or raw_prices["ticker"].isna().any():
        raise ValueError("Data-quality failure: raw date or ticker is missing.")
    if (raw_prices.loc[raw_prices["adj_close"].notna(), "adj_close"] <= 0).any():
        raise ValueError("Data-quality failure: non-positive adjusted close found.")
    if complete_prices.isna().any().any() or complete_returns.isna().any().any():
        raise ValueError("Data-quality failure: incomplete observations remain in final panels.")
    if not complete_prices.index.equals(complete_returns.index):
        raise ValueError("Data-quality failure: final price and return indices differ.")
    if list(complete_prices.columns) != list(complete_returns.columns):
        raise ValueError("Data-quality failure: final price and return columns differ.")

    downloaded = int(raw_prices.loc[raw_prices["adj_close"].notna(), "ticker"].nunique())
    if downloaded != len(unfiltered_prices.columns):
        raise ValueError(
            f"Data-quality failure: adjusted data are missing for {len(unfiltered_prices.columns) - downloaded} ticker(s)."
        )
    raw_dates = pd.to_datetime(raw_prices.loc[raw_prices["adj_close"].notna(), "date"])
    return QualityReport(
        downloaded_ticker_count=downloaded,
        raw_date_range={"start": raw_dates.min().date().isoformat(), "end": raw_dates.max().date().isoformat()},
        final_common_date_range={
            "start": complete_returns.index.min().date().isoformat(),
            "end": complete_returns.index.max().date().isoformat(),
        },
        return_rows_before_cleaning=return_rows_before_cleaning,
        return_rows_removed=return_rows_removed,
        missing_observations_per_ticker={ticker: int(count) for ticker, count in unfiltered_prices.isna().sum().items()},
        final_adjusted_close_shape=complete_prices.shape,
        final_return_shape=complete_returns.shape,
    )


def run_stage1(root: Path, start: str = DEFAULT_START, end_exclusive: str = DEFAULT_END_EXCLUSIVE) -> QualityReport:
    """Execute the complete Stage 1 data pipeline and persist its three artifacts."""
    paths = Stage1Paths.from_root(root)
    universe = load_universe(paths.universe)
    tickers = universe["ticker"].tolist()

    raw_prices = download_yahoo_prices(tickers, start=start, end_exclusive=end_exclusive)
    paths.raw_prices.parent.mkdir(parents=True, exist_ok=True)
    raw_prices.to_parquet(paths.raw_prices, index=False)

    unfiltered_prices = build_adjusted_close_panel(raw_prices, tickers)
    complete_prices, complete_returns, rows_before, rows_removed = build_complete_return_panel(unfiltered_prices)
    paths.adjusted_close.parent.mkdir(parents=True, exist_ok=True)
    complete_prices.to_parquet(paths.adjusted_close)
    complete_returns.to_parquet(paths.daily_returns)

    return make_quality_report(
        raw_prices,
        unfiltered_prices,
        complete_prices,
        complete_returns,
        rows_before,
        rows_removed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Yahoo Finance research data panels.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive Yahoo download date (YYYY-MM-DD).")
    parser.add_argument(
        "--end-exclusive",
        default=DEFAULT_END_EXCLUSIVE,
        help="Exclusive Yahoo download date (YYYY-MM-DD). Default includes 2026-08-31.",
    )
    args = parser.parse_args()
    report = run_stage1(args.root, start=args.start, end_exclusive=args.end_exclusive)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
