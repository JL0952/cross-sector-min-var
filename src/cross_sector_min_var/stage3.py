"""Stage 3 constrained global minimum-variance portfolio diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from cross_sector_min_var.covariance import ewma_covariance, ledoit_wolf_covariance, sample_covariance
from cross_sector_min_var.optimizer import (
    DEFAULT_SECTOR_CAP,
    DEFAULT_STOCK_CAP,
    DEFAULT_TOLERANCE,
    equal_weight_portfolio,
    minimum_variance_weights,
    predicted_risk_metrics,
    sector_weights,
    validate_optimization_result,
)
from cross_sector_min_var.stage1 import load_universe
from cross_sector_min_var.stage2 import DEFAULT_LAMBDA, DEFAULT_WINDOW_SIZE, load_latest_return_window


def load_sector_map(path: Path, ticker_order: list[str]) -> pd.Series:
    """Load metadata and require exact ticker alignment with the return/covariance order."""
    universe = load_universe(path)
    if universe["ticker"].tolist() != ticker_order:
        raise ValueError("Universe ticker order must exactly match the return/covariance ticker order.")
    return pd.Series(universe["sector"].to_numpy(), index=universe["ticker"], name="sector")


def covariance_estimates(returns_window: pd.DataFrame, lambda_: float = DEFAULT_LAMBDA) -> dict[str, pd.DataFrame]:
    """Reuse the Stage 2 estimators to create the three model inputs for optimization."""
    ledoit_wolf, _ = ledoit_wolf_covariance(returns_window)
    return {
        "sample": sample_covariance(returns_window),
        "ewma": ewma_covariance(returns_window, lambda_),
        "ledoit_wolf": ledoit_wolf,
    }


def _sector_column_name(sector: str) -> str:
    return "sector_weight_" + "".join(character.lower() if character.isalnum() else "_" for character in sector).strip("_")


def _portfolio_record(
    model: str,
    optimized_weights: pd.Series,
    optimized_metrics: Mapping[str, float],
    equal_weight_metrics: Mapping[str, float],
    sector_map: pd.Series,
    max_stock_weight: float,
    max_sector_weight: float,
    tolerance: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create compact CSV fields plus richer terminal-report fields for one model."""
    sector_allocation = sector_weights(optimized_weights, sector_map)
    stock_cap_tickers = optimized_weights.index[optimized_weights >= max_stock_weight - tolerance].tolist()
    sector_cap_names = sector_allocation.index[sector_allocation >= max_sector_weight - tolerance].tolist()
    hhi = float(np.square(optimized_weights.to_numpy()).sum())

    record: dict[str, object] = {
        "model": model,
        **optimized_metrics,
        "equal_weight_daily_predicted_variance": equal_weight_metrics["daily_predicted_variance"],
        "equal_weight_daily_predicted_volatility": equal_weight_metrics["daily_predicted_volatility"],
        "equal_weight_annualized_predicted_volatility": equal_weight_metrics["annualized_predicted_volatility"],
        "largest_stock_weight": float(optimized_weights.max()),
        "smallest_stock_weight": float(optimized_weights.min()),
        "stocks_over_1pct": int((optimized_weights > 0.01).sum()),
        "stocks_effectively_zero": int((optimized_weights <= tolerance).sum()),
        "hhi": hhi,
        "effective_holdings": float(1.0 / hhi),
        "stock_cap_binding": bool(stock_cap_tickers),
        "stock_cap_binding_tickers": "|".join(stock_cap_tickers) or "<none>",
        "sector_cap_binding": bool(sector_cap_names),
        "sector_cap_binding_sectors": "|".join(sector_cap_names) or "<none>",
        "constraint_tolerance": tolerance,
    }
    record.update({_sector_column_name(sector): float(weight) for sector, weight in sector_allocation.items()})
    display = {
        "model": model,
        **record,
        "sector_weights": {sector: float(weight) for sector, weight in sector_allocation.items()},
        "stock_cap_binding_tickers": stock_cap_tickers,
        "sector_cap_binding_sectors": sector_cap_names,
    }
    return record, display


def build_stage3_diagnostics(
    returns_window: pd.DataFrame,
    sector_map: pd.Series,
    lambda_: float = DEFAULT_LAMBDA,
    max_stock_weight: float = DEFAULT_STOCK_CAP,
    max_sector_weight: float = DEFAULT_SECTOR_CAP,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Optimize all three models and produce the requested Stage 3 comparisons."""
    covariances = covariance_estimates(returns_window, lambda_)
    equal_weight = equal_weight_portfolio(returns_window.columns.tolist())
    optimized_by_model: dict[str, pd.Series] = {}
    metric_records: list[dict[str, object]] = []
    display_records: list[dict[str, object]] = []

    for model, covariance in covariances.items():
        optimized = minimum_variance_weights(
            covariance,
            sector_map,
            max_stock_weight=max_stock_weight,
            max_sector_weight=max_sector_weight,
        )
        validate_optimization_result(
            optimized,
            covariance,
            sector_map,
            max_stock_weight=max_stock_weight,
            max_sector_weight=max_sector_weight,
            tolerance=tolerance,
        )
        optimized_risk = predicted_risk_metrics(optimized.weights, covariance)
        equal_weight_risk = predicted_risk_metrics(equal_weight, covariance)
        if optimized_risk["daily_predicted_variance"] > equal_weight_risk["daily_predicted_variance"] + tolerance:
            raise ValueError(f"{model} optimized variance exceeds feasible equal-weight variance.")

        optimized_by_model[model] = optimized.weights
        record, display = _portfolio_record(
            model,
            optimized.weights,
            optimized_risk,
            equal_weight_risk,
            sector_map,
            max_stock_weight,
            max_sector_weight,
            tolerance,
        )
        metric_records.append(record)
        display_records.append(display)

    weight_table = pd.DataFrame(
        {
            "ticker": returns_window.columns,
            "sector": sector_map.reindex(returns_window.columns).to_numpy(),
            "sample": optimized_by_model["sample"].to_numpy(),
            "ewma": optimized_by_model["ewma"].to_numpy(),
            "ledoit_wolf": optimized_by_model["ledoit_wolf"].to_numpy(),
            "equal_weight": equal_weight.to_numpy(),
        }
    )
    l1_distances = {
        "sample_vs_ewma": float(np.abs(optimized_by_model["sample"] - optimized_by_model["ewma"]).sum()),
        "sample_vs_ledoit_wolf": float(
            np.abs(optimized_by_model["sample"] - optimized_by_model["ledoit_wolf"]).sum()
        ),
        "ewma_vs_ledoit_wolf": float(np.abs(optimized_by_model["ewma"] - optimized_by_model["ledoit_wolf"]).sum()),
    }
    report = {
        "window_start": returns_window.index.min().date().isoformat(),
        "window_end": returns_window.index.max().date().isoformat(),
        "observations": len(returns_window),
        "assets": returns_window.shape[1],
        "constraint_tolerance": tolerance,
        "portfolio_diagnostics": display_records,
        "l1_weight_distances": l1_distances,
    }
    return weight_table, pd.DataFrame(metric_records), report


def run_stage3(
    root: Path,
    observations: int = DEFAULT_WINDOW_SIZE,
    lambda_: float = DEFAULT_LAMBDA,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run the fixed-window Stage 3 diagnostic and save the two compact outputs."""
    root = root.resolve()
    returns_window = load_latest_return_window(
        root / "data" / "processed" / "daily_returns.parquet",
        observations=observations,
    )
    sector_map = load_sector_map(root / "data" / "metadata" / "universe.csv", returns_window.columns.tolist())
    weight_table, diagnostics, report = build_stage3_diagnostics(returns_window, sector_map, lambda_)

    weights_path = root / "results" / "weights" / "stage3_weights.csv"
    metrics_path = root / "results" / "metrics" / "stage3_portfolio_diagnostics.csv"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    weight_table.to_csv(weights_path, index=False)
    diagnostics.to_csv(metrics_path, index=False)

    report["weights_path"] = str(weights_path.relative_to(root))
    report["metrics_path"] = str(metrics_path.relative_to(root))
    return weight_table, diagnostics, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-window constrained-GMV diagnostics.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    parser.add_argument("--observations", type=int, default=DEFAULT_WINDOW_SIZE, help="Latest return observations to use.")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA, help="EWMA decay parameter.")
    args = parser.parse_args()

    weights, diagnostics, report = run_stage3(args.root, observations=args.observations, lambda_=args.lambda_)
    report["weight_comparison"] = weights.to_dict(orient="records")
    report["diagnostics_table"] = diagnostics.to_dict(orient="records")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
