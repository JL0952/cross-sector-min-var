"""Stage 6 turnover-cost and regime-definition robustness analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cross_sector_min_var.analysis import (
    equal_weight_high_volatility_months,
    high_volatility_months,
    regime_forecast_metrics,
    transaction_cost_analysis,
)


def load_stage4_outputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the saved Stage 4 forecasts and daily OOS portfolio returns."""
    forecasts = pd.read_csv(
        root / "results" / "forecasts" / "stage4_forecasts.csv",
        parse_dates=["rebalance_date", "holding_start", "holding_end"],
    )
    daily_returns = pd.read_csv(
        root / "results" / "portfolio_returns" / "stage4_daily_portfolio_returns.csv",
        parse_dates=["date"],
    )
    return forecasts, daily_returns


def compare_regime_definitions(
    stage5_metrics: pd.DataFrame, equal_weight_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Keep the Stage 5 definition visible while reporting the equal-weight robustness check."""
    keys = ["model", "regime"]
    metrics = ["months", "mae", "rmse", "bias", "underprediction_frequency"]
    left = stage5_metrics[keys + metrics].rename(columns={column: f"stage5_{column}" for column in metrics})
    right = equal_weight_metrics[keys + metrics].rename(
        columns={column: f"equal_weight_reference_{column}" for column in metrics}
    )
    return left.merge(right, on=keys, validate="one_to_one")


def run_stage6(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Persist compact Stage 6 cost and robustness tables from saved Stage 4 results."""
    root = root.resolve()
    forecasts, daily_returns = load_stage4_outputs(root)

    cost_metrics = transaction_cost_analysis(forecasts, daily_returns)
    stage5_regimes, stage5_threshold = high_volatility_months(forecasts)
    stage5_metrics = regime_forecast_metrics(forecasts, stage5_regimes, stage5_threshold)
    equal_weight_regimes, equal_weight_threshold = equal_weight_high_volatility_months(forecasts)
    equal_weight_metrics = regime_forecast_metrics(forecasts, equal_weight_regimes, equal_weight_threshold)
    comparison = compare_regime_definitions(stage5_metrics, equal_weight_metrics)

    metrics_directory = root / "results" / "metrics"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "stage6_transaction_cost_metrics.csv": cost_metrics,
        "stage6_equal_weight_regime_metrics.csv": equal_weight_metrics,
        "stage6_regime_definition_comparison.csv": comparison,
    }
    for filename, frame in outputs.items():
        frame.to_csv(metrics_directory / filename, index=False)

    report = {
        "cost_convention": "Cost is charged once on each holding-period start: 2 * turnover * (bps / 10000).",
        "first_rebalance_cost": "zero because Stage 4 turnover is undefined without prior pre-trade weights",
        "stage5_high_volatility_threshold": stage5_threshold,
        "equal_weight_high_volatility_threshold": equal_weight_threshold,
        "stage5_high_volatility_months": int(stage5_regimes["high_volatility"].sum()),
        "equal_weight_high_volatility_months": int(equal_weight_regimes["high_volatility"].sum()),
    }
    return outputs, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate turnover costs and the regime-definition robustness check.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    args = parser.parse_args()

    outputs, report = run_stage6(args.root)
    report["transaction_cost_metrics"] = outputs["stage6_transaction_cost_metrics.csv"].to_dict(orient="records")
    report["equal_weight_regime_metrics"] = outputs["stage6_equal_weight_regime_metrics.csv"].to_dict(orient="records")
    report["regime_definition_comparison"] = outputs["stage6_regime_definition_comparison.csv"].to_dict(orient="records")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
