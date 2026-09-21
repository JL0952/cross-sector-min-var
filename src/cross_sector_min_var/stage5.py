"""Stage 5 analysis and figures for saved Stage 4 out-of-sample results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from cross_sector_min_var.analysis import (
    ALL_MODELS,
    OPTIMIZED_MODELS,
    concentration_metrics,
    forecast_misses,
    high_volatility_months,
    pairwise_weight_distances,
    regime_forecast_metrics,
    sector_allocations,
    summarize_pairwise_distances,
    turnover_summary,
    weight_stability,
)


MODEL_LABELS = {
    "sample": "Sample",
    "ewma": "EWMA",
    "ledoit_wolf": "Ledoit-Wolf",
    "equal_weight": "Equal Weight",
}


def model_label(model: str) -> str:
    """Return a presentation label for a stored model identifier."""
    return MODEL_LABELS[model]


def load_stage4_outputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four saved Stage 4 artifacts without recreating the backtest."""
    root = root.resolve()
    forecasts = pd.read_csv(
        root / "results" / "forecasts" / "stage4_forecasts.csv",
        parse_dates=["rebalance_date", "estimation_start", "estimation_end", "holding_start", "holding_end"],
    )
    weights = pd.read_csv(root / "results" / "weights" / "stage4_weights.csv", parse_dates=["rebalance_date"])
    daily_returns = pd.read_csv(
        root / "results" / "portfolio_returns" / "stage4_daily_portfolio_returns.csv",
        parse_dates=["date"],
    )
    summary_metrics = pd.read_csv(root / "results" / "metrics" / "stage4_summary_metrics.csv")
    return forecasts, weights, daily_returns, summary_metrics


def _format_dates(axis) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(alpha=0.25)


def create_figures(
    forecasts: pd.DataFrame,
    daily_returns: pd.DataFrame,
    pairwise_distances: pd.DataFrame,
    concentration: pd.DataFrame,
    figure_directory: Path,
) -> list[Path]:
    """Create the six saved research figures."""
    figure_directory.mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for axis, model in zip(axes, OPTIMIZED_MODELS):
        data = forecasts.loc[forecasts["model"] == model].sort_values("rebalance_date")
        axis.plot(data["rebalance_date"], data["predicted_vol"], label="Predicted", linewidth=1.4)
        axis.plot(data["rebalance_date"], data["realized_vol"], label="Realized", linewidth=1.2)
        axis.set_title(model_label(model))
        axis.set_ylabel("Annualized vol.")
        axis.legend(loc="upper right")
        _format_dates(axis)
    figure.suptitle("Predicted vs. realized holding-period volatility")
    figure.tight_layout()
    path = figure_directory / "predicted_vs_realized_volatility.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, model in zip(axes, OPTIMIZED_MODELS):
        data = forecasts.loc[forecasts["model"] == model].sort_values("rebalance_date")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(data["rebalance_date"], data["forecast_error"], linewidth=1.1)
        axis.set_title(model_label(model))
        axis.set_ylabel("Pred. − realized")
        _format_dates(axis)
    figure.suptitle("Forecast error through time")
    figure.tight_layout()
    path = figure_directory / "forecast_errors.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(11, 4))
    for model in ALL_MODELS:
        data = forecasts.loc[forecasts["model"] == model].sort_values("rebalance_date")
        axis.plot(data["rebalance_date"], data["turnover"], label=model_label(model), linewidth=1.1)
    axis.set_title("Monthly turnover")
    axis.set_ylabel("One-sided turnover")
    axis.legend(ncol=4)
    _format_dates(axis)
    figure.tight_layout()
    path = figure_directory / "monthly_turnover.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(11, 4))
    for column in ["sample_vs_ewma", "sample_vs_ledoit_wolf", "ewma_vs_ledoit_wolf"]:
        axis.plot(pairwise_distances["rebalance_date"], pairwise_distances[column], label=column.replace("_", " "))
    axis.set_title("Pairwise target-weight L1 distance")
    axis.set_ylabel("L1 distance")
    axis.legend(ncol=3, fontsize=8)
    _format_dates(axis)
    figure.tight_layout()
    path = figure_directory / "pairwise_target_weight_distances.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for model in ALL_MODELS:
        data = concentration.loc[concentration["model"] == model].sort_values("rebalance_date")
        axes[0].plot(data["rebalance_date"], data["hhi"], label=model_label(model), linewidth=1.1)
        axes[1].plot(data["rebalance_date"], data["effective_holdings"], label=model_label(model), linewidth=1.1)
    axes[0].set_title("Concentration through time")
    axes[0].set_ylabel("HHI")
    axes[1].set_ylabel("Effective holdings")
    axes[0].legend(ncol=4, fontsize=8)
    for axis in axes:
        _format_dates(axis)
    figure.tight_layout()
    path = figure_directory / "concentration.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(11, 4))
    for model in ALL_MODELS:
        data = daily_returns.loc[daily_returns["model"] == model].sort_values("date").copy()
        wealth = (1.0 + data["portfolio_return"]).cumprod()
        axis.plot(data["date"], wealth, label=model_label(model), linewidth=1.1)
    axis.set_title("Cumulative out-of-sample portfolio growth")
    axis.set_ylabel("Growth of $1")
    axis.legend(ncol=4, fontsize=8)
    _format_dates(axis)
    figure.tight_layout()
    path = figure_directory / "cumulative_oos_growth.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    return paths


def run_stage5(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Analyze Stage 4 outputs, persist compact Stage 5 artifacts, and create figures."""
    root = root.resolve()
    forecasts, weights, daily_returns, stage4_summary = load_stage4_outputs(root)

    regimes, threshold = high_volatility_months(forecasts)
    regime_metrics = regime_forecast_metrics(forecasts, regimes, threshold)
    misses = forecast_misses(forecasts)
    stability_summary, stock_volatility, target_changes = weight_stability(weights)
    pairwise_distances = pairwise_weight_distances(weights)
    pairwise_summary = summarize_pairwise_distances(pairwise_distances)
    turnover = turnover_summary(forecasts, target_changes)
    concentration_timeseries, concentration_summary = concentration_metrics(weights)
    sector_timeseries, sector_summary = sector_allocations(weights)

    metrics_directory = root / "results" / "metrics"
    figure_directory = root / "results" / "figures"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "stage5_regime_forecast_metrics.csv": regime_metrics,
        "stage5_largest_forecast_misses.csv": misses,
        "stage5_weight_stability.csv": stability_summary,
        "stage5_stock_weight_volatility.csv": stock_volatility,
        "stage5_turnover_summary.csv": turnover,
        "stage5_concentration_summary.csv": concentration_summary,
        "stage5_pairwise_weight_distances.csv": pairwise_distances,
        "stage5_pairwise_weight_distance_summary.csv": pairwise_summary,
        "stage5_sector_allocation_timeseries.csv": sector_timeseries,
        "stage5_sector_allocation_summary.csv": sector_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(metrics_directory / filename, index=False)

    figure_paths = create_figures(
        forecasts, daily_returns, pairwise_distances, concentration_timeseries, figure_directory
    )
    report = {
        "high_volatility_threshold": threshold,
        "high_volatility_months": int(regimes["high_volatility"].sum()),
        "remaining_months": int((~regimes["high_volatility"]).sum()),
        "figures": [str(path.relative_to(root)) for path in figure_paths],
        "largest_underprediction_months": misses.loc[misses["miss_type"] == "largest_underprediction"]
        .groupby("model", sort=False)
        .first()
        .reset_index()
        .to_dict(orient="records"),
        "largest_overprediction_months": misses.loc[misses["miss_type"] == "largest_overprediction"]
        .groupby("model", sort=False)
        .first()
        .reset_index()
        .to_dict(orient="records"),
        "stage4_benchmark_metrics": stage4_summary.to_dict(orient="records"),
    }
    return outputs, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved rolling out-of-sample results.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    args = parser.parse_args()

    outputs, report = run_stage5(args.root)
    report["regime_forecast_metrics"] = outputs["stage5_regime_forecast_metrics.csv"].to_dict(orient="records")
    report["pairwise_weight_distance_summary"] = outputs[
        "stage5_pairwise_weight_distance_summary.csv"
    ].to_dict(orient="records")
    report["turnover_summary"] = outputs["stage5_turnover_summary.csv"].to_dict(orient="records")
    report["concentration_summary"] = outputs["stage5_concentration_summary.csv"].to_dict(orient="records")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
