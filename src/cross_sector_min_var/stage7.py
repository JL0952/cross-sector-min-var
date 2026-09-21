"""Stage 7 historical VaR and Expected Shortfall analysis of Stage 4 returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from cross_sector_min_var.backtest import MODEL_ORDER
from cross_sector_min_var.risk import (
    DEFAULT_CONFIDENCE_LEVELS,
    DEFAULT_RISK_LOOKBACK,
    generate_historical_var_forecasts,
    summarize_var_backtest,
    validate_daily_portfolio_returns,
)


MODEL_LABELS = {
    "sample": "Sample",
    "ewma": "EWMA",
    "ledoit_wolf": "Ledoit-Wolf",
    "equal_weight": "Equal Weight",
}


def load_stage4_daily_returns(root: Path) -> pd.DataFrame:
    """Load and validate the fixed Stage 4 gross daily-return artifact."""
    root = root.resolve()
    daily_returns = pd.read_csv(
        root / "results" / "portfolio_returns" / "stage4_daily_portfolio_returns.csv",
        parse_dates=["date"],
    )
    validate_daily_portfolio_returns(daily_returns, expected_models=MODEL_ORDER)
    return daily_returns


def _format_dates(axis) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(alpha=0.25)


def create_var_figure(daily_var: pd.DataFrame, output_path: Path) -> None:
    """Plot 99% historical VaR against realized losses for each Stage 4 portfolio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for axis, model in zip(axes, MODEL_ORDER):
        data = daily_var.loc[
            (daily_var["model"] == model) & (daily_var["confidence_level"] == 0.99)
        ].sort_values("date")
        axis.plot(data["date"], data["realized_loss"], label="Realized loss", linewidth=0.8, alpha=0.75)
        axis.plot(data["date"], data["historical_var"], label="99% historical VaR", linewidth=1.2)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(MODEL_LABELS[model])
        axis.set_ylabel("Loss")
        axis.legend(loc="upper right", fontsize=8)
        _format_dates(axis)
    figure.suptitle("Rolling 99% historical VaR versus realized daily losses")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_stage7(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create auditable tail-risk forecasts, coverage diagnostics, and one figure."""
    root = root.resolve()
    daily_returns = load_stage4_daily_returns(root)
    daily_var = generate_historical_var_forecasts(
        daily_returns,
        lookback=DEFAULT_RISK_LOOKBACK,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    )
    summary = summarize_var_backtest(
        daily_var,
        models=MODEL_ORDER,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    )

    daily_var_path = root / "results" / "risk" / "stage7_daily_var.csv"
    summary_path = root / "results" / "metrics" / "stage7_var_backtest_summary.csv"
    figure_path = root / "results" / "figures" / "stage7_rolling_99pct_var_vs_realized_losses.png"
    daily_var_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    daily_var.to_csv(daily_var_path, index=False)
    summary.to_csv(summary_path, index=False)
    create_var_figure(daily_var, figure_path)

    return daily_var, summary, {
        "source_returns_path": "results/portfolio_returns/stage4_daily_portfolio_returns.csv",
        "lookback_observations": DEFAULT_RISK_LOOKBACK,
        "confidence_levels": list(DEFAULT_CONFIDENCE_LEVELS),
        "quantile_method": "higher empirical order statistic",
        "loss_convention": "loss = -portfolio_return; VaR and ES are not floored at zero",
        "first_forecast_date": daily_var["date"].min().date().isoformat() if not daily_var.empty else None,
        "daily_var_path": str(daily_var_path.relative_to(root)),
        "summary_path": str(summary_path.relative_to(root)),
        "figure_path": str(figure_path.relative_to(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical VaR and Expected Shortfall analysis on Stage 4 returns.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    args = parser.parse_args()

    _, summary, report = run_stage7(args.root)
    report["var_backtest_summary"] = summary.to_dict(orient="records")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
