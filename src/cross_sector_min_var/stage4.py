"""Stage 4 rolling monthly out-of-sample covariance-model backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cross_sector_min_var.backtest import (
    DEFAULT_LOOKBACK,
    MODEL_ORDER,
    build_rebalance_schedule,
    forecast_error_metrics,
    one_sided_turnover,
    realized_annualized_volatility,
    reindex_weights,
    simulate_drifted_holding,
    validate_backtest_outputs,
)
from cross_sector_min_var.optimizer import (
    equal_weight_portfolio,
    minimum_variance_weights,
    predicted_risk_metrics,
    validate_optimization_result,
)
from cross_sector_min_var.stage2 import DEFAULT_LAMBDA
from cross_sector_min_var.stage3 import covariance_estimates, load_sector_map


OPTIMIZED_MODELS = ["sample", "ewma", "ledoit_wolf"]


def rolling_backtest(
    returns: pd.DataFrame,
    sector_map: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
    lambda_: float = DEFAULT_LAMBDA,
    schedule: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a causally ordered monthly GMV backtest with explicit daily weight drift."""
    schedule = build_rebalance_schedule(returns, lookback) if schedule is None else schedule.copy()
    ticker_order = returns.columns.tolist()
    equal_weight = equal_weight_portfolio(ticker_order)

    forecast_records: list[dict[str, object]] = []
    target_weight_records: list[dict[str, object]] = []
    daily_records: list[dict[str, object]] = []
    pretrade_by_model: dict[str, pd.Series] = {}

    for _, period in schedule.iterrows():
        rebalance_date = pd.Timestamp(period["rebalance_date"])
        estimation_start = pd.Timestamp(period["estimation_start"])
        estimation_end = pd.Timestamp(period["estimation_end"])
        holding_start = pd.Timestamp(period["holding_start"])
        holding_end = pd.Timestamp(period["holding_end"])

        estimation_returns = returns.loc[estimation_start:estimation_end]
        holding_returns = returns.loc[holding_start:holding_end]
        if len(estimation_returns) != lookback:
            raise ValueError("Backtest estimation window length differs from lookback.")
        if not (estimation_end <= rebalance_date < holding_start):
            raise ValueError("Backtest date ordering violates the no-look-ahead requirement.")

        covariances = covariance_estimates(estimation_returns, lambda_)
        model_targets: dict[str, pd.Series] = {}
        predicted_volatilities: dict[str, float] = {}
        optimizer_success: dict[str, Optional[bool]] = {}

        for model, covariance in covariances.items():
            optimized = minimum_variance_weights(covariance, sector_map)
            validate_optimization_result(optimized, covariance, sector_map)
            target = reindex_weights(optimized.weights, ticker_order)
            model_targets[model] = target
            predicted_volatilities[model] = predicted_risk_metrics(target, covariance)[
                "annualized_predicted_volatility"
            ]
            optimizer_success[model] = optimized.success

        model_targets["equal_weight"] = equal_weight.copy()
        predicted_volatilities["equal_weight"] = np.nan
        optimizer_success["equal_weight"] = None

        for model in MODEL_ORDER:
            target = model_targets[model]
            pretrade = pretrade_by_model.get(model)
            turnover = np.nan if pretrade is None else one_sided_turnover(target, pretrade)
            daily_returns, end_weights = simulate_drifted_holding(target, holding_returns)
            realized_vol = realized_annualized_volatility(daily_returns)

            if model in OPTIMIZED_MODELS:
                errors = forecast_error_metrics(predicted_volatilities[model], realized_vol)
            else:
                errors = {"forecast_error": np.nan, "absolute_error": np.nan, "squared_error": np.nan}

            forecast_records.append(
                {
                    "rebalance_date": rebalance_date,
                    "estimation_start": estimation_start,
                    "estimation_end": estimation_end,
                    "holding_start": holding_start,
                    "holding_end": holding_end,
                    "holding_days": len(holding_returns),
                    "estimation_observations": len(estimation_returns),
                    "model": model,
                    "predicted_vol": predicted_volatilities[model],
                    "realized_vol": realized_vol,
                    **errors,
                    "turnover": turnover,
                    "optimizer_success": optimizer_success[model],
                }
            )
            for ticker in ticker_order:
                target_weight_records.append(
                    {
                        "rebalance_date": rebalance_date,
                        "model": model,
                        "ticker": ticker,
                        "sector": sector_map.loc[ticker],
                        "target_weight": float(target.loc[ticker]),
                        "pretrade_weight": np.nan if pretrade is None else float(pretrade.loc[ticker]),
                    }
                )
            daily_records.extend(
                {
                    "date": date,
                    "model": model,
                    "portfolio_return": float(portfolio_return),
                }
                for date, portfolio_return in daily_returns.items()
            )
            pretrade_by_model[model] = end_weights

    forecasts = pd.DataFrame(forecast_records)
    weights = pd.DataFrame(target_weight_records)
    daily_portfolio_returns = pd.DataFrame(daily_records)
    validate_backtest_outputs(schedule, forecasts, weights, daily_portfolio_returns, lookback)
    return schedule, forecasts, weights, daily_portfolio_returns


def _maximum_drawdown(portfolio_returns: pd.Series) -> float:
    wealth = (1.0 + portfolio_returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def full_period_metrics(
    forecasts: pd.DataFrame, daily_portfolio_returns: pd.DataFrame
) -> pd.DataFrame:
    """Calculate OOS geometric-return and risk metrics from continuous daily returns."""
    records = []
    for model in MODEL_ORDER:
        daily = daily_portfolio_returns.loc[daily_portfolio_returns["model"] == model].sort_values("date")
        returns = daily["portfolio_return"]
        compounded_growth = float((1.0 + returns).prod())
        if compounded_growth <= 0.0:
            raise ValueError(f"Non-positive compounded growth for {model}.")
        annualized_return = float(compounded_growth ** (252.0 / len(returns)) - 1.0)
        annualized_volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
        model_forecasts = forecasts.loc[forecasts["model"] == model]
        record: dict[str, object] = {
            "model": model,
            "annualized_return": annualized_return,
            "annualized_realized_volatility": annualized_volatility,
            "sharpe_ratio_rf0": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0))
            if annualized_volatility > 0.0
            else np.nan,
            "maximum_drawdown": _maximum_drawdown(returns),
            "average_monthly_turnover": float(model_forecasts["turnover"].dropna().mean()),
            "mean_realized_volatility": float(model_forecasts["realized_vol"].mean()),
        }
        if model in OPTIMIZED_MODELS:
            errors = model_forecasts["forecast_error"]
            record.update(
                {
                    "mae": float(model_forecasts["absolute_error"].mean()),
                    "rmse": float(np.sqrt(model_forecasts["squared_error"].mean())),
                    "bias": float(errors.mean()),
                    "mean_predicted_volatility": float(model_forecasts["predicted_vol"].mean()),
                    "underprediction_frequency": float((errors < 0.0).mean()),
                }
            )
        else:
            record.update(
                {
                    "mae": np.nan,
                    "rmse": np.nan,
                    "bias": np.nan,
                    "mean_predicted_volatility": np.nan,
                    "underprediction_frequency": np.nan,
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def _manual_audit(
    schedule: pd.DataFrame, forecasts: pd.DataFrame, weights: pd.DataFrame
) -> dict[str, object]:
    """Create an auditable middle-of-sample rebalance record without future inputs."""
    audit_position = len(schedule) // 2
    period = schedule.iloc[audit_position]
    next_period = schedule.iloc[audit_position + 1]
    rebalance_date = pd.Timestamp(period["rebalance_date"])
    next_rebalance_date = pd.Timestamp(next_period["rebalance_date"])
    audit_models = {}

    for model in MODEL_ORDER:
        current_forecast = forecasts.loc[
            (forecasts["rebalance_date"] == rebalance_date) & (forecasts["model"] == model)
        ].iloc[0]
        current_target = weights.loc[
            (weights["rebalance_date"] == rebalance_date) & (weights["model"] == model),
            ["ticker", "target_weight"],
        ]
        next_pretrade = weights.loc[
            (weights["rebalance_date"] == next_rebalance_date) & (weights["model"] == model),
            ["ticker", "pretrade_weight"],
        ]
        next_turnover = forecasts.loc[
            (forecasts["rebalance_date"] == next_rebalance_date) & (forecasts["model"] == model),
            "turnover",
        ].iloc[0]
        audit_models[model] = {
            "target_weights": {
                row.ticker: float(row.target_weight) for row in current_target.itertuples(index=False)
            },
            "predicted_vol": None if pd.isna(current_forecast.predicted_vol) else float(current_forecast.predicted_vol),
            "realized_vol": float(current_forecast.realized_vol),
            "pretrade_weights_at_next_rebalance": {
                row.ticker: float(row.pretrade_weight) for row in next_pretrade.itertuples(index=False)
            },
            "turnover_into_next_target": float(next_turnover),
        }

    return {
        "rebalance_date": rebalance_date.date().isoformat(),
        "estimation_start": pd.Timestamp(period["estimation_start"]).date().isoformat(),
        "estimation_end": pd.Timestamp(period["estimation_end"]).date().isoformat(),
        "covariance_estimation_date": pd.Timestamp(period["estimation_end"]).date().isoformat(),
        "holding_start": pd.Timestamp(period["holding_start"]).date().isoformat(),
        "holding_end": pd.Timestamp(period["holding_end"]).date().isoformat(),
        "holding_days": int(period["holding_days"]),
        "next_rebalance_date": next_rebalance_date.date().isoformat(),
        "models": audit_models,
    }


def run_stage4(
    root: Path,
    lookback: int = DEFAULT_LOOKBACK,
    lambda_: float = DEFAULT_LAMBDA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run Stage 4 and save only tidy forecasts, targets, returns, and summary metrics."""
    root = root.resolve()
    returns = pd.read_parquet(root / "data" / "processed" / "daily_returns.parquet")
    sector_map = load_sector_map(root / "data" / "metadata" / "universe.csv", returns.columns.tolist())
    schedule, forecasts, weights, daily_returns = rolling_backtest(
        returns, sector_map, lookback=lookback, lambda_=lambda_
    )
    summary = full_period_metrics(forecasts, daily_returns)

    forecast_path = root / "results" / "forecasts" / "stage4_forecasts.csv"
    weight_path = root / "results" / "weights" / "stage4_weights.csv"
    returns_path = root / "results" / "portfolio_returns" / "stage4_daily_portfolio_returns.csv"
    metrics_path = root / "results" / "metrics" / "stage4_summary_metrics.csv"
    for path, frame in [
        (forecast_path, forecasts),
        (weight_path, weights),
        (returns_path, daily_returns),
        (metrics_path, summary),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    report = {
        "total_rebalances": len(schedule),
        "first_rebalance_date": pd.Timestamp(schedule.iloc[0]["rebalance_date"]).date().isoformat(),
        "last_rebalance_date": pd.Timestamp(schedule.iloc[-1]["rebalance_date"]).date().isoformat(),
        "out_of_sample_start": pd.Timestamp(daily_returns["date"].min()).date().isoformat(),
        "out_of_sample_end": pd.Timestamp(daily_returns["date"].max()).date().isoformat(),
        "timeline_example": {
            key: pd.Timestamp(schedule.iloc[len(schedule) // 2][key]).date().isoformat()
            for key in ["estimation_start", "estimation_end", "rebalance_date", "holding_start", "holding_end"]
        },
        "manual_audit": _manual_audit(schedule, forecasts, weights),
        "forecast_path": str(forecast_path.relative_to(root)),
        "weights_path": str(weight_path.relative_to(root)),
        "returns_path": str(returns_path.relative_to(root)),
        "metrics_path": str(metrics_path.relative_to(root)),
    }
    return schedule, forecasts, weights, daily_returns, summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rolling out-of-sample covariance backtest.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Historical daily-return observations.")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA, help="EWMA decay parameter.")
    args = parser.parse_args()

    _, _, _, _, summary, report = run_stage4(args.root, lookback=args.lookback, lambda_=args.lambda_)
    report["summary_metrics"] = summary.to_dict(orient="records")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
