"""Deterministic analysis functions for existing Stage 4 out-of-sample outputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


OPTIMIZED_MODELS = ["sample", "ewma", "ledoit_wolf"]
ALL_MODELS = OPTIMIZED_MODELS + ["equal_weight"]
ZERO_WEIGHT_TOLERANCE = 1e-6


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def hhi(weights: pd.Series) -> float:
    """Return Herfindahl-Hirschman concentration from one vector of weights."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Weights must be a non-empty finite one-dimensional vector.")
    return float(np.square(values).sum())


def effective_holdings(weights: pd.Series) -> float:
    """Return inverse-HHI effective holdings."""
    concentration = hhi(weights)
    if concentration <= 0.0:
        raise ValueError("HHI must be positive.")
    return float(1.0 / concentration)


def l1_weight_distance(left: pd.Series, right: pd.Series) -> float:
    """Return sum(abs(left - right)) after exact label/order validation."""
    if list(left.index) != list(right.index):
        raise ValueError("Weight vectors must have identical ticker labels and order.")
    return float(np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float)).sum())


def _model_weight_pivots(weights: pd.DataFrame, models: Sequence[str]) -> dict[str, pd.DataFrame]:
    _require_columns(weights, ["rebalance_date", "model", "ticker", "target_weight"], "Stage 4 weights")
    pivots = {}
    for model in models:
        subset = weights.loc[weights["model"] == model]
        if subset.empty or subset.duplicated(["rebalance_date", "ticker"]).any():
            raise ValueError(f"Stage 4 weights are incomplete or duplicated for {model}.")
        pivot = subset.pivot(index="rebalance_date", columns="ticker", values="target_weight").sort_index()
        if pivot.isna().any().any():
            raise ValueError(f"Stage 4 target weights are incomplete for {model}.")
        pivots[model] = pivot
    return pivots


def consecutive_target_weight_changes(weights: pd.DataFrame) -> pd.DataFrame:
    """Calculate L1 changes between consecutive monthly target vectors."""
    pivots = _model_weight_pivots(weights, ALL_MODELS)
    records = []
    for model, pivot in pivots.items():
        changes = pivot.diff().abs().sum(axis=1)
        changes.iloc[0] = np.nan
        records.extend(
            {
                "rebalance_date": date,
                "model": model,
                "l1_target_weight_change": float(change) if pd.notna(change) else np.nan,
            }
            for date, change in changes.items()
        )
    return pd.DataFrame(records)


def pairwise_weight_distances(weights: pd.DataFrame) -> pd.DataFrame:
    """Calculate same-date L1 distances among the three optimized target portfolios."""
    pivots = _model_weight_pivots(weights, OPTIMIZED_MODELS)
    reference_dates = pivots["sample"].index
    reference_tickers = pivots["sample"].columns.tolist()
    for model, pivot in pivots.items():
        if not pivot.index.equals(reference_dates) or list(pivot.columns) != reference_tickers:
            raise ValueError(f"Target-weight dates or ticker order do not align for {model}.")

    return pd.DataFrame(
        {
            "rebalance_date": reference_dates,
            "sample_vs_ewma": [
                l1_weight_distance(pivots["sample"].loc[date], pivots["ewma"].loc[date])
                for date in reference_dates
            ],
            "sample_vs_ledoit_wolf": [
                l1_weight_distance(pivots["sample"].loc[date], pivots["ledoit_wolf"].loc[date])
                for date in reference_dates
            ],
            "ewma_vs_ledoit_wolf": [
                l1_weight_distance(pivots["ewma"].loc[date], pivots["ledoit_wolf"].loc[date])
                for date in reference_dates
            ],
        }
    )


def summarize_pairwise_distances(distances: pd.DataFrame) -> pd.DataFrame:
    """Summarize date-level pairwise target-weight L1 distances."""
    _require_columns(
        distances,
        ["sample_vs_ewma", "sample_vs_ledoit_wolf", "ewma_vs_ledoit_wolf"],
        "Pairwise distances",
    )
    records = []
    for column in ["sample_vs_ewma", "sample_vs_ledoit_wolf", "ewma_vs_ledoit_wolf"]:
        values = distances[column]
        records.append(
            {
                "pair": column,
                "mean_l1_distance": float(values.mean()),
                "median_l1_distance": float(values.median()),
                "p95_l1_distance": float(values.quantile(0.95)),
                "maximum_l1_distance": float(values.max()),
            }
        )
    return pd.DataFrame(records)


def high_volatility_months(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Classify common high-volatility months from cross-model median realized volatility."""
    _require_columns(forecasts, ["rebalance_date", "model", "realized_vol"], "Stage 4 forecasts")
    realized = forecasts.loc[forecasts["model"].isin(OPTIMIZED_MODELS)].pivot(
        index="rebalance_date", columns="model", values="realized_vol"
    )
    realized = realized.reindex(columns=OPTIMIZED_MODELS).sort_index()
    if realized.isna().any().any():
        raise ValueError("Optimized-model realized-volatility panel is incomplete.")
    reference = realized.median(axis=1)
    threshold = float(reference.quantile(0.75))
    return pd.DataFrame(
        {
            "rebalance_date": reference.index,
            "reference_realized_volatility": reference.to_numpy(),
            "high_volatility": (reference > threshold).to_numpy(),
        }
    ), threshold


def equal_weight_high_volatility_months(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Classify common high-volatility months from equal-weight realized volatility only."""
    _require_columns(forecasts, ["rebalance_date", "model", "realized_vol"], "Stage 4 forecasts")
    equal_weight = forecasts.loc[forecasts["model"] == "equal_weight", ["rebalance_date", "realized_vol"]]
    if equal_weight.empty or equal_weight.duplicated("rebalance_date").any():
        raise ValueError("Equal-weight realized-volatility observations are incomplete or duplicated.")
    reference = equal_weight.set_index("rebalance_date")["realized_vol"].sort_index()
    if reference.isna().any():
        raise ValueError("Equal-weight realized-volatility observations are incomplete.")
    threshold = float(reference.quantile(0.75))
    return pd.DataFrame(
        {
            "rebalance_date": reference.index,
            "reference_realized_volatility": reference.to_numpy(),
            "high_volatility": (reference > threshold).to_numpy(),
        }
    ), threshold


def regime_forecast_metrics(forecasts: pd.DataFrame, regimes: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Summarize forecast accuracy on a common high-volatility versus remaining-month split."""
    _require_columns(
        forecasts,
        ["rebalance_date", "model", "forecast_error", "absolute_error", "squared_error"],
        "Stage 4 forecasts",
    )
    merged = forecasts.loc[forecasts["model"].isin(OPTIMIZED_MODELS)].merge(
        regimes[["rebalance_date", "high_volatility"]],
        on="rebalance_date",
        validate="many_to_one",
    )
    records = []
    for model in OPTIMIZED_MODELS:
        for high_volatility, label in [(True, "high_volatility"), (False, "remaining_months")]:
            subset = merged.loc[(merged["model"] == model) & (merged["high_volatility"] == high_volatility)]
            records.append(
                {
                    "model": model,
                    "regime": label,
                    "months": len(subset),
                    "high_volatility_threshold": threshold,
                    "mae": float(subset["absolute_error"].mean()),
                    "rmse": float(np.sqrt(subset["squared_error"].mean())),
                    "bias": float(subset["forecast_error"].mean()),
                    "underprediction_frequency": float((subset["forecast_error"] < 0.0).mean()),
                }
            )
    return pd.DataFrame(records)


def transaction_cost_fraction(turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Return one-time cost fractions from one-sided turnover and bps per traded dollar."""
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("Transaction cost in basis points must be finite and non-negative.")
    values = pd.to_numeric(turnover, errors="raise")
    if (values.dropna() < 0.0).any() or not np.isfinite(values.dropna()).all():
        raise ValueError("Turnover must be non-negative and finite where observed.")
    return values.fillna(0.0) * 2.0 * (cost_bps / 10_000.0)


def apply_monthly_transaction_costs(
    forecasts: pd.DataFrame, daily_returns: pd.DataFrame, cost_bps: float
) -> pd.DataFrame:
    """Deduct each rebalance cost once, on the corresponding first holding-day return."""
    _require_columns(forecasts, ["model", "holding_start", "turnover"], "Stage 4 forecasts")
    _require_columns(daily_returns, ["date", "model", "portfolio_return"], "Stage 4 daily portfolio returns")
    costs = forecasts[["model", "holding_start", "turnover"]].copy()
    costs["transaction_cost_fraction"] = transaction_cost_fraction(costs["turnover"], cost_bps)
    costs = costs.rename(columns={"holding_start": "date"})[
        ["model", "date", "transaction_cost_fraction"]
    ]
    if costs.duplicated(["model", "date"]).any():
        raise ValueError("Stage 4 forecasts contain duplicate model holding-start dates.")

    net = daily_returns.merge(costs, on=["model", "date"], how="left", validate="one_to_one")
    net["transaction_cost_fraction"] = net["transaction_cost_fraction"].fillna(0.0)
    if (net["portfolio_return"] <= -1.0).any():
        raise ValueError("Daily portfolio returns must exceed -100%.")
    net["net_portfolio_return"] = (
        (1.0 + net["portfolio_return"]) * (1.0 - net["transaction_cost_fraction"]) - 1.0
    )
    return net


def _return_metrics(returns: pd.Series) -> dict[str, float]:
    """Calculate full-sample geometric return and risk metrics from daily portfolio returns."""
    values = pd.to_numeric(returns, errors="raise")
    if values.empty or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("Daily returns must be finite, non-empty, and greater than -100%.")
    compounded_growth = float((1.0 + values).prod())
    annualized_return = float(compounded_growth ** (252.0 / len(values)) - 1.0)
    annualized_volatility = float(values.std(ddof=1) * np.sqrt(252.0))
    wealth = (1.0 + values).cumprod()
    maximum_drawdown = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "annualized_return": annualized_return,
        "annualized_realized_volatility": annualized_volatility,
        "sharpe_ratio_rf0": float(values.mean() / values.std(ddof=1) * np.sqrt(252.0))
        if annualized_volatility > 0.0
        else np.nan,
        "maximum_drawdown": maximum_drawdown,
    }


def transaction_cost_analysis(
    forecasts: pd.DataFrame, daily_returns: pd.DataFrame, cost_bps_values: Sequence[float] = (5.0, 10.0, 20.0)
) -> pd.DataFrame:
    """Evaluate full-period net performance after monthly turnover-based transaction costs."""
    if not cost_bps_values:
        raise ValueError("At least one transaction-cost assumption is required.")
    models = [model for model in ALL_MODELS if model in set(daily_returns["model"])]
    records = []
    for cost_bps in cost_bps_values:
        net_daily = apply_monthly_transaction_costs(forecasts, daily_returns, float(cost_bps))
        for model in models:
            model_daily = net_daily.loc[net_daily["model"] == model].sort_values("date")
            gross_metrics = _return_metrics(model_daily["portfolio_return"])
            net_metrics = _return_metrics(model_daily["net_portfolio_return"])
            records.append(
                {
                    "model": model,
                    "cost_bps_per_traded_dollar": float(cost_bps),
                    "gross_annualized_return": gross_metrics["annualized_return"],
                    "gross_annualized_realized_volatility": gross_metrics["annualized_realized_volatility"],
                    "gross_sharpe_ratio_rf0": gross_metrics["sharpe_ratio_rf0"],
                    "gross_maximum_drawdown": gross_metrics["maximum_drawdown"],
                    "net_annualized_return": net_metrics["annualized_return"],
                    "net_annualized_realized_volatility": net_metrics["annualized_realized_volatility"],
                    "net_sharpe_ratio_rf0": net_metrics["sharpe_ratio_rf0"],
                    "net_maximum_drawdown": net_metrics["maximum_drawdown"],
                    "annualized_transaction_cost_drag": gross_metrics["annualized_return"]
                    - net_metrics["annualized_return"],
                    "cumulative_total_transaction_cost": float(
                        model_daily["transaction_cost_fraction"].sum()
                    ),
                    "charged_rebalances": int((model_daily["transaction_cost_fraction"] > 0.0).sum()),
                }
            )
    return pd.DataFrame(records)


def forecast_misses(forecasts: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    """Return the largest forecast underprediction and overprediction months per estimator."""
    _require_columns(
        forecasts,
        ["rebalance_date", "holding_start", "holding_end", "model", "predicted_vol", "realized_vol", "forecast_error"],
        "Stage 4 forecasts",
    )
    records = []
    for model in OPTIMIZED_MODELS:
        subset = forecasts.loc[forecasts["model"] == model]
        for miss_type, ordered in [
            ("largest_underprediction", subset.nsmallest(count, "forecast_error")),
            ("largest_overprediction", subset.nlargest(count, "forecast_error")),
        ]:
            for rank, row in enumerate(ordered.itertuples(index=False), start=1):
                records.append(
                    {
                        "model": model,
                        "miss_type": miss_type,
                        "rank": rank,
                        "rebalance_date": row.rebalance_date,
                        "holding_start": row.holding_start,
                        "holding_end": row.holding_end,
                        "predicted_vol": row.predicted_vol,
                        "realized_vol": row.realized_vol,
                        "forecast_error": row.forecast_error,
                    }
                )
    return pd.DataFrame(records)


def weight_stability(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize target-vector changes and ticker-level target-weight volatility."""
    changes = consecutive_target_weight_changes(weights)
    _require_columns(weights, ["model", "ticker", "target_weight"], "Stage 4 weights")
    stock_volatility = (
        weights.groupby(["model", "ticker"], as_index=False)["target_weight"]
        .std(ddof=1)
        .rename(columns={"target_weight": "target_weight_std"})
    )
    records = []
    for model in ALL_MODELS:
        model_changes = changes.loc[(changes["model"] == model) & changes["l1_target_weight_change"].notna()]
        model_stock_volatility = stock_volatility.loc[stock_volatility["model"] == model]
        most_unstable = model_stock_volatility.loc[model_stock_volatility["target_weight_std"].idxmax()]
        records.append(
            {
                "model": model,
                "average_monthly_l1_target_change": float(model_changes["l1_target_weight_change"].mean()),
                "median_monthly_l1_target_change": float(model_changes["l1_target_weight_change"].median()),
                "p95_monthly_l1_target_change": float(model_changes["l1_target_weight_change"].quantile(0.95)),
                "largest_monthly_l1_target_change": float(model_changes["l1_target_weight_change"].max()),
                "most_unstable_ticker": most_unstable["ticker"],
                "most_unstable_target_weight_std": float(most_unstable["target_weight_std"]),
            }
        )
    return pd.DataFrame(records), stock_volatility, changes


def turnover_summary(forecasts: pd.DataFrame, target_changes: pd.DataFrame) -> pd.DataFrame:
    """Summarize turnover and its descriptive relation to target and forecast changes."""
    _require_columns(forecasts, ["rebalance_date", "model", "turnover", "predicted_vol"], "Stage 4 forecasts")
    records = []
    for model in ALL_MODELS:
        model_forecasts = forecasts.loc[forecasts["model"] == model].sort_values("rebalance_date").copy()
        model_forecasts["abs_predicted_vol_change"] = model_forecasts["predicted_vol"].diff().abs()
        model_forecasts = model_forecasts.merge(
            target_changes.loc[target_changes["model"] == model],
            on=["rebalance_date", "model"],
            validate="one_to_one",
        )
        turnover = model_forecasts["turnover"].dropna()
        largest = model_forecasts.loc[model_forecasts["turnover"].idxmax()]
        def correlation_if_variable(left: pd.Series, right: pd.Series) -> float:
            paired = pd.concat([left, right], axis=1).dropna()
            if len(paired) < 2 or (paired.iloc[:, 0].std(ddof=1) == 0.0) or (paired.iloc[:, 1].std(ddof=1) == 0.0):
                return np.nan
            return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))

        records.append(
            {
                "model": model,
                "mean_turnover": float(turnover.mean()),
                "median_turnover": float(turnover.median()),
                "turnover_std": float(turnover.std(ddof=1)),
                "p95_turnover": float(turnover.quantile(0.95)),
                "maximum_turnover": float(turnover.max()),
                "largest_turnover_rebalance_date": largest["rebalance_date"],
                "turnover_target_l1_change_correlation": correlation_if_variable(
                    model_forecasts["turnover"], model_forecasts["l1_target_weight_change"]
                ),
                "turnover_abs_predicted_vol_change_correlation": correlation_if_variable(
                    model_forecasts["turnover"], model_forecasts["abs_predicted_vol_change"]
                ),
            }
        )
    return pd.DataFrame(records)


def concentration_metrics(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate concentration and position-count diagnostics at every rebalance."""
    _require_columns(weights, ["rebalance_date", "model", "ticker", "sector", "target_weight"], "Stage 4 weights")
    records = []
    for (rebalance_date, model), group in weights.groupby(["rebalance_date", "model"], sort=True):
        vector = group.set_index("ticker")["target_weight"]
        sector_totals = group.groupby("sector")["target_weight"].sum()
        records.append(
            {
                "rebalance_date": rebalance_date,
                "model": model,
                "hhi": hhi(vector),
                "effective_holdings": effective_holdings(vector),
                "positions_over_1pct": int((vector > 0.01).sum()),
                "positions_effectively_zero": int((vector <= ZERO_WEIGHT_TOLERANCE).sum()),
                "largest_stock_weight": float(vector.max()),
                "largest_sector_weight": float(sector_totals.max()),
            }
        )
    timeseries = pd.DataFrame(records)
    summary = (
        timeseries.groupby("model", as_index=False)
        .agg(
            mean_hhi=("hhi", "mean"),
            mean_effective_holdings=("effective_holdings", "mean"),
            minimum_effective_holdings=("effective_holdings", "min"),
            maximum_effective_holdings=("effective_holdings", "max"),
            mean_positions_over_1pct=("positions_over_1pct", "mean"),
            mean_positions_effectively_zero=("positions_effectively_zero", "mean"),
            mean_largest_stock_weight=("largest_stock_weight", "mean"),
            maximum_largest_sector_weight=("largest_sector_weight", "max"),
        )
        .set_index("model")
        .reindex(ALL_MODELS)
        .reset_index()
    )
    return timeseries, summary


def sector_allocations(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate metadata-driven target weights by rebalance date and sector."""
    _require_columns(weights, ["rebalance_date", "model", "sector", "target_weight"], "Stage 4 weights")
    timeseries = (
        weights.groupby(["rebalance_date", "model", "sector"], as_index=False)["target_weight"]
        .sum()
        .rename(columns={"target_weight": "sector_weight"})
    )
    summary = (
        timeseries.groupby(["model", "sector"], as_index=False)["sector_weight"]
        .agg(average_sector_weight="mean", minimum_sector_weight="min", maximum_sector_weight="max")
    )
    sector_means = summary.pivot(index="sector", columns="model", values="average_sector_weight")
    required = {"sample", "ewma", "ledoit_wolf"}
    if required.issubset(sector_means.columns):
        ewma_difference = sector_means["ewma"] - (sector_means["sample"] + sector_means["ledoit_wolf"]) / 2.0
        summary = summary.merge(
            ewma_difference.rename("ewma_minus_mean_sample_ledoit_wolf"),
            on="sector",
            how="left",
        )
    return timeseries, summary
