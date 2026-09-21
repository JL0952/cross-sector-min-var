"""Stage 2 covariance diagnostics using the fixed Stage 1 return panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cross_sector_min_var.covariance import (
    ewma_covariance,
    ewma_weights,
    ledoit_wolf_covariance,
    sample_covariance,
    validate_covariance_matrix,
    validate_returns_window,
)


DEFAULT_WINDOW_SIZE = 504
DEFAULT_LAMBDA = 0.94


def load_latest_return_window(path: Path, observations: int = DEFAULT_WINDOW_SIZE) -> pd.DataFrame:
    """Load the saved Stage 1 returns and select its latest valid historical window."""
    returns = pd.read_parquet(path)
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Stage 1 returns must use a DatetimeIndex.")
    if returns.index.name != "date":
        raise ValueError("Stage 1 return index must be named 'date'.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Stage 1 returns must be sorted ascending by date.")
    validate_returns_window(returns)
    if len(returns) < observations:
        raise ValueError(f"Need {observations} returns, but saved dataset contains {len(returns)}.")

    window = returns.tail(observations).copy()
    validate_returns_window(window)
    return window


def covariance_diagnostic(
    estimator: str,
    matrix: pd.DataFrame,
    returns_window: pd.DataFrame,
    *,
    lambda_: Optional[float] = None,
    shrinkage: Optional[float] = None,
) -> dict[str, object]:
    """Validate one covariance matrix and create its one-row diagnostic record."""
    validate_covariance_matrix(matrix, returns_window.columns)
    values = matrix.to_numpy(dtype=float)
    eigenvalues = np.linalg.eigvalsh(values)
    diagonal = np.diag(values)

    return {
        "estimator": estimator,
        "window_start": returns_window.index.min().date().isoformat(),
        "window_end": returns_window.index.max().date().isoformat(),
        "observations": len(returns_window),
        "assets": returns_window.shape[1],
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(np.linalg.cond(values)),
        "mean_diagonal_daily_variance": float(diagonal.mean()),
        "minimum_diagonal_daily_variance": float(diagonal.min()),
        "maximum_diagonal_daily_variance": float(diagonal.max()),
        "lambda": lambda_,
        "half_life_days": float(np.log(0.5) / np.log(lambda_)) if lambda_ is not None else None,
        "ledoit_wolf_shrinkage": shrinkage,
    }


def build_stage2_diagnostics(
    returns_window: pd.DataFrame, lambda_: float = DEFAULT_LAMBDA
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estimate all Stage 2 matrices and return their diagnostic table and distances."""
    sample = sample_covariance(returns_window)
    ewma = ewma_covariance(returns_window, lambda_)
    ledoit_wolf, shrinkage = ledoit_wolf_covariance(returns_window)

    diagnostics = pd.DataFrame(
        [
            covariance_diagnostic("sample", sample, returns_window),
            covariance_diagnostic("ewma", ewma, returns_window, lambda_=lambda_),
            covariance_diagnostic(
                "ledoit_wolf",
                ledoit_wolf,
                returns_window,
                shrinkage=shrinkage,
            ),
        ]
    )
    matrices = {"sample": sample, "ewma": ewma, "ledoit_wolf": ledoit_wolf}
    distances = {
        "sample_vs_ewma": float(np.linalg.norm(sample.to_numpy() - ewma.to_numpy(), ord="fro")),
        "sample_vs_ledoit_wolf": float(np.linalg.norm(sample.to_numpy() - ledoit_wolf.to_numpy(), ord="fro")),
        "ewma_vs_ledoit_wolf": float(np.linalg.norm(ewma.to_numpy() - ledoit_wolf.to_numpy(), ord="fro")),
    }
    return diagnostics, distances


def run_stage2(
    root: Path,
    observations: int = DEFAULT_WINDOW_SIZE,
    lambda_: float = DEFAULT_LAMBDA,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run Stage 2 on saved data and persist only the compact diagnostics table."""
    root = root.resolve()
    window = load_latest_return_window(root / "data" / "processed" / "daily_returns.parquet", observations)
    diagnostics, distances = build_stage2_diagnostics(window, lambda_)

    output_path = root / "results" / "metrics" / "stage2_covariance_diagnostics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)

    weights = ewma_weights(len(window), lambda_)
    report = {
        "window_start": window.index.min().date().isoformat(),
        "window_end": window.index.max().date().isoformat(),
        "observations": len(window),
        "assets": window.shape[1],
        "ticker_order": window.columns.tolist(),
        "ewma": {
            "lambda": lambda_,
            "newest_observation_weight": float(weights[-1]),
            "oldest_observation_weight": float(weights[0]),
            "half_life_days": float(np.log(0.5) / np.log(lambda_)),
        },
        "frobenius_distances": distances,
        "diagnostic_path": str(output_path.relative_to(root)),
    }
    return diagnostics, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run covariance-estimation diagnostics.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory).")
    parser.add_argument("--observations", type=int, default=DEFAULT_WINDOW_SIZE, help="Latest return observations to use.")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA, help="EWMA decay parameter.")
    args = parser.parse_args()

    diagnostics, report = run_stage2(args.root, observations=args.observations, lambda_=args.lambda_)
    report["estimators"] = diagnostics.to_dict(orient="records")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
