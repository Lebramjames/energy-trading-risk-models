from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

class CointegrationValidationError(Exception):
    """Base error for cointegration input validation."""

def _to_numeric_series(series: pd.Series, name: str | None = None) -> pd.Series:
    s = pd.Series(series).copy()
    s = pd.to_numeric(s, errors="coerce")
    s.name = name or getattr(series, "name", None) or "series"
    return s

def _align_two_series(
    y: pd.Series,
    x: pd.Series,
    *,
    min_obs: int = 30,
) -> tuple[pd.Series, pd.Series]:
    """
    Align two series on index, coerce numeric, drop NaNs, and validate.
    """
    y_clean = _to_numeric_series(y, name=y.name or "y")
    x_clean = _to_numeric_series(x, name=x.name or "x")

    df = pd.concat([y_clean, x_clean], axis=1, join="inner").dropna()
    if df.empty:
        raise CointegrationValidationError("No overlapping non-null observations after alignment.")

    if len(df) < min_obs:
        raise CointegrationValidationError(
            f"Only {len(df)} overlapping observations; need at least {min_obs} for cointegration testing."
        )

    if df.iloc[:, 0].nunique() < 3 or df.iloc[:, 1].nunique() < 3:
        raise CointegrationValidationError("One or both series have too few unique values.")

    return df.iloc[:, 0], df.iloc[:, 1]

@dataclass
class EngleGrangerCointegrationResult:
    y_name: str
    x_name: str
    test_statistic: float
    p_value: float
    critical_values: dict[str, float]
    is_cointegrated: bool
    alpha: float
    trend: str
    hedge_ratio_beta: float
    intercept_alpha: float
    spread_name: str

    def summary_dict(self) -> dict[str, float | str | bool]:
        return {
            "y": self.y_name,
            "x": self.x_name,
            "test": "Engle-Granger",
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "is_cointegrated": self.is_cointegrated,
            "trend": self.trend,
            "hedge_ratio_beta": self.hedge_ratio_beta,
            "intercept_alpha": self.intercept_alpha,
            "spread_name": self.spread_name,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary_dict()])
    
@dataclass
class JohansenCointegrationResult:
    column_names: list[str]
    det_order: int
    k_ar_diff: int
    trace_stats: list[float]
    trace_crit_vals_95: list[float]
    maxeig_stats: list[float]
    maxeig_crit_vals_95: list[float]
    rank_trace_95: int
    rank_maxeig_95: int
    eigenvectors: pd.DataFrame

    def summary_dict(self) -> dict[str, object]:
        return {
            "test": "Johansen",
            "columns": self.column_names,
            "det_order": self.det_order,
            "k_ar_diff": self.k_ar_diff,
            "rank_trace_95": self.rank_trace_95,
            "rank_maxeig_95": self.rank_maxeig_95,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary_dict()])
    
def estimate_hedge_ratio_ols(
    y: pd.Series,
    x: pd.Series,
    *,
    add_intercept: bool = True,
) -> tuple[float, float]:
    """
    Estimate y = alpha + beta * x via OLS (numpy least squares).
    Returns (alpha, beta).

    This is useful for constructing the spread:
        spread = y - (alpha + beta*x)
    """
    y_aligned, x_aligned = _align_two_series(y, x)

    yv = y_aligned.to_numpy(dtype=float)
    xv = x_aligned.to_numpy(dtype=float)

    if add_intercept:
        X = np.column_stack([np.ones(len(xv)), xv])
        alpha, beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    else:
        X = xv.reshape(-1, 1)
        beta = float(np.linalg.lstsq(X, yv, rcond=None)[0][0])
        alpha = 0.0

    return float(alpha), float(beta)

def compute_spread(
    y: pd.Series,
    x: pd.Series,
    *,
    alpha: float,
    beta: float,
    spread_name: str | None = None,
) -> pd.Series:
    """
    Compute spread = y - (alpha + beta*x), aligned on common index.
    """
    y_aligned, x_aligned = _align_two_series(y, x)
    spread = y_aligned - (alpha + beta * x_aligned)
    spread.name = spread_name or f"spread_{y_aligned.name}_vs_{x_aligned.name}"
    return spread

def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    *,
    alpha: float = 0.05,
    trend: Literal["c", "ct", "ctt", "n"] = "c",
    autolag: str = "AIC",
    return_spread: bool = False,
) -> EngleGrangerCointegrationResult | tuple[EngleGrangerCointegrationResult, pd.Series]:
    """
    Engle-Granger cointegration test for two series.

    Notes
    -----
    statsmodels.tsa.stattools.coint tests the null hypothesis of NO cointegration.
    We conclude cointegration if p_value < alpha.
    """
    from statsmodels.tsa.stattools import coint

    y_aligned, x_aligned = _align_two_series(y, x)

    # 1) Estimate hedge ratio / equilibrium relation via OLS
    intercept_alpha, hedge_beta = estimate_hedge_ratio_ols(y_aligned, x_aligned, add_intercept=(trend != "n"))

    # 2) Compute spread (for trading / diagnostics)
    spread_name = f"spread_{y_aligned.name}_vs_{x_aligned.name}"
    spread = compute_spread(y_aligned, x_aligned, alpha=intercept_alpha, beta=hedge_beta, spread_name=spread_name)

    # 3) Cointegration test (null = no cointegration)
    test_stat, p_value, crit_vals = coint(y_aligned, x_aligned, trend=trend, autolag=autolag)

    crit_map = {
        "1%": float(crit_vals[0]),
        "5%": float(crit_vals[1]),
        "10%": float(crit_vals[2]),
    }

    result = EngleGrangerCointegrationResult(
        y_name=str(y_aligned.name),
        x_name=str(x_aligned.name),
        test_statistic=float(test_stat),
        p_value=float(p_value),
        critical_values=crit_map,
        is_cointegrated=bool(float(p_value) < alpha),
        alpha=float(alpha),
        trend=trend,
        hedge_ratio_beta=float(hedge_beta),
        intercept_alpha=float(intercept_alpha),
        spread_name=spread_name,
    )

    if return_spread:
        return result, spread
    return result

def johansen_test(
    df: pd.DataFrame,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenCointegrationResult:
    """
    Johansen cointegration test for 2+ series.

    Parameters
    ----------
    df : pd.DataFrame
        Columns are the time series (same integration order, typically I(1)).
    det_order : int
        Deterministic term assumption in statsmodels coint_johansen.
        Common values:
          -1: no deterministic terms
           0: constant term
           1: linear trend
    k_ar_diff : int
        Number of lagged differences in the VECM.

    Returns
    -------
    JohansenCointegrationResult
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    clean = _clean_dataframe_for_johansen(df)

    joh = coint_johansen(clean, det_order=det_order, k_ar_diff=k_ar_diff)

    # Trace test rank at 95%
    # joh.lr1 = trace stats
    # joh.cvt = trace critical values columns [90,95,99]
    trace_stats = [float(v) for v in joh.lr1]
    trace_cv_95 = [float(v) for v in joh.cvt[:, 1]]

    rank_trace_95 = 0
    for stat, cv in zip(trace_stats, trace_cv_95):
        if stat > cv:
            rank_trace_95 += 1

    # Max eigenvalue test rank at 95%
    # joh.lr2 = max eig stats
    # joh.cvm = max eig critical values [90,95,99]
    maxeig_stats = [float(v) for v in joh.lr2]
    maxeig_cv_95 = [float(v) for v in joh.cvm[:, 1]]

    rank_maxeig_95 = 0
    for stat, cv in zip(maxeig_stats, maxeig_cv_95):
        if stat > cv:
            rank_maxeig_95 += 1

    eigvec_df = pd.DataFrame(
        joh.evec,
        index=clean.columns,
        columns=[f"vec_{i+1}" for i in range(joh.evec.shape[1])],
    )

    return JohansenCointegrationResult(
        column_names=[str(c) for c in clean.columns],
        det_order=int(det_order),
        k_ar_diff=int(k_ar_diff),
        trace_stats=trace_stats,
        trace_crit_vals_95=trace_cv_95,
        maxeig_stats=maxeig_stats,
        maxeig_crit_vals_95=maxeig_cv_95,
        rank_trace_95=rank_trace_95,
        rank_maxeig_95=rank_maxeig_95,
        eigenvectors=eigvec_df,
    )

def pair_cointegration_report(
    y: pd.Series,
    x: pd.Series,
    *,
    alpha: float = 0.05,
    trend: Literal["c", "ct", "ctt", "n"] = "c",
    autolag: str = "AIC",
) -> dict[str, object]:
    """
    Convenience wrapper for a pair:
      - Engle-Granger result
      - Spread series
      - Basic spread diagnostics (mean/std)
    """
    eg_result, spread = engle_granger_test(
        y,
        x,
        alpha=alpha,
        trend=trend,
        autolag=autolag,
        return_spread=True,
    )

    diagnostics = {
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std(ddof=1)),
        "spread_min": float(spread.min()),
        "spread_max": float(spread.max()),
        "observations": int(spread.shape[0]),
    }

    return {
        "engle_granger": eg_result,
        "spread": spread,
        "spread_diagnostics": diagnostics,
    }

def _clean_dataframe_for_johansen(
    df: pd.DataFrame,
    *,
    min_obs: int = 50,
    min_cols: int = 2,
) -> pd.DataFrame:
    """
    Coerce all columns to numeric, drop rows with NaNs, validate shape.
    """
    if not isinstance(df, pd.DataFrame):
        raise CointegrationValidationError("Johansen test requires a pandas DataFrame.")

    clean = df.copy()
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna()

    if clean.shape[1] < min_cols:
        raise CointegrationValidationError(f"Need at least {min_cols} columns for Johansen test.")

    if clean.shape[0] < min_obs:
        raise CointegrationValidationError(
            f"Only {clean.shape[0]} observations after cleaning; need at least {min_obs}."
        )

    for col in clean.columns:
        if clean[col].nunique() < 3:
            raise CointegrationValidationError(f"Column '{col}' has too few unique values.")

    return clean