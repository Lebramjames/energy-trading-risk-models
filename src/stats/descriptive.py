from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import jarque_bera


def _safe_series(series: pd.Series) -> pd.Series:
    """Convert to numeric and drop NaNs."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s

def compute_returns(
    series: pd.Series,
    method: str = "log",
) -> pd.Series:
    """
    Compute returns from a price series.
    method: 'log' or 'simple'
    """
    s = _safe_series(series)

    if method == "log":
        ret = np.log(s / s.shift(1))
    elif method == "simple":
        ret = s.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

    ret.name = f"{series.name or 'price'}_{method}_return"
    return ret.dropna()


def compute_descriptive_stats(series: pd.Series, series_name: str | None = None) -> pd.DataFrame:
    """
    Compute descriptive statistics for a single series.
    Returns a one-row DataFrame (nice for concatenating across assets).
    """
    s = _safe_series(series)

    if s.empty:
        raise ValueError("Series is empty after cleaning.")

    jb_stat, jb_p = jarque_bera(s)

    result = {
        "series": series_name or (series.name if series.name else "value"),
        "From": s.index.min() if s.index.is_all_dates else None,
        "To": s.index.max() if s.index.is_all_dates else None,
        "observations": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "maximum": float(s.max()),
        "minimum": float(s.min()),
        "std_dev": float(s.std(ddof=1)),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurt()),  # pandas excess kurtosis by default
        "jarque_bera": float(jb_stat),
        "jb_pvalue": float(jb_p),
    }

    return pd.DataFrame([result])

def compute_multi_descriptive_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Compute descriptive stats for multiple columns in a DataFrame.
    Returns one row per column.
    """
    cols = columns or list(df.columns)
    tables = [compute_descriptive_stats(df[col], series_name=col) for col in cols]
    out = pd.concat(tables, ignore_index=True)
    return out

