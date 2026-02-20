# src/analysis/validation.py
from __future__ import annotations

import pandas as pd

from dataclasses import dataclass

@dataclass(frozen=True)
class SeriesValidationConfig:
    min_obs: int = 20
    min_unique: int = 3
    allow_infinite: bool = False

def validate_series_for_stationarity_test(
    series: pd.Series,
    *,
    min_obs: int = 20,
    min_unique: int = 3,
    allow_infinite: bool = False,
    validate_series_for_stationarity_test_config: SeriesValidationConfig | None = None,
) -> pd.Series:
    """
    Clean and validate a series before stationarity tests.

    Returns
    -------
    pd.Series
        Clean numeric series with NaNs removed.

    Raises
    ------
    EmptySeriesError
    ShortSeriesError
    ConstantSeriesError
    NonNumericSeriesError
    """
    if validate_series_for_stationarity_test_config is not None:
        min_obs = validate_series_for_stationarity_test_config.min_obs
        min_unique = validate_series_for_stationarity_test_config.min_unique
        allow_infinite = validate_series_for_stationarity_test_config.allow_infinite

    if series is None:
        raise EmptySeriesError("Series is None.")

    # Force to Series if someone passes a list/array
    s = pd.Series(series)

    # Convert to numeric
    s = pd.to_numeric(s, errors="coerce")

    if s.isna().all():
        raise NonNumericSeriesError("Series could not be converted to numeric values.")

    # Handle inf values
    if not allow_infinite:
        s = s.replace([float("inf"), float("-inf")], pd.NA)

    # Drop NA after conversion/cleanup
    s = s.dropna()

    if s.empty:
        raise EmptySeriesError("Series is empty after cleaning.")

    if len(s) < min_obs:
        raise ShortSeriesError(
            f"Series has {len(s)} observations; need at least {min_obs} for reliable stationarity testing."
        )

    if s.nunique() < min_unique:
        raise ConstantSeriesError(
            f"Series has only {s.nunique()} unique values; stationarity tests are not meaningful."
        )

    return s

class SeriesValidationError(Exception):
    """Base error for invalid time series input."""


class EmptySeriesError(SeriesValidationError):
    """Raised when the series is empty after cleaning."""


class ShortSeriesError(SeriesValidationError):
    """Raised when the series is too short for a reliable test."""


class ConstantSeriesError(SeriesValidationError):
    """Raised when the series has too few unique values / is constant."""


class NonNumericSeriesError(SeriesValidationError):
    """Raised when the series cannot be converted to numeric."""


