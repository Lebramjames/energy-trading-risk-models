# to implement: 
# - ADF test for stationarity
# - KPSS test for stationarity
# - Phillips-Perron test for stationarity

from dataclasses import dataclass
import pandas as pd
import warnings
from typing import Literal
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings("ignore", category=InterpolationWarning)

@dataclass
class StationarityTestResult:
    test_name: str
    test_statistic: float
    p_value: float
    critical_values: dict[str, float]
    is_stationary: bool

@dataclass
class StationarityTestSummary:
    series_name: str
    adf_result: StationarityTestResult
    kpss_result: StationarityTestResult
    pp_result: StationarityTestResult

def adf_test(series: pd.Series, alpha: float = 0.05, regression: Literal['c', 'ct', 'ctt', 'n']= "c", autolag: str = "AIC"):
    """
    Perform the Augmented Dickey-Fuller test for stationarity.
    
    Arguments:
        series: The time series to test for stationarity.
        alpha: Significance level for the test (default 0.05).
        regression: Type of regression to include in the test ('c' for constant, 'ct' for constant and trend, 'ctt' for constant, trend, and trend squared, 'n' for no constant).
        autolag: Method to use when automatically determining the lag length (default 'AIC').

    Returns:
        StationarityTestResult: The result of the ADF test, including test statistic, p-value, critical values, and stationarity conclusion.
    """
    from statsmodels.tsa.stattools import adfuller

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Series is empty after cleaning.")

    result = adfuller(s, regression=regression, autolag=autolag)
    return StationarityTestResult(
        test_name="ADF",
        test_statistic=float(result[0]),
        p_value=float(result[1]),
        critical_values={str(k): float(v) for k, v in result[4].items()},
        is_stationary=bool(result[1] < alpha) # Null hypothesis is non-stationarity, so we want p-value < alpha to conclude stationarity
    )


def kpss_test(
    series: pd.Series,
    regression: Literal['c', 'ct'] = "c",
    alpha: float = 0.05
):
    from statsmodels.tsa.stattools import kpss

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Series is empty after cleaning.")

    result = kpss(s, nlags="auto", regression=regression)
    return StationarityTestResult(
        test_name="KPSS",
        test_statistic=result[0],
        p_value=result[1],
        critical_values=result[3],
        is_stationary=result[1] > alpha, # Null hypothesis is stationarity, so we want p-value > alpha to conclude stationarity
    )

def pp_test(series: pd.Series, regression: Literal['c', 'ct'] = "c", lags: int = 0, alpha: float = 0.05):
    from arch.unitroot import PhillipsPerron

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Series is empty after cleaning.")

    pp = PhillipsPerron(s, trend=regression, lags=lags)
    return StationarityTestResult(
        test_name="Phillips-Perron",
        test_statistic=float(pp.stat),
        p_value=float(pp.pvalue),
        critical_values={str(k): float(v) for k, v in pp.critical_values.items()},
        is_stationary=float(pp.pvalue) < alpha, # Null hypothesis is non-stationarity, so we want p-value < alpha to conclude stationarity
    )

def run_stationarity_tests(series: pd.Series, series_name: str | None = None) -> StationarityTestSummary:
    """
    Run ADF, KPSS, and Phillips-Perron tests on the given series and return a summary.
    
    Arguments:
        series: The time series to test for stationarity.
        series_name: Optional name for the series (used in the summary).

    Returns:
        StationarityTestSummary: A summary of the stationarity test results.

    Example usage:
    >> run_stationarity_tests(my_dataset.df['price'], series_name='My Dataset Price')
    >> StationarityTestSummary(
        series_name='My Dataset Price',
        adf_result=StationarityTestResult(...),
        kpss_result=StationarityTestResult(...),
        pp_result=StationarityTestResult(...),
    )
    """
    name = series_name or (series.name if series.name else "series")
    adf_result = adf_test(series)
    kpss_result = kpss_test(series)
    pp_result = pp_test(series)

    return StationarityTestSummary(
        series_name=name,
        adf_result=adf_result,
        kpss_result=kpss_result,
        pp_result=pp_result,
    )
