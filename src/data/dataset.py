# src/data/dataset.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from src.stats.descriptive import (
    compute_descriptive_stats,
    compute_returns,
    compute_multi_descriptive_stats,
)

@dataclass
class TimeSeriesDataset:
    """A simple wrapper for a time series dataset, including metadata and convenience methods.
    
    Methods:
    - preview(n): Show the first n rows of the dataset.
    - coverage(): Get basic coverage info (row count, date range, columns).
    - missing_report(): Get a report of missing values by column.
    - describe_prices(price_col=None): Descriptive stats on the price column.
    - describe_returns(price_col=None, method='log'): Descriptive stats on returns derived from the price column.
    - describe_all_numeric(): Descriptive stats for all numeric columns.
    """
    name: str
    df: pd.DataFrame
    meta: object  # DatasetMeta

    def preview(self, n: int = 5) -> pd.DataFrame:
        """
        Returns the first n rows of the dataset for previewing.
        
        Args:
            n (int): The number of rows to return. Default is 5.
        
        Returns:
            pd.DataFrame: A DataFrame containing the first n rows of the dataset.
        """
        return self.df.head(n)

    def coverage(self) -> dict:
        """
        Returns basic coverage info about the dataset.
        
        Returns:
            dict: A dictionary containing the number of rows, start and end dates, and column names.
        """
        idx = self.df.index
        return {
            "rows": len(self.df),
            "start": str(idx.min()) if len(idx) else None,
            "end": str(idx.max()) if len(idx) else None,
            "cols": list(self.df.columns),
        }

    def _resolve_price_column(self, price_col: str | None = None) -> str:
        """
        Resolve which column to use as the price series.
        Priority:
        1) explicit arg
        2) metadata price_col
        3) 'price'
        """
        if price_col and price_col in self.df.columns:
            return price_col

        meta_price_col = getattr(self.meta, "price_col", None)
        if meta_price_col and meta_price_col in self.df.columns:
            return meta_price_col

        if "price" in self.df.columns:
            return "price"

        raise ValueError(
            f"Could not resolve price column for dataset '{self.name}'. "
            f"Available columns: {list(self.df.columns)}"
        )

    def missing_report(self) -> pd.Series:
        """
        Returns a report of missing values in the dataset.

        Returns:
            pd.Series: A Series containing the percentage of missing values for each column, sorted in descending order.        
        """
        return self.df.isna().mean().sort_values(ascending=False)

    def describe_prices(self, price_col: str | None = None) -> pd.DataFrame:
        """
        Descriptive stats on the price column.
        """
        col = self._resolve_price_column(price_col)
        return compute_descriptive_stats(self.df[col], series_name=f"{self.name}:{col}")

    def describe_returns(
        self,
        price_col: str | None = None,
        method: str = "log",
    ) -> pd.DataFrame:
        """
        Descriptive stats on returns derived from the price column.
        """
        col = self._resolve_price_column(price_col)
        returns = compute_returns(self.df[col], method=method)
        return compute_descriptive_stats(returns, series_name=f"{self.name}:{col}:{method}_ret")

    def describe_all_numeric(self) -> pd.DataFrame:
        """
        Descriptive stats for all numeric columns in the dataset.
        Useful when you have OHLCV or multiple price columns.
        """
        numeric_df = self.df.select_dtypes(include=["number"])
        if numeric_df.empty:
            raise ValueError(f"No numeric columns found in dataset '{self.name}'.")
        return compute_multi_descriptive_stats(numeric_df)