# src/data/interface.py
import pandas as pd
from .catalog import DataCatalog
from .source import LocalFileSource
from .dataset import TimeSeriesDataset

class DataInterface:
    """Interface for accessing datasets from the catalog and loading them using the source."""
    def __init__(self, catalog: DataCatalog, source=None):
        self.catalog = catalog
        self.source = source or LocalFileSource()

    def available(self) -> pd.DataFrame:
        """
        Returns a DataFrame listing all available datasets in the catalog.
        
        Returns:
            pd.DataFrame: A DataFrame containing metadata of all datasets in the catalog.
        """
        return self.catalog.to_dataframe()

    def load(self, name: str) -> TimeSeriesDataset:
        """Loads a dataset by name and returns a TimeSeriesDataset object."""
        meta = self.catalog.meta(name)
        df = self.source.load(meta.path)

        # standardize: parse datetime index if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        return TimeSeriesDataset(name=name, df=df, meta=meta)