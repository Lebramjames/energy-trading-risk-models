import pandas as pd
from ..data.dataset import TimeSeriesDataset

def descriptive_table_for_datasets(datasets: dict[str, "TimeSeriesDataset"]) -> pd.DataFrame:
    tables = []
    for name, ds in datasets.items():
        t = ds.describe_prices()
        t = t.set_index("series").T
        t.columns = [name]
        tables.append(t)

    return pd.concat(tables, axis=1)