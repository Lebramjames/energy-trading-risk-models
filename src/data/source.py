# src/data/source.py
from pathlib import Path
import pandas as pd

class LocalFileSource:
    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")
