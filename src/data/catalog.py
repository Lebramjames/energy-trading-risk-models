# src/data/catalog.py

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yaml

@dataclass(frozen=True)
class DatasetMeta:
    name: str
    path: Path
    kind: str
    symbol: str | None = None
    source: str | None = None
    freq: str | None = None
    tz: str | None = None
    price_col: str | None = None
    currency: str | None = None
    unit: str | None = None
    tags: list[str] | None = None

class DataCatalog:
    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path)
        self._meta = self._load()

    def _load(self) -> dict[str, DatasetMeta]:
        raw = yaml.safe_load(self.catalog_path.read_text())
        out = {}
        for name, d in raw.get("datasets", {}).items():
            out[name] = DatasetMeta(
                name=name,
                path=Path(d["path"]),
                kind=d.get("kind", "timeseries"),
                symbol=d.get("symbol"),
                source=d.get("source"),
                freq=d.get("freq"),
                tz=d.get("tz"),
                price_col=d.get("price_col", "price"),
                currency=d.get("currency"),
                unit=d.get("unit"),
                tags=d.get("tags", []),
            )
        return out

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for m in self._meta.values():
            rows.append({
                "name": m.name,
                "kind": m.kind,
                "symbol": m.symbol,
                "source": m.source,
                "freq": m.freq,
                "currency": m.currency,
                "unit": m.unit,
                "tags": ",".join(m.tags or []),
                "path": str(m.path),
            })
        return pd.DataFrame(rows).sort_values(["kind", "symbol", "name"])

    def filter(self, *, tag: str | None = None, source: str | None = None) -> list[str]:
        names = []
        for m in self._meta.values():
            if tag and tag not in (m.tags or []):
                continue
            if source and source != m.source:
                continue
            names.append(m.name)
        return sorted(names)

    def meta(self, name: str) -> DatasetMeta:
        return self._meta[name]