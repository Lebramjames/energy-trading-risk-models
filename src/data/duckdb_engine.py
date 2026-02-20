# src/data/duckdb_engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import duckdb
import pandas as pd

@dataclass(frozen=True)
class DatasetSummary:
    name: str
    rows: int
    start: str | None
    end: str | None
    cols: int
    path: str

class DuckDBEngine:
    def __init__(self, db_path: str | Path = "data/.duckdb/energy.duckdb", read_only: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path), read_only=read_only)

        # Sensible defaults
        self.con.execute("PRAGMA threads=4;")
        self.con.execute("PRAGMA enable_progress_bar=false;")

    def close(self):
        self.con.close()

    def summarize_parquet(self, *, name: str, path: str | Path, date_col: str = "date") -> DatasetSummary:
        path = str(Path(path))

        # row count + min/max date: DuckDB can push this down efficiently in many cases
        q = f"""
        SELECT
            COUNT(*) AS rows,
            MIN({date_col}) AS start,
            MAX({date_col}) AS end
        FROM read_parquet('{path}')
        """
        rows, start, end = self.con.execute(q).fetchone()

        # column count (cheap)
        col_q = f"SELECT COUNT(*) FROM parquet_schema('{path}')"
        cols = self.con.execute(col_q).fetchone()[0]

        return DatasetSummary(
            name=name,
            rows=int(rows),
            start=str(start) if start is not None else None,
            end=str(end) if end is not None else None,
            cols=int(cols),
            path=path,
        )

    def load_parquet(
        self,
        *,
        path: str | Path,
        columns: list[str] | None = None,
        date_col: str = "date",
        start: str | None = None,
        end: str | None = None,
        where: str | None = None,
    ) -> pd.DataFrame:
        path = str(Path(path))
        cols_sql = "*" if not columns else ", ".join(columns)

        filters = []
        if start:
            filters.append(f"{date_col} >= '{start}'")
        if end:
            filters.append(f"{date_col} <= '{end}'")
        if where:
            filters.append(f"({where})")

        where_sql = ""
        if filters:
            where_sql = "WHERE " + " AND ".join(filters)

        q = f"SELECT {cols_sql} FROM read_parquet('{path}') {where_sql} ORDER BY {date_col}"
        return self.con.execute(q).df()