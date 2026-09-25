import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE = PROJECT_ROOT / "data" / "processed" / "train_source2.parquet"

df = pl.read_parquet(FILE)

print("Rows:", f"{df.height:,}")
print("Columns:", df.columns)
print()

print("Schema:")
print(df.schema)

print()
print("First 5 rows:")
print(df.head(5))