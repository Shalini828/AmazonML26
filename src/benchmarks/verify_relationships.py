import polars as pl
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_relationships.parquet"
)


df = pl.read_parquet(FILE)

print("Relationship table")
print("=" * 40)

print(f"Rows: {df.height:,}")
print(f"Columns: {df.columns}")
print()

print("Schema:")
print(df.schema)
print()

print("First 10 relationships:")
print(df.head(10))