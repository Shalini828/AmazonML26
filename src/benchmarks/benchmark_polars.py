import polars as pl
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE = PROJECT_ROOT / "data" / "dataset" / "train" / "train_source2.tsv"

print("File:", FILE)
print()
print("Starting Polars benchmark...")

start = time.perf_counter()

df = pl.read_csv(
    FILE,
    separator="\t",
    infer_schema_length=10000,
)

elapsed = time.perf_counter() - start

print(f"Rows: {df.height:,}")
print(f"Columns: {df.width}")
print(f"Time: {elapsed:.2f} seconds")
print()
print(df.head(5))