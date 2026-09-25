import polars as pl
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_source2.parquet"
)


# ---------------------------------------------------------
# Get 1,000 real IDs
# ---------------------------------------------------------

df_sample = (
    pl.read_parquet(FILE)
    .select(["entity_id"])
    .head(1000)
)

ids = df_sample["entity_id"].to_list()

print("IDs to lookup:", len(ids))
print()


# ---------------------------------------------------------
# Method 1: Polars Parquet filtering
# ---------------------------------------------------------

start = time.perf_counter()

result = (
    pl.scan_parquet(FILE)
    .filter(
        pl.col("entity_id").is_in(ids)
    )
    .collect()
)

polars_time = time.perf_counter() - start

print("Method 1: Polars Parquet")
print(f"Records found: {result.height}")
print(f"Time: {polars_time:.4f} seconds")
print()


# ---------------------------------------------------------
# Method 2: In-memory dictionary
# ---------------------------------------------------------

start = time.perf_counter()

df = pl.read_parquet(FILE)

lookup = {
    row["entity_id"]: row
    for row in df.iter_rows(named=True)
}

build_time = time.perf_counter() - start


start = time.perf_counter()

results = [
    lookup[entity_id]
    for entity_id in ids
    if entity_id in lookup
]

lookup_time = time.perf_counter() - start

print("Method 2: In-memory dictionary")
print(f"Dictionary build time: {build_time:.4f} seconds")
print(f"Lookup time: {lookup_time:.6f} seconds")
print(f"Records found: {len(results)}")