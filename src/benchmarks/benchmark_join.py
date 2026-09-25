import polars as pl
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RELATIONSHIPS = PROCESSED_DIR / "train_relationships.parquet"
SOURCE1 = PROCESSED_DIR / "train_source1.parquet"


print("Starting lazy join benchmark...")
print()

start = time.perf_counter()

result = (
    pl.scan_parquet(RELATIONSHIPS)
    .join(
        pl.scan_parquet(SOURCE1),
        left_on="source1_entity_id",
        right_on="entity_id",
        how="inner",
    )
    .select(
        [
            "source1_entity_id",
            "matched_entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    )
    .collect()
)

elapsed = time.perf_counter() - start

print(f"Joined rows: {result.height:,}")
print(f"Columns: {result.columns}")
print(f"Time: {elapsed:.2f} seconds")
print()
print(result.head(10))