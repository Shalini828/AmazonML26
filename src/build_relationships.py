import polars as pl
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_ground_truth.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_relationships.parquet"
)


print("Building relationship table...")
print()

(
    pl.scan_parquet(INPUT_FILE)
    .with_columns(
        pl.col("matched_entity_ids")
        .fill_null("")
        .str.split(",")
        .alias("matched_entity_ids")
    )
    .explode("matched_entity_ids")
    .rename({
        "matched_entity_ids": "matched_entity_id"
    })
    .filter(
        pl.col("matched_entity_id").str.strip_chars() != ""
    )
    .sink_parquet(OUTPUT_FILE)
)

print("Done.")
print("Output:", OUTPUT_FILE)