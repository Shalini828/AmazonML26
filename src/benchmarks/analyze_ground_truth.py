import polars as pl
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE = PROJECT_ROOT / "data" / "processed" / "train_ground_truth.parquet"


print("Loading ground truth...")
print()

# Read only the two required columns
df = pl.read_parquet(FILE)

print(f"Total Source 1 entities: {df.height:,}")
print()


# ---------------------------------------------------------
# 1. Count number of matches for every Source 1 entity
# ---------------------------------------------------------

df = df.with_columns(
    pl.when(
        pl.col("matched_entity_ids").is_null()
        | (pl.col("matched_entity_ids").str.strip_chars() == "")
    )
    .then(0)
    .otherwise(
        pl.col("matched_entity_ids")
        .str.split(",")
        .list.len()
    )
    .alias("match_count")
)


# ---------------------------------------------------------
# 2. Distribution of match counts
# ---------------------------------------------------------

distribution = (
    df.group_by("match_count")
    .agg(pl.len().alias("source1_count"))
    .sort("match_count")
)

print("Match-count distribution:")
print(distribution)
print()


# ---------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------

summary = df.select(
    [
        pl.col("match_count").min().alias("minimum_matches"),
        pl.col("match_count").max().alias("maximum_matches"),
        pl.col("match_count").mean().alias("average_matches"),
        pl.col("match_count").median().alias("median_matches"),
    ]
)

print("Summary:")
print(summary)
print()


# ---------------------------------------------------------
# 4. Entities with zero matches
# ---------------------------------------------------------

zero_matches = df.filter(
    pl.col("match_count") == 0
).height

one_match = df.filter(
    pl.col("match_count") == 1
).height

multiple_matches = df.filter(
    pl.col("match_count") > 1
).height

print("Important counts:")
print(f"Zero matches      : {zero_matches:,}")
print(f"Exactly one match : {one_match:,}")
print(f"Multiple matches  : {multiple_matches:,}")
