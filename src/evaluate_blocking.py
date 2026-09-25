import polars as pl

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from normalization import normalize_text


# --------------------------------------------------
# Normalization + blocking keys
# --------------------------------------------------

def add_blocking_keys(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create the same blocking keys used by blocking.py.

    key1 = country + first character
    key2 = country + first 4 characters
    key3 = country + first 3 characters
    """

    df = df.with_columns(
        [
            pl.col("business_name")
            .map_elements(
                normalize_text,
                return_dtype=pl.String,
            )
            .fill_null(pl.lit(""))
            .alias("name_normalized"),

            pl.col("country")
            .map_elements(
                normalize_text,
                return_dtype=pl.String,
            )
            .fill_null(pl.lit(""))
            .alias("country_normalized"),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("country_normalized")
                + pl.lit("_")
                + pl.col("name_normalized").str.slice(0, 1)
            ).alias("key1"),

            (
                pl.col("country_normalized")
                + pl.lit("_")
                + pl.col("name_normalized").str.slice(0, 4)
            ).alias("key2"),

            (
                pl.col("country_normalized")
                + pl.lit("_")
                + pl.col("name_normalized").str.slice(0, 3)
            ).alias("key3"),
        ]
    )

    return df.select(
        [
            "entity_id",
            "key1",
            "key2",
            "key3",
        ]
    )


# --------------------------------------------------
# Ground truth
# --------------------------------------------------

def load_ground_truth() -> pl.DataFrame:
    """
    Load ground truth and explode matched IDs.

    Output:
        source1_entity_id | target_id
    """

    ground_truth = (
        pl.scan_csv(
            TRAIN_GROUND_TRUTH,
            separator="\t",
            infer_schema=False,
        )
        .select(
            [
                pl.col("source1_entity_id")
                .cast(pl.String),

                pl.col("matched_entity_ids")
                .cast(pl.String),
            ]
        )
        .collect()
    )

    matches = (
        ground_truth
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null(pl.lit(""))
            .str.split(by=",")
            .alias("target_id")
        )
        .explode(
            "target_id",
            empty_as_null=True,
        )
        .with_columns(
            pl.col("target_id")
            .str.strip_chars()
        )
        .filter(
            pl.col("target_id").is_not_null()
            & (pl.col("target_id") != pl.lit(""))
        )
        .select(
            [
                "source1_entity_id",
                pl.col("target_id").alias("target_id"),
            ]
        )
    )

    return matches


# --------------------------------------------------
# Load relevant source records
# --------------------------------------------------

def load_relevant_source(
    path,
    required_ids: set[str],
) -> pl.DataFrame:
    """
    Read only the records whose entity IDs are required.

    Uses Polars lazy scanning and filtering.
    """

    ids_df = pl.DataFrame(
        {
            "entity_id": list(required_ids)
        }
    )

    df = (
        pl.scan_csv(
            path,
            separator="\t",
            infer_schema=False,
        )
        .select(
            [
                "entity_id",
                "business_name",
                "country",
            ]
        )
        .join(
            ids_df.lazy(),
            on="entity_id",
            how="inner",
        )
        .collect()
    )

    return add_blocking_keys(df)


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    print("Loading ground truth...")

    matches = load_ground_truth()

    source1_count = (
        matches
        .select("source1_entity_id")
        .unique()
        .height
    )

    print(
        "Source 1 entities in ground truth:",
        source1_count,
    )

    target_ids = (
        matches
        .select("target_id")
        .unique()
    )

    print(
        "True matched Source 2/3 IDs:",
        target_ids.height,
    )

    # --------------------------------------------------
    # Source 1
    # --------------------------------------------------

    print("\nReading Source 1...")

    source1_ids = (
        matches
        .select(
            pl.col("source1_entity_id")
            .alias("entity_id")
        )
        .unique()
    )

    source1 = load_relevant_source(
        TRAIN_SOURCE1,
        set(source1_ids["entity_id"].to_list()),
    )

    print(
        "Source 1 records loaded:",
        source1.height,
    )

    # --------------------------------------------------
    # Source 2
    # --------------------------------------------------

    print("\nReading Source 2...")

    source2 = load_relevant_source(
        TRAIN_SOURCE2,
        set(target_ids["target_id"].to_list()),
    )

    print(
        "Relevant Source 2 records:",
        source2.height,
    )

    # --------------------------------------------------
    # Source 3
    # --------------------------------------------------

    print("\nReading Source 3...")

    source3 = load_relevant_source(
        TRAIN_SOURCE3,
        set(target_ids["target_id"].to_list()),
    )

    print(
        "Relevant Source 3 records:",
        source3.height,
    )

    # --------------------------------------------------
    # Combine target blocking keys
    # --------------------------------------------------

    targets = pl.concat(
        [
            source2.select(
                [
                    "entity_id",
                    "key1",
                    "key2",
                     "key3",
                ]
            ),
            source3.select(
                [
                    "entity_id",
                    "key1",
                    "key2",
                    "key3",
                ]
            ),
        ]
    )

    # --------------------------------------------------
    # Evaluate blocking
    # --------------------------------------------------

    evaluated = (
        matches

        # Attach Source 1 blocking keys
        .join(
            source1.rename(
                {
                    "entity_id": "source1_entity_id",
                    "key1": "source1_key1",
                    "key2": "source1_key2",
                    "key3": "source1_key3",
                }
            ),
            on="source1_entity_id",
            how="left",
        )

        # Attach target blocking keys
        .join(
            targets.rename(
                {
                    "entity_id": "target_id",
                    "key1": "target_key1",
                    "key2": "target_key2",
                    "key3": "target_key3",
                }
            ),
            on="target_id",
            how="left",
        )

        # Match survives if ANY blocking key matches.
        .with_columns(
           (
    (
        (pl.col("source1_key1") != pl.lit(""))
        & (pl.col("target_key1") != pl.lit(""))
        & (pl.col("source1_key1") == pl.col("target_key1"))
    )
    |
    (
        (pl.col("source1_key2") != pl.lit(""))
        & (pl.col("target_key2") != pl.lit(""))
        & (pl.col("source1_key2") == pl.col("target_key2"))
    )
    |
    (
        (pl.col("source1_key3") != pl.lit(""))
        & (pl.col("target_key3") != pl.lit(""))
        & (pl.col("source1_key3") == pl.col("target_key3"))
    )
).alias("retained")
        )
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total_true_matches = evaluated.height

    retained_true_matches = (
        evaluated
        .filter(pl.col("retained"))
        .height
    )

    lost_matches = (
        total_true_matches
        - retained_true_matches
    )

    if total_true_matches == 0:
        print("\nNo true matches found.")
        return

    recall = (
        retained_true_matches
        / total_true_matches
        * 100
    )

    print("\n" + "=" * 50)
    print("BLOCKING EVALUATION")
    print("=" * 50)

    print(
        f"Total true matches:       "
        f"{total_true_matches:,}"
    )

    print(
        f"Retained by blocking:     "
        f"{retained_true_matches:,}"
    )

    print(
        f"Lost during blocking:     "
        f"{lost_matches:,}"
    )

    print(
        f"Blocking recall:          "
        f"{recall:.2f}%"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()