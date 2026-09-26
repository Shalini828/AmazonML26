import polars as pl

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)

from normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_pincode,
)

# --------------------------------------------------
# Normalization + blocking keys
# --------------------------------------------------

def add_blocking_keys(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create exactly the same blocking keys as blocking.py.

    This is important: the evaluator and production matcher
    must use identical blocking logic.
    """

    # --------------------------------------------------
    # Normalize name and country
    # --------------------------------------------------

    df = df.with_columns(
        [
            pl.col("business_name")
            .map_elements(
                normalize_business_name,
                return_dtype=pl.String,
            )
            .fill_null("")
            .alias("name_normalized"),

            pl.col("country")
            .map_elements(
                normalize_country,
                return_dtype=pl.String,
            )
            .fill_null("")
            .alias("country_normalized"),
        ]
    )

    # --------------------------------------------------
    # Pincode
    # --------------------------------------------------

    df = df.with_columns(
        pl.struct(["business_address", "country"])
        .map_elements(
            lambda row: extract_pincode(
                row["business_address"],
                row["country"],
            ),
            return_dtype=pl.String,
        )
        .fill_null("")
        .alias("pincode")
    )

    # --------------------------------------------------
    # Longest meaningful word
    # --------------------------------------------------

    def get_longest_word(name):
        if name is None:
            return ""

        name = str(name).strip()

        if not name:
            return ""

        words = [
            word
            for word in name.split()
            if len(word) >= 4
        ]

        if not words:
            return ""

        return max(words, key=len)

    df = df.with_columns(
        pl.col("name_normalized")
        .map_elements(
            get_longest_word,
            return_dtype=pl.String,
        )
        .fill_null("")
        .alias("longest_name_word")
    )

    # --------------------------------------------------
    # Seven blocking keys
    # --------------------------------------------------

    country = pl.col("country_normalized").fill_null("")
    name = pl.col("name_normalized").fill_null("")
    longest = pl.col("longest_name_word").fill_null("")
    pincode = pl.col("pincode").fill_null("")

    df = df.with_columns(
        [
            # Rule 1
            (
                country
                + pl.lit("_")
                + name.str.slice(0, 1)
            ).alias("key1"),

            # Rule 2
            (
                country
                + pl.lit("_")
                + longest.str.slice(0, 4)
            ).alias("key2"),

            # Rule 3
            (
                country
                + pl.lit("_")
                + longest.str.slice(0, 3)
            ).alias("key3"),

            # Rule 4
            (
                country
                + pl.lit("_")
                + longest.str.slice(0, 2)
            ).alias("key4"),

            # Rule 5
            (
                country
                + pl.lit("_")
                + pincode
            ).alias("key5"),

            # Rule 6
            (
                country
                + pl.lit("_")
                + name.str.slice(0, 3)
            ).alias("key6"),

            # Rule 7
            (
                country
                + pl.lit("_")
                + name
            ).alias("key7"),
        ]
    )

# Address-based blocking keys
    address = (
        pl.col("business_address")
        .map_elements(
            normalize_address,
            return_dtype=pl.String,
        )
        .fill_null("")
    )

    df = df.with_columns(
        [
            (
                country
                + pl.lit("_")
                + address.str.split(" ").list.first()
            ).alias("key8"),

            (
                country
                + pl.lit("_")
                + address.str.split(" ").list.slice(0, 2).list.join("_")
            ).alias("key9"),

            (
                country
                + pl.lit("_")
                + address.str.extract(r"(\d+)", 1)
            ).alias("key10"),

                        # Rule 11: country + first address number, leading zeros removed
            (
                country
                + "_"
                + address
                .str.extract(r"(\d+)", 1)
                .fill_null("")
                .str.replace(r"^0+", "")
            ).alias("key11"),

            # Rule 12: country + first two useful address tokens
(
    country
    + "_"
    + address.str.to_lowercase()
        .str.extract_all(r"[a-z]{4,}")
        .list.eval(
            pl.element().filter(
                ~pl.element().is_in([
                    "street", "road", "drive", "avenue", "lane",
                    "court", "place", "boulevard", "highway",
                    "north", "south", "east", "west", "india",
                    "city", "county", "floor", "unit", "suite",
                    "near", "area", "district"
                ])
            )
        )
        .list.unique()
        .list.sort()
        .list.head(2)
        .list.join("_")
).alias("key12"),
        ]
    )

    return df



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
                pl.col("source1_entity_id").cast(pl.String),
                pl.col("matched_entity_ids").cast(pl.String),
            ]
        )
        .collect()
    )

    matches = (
        ground_truth
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null("")
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
            & (pl.col("target_id") != "")
        )
        .select(
            [
                "source1_entity_id",
                "target_id",
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
    Read only records whose entity IDs are required.
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
                "business_address",
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

    target_columns = [
    "entity_id",
    "key1",
    "key2",
    "key3",
    "key4",
    "key5",
    "key6",
    "key7",
    "key8",
    "key9",
    "key10",
    "key11",
    "key12"
]

    targets = pl.concat(
        [
            source2.select(target_columns),
            source3.select(target_columns),
        ]
    )

    # --------------------------------------------------
    # Attach Source 1 blocking keys
    # --------------------------------------------------

    source1_keys = source1.rename(
        {
            "entity_id": "source1_entity_id",
            "key1": "source1_key1",
            "key2": "source1_key2",
            "key3": "source1_key3",
            "key4": "source1_key4",
            "key5": "source1_key5",
            "key6": "source1_key6",
            "key7": "source1_key7",
            "key8": "source1_key8",
            "key9": "source1_key9",
            "key10": "source1_key10",
            "key11": "source1_key11",
            "key12": "source1_key12",
        }
    )

    # --------------------------------------------------
    # Attach target blocking keys
    # --------------------------------------------------

    target_keys = targets.rename(
        {
            "entity_id": "target_id",
            "key1": "target_key1",
            "key2": "target_key2",
            "key3": "target_key3",
            "key4": "target_key4",
            "key5": "target_key5",
            "key6": "target_key6",
            "key7": "target_key7",
            "key8": "target_key8",
            "key9": "target_key9",
            "key10": "target_key10",
            "key11": "target_key11",
            "key12": "target_key12",
        }
    )

    # --------------------------------------------------
    # Evaluate blocking
    # --------------------------------------------------

    evaluated = (
        matches
        .join(
            source1_keys,
            on="source1_entity_id",
            how="left",
        )
        .join(
            target_keys,
            on="target_id",
            how="left",
        )
        .with_columns(
            (
                (
                    (pl.col("source1_key1") != "")
                    & (pl.col("target_key1") != "")
                    & (
                        pl.col("source1_key1")
                        == pl.col("target_key1")
                    )
                )
                |
                (
                    (pl.col("source1_key2") != "")
                    & (pl.col("target_key2") != "")
                    & (
                        pl.col("source1_key2")
                        == pl.col("target_key2")
                    )
                )
                |
                (
                    (pl.col("source1_key3") != "")
                    & (pl.col("target_key3") != "")
                    & (
                        pl.col("source1_key3")
                        == pl.col("target_key3")
                    )
                )
                |
                (
                    (pl.col("source1_key4") != "")
                    & (pl.col("target_key4") != "")
                    & (
                        pl.col("source1_key4")
                        == pl.col("target_key4")
                    )
                )
                |
                (
                    (pl.col("source1_key5") != "")
                    & (pl.col("target_key5") != "")
                    & (
                        pl.col("source1_key5")
                        == pl.col("target_key5")
                    )
                )
                |
                (
                    (pl.col("source1_key6") != "")
                    & (pl.col("target_key6") != "")
                    & (
                        pl.col("source1_key6")
                        == pl.col("target_key6")
                    )
                )
                |
                (
                    (pl.col("source1_key7") != "")
                    & (pl.col("target_key7") != "")
                    & (
                        pl.col("source1_key7")
                        == pl.col("target_key7")
                    )
                )
                |
                (
                    (pl.col("source1_key8") != "")
                    & (pl.col("target_key8") != "")
                    & (
                        pl.col("source1_key8")
                        == pl.col("target_key8")
                    )
                )
                |
                (
                    (pl.col("source1_key9") != "")
                    & (pl.col("target_key9") != "")
                    & (
                        pl.col("source1_key9")
                        == pl.col("target_key9")
                    )
                )
                |
                (
                    (pl.col("source1_key10") != "")
                    & (pl.col("target_key10") != "")
                    & (
                        pl.col("source1_key10")
                        == pl.col("target_key10")
                    )
                )
                |
                (
                    (pl.col("source1_key11") != "")
                    & (pl.col("target_key11") != "")
                    & (
                        pl.col("source1_key11")
                        == pl.col("target_key11")
                    )
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

    print("\n" + "=" * 60)
    print("BLOCKING EVALUATION")
    print("=" * 60)

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

    print("=" * 60)


if __name__ == "__main__":
    main()
