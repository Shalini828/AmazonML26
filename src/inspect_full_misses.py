import polars as pl

from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
from normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_pincode,
)


def add_blocking_keys(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("business_name")
            .map_elements(normalize_business_name, return_dtype=pl.String)
            .fill_null("")
            .alias("name_normalized"),

            pl.col("country")
            .map_elements(normalize_country, return_dtype=pl.String)
            .fill_null("")
            .alias("country_normalized"),
        ]
    )

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

    def longest_word(name):
        if not name:
            return ""
        words = [w for w in str(name).split() if len(w) >= 4]
        return max(words, key=len) if words else ""

    df = df.with_columns(
        pl.col("name_normalized")
        .map_elements(longest_word, return_dtype=pl.String)
        .fill_null("")
        .alias("longest_name_word")
    )

    country = pl.col("country_normalized")
    name = pl.col("name_normalized")
    longest = pl.col("longest_name_word")
    pincode = pl.col("pincode")

    address = (
        pl.col("business_address")
        .map_elements(normalize_address, return_dtype=pl.String)
        .fill_null("")
    )

    return df.with_columns(
        [
            (country + "_" + name.str.slice(0, 1)).alias("key1"),
            (country + "_" + longest.str.slice(0, 4)).alias("key2"),
            (country + "_" + longest.str.slice(0, 3)).alias("key3"),
            (country + "_" + longest.str.slice(0, 2)).alias("key4"),
            (country + "_" + pincode).alias("key5"),
            (country + "_" + name.str.slice(0, 3)).alias("key6"),
            (country + "_" + name).alias("key7"),

            (
                country
                + "_"
                + address.str.split(" ").list.first()
            ).alias("key8"),

            (
                country
                + "_"
                + address.str.split(" ").list.slice(0, 2).list.join("_")
            ).alias("key9"),

            (
                country
                + "_"
                + address.str.extract(r"(\d+)", 1).fill_null("")
            ).alias("key10"),

            (
                country
                + "_"
                + address.str.extract(r"(\d+)", 1)
                    .fill_null("")
                    .str.replace(r"^0+", "")
            ).alias("key11"),

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


def main():
    print("Loading ground truth...")

    gt = pl.read_csv(
        TRAIN_GROUND_TRUTH,
        separator="\t",
        infer_schema_length=10000,
    )

    # Ground truth target IDs are comma-separated.
    matches = (
        gt.select(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null("")
            .str.split(",")
            .alias("target_id")
        )
        .explode("target_id")
        .with_columns(
            pl.col("target_id").str.strip_chars()
        )
        .filter(pl.col("target_id") != "")
    )

    print(f"True pairs: {matches.height:,}")

    print("Reading source files...")

    s1 = pl.read_csv(
        TRAIN_SOURCE1,
        separator="\t",
        infer_schema_length=10000,
    )

    s2 = pl.read_csv(
        TRAIN_SOURCE2,
        separator="\t",
        infer_schema_length=10000,
    )

    s3 = pl.read_csv(
        TRAIN_SOURCE3,
        separator="\t",
        infer_schema_length=10000,
    )

    print("Creating blocking keys...")

    s1 = add_blocking_keys(s1)
    s2 = add_blocking_keys(s2)
    s3 = add_blocking_keys(s3)

    targets = pl.concat(
        [
            s2,
            s3,
        ]
    )

    # Only keep columns needed for inspection.
    target_cols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        *[f"key{i}" for i in range(1, 13)],
    ]

    source1_cols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        *[f"key{i}" for i in range(1, 13)],
    ]

    s1 = s1.select(source1_cols)
    targets = targets.select(target_cols)

    print("Joining true pairs...")

    evaluated = (
        matches
        .join(
            s1,
            left_on="source1_entity_id",
            right_on="entity_id",
            how="inner",
        )
        .join(
            targets,
            left_on="target_id",
            right_on="entity_id",
            how="inner",
            suffix="_target",
        )
    )

    # A true pair is retained if ANY blocking key matches.
    retained_expr = None

    for i in range(1, 13):
        expr = (
            (pl.col(f"key{i}") != "")
            & (pl.col(f"key{i}_target") != "")
            & (
                pl.col(f"key{i}")
                == pl.col(f"key{i}_target")
            )
        )

        retained_expr = expr if retained_expr is None else retained_expr | expr

    evaluated = evaluated.with_columns(
        retained_expr.alias("retained")
    )

    misses = (
        evaluated
        .filter(~pl.col("retained"))
        .select(
            [
                "source1_entity_id",
                "target_id",
                "country",
                "country_target",
                "business_name",
                "business_name_target",
                "business_address",
                "business_address_target",
                "key1",
                "key1_target",
                "key2",
                "key2_target",
                "key3",
                "key3_target",
                "key4",
                "key4_target",
                "key5",
                "key5_target",
                "key6",
                "key6_target",
                "key7",
                "key7_target",
                "key8",
                "key8_target",
                "key9",
                "key9_target",
                "key10",
                "key10_target",
                "key11",
                "key11_target",
                "key12",
                "key12_target",
               
            ]
        )
    )

    print()
    print("=" * 70)
    print("FULL TRAIN MISSED TRUE PAIRS")
    print("=" * 70)
    print(f"Total missed pairs: {misses.height:,}")
    print()

    # Save all misses so we don't need to rerun the expensive computation.
    output_path = "data/full_train_blocking_misses.tsv"

    misses.write_csv(
        output_path,
        separator="\t",
    )

    print(f"Saved all misses to: {output_path}")
    print()
    print("First 100 missed pairs:")
    print()

    print(
        misses.head(100).to_pandas().to_string(index=False)
    )


if __name__ == "__main__":
    main()