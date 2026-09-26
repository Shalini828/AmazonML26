import polars as pl

from normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_pincode,
)


def create_blocking_key(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create high-recall blocking keys.

    Keys:
      1. country + first character of normalized business name
      2. country + first 4 chars of longest meaningful name word
      3. country + first 3 chars of longest meaningful name word
      4. country + first 2 chars of longest meaningful name word
      5. country + pincode
      6. country + first 3 chars of normalized business name
      7. country + exact normalized business name

    Multiple independent keys are intentionally used to improve blocking recall.
    """

    # --------------------------------------------------
    # Normalize fields
    # --------------------------------------------------

    df = df.with_columns(
        [
            pl.col("business_name")
            .map_elements(
                normalize_business_name,
                return_dtype=pl.String,
            )
            .fill_null("")
            .alias("business_name_normalized"),

            pl.col("business_address")
            .map_elements(
                normalize_address,
                return_dtype=pl.String,
            )
            .fill_null("")
            .alias("business_address_normalized"),

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
        pl.col("business_name_normalized")
        .map_elements(
            get_longest_word,
            return_dtype=pl.String,
        )
        .fill_null("")
        .alias("longest_name_word")
    )

    # --------------------------------------------------
    # Blocking keys
    # --------------------------------------------------

    country = pl.col("country_normalized").fill_null("")
    name = pl.col("business_name_normalized").fill_null("")
    longest = pl.col("longest_name_word").fill_null("")
    pincode = pl.col("pincode").fill_null("")
    address = pl.col("business_address_normalized").fill_null("")

    df = df.with_columns(
        [
            # Rule 1:
            # Country + first character of business name
            (
                country
                + pl.lit("_")
                + name.str.slice(0, 1)
            ).alias("blocking_key_1"),

            # Rule 2:
            # Country + first 4 characters of longest meaningful word
            (
                country
                + pl.lit("_")
                + longest.str.slice(0, 4)
            ).alias("blocking_key_2"),

            # Rule 3:
            # Country + first 3 characters of longest meaningful word
            (
                country
                + pl.lit("_")
                + longest.str.slice(0, 3)
            ).alias("blocking_key_3"),

            # Rule 4:
            # Country + first 2 characters of longest meaningful word
            (
                country
                + pl.lit("_")
                + longest.str.slice(0, 2)
            ).alias("blocking_key_4"),

            # Rule 5:
            # Country + pincode
            (
                country
                + pl.lit("_")
                + pincode
            ).alias("blocking_key_5"),

            # Rule 6:
            # Country + first 3 chars of normalized business name
            (
                country
                + pl.lit("_")
                + name.str.slice(0, 3)
            ).alias("blocking_key_6"),

            # Rule 7:
            # Country + exact normalized business name
            (
                country
                + pl.lit("_")
                + name
            ).alias("blocking_key_7"),

            # Rule 8: Country + first address token
            (
                country
                + pl.lit("_")
                + address.str.split(" ").list.first()
            ).alias("blocking_key_8"),

            # Rule 9: Country + first two address tokens
            (
                country
                + pl.lit("_")
                + address.str.split(" ").list.slice(0, 2).list.join("_")
            ).alias("blocking_key_9"),

            # Rule 10: Country + first address number
            (
                country
                + pl.lit("_")
                + address.str.extract(r"(\d+)", 1).fill_null("")
            ).alias("blocking_key_10"),

            # Rule 11: Country + normalized first address number
            (
                country
                + pl.lit("_")
                + address.str.extract(r"(\d+)", 1)
                    .fill_null("")
                    .str.replace(r"^0+", "")
            ).alias("blocking_key_11"),

            # Rule 12: Country + first two useful address tokens
            (
                country
                + pl.lit("_")
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
            ).alias("blocking_key_12"),
                    ]
                )

    return df


def build_block_index(df: pl.DataFrame):
    """
    Build blocking index:

        blocking_key -> entity IDs

    Multiple blocking rules are stored independently.
    """

    df = create_blocking_key(df)

    key_columns = [
        "blocking_key_1",
        "blocking_key_2",
        "blocking_key_3",
        "blocking_key_4",
        "blocking_key_5",
        "blocking_key_6",
        "blocking_key_7",
        "blocking_key_8",
        "blocking_key_9",
        "blocking_key_10",
        "blocking_key_11",
        "blocking_key_12",
    ]

    index = {}

    for key_column in key_columns:
        grouped = (
            df
            .filter(
                pl.col(key_column).is_not_null()
                & (pl.col(key_column) != "")
                & (pl.col(key_column) != "_")
            )
            .group_by(key_column)
            .agg(
                pl.col("entity_id").alias("entity_ids")
            )
        )

        index[key_column] = {
            row[key_column]: row["entity_ids"]
            for row in grouped.iter_rows(named=True)
        }

    return df, index


def get_candidates(row, index):
    """
    Retrieve candidate entity IDs for one Source-1 record.

    A target is a candidate if it shares ANY blocking key.
    """

    candidates = set()

    key_columns = [
        "blocking_key_1",
        "blocking_key_2",
        "blocking_key_3",
        "blocking_key_4",
        "blocking_key_5",
        "blocking_key_6",
        "blocking_key_7",
        "blocking_key_8",
        "blocking_key_9",
        "blocking_key_10",
        "blocking_key_11",
        "blocking_key_12",
    ]

    for key_column in key_columns:
        key = row.get(key_column)

        if key is None or key == "" or key == "_":
            continue

        matches = index.get(key_column, {}).get(key, [])

        candidates.update(matches)

    return candidates


if __name__ == "__main__":
    from config import TRAIN_SOURCE1

    print("Loading training Source 1 sample...")

    df = pl.read_csv(
        TRAIN_SOURCE1,
        separator="\t",
        n_rows=10,
        infer_schema_length=1000,
    )

    print("Creating blocking keys...")

    df = create_blocking_key(df)

    print("\nBlocking columns:")

    print(
        df.select(
            [
                "entity_id",
                "business_name",
                "country",
                "business_name_normalized",
                "country_normalized",
                "pincode",
                "blocking_key_1",
                "blocking_key_2",
                "blocking_key_3",
                "blocking_key_4",
                "blocking_key_5",
                "blocking_key_6",
                "blocking_key_7",
                "blocking_key_8",
                "blocking_key_9",
                "blocking_key_10",
                "blocking_key_11",
                "blocking_key_12",
            ]
        )
    )

    print("\nBlocking smoke test PASSED.")