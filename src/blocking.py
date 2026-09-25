import polars as pl

from normalization import normalize_text


BLOCKING_KEYS = [
    "blocking_key_1",
    "blocking_key_2",
    "blocking_key_3",
]


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add normalized columns using the existing normalization logic.
    """

    return df.with_columns(
        [
            pl.col("business_name")
            .map_elements(normalize_text, return_dtype=pl.String)
            .alias("business_name_normalized"),

            pl.col("business_address")
            .map_elements(normalize_text, return_dtype=pl.String)
            .alias("business_address_normalized"),

            pl.col("country")
            .map_elements(normalize_text, return_dtype=pl.String)
            .alias("country_normalized"),
        ]
    )


def create_blocking_key(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create multiple blocking keys for candidate generation.
    """

    df = normalize_columns(df)

    name = pl.col("business_name_normalized").fill_null("")
    country = pl.col("country_normalized").fill_null("")

    return df.with_columns(
        [
            (country + "_" + name.str.slice(0, 1))
            .alias("blocking_key_1"),

            (country + "_" + name.str.slice(0, 4))
            .alias("blocking_key_2"),

            (country + "_" + name.str.slice(0, 3))
            .alias("blocking_key_3"),
        ]
    )


def build_block_index(df: pl.DataFrame):
    """
    Build blocking indexes:

        blocking_key -> entity IDs

    """

    df = create_blocking_key(df)

    index = {}

    for key_column in BLOCKING_KEYS:

        grouped = (
            df.filter(pl.col(key_column) != "")
            .group_by(key_column)
            .agg(pl.col("entity_id"))
        )

        index[key_column] = {
            row[key_column]: row["entity_id"]
            for row in grouped.iter_rows(named=True)
        }

    return df, index


def get_candidates(row, index):
    """
    Retrieve candidate entity IDs for one Source-1 record.
    """

    candidates = set()

    for key_column in BLOCKING_KEYS:

        key = row[key_column]

        if not key:
            continue

        candidates.update(
            index.get(key_column, {}).get(key, [])
        )

    return candidates


if __name__ == "__main__":

    from config import TRAIN_SOURCE1

    df = pl.read_csv(
        TRAIN_SOURCE1,
        separator="\t",
        n_rows=1000,
    )

    df, index = build_block_index(df)

    print("Rows processed:", df.height)

    print("\nBlocking columns:")

    print(
        df.select(
            [
                "entity_id",
                "business_name",
                "country",
                "blocking_key_1",
                "blocking_key_2",
                "blocking_key_3",
            ]
        ).head(10)
    )

    print("\nIndex sizes:")

    for key_column, values in index.items():
        print(
            key_column,
            "->",
            len(values),
            "unique blocks",
        )