import pandas as pd
from normalization import normalize_dataframe


def create_blocking_key(df):
    """
    Create multiple blocking keys for fast candidate generation.
    """

    df = normalize_dataframe(df)

    # Make sure normalized columns are strings
    name = df["business_name_normalized"].fillna("").astype(str)
    country = df["country_normalized"].fillna("").astype(str)

    # Blocking key 1:
    # Country + first character of business name
    df["blocking_key_1"] = (
        country + "_" + name.str[:1]
    )

    # Blocking key 2:
    # Country + first 4 characters of business name
    df["blocking_key_2"] = (
        country + "_" + name.str[:4]
    )

    # Blocking key 3:
    # Country + first 3 characters
    # Useful when names are slightly different
    df["blocking_key_3"] = (
        country + "_" + name.str[:3]
    )

    return df


def build_block_index(df):
    """
    Build an in-memory index:
        blocking_key -> entity IDs

    This lets us retrieve candidates without comparing
    every Source-1 row with every Source-2/3 row.
    """

    df = create_blocking_key(df)

    index = {}

    for key_column in [
        "blocking_key_1",
        "blocking_key_2",
        "blocking_key_3",
    ]:

        grouped = df.groupby(key_column)["entity_id"].apply(list)

        for key, entity_ids in grouped.items():

            if not key:
                continue

            index.setdefault(key_column, {})
            index[key_column][key] = entity_ids

    return df, index


def get_candidates(row, index):
    """
    Retrieve candidate entity IDs for one Source-1 record.
    """

    candidates = set()

    for key_column in [
        "blocking_key_1",
        "blocking_key_2",
        "blocking_key_3",
    ]:

        key = row[key_column]

        if not key:
            continue

        matches = index.get(key_column, {}).get(key, [])

        candidates.update(matches)

    return candidates


if __name__ == "__main__":

    from config import TRAIN_SOURCE1

    df = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        nrows=1000
    )

    df, index = build_block_index(df)

    print("Rows processed:", len(df))

    print("\nBlocking columns:")
    print(
        df[
            [
                "entity_id",
                "business_name",
                "country",
                "blocking_key_1",
                "blocking_key_2",
                "blocking_key_3",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nIndex sizes:")

    for key_column, values in index.items():
        print(
            key_column,
            "→",
            len(values),
            "unique blocks"
        )
