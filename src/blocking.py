import re

from normalization import normalize_dataframe, normalize_text


def get_address_number(address):
    if address is None:
        return ""

    address = normalize_text(address)
    match = re.search(r"\b\d+\b", address)

    if match:
        return match.group()

    return ""


def create_blocking_key(df):
    """Create basic blocking keys for candidate generation."""
    df = normalize_dataframe(df)

    # Rule 1: Country + first character of business name
    df["blocking_key_1"] = (
        df["country_normalized"]
        + "_"
        + df["business_name_normalized"].str[:1]
    )

    # Rule 2: Country + first 4 characters of the longest name word
    def get_name_prefix(name):
        words = name.split()
        meaningful_words = [word for word in words if len(word) >= 4]

        if not meaningful_words:
            return ""

        longest_word = max(meaningful_words, key=len)
        return longest_word[:4]

    df["name_prefix"] = df["business_name_normalized"].apply(get_name_prefix)
    df["blocking_key_2"] = (
        df["country_normalized"] + "_" + df["name_prefix"]
    )

    # Rule 3: Country + street address number
    df["address_number"] = df["business_address_normalized"].apply(
        get_address_number
    )
    df["blocking_key_3"] = (
        df["country_normalized"] + "_addr_" + df["address_number"]
    )

    return df


if __name__ == "__main__":
    import pandas as pd

    from config import TRAIN_SOURCE1

    df = pd.read_csv(TRAIN_SOURCE1, sep="\t", nrows=10)
    df = create_blocking_key(df)

    print(
        df[
            [
                "entity_id",
                "business_name",
                "country",
                "business_name_normalized",
                "blocking_key_1",
                "blocking_key_2",
            ]
        ].to_string(index=False)
    )
