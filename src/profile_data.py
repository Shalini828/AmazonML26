import pandas as pd

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)


def profile_source(name, path):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    total_rows = 0
    missing_name = 0
    missing_address = 0
    missing_country = 0
    country_counts = {}

    for chunk in pd.read_csv(path, sep="\t", chunksize=100_000):

        total_rows += len(chunk)

        missing_name += chunk["business_name"].isna().sum()
        missing_address += chunk["business_address"].isna().sum()
        missing_country += chunk["country"].isna().sum()

        counts = chunk["country"].value_counts(dropna=False)

        for country, count in counts.items():
            country = str(country)
            country_counts[country] = (
                country_counts.get(country, 0) + int(count)
            )

    print("Rows:", total_rows)
    print("Missing business_name:", missing_name)
    print("Missing business_address:", missing_address)
    print("Missing country:", missing_country)

    print("\nCountry distribution:")
    for country, count in sorted(
        country_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {country}: {count}")


def profile_ground_truth(path):
    print("\n" + "=" * 80)
    print("TRAIN GROUND TRUTH")
    print("=" * 80)

    total_rows = 0
    zero_matches = 0
    match_counts = []

    for chunk in pd.read_csv(path, sep="\t", chunksize=100_000):

        total_rows += len(chunk)

        for value in chunk["matched_entity_ids"].fillna(""):
            value = str(value).strip()

            if not value:
                count = 0
            else:
                count = len(value.split(","))

            match_counts.append(count)

            if count == 0:
                zero_matches += 1

    print("Source 1 entities:", total_rows)
    print("Entities with zero matches:", zero_matches)

    if match_counts:
        print("Average matches per Source 1:", sum(match_counts) / len(match_counts))
        print("Maximum matches for one Source 1:", max(match_counts))


profile_source("TRAIN SOURCE 1", TRAIN_SOURCE1)
profile_source("TRAIN SOURCE 2", TRAIN_SOURCE2)
profile_source("TRAIN SOURCE 3", TRAIN_SOURCE3)

profile_source("TEST SOURCE 1", TEST_SOURCE1)
profile_source("TEST SOURCE 2", TEST_SOURCE2)
profile_source("TEST SOURCE 3", TEST_SOURCE3)

profile_ground_truth(TRAIN_GROUND_TRUTH)