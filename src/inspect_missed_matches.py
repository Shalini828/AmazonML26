import pandas as pd

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from normalization import normalize_text


def get_blocking_keys(name, country):
    name = normalize_text(name)
    country = normalize_text(country)

    keys = set()

    # Rule 1
    if name:
        keys.add(country + "_" + name[0])

    # Rule 2
    words = [
        word
        for word in name.split()
        if len(word) >= 4
    ]

    if words:
        longest_word = max(words, key=len)
        keys.add(country + "_" + longest_word[:4])

    return keys


def load_ground_truth():

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t"
    )

    truth = {}

    for _, row in gt.iterrows():

        source1_id = str(row["source1_entity_id"])
        matches = str(row["matched_entity_ids"])

        if matches.lower() == "nan" or not matches.strip():
            truth[source1_id] = set()
        else:
            truth[source1_id] = {
                x.strip()
                for x in matches.split(",")
                if x.strip()
            }

    return truth


def load_records(path, wanted_ids):

    records = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100_000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ):

        chunk["entity_id"] = chunk["entity_id"].astype(str)

        relevant = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        for _, row in relevant.iterrows():

            records[row["entity_id"]] = {
                "business_name": row["business_name"],
                "business_address": row["business_address"],
                "country": row["country"],
            }

    return records


def main():

    print("Loading ground truth...")

    truth = load_ground_truth()

    # We only need a small number of missed pairs
    missed_pairs = []

    print("Finding candidate misses...")

    # First load all Source 1 records needed by the ground truth.
    source1_ids = set(truth.keys())

    source1_records = load_records(
        TRAIN_SOURCE1,
        source1_ids
    )

    # Get all target IDs appearing in ground truth.
    target_ids = set()

    for matches in truth.values():
        target_ids.update(matches)

    print("Loading Source 2...")
    source2_records = load_records(
        TRAIN_SOURCE2,
        target_ids
    )

    print("Loading Source 3...")
    source3_records = load_records(
        TRAIN_SOURCE3,
        target_ids
    )

    target_records = {}
    target_records.update(source2_records)
    target_records.update(source3_records)

    # Find first 20 missed true matches
    for source1_id, matches in truth.items():

        if source1_id not in source1_records:
            continue

        source1 = source1_records[source1_id]

        source1_keys = get_blocking_keys(
            source1["business_name"],
            source1["country"]
        )

        for target_id in matches:

            if target_id not in target_records:
                continue

            target = target_records[target_id]

            target_keys = get_blocking_keys(
                target["business_name"],
                target["country"]
            )

            if not source1_keys & target_keys:

                missed_pairs.append(
                    (
                        source1_id,
                        target_id,
                        source1,
                        target,
                    )
                )

                if len(missed_pairs) >= 20:
                    break

        if len(missed_pairs) >= 20:
            break

    print("\n" + "=" * 70)
    print("FIRST MISSED TRUE MATCHES")
    print("=" * 70)

    for (
        source1_id,
        target_id,
        source1,
        target
    ) in missed_pairs:

        print("\nSource 1:", source1_id)
        print("  Name:    ", source1["business_name"])
        print("  Address: ", source1["business_address"])
        print("  Country: ", source1["country"])
        print("  Keys:    ", get_blocking_keys(
            source1["business_name"],
            source1["country"]
        ))

        print("\nMatched target:", target_id)
        print("  Name:    ", target["business_name"])
        print("  Address: ", target["business_address"])
        print("  Country: ", target["country"])
        print("  Keys:    ", get_blocking_keys(
            target["business_name"],
            target["country"]
        ))

        print("-" * 70)


if __name__ == "__main__":
    main()