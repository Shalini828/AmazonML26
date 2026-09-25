import pandas as pd

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from normalization import normalize_text


def get_blocking_keys(name, country):
    """
    Generate the same blocking keys used in blocking.py.
    """

    name = normalize_text(name)
    country = normalize_text(country)

    # Rule 1: country + first character
    key1 = ""
    if name:
        key1 = country + "_" + name[0]

    # Rule 2: country + first 4 characters
    # of the longest meaningful word
    words = [
        word
        for word in name.split()
        if len(word) >= 4
    ]

    key2 = ""

    if words:
        longest_word = max(words, key=len)
        key2 = country + "_" + longest_word[:4]

    return {key1, key2} - {""}


def load_ground_truth():
    """
    Load ground truth and create:
        source1_id -> set of true matched source2/source3 IDs
    """

    ground_truth = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t"
    )

    truth = {}

    for _, row in ground_truth.iterrows():
        source1_id = str(row["source1_entity_id"])

        matches = str(row["matched_entity_ids"])

        if matches.lower() == "nan" or not matches.strip():
            truth[source1_id] = set()
        else:
            truth[source1_id] = {
                match.strip()
                for match in matches.split(",")
                if match.strip()
            }

    return truth


def load_source_keys(path, required_ids):
    """
    Read a source file in chunks and calculate blocking keys
    only for IDs that actually appear in the ground truth.
    """

    result = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100_000,
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ]
    ):
        chunk["entity_id"] = chunk["entity_id"].astype(str)

        relevant = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in relevant.iterrows():
            result[row["entity_id"]] = get_blocking_keys(
                row["business_name"],
                row["country"]
            )

    return result


def main():

    print("Loading ground truth...")

    truth = load_ground_truth()

    print("Source 1 entities in ground truth:", len(truth))

    # IDs that appear as true matches
    target_ids = set()

    for matches in truth.values():
        target_ids.update(matches)

    print("True matched Source 2/3 IDs:", len(target_ids))

    print("\nReading Source 1...")
    source1_keys = load_source_keys(
        TRAIN_SOURCE1,
        set(truth.keys())
    )

    print("Source 1 records loaded:", len(source1_keys))

    print("\nReading Source 2...")
    source2_keys = load_source_keys(
        TRAIN_SOURCE2,
        target_ids
    )

    print("Relevant Source 2 records:", len(source2_keys))

    print("\nReading Source 3...")
    source3_keys = load_source_keys(
        TRAIN_SOURCE3,
        target_ids
    )

    print("Relevant Source 3 records:", len(source3_keys))

    target_keys = {}
    target_keys.update(source2_keys)
    target_keys.update(source3_keys)

    # --------------------------------------------------
    # Evaluate blocking recall
    # --------------------------------------------------

    total_true_matches = 0
    retained_true_matches = 0
    missing_matches = []

    for source1_id, true_matches in truth.items():

        if source1_id not in source1_keys:
            continue

        source1_blocking_keys = source1_keys[source1_id]

        for target_id in true_matches:

            total_true_matches += 1

            if target_id not in target_keys:
                missing_matches.append(
                    (source1_id, target_id, "target_not_found")
                )
                continue

            target_blocking_keys = target_keys[target_id]

            # Candidate survives if ANY blocking rule matches
            if source1_blocking_keys & target_blocking_keys:
                retained_true_matches += 1
            else:
                missing_matches.append(
                    (source1_id, target_id, "blocked_out")
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

    print(f"Total true matches:       {total_true_matches:,}")
    print(f"Retained by blocking:     {retained_true_matches:,}")
    print(f"Lost during blocking:     {len(missing_matches):,}")
    print(f"Blocking recall:          {recall:.2f}%")

    print("=" * 50)

    if missing_matches:
        print("\nFirst 10 missed matches:")

        for item in missing_matches[:10]:
            print(item)


if __name__ == "__main__":
    main()