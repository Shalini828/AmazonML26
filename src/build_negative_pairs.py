import pandas as pd
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSITIVE_PATH = PROJECT_ROOT / "data" / "training_positive_pairs.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_pairs.tsv"

RANDOM_SEED = 42


def main():

    random.seed(RANDOM_SEED)

    print("Loading positive pairs...")

    positive = pd.read_csv(
        POSITIVE_PATH,
        sep="\t"
    )

    print("Positive pairs:", len(positive))

    # ---------------------------------------------------------
    # Build ground-truth lookup
    # source1_entity_id -> set of valid matched candidate IDs
    # ---------------------------------------------------------

    true_matches = (
        positive
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    # ---------------------------------------------------------
    # Create shuffled candidate pool
    # ---------------------------------------------------------

    candidate_pool = positive[
        [
            "candidate_entity_id",
            "candidate_name",
            "candidate_address",
            "candidate_country"
        ]
    ].drop_duplicates(
        subset=["candidate_entity_id"]
    ).reset_index(drop=True)

    print("Unique candidate entities:", len(candidate_pool))

    # ---------------------------------------------------------
    # Generate negative pairs
    # ---------------------------------------------------------

    negative_rows = []

    candidate_count = len(candidate_pool)

    for _, row in positive.iterrows():

        source1_id = row["source1_entity_id"]

        # Try random candidates until we find
        # one that is NOT a true match.
        for _ in range(20):

            idx = random.randrange(candidate_count)

            candidate = candidate_pool.iloc[idx]

            candidate_id = candidate["candidate_entity_id"]

            if candidate_id not in true_matches.get(
                source1_id,
                set()
            ):
                negative_rows.append({
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                    "s1_name": row["s1_name"],
                    "s1_address": row["s1_address"],
                    "s1_country": row["s1_country"],
                    "candidate_name": candidate["candidate_name"],
                    "candidate_address": candidate["candidate_address"],
                    "candidate_country": candidate["candidate_country"],
                    "label": 0
                })

                break

    negative = pd.DataFrame(negative_rows)

    print("Negative pairs:", len(negative))

    # ---------------------------------------------------------
    # Combine positive + negative
    # ---------------------------------------------------------

    combined = pd.concat(
        [
            positive,
            negative
        ],
        ignore_index=True
    )

    # Shuffle final dataset
    combined = combined.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print("\n" + "=" * 60)
    print("TRAINING DATASET CREATED")
    print("=" * 60)

    print("Total pairs:", len(combined))

    print("\nLabel distribution:")
    print(combined["label"].value_counts())

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()