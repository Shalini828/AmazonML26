import pandas as pd
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSITIVE_PATH = PROJECT_ROOT / "data" / "training_positive_pairs.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_pairs.tsv"

RANDOM_SEED = 42

# Number of negative candidates to test per positive row.
# Keep this small so the script runs comfortably on a laptop.
CANDIDATES_PER_ROW = 5


def normalize_text(value):
    """
    Normalize text for simple comparison.
    """
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    return " ".join(value.split())


def simple_similarity(a, b):
    """
    Lightweight similarity based on exact normalized equality
    and substring/token overlap.

    This is intentionally much cheaper than running RapidFuzz
    extraction for every source against 200,000 candidates.
    """

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    # Exact match
    if a == b:
        return 1.0

    # Substring match
    if a in b or b in a:
        return 0.85

    # Token-based Jaccard similarity
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("BUILDING LIGHTWEIGHT HARD NEGATIVE PAIRS")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load positive pairs
    # ---------------------------------------------------------
    print("\nLoading positive pairs...")

    positive = pd.read_csv(
        POSITIVE_PATH,
        sep="\t"
    )

    print("Positive pairs:", len(positive))

    # ---------------------------------------------------------
    # Ground-truth lookup
    # ---------------------------------------------------------
    print("\nBuilding ground-truth lookup...")

    true_matches = (
        positive
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    print("Unique sources:", len(true_matches))

    # ---------------------------------------------------------
    # Candidate pool
    # ---------------------------------------------------------
    print("\nBuilding candidate pool...")

    candidate_columns = [
        "candidate_entity_id",
        "candidate_name",
        "candidate_address",
        "candidate_country",
    ]

    candidate_pool = (
        positive[candidate_columns]
        .drop_duplicates(subset=["candidate_entity_id"])
        .reset_index(drop=True)
    )

    print(
        "Unique candidates:",
        len(candidate_pool)
    )

    # ---------------------------------------------------------
    # Normalize candidate names once
    # ---------------------------------------------------------
    print("\nNormalizing candidate names...")

    candidate_pool["candidate_name_norm"] = (
        candidate_pool["candidate_name"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )

    # ---------------------------------------------------------
    # Generate hard negatives
    # ---------------------------------------------------------
    print("\nGenerating lightweight hard negatives...")

    negative_rows = []

    candidate_count = len(candidate_pool)
    total_rows = len(positive)

    for row_number, (_, row) in enumerate(
        positive.iterrows(),
        start=1
    ):

        source_id = row["source1_entity_id"]

        valid_candidates = true_matches.get(
            source_id,
            set()
        )

        source_name = normalize_text(
            row["s1_name"]
        )

        # -----------------------------------------------------
        # Try a few random candidates
        # -----------------------------------------------------
        selected_negative = None

        for _ in range(CANDIDATES_PER_ROW):

            random_index = random.randrange(
                candidate_count
            )

            candidate = candidate_pool.iloc[
                random_index
            ]

            candidate_id = candidate[
                "candidate_entity_id"
            ]

            # Never select a known positive match.
            if candidate_id in valid_candidates:
                continue

            candidate_name = candidate[
                "candidate_name_norm"
            ]

            similarity = simple_similarity(
                source_name,
                candidate_name
            )

            # Prefer candidates with some name similarity.
            if similarity >= 0.30:
                selected_negative = candidate
                break

            # Keep the first valid candidate as fallback.
            if selected_negative is None:
                selected_negative = candidate

        # -----------------------------------------------------
        # Save negative pair
        # -----------------------------------------------------
        if selected_negative is not None:

            negative_rows.append({
                "source1_entity_id": source_id,

                "candidate_entity_id":
                    selected_negative[
                        "candidate_entity_id"
                    ],

                "s1_name":
                    row["s1_name"],

                "s1_address":
                    row["s1_address"],

                "s1_country":
                    row["s1_country"],

                "candidate_name":
                    selected_negative[
                        "candidate_name"
                    ],

                "candidate_address":
                    selected_negative[
                        "candidate_address"
                    ],

                "candidate_country":
                    selected_negative[
                        "candidate_country"
                    ],

                "label": 0
            })

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------
        if row_number % 10000 == 0:

            print(
                f"Processed: {row_number} / "
                f"{total_rows} | "
                f"Negatives: {len(negative_rows)}"
            )

    # ---------------------------------------------------------
    # Create negative dataframe
    # ---------------------------------------------------------
    negative = pd.DataFrame(
        negative_rows
    )

    print(
        "\nNegative pairs:",
        len(negative)
    )

    # ---------------------------------------------------------
    # Combine positive + negative
    # ---------------------------------------------------------
    print(
        "\nCombining positive and negative pairs..."
    )

    combined = pd.concat(
        [
            positive,
            negative
        ],
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Shuffle
    # ---------------------------------------------------------
    combined = (
        combined
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

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

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------
    print(
        "\n" + "=" * 60
    )

    print(
        "HARD NEGATIVE DATASET CREATED"
    )

    print(
        "=" * 60
    )

    print(
        "Positive pairs:",
        len(positive)
    )

    print(
        "Negative pairs:",
        len(negative)
    )

    print(
        "Total pairs:",
        len(combined)
    )

    print(
        "\nLabel distribution:"
    )

    print(
        combined["label"].value_counts()
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()