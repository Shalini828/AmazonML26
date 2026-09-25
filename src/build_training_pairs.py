from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_ROOT / "dataset" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

OUTPUT_PATH = DATA_ROOT / "training_positive_pairs.tsv"


# ============================================================
# SETTINGS
# ============================================================

# For development, don't generate millions of pairs.
# Increase later after the pipeline works.
MAX_POSITIVE_PAIRS = 200_000

CHUNK_SIZE = 100_000


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_source_data():

    print("Loading source tables...")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    s3 = pd.read_csv(
        S3_PATH,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    print("Source 1:", len(s1))
    print("Source 2:", len(s2))
    print("Source 3:", len(s3))

    return s1, s2, s3


# ============================================================
# CREATE LOOKUPS
# ============================================================

def create_lookup(df, prefix):

    result = df.copy()

    result = result.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "candidate_name",
            "business_address": "candidate_address",
            "country": "candidate_country"
        }
    )

    return result


# ============================================================
# BUILD POSITIVE PAIRS
# ============================================================

def build_positive_pairs():

    s1, s2, s3 = load_source_data()

    # Create fast O(1) lookup for Source 1
    s1_lookup = s1.set_index("entity_id")

    # Combine S2 and S3
    candidates = pd.concat(
        [
            create_lookup(s2, "S2"),
            create_lookup(s3, "S3")
        ],
        ignore_index=True
    )

    print("Combined candidate records:", len(candidates))

    # Fast lookup for S2/S3
    candidates = candidates.set_index("candidate_entity_id")

    output_parts = []

    total_pairs = 0

    print("\nProcessing ground truth in chunks...")

    for chunk_number, gt in enumerate(
        pd.read_csv(
            GT_PATH,
            sep="\t",
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(f"Processing ground-truth chunk {chunk_number}...")

        # Remove empty matches
        gt = gt.dropna(
            subset=["matched_entity_ids"]
        )

        gt = gt[
            gt["matched_entity_ids"]
            .astype(str)
            .str.strip()
            != ""
        ]

        if len(gt) == 0:
            continue

        # One row per matched entity
        gt = gt.assign(
            candidate_entity_id=
            gt["matched_entity_ids"]
            .astype(str)
            .str.split(",")
        )

        gt = gt.explode(
            "candidate_entity_id"
        )

        gt["candidate_entity_id"] = (
            gt["candidate_entity_id"]
            .astype(str)
            .str.strip()
        )

        # Join Source 1 information
        gt = gt.join(
            s1_lookup[
                [
                    "business_name",
                    "business_address",
                    "country"
                ]
            ],
            on="source1_entity_id"
        )

        gt = gt.rename(
            columns={
                "business_name": "s1_name",
                "business_address": "s1_address",
                "country": "s1_country"
            }
        )

        # Join S2/S3 candidate information
        gt = gt.join(
            candidates[
                [
                    "candidate_name",
                    "candidate_address",
                    "candidate_country"
                ]
            ],
            on="candidate_entity_id"
        )

        # Remove candidates that could not be found
        gt = gt.dropna(
            subset=["candidate_name"]
        )

        gt["label"] = 1

        result = gt[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "s1_name",
                "s1_address",
                "s1_country",
                "candidate_name",
                "candidate_address",
                "candidate_country",
                "label"
            ]
        ]

        output_parts.append(result)

        total_pairs += len(result)

        print(
            f"  Positive pairs collected: {total_pairs:,}"
        )

        # Stop during development
        if total_pairs >= MAX_POSITIVE_PAIRS:
            print(
                f"\nReached development limit "
                f"of {MAX_POSITIVE_PAIRS:,} pairs."
            )
            break

    if not output_parts:
        print("No positive pairs generated.")
        return

    pairs = pd.concat(
        output_parts,
        ignore_index=True
    )

    # Exact limit
    pairs = pairs.head(
        MAX_POSITIVE_PAIRS
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pairs.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print("\n" + "=" * 60)
    print("POSITIVE PAIR GENERATION COMPLETE")
    print("=" * 60)

    print(
        "Positive pairs:",
        len(pairs)
    )

    print(
        "Saved to:",
        OUTPUT_PATH
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_positive_pairs()