import pandas as pd
import re
from pathlib import Path
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "training_pairs.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_features.tsv"


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Keep letters and numbers
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ---------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------

def string_similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def token_similarity(a, b):
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    return intersection / union


# ---------------------------------------------------------
# Feature generation
# ---------------------------------------------------------

def create_features(df):

    print("Creating name similarity...")

    df["name_similarity"] = [
        string_similarity(a, b)
        for a, b in zip(
            df["s1_name"],
            df["candidate_name"]
        )
    ]

    print("Creating address similarity...")

    df["address_similarity"] = [
        string_similarity(a, b)
        for a, b in zip(
            df["s1_address"],
            df["candidate_address"]
        )
    ]

    print("Creating name token similarity...")

    df["name_token_similarity"] = [
        token_similarity(a, b)
        for a, b in zip(
            df["s1_name"],
            df["candidate_name"]
        )
    ]

    print("Creating address token similarity...")

    df["address_token_similarity"] = [
        token_similarity(a, b)
        for a, b in zip(
            df["s1_address"],
            df["candidate_address"]
        )
    ]

    print("Creating country match...")

    df["country_match"] = (
        df["s1_country"].fillna("").str.lower()
        ==
        df["candidate_country"].fillna("").str.lower()
    ).astype(int)

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading training pairs...")

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t"
    )

    print("Training pairs:", len(df))

    df = create_features(df)

    feature_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "name_similarity",
        "address_similarity",
        "name_token_similarity",
        "address_token_similarity",
        "country_match",
        "label"
    ]

    features = df[feature_columns]

    features.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 60)

    print("Rows:", len(features))

    print("\nFeatures:")
    print(
        features[
            [
                "name_similarity",
                "address_similarity",
                "name_token_similarity",
                "address_token_similarity",
                "country_match"
            ]
        ].head()
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()