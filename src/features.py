import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

try:
    from rapidfuzz.fuzz import (
        ratio,
        token_sort_ratio,
        token_set_ratio,
        WRatio,
    )
    from rapidfuzz.distance import JaroWinkler
except ImportError:
    raise ImportError(
        "rapidfuzz is required. Install it with: pip install rapidfuzz"
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "training_pairs.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_features.tsv"


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value):
    """
    Normalize text for comparison.

    - lowercase
    - replace punctuation with spaces
    - keep letters and numbers
    - collapse repeated whitespace
    """
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Replace punctuation/special characters with spaces
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_compact(value):
    """
    Compact normalization used for exact comparisons.
    """
    return re.sub(r"\s+", "", normalize_text(value))


# =========================================================
# BASIC SIMILARITY FUNCTIONS
# =========================================================

def sequence_similarity(a, b):
    """
    SequenceMatcher similarity.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def rapid_ratio(a, b):
    """
    Character-level RapidFuzz ratio.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def token_sort_similarity(a, b):
    """
    Token-sort similarity.

    Useful when word ordering differs.

    Example:
        "ABC Trading Pvt Ltd"
        "Pvt Ltd ABC Trading"
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return token_sort_ratio(a, b) / 100.0


def token_set_similarity(a, b):
    """
    Token-set similarity.

    Useful when one string contains additional words.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return token_set_ratio(a, b) / 100.0


def weighted_similarity(a, b):
    """
    RapidFuzz WRatio.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return WRatio(a, b) / 100.0


def jaro_winkler_similarity(a, b):
    """
    Jaro-Winkler similarity.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return JaroWinkler.normalized_similarity(a, b)


# =========================================================
# TOKEN FEATURES
# =========================================================

def token_jaccard_similarity(a, b):
    """
    Jaccard similarity between word sets.
    """
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def token_overlap_ratio(a, b):
    """
    Measures how much of the smaller token set
    is present in the larger token set.
    """
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    minimum = min(len(a_tokens), len(b_tokens))

    if minimum == 0:
        return 0.0

    return intersection / minimum


# =========================================================
# CHARACTER N-GRAM FEATURES
# =========================================================

def character_ngrams(value, n=3):
    """
    Generate character n-grams.
    """
    value = normalize_compact(value)

    if len(value) < n:
        return set()

    return {
        value[i:i + n]
        for i in range(len(value) - n + 1)
    }


def character_ngram_similarity(a, b, n=3):
    """
    Jaccard similarity over character n-grams.

    Useful for:
    - spelling variations
    - typos
    - abbreviations
    - transliteration differences
    """
    a_grams = character_ngrams(a, n)
    b_grams = character_ngrams(b, n)

    if not a_grams or not b_grams:
        return 0.0

    intersection = len(a_grams & b_grams)
    union = len(a_grams | b_grams)

    if union == 0:
        return 0.0

    return intersection / union


# =========================================================
# EXACT MATCH FEATURES
# =========================================================

def exact_normalized_match(a, b):
    """
    Exact match after normalization.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0

    return int(a == b)


def exact_compact_match(a, b):
    """
    Exact match after removing whitespace and punctuation.
    """
    a = normalize_compact(a)
    b = normalize_compact(b)

    if not a or not b:
        return 0

    return int(a == b)


# =========================================================
# NUMERIC / ADDRESS FEATURES
# =========================================================

def extract_house_numbers(value):
    """
    Extract numeric components from an address.

    Example:
        '123 Main Street Apt 4'
        -> {'123', '4'}
    """
    value = normalize_text(value)

    if not value:
        return set()

    return set(re.findall(r"\b\d+[a-z]?\b", value))


def house_number_match(a, b):
    """
    Detect whether address house numbers overlap.
    """
    a_numbers = extract_house_numbers(a)
    b_numbers = extract_house_numbers(b)

    if not a_numbers or not b_numbers:
        return 0

    return int(bool(a_numbers & b_numbers))


def house_number_exact_match(a, b):
    """
    Stronger house-number feature.

    Returns 1 when the first numeric component matches.
    """
    a_numbers = extract_house_numbers(a)
    b_numbers = extract_house_numbers(b)

    if not a_numbers or not b_numbers:
        return 0

    return int(next(iter(a_numbers)) == next(iter(b_numbers)))


def length_ratio(a, b):
    """
    Ratio of the shorter normalized string to the longer one.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    longer = max(len(a), len(b))
    shorter = min(len(a), len(b))

    if longer == 0:
        return 0.0

    return shorter / longer


# =========================================================
# COLUMN HELPERS
# =========================================================

def get_column(df, possible_names):
    """
    Return the first available column from possible_names.

    This makes the feature generator robust to datasets
    having slightly different structured-field names.
    """
    for name in possible_names:
        if name in df.columns:
            return df[name]

    return None


def add_optional_exact_feature(
    df,
    output_name,
    source_candidates,
    target_candidates,
):
    """
    Add an exact-match feature if both source and target
    columns exist.

    Otherwise create a zero column.
    """
    source_col = get_column(df, source_candidates)
    target_col = get_column(df, target_candidates)

    if source_col is None or target_col is None:
        df[output_name] = 0
        return

    source = source_col.fillna("").astype(str).map(normalize_compact)
    target = target_col.fillna("").astype(str).map(normalize_compact)

    # Do not treat two missing values as a match.
    df[output_name] = (
        (source != "")
        & (target != "")
        & (source == target)
    ).astype(int)


# =========================================================
# NAME FEATURES
# =========================================================

def create_name_features(df):
    print("Creating name features...")

    source = df["s1_name"].fillna("")
    target = df["candidate_name"].fillna("")

    df["name_exact"] = [
        exact_normalized_match(a, b)
        for a, b in zip(source, target)
    ]

    df["name_compact_exact"] = [
        exact_compact_match(a, b)
        for a, b in zip(source, target)
    ]

    df["name_similarity"] = [
        sequence_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["name_ratio"] = [
        rapid_ratio(a, b)
        for a, b in zip(source, target)
    ]

    df["name_token_sort_similarity"] = [
        token_sort_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["name_token_similarity"] = [
        token_jaccard_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["name_token_set_similarity"] = [
        token_set_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["name_token_overlap"] = [
        token_overlap_ratio(a, b)
        for a, b in zip(source, target)
    ]

    df["name_jaro_winkler"] = [
        jaro_winkler_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["name_ngram_similarity"] = [
        character_ngram_similarity(a, b, n=3)
        for a, b in zip(source, target)
    ]

    df["name_weighted_similarity"] = [
        weighted_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["name_length_ratio"] = [
        length_ratio(a, b)
        for a, b in zip(source, target)
    ]

    return df


# =========================================================
# ADDRESS FEATURES
# =========================================================

def create_address_features(df):
    print("Creating address features...")

    source = df["s1_address"].fillna("")
    target = df["candidate_address"].fillna("")

    df["address_exact"] = [
        exact_normalized_match(a, b)
        for a, b in zip(source, target)
    ]

    df["address_compact_exact"] = [
        exact_compact_match(a, b)
        for a, b in zip(source, target)
    ]

    df["address_similarity"] = [
        sequence_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["address_ratio"] = [
        rapid_ratio(a, b)
        for a, b in zip(source, target)
    ]

    df["address_token_sort_similarity"] = [
        token_sort_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["address_token_similarity"] = [
        token_jaccard_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["address_token_set_similarity"] = [
        token_set_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["address_token_overlap"] = [
        token_overlap_ratio(a, b)
        for a, b in zip(source, target)
    ]

    df["address_jaro_winkler"] = [
        jaro_winkler_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["address_ngram_similarity"] = [
        character_ngram_similarity(a, b, n=3)
        for a, b in zip(source, target)
    ]

    df["address_weighted_similarity"] = [
        weighted_similarity(a, b)
        for a, b in zip(source, target)
    ]

    df["address_length_ratio"] = [
        length_ratio(a, b)
        for a, b in zip(source, target)
    ]

    df["house_number_match"] = [
        house_number_match(a, b)
        for a, b in zip(source, target)
    ]

    df["house_number_exact_match"] = [
        house_number_exact_match(a, b)
        for a, b in zip(source, target)
    ]

    return df


# =========================================================
# COUNTRY FEATURES
# =========================================================

def create_country_features(df):
    print("Creating country features...")

    source = df["s1_country"].fillna("").astype(str)
    target = df["candidate_country"].fillna("").astype(str)

    source_norm = source.map(normalize_text)
    target_norm = target.map(normalize_text)

    # Exact country match, but missing + missing is NOT a match.
    df["country_match"] = (
        (source_norm != "")
        & (target_norm != "")
        & (source_norm == target_norm)
    ).astype(int)

    df["source_country_missing"] = (
        source_norm == ""
    ).astype(int)

    df["candidate_country_missing"] = (
        target_norm == ""
    ).astype(int)

    df["country_both_present"] = (
        (source_norm != "")
        & (target_norm != "")
    ).astype(int)

    return df


# =========================================================
# OPTIONAL STRUCTURED FEATURES
# =========================================================

def create_optional_structured_features(df):
    print("Creating optional structured-field features...")

    # -----------------------------------------------------
    # Postal / ZIP code
    # -----------------------------------------------------
    add_optional_exact_feature(
        df,
        "postal_code_match",
        [
            "s1_postal_code",
            "s1_postcode",
            "s1_zip",
            "s1_zipcode",
            "source_postal_code",
            "source_postcode",
        ],
        [
            "candidate_postal_code",
            "candidate_postcode",
            "candidate_zip",
            "candidate_zipcode",
            "target_postal_code",
            "target_postcode",
        ],
    )

    # -----------------------------------------------------
    # Phone
    # -----------------------------------------------------
    add_optional_exact_feature(
        df,
        "phone_match",
        [
            "s1_phone",
            "s1_phone_number",
            "source_phone",
            "source_phone_number",
        ],
        [
            "candidate_phone",
            "candidate_phone_number",
            "target_phone",
            "target_phone_number",
        ],
    )

    # -----------------------------------------------------
    # Email
    # -----------------------------------------------------
    add_optional_exact_feature(
        df,
        "email_match",
        [
            "s1_email",
            "source_email",
        ],
        [
            "candidate_email",
            "target_email",
        ],
    )

    # -----------------------------------------------------
    # Website / domain
    # -----------------------------------------------------
    add_optional_exact_feature(
        df,
        "website_match",
        [
            "s1_website",
            "s1_url",
            "source_website",
            "source_url",
            "s1_domain",
            "source_domain",
        ],
        [
            "candidate_website",
            "candidate_url",
            "target_website",
            "target_url",
            "candidate_domain",
            "target_domain",
        ],
    )

    return df



def main():
    print("Loading training pairs...")

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t"
    )

    print("Training pairs:", len(df))

    # Create all available Member-2 features
    df = create_name_features(df)
    df = create_address_features(df)
    df = create_country_features(df)
    df = create_optional_structured_features(df)

    # All features that are actually generated from the
    # available training-pairs columns.
    feature_columns = [
        "source1_entity_id",
        "candidate_entity_id",

        # Name
        "name_exact",
        "name_compact_exact",
        "name_similarity",
        "name_ratio",
        "name_token_sort_similarity",
        "name_token_similarity",
        "name_token_set_similarity",
        "name_token_overlap",
        "name_jaro_winkler",
        "name_ngram_similarity",
        "name_weighted_similarity",
        "name_length_ratio",

        # Address
        "address_exact",
        "address_compact_exact",
        "address_similarity",
        "address_ratio",
        "address_token_sort_similarity",
        "address_token_similarity",
        "address_token_set_similarity",
        "address_token_overlap",
        "address_jaro_winkler",
        "address_ngram_similarity",
        "address_weighted_similarity",
        "address_length_ratio",
        "house_number_match",
        "house_number_exact_match",

        # Country
        "country_match",
        "source_country_missing",
        "candidate_country_missing",
        "country_both_present",

        # Optional structured fields
        "postal_code_match",
        "phone_match",
        "email_match",
        "website_match",

        # Label
        "label"
    ]

    # Keep only features that were actually created.
    available_columns = [
        column for column in feature_columns
        if column in df.columns
    ]

    features = df[available_columns]

    features.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 60)
    print("Rows:", len(features))
    print("Columns:", len(features.columns))

    print("\nSaved features:")
    print(features.columns.tolist())

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
