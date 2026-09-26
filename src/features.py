from __future__ import annotations

import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio,
    WRatio,
)
from rapidfuzz.distance import JaroWinkler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "training_pairs.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_features.tsv"


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_compact(value) -> str:
    return re.sub(r"\s+", "", normalize_text(value))


def exact_normalized_match(a, b) -> int:
    a = normalize_text(a)
    b = normalize_text(b)
    return int(bool(a) and bool(b) and a == b)


def exact_compact_match(a, b) -> int:
    a = normalize_compact(a)
    b = normalize_compact(b)
    return int(bool(a) and bool(b) and a == b)


# =========================================================
# BASIC SIMILARITY FUNCTIONS
# =========================================================

def sequence_similarity(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def rapid_ratio(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return ratio(a, b) / 100.0


def token_sort_similarity(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return token_sort_ratio(a, b) / 100.0


def token_set_similarity(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return token_set_ratio(a, b) / 100.0


def weighted_similarity(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return WRatio(a, b) / 100.0


def jaro_winkler_similarity(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return JaroWinkler.normalized_similarity(a, b)


# =========================================================
# TOKEN FEATURES
# =========================================================

def token_jaccard_similarity(a, b) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def token_overlap_ratio(a, b) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens))


def character_ngram_similarity(a, b, n=3) -> float:
    a = normalize_compact(a)
    b = normalize_compact(b)
    if not a or not b:
        return 0.0

    def grams(s):
        if len(s) < n:
            return {s}
        return {s[i:i+n] for i in range(len(s) - n + 1)}

    ag = grams(a)
    bg = grams(b)
    union = ag | bg
    return len(ag & bg) / len(union) if union else 0.0


def length_ratio(a, b) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


# =========================================================
# ADDRESS / HOUSE NUMBER FEATURES
# =========================================================

def _house_numbers(value):
    return re.findall(r"\d+[a-zA-Z]?", normalize_text(value))


def house_number_match(a, b) -> int:
    aa = set(_house_numbers(a))
    bb = set(_house_numbers(b))
    return int(bool(aa) and bool(bb) and bool(aa & bb))


def house_number_exact_match(a, b) -> int:
    aa = _house_numbers(a)
    bb = _house_numbers(b)
    return int(bool(aa) and bool(bb) and aa == bb)


# =========================================================
# OPTIONAL STRUCTURED FEATURES
# =========================================================

def _first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def add_optional_exact_feature(df, output_name, source_candidates, target_candidates):
    source_col = _first_existing_column(df, source_candidates)
    target_col = _first_existing_column(df, target_candidates)

    if source_col is None or target_col is None:
        df[output_name] = 0
        return df

    source = df[source_col].fillna("").astype(str)
    target = df[target_col].fillna("").astype(str)

    values = []
    for a, b in zip(source, target):
        a_norm = normalize_compact(a)
        b_norm = normalize_compact(b)
        values.append(int(bool(a_norm) and bool(b_norm) and a_norm == b_norm))

    df[output_name] = values
    return df


# =========================================================
# NAME FEATURES
# =========================================================

def create_name_features(df):
    print("Creating name features...")

    source = df["s1_name"].fillna("")
    target = df["candidate_name"].fillna("")

    df["name_exact"] = [
        exact_normalized_match(a, b) for a, b in zip(source, target)
    ]
    df["name_compact_exact"] = [
        exact_compact_match(a, b) for a, b in zip(source, target)
    ]
    df["name_similarity"] = [
        sequence_similarity(a, b) for a, b in zip(source, target)
    ]
    df["name_ratio"] = [
        rapid_ratio(a, b) for a, b in zip(source, target)
    ]
    df["name_token_sort_similarity"] = [
        token_sort_similarity(a, b) for a, b in zip(source, target)
    ]
    df["name_token_similarity"] = [
        token_jaccard_similarity(a, b) for a, b in zip(source, target)
    ]
    df["name_token_set_similarity"] = [
        token_set_similarity(a, b) for a, b in zip(source, target)
    ]
    df["name_token_overlap"] = [
        token_overlap_ratio(a, b) for a, b in zip(source, target)
    ]
    df["name_jaro_winkler"] = [
        jaro_winkler_similarity(a, b) for a, b in zip(source, target)
    ]
    df["name_ngram_similarity"] = [
        character_ngram_similarity(a, b, n=3) for a, b in zip(source, target)
    ]
    df["name_weighted_similarity"] = [
        weighted_similarity(a, b) for a, b in zip(source, target)
    ]
    df["name_length_ratio"] = [
        length_ratio(a, b) for a, b in zip(source, target)
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
        exact_normalized_match(a, b) for a, b in zip(source, target)
    ]
    df["address_compact_exact"] = [
        exact_compact_match(a, b) for a, b in zip(source, target)
    ]
    df["address_similarity"] = [
        sequence_similarity(a, b) for a, b in zip(source, target)
    ]
    df["address_ratio"] = [
        rapid_ratio(a, b) for a, b in zip(source, target)
    ]
    df["address_token_sort_similarity"] = [
        token_sort_similarity(a, b) for a, b in zip(source, target)
    ]
    df["address_token_similarity"] = [
        token_jaccard_similarity(a, b) for a, b in zip(source, target)
    ]
    df["address_token_set_similarity"] = [
        token_set_similarity(a, b) for a, b in zip(source, target)
    ]
    df["address_token_overlap"] = [
        token_overlap_ratio(a, b) for a, b in zip(source, target)
    ]
    df["address_jaro_winkler"] = [
        jaro_winkler_similarity(a, b) for a, b in zip(source, target)
    ]
    df["address_ngram_similarity"] = [
        character_ngram_similarity(a, b, n=3) for a, b in zip(source, target)
    ]
    df["address_weighted_similarity"] = [
        weighted_similarity(a, b) for a, b in zip(source, target)
    ]
    df["address_length_ratio"] = [
        length_ratio(a, b) for a, b in zip(source, target)
    ]
    df["house_number_match"] = [
        house_number_match(a, b) for a, b in zip(source, target)
    ]
    df["house_number_exact_match"] = [
        house_number_exact_match(a, b) for a, b in zip(source, target)
    ]

    return df


# =========================================================
# COUNTRY FEATURES
# =========================================================

def create_country_features(df):
    print("Creating country features...")

    source = df["s1_country"].fillna("").astype(str).map(normalize_text)
    target = df["candidate_country"].fillna("").astype(str).map(normalize_text)

    df["country_match"] = (
        (source != "") & (target != "") & (source == target)
    ).astype(int)

    df["source_country_missing"] = (source == "").astype(int)
    df["candidate_country_missing"] = (target == "").astype(int)
    df["country_both_present"] = (
        (source != "") & (target != "")
    ).astype(int)

    return df


# =========================================================
# OPTIONAL STRUCTURED FIELDS
# =========================================================

def create_optional_structured_features(df):
    print("Creating optional structured-field features...")

    add_optional_exact_feature(
        df, "postal_code_match",
        [
            "s1_postal_code", "s1_postcode", "s1_zip", "s1_zipcode",
            "source_postal_code", "source_postcode",
        ],
        [
            "candidate_postal_code", "candidate_postcode",
            "candidate_zip", "candidate_zipcode",
            "target_postal_code", "target_postcode",
        ],
    )

    add_optional_exact_feature(
        df, "phone_match",
        [
            "s1_phone", "s1_phone_number",
            "source_phone", "source_phone_number",
        ],
        [
            "candidate_phone", "candidate_phone_number",
            "target_phone", "target_phone_number",
        ],
    )

    add_optional_exact_feature(
        df, "email_match",
        ["s1_email", "source_email"],
        ["candidate_email", "target_email"],
    )

    add_optional_exact_feature(
        df, "website_match",
        [
            "s1_website", "s1_url", "source_website", "source_url",
            "s1_domain", "source_domain",
        ],
        [
            "candidate_website", "candidate_url",
            "target_website", "target_url",
            "candidate_domain", "target_domain",
        ],
    )

    return df


# =========================================================
# EXACT MODEL FEATURE ORDER
# =========================================================

FEATURES = [
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

    "country_match",
    "source_country_missing",
    "candidate_country_missing",
    "country_both_present",

    "postal_code_match",
    "phone_match",
    "email_match",
    "website_match",
]


def main():
    print("Loading training pairs...")
    df = pd.read_csv(INPUT_PATH, sep="\t")
    print("Training pairs:", len(df))

    required = [
        "s1_name", "candidate_name",
        "s1_address", "candidate_address",
        "s1_country", "candidate_country",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required training-pair columns: {missing}")

    df = create_name_features(df)
    df = create_address_features(df)
    df = create_country_features(df)
    df = create_optional_structured_features(df)

    missing_features = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"Missing generated model features: {missing_features}")

    output_columns = []
    if "source1_entity_id" in df.columns:
        output_columns.append("source1_entity_id")
    if "candidate_entity_id" in df.columns:
        output_columns.append("candidate_entity_id")

    output_columns.extend(MODEL_FEATURES)

    if "label" in df.columns:
        output_columns.append("label")

    features = df[output_columns]
    features.to_csv(OUTPUT_PATH, sep="\t", index=False)

    print("=" * 60)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 60)
    print("Rows:", len(features))
    print("Columns:", len(features.columns))
    print("Model features:", len(MODEL_FEATURES))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
