import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from normalization import normalize_text


SAMPLE_ROWS = 5000
MAX_MISSES = 20

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "output" / "member3_missed_matches.csv"


# ============================================================
# BLOCKING KEYS
# ============================================================

def get_blocking_keys(name, country):
    name = normalize_text(name)
    country = normalize_text(country)

    keys = set()

    if name:
        keys.add(country + "_" + name[0])

    words = [
        word
        for word in name.split()
        if len(word) >= 4
    ]

    if words:
        longest_word = max(words, key=len)
        keys.add(country + "_" + longest_word[:4])

    return keys


# ============================================================
# HELPERS
# ============================================================

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
        ],
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

        if len(records) >= len(wanted_ids):
            break

    return records


def script_types(text):
    """
    Return broad script categories present in the text.
    Used only for error analysis.
    """

    scripts = set()

    for char in str(text):

        if not char.isalpha():
            continue

        name = unicodedata.name(char, "")

        if "LATIN" in name:
            scripts.add("latin")
        elif "DEVANAGARI" in name:
            scripts.add("devanagari")
        elif "TAMIL" in name:
            scripts.add("tamil")
        elif "TELUGU" in name:
            scripts.add("telugu")
        elif "BENGALI" in name:
            scripts.add("bengali")
        elif "GUJARATI" in name:
            scripts.add("gujarati")
        elif "GURMUKHI" in name:
            scripts.add("gurmukhi")
        elif "KANNADA" in name:
            scripts.add("kannada")
        elif "MALAYALAM" in name:
            scripts.add("malayalam")
        else:
            scripts.add("other")

    return scripts


def has_non_latin_difference(name1, name2):
    scripts1 = script_types(name1)
    scripts2 = script_types(name2)

    return (
        scripts1
        and scripts2
        and scripts1 != scripts2
        and (
            "latin" in scripts1
            or "latin" in scripts2
        )
    )


def normalized_tokens(text):
    return set(normalize_text(text).split())


def token_overlap(name1, name2):
    tokens1 = normalized_tokens(name1)
    tokens2 = normalized_tokens(name2)

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def address_overlap(address1, address2):
    tokens1 = normalized_tokens(address1)
    tokens2 = normalized_tokens(address2)

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def classify_error(source1, target):
    """
    Assign a descriptive error category.

    This is for analysis/reporting only.
    It does NOT affect matching.
    """

    name1 = str(source1["business_name"])
    name2 = str(target["business_name"])

    address1 = str(source1["business_address"])
    address2 = str(target["business_address"])

    normalized_name1 = normalize_text(name1)
    normalized_name2 = normalize_text(name2)

    # --------------------------------------------------------
    # Cross-script
    # --------------------------------------------------------

    if has_non_latin_difference(name1, name2):
        return "cross_script"

    # --------------------------------------------------------
    # DBA / AKA / alias indicators
    # --------------------------------------------------------

    combined = (
        name1.lower()
        + " "
        + name2.lower()
    )

    if re.search(
        r"\b(dba|aka|formerly|trading as)\b",
        combined,
    ):
        return "dba_or_alias"

    # --------------------------------------------------------
    # Address-dominant match
    # --------------------------------------------------------

    addr_score = address_overlap(
        address1,
        address2,
    )

    name_score = token_overlap(
        name1,
        name2,
    )

    if addr_score >= 0.5 and name_score < 0.2:
        return "address_dominant"

    # --------------------------------------------------------
    # Word-order / token variation
    # --------------------------------------------------------

    if (
        name_score >= 0.5
        and normalized_name1 != normalized_name2
    ):
        return "word_order_or_name_variation"

    # --------------------------------------------------------
    # Name noise / typo
    # --------------------------------------------------------

    if name_score > 0:
        return "name_noise"

    # --------------------------------------------------------
    # Address variation
    # --------------------------------------------------------

    if addr_score > 0:
        return "address_variation"

    return "mixed"


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Loading first {SAMPLE_ROWS:,} ground-truth rows..."
    )

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        nrows=SAMPLE_ROWS,
    )

    truth = {}

    for _, row in gt.iterrows():

        source1_id = str(row["source1_entity_id"])
        matches = str(row["matched_entity_ids"])

        if (
            matches.lower() == "nan"
            or not matches.strip()
        ):
            truth[source1_id] = set()

        else:
            truth[source1_id] = {
                x.strip()
                for x in matches.split(",")
                if x.strip()
            }

    source1_ids = set(truth.keys())

    target_ids = set()

    for matches in truth.values():
        target_ids.update(matches)

    print(
        f"Source 1 IDs needed : {len(source1_ids):,}"
    )

    print(
        f"Target IDs needed   : {len(target_ids):,}"
    )

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    print("\nLoading required Source 1 records...")

    source1_records = load_records(
        TRAIN_SOURCE1,
        source1_ids,
    )

    print(
        f"Loaded Source 1 records: "
        f"{len(source1_records):,}"
    )

    print("\nLoading required Source 2 records...")

    source2_records = load_records(
        TRAIN_SOURCE2,
        target_ids,
    )

    print(
        f"Loaded Source 2 records: "
        f"{len(source2_records):,}"
    )

    print("\nLoading required Source 3 records...")

    source3_records = load_records(
        TRAIN_SOURCE3,
        target_ids,
    )

    print(
        f"Loaded Source 3 records: "
        f"{len(source3_records):,}"
    )

    target_records = {}

    target_records.update(source2_records)
    target_records.update(source3_records)

    # --------------------------------------------------------
    # Find missed blocking pairs
    # --------------------------------------------------------

    missed_pairs = []

    print("\nSearching for missed blocking pairs...")

    for source1_id, matches in truth.items():

        source1 = source1_records.get(source1_id)

        if source1 is None:
            continue

        source1_keys = get_blocking_keys(
            source1["business_name"],
            source1["country"],
        )

        for target_id in matches:

            target = target_records.get(target_id)

            if target is None:
                continue

            target_keys = get_blocking_keys(
                target["business_name"],
                target["country"],
            )

            if not source1_keys.intersection(target_keys):

                category = classify_error(
                    source1,
                    target,
                )

                missed_pairs.append(
                    {
                        "source1_entity_id": source1_id,
                        "target_entity_id": target_id,
                        "source1_name": source1["business_name"],
                        "target_name": target["business_name"],
                        "source1_address": source1["business_address"],
                        "target_address": target["business_address"],
                        "country": source1["country"],
                        "error_category": category,
                    }
                )

                if len(missed_pairs) >= MAX_MISSES:
                    break

        if len(missed_pairs) >= MAX_MISSES:
            break

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = pd.DataFrame(missed_pairs)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MEMBER 3 — MISSED MATCH ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\nMissed pairs found : {len(report):,}"
    )

    print(
        f"Saved report       : {OUTPUT_FILE}"
    )

    if not report.empty:

        print("\nError categories:")

        counts = (
            report["error_category"]
            .value_counts()
        )

        for category, count in counts.items():

            print(
                f"  {category:30s} {count:>4}"
            )

        print("\nExamples:")

        for _, row in report.iterrows():

            print("\n" + "-" * 70)

            print(
                f"Category : {row['error_category']}"
            )

            print(
                f"S1       : {row['source1_entity_id']}"
            )

            print(
                f"Target   : {row['target_entity_id']}"
            )

            print(
                f"S1 name  : {row['source1_name']}"
            )

            print(
                f"Tgt name : {row['target_name']}"
            )

            print(
                f"S1 addr  : {row['source1_address']}"
            )

            print(
                f"Tgt addr : {row['target_address']}"
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()