from pathlib import Path
import polars as pl
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"

S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"

CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"
MATCH_OUTPUT = OUTPUT_DIR / "matching_results.tsv"


# ============================================================
# FAST POLARS NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = [
    "incorporated",
    "corporation",
    "company",
    "private",
    "limited",
    "proprietor",
    "inc",
    "corp",
    "llc",
    "ltd",
    "llp",
    "plc",
    "pvt",
    "opc",
    "huf",
    "prop",
    "co",
]


def normalized_expr(column: str) -> pl.Expr:
    """
    Fast native Polars normalization.

    No Python apply/map_elements on millions of rows.
    """

    expr = (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.to_lowercase()
    )

    # Ampersand -> and
    expr = expr.str.replace_all("&", " and ")

    # Punctuation -> space
    expr = expr.str.replace_all(r"[^[:alnum:][:space:]]", " ")

    # Collapse whitespace
    expr = expr.str.replace_all(r"\s+", " ")

    return expr.str.strip_chars()


def name_expr() -> pl.Expr:
    """
    Normalize business names and remove common legal suffixes.
    """

    expr = normalized_expr("business_name")

    for suffix in LEGAL_SUFFIXES:
        expr = expr.str.replace_all(
            rf"(?i)\b{re.escape(suffix)}\b",
            " ",
        )

    expr = expr.str.replace_all(r"\s+", " ")
    return expr.str.strip_chars()


def address_expr() -> pl.Expr:
    """
    Normalize addresses.
    """

    expr = normalized_expr("business_address")

    replacements = {
        r"\brd\b": "road",
        r"\bst\b": "street",
        r"\bave\b": "avenue",
        r"\bav\b": "avenue",
        r"\bblvd\b": "boulevard",
        r"\bdr\b": "drive",
        r"\bln\b": "lane",
        r"\bct\b": "court",
        r"\bpl\b": "place",
        r"\bsq\b": "square",
        r"\bhwy\b": "highway",
        r"\bpkwy\b": "parkway",
        r"\bapt\b": "apartment",
        r"\bste\b": "suite",
        r"\bfl\b": "floor",
    }

    for pattern, replacement in replacements.items():
        expr = expr.str.replace_all(pattern, replacement)

    expr = expr.str.replace_all(r"\s+", " ")
    return expr.str.strip_chars()


def country_expr() -> pl.Expr:
    """
    Normalize country names.
    """

    expr = normalized_expr("country")

    return (
        pl.when(expr.is_in(["usa", "united states", "united states of america", "us"]))
        .then(pl.lit("us"))
        .when(expr.is_in(["india", "bharat", "in"]))
        .then(pl.lit("india"))
        .when(expr.is_in(["france", "fr"]))
        .then(pl.lit("france"))
        .otherwise(expr)
    )


# ============================================================
# LOAD + NORMALIZE
# ============================================================

def load_source(path: Path) -> pl.DataFrame:

    print(f"Loading {path.name}...", flush=True)

    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema=False,
        ignore_errors=False,
    )

    required = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    df = df.select(required)

    df = df.with_columns(
        [
            name_expr().alias("name_norm"),
            address_expr().alias("address_norm"),
            country_expr().alias("country_norm"),
        ]
    )

    # Strong matching keys
    df = df.with_columns(
        [
            (
                pl.col("country_norm")
                + pl.lit("|")
                + pl.col("name_norm")
            ).alias("name_key"),

            (
                pl.col("country_norm")
                + pl.lit("|")
                + pl.col("address_norm")
            ).alias("address_key"),

            (
                pl.col("country_norm")
                + pl.lit("|")
                + pl.col("name_norm").str.slice(0, 4)
            ).alias("name4_key"),

            (
                pl.col("country_norm")
                + pl.lit("|")
                + pl.col("name_norm").str.slice(0, 6)
            ).alias("name6_key"),
        ]
    )

    return df


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_candidates(
    source1: pl.DataFrame,
    targets: pl.DataFrame,
) -> pl.DataFrame:

    print("Building exact-name candidates...", flush=True)

    # --------------------------------------------------------
    # 1. Exact normalized name + country
    # --------------------------------------------------------

    name_candidates = (
        source1
        .select(
            [
                "entity_id",
                "name_key",
            ]
        )
        .filter(pl.col("name_key") != "")
        .join(
            targets.select(
                [
                    "entity_id",
                    "name_key",
                ]
            ),
            on="name_key",
            how="inner",
            suffix="_target",
        )
        .select(
            [
                pl.col("entity_id").alias("source1_entity_id"),
                pl.col("entity_id_target").alias("candidate_entity_id"),
            ]
        )
    )

    print(
        f"Exact-name candidate rows: {name_candidates.height:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # 2. Exact normalized address + country
    # --------------------------------------------------------

    print("Building exact-address candidates...", flush=True)

    address_candidates = (
        source1
        .select(
            [
                "entity_id",
                "address_key",
            ]
        )
        .filter(pl.col("address_key") != "")
        .join(
            targets.select(
                [
                    "entity_id",
                    "address_key",
                ]
            ),
            on="address_key",
            how="inner",
            suffix="_target",
        )
        .select(
            [
                pl.col("entity_id").alias("source1_entity_id"),
                pl.col("entity_id_target").alias("candidate_entity_id"),
            ]
        )
    )

    print(
        f"Exact-address candidate rows: {address_candidates.height:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # 3. Exact name + exact address
    #    Very high precision.
    # --------------------------------------------------------

    print("Building exact name+address candidates...", flush=True)

    strong_candidates = (
        source1
        .select(
            [
                "entity_id",
                "name_key",
                "address_key",
            ]
        )
        .filter(
            (pl.col("name_key") != "")
            & (pl.col("address_key") != "")
        )
        .join(
            targets.select(
                [
                    "entity_id",
                    "name_key",
                    "address_key",
                ]
            ),
            on=["name_key", "address_key"],
            how="inner",
            suffix="_target",
        )
        .select(
            [
                pl.col("entity_id").alias("source1_entity_id"),
                pl.col("entity_id_target").alias("candidate_entity_id"),
            ]
        )
    )

    print(
        f"Strong candidates: {strong_candidates.height:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Combine candidates
    # --------------------------------------------------------

    candidates = (
        pl.concat(
            [
                name_candidates,
                address_candidates,
                strong_candidates,
            ],
            how="vertical",
        )
        .unique(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
    )

    print(
        f"Unique candidate pairs: {candidates.height:,}",
        flush=True,
    )

    return candidates


# ============================================================
# SCORE CANDIDATES
# ============================================================

def build_matches(
    source1: pl.DataFrame,
    targets: pl.DataFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:

    print("Scoring candidates...", flush=True)

    s1 = source1.select(
        [
            pl.col("entity_id").alias("source1_entity_id"),
            "name_key",
            "address_key",
            "country_norm",
            "name_norm",
            "address_norm",
        ]
    )

    tgt = targets.select(
        [
            pl.col("entity_id").alias("candidate_entity_id"),
            "name_key",
            "address_key",
            "country_norm",
            "name_norm",
            "address_norm",
        ]
    )

    scored = (
        candidates
        .join(s1, on="source1_entity_id", how="left")
        .join(tgt, on="candidate_entity_id", how="left", suffix="_target")
        .with_columns(
            [
                (
                    pl.col("name_key")
                    == pl.col("name_key_target")
                ).cast(pl.Int8).alias("exact_name"),

                (
                    pl.col("address_key")
                    == pl.col("address_key_target")
                ).cast(pl.Int8).alias("exact_address"),

                (
                    pl.col("country_norm")
                    == pl.col("country_norm_target")
                ).cast(pl.Int8).alias("same_country"),

                (
                    pl.col("name_norm")
                    .str.len_chars()
                    ==
                    pl.col("name_norm_target")
                    .str.len_chars()
                ).cast(pl.Int8).alias("same_name_length"),
            ]
        )
        .with_columns(
            (
                pl.col("exact_name") * 100
                + pl.col("exact_address") * 80
                + pl.col("same_country") * 20
                + pl.col("same_name_length") * 5
            ).alias("score")
        )
    )

    # --------------------------------------------------------
    # Important:
    # Keep all exact-name matches when the normalized name is
    # reasonably specific.
    #
    # For very common names, require address agreement.
    # This protects precision/F0.5.
    # --------------------------------------------------------

    name_frequency = (
        targets
        .group_by("name_key")
        .agg(
            pl.len().alias("target_name_count")
        )
    )

    scored = scored.join(
        name_frequency,
        on="name_key_target",
        how="left",
    )

    # Rules:
    #
    # A) exact name + same country and name occurs <= 10 times
    # B) exact name + exact address
    # C) exact address + same country
    #
    matched = scored.filter(
        (
            (
                (pl.col("exact_name") == 1)
                & (pl.col("same_country") == 1)
                & (pl.col("target_name_count") <= 10)
            )
            |
            (
                (pl.col("exact_name") == 1)
                & (pl.col("exact_address") == 1)
            )
            |
            (
                (pl.col("exact_address") == 1)
                & (pl.col("same_country") == 1)
            )
        )
    )

    print(
        f"Candidate matches before grouping: {matched.height:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Group back to one row per Source-1.
    # --------------------------------------------------------

    result = (
        matched
        .group_by("source1_entity_id")
        .agg(
            pl.col("candidate_entity_id")
            .unique()
            .sort()
            .str.join(",")
            .alias("matched_entity_ids")
        )
    )

    return result


# ============================================================
# WRITE OUTPUTS
# ============================================================

def write_outputs(
    source1: pl.DataFrame,
    candidates: pl.DataFrame,
    matches: pl.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Candidate output
    # --------------------------------------------------------

    candidate_output = (
        source1
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id")
        )
        .join(
            candidates
            .group_by("source1_entity_id")
            .agg(
                pl.col("candidate_entity_id")
                .unique()
                .sort()
                .str.join(",")
                .alias("candidate_entity_ids")
            ),
            on="source1_entity_id",
            how="left",
        )
        .with_columns(
            pl.col("candidate_entity_ids")
            .fill_null("")
        )
    )

    print("Writing candidate_pairs.tsv...", flush=True)

    candidate_output.write_csv(
        CANDIDATE_OUTPUT,
        separator="\t",
    )

    # --------------------------------------------------------
    # Matching output
    # --------------------------------------------------------

    matching_output = (
        source1
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id")
        )
        .join(
            matches,
            on="source1_entity_id",
            how="left",
        )
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null("")
        )
    )

    print("Writing matching_results.tsv...", flush=True)

    matching_output.write_csv(
        MATCH_OUTPUT,
        separator="\t",
    )

    print()
    print("=" * 60)
    print("MATCHING COMPLETE")
    print("=" * 60)
    print(
        f"Source 1 rows:       {source1.height:,}"
    )
    print(
        f"Candidate pairs:     {candidates.height:,}"
    )
    print(
        f"Matched S1 entities:  {matches.height:,}"
    )
    print(
        f"Candidate file:      {CANDIDATE_OUTPUT}"
    )
    print(
        f"Matching file:       {MATCH_OUTPUT}"
    )
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FAST POLARS ENTITY MATCHER")
    print("=" * 60)

    source1 = load_source(S1_PATH)

    source2 = load_source(S2_PATH)
    source3 = load_source(S3_PATH)

    print()
    print(
        f"Source 1: {source1.height:,}"
    )
    print(
        f"Source 2: {source2.height:,}"
    )
    print(
        f"Source 3: {source3.height:,}"
    )

    print()
    print("Combining target sources...", flush=True)

    targets = pl.concat(
        [
            source2,
            source3,
        ],
        how="vertical",
    )

    print(
        f"Total targets: {targets.height:,}",
        flush=True,
    )

    print()
    print("Generating candidates...", flush=True)

    candidates = build_candidates(
        source1,
        targets,
    )

    print()
    print("Generating matches...", flush=True)

    matches = build_matches(
        source1,
        targets,
        candidates,
    )

    print()
    print("Writing outputs...", flush=True)

    write_outputs(
        source1,
        candidates,
        matches,
    )


if __name__ == "__main__":
    main()