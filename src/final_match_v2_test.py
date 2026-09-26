"""Build TEST candidates and RF matches with a disk-backed DuckDB pipeline."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
OUT = ROOT / "output"

DB_PATH = OUT / "matcher_test_v2.duckdb"
TEMP_DIR = OUT / "duckdb_temp_test_v2"
MODEL_PATH = DATA / "rf_model.joblib"

S1_PATH = PROCESSED / "test_source1.parquet"
S2_PATH = PROCESSED / "test_source2.parquet"
S3_PATH = PROCESSED / "test_source3.parquet"

CANDIDATE_OUTPUT = OUT / "candidate_pairs.tsv"
MATCH_OUTPUT = OUT / "matching_results.tsv"

MEMORY_LIMIT = "2GB"
THREADS = 4

# Keep the candidate pool bounded.
BLOCK_CAP = 20

MATCH_THRESHOLD = 0.999
MAX_MATCHES_PER_SOURCE = 3

# Larger batch = fewer DuckDB/Pandas registrations.
SCORE_BATCH_SIZE = 25_000


# IMPORTANT:
# This order MUST exactly match rf_model.joblib.
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


# For speed we are using the exact-name block.
# The candidate ranking inside this block uses address/name/country signals.
BLOCKS = [
    ("exact_name", "k_exact", BLOCK_CAP),
]


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def normalized(expr: str) -> str:
    value = f"lower(coalesce(cast({expr} AS VARCHAR), ''))"
    value = f"regexp_replace({value}, '[^[:alnum:][:space:]]', ' ', 'g')"
    return f"trim(regexp_replace({value}, '[[:space:]]+', ' ', 'g'))"


def country_normalized(expr: str) -> str:
    value = normalized(expr)

    return f"""
    CASE
        WHEN {value} IN (
            'usa',
            'united states',
            'united states of america',
            'us'
        ) THEN 'us'

        WHEN {value} IN (
            'india',
            'bharat',
            'in'
        ) THEN 'india'

        WHEN {value} IN (
            'france',
            'fr'
        ) THEN 'france'

        ELSE {value}
    END
    """


def create_normalized_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    path_sql: str,
) -> None:

    n_name = normalized("business_name")
    n_address = normalized("business_address")
    n_country = country_normalized("country")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS

        WITH normalized_rows AS (
            SELECT
                CAST(entity_id AS VARCHAR) AS entity_id,

                COALESCE(
                    CAST(business_name AS VARCHAR),
                    ''
                ) AS business_name,

                COALESCE(
                    CAST(business_address AS VARCHAR),
                    ''
                ) AS business_address,

                COALESCE(
                    CAST(country AS VARCHAR),
                    ''
                ) AS country,

                {n_name} AS n_name,
                {n_address} AS n_address,
                {n_country} AS n_country

            FROM read_parquet('{path_sql}')
        ),

        keyed AS (
            SELECT
                *,

                regexp_extract(
                    regexp_replace(
                        n_name,
                        '^(the|a|an|and|of|company|co|inc|ltd|llc) +',
                        ''
                    ),
                    '^[^ ]+',
                    0
                ) AS name_token,

                regexp_extract(
                    n_address,
                    '^[^ ]+',
                    0
                ) AS address_token,

                regexp_extract(
                    n_address,
                    '[0-9]{{5,6}}',
                    0
                ) AS postal,

                regexp_extract(
                    n_address,
                    '[0-9]+',
                    0
                ) AS address_number

            FROM normalized_rows
        )

        SELECT
            *,

            n_name AS k_exact,

            n_country || '|' || name_token
                AS k_token,

            n_country || '|' || substr(n_name, 1, 2)
                AS k_c2,

            n_country || '|' || substr(n_name, 1, 3)
                AS k_c3,

            n_country || '|' || substr(n_name, 1, 4)
                AS k_c4,

            n_country || '|' || address_token
                AS k_addrtoken,

            n_country || '|' || substr(n_address, 1, 10)
                AS k_addrprefix,

            CASE
                WHEN address_number <> ''
                THEN n_country || '|' || address_number
                ELSE ''
            END AS k_addrnumber,

            n_country || '|' || postal
                AS k_postal

        FROM keyed
        """
    )


def build_tables(con: duckdb.DuckDBPyConnection) -> None:

    create_normalized_table(
        con,
        "source1",
        sql_path(S1_PATH),
    )

    source2 = sql_path(S2_PATH)
    source3 = sql_path(S3_PATH)

    n_name = normalized("business_name")
    n_address = normalized("business_address")
    n_country = country_normalized("country")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE target AS

        WITH raw AS (

            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM read_parquet('{source2}')

            UNION ALL

            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM read_parquet('{source3}')
        ),

        normalized_rows AS (

            SELECT
                CAST(entity_id AS VARCHAR) AS entity_id,

                COALESCE(
                    CAST(business_name AS VARCHAR),
                    ''
                ) AS business_name,

                COALESCE(
                    CAST(business_address AS VARCHAR),
                    ''
                ) AS business_address,

                COALESCE(
                    CAST(country AS VARCHAR),
                    ''
                ) AS country,

                {n_name} AS n_name,
                {n_address} AS n_address,
                {n_country} AS n_country

            FROM raw
        ),

        keyed AS (

            SELECT
                *,

                regexp_extract(
                    regexp_replace(
                        n_name,
                        '^(the|a|an|and|of|company|co|inc|ltd|llc) +',
                        ''
                    ),
                    '^[^ ]+',
                    0
                ) AS name_token,

                regexp_extract(
                    n_address,
                    '^[^ ]+',
                    0
                ) AS address_token,

                regexp_extract(
                    n_address,
                    '[0-9]{{5,6}}',
                    0
                ) AS postal,

                regexp_extract(
                    n_address,
                    '[0-9]+',
                    0
                ) AS address_number

            FROM normalized_rows
        )

        SELECT
            *,

            n_name AS k_exact,

            n_country || '|' || name_token
                AS k_token,

            n_country || '|' || substr(n_name, 1, 2)
                AS k_c2,

            n_country || '|' || substr(n_name, 1, 3)
                AS k_c3,

            n_country || '|' || substr(n_name, 1, 4)
                AS k_c4,

            n_country || '|' || address_token
                AS k_addrtoken,

            n_country || '|' || substr(n_address, 1, 10)
                AS k_addrprefix,

            CASE
                WHEN address_number <> ''
                THEN n_country || '|' || address_number
                ELSE ''
            END AS k_addrnumber,

            n_country || '|' || postal
                AS k_postal

        FROM keyed
        """
    )


def build_candidates(con: duckdb.DuckDBPyConnection) -> None:

    con.execute(
        """
        CREATE OR REPLACE TABLE selected_candidates (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR
        )
        """
    )

    for strategy, key, cap in BLOCKS:

        started = time.perf_counter()

        print(
            f"Generating block {strategy}...",
            flush=True,
        )

        con.execute(
            f"""
            INSERT INTO selected_candidates

            WITH raw AS (

                SELECT
                    s.entity_id AS source1_entity_id,
                    t.entity_id AS candidate_entity_id,

                    (
                        s.n_name <> ''
                        AND s.n_name = t.n_name
                    )::INT AS exact_name,

                    (
                        s.n_address <> ''
                        AND s.n_address = t.n_address
                    )::INT AS address_exact,

                    (
                        s.n_address <> ''
                        AND t.n_address <> ''
                        AND (
                            starts_with(
                                s.n_address,
                                t.n_address
                            )
                            OR starts_with(
                                t.n_address,
                                s.n_address
                            )
                            OR contains(
                                s.n_address,
                                t.n_address
                            )
                            OR contains(
                                t.n_address,
                                s.n_address
                            )
                        )
                    )::INT AS address_partial,

                    (
                        (length(s.n_name) >= 1
                         AND substr(s.n_name,1,1)
                         = substr(t.n_name,1,1))::INT

                        +

                        (length(s.n_name) >= 2
                         AND substr(s.n_name,1,2)
                         = substr(t.n_name,1,2))::INT

                        +

                        (length(s.n_name) >= 3
                         AND substr(s.n_name,1,3)
                         = substr(t.n_name,1,3))::INT

                        +

                        (length(s.n_name) >= 4
                         AND substr(s.n_name,1,4)
                         = substr(t.n_name,1,4))::INT

                        +

                        (length(s.n_name) >= 8
                         AND substr(s.n_name,1,8)
                         = substr(t.n_name,1,8))::INT
                    ) AS name_prefix_agreement,

                    CASE
                        WHEN length(s.n_name)
                           + length(t.n_name) > 0

                        THEN
                            length(
                                list_intersect(
                                    list_distinct(
                                        string_split(
                                            s.n_name,
                                            ' '
                                        )
                                    ),
                                    list_distinct(
                                        string_split(
                                            t.n_name,
                                            ' '
                                        )
                                    )
                                )
                            )::DOUBLE

                            /

                            greatest(
                                1,

                                length(
                                    list_distinct(
                                        string_split(
                                            s.n_name,
                                            ' '
                                        )
                                    )
                                )

                                +

                                length(
                                    list_distinct(
                                        string_split(
                                            t.n_name,
                                            ' '
                                        )
                                    )
                                )

                                -

                                length(
                                    list_intersect(
                                        list_distinct(
                                            string_split(
                                                s.n_name,
                                                ' '
                                            )
                                        ),
                                        list_distinct(
                                            string_split(
                                                t.n_name,
                                                ' '
                                            )
                                        )
                                    )
                                )
                            )

                        ELSE 0
                    END AS name_token_overlap,

                    CASE
                        WHEN length(s.n_address)
                           + length(t.n_address) > 0

                        THEN
                            length(
                                list_intersect(
                                    list_distinct(
                                        string_split(
                                            s.n_address,
                                            ' '
                                        )
                                    ),
                                    list_distinct(
                                        string_split(
                                            t.n_address,
                                            ' '
                                        )
                                    )
                                )
                            )::DOUBLE

                            /

                            greatest(
                                1,

                                length(
                                    list_distinct(
                                        string_split(
                                            s.n_address,
                                            ' '
                                        )
                                    )
                                )

                                +

                                length(
                                    list_distinct(
                                        string_split(
                                            t.n_address,
                                            ' '
                                        )
                                    )
                                )

                                -

                                length(
                                    list_intersect(
                                        list_distinct(
                                            string_split(
                                                s.n_address,
                                                ' '
                                            )
                                        ),
                                        list_distinct(
                                            string_split(
                                                t.n_address,
                                                ' '
                                            )
                                        )
                                    )
                                )
                            )

                        ELSE 0
                    END AS address_token_overlap,

                    (
                        s.n_country <> ''
                        AND s.n_country = t.n_country
                    )::INT AS country_equal

                FROM source1 s

                JOIN target t
                    ON s.{key} = t.{key}

                WHERE s.{key} <> ''
                  AND s.{key} NOT LIKE '%|'
            ),

            ranked AS (

                SELECT
                    *,

                    row_number() OVER (
                        PARTITION BY source1_entity_id

                        ORDER BY
                            exact_name DESC,
                            address_exact DESC,
                            address_partial DESC,
                            name_prefix_agreement DESC,
                            name_token_overlap DESC,
                            address_token_overlap DESC,
                            country_equal DESC,
                            candidate_entity_id
                    ) AS rn

                FROM raw
            )

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM ranked

            WHERE rn <= {cap}
            """
        )

        pair_count = con.execute(
            """
            SELECT count(*)
            FROM (
                SELECT DISTINCT
                    source1_entity_id,
                    candidate_entity_id
                FROM selected_candidates
            )
            """
        ).fetchone()[0]

        print(
            f"  cumulative unique pairs: "
            f"{pair_count:,} | "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )

    con.execute(
        """
        CREATE OR REPLACE TABLE candidate_pairs_internal AS

        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id

        FROM selected_candidates
        """
    )


def score_candidates(
    con: duckdb.DuckDBPyConnection,
    model,
) -> None:

    # Import ONLY once.
    from features import (
        exact_normalized_match,
        exact_compact_match,
        sequence_similarity,
        rapid_ratio,
        token_sort_similarity,
        token_jaccard_similarity,
        token_set_similarity,
        token_overlap_ratio,
        jaro_winkler_similarity,
        character_ngram_similarity,
        weighted_similarity,
        length_ratio,
        house_number_match,
        house_number_exact_match,
        normalize_text,
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE scored_candidates (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR,
            probability DOUBLE
        )
        """
    )

    # IMPORTANT: DuckDB connection.execute() shares the connection's current
    # result. Later INSERT/REGISTER queries would overwrite it, causing the
    # next fetchmany() to return something like (25000,) instead of the
    # candidate row. Use a dedicated cursor for the long-running SELECT.
    query_cursor = con.cursor()
    query = query_cursor.execute(
        """
        SELECT
            CAST(
                c.source1_entity_id AS VARCHAR
            ) AS source1_entity_id,

            CAST(
                c.candidate_entity_id AS VARCHAR
            ) AS candidate_entity_id,

            COALESCE(
                s.n_name,
                ''
            ) AS source_name,

            COALESCE(
                t.n_name,
                ''
            ) AS target_name,

            COALESCE(
                s.n_address,
                ''
            ) AS source_address,

            COALESCE(
                t.n_address,
                ''
            ) AS target_address,

            COALESCE(
                s.n_country,
                ''
            ) AS source_country,

            COALESCE(
                t.n_country,
                ''
            ) AS target_country

        FROM candidate_pairs_internal c

        INNER JOIN source1 s
            ON s.entity_id = c.source1_entity_id

        INNER JOIN target t
            ON t.entity_id = c.candidate_entity_id

        ORDER BY
            c.source1_entity_id,
            c.candidate_entity_id
        """
    )

    batch_number = 0
    total_scored = 0
    started = time.perf_counter()

    while True:

        rows = query.fetchmany(SCORE_BATCH_SIZE)

        if not rows:
            break

        feature_rows = []

        output_source_ids = []
        output_candidate_ids = []

        for row in rows:
            # DuckDB should return 8 columns for every candidate row.
            # Keep this explicit so a malformed query result fails clearly.
            if len(row) != 8:
                raise RuntimeError(
                    f"Expected 8 columns from candidate query, got {len(row)}: {row!r}"
                )

            (
                source_entity_id,
                candidate_entity_id,
                s_name,
                t_name,
                s_addr,
                t_addr,
                s_country,
                t_country,
            ) = row

            feature_values = {
                    "name_exact":
                        exact_normalized_match(
                            s_name,
                            t_name,
                        ),

                    "name_compact_exact":
                        exact_compact_match(
                            s_name,
                            t_name,
                        ),

                    "name_similarity":
                        sequence_similarity(
                            s_name,
                            t_name,
                        ),

                    "name_ratio":
                        rapid_ratio(
                            s_name,
                            t_name,
                        ),

                    "name_token_sort_similarity":
                        token_sort_similarity(
                            s_name,
                            t_name,
                        ),

                    "name_token_similarity":
                        token_jaccard_similarity(
                            s_name,
                            t_name,
                        ),

                    "name_token_set_similarity":
                        token_set_similarity(
                            s_name,
                            t_name,
                        ),

                    "name_token_overlap":
                        token_overlap_ratio(
                            s_name,
                            t_name,
                        ),

                    "name_jaro_winkler":
                        jaro_winkler_similarity(
                            s_name,
                            t_name,
                        ),

                    "name_ngram_similarity":
                        character_ngram_similarity(
                            s_name,
                            t_name,
                            n=3,
                        ),

                    "name_weighted_similarity":
                        weighted_similarity(
                            s_name,
                            t_name,
                        ),

                    "name_length_ratio":
                        length_ratio(
                            s_name,
                            t_name,
                        ),

                    "address_exact":
                        exact_normalized_match(
                            s_addr,
                            t_addr,
                        ),

                    "address_compact_exact":
                        exact_compact_match(
                            s_addr,
                            t_addr,
                        ),

                    "address_similarity":
                        sequence_similarity(
                            s_addr,
                            t_addr,
                        ),

                    "address_ratio":
                        rapid_ratio(
                            s_addr,
                            t_addr,
                        ),

                    "address_token_sort_similarity":
                        token_sort_similarity(
                            s_addr,
                            t_addr,
                        ),

                    "address_token_similarity":
                        token_jaccard_similarity(
                            s_addr,
                            t_addr,
                        ),

                    "address_token_set_similarity":
                        token_set_similarity(
                            s_addr,
                            t_addr,
                        ),

                    "address_token_overlap":
                        token_overlap_ratio(
                            s_addr,
                            t_addr,
                        ),

                    "address_jaro_winkler":
                        jaro_winkler_similarity(
                            s_addr,
                            t_addr,
                        ),

                    "address_ngram_similarity":
                        character_ngram_similarity(
                            s_addr,
                            t_addr,
                            n=3,
                        ),

                    "address_weighted_similarity":
                        weighted_similarity(
                            s_addr,
                            t_addr,
                        ),

                    "address_length_ratio":
                        length_ratio(
                            s_addr,
                            t_addr,
                        ),

                    "house_number_match":
                        house_number_match(
                            s_addr,
                            t_addr,
                        ),

                    "house_number_exact_match":
                        house_number_exact_match(
                            s_addr,
                            t_addr,
                        ),

                    "country_match":
                        int(
                            bool(s_country)
                            and bool(t_country)
                            and normalize_text(
                                s_country
                            )
                            == normalize_text(
                                t_country
                            )
                        ),

                    "source_country_missing":
                        int(not bool(s_country)),

                    "candidate_country_missing":
                        int(not bool(t_country)),

                    "country_both_present":
                        int(
                            bool(s_country)
                            and bool(t_country)
                        ),

                    # These are not available in the
                    # normalized DuckDB tables.
                    "postal_code_match": 0,
                    "phone_match": 0,
                    "email_match": 0,
                    "website_match": 0,
                }

            feature_rows.append(
                    [
                        feature_values[name]
                        for name in FEATURES
                    ]
                )

            output_source_ids.append(
                    source_entity_id
                )

            output_candidate_ids.append(
                    candidate_entity_id
                )

        feature_matrix = np.asarray(
            feature_rows,
            dtype=np.float64,
        )

        probabilities = model.predict_proba(
            feature_matrix
        )[
            :,
            list(model.classes_).index(1),
        ]

        batch_df = pd.DataFrame(
            {
                "source1_entity_id":
                    output_source_ids,

                "candidate_entity_id":
                    output_candidate_ids,

                "probability":
                    probabilities,
            }
        )

        con.register(
            "score_batch",
            batch_df,
        )

        try:
            con.execute(
                """
                INSERT INTO scored_candidates
                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    probability
                FROM score_batch
                """
            )
        finally:
            con.unregister("score_batch")

        batch_number += 1
        total_scored += len(rows)

        elapsed = time.perf_counter() - started

        rate = (
            total_scored / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"  scored batch {batch_number:,} | "
            f"rows: {len(rows):,} | "
            f"total: {total_scored:,} | "
            f"{rate:,.0f} rows/s",
            flush=True,
        )

    query_cursor.close()


def build_final_matches(
    con: duckdb.DuckDBPyConnection,
) -> None:

    con.execute(
        f"""
        CREATE OR REPLACE TABLE final_matches AS

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM (

            SELECT
                source1_entity_id,
                candidate_entity_id,
                probability,

                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id

                    ORDER BY
                        probability DESC,
                        candidate_entity_id
                ) AS rn

            FROM scored_candidates

            WHERE probability >= {
                MATCH_THRESHOLD
            }
        ) ranked

        WHERE rn <= {
            MAX_MATCHES_PER_SOURCE
        }
        """
    )

    count = con.execute(
        """
        SELECT count(*)
        FROM final_matches
        """
    ).fetchone()[0]

    print(
        f"Final matches selected: {count:,}",
        flush=True,
    )


def copy_staged_outputs(
    con: duckdb.DuckDBPyConnection,
) -> tuple[Path, Path]:

    candidate_fd, candidate_name = tempfile.mkstemp(
        prefix="candidate_pairs_v2_",
        suffix=".tsv",
        dir=OUT,
    )

    match_fd, match_name = tempfile.mkstemp(
        prefix="matching_results_v2_",
        suffix=".tsv",
        dir=OUT,
    )

    os.close(candidate_fd)
    os.close(match_fd)

    candidate_tmp = Path(candidate_name)
    match_tmp = Path(match_name)

    con.execute(
        f"""
        COPY (

            SELECT
                s.entity_id AS source1_entity_id,

                COALESCE(
                    string_agg(
                        c.candidate_entity_id,
                        ','
                        ORDER BY c.candidate_entity_id
                    ),
                    ''
                ) AS candidate_entity_ids

            FROM source1 s

            LEFT JOIN candidate_pairs_internal c
                ON c.source1_entity_id = s.entity_id

            GROUP BY s.entity_id

            ORDER BY s.entity_id

        )

        TO '{sql_path(candidate_tmp)}'

        (
            DELIMITER '\\t',
            HEADER TRUE,
            QUOTE '"',
            ESCAPE '"'
        )
        """
    )

    con.execute(
        f"""
        COPY (

            SELECT
                s.entity_id AS source1_entity_id,

                COALESCE(
                    string_agg(
                        m.candidate_entity_id,
                        ','
                        ORDER BY
                            m.probability DESC,
                            m.candidate_entity_id
                    ),
                    ''
                ) AS matched_entity_ids

            FROM source1 s

            LEFT JOIN (

                SELECT
                    f.source1_entity_id,
                    f.candidate_entity_id,
                    p.probability

                FROM final_matches f

                JOIN scored_candidates p
                    USING (
                        source1_entity_id,
                        candidate_entity_id
                    )

            ) m

                ON m.source1_entity_id = s.entity_id

            GROUP BY s.entity_id

            ORDER BY s.entity_id

        )

        TO '{sql_path(match_tmp)}'

        (
            DELIMITER '\\t',
            HEADER TRUE,
            QUOTE '"',
            ESCAPE '"'
        )
        """
    )

    return candidate_tmp, match_tmp


def validate_staged_outputs(
    con: duckdb.DuckDBPyConnection,
    candidate_tmp: Path,
    match_tmp: Path,
) -> None:

    expected_source_count = con.execute(
        "SELECT count(*) FROM source1"
    ).fetchone()[0]

    candidate_csv = sql_path(candidate_tmp)
    match_csv = sql_path(match_tmp)

    expected_headers = {
        candidate_tmp:
            "source1_entity_id\tcandidate_entity_ids",

        match_tmp:
            "source1_entity_id\tmatched_entity_ids",
    }

    for path, header in expected_headers.items():

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:

            actual = stream.readline().rstrip(
                "\r\n"
            )

        if actual != header:
            raise RuntimeError(
                f"Unexpected TSV header in "
                f"{path.name}: {actual!r}"
            )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW staged_candidates AS

        SELECT *

        FROM read_csv(
            '{candidate_csv}',
            delim='\\t',
            header=true,

            columns={{
                'source1_entity_id':'VARCHAR',
                'candidate_entity_ids':'VARCHAR'
            }}
        )
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW staged_matches AS

        SELECT *

        FROM read_csv(
            '{match_csv}',
            delim='\\t',
            header=true,

            columns={{
                'source1_entity_id':'VARCHAR',
                'matched_entity_ids':'VARCHAR'
            }}
        )
        """
    )

    for view in (
        "staged_candidates",
        "staged_matches",
    ):

        row_count, distinct_count = con.execute(
            f"""
            SELECT
                count(*),
                count(DISTINCT source1_entity_id)

            FROM {view}
            """
        ).fetchone()

        if (
            row_count != expected_source_count
            or distinct_count != expected_source_count
        ):
            raise RuntimeError(
                f"{view} row/unique Source1 count "
                f"{row_count}/{distinct_count}; "
                f"expected {expected_source_count}"
            )

    invalid_target_count = con.execute(
        """
        SELECT count(*)

        FROM staged_candidates c,

        unnest(
            string_split(
                coalesce(
                    c.candidate_entity_ids,
                    ''
                ),
                ','
            )
        ) u(candidate_id)

        LEFT JOIN target t
            ON t.entity_id = u.candidate_id

        WHERE u.candidate_id <> ''
          AND t.entity_id IS NULL
        """
    ).fetchone()[0]

    if invalid_target_count:
        raise RuntimeError(
            f"Candidate output contains "
            f"{invalid_target_count} invalid target IDs"
        )

    subset_violations = con.execute(
        """
        SELECT count(*)

        FROM staged_matches m,

        unnest(
            string_split(
                coalesce(
                    m.matched_entity_ids,
                    ''
                ),
                ','
            )
        ) u(matched_id)

        WHERE u.matched_id <> ''

          AND NOT EXISTS (

              SELECT 1

              FROM staged_candidates c,

              unnest(
                  string_split(
                      coalesce(
                          c.candidate_entity_ids,
                          ''
                      ),
                      ','
                  )
              ) v(candidate_id)

              WHERE
                  c.source1_entity_id
                  = m.source1_entity_id

                AND v.candidate_id
                    = u.matched_id
          )
        """
    ).fetchone()[0]

    if subset_violations:
        raise RuntimeError(
            f"Found {subset_violations} matches "
            f"absent from candidate lists"
        )

    print(
        f"Validated outputs: "
        f"{expected_source_count:,} Source1 rows; "
        f"unique IDs, target IDs, and match subsets "
        f"are valid.",
        flush=True,
    )


def main() -> None:

    if duckdb.__version__ != "1.5.5":
        raise RuntimeError(
            f"DuckDB 1.5.5 is required; "
            f"found {duckdb.__version__}"
        )

    for path in (
        S1_PATH,
        S2_PATH,
        S3_PATH,
        MODEL_PATH,
    ):

        if not path.is_file():
            raise FileNotFoundError(path)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]

    model_features = list(
        getattr(
            model,
            "feature_names_in_",
            FEATURES,
        )
    )

    if model_features != FEATURES:
        raise RuntimeError(
            "RF feature order mismatch.\n"
            f"Expected: {FEATURES}\n"
            f"Got:      {model_features}"
        )

    if 1 not in model.classes_:
        raise RuntimeError(
            f"RF has no positive class 1: "
            f"{model.classes_}"
        )

    print(
        f"Model loaded: {len(FEATURES)} features",
        flush=True,
    )

    con = duckdb.connect(
        str(DB_PATH)
    )

    candidate_tmp = None
    match_tmp = None

    try:

        con.execute(
            f"SET memory_limit='{MEMORY_LIMIT}'"
        )

        con.execute(
            f"SET temp_directory="
            f"'{sql_path(TEMP_DIR)}'"
        )

        con.execute(
            f"SET threads={THREADS}"
        )

        con.execute(
            "SET preserve_insertion_order=false"
        )

        print(
            "Building normalized tables...",
            flush=True,
        )

        build_tables(con)

        source_count = con.execute(
            "SELECT count(*) FROM source1"
        ).fetchone()[0]

        target_count = con.execute(
            "SELECT count(*) FROM target"
        ).fetchone()[0]

        print(
            f"Loaded {source_count:,} Source1 rows "
            f"and {target_count:,} target rows.",
            flush=True,
        )

        build_candidates(con)

        candidate_count = con.execute(
            "SELECT count(*) "
            "FROM candidate_pairs_internal"
        ).fetchone()[0]

        print(
            f"Candidate pairs to score: "
            f"{candidate_count:,}",
            flush=True,
        )

        score_candidates(
            con,
            model,
        )

        build_final_matches(con)

        candidate_tmp, match_tmp = (
            copy_staged_outputs(con)
        )

        validate_staged_outputs(
            con,
            candidate_tmp,
            match_tmp,
        )

        # Only replace final submission files
        # AFTER successful validation.

        os.replace(
            candidate_tmp,
            CANDIDATE_OUTPUT,
        )

        candidate_tmp = None

        os.replace(
            match_tmp,
            MATCH_OUTPUT,
        )

        match_tmp = None

        print(
            f"Wrote {CANDIDATE_OUTPUT}",
            flush=True,
        )

        print(
            f"Wrote {MATCH_OUTPUT}",
            flush=True,
        )

        print(
            "\nDONE. matching_results.tsv is ready.",
            flush=True,
        )

    finally:

        con.close()

        for path in (
            candidate_tmp,
            match_tmp,
        ):

            if path is not None:
                path.unlink(
                    missing_ok=True
                )


if __name__ == "__main__":
    main()