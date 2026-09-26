"""Build TEST candidates and RF matches with a disk-backed DuckDB pipeline."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
import time

import duckdb
import joblib
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio


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
BLOCK_CAP = 20
MATCH_THRESHOLD = 0.999
MAX_MATCHES_PER_SOURCE = 3
SCORE_BATCH_SIZE = 25_000

FEATURES = [
    "name_similarity",
    "address_similarity",
    "name_token_similarity",
    "address_token_similarity",
    "country_match",
]

BLOCKS = [
    ("exact_name", "k_exact", BLOCK_CAP),
    ("country_first_meaningful_token", "k_token", BLOCK_CAP),
    ("country_name_2", "k_c2", BLOCK_CAP),
    ("country_name_3", "k_c3", BLOCK_CAP),
    ("country_name_4", "k_c4", BLOCK_CAP),
    ("country_address_first_token", "k_addrtoken", BLOCK_CAP),
    ("country_address_prefix_10", "k_addrprefix", BLOCK_CAP),
    ("country_address_number", "k_addrnumber", BLOCK_CAP),
    ("country_postal_code", "k_postal", BLOCK_CAP),
]


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def normalized(expr: str) -> str:
    value = f"lower(coalesce(cast({expr} AS VARCHAR), ''))"
    value = f"regexp_replace({value}, '[^[:alnum:][:space:]]', ' ', 'g')"
    return f"trim(regexp_replace({value}, '[[:space:]]+', ' ', 'g'))"


def country_normalized(expr: str) -> str:
    value = normalized(expr)
    return f"""CASE
        WHEN {value} IN ('usa', 'united states', 'united states of america', 'us') THEN 'us'
        WHEN {value} IN ('india', 'bharat', 'in') THEN 'india'
        WHEN {value} IN ('france', 'fr') THEN 'france'
        ELSE {value}
    END"""


def create_normalized_table(con: duckdb.DuckDBPyConnection, table: str, path_sql: str) -> None:
    n_name = normalized("business_name")
    n_address = normalized("business_address")
    n_country = country_normalized("country")
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH normalized_rows AS (
            SELECT
                cast(entity_id AS VARCHAR) AS entity_id,
                coalesce(cast(business_name AS VARCHAR), '') AS business_name,
                coalesce(cast(business_address AS VARCHAR), '') AS business_address,
                coalesce(cast(country AS VARCHAR), '') AS country,
                {n_name} AS n_name,
                {n_address} AS n_address,
                {n_country} AS n_country
            FROM read_parquet('{path_sql}')
        ), keyed AS (
            SELECT *,
                regexp_extract(
                    regexp_replace(
                        n_name,
                        '^(the|a|an|and|of|company|co|inc|ltd|llc) +',
                        ''
                    ),
                    '^[^ ]+',
                    0
                ) AS name_token,
                regexp_extract(n_address, '^[^ ]+', 0) AS address_token,
                regexp_extract(n_address, '[0-9]{{5,6}}', 0) AS postal,
                regexp_extract(n_address, '[0-9]+', 0) AS address_number,
            FROM normalized_rows
        )
        SELECT *,
            n_name AS k_exact,
            n_country || '|' || name_token AS k_token,
            n_country || '|' || substr(n_name,1,2) AS k_c2,
            n_country || '|' || substr(n_name,1,3) AS k_c3,
            n_country || '|' || substr(n_name,1,4) AS k_c4,
            n_country || '|' || address_token AS k_addrtoken,
            n_country || '|' || substr(n_address,1,10) AS k_addrprefix,
            CASE
                WHEN address_number <> ''
                THEN n_country || '|' || address_number
                ELSE ''
            END AS k_addrnumber,
            n_country || '|' || postal AS k_postal
        FROM keyed
    """)


def build_tables(con: duckdb.DuckDBPyConnection) -> None:
    create_normalized_table(con, "source1", sql_path(S1_PATH))

    source2 = sql_path(S2_PATH)
    source3 = sql_path(S3_PATH)
    n_name = normalized("business_name")
    n_address = normalized("business_address")
    n_country = country_normalized("country")
    con.execute(f"""
        CREATE OR REPLACE TABLE target AS
        WITH raw AS (
            SELECT entity_id, business_name, business_address, country
            FROM read_parquet('{source2}')
            UNION ALL
            SELECT entity_id, business_name, business_address, country
            FROM read_parquet('{source3}')
        ), normalized_rows AS (
            SELECT
                cast(entity_id AS VARCHAR) AS entity_id,
                coalesce(cast(business_name AS VARCHAR), '') AS business_name,
                coalesce(cast(business_address AS VARCHAR), '') AS business_address,
                coalesce(cast(country AS VARCHAR), '') AS country,
                {n_name} AS n_name,
                {n_address} AS n_address,
                {n_country} AS n_country
            FROM raw
        ), keyed AS (
            SELECT *,
                regexp_extract(
                    regexp_replace(
                        n_name,
                        '^(the|a|an|and|of|company|co|inc|ltd|llc) +',
                        ''
                    ),
                    '^[^ ]+',
                    0
                ) AS name_token,
                regexp_extract(n_address, '^[^ ]+', 0) AS address_token,
                regexp_extract(n_address, '[0-9]{{5,6}}', 0) AS postal,
                regexp_extract(n_address, '[0-9]+', 0) AS address_number,
            FROM normalized_rows
        )
        SELECT *,
            n_name AS k_exact,
            n_country || '|' || name_token AS k_token,
            n_country || '|' || substr(n_name,1,2) AS k_c2,
            n_country || '|' || substr(n_name,1,3) AS k_c3,
            n_country || '|' || substr(n_name,1,4) AS k_c4,
            n_country || '|' || address_token AS k_addrtoken,
            n_country || '|' || substr(n_address,1,10) AS k_addrprefix,
            CASE
                WHEN address_number <> ''
                THEN n_country || '|' || address_number
                ELSE ''
                END AS k_addrnumber,
            n_country || '|' || postal AS k_postal
        FROM keyed
    """)


def build_candidates(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE selected_candidates (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR
        )
    """)

    for strategy, key, cap in BLOCKS:
        started = time.perf_counter()
        print(f"Generating block {strategy}...", flush=True)
        if cap is None:
            con.execute(f"""
                INSERT INTO selected_candidates
                SELECT DISTINCT s.entity_id, t.entity_id
                FROM source1 s
                JOIN target t ON s.{key}=t.{key}
                WHERE s.{key} <> ''
                  AND s.{key} NOT LIKE '%|'
            """)
        else:
            con.execute(f"""
                INSERT INTO selected_candidates
                WITH raw AS (
                    SELECT
                        s.entity_id AS source1_entity_id,
                        t.entity_id AS candidate_entity_id,
                        (s.n_name <> '' AND s.n_name=t.n_name)::INT AS exact_name,
                        (s.n_address <> '' AND s.n_address=t.n_address)::INT AS address_exact,
                        (
                            s.n_address <> '' AND t.n_address <> '' AND (
                                starts_with(s.n_address,t.n_address)
                                OR starts_with(t.n_address,s.n_address)
                                OR contains(s.n_address,t.n_address)
                                OR contains(t.n_address,s.n_address)
                            )
                        )::INT AS address_partial,
                        (
                            (length(s.n_name)>=1 AND substr(s.n_name,1,1)=substr(t.n_name,1,1))::INT
                            + (length(s.n_name)>=2 AND substr(s.n_name,1,2)=substr(t.n_name,1,2))::INT
                            + (length(s.n_name)>=3 AND substr(s.n_name,1,3)=substr(t.n_name,1,3))::INT
                            + (length(s.n_name)>=4 AND substr(s.n_name,1,4)=substr(t.n_name,1,4))::INT
                            + (length(s.n_name)>=8 AND substr(s.n_name,1,8)=substr(t.n_name,1,8))::INT
                        ) AS name_prefix_agreement,
                        CASE
                            WHEN length(s.n_name)+length(t.n_name)>0 THEN
                                length(list_intersect(
                                    list_distinct(string_split(s.n_name,' ')),
                                    list_distinct(string_split(t.n_name,' '))
                                ))::DOUBLE / greatest(1,
                                    length(list_distinct(string_split(s.n_name,' ')))
                                    + length(list_distinct(string_split(t.n_name,' ')))
                                    - length(list_intersect(
                                        list_distinct(string_split(s.n_name,' ')),
                                        list_distinct(string_split(t.n_name,' '))
                                    ))
                                )
                            ELSE 0
                        END AS name_token_overlap,
                        CASE
                            WHEN length(s.n_address)+length(t.n_address)>0 THEN
                                length(list_intersect(
                                    list_distinct(string_split(s.n_address,' ')),
                                    list_distinct(string_split(t.n_address,' '))
                                ))::DOUBLE / greatest(1,
                                    length(list_distinct(string_split(s.n_address,' ')))
                                    + length(list_distinct(string_split(t.n_address,' ')))
                                    - length(list_intersect(
                                        list_distinct(string_split(s.n_address,' ')),
                                        list_distinct(string_split(t.n_address,' '))
                                    ))
                                )
                            ELSE 0
                        END AS address_token_overlap,
                        (s.n_country <> '' AND s.n_country=t.n_country)::INT AS country_equal
                    FROM source1 s
                    JOIN target t ON s.{key}=t.{key}
                    WHERE s.{key} <> ''
                      AND s.{key} NOT LIKE '%|'
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY exact_name DESC, address_exact DESC,
                            address_partial DESC, name_prefix_agreement DESC,
                            name_token_overlap DESC, address_token_overlap DESC,
                            country_equal DESC, candidate_entity_id
                    ) AS rn
                    FROM raw
                )
                SELECT source1_entity_id, candidate_entity_id
                FROM ranked
                WHERE rn <= {cap}
            """)

        print(
            f"  cumulative unique pairs: {con.execute('SELECT count(*) FROM (SELECT DISTINCT source1_entity_id, candidate_entity_id FROM selected_candidates)').fetchone()[0]:,}"
            f" | {time.perf_counter()-started:.1f}s",
            flush=True,
        )

    con.execute("""
        CREATE OR REPLACE TABLE candidate_pairs_internal AS
        SELECT DISTINCT source1_entity_id, candidate_entity_id
        FROM selected_candidates
    """)


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def score_candidates(con: duckdb.DuckDBPyConnection, model) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE scored_candidates (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR,
            probability DOUBLE
        )
    """)

    query = con.execute("""
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            s.n_name,
            t.n_name,
            s.n_address,
            t.n_address,
            s.n_country,
            t.n_country
        FROM candidate_pairs_internal c
        JOIN source1 s ON s.entity_id=c.source1_entity_id
        JOIN target t ON t.entity_id=c.candidate_entity_id
        ORDER BY c.source1_entity_id, c.candidate_entity_id
    """)

    batch_number = 0
    while rows := query.fetchmany(SCORE_BATCH_SIZE):
        features = np.empty((len(rows), len(FEATURES)), dtype=np.float64)
        for i, row in enumerate(rows):
            features[i, 0] = ratio(row[2], row[3]) / 100.0
            features[i, 1] = ratio(row[4], row[5]) / 100.0
            features[i, 2] = token_jaccard(row[2], row[3])
            features[i, 3] = token_jaccard(row[4], row[5])
            features[i, 4] = int(bool(row[6]) and row[6] == row[7])

        probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
        con.register("score_batch", pd.DataFrame({
            "source1_entity_id": [r[0] for r in rows],
            "candidate_entity_id": [r[1] for r in rows],
            "probability": probabilities,
        }))
        con.execute("INSERT INTO scored_candidates SELECT * FROM score_batch")
        con.unregister("score_batch")
        batch_number += 1
        if batch_number % 10 == 0:
            print(f"Scored {query.rowcount if query.rowcount >= 0 else 'batch'} rows in {batch_number} batches...", flush=True)

    con.execute("""
        CREATE OR REPLACE TABLE final_matches AS
        SELECT source1_entity_id, candidate_entity_id
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY source1_entity_id
                ORDER BY probability DESC, candidate_entity_id
            ) AS rn
            FROM scored_candidates
            WHERE probability >= ?
        )
        WHERE rn <= ?
    """, [MATCH_THRESHOLD, MAX_MATCHES_PER_SOURCE])


def copy_staged_outputs(con: duckdb.DuckDBPyConnection) -> tuple[Path, Path]:
    candidate_fd, candidate_name = tempfile.mkstemp(
        prefix="candidate_pairs_v2_", suffix=".tsv", dir=OUT
    )
    match_fd, match_name = tempfile.mkstemp(
        prefix="matching_results_v2_", suffix=".tsv", dir=OUT
    )
    os.close(candidate_fd)
    os.close(match_fd)
    candidate_tmp = Path(candidate_name)
    match_tmp = Path(match_name)

    con.execute(f"""
        COPY (
            SELECT s.entity_id AS source1_entity_id,
                coalesce(string_agg(c.candidate_entity_id, ',' ORDER BY c.candidate_entity_id), '')
                    AS candidate_entity_ids
            FROM source1 s
            LEFT JOIN candidate_pairs_internal c
              ON c.source1_entity_id=s.entity_id
            GROUP BY s.entity_id
            ORDER BY s.entity_id
        ) TO '{sql_path(candidate_tmp)}'
        (DELIMITER '\t', HEADER TRUE, QUOTE '"', ESCAPE '"')
    """)
    con.execute(f"""
        COPY (
            SELECT s.entity_id AS source1_entity_id,
                coalesce(string_agg(m.candidate_entity_id, ',' ORDER BY
                    m.probability DESC, m.candidate_entity_id), '') AS matched_entity_ids
            FROM source1 s
            LEFT JOIN (
                SELECT f.source1_entity_id, f.candidate_entity_id, p.probability
                FROM final_matches f
                JOIN scored_candidates p USING (source1_entity_id, candidate_entity_id)
            ) m ON m.source1_entity_id=s.entity_id
            GROUP BY s.entity_id
            ORDER BY s.entity_id
        ) TO '{sql_path(match_tmp)}'
        (DELIMITER '\t', HEADER TRUE, QUOTE '"', ESCAPE '"')
    """)
    return candidate_tmp, match_tmp


def validate_staged_outputs(
    con: duckdb.DuckDBPyConnection, candidate_tmp: Path, match_tmp: Path
) -> None:
    expected_source_count = con.execute("SELECT count(*) FROM source1").fetchone()[0]
    candidate_csv = sql_path(candidate_tmp)
    match_csv = sql_path(match_tmp)

    expected_headers = {
        candidate_tmp: "source1_entity_id\tcandidate_entity_ids",
        match_tmp: "source1_entity_id\tmatched_entity_ids",
    }
    for path, header in expected_headers.items():
        with path.open("r", encoding="utf-8", newline="") as stream:
            actual = stream.readline().rstrip("\r\n")
        if actual != header:
            raise RuntimeError(f"Unexpected TSV header in {path.name}: {actual!r}")

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW staged_candidates AS
        SELECT * FROM read_csv('{candidate_csv}', delim='\\t', header=true,
            columns={{'source1_entity_id':'VARCHAR','candidate_entity_ids':'VARCHAR'}})
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW staged_matches AS
        SELECT * FROM read_csv('{match_csv}', delim='\\t', header=true,
            columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
    """)

    for view in ("staged_candidates", "staged_matches"):
        row_count, distinct_count = con.execute(
            f"SELECT count(*), count(DISTINCT source1_entity_id) FROM {view}"
        ).fetchone()
        if row_count != expected_source_count or distinct_count != expected_source_count:
            raise RuntimeError(
                f"{view} row/unique Source1 count {row_count}/{distinct_count}; "
                f"expected {expected_source_count}"
            )

    invalid_target_count = con.execute("""
        SELECT count(*)
        FROM staged_candidates c,
             unnest(string_split(coalesce(c.candidate_entity_ids,''), ',')) u(candidate_id)
        LEFT JOIN target t ON t.entity_id=u.candidate_id
        WHERE u.candidate_id <> '' AND t.entity_id IS NULL
    """).fetchone()[0]
    if invalid_target_count:
        raise RuntimeError(f"Candidate output contains {invalid_target_count} invalid target IDs")

    subset_violations = con.execute("""
        SELECT count(*)
        FROM staged_matches m,
             unnest(string_split(coalesce(m.matched_entity_ids,''), ',')) u(matched_id)
        WHERE u.matched_id <> ''
          AND NOT EXISTS (
              SELECT 1
              FROM staged_candidates c,
                   unnest(string_split(coalesce(c.candidate_entity_ids,''), ',')) v(candidate_id)
              WHERE c.source1_entity_id=m.source1_entity_id
                AND v.candidate_id=u.matched_id
          )
    """).fetchone()[0]
    if subset_violations:
        raise RuntimeError(f"Found {subset_violations} matches absent from candidate lists")

    print(
        f"Validated outputs: {expected_source_count:,} Source1 rows; "
        "unique IDs, target IDs, and match subsets are valid.",
        flush=True,
    )


def main() -> None:
    if duckdb.__version__ != "1.5.5":
        raise RuntimeError(
            f"DuckDB 1.5.5 is required; found {duckdb.__version__}"
        )
    for path in (S1_PATH, S2_PATH, S3_PATH, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    model_features = list(getattr(model, "feature_names_in_", FEATURES))
    if model_features != FEATURES:
        raise RuntimeError(
            f"RF feature order mismatch: expected {FEATURES}, got {model_features}"
        )
    if 1 not in model.classes_:
        raise RuntimeError(f"RF has no positive class 1: {model.classes_}")

    con = duckdb.connect(str(DB_PATH))
    candidate_tmp = None
    match_tmp = None
    try:
        con.execute("SET memory_limit='2GB'")
        con.execute(f"SET temp_directory='{sql_path(TEMP_DIR)}'")
        con.execute(f"SET threads={THREADS}")
        con.execute("SET preserve_insertion_order=false")

        build_tables(con)
        print(
            f"Loaded {con.execute('SELECT count(*) FROM source1').fetchone()[0]:,} Source1 rows and "
            f"{con.execute('SELECT count(*) FROM target').fetchone()[0]:,} target rows.",
            flush=True,
        )
        build_candidates(con)
        score_candidates(con, model)
        candidate_tmp, match_tmp = copy_staged_outputs(con)
        validate_staged_outputs(con, candidate_tmp, match_tmp)

        # Replace existing submission paths only after both staged files pass validation.
        os.replace(candidate_tmp, CANDIDATE_OUTPUT)
        candidate_tmp = None
        os.replace(match_tmp, MATCH_OUTPUT)
        match_tmp = None
        print(f"Wrote {CANDIDATE_OUTPUT}", flush=True)
        print(f"Wrote {MATCH_OUTPUT}", flush=True)
    finally:
        con.close()
        for path in (candidate_tmp, match_tmp):
            if path is not None:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
