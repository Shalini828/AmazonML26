"""Disk-backed emergency entity matcher.

Target records and joins live in DuckDB; Python only configures queries and
never builds per-entity or per-candidate dictionaries.
"""
from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"
MATCH_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
DB_PATH = OUTPUT_DIR / "matcher_work.duckdb"
MAX_CANDIDATES_PER_BLOCK = 5


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUT_DIR / "duckdb_temp"
    temp_dir.mkdir(exist_ok=True)

    print("Starting disk-backed matcher (DuckDB memory limit: 2 GB)...", flush=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")

    def source_sql(path: Path) -> str:
        return f"read_csv('{sql_path(path)}', delim='\\t', header=true, all_varchar=true, quote='\"', escape='\"')"

    # Match normalization.py's lowercase, ampersand expansion and punctuation
    # removal while doing the work once in the source staging tables.
    def norm(column: str) -> str:
        value = f"lower(coalesce(cast({column} AS VARCHAR), ''))"
        value = f"replace({value}, '&', ' and ')"
        value = f"regexp_replace({value}, '[^[:alnum:][:space:]]', ' ', 'g')"
        return f"trim(regexp_replace({value}, '[[:space:]]+', ' ', 'g'))"

    existing_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "target" not in existing_tables:
        print("Staging Source 2 and Source 3 on disk...", flush=True)
        con.execute(f"""
        CREATE TABLE target AS
        SELECT entity_id, business_name, business_address, country,
               n_name, n_address, n_country,
               n_country || '*' || substr(n_name, 1, 1) AS key1,
               n_country || '_' || substr(n_name, 1, 4) AS key2,
               n_country || '_' || substr(n_name, 1, 3) AS key3
        FROM (
            SELECT cast(entity_id AS VARCHAR) entity_id,
                   cast(business_name AS VARCHAR) business_name,
                   cast(business_address AS VARCHAR) business_address,
                   cast(country AS VARCHAR) country,
                   {norm('business_name')} n_name,
                   {norm('business_address')} n_address,
                   {norm('country')} n_country
            FROM {source_sql(S2_PATH)}
            UNION ALL
            SELECT cast(entity_id AS VARCHAR), cast(business_name AS VARCHAR),
                   cast(business_address AS VARCHAR), cast(country AS VARCHAR),
                   {norm('business_name')}, {norm('business_address')}, {norm('country')}
            FROM {source_sql(S3_PATH)}
        )
        """)
    print(f"Targets staged: {con.execute('SELECT count(*) FROM target').fetchone()[0]:,}", flush=True)

    if "source1" not in existing_tables:
        print("Staging Source 1...", flush=True)
        con.execute(f"""
        CREATE TABLE source1 AS
        SELECT entity_id, business_name, business_address, country,
               n_name, n_address, n_country,
               n_country || '*' || substr(n_name, 1, 1) AS key1,
               n_country || '_' || substr(n_name, 1, 4) AS key2,
               n_country || '_' || substr(n_name, 1, 3) AS key3
        FROM (
            SELECT cast(entity_id AS VARCHAR) entity_id,
                   cast(business_name AS VARCHAR) business_name,
                   cast(business_address AS VARCHAR) business_address,
                   cast(country AS VARCHAR) country,
                   {norm('business_name')} n_name,
                   {norm('business_address')} n_address,
                   {norm('country')} n_country
            FROM {source_sql(S1_PATH)}
        )
        """)

    # Aggregate each of the three prescribed blocks before joining. max_by
    # keeps a small, deterministic ID sample per block and prevents any all-pairs
    # join or unbounded candidate list from being materialized.
    if "candidate_rows" in existing_tables:
        con.execute("DROP TABLE candidate_rows")
    if "block_candidates" in existing_tables:
        con.execute("DROP TABLE block_candidates")
    print("Aggregating bounded blocked candidate lists...", flush=True)
    con.execute(f"""
        CREATE TABLE block_candidates AS
        SELECT 'key1' AS block_type, key1 AS block_key,
               max_by(entity_id, entity_id, {MAX_CANDIDATES_PER_BLOCK}) AS ids
        FROM target WHERE n_name<>'' GROUP BY key1
        UNION ALL
        SELECT 'key2', key2,
               max_by(entity_id, entity_id, {MAX_CANDIDATES_PER_BLOCK})
        FROM target WHERE n_name<>'' GROUP BY key2
        UNION ALL
        SELECT 'key3', key3,
               max_by(entity_id, entity_id, {MAX_CANDIDATES_PER_BLOCK})
        FROM target WHERE n_name<>'' GROUP BY key3
        """)
    con.execute("""
        CREATE TABLE candidate_rows AS
        SELECT s.entity_id AS s1_id,
               list_sort(list_distinct(list_concat(
                   coalesce(k1.ids, []::VARCHAR[]),
                   coalesce(k2.ids, []::VARCHAR[]),
                   coalesce(k3.ids, []::VARCHAR[])
               ))) AS ids
        FROM source1 s
        LEFT JOIN block_candidates k1 ON k1.block_type='key1' AND s.key1=k1.block_key AND s.n_name<>''
        LEFT JOIN block_candidates k2 ON k2.block_type='key2' AND s.key2=k2.block_key AND s.n_name<>''
        LEFT JOIN block_candidates k3 ON k3.block_type='key3' AND s.key3=k3.block_key AND s.n_name<>''
    """)
    candidate_count = con.execute("SELECT sum(len(ids)) FROM candidate_rows").fetchone()[0] or 0
    print(f"Candidate pairs retained: {candidate_count:,}", flush=True)

    print("Writing candidate output...", flush=True)
    con.execute(f"""
        COPY (
            SELECT s1_id AS source1_entity_id,
                   array_to_string(ids, ',') AS candidate_entity_ids
            FROM candidate_rows
        ) TO '{sql_path(CANDIDATE_OUTPUT)}'
        (DELIMITER '\t', HEADER true, QUOTE '"', ESCAPE '"')
    """)

    print("Writing deterministic exact-match output...", flush=True)
    # Exact normalized-name agreement is the stable emergency decision rule;
    # exact address agreement breaks ties when several names are identical.
    con.execute(f"""
        COPY (
            WITH choices AS (
                SELECT s.entity_id AS s1_id, u.candidate_id,
                       row_number() OVER (
                           PARTITION BY s.entity_id
                           ORDER BY (s.n_name=t.n_name AND s.n_name<>'') DESC,
                                    (s.n_address=t.n_address AND s.n_address<>'') DESC,
                                     u.candidate_id
                       ) AS choice_rank
                FROM source1 s
                JOIN candidate_rows c ON c.s1_id=s.entity_id
                CROSS JOIN UNNEST(c.ids) u(candidate_id)
                JOIN target t ON t.key2=s.key2 AND t.n_name=s.n_name
                             AND t.entity_id=u.candidate_id
                WHERE s.n_name<>''
            ), selected AS (
                SELECT s1_id, candidate_id FROM choices WHERE choice_rank=1
            )
            SELECT s.entity_id AS source1_entity_id,
                   coalesce(x.candidate_id, '') AS matched_entity_ids
            FROM source1 s LEFT JOIN selected x ON x.s1_id=s.entity_id
        ) TO '{sql_path(MATCH_OUTPUT)}'
        (DELIMITER '\t', HEADER true, QUOTE '"', ESCAPE '"')
    """)
    source_count = con.execute("SELECT count(*) FROM source1").fetchone()[0]
    matched_count = con.execute("SELECT count(*) FROM (SELECT DISTINCT s1_id FROM candidate_rows c CROSS JOIN UNNEST(c.ids) u(candidate_id) JOIN target t ON t.entity_id=u.candidate_id JOIN source1 s ON s.entity_id=c.s1_id WHERE s.n_name=t.n_name AND s.n_name<>'')").fetchone()[0]
    con.close()

    print("MATCHER COMPLETE", flush=True)
    print(f"Source 1 rows: {source_count:,}", flush=True)
    print(f"Candidate pairs: {candidate_count:,}", flush=True)
    print(f"Predicted matches: {matched_count:,}", flush=True)
    print(f"Candidate output: {CANDIDATE_OUTPUT} ({CANDIDATE_OUTPUT.stat().st_size:,} bytes)", flush=True)
    print(f"Match output: {MATCH_OUTPUT} ({MATCH_OUTPUT.stat().st_size:,} bytes)", flush=True)


if __name__ == '__main__':
    main()
