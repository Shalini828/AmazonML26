from pathlib import Path
import duckdb


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

DB_PATH = OUTPUT_DIR / "matcher_work_v2.duckdb"
OUTPUT_PATH = OUTPUT_DIR / "matching_results_v2.tsv"


# ============================================================
# SETTINGS
# ============================================================

MEMORY_LIMIT = "2GB"
THREADS = 4


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STARTING FINAL MATCHING V2")
    print("=" * 70)

    print(f"DuckDB: {DB_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    if not DB_PATH.exists():
        print("ERROR: DuckDB database does not exist:")
        print(DB_PATH)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(
        str(DB_PATH),
        read_only=False,
    )

    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")

    # --------------------------------------------------------
    # Check tables
    # --------------------------------------------------------

    print("Checking database...")

    tables = {
        row[0]
        for row in con.execute("SHOW TABLES").fetchall()
    }

    print("Tables:", sorted(tables))
    print()

    if "source1" not in tables or "target" not in tables:
        print("ERROR: source1/target tables are missing.")
        con.close()
        return

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    source_count = con.execute(
        "SELECT COUNT(*) FROM source1"
    ).fetchone()[0]

    target_count = con.execute(
        "SELECT COUNT(*) FROM target"
    ).fetchone()[0]

    print(f"Source 1 rows: {source_count:,}")
    print(f"Target rows:   {target_count:,}")
    print()

    # --------------------------------------------------------
    # Step 1
    # Unique exact country + normalized name
    # --------------------------------------------------------

    print("1/4 Finding unique exact country + name matches...")

    con.execute("DROP TABLE IF EXISTS unique_name_matches")

    con.execute("""
        CREATE TABLE unique_name_matches AS
        WITH grouped AS (
            SELECT
                n_country,
                n_name,
                COUNT(*) AS target_count,
                MIN(entity_id) AS target_id
            FROM target
            WHERE
                n_country <> ''
                AND n_name <> ''
            GROUP BY
                n_country,
                n_name
        )
        SELECT
            s.entity_id AS source1_entity_id,
            g.target_id AS matched_entity_id
        FROM source1 s
        JOIN grouped g
            ON s.n_country = g.n_country
            AND s.n_name = g.n_name
        WHERE
            s.n_country <> ''
            AND s.n_name <> ''
            AND g.target_count = 1
    """)

    unique_count = con.execute(
        "SELECT COUNT(*) FROM unique_name_matches"
    ).fetchone()[0]

    print(f"Unique exact-name matches: {unique_count:,}")
    print()

    # --------------------------------------------------------
    # Step 2
    # Exact address match for unresolved records
    # --------------------------------------------------------

    print("2/4 Finding exact address matches...")

    con.execute("DROP TABLE IF EXISTS exact_address_matches")

    con.execute("""
        CREATE TABLE exact_address_matches AS
        WITH candidates AS (
            SELECT
                s.entity_id AS source1_entity_id,
                t.entity_id AS target_id,
                ROW_NUMBER() OVER (
                    PARTITION BY s.entity_id
                    ORDER BY t.entity_id
                ) AS rn
            FROM source1 s
            JOIN target t
                ON s.n_country = t.n_country
                AND s.n_name = t.n_name
                AND s.n_address = t.n_address
            WHERE
                s.n_country <> ''
                AND s.n_name <> ''
                AND s.n_address <> ''
                AND NOT EXISTS (
                    SELECT 1
                    FROM unique_name_matches u
                    WHERE u.source1_entity_id = s.entity_id
                )
        )
        SELECT
            source1_entity_id,
            target_id
        FROM candidates
        WHERE rn = 1
    """)

    address_count = con.execute(
        "SELECT COUNT(*) FROM exact_address_matches"
    ).fetchone()[0]

    print(f"Exact-address matches: {address_count:,}")
    print()

    # --------------------------------------------------------
    # Step 3
    # Resolve remaining exact-name ambiguities
    #
    # Prefer:
    #   1. exact address
    #   2. exact pincode when available
    #   3. deterministic target ID
    # --------------------------------------------------------

    print("3/4 Resolving remaining ambiguous names...")

    con.execute("DROP TABLE IF EXISTS ambiguous_matches")

    con.execute("""
        CREATE TABLE ambiguous_matches AS
        WITH unresolved AS (
            SELECT
                s.entity_id,
                s.n_country,
                s.n_name,
                s.n_address,
                s.pincode
            FROM source1 s
            WHERE
                s.n_country <> ''
                AND s.n_name <> ''
                AND NOT EXISTS (
                    SELECT 1
                    FROM unique_name_matches u
                    WHERE u.source1_entity_id = s.entity_id
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM exact_address_matches a
                    WHERE a.source1_entity_id = s.entity_id
                )
        ),

        ranked AS (
            SELECT
                s.entity_id AS source1_entity_id,
                t.entity_id AS target_id,

                CASE
                    WHEN
                        s.n_address <> ''
                        AND t.n_address <> ''
                        AND s.n_address = t.n_address
                    THEN 3

                    WHEN
                        s.pincode <> ''
                        AND t.pincode <> ''
                        AND s.pincode = t.pincode
                    THEN 2

                    ELSE 1
                END AS match_score,

                ROW_NUMBER() OVER (
                    PARTITION BY s.entity_id
                    ORDER BY
                        CASE
                            WHEN
                                s.n_address <> ''
                                AND t.n_address <> ''
                                AND s.n_address = t.n_address
                            THEN 3

                            WHEN
                                s.pincode <> ''
                                AND t.pincode <> ''
                                AND s.pincode = t.pincode
                            THEN 2

                            ELSE 1
                        END DESC,
                        t.entity_id
                ) AS rn

            FROM unresolved s
            JOIN target t
                ON s.n_country = t.n_country
                AND s.n_name = t.n_name
        )

        SELECT
            source1_entity_id,
            target_id,
            match_score
        FROM ranked
        WHERE rn = 1
    """)

    ambiguous_count = con.execute(
        "SELECT COUNT(*) FROM ambiguous_matches"
    ).fetchone()[0]

    print(f"Resolved ambiguous matches: {ambiguous_count:,}")
    print()

    # --------------------------------------------------------
    # Step 4
    # Final output
    # --------------------------------------------------------

    print("4/4 Writing final matching_results_v2.tsv...")

    con.execute(f"""
        COPY (
            SELECT
                s.entity_id AS source1_entity_id,

                COALESCE(
                    CAST(u.matched_entity_id AS VARCHAR),
                    CAST(a.target_id AS VARCHAR),
                    CAST(m.target_id AS VARCHAR),
                    ''
                ) AS matched_entity_ids

            FROM source1 s

            LEFT JOIN unique_name_matches u
                ON s.entity_id = u.source1_entity_id

            LEFT JOIN exact_address_matches a
                ON s.entity_id = a.source1_entity_id

            LEFT JOIN ambiguous_matches m
                ON s.entity_id = m.source1_entity_id

            ORDER BY s.entity_id
        )
        TO '{OUTPUT_PATH.as_posix()}'
        (
            DELIMITER '\\t',
            HEADER true,
            QUOTE '"',
            ESCAPE '"'
        )
    """)

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    matched_count = con.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT
                s.entity_id,
                COALESCE(
                    u.matched_entity_id,
                    a.target_id,
                    m.target_id
                ) AS matched_id
            FROM source1 s

            LEFT JOIN unique_name_matches u
                ON s.entity_id = u.source1_entity_id

            LEFT JOIN exact_address_matches a
                ON s.entity_id = a.source1_entity_id

            LEFT JOIN ambiguous_matches m
                ON s.entity_id = m.source1_entity_id
        )
        WHERE matched_id IS NOT NULL
    """).fetchone()[0]

    unmatched_count = source_count - matched_count

    con.close()

    print()
    print("=" * 70)
    print("FINAL MATCHING V2 COMPLETE")
    print("=" * 70)
    print(f"Source 1 rows:     {source_count:,}")
    print(f"Matched rows:      {matched_count:,}")
    print(f"Unmatched rows:    {unmatched_count:,}")
    print(f"Output:")
    print(OUTPUT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()