import duckdb
from pathlib import Path

DB = Path("output/matcher_work_v2.duckdb")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

con = duckdb.connect(str(DB), read_only=True)
con.execute("SET threads=4")
con.execute("SET memory_limit='2GB'")

print("Starting final matching...", flush=True)

print("1/4 Finding unique exact name matches...", flush=True)

con.execute("""
CREATE OR REPLACE TEMP TABLE unique_matches AS
SELECT
    s.entity_id AS source1_entity_id,
    MIN(t.entity_id) AS matched_entity_id
FROM source1 s
JOIN target t
  ON s.n_country = t.n_country
 AND s.n_name = t.n_name
WHERE s.n_name <> ''
  AND t.n_name <> ''
GROUP BY s.entity_id
HAVING COUNT(*) = 1
""")

unique_count = con.execute("""
SELECT COUNT(*) FROM unique_matches
""").fetchone()[0]

print(f"Unique exact-name matches: {unique_count:,}", flush=True)


print("2/4 Finding exact address matches...", flush=True)

con.execute("""
CREATE OR REPLACE TEMP TABLE address_matches AS
SELECT
    s.entity_id AS source1_entity_id,
    MIN(t.entity_id) AS matched_entity_id
FROM source1 s
JOIN target t
  ON s.n_country = t.n_country
 AND s.n_name = t.n_name
 AND s.n_address = t.n_address
WHERE s.n_name <> ''
  AND s.n_address <> ''
  AND t.n_name <> ''
  AND t.n_address <> ''
GROUP BY s.entity_id
""")

address_count = con.execute("""
SELECT COUNT(*) FROM address_matches
""").fetchone()[0]

print(f"Exact address matches: {address_count:,}", flush=True)


print("3/4 Resolving ambiguous names...", flush=True)

con.execute("""
CREATE OR REPLACE TEMP TABLE final_matches AS

SELECT
    s.entity_id AS source1_entity_id,

    COALESCE(
        a.matched_entity_id,
        u.matched_entity_id,
        ''
    ) AS matched_entity_ids

FROM source1 s

LEFT JOIN address_matches a
    ON s.entity_id = a.source1_entity_id

LEFT JOIN unique_matches u
    ON s.entity_id = u.source1_entity_id
""")

matched_count = con.execute("""
SELECT COUNT(*)
FROM final_matches
WHERE matched_entity_ids <> ''
""").fetchone()[0]

print(f"Total resolved matches: {matched_count:,}", flush=True)


print("4/4 Writing matching_results.tsv...", flush=True)

output_path = (OUT / "matching_results.tsv").resolve()
output_sql = str(output_path).replace("\\", "/").replace("'", "''")

con.execute(f"""
COPY (
    SELECT
        source1_entity_id,
        matched_entity_ids
    FROM final_matches
) TO '{output_sql}'
(
    DELIMITER '\\t',
    HEADER TRUE,
    QUOTE '"',
    ESCAPE '"'
)
""")

total = con.execute("""
SELECT COUNT(*) FROM final_matches
""").fetchone()[0]

unmatched = total - matched_count

con.close()

print("")
print("=" * 60)
print("FINAL MATCHING COMPLETE")
print("=" * 60)
print(f"Source 1 rows:       {total:,}")
print(f"Matched rows:        {matched_count:,}")
print(f"Unmatched rows:      {unmatched:,}")
print(f"Output:              {output_path}")
print("=" * 60)
