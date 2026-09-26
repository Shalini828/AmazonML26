import duckdb

con = duckdb.connect("output/matcher_work_v2.duckdb", read_only=True)

q = """
SELECT
    COUNT(*) AS ambiguous_sources,
    SUM(CASE WHEN candidate_count = 1 THEN 1 ELSE 0 END) AS one_candidate,
    SUM(CASE WHEN candidate_count = 2 THEN 1 ELSE 0 END) AS two_candidates,
    SUM(CASE WHEN candidate_count BETWEEN 3 AND 5 THEN 1 ELSE 0 END) AS three_to_five,
    SUM(CASE WHEN candidate_count > 5 THEN 1 ELSE 0 END) AS more_than_five,
    AVG(candidate_count) AS avg_candidates,
    MAX(candidate_count) AS max_candidates
FROM (
    SELECT
        s.entity_id,
        COUNT(t.entity_id) AS candidate_count
    FROM source1 s
    JOIN target t
        ON s.n_country = t.n_country
        AND s.n_name = t.n_name
    WHERE
        s.n_country <> ''
        AND s.n_name <> ''
    GROUP BY s.entity_id
    HAVING COUNT(t.entity_id) > 1
)
"""

result = con.execute(q).fetchone()

print("AMBIGUITY ANALYSIS")
print("==================")
print("Ambiguous sources :", result[0])
print("1 candidate       :", result[1])
print("2 candidates      :", result[2])
print("3-5 candidates    :", result[3])
print(">5 candidates     :", result[4])
print("Average candidates:", result[5])
print("Maximum candidates:", result[6])

con.close()
