"""Lightweight integrity checks for the emergency TSV submission."""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DB = OUT / "matcher_work.duckdb"
CANDIDATES = OUT / "candidate_pairs.tsv"
MATCHES = OUT / "matching_results.tsv"


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> None:
    assert CANDIDATES.is_file() and MATCHES.is_file(), "submission output file missing"
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET memory_limit='2GB'")
    cand = f"read_csv('{qpath(CANDIDATES)}', delim='\\t', header=true, all_varchar=true)"
    match = f"read_csv('{qpath(MATCHES)}', delim='\\t', header=true, all_varchar=true)"

    def scalar(sql: str):
        return con.execute(sql).fetchone()[0]

    expected = scalar("SELECT count(*) FROM source1")
    source_unique = scalar("SELECT count(DISTINCT entity_id) FROM source1")
    c_rows = scalar(f"SELECT count(*) FROM {cand}")
    c_unique = scalar(f"SELECT count(DISTINCT source1_entity_id) FROM {cand}")
    m_rows = scalar(f"SELECT count(*) FROM {match}")
    m_unique = scalar(f"SELECT count(DISTINCT source1_entity_id) FROM {match}")
    assert c_rows == expected == c_unique, f"candidate rows/unique IDs {c_rows}/{c_unique}, expected {expected}"
    assert m_rows == expected == m_unique, f"match rows/unique IDs {m_rows}/{m_unique}, expected {expected}"
    assert source_unique == expected, "Source 1 itself contains duplicate IDs"

    missing_c = scalar(f"SELECT count(*) FROM source1 s LEFT JOIN {cand} c ON s.entity_id=c.source1_entity_id WHERE c.source1_entity_id IS NULL")
    extra_c = scalar(f"SELECT count(*) FROM {cand} c LEFT JOIN source1 s ON s.entity_id=c.source1_entity_id WHERE s.entity_id IS NULL")
    missing_m = scalar(f"SELECT count(*) FROM source1 s LEFT JOIN {match} m ON s.entity_id=m.source1_entity_id WHERE m.source1_entity_id IS NULL")
    assert missing_c == extra_c == missing_m == 0, "Source 1 coverage mismatch"

    dup_candidates = scalar(f"""
        SELECT count(*) FROM {cand}
        WHERE candidate_entity_ids<>'' AND
              array_length(string_split(candidate_entity_ids, ',')) <>
              list_unique(string_split(candidate_entity_ids, ','))
    """)
    invalid_candidates = scalar(f"""
        SELECT count(*) FROM (
            SELECT c.source1_entity_id, unnest(string_split(c.candidate_entity_ids, ',')) AS id
            FROM {cand} c WHERE c.candidate_entity_ids<>''
        ) x LEFT JOIN target t ON t.entity_id=x.id WHERE t.entity_id IS NULL
    """)
    dup_matches = scalar(f"""
        SELECT count(*) FROM {match}
        WHERE matched_entity_ids<>'' AND
              array_length(string_split(matched_entity_ids, ',')) <>
              list_unique(string_split(matched_entity_ids, ','))
    """)
    invalid_matches = scalar(f"""
        SELECT count(*) FROM (
            SELECT m.source1_entity_id, unnest(string_split(m.matched_entity_ids, ',')) AS id
            FROM {match} m WHERE m.matched_entity_ids<>''
        ) x LEFT JOIN target t ON t.entity_id=x.id WHERE t.entity_id IS NULL
    """)
    not_candidates = scalar(f"""
        SELECT count(*) FROM {match} m JOIN {cand} c USING (source1_entity_id)
        WHERE m.matched_entity_ids<>'' AND
              NOT list_contains(string_split(c.candidate_entity_ids, ','), m.matched_entity_ids)
    """)
    assert dup_candidates == invalid_candidates == dup_matches == invalid_matches == not_candidates == 0, (
        f"invalid IDs: candidate duplicates={dup_candidates}, candidate unknown={invalid_candidates}, "
        f"match duplicates={dup_matches}, match unknown={invalid_matches}, not candidate={not_candidates}"
    )
    print(f"VALIDATION PASSED: {expected:,} Source 1 rows appear exactly once in both files")
    print("All candidate/matched IDs are valid; no duplicates; every match is a candidate.")
    con.close()


if __name__ == "__main__":
    main()
