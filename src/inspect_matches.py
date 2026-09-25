import duckdb
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "amazon_ml.duckdb"
TRAIN_DIR = PROJECT_ROOT / "data" / "dataset" / "train"


con = duckdb.connect(str(DB_PATH))


# Pick one Source 1 entity
source1_id = "S1-965667"


# Get Source 1 record
source1 = con.execute(f"""
    SELECT *
    FROM read_csv_auto(
        '{(TRAIN_DIR / "train_source1.tsv").as_posix()}',
        delim='\t'
    )
    WHERE entity_id = '{source1_id}'
""").fetchdf()


print("\nSOURCE 1")
print("=" * 80)
print(source1.to_string(index=False))


# Get ground truth
truth = con.execute(f"""
    SELECT *
    FROM read_csv_auto(
        '{(TRAIN_DIR / "train_ground_truth.tsv").as_posix()}',
        delim='\t'
    )
    WHERE source1_entity_id = '{source1_id}'
""").fetchdf()


print("\nGROUND TRUTH")
print("=" * 80)
print(truth.to_string(index=False))


# Extract matching IDs
matched_ids = truth.iloc[0]["matched_entity_ids"].split(",")


print("\nTRUE MATCHES")
print("=" * 80)


# Source 2 matches
source2_ids = [
    x.strip()
    for x in matched_ids
    if x.strip().startswith("S2-")
]

if source2_ids:

    ids = ",".join(f"'{x}'" for x in source2_ids)

    source2 = con.execute(f"""
        SELECT *
        FROM read_csv_auto(
            '{(TRAIN_DIR / "train_source2.tsv").as_posix()}',
            delim='\t'
        )
        WHERE entity_id IN ({ids})
    """).fetchdf()

    print("\nSOURCE 2 MATCHES")
    print(source2.to_string(index=False))


# Source 3 matches
source3_ids = [
    x.strip()
    for x in matched_ids
    if x.strip().startswith("S3-")
]

if source3_ids:

    ids = ",".join(f"'{x}'" for x in source3_ids)

    source3 = con.execute(f"""
        SELECT *
        FROM read_csv_auto(
            '{(TRAIN_DIR / "train_source3.tsv").as_posix()}',
            delim='\t'
        )
        WHERE entity_id IN ({ids})
    """).fetchdf()

    print("\nSOURCE 3 MATCHES")
    print(source3.to_string(index=False))


con.close()