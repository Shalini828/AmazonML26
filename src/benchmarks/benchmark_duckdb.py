import duckdb
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE = PROJECT_ROOT / "data" / "dataset" / "train" / "train_source2.tsv"

print("File:", FILE)
print()
print("Starting DuckDB benchmark...")

start = time.perf_counter()

con = duckdb.connect()

result = con.execute(f"""
    SELECT *
    FROM read_csv_auto(
        '{FILE.as_posix()}',
        delim='\t'
    )
""").fetchall()

elapsed = time.perf_counter() - start

print(f"Rows: {len(result):,}")
print(f"Columns: {len(result[0])}")
print(f"Time: {elapsed:.2f} seconds")

print()
print(result[:5])

con.close()