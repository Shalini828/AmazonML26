import polars as pl
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


FILES = {
    "Source 1": "train_source1.parquet",
    "Source 2": "train_source2.parquet",
    "Source 3": "train_source3.parquet",
    "Ground Truth": "train_ground_truth.parquet",
}


for name, filename in FILES.items():

    path = PROCESSED_DIR / filename

    df = pl.read_parquet(path)

    print("=" * 60)
    print(name)
    print("Rows:", f"{df.height:,}")
    print("Columns:", df.columns)
    print("Schema:", df.schema)
    print()