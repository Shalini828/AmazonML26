import polars as pl
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "dataset" / "train"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "source1": (
        TRAIN_DIR / "train_source1.tsv",
        PROCESSED_DIR / "train_source1.parquet",
    ),

    "source2": (
    TRAIN_DIR / "train_source2.tsv",
    PROCESSED_DIR / "train_source2.parquet",
),

    "source3": (
        TRAIN_DIR / "train_source3.tsv",
        PROCESSED_DIR / "train_source3.parquet",
    ),
    "ground_truth": (
        TRAIN_DIR / "train_ground_truth.tsv",
        PROCESSED_DIR / "train_ground_truth.parquet",
    ),
}

TEST_DIR = PROJECT_ROOT / "data" / "dataset" / "test"

FILES.update({
    "test_source1": (
        TEST_DIR / "test_source1.tsv",
        PROCESSED_DIR / "test_source1.parquet",
    ),
    "test_source2": (
        TEST_DIR / "test_source2.tsv",
        PROCESSED_DIR / "test_source2.parquet",
    ),
    "test_source3": (
        TEST_DIR / "test_source3.tsv",
        PROCESSED_DIR / "test_source3.parquet",
    ),
})

for name, (input_file, output_file) in FILES.items():

    print("=" * 60)
    print(f"Converting {name}")
    print("Input :", input_file)
    print("Output:", output_file)
    print()

    start = time.perf_counter()

    (
        pl.scan_csv(
            input_file,
            separator="\t",
            infer_schema_length=10000,
        )
        .sink_parquet(output_file)
    )

    elapsed = time.perf_counter() - start

    size_mb = output_file.stat().st_size / (1024 ** 2)

    print(f"Done in {elapsed:.2f} seconds")
    print(f"Parquet size: {size_mb:.2f} MB")
    print()