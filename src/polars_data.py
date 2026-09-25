import polars as pl
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "dataset" / "test"


def scan_tsv(path: Path):
    """Lazily scan a TSV file using Polars."""
    return pl.scan_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
    )


def train_source1():
    return scan_tsv(TRAIN_DIR / "train_source1.tsv")


def train_source2():
    return scan_tsv(TRAIN_DIR / "train_source2.tsv")


def train_source3():
    return scan_tsv(TRAIN_DIR / "train_source3.tsv")


def train_ground_truth():
    return scan_tsv(TRAIN_DIR / "train_ground_truth.tsv")


if __name__ == "__main__":
    df = train_source2()

    print("Lazy Source 2 scan created.")
    print()

    print(
        df.select(
            pl.len().alias("rows")
        ).collect()
    )

    print()
    print(
        df.select(
            [
                pl.col("country").value_counts()
            ]
        ).collect()
    )