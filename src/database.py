import polars as pl
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


FILES = {
    "source1": PROCESSED_DIR / "train_source1.parquet",
    "source2": PROCESSED_DIR / "train_source2.parquet",
    "source3": PROCESSED_DIR / "train_source3.parquet",
    "ground_truth": PROCESSED_DIR / "train_ground_truth.parquet",
    "relationships": PROCESSED_DIR / "train_relationships.parquet",
}


def get_file(name: str) -> Path:
    """Return the Parquet file for a dataset."""

    if name not in FILES:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(FILES.keys())}"
        )

    return FILES[name]


def scan(name: str):
    """Lazily scan a Parquet dataset."""

    path = get_file(name)

    return pl.scan_parquet(path)


def read(name: str) -> pl.DataFrame:
    """Read an entire dataset into a Polars DataFrame."""

    return pl.read_parquet(get_file(name))


def count(name: str) -> int:
    """Return the number of records without loading the full dataset."""

    return (
        scan(name)
        .select(pl.len())
        .collect()
        .item()
    )


def filter_by_country(name: str, country: str) -> pl.DataFrame:
    """Return records belonging to a specific country."""

    return (
        scan(name)
        .filter(
            pl.col("country") == country
        )
        .collect()
    )


def select_columns(
    name: str,
    columns: list[str]
) -> pl.DataFrame:
    """Return only the requested columns."""

    available = get_columns(name)

    invalid = [
        column
        for column in columns
        if column not in available
    ]

    if invalid:
        raise ValueError(
            f"Invalid columns: {invalid}. "
            f"Available columns: {available}"
        )

    return (
        scan(name)
        .select(columns)
        .collect()
    )

def get_by_id(name: str, entity_id: str) -> pl.DataFrame:
    """Return the record with the specified entity ID."""

    return (
        scan(name)
        .filter(
            pl.col("entity_id") == entity_id
        )
        .collect()
    )

def get_by_ids(name: str, entity_ids: list[str]) -> pl.DataFrame:
    """Return records matching any of the supplied entity IDs."""

    if not entity_ids:
        return pl.DataFrame()

    return (
        scan(name)
        .filter(
            pl.col("entity_id").is_in(entity_ids)
        )
        .collect()
    )

def get_matches(source1_entity_id: str) -> pl.DataFrame:
    """Return all matched S2/S3 entity IDs for a Source 1 entity."""

    return (
        scan("relationships")
        .filter(
            pl.col("source1_entity_id") == source1_entity_id
        )
        .select("matched_entity_id")
        .collect()
    )

def get_columns(name: str) -> list[str]:
    """Return the available columns for a dataset."""

    return scan(name).collect_schema().names()

if __name__ == "__main__":

    print("AmazonML26 Data Layer")
    print("=" * 40)

    source1_id = "S1-965667"

    matches = get_matches(source1_id)

    print(f"Source 1: {source1_id}")
    print(f"Number of matches: {matches.height}")
    print()
    print(matches)