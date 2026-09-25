from pathlib import Path


# Project root
DATA_ROOT = Path(
    r"C:\Users\R K SINGH\Downloads\6ab10eb3b23ba_student_resource\student_resource"
)


# Data directories
TRAIN_DIR = DATA_ROOT / "dataset" / "train"
TEST_DIR = DATA_ROOT / "dataset" / "test"


# Training files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


# Test files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"