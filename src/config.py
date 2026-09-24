from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EDA_DATA_DIR = DATA_DIR / "eda"

# Other directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


# Training files
TRAIN_SOURCE1 = RAW_DATA_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = RAW_DATA_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = RAW_DATA_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = RAW_DATA_DIR / "train_ground_truth.tsv"

# Test files
TEST_SOURCE1 = RAW_DATA_DIR / "test_source1.tsv"
TEST_SOURCE2 = RAW_DATA_DIR / "test_source2.tsv"
TEST_SOURCE3 = RAW_DATA_DIR / "test_source3.tsv"