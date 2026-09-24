import pandas as pd
from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)

files = {
    "TRAIN SOURCE 1": TRAIN_SOURCE1,
    "TRAIN SOURCE 2": TRAIN_SOURCE2,
    "TRAIN SOURCE 3": TRAIN_SOURCE3,
    "TRAIN GROUND TRUTH": TRAIN_GROUND_TRUTH,
    "TEST SOURCE 1": TEST_SOURCE1,
    "TEST SOURCE 2": TEST_SOURCE2,
    "TEST SOURCE 3": TEST_SOURCE3,
}

for name, file in files.items():
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    df = pd.read_csv(file, sep="\t", nrows=5)

    print("Columns:")
    print(df.columns.tolist())

    print("\nShape of sample:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))