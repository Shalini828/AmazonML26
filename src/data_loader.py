import pandas as pd

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)


def load_training_data():
    """
    Load all training datasets.
    """

    source1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t"
    )

    source2 = pd.read_csv(
        TRAIN_SOURCE2,
        sep="\t"
    )

    source3 = pd.read_csv(
        TRAIN_SOURCE3,
        sep="\t"
    )

    ground_truth = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t"
    )

    return source1, source2, source3, ground_truth


if __name__ == "__main__":

    source1, source2, source3, ground_truth = load_training_data()

    print("Source 1:", source1.shape)
    print("Source 2:", source2.shape)
    print("Source 3:", source3.shape)
    print("Ground Truth:", ground_truth.shape)