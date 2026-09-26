import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_PAIRS_PATH = (
    PROJECT_ROOT / "data" / "training_pairs.tsv"
)

NEW_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "model_predictions.tsv"
)

OLD_PREDICTION_CANDIDATES = [
    PROJECT_ROOT / "data" / "model_predictions_old.tsv",
    PROJECT_ROOT / "data" / "model_predictions_random.tsv",
    PROJECT_ROOT / "data" / "model_predictions_backup.tsv",
    PROJECT_ROOT / "data" / "model_predictions_v1.tsv",
]


# =========================================================
# FIND OLD PREDICTIONS
# =========================================================

def find_old_prediction_file():

    for path in OLD_PREDICTION_CANDIDATES:

        if path.exists():
            return path

    return None


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

def load_ground_truth():

    print()
    print("Loading ground-truth labels...")

    df = pd.read_csv(
        TRAINING_PAIRS_PATH,
        sep="\t"
    )

    print(
        "Training pairs:",
        len(df)
    )

    required_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # -----------------------------------------------------
    # Check duplicate pair IDs
    # -----------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ).sum()

    print(
        "Duplicate pair rows:",
        duplicate_count
    )

    # -----------------------------------------------------
    # Check whether duplicate pairs have conflicting labels
    # -----------------------------------------------------

    conflicts = (
        df.groupby(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )["label"]
        .nunique()
    )

    conflicting_pairs = int(
        (conflicts > 1).sum()
    )

    print(
        "Conflicting duplicate pairs:",
        conflicting_pairs
    )

    if conflicting_pairs > 0:

        raise ValueError(
            "Some source/candidate pairs have "
            "both label 0 and label 1."
        )

    # -----------------------------------------------------
    # Remove exact duplicate pairs
    # -----------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ).copy()

    print(
        "Unique ground-truth pairs:",
        len(df)
    )

    print()
    print("Label distribution:")

    print(
        df["label"].value_counts()
    )

    return df


# =========================================================
# LOAD PREDICTIONS
# =========================================================

def load_predictions(
    prediction_path,
    ground_truth,
):

    print()
    print("Loading:")
    print(prediction_path)

    predictions = pd.read_csv(
        prediction_path,
        sep="\t"
    )

    required_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "prediction",
        "match_probability",
    ]

    for column in required_columns:

        if column not in predictions.columns:

            raise ValueError(
                f"Missing column '{column}' "
                f"in {prediction_path.name}"
            )

    print(
        "Prediction rows:",
        len(predictions)
    )

    # -----------------------------------------------------
    # Convert IDs to strings
    # -----------------------------------------------------

    id_columns = [
        "source1_entity_id",
        "candidate_entity_id",
    ]

    for column in id_columns:

        predictions[column] = (
            predictions[column]
            .astype(str)
        )

        ground_truth[column] = (
            ground_truth[column]
            .astype(str)
        )

    # -----------------------------------------------------
    # Remove duplicate predictions
    # -----------------------------------------------------

    predictions = predictions.drop_duplicates(
        subset=id_columns,
        keep="first"
    ).copy()

    # -----------------------------------------------------
    # Merge with ground truth
    # -----------------------------------------------------

    merged = predictions.merge(
        ground_truth,
        on=id_columns,
        how="inner",
        validate="one_to_one"
    )

    print(
        "Unique prediction rows:",
        len(predictions)
    )

    print(
        "Rows matched with ground truth:",
        len(merged)
    )

    if len(merged) == 0:

        raise ValueError(
            "No prediction rows matched "
            "the ground truth."
        )

    return merged


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(df):

    y_true = df["label"]

    y_pred = df["prediction"]

    y_probability = (
        df["match_probability"]
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix,
    }


# =========================================================
# PRINT MODEL RESULTS
# =========================================================

def print_model_results(
    title,
    df,
    metrics,
):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print()

    print(
        "Rows evaluated:",
        len(df)
    )

    print()

    print(
        "Accuracy :",
        f"{metrics['accuracy']:.6f}"
    )

    print(
        "Precision:",
        f"{metrics['precision']:.6f}"
    )

    print(
        "Recall   :",
        f"{metrics['recall']:.6f}"
    )

    print(
        "F1 Score :",
        f"{metrics['f1']:.6f}"
    )

    print(
        "ROC-AUC  :",
        f"{metrics['roc_auc']:.6f}"
    )

    print()

    print("Confusion Matrix:")

    print(
        metrics["confusion_matrix"]
    )

    predicted_matches = int(
        (df["prediction"] == 1).sum()
    )

    predicted_non_matches = int(
        (df["prediction"] == 0).sum()
    )

    print()

    print(
        "Predicted matches    :",
        predicted_matches
    )

    print(
        "Predicted non-matches:",
        predicted_non_matches
    )

    print(
        "Match rate           :",
        f"{predicted_matches / len(df):.6f}"
    )

    print()

    print(
        "Average probability:",
        f"{df['match_probability'].mean():.6f}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("MODEL SCORE COMPARISON")
    print("=" * 60)

    # -----------------------------------------------------
    # Check required files
    # -----------------------------------------------------

    if not TRAINING_PAIRS_PATH.exists():

        raise FileNotFoundError(
            f"Training pairs not found:\n"
            f"{TRAINING_PAIRS_PATH}"
        )

    if not NEW_PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            f"New predictions not found:\n"
            f"{NEW_PREDICTIONS_PATH}"
        )

    # -----------------------------------------------------
    # Load ground truth
    # -----------------------------------------------------

    ground_truth = load_ground_truth()

    # -----------------------------------------------------
    # NEW MODEL
    # -----------------------------------------------------

    new_predictions = load_predictions(
        NEW_PREDICTIONS_PATH,
        ground_truth.copy()
    )

    new_metrics = calculate_metrics(
        new_predictions
    )

    print_model_results(
        "NEW HARD-NEGATIVE MODEL",
        new_predictions,
        new_metrics
    )

    # -----------------------------------------------------
    # Find OLD model
    # -----------------------------------------------------

    old_prediction_path = (
        find_old_prediction_file()
    )

    if old_prediction_path is None:

        print()
        print("=" * 60)
        print("OLD MODEL PREDICTIONS NOT FOUND")
        print("=" * 60)

        print()

        print(
            "New model score is valid."
        )

        print()

        print(
            "For OLD vs NEW comparison, "
            "place the old prediction file "
            "inside:"
        )

        print(
            PROJECT_ROOT / "data"
        )

        print()

        print(
            "Accepted filenames:"
        )

        for path in OLD_PREDICTION_CANDIDATES:

            print(
                "  -",
                path.name
            )

        print()

        print(
            "Then run:"
        )

        print(
            "python .\\src\\compare_scores.py"
        )

        return

    # -----------------------------------------------------
    # OLD MODEL
    # -----------------------------------------------------

    old_predictions = load_predictions(
        old_prediction_path,
        ground_truth.copy()
    )

    old_metrics = calculate_metrics(
        old_predictions
    )

    print_model_results(
        "OLD RANDOM-NEGATIVE MODEL",
        old_predictions,
        old_metrics
    )

    # -----------------------------------------------------
    # OLD VS NEW
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("OLD vs NEW COMPARISON")
    print("=" * 60)

    print()

    print(
        f"{'Metric':<20}"
        f"{'Old Model':>15}"
        f"{'New Model':>15}"
        f"{'Difference':>15}"
    )

    print("-" * 65)

    comparison = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]

    for display_name, key in comparison:

        old_value = old_metrics[key]

        new_value = new_metrics[key]

        difference = (
            new_value - old_value
        )

        print(
            f"{display_name:<20}"
            f"{old_value:>15.6f}"
            f"{new_value:>15.6f}"
            f"{difference:>15.6f}"
        )

    # -----------------------------------------------------
    # Match count comparison
    # -----------------------------------------------------

    old_matches = int(
        (old_predictions["prediction"] == 1).sum()
    )

    new_matches = int(
        (new_predictions["prediction"] == 1).sum()
    )

    print()

    print(
        "Predicted match count:"
    )

    print(
        "Old:",
        old_matches
    )

    print(
        "New:",
        new_matches
    )

    print(
        "Difference:",
        new_matches - old_matches
    )

    # -----------------------------------------------------
    # Probability comparison
    # -----------------------------------------------------

    old_probability = (
        old_predictions[
            "match_probability"
        ].mean()
    )

    new_probability = (
        new_predictions[
            "match_probability"
        ].mean()
    )

    print()

    print(
        "Average match probability:"
    )

    print(
        "Old:",
        f"{old_probability:.6f}"
    )

    print(
        "New:",
        f"{new_probability:.6f}"
    )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print()

    print("=" * 60)
    print("SCORE COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()