import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_PATH = PROJECT_ROOT / "data" / "training_features.tsv"
MODEL_PATH = PROJECT_ROOT / "data" / "rf_model.joblib"


FEATURE_COLUMNS = [
    "name_exact",
    "name_compact_exact",
    "name_similarity",
    "name_ratio",
    "name_token_sort_similarity",
    "name_token_similarity",
    "name_token_set_similarity",
    "name_token_overlap",
    "name_jaro_winkler",
    "name_ngram_similarity",
    "name_weighted_similarity",
    "name_length_ratio",

    "address_exact",
    "address_compact_exact",
    "address_similarity",
    "address_ratio",
    "address_token_sort_similarity",
    "address_token_similarity",
    "address_token_set_similarity",
    "address_token_overlap",
    "address_jaro_winkler",
    "address_ngram_similarity",
    "address_weighted_similarity",
    "address_length_ratio",

    "house_number_match",
    "house_number_exact_match",

    "country_match",
    "source_country_missing",
    "candidate_country_missing",
    "country_both_present",

    "postal_code_match",
    "phone_match",
    "email_match",
    "website_match",
]


def main():

    print("Loading training features...")

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    available_features = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    print("\nAvailable model features:")

    for feature in available_features:
        print("  +", feature)

    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:
            print("  -", feature)

    if not available_features:
        raise RuntimeError(
            "No model features found in training_features.tsv"
        )

    X = df[available_features].fillna(0)
    y = df["label"].astype(int)

    print("\nLabel distribution:")
    print(y.value_counts().sort_index())

    print(
        "\nPositive rate:",
        round(y.mean(), 6)
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_valid))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("\nEvaluating validation set...")

    predictions = model.predict(
        X_valid
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    print("\nClassification report:")

    print(
        classification_report(
            y_valid,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    try:

        auc = roc_auc_score(
            y_valid,
            probabilities
        )

        print(
            "Validation ROC-AUC:",
            round(auc, 6)
        )

    except ValueError:

        print(
            "ROC-AUC could not be calculated."
        )

    importance = pd.Series(
        model.feature_importances_,
        index=available_features
    ).sort_values(
        ascending=False
    )

    print("\nTop feature importances:")

    for feature, value in importance.head(15).items():

        print(
            f"  {feature:35s} {value:.6f}"
        )

    model_bundle = {
        "model": model,
        "feature_columns": available_features
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Features used:",
        len(available_features)
    )

    print(
        "Training rows:",
        len(X_train)
    )

    print(
        "Validation rows:",
        len(X_valid)
    )

    print("\nModel saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()