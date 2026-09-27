import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    fbeta_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_PATH = PROJECT_ROOT / "data" / "training_features.tsv"
MODEL_PATH = PROJECT_ROOT / "data" / "rf_model.joblib"


# ============================================================
# EXACT 30 FEATURES
# Must match features.py and inference feature order.
# ============================================================

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
]


def main():

    print("=" * 70)
    print("LOADING TRAINING FEATURES")
    print("=" * 70)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Training feature file not found:\n{FEATURE_PATH}"
        )

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # ========================================================
    # STRICT 30-FEATURE CHECK
    # ========================================================

    if len(FEATURE_COLUMNS) != 30:
        raise RuntimeError(
            f"Internal error: expected 30 features, "
            f"but FEATURE_COLUMNS contains {len(FEATURE_COLUMNS)}"
        )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:
        print("\nMISSING FEATURES:")

        for feature in missing_features:
            print("  -", feature)

        raise RuntimeError(
            "training_features.tsv does not contain all 30 required features."
        )

    if "label" not in df.columns:
        raise RuntimeError(
            "training_features.tsv does not contain a 'label' column."
        )

    print("\nUsing EXACTLY these 30 model features:")

    for i, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"  {i:02d}. {feature}")

    # IMPORTANT:
    # Ignore any extra columns such as:
    # postal_code_match
    # phone_match
    # email_match
    # website_match
    #
    # They will NOT enter this model.

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["label"].astype(int)

    print("\nFeature matrix shape:", X.shape)

    if X.shape[1] != 30:
        raise RuntimeError(
            f"Expected 30 feature columns, got {X.shape[1]}"
        )

    # ========================================================
    # LABEL CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    print(y.value_counts().sort_index())

    print(
        "\nPositive rate:",
        round(y.mean(), 6)
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_valid))

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # VERIFY MODEL DIMENSION
    # ========================================================

    print("\nModel feature count:", model.n_features_in_)

    if model.n_features_in_ != 30:
        raise RuntimeError(
            f"MODEL FEATURE MISMATCH: "
            f"expected 30, got {model.n_features_in_}"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

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

    # ========================================================
    # ROC-AUC
    # ========================================================

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

    # ========================================================
    # F0.5 THRESHOLD SEARCH
    # ========================================================

    print("\n" + "=" * 70)
    print("SEARCHING BEST F0.5 THRESHOLD")
    print("=" * 70)

    best_threshold = 0.50
    best_f05 = -1.0

    for i in range(100, 1000):

        threshold = i / 1000.0

        threshold_predictions = (
            probabilities >= threshold
        ).astype(int)

        score = fbeta_score(
            y_valid,
            threshold_predictions,
            beta=0.5,
            zero_division=0
        )

        if score > best_f05:

            best_f05 = score
            best_threshold = threshold

    print(
        f"Best validation F0.5: {best_f05:.6f}"
    )

    print(
        f"Best threshold: {best_threshold:.3f}"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(
        ascending=False
    )

    print("\n" + "=" * 70)
    print("TOP FEATURE IMPORTANCES")
    print("=" * 70)

    for feature, value in importance.head(15).items():

        print(
            f"  {feature:35s} {value:.6f}"
        )

    # ========================================================
    # SAVE MODEL BUNDLE
    # ========================================================

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best_threshold,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Features used:",
        len(FEATURE_COLUMNS)
    )

    print(
        "Model n_features_in_:",
        model.n_features_in_
    )

    print(
        "Training rows:",
        len(X_train)
    )

    print(
        "Validation rows:",
        len(X_valid)
    )

    print(
        "Validation F0.5:",
        round(best_f05, 6)
    )

    print(
        "Selected threshold:",
        round(best_threshold, 3)
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\n" + "=" * 70)
    print("SAVED MODEL CHECK")
    print("=" * 70)

    saved = joblib.load(MODEL_PATH)

    saved_model = saved["model"]
    saved_features = saved["feature_columns"]
    saved_threshold = saved.get("threshold")

    print(
        "Saved model features:",
        saved_model.n_features_in_
    )

    print(
        "Saved feature list length:",
        len(saved_features)
    )

    print(
        "Saved threshold:",
        saved_threshold
    )

    if saved_model.n_features_in_ != 30:
        raise RuntimeError(
            "Saved model is NOT a 30-feature model."
        )

    if len(saved_features) != 30:
        raise RuntimeError(
            "Saved feature list is NOT 30 features."
        )

    print("\nSUCCESS: 30-feature model saved correctly.")


if __name__ == "__main__":
    main()