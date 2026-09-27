import pandas as pd
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "data" / "rf_model.joblib"
FEATURE_PATH = PROJECT_ROOT / "data" / "training_features.tsv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "model_predictions.tsv"


MODEL_FEATURES = [
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

    print("=" * 60)
    print("MODEL MATCHING")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}"
        )

    print("\nLoading trained model...")

    saved_object = joblib.load(MODEL_PATH)

    print("Saved object type:")
    print(type(saved_object).__name__)

    if isinstance(saved_object, dict):

        if "model" not in saved_object:
            raise ValueError(
                "rf_model.joblib is a dictionary, "
                "but it does not contain a 'model' key."
            )

        model = saved_object["model"]

        print("Extracted model from dictionary.")

    else:

        model = saved_object

        print("Loaded model directly.")

    print("Actual model type:")
    print(type(model).__name__)

    print("\nLoading feature data...")

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    required_columns = [
        "source1_entity_id",
        "candidate_entity_id",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    print("\nChecking model features...")

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:
            print("  -", feature)

        raise ValueError(
            "Feature mismatch detected."
        )

    print("All model features available.")

    print("\nBuilding model input...")

    X = df[MODEL_FEATURES].copy()

    for column in MODEL_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X = X.fillna(0)

    print(
        "Feature matrix shape:",
        X.shape
    )

    print("\nGenerating predictions...")

    predictions = model.predict(X)

    print("Predictions generated.")

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        if hasattr(model, "classes_"):

            classes = list(model.classes_)

            if 1 in classes:

                positive_index = classes.index(1)

                match_probability = probabilities[
                    :, positive_index
                ]

            else:

                match_probability = probabilities[:, -1]

        else:

            match_probability = probabilities[:, -1]

    else:

        match_probability = predictions.astype(float)

    print("Probabilities generated.")

    output = df[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].copy()

    output["match_probability"] = match_probability

    output["prediction"] = predictions

    output.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print("\n" + "=" * 60)
    print("MODEL MATCHING COMPLETE")
    print("=" * 60)

    print(
        "Rows scored:",
        len(output)
    )

    print(
        "Predicted matches:",
        int(
            (output["prediction"] == 1).sum()
        )
    )

    print(
        "Predicted non-matches:",
        int(
            (output["prediction"] == 0).sum()
        )
    )

    print(
        "Match rate:",
        round(
            float(
                output["prediction"].mean()
            ),
            6
        )
    )

    print("\nProbability statistics:")

    print(
        output["match_probability"].describe()
    )

    print("\nSaved predictions to:")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()