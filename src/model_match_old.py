import pandas as pd
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "data" / "rf_model_old.joblib"
FEATURE_PATH = PROJECT_ROOT / "data" / "training_features.tsv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "model_predictions_old.tsv"


OLD_MODEL_FEATURES = [
    "name_similarity",
    "address_similarity",
    "name_token_similarity",
    "address_token_similarity",
    "country_match",
]


def main():

    print("=" * 60)
    print("OLD MODEL MATCHING")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Old model not found: {MODEL_PATH}"
        )

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}"
        )

    print("\nLoading old trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model type:")
    print(type(model).__name__)

    print("\nLoading feature data...")

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nChecking old model features...")

    missing_features = [
        feature
        for feature in OLD_MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:
            print("  -", feature)

        raise ValueError(
            "Old model feature mismatch detected."
        )

    print("All old model features available.")

    print("\nBuilding old model input...")

    X = df[OLD_MODEL_FEATURES].copy()

    for column in OLD_MODEL_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X = X.fillna(0)

    print(
        "Feature matrix shape:",
        X.shape
    )

    print("\nGenerating old model predictions...")

    predictions = model.predict(X)

    print("Predictions generated.")

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        if hasattr(model, "classes_"):

            classes = list(model.classes_)

            if 1 in classes:

                positive_index = classes.index(1)

                match_probability = probabilities[
                    :,
                    positive_index
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
    print("OLD MODEL MATCHING COMPLETE")
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

    print("\nSaved old predictions to:")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()