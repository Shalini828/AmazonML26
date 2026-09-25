import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

FEATURE_PATH = PROJECT_ROOT / "data" / "training_features.tsv"
MODEL_PATH = PROJECT_ROOT / "data" / "rf_model.joblib"


def main():

    print("Loading training features...")

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t"
    )

    feature_columns = [
        "name_similarity",
        "address_similarity",
        "name_token_similarity",
        "address_token_similarity",
        "country_match"
    ]

    X = df[feature_columns]
    y = df["label"]

    print("Rows:", len(df))
    print("Features:", feature_columns)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("\nTraining Random Forest...")

    model.fit(X, y)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\n==============================")
    print("MODEL SAVED")
    print("==============================")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()