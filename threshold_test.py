import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv(
    "data/training_features.tsv",
    sep="\t"
)

features = [
    "name_similarity",
    "address_similarity",
    "name_token_similarity",
    "address_token_similarity",
    "country_match"
]

X = df[features]
y = df["label"]

groups = df["source1_entity_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

model = joblib.load("data/rf_model.joblib")

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

prob = model.predict_proba(X_val)[:, 1]

print("THRESHOLD ANALYSIS")
print("==================")

for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]:

    pred = (prob >= threshold).astype(int)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        pred,
        zero_division=0
    )

    accepted = pred.sum()

    print(
        f"threshold={threshold:.2f} "
        f"accepted={accepted:,} "
        f"precision={precision:.4f} "
        f"recall={recall:.4f} "
        f"f1={f1:.4f}"
    )
