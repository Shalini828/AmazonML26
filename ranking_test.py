import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit

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

groups = df["source1_entity_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, df["label"], groups=groups)
)

train = df.iloc[train_idx].copy()
val = df.iloc[val_idx].copy()

model = joblib.load("data/rf_model.joblib")

val["probability"] = model.predict_proba(
    val[features]
)[:, 1]

# Rank candidates within each source record
val["rank"] = (
    val.groupby("source1_entity_id")["probability"]
    .rank(
        method="first",
        ascending=False
    )
)

# Get best candidate per source
best = (
    val.sort_values(
        ["source1_entity_id", "probability"],
        ascending=[True, False]
    )
    .groupby("source1_entity_id")
    .head(1)
    .copy()
)

# Get second-best probability
second = (
    val[val["rank"] == 2]
    [
        [
            "source1_entity_id",
            "probability"
        ]
    ]
    .rename(
        columns={
            "probability": "second_probability"
        }
    )
)

best = best.merge(
    second,
    on="source1_entity_id",
    how="left"
)

best["second_probability"] = (
    best["second_probability"].fillna(0)
)

best["margin"] = (
    best["probability"]
    - best["second_probability"]
)

# Correct if top candidate is actually positive
best["correct"] = (
    best["label"] == 1
)

print("RANKING ANALYSIS")
print("================")

print("Validation source records:", len(best))

print()
print("TOP CANDIDATE CORRECT:")
print(
    best["correct"]
    .mean()
)

print()
print("MARGIN DISTRIBUTION:")
print(
    best["margin"].describe()
)

print()
print("TOP PROBABILITY DISTRIBUTION:")
print(
    best["probability"].describe()
)

print()
print("MARGIN THRESHOLDS")
print("=================")

for margin in [0.01, 0.03, 0.05, 0.10, 0.20]:

    accepted = best[
        (best["probability"] >= 0.95)
        &
        (best["margin"] >= margin)
    ]

    if len(accepted) == 0:
        continue

    precision = accepted["correct"].mean()

    coverage = (
        len(accepted) / len(best)
    )

    print(
        f"margin={margin:.2f} "
        f"accepted={len(accepted):,} "
        f"precision={precision:.4f} "
        f"coverage={coverage:.4f}"
    )
