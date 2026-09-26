import pandas as pd
import joblib

model = joblib.load("data/rf_model.joblib")

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

probs = model.predict_proba(X)[:, 1]

df["probability"] = probs

print("MODEL SCORE ANALYSIS")
print("====================")

print("Minimum :", df["probability"].min())
print("Maximum :", df["probability"].max())
print("Mean    :", df["probability"].mean())

print()
print("POSITIVE LABEL SCORES")
print(df[df["label"] == 1]["probability"].describe())

print()
print("NEGATIVE LABEL SCORES")
print(df[df["label"] == 0]["probability"].describe())

print()
print("SCORES >= 0.90:")
print((df["probability"] >= 0.90).sum())

print("SCORES >= 0.95:")
print((df["probability"] >= 0.95).sum())

print("SCORES >= 0.99:")
print((df["probability"] >= 0.99).sum())
