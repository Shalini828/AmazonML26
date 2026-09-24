import pandas as pd

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)

# Read a small sample of Source 1
source1 = pd.read_csv(
    TRAIN_SOURCE1,
    sep="\t",
    nrows=10000
)

# Read source 2 and 3 IDs + data
source2 = pd.read_csv(
    TRAIN_SOURCE2,
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

source3 = pd.read_csv(
    TRAIN_SOURCE3,
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

ground_truth = pd.read_csv(
    TRAIN_GROUND_TRUTH,
    sep="\t",
    nrows=10000
)

source2 = source2.set_index("entity_id")
source3 = source3.set_index("entity_id")

print("\n" + "=" * 100)
print("MATCH ANALYSIS")
print("=" * 100)

shown = 0

for _, gt in ground_truth.iterrows():

    s1_id = gt["source1_entity_id"]
    matched_ids = str(gt["matched_entity_ids"])

    if s1_id not in set(source1["entity_id"]):
        continue

    s1 = source1[source1["entity_id"] == s1_id].iloc[0]

    print("\n" + "-" * 100)

    print("SOURCE 1")
    print("ID      :", s1["entity_id"])
    print("NAME    :", s1["business_name"])
    print("ADDRESS :", s1["business_address"])
    print("COUNTRY :", s1["country"])

    print("\nTRUE MATCHES:")

    for match_id in matched_ids.split(","):

        match_id = match_id.strip()

        if match_id.startswith("S2-") and match_id in source2.index:
            row = source2.loc[match_id]
            source = "SOURCE 2"

        elif match_id.startswith("S3-") and match_id in source3.index:
            row = source3.loc[match_id]
            source = "SOURCE 3"

        else:
            continue

        print(
            f"\n{source} - {match_id}\n"
            f"NAME    : {row['business_name']}\n"
            f"ADDRESS : {row['business_address']}\n"
            f"COUNTRY : {row['country']}"
        )

    shown += 1

    if shown >= 20:
        break