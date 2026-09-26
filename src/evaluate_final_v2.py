import polars as pl
from pathlib import Path

GT = Path("data/dataset/train/train_ground_truth.tsv")
PRED = Path("output/matching_results_v2.tsv")

print("=" * 70)
print("FINAL MATCHING EVALUATION V2")
print("=" * 70)

gt = pl.read_csv(
    GT,
    separator="\t",
    infer_schema=False,
    ignore_errors=True
)

pred = pl.read_csv(
    PRED,
    separator="\t",
    infer_schema=False,
    ignore_errors=True
)

print(f"Ground-truth rows: {gt.height:,}")
print(f"Prediction rows:   {pred.height:,}")
print()

required = {"source1_entity_id", "matched_entity_ids"}

if not required.issubset(set(gt.columns)):
    raise RuntimeError(f"GT columns are wrong: {gt.columns}")

if not required.issubset(set(pred.columns)):
    raise RuntimeError(f"Prediction columns are wrong: {pred.columns}")

# Clean IDs
gt = gt.with_columns([
    pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars(),
    pl.col("matched_entity_ids")
      .cast(pl.Utf8)
      .fill_null("")
      .str.strip_chars()
])

pred = pred.with_columns([
    pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars(),
    pl.col("matched_entity_ids")
      .cast(pl.Utf8)
      .fill_null("")
      .str.strip_chars()
])

# In case either file has duplicate source1 rows, keep one deterministic row.
gt = gt.unique(
    subset=["source1_entity_id"],
    keep="first"
)

pred = pred.unique(
    subset=["source1_entity_id"],
    keep="first"
)

print(f"Unique GT source IDs:   {gt.height:,}")
print(f"Unique prediction IDs: {pred.height:,}")
print()

# Join GT to predictions.
joined = gt.join(
    pred.select([
        "source1_entity_id",
        pl.col("matched_entity_ids").alias("predicted_ids")
    ]),
    on="source1_entity_id",
    how="left"
)

joined = joined.with_columns(
    pl.col("predicted_ids").fill_null("")
)

# Normalize comma-separated ID lists.
def normalize_ids(x):
    if x is None:
        return set()
    x = str(x).strip()
    if not x:
        return set()
    return {
        v.strip()
        for v in x.split(",")
        if v.strip()
    }

gt_sets = joined["matched_entity_ids"].to_list()
pred_sets = joined["predicted_ids"].to_list()

total = len(gt_sets)

exact = 0
correct_nonempty = 0
incorrect = 0
gt_matched = 0
pred_matched = 0
unmatched_gt = 0
unmatched_pred = 0

for g, p in zip(gt_sets, pred_sets):
    gs = normalize_ids(g)
    ps = normalize_ids(p)

    if gs:
        gt_matched += 1
    else:
        unmatched_gt += 1

    if ps:
        pred_matched += 1
    else:
        unmatched_pred += 1

    if gs == ps:
        exact += 1

    if gs and ps and gs.intersection(ps):
        correct_nonempty += 1
    elif gs != ps:
        incorrect += 1

precision = correct_nonempty / pred_matched if pred_matched else 0.0
recall = correct_nonempty / gt_matched if gt_matched else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall
    else 0.0
)

print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Evaluated rows:        {total:,}")
print(f"Ground-truth matched:  {gt_matched:,}")
print(f"Ground-truth unmatched:{unmatched_gt:,}")
print(f"Predicted matched:     {pred_matched:,}")
print(f"Predicted unmatched:   {unmatched_pred:,}")
print()
print(f"Exact correct rows:    {exact:,}")
print(f"Correct non-empty:     {correct_nonempty:,}")
print(f"Incorrect:             {incorrect:,}")
print()
print(f"Precision: {precision:.4%}")
print(f"Recall:    {recall:.4%}")
print(f"F1:        {f1:.4%}")
print("=" * 70)

# Show overlap sanity check
gt_ids = set(gt["source1_entity_id"].to_list())
pred_ids = set(pred["source1_entity_id"].to_list())

overlap = len(gt_ids & pred_ids)

print()
print("ID OVERLAP CHECK")
print(f"GT IDs:       {len(gt_ids):,}")
print(f"Prediction:   {len(pred_ids):,}")
print(f"Overlap:      {overlap:,}")
print("=" * 70)
