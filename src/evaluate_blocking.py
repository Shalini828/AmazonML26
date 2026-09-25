# src/evaluate_blocking.py — SMOKE TEST ONLY
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import TRAIN_GROUND_TRUTH
from cache_utils import load_normalized
from blocking import get_blocking_keys, build_index

S1_SAMPLE = 500
S2_SAMPLE_RATE = 0.02
RANDOM_SEED = 42


def load_ground_truth():
    gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", dtype=str).fillna("")
    truth = {}
    for s1, txt in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        txt = txt.strip()
        truth[s1] = {m.strip() for m in txt.split(",") if m.strip()} if txt else set()
    return truth


def main():
    t0 = time.time()
    print("Loading GT...")
    truth = load_ground_truth()
    print(f"  {len(truth):,}  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    print("Loading normalized sources...")
    s1 = load_normalized("s1")
    s2 = load_normalized("s2")
    s3 = load_normalized("s3")
    print(f"  S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    s2_s = s2.sample(frac=S2_SAMPLE_RATE, random_state=RANDOM_SEED)
    s3_s = s3.sample(frac=S2_SAMPLE_RATE, random_state=RANDOM_SEED)
    print(f"  Sampled S2={len(s2_s):,} S3={len(s3_s):,}  [{time.time()-t0:.1f}s]")
    del s2, s3

    sampled_ids = set(s2_s["entity_id"]) | set(s3_s["entity_id"])
    valid = [s1_id for s1_id, m in truth.items() if m and all(x in sampled_ids for x in m)]
    print(f"  Valid S1: {len(valid):,}")
    if S1_SAMPLE < len(valid):
        import random
        random.seed(RANDOM_SEED)
        valid = random.sample(valid, S1_SAMPLE)
    print(f"  Sampled S1 to {len(valid):,}")

    t0 = time.time()
    print("Building index...")
    raw_index = build_index(s2_s.to_dict("records") + s3_s.to_dict("records"))
    index = {k: v for k, v in raw_index.items() if len(v) <= 200}
    print(f"  {len(raw_index):,} raw keys → {len(index):,} after bucket filter")
    print(f"  {len(index):,} keys  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    print("Evaluating...")
    valid_set = set(valid)
    s1_eval = s1[s1["entity_id"].isin(valid_set)]
    total = retained = 0
    missing = []
    cand_counts = []
    for row in s1_eval.to_dict("records"):
        keys = get_blocking_keys(row)
        cand = set()
        for key in sorted(keys, key=lambda item: 0 if "pin" in item[0] else 1):
            cand.update(index.get(key, []))
        if len(cand) > 20:
            cand = set(sorted(cand)[:20])
        cand_counts.append(len(cand))
        for tgt in truth[row["entity_id"]]:
            total += 1
            if tgt in cand: retained += 1
            else: missing.append((row["entity_id"], tgt))
    print(f"  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 55)
    print("SMOKE TEST")
    print("=" * 55)
    print(f"S1 evaluated:       {len(s1_eval):,}")
    print(f"Total true matches: {total:,}")
    print(f"Retained:           {retained:,}")
    recall = retained/total*100 if total else 0
    print(f"BLOCKING RECALL:    {recall:.2f}%")
    if cand_counts:
        print(f"Avg candidates / S1: {sum(cand_counts)/len(cand_counts):.1f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
