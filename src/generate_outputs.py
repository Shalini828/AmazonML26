# src/generate_outputs.py
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import OUTPUT_DIR
from cache_utils import load_normalized
from blocking import generate_candidates, get_blocking_keys

# For the test set — we need a parallel loader
from config import TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
from normalization import normalize_dataframe

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

TEST_SOURCES = {"t1": TEST_SOURCE1, "t2": TEST_SOURCE2, "t3": TEST_SOURCE3}


def load_test_normalized(name, use_cache=True):
    cache_file = CACHE_DIR / f"test_{name}_norm.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    print(f"[cache miss] building {cache_file.name} ...")
    df = pd.read_csv(TEST_SOURCES[name], sep="\t")
    df = normalize_dataframe(df)
    df.to_parquet(cache_file, index=False)
    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    print("Loading test S1/S2/S3 (cached after first run)...")
    s1 = load_test_normalized("t1")
    s2 = load_test_normalized("t2")
    s3 = load_test_normalized("t3")
    print(f"  S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    print("Generating candidates...")
    cands = generate_candidates(
        s1.to_dict("records"),
        s2.to_dict("records"),
        s3.to_dict("records"),
        max_per_s1=20,
        max_key_bucket=200,
    )
    print(f"  [{time.time()-t0:.1f}s]")

    # --- candidate_pairs.tsv ---
    t0 = time.time()
    print("Writing candidate_pairs.tsv...")
    with open(OUTPUT_DIR / "candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for _, r in s1.iterrows():
            ids = ",".join(cands.get(r["entity_id"], []))
            f.write(f"{r['entity_id']}\t{ids}\n")
    print(f"  [{time.time()-t0:.1f}s]")

    # --- matching_results.tsv — PLACEHOLDER until matcher is ready ---
    # Just copies candidates. Matcher teammate will overwrite this.
    with open(OUTPUT_DIR / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for _, r in s1.iterrows():
            ids = ",".join(cands.get(r["entity_id"], []))
            f.write(f"{r['entity_id']}\t{ids}\n")

    total_cands = sum(len(v) for v in cands.values())
    avg_c = total_cands / max(len(cands), 1)
    zero_c = sum(1 for v in cands.values() if not v)
    print(f"\nStats:")
    print(f"  Total S1: {len(cands):,}")
    print(f"  Avg candidates / S1: {avg_c:.1f}")
    print(f"  S1 with 0 candidates: {zero_c:,}")
    print(f"  Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()