# src/generate_outputs_fast.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import OUTPUT_DIR, TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
from normalization import normalize_dataframe
from blocking import get_blocking_keys, build_index

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
TEST_SOURCES = {"t1": TEST_SOURCE1, "t2": TEST_SOURCE2, "t3": TEST_SOURCE3}

MAX_PER_S1 = 20
MAX_KEY_BUCKET = 200


def load_test_normalized(name):
    f = CACHE_DIR / f"test_{name}_norm.parquet"
    if f.exists():
        return pd.read_parquet(f)
    print(f"[cache miss] building {f.name}")
    df = pd.read_csv(TEST_SOURCES[name], sep="\t")
    df = normalize_dataframe(df)
    df.to_parquet(f, index=False)
    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    print("Loading cache...")
    s1 = load_test_normalized("t1")
    s2 = load_test_normalized("t2")
    s3 = load_test_normalized("t3")
    print(f"  S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}  [{time.time()-t0:.1f}s]")

    # --- Build index ---
    t0 = time.time()
    print("Building index...")
    idx = {}
    for df in (s2, s3):
        for row in df.itertuples(index=False):
            rec = {
                "entity_id": row.entity_id,
                "country_normalized": row.country_normalized,
                "business_name_normalized": row.business_name_normalized,
                "business_address_normalized": row.business_address_normalized,
                "pincode": row.pincode if pd.notna(row.pincode) else "",
            }
            for k in get_blocking_keys(rec):
                idx.setdefault(k, []).append(rec["entity_id"])
    raw_keys = len(idx)
    idx = {k: v for k, v in idx.items() if len(v) <= MAX_KEY_BUCKET}
    print(f"  {raw_keys:,} raw keys -> {len(idx):,} kept  [{time.time()-t0:.1f}s]")

    # --- Generate candidates ---
    t0 = time.time()
    print("Generating candidates for S1...")
    out_rows = []
    stats = {"total": 0, "empty": 0, "sum_c": 0}
    progress_step = 100_000

    for i, row in enumerate(s1.itertuples(index=False)):
        rec = {
            "country_normalized": row.country_normalized,
            "business_name_normalized": row.business_name_normalized,
            "business_address_normalized": row.business_address_normalized,
            "pincode": row.pincode if pd.notna(row.pincode) else "",
        }
        cand = set()
        keys = get_blocking_keys(rec)
        for k in sorted(keys, key=lambda x: 0 if "pin" in x[0] else 1):
            cand.update(idx.get(k, []))
        if len(cand) > MAX_PER_S1:
            cand = set(sorted(cand)[:MAX_PER_S1])

        out_rows.append((row.entity_id, ",".join(sorted(cand))))
        stats["total"] += 1
        stats["sum_c"] += len(cand)
        if not cand:
            stats["empty"] += 1

        if (i + 1) % progress_step == 0:
            print(f"    {i+1:,} / {len(s1):,}  [{time.time()-t0:.1f}s]")

    print(f"  Done generating  [{time.time()-t0:.1f}s]")

    # --- Write files ---
    t0 = time.time()
    print("Writing output files...")
    with open(OUTPUT_DIR / "candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, ids in out_rows:
            f.write(f"{s1_id}\t{ids}\n")

    # Placeholder matching_results = same as candidates
    with open(OUTPUT_DIR / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, ids in out_rows:
            f.write(f"{s1_id}\t{ids}\n")
    print(f"  Written  [{time.time()-t0:.1f}s]")

    print("\nStats:")
    print(f"  Total S1: {stats['total']:,}")
    print(f"  Avg candidates: {stats['sum_c']/max(stats['total'],1):.1f}")
    print(f"  S1 with 0 candidates: {stats['empty']:,}")


if __name__ == "__main__":
    main()