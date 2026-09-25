# src/cache_utils.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SOURCES = {"s1": TRAIN_SOURCE1, "s2": TRAIN_SOURCE2, "s3": TRAIN_SOURCE3}


def load_normalized(name, use_cache=True):
    cache_file = CACHE_DIR / f"train_{name}_norm.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    print(f"[cache miss] building {cache_file.name} ...")
    from normalization import normalize_dataframe
    df = pd.read_csv(SOURCES[name], sep="\t")
    df = normalize_dataframe(df)
    df.to_parquet(cache_file, index=False)
    return df