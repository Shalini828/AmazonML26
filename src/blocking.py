# src/blocking.py
from collections import defaultdict
from normalization import normalize_dataframe


def _name_tokens(name):
    return [t for t in name.split() if len(t) >= 2]


def _address_tokens(addr):
    return [t for t in addr.split() if len(t) >= 3 and not t.isdigit()]


def get_blocking_keys(record):
    keys = set()
    c = record.get("country_normalized") or "xx"
    name = record.get("business_name_normalized", "")
    addr = record.get("business_address_normalized", "")
    pin = record.get("pincode", "") or ""

    ntoks = _name_tokens(name)
    atoks = _address_tokens(addr)

    if pin:
        keys.add(("pin", c, pin))
        if ntoks:
            keys.add(("pin_name1", c, pin, ntoks[0]))
        if len(ntoks) >= 2:
            keys.add(("pin_name12", c, pin, ntoks[0], ntoks[1]))
        if atoks:
            keys.add(("pin_addr1", c, pin, atoks[0]))

    if len(ntoks) >= 2:
        keys.add(("nameset", c, " ".join(sorted(set(ntoks[:4])))))
        keys.add(("name12", c, ntoks[0], ntoks[1]))
    if ntoks:
        keys.add(("name1", c, ntoks[0]))

    if len(atoks) >= 2:
        keys.add(("addr12", c, atoks[0], atoks[1]))

    return keys


def build_index(records):
    idx = defaultdict(list)
    for rec in records:
        for k in get_blocking_keys(rec):
            idx[k].append(rec["entity_id"])
    return idx


def generate_candidates(s1_records, s2_records, s3_records,
                        max_per_s1=20,
                        max_key_bucket=200):
    idx = build_index(s2_records + s3_records)
    filtered = {k: v for k, v in idx.items() if len(v) <= max_key_bucket}

    out = {}
    for rec in s1_records:
        cand = set()
        keys = get_blocking_keys(rec)
        for k in sorted(keys, key=lambda x: 0 if "pin" in x[0] else 1):
            cand.update(filtered.get(k, []))
        if len(cand) > max_per_s1:
            cand = set(sorted(cand)[:max_per_s1])
        out[rec["entity_id"]] = sorted(cand)
    return out


def records_from_df(df):
    return normalize_dataframe(df).to_dict("records")


# ---- Backwards-compat shim for teammates using old API ----
def create_blocking_key(df):
    """Legacy shim — returns df with a basic blocking_key column."""
    df = normalize_dataframe(df)
    df["blocking_key_1"] = df["country_normalized"] + "_" + df["business_name_normalized"].str[:1]
    return df