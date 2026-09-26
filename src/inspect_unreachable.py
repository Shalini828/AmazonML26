from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "v2_train_eval_v2.duckdb"
FEATURES = DATA / "training_features.tsv"

SAMPLE_N = 1000

con = duckdb.connect(str(DB), read_only=True)

seen = pd.read_csv(
    FEATURES,
    sep="\t",
    usecols=["source1_entity_id"],
    dtype=str,
)

con.register(
    "seen_ids",
    seen.rename(columns={"source1_entity_id": "entity_id"}),
)

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE sample_source AS
    SELECT s.*
    FROM s1 s
    ANTI JOIN seen_ids USING(entity_id)
    ORDER BY hash(s.entity_id)
    LIMIT {SAMPLE_N}
""")

con.unregister("seen_ids")

query = """
SELECT
    t.s1_id,
    t.target_id,

    s.n_country AS source1_country,
    x.n_country AS target_country,

    s.n_name AS source1_name,
    x.n_name AS target_name,

    s.n_address AS source1_address,
    x.n_address AS target_address,

    s.k_token AS source1_token_key,
    x.k_token AS target_token_key,

    s.k_c2 AS source1_c2,
    x.k_c2 AS target_c2,

    s.k_c3 AS source1_c3,
    x.k_c3 AS target_c3,

    s.k_c4 AS source1_c4,
    x.k_c4 AS target_c4

FROM truth t

JOIN sample_source s
  ON s.entity_id = t.s1_id

JOIN target x
  ON x.entity_id = t.target_id

WHERE
    NOT (
        s.k_exact <> ''
        AND s.k_exact = x.k_exact
    )

    AND NOT (
        s.k_token <> ''
        AND s.k_token = x.k_token
    )

    AND NOT (
        s.k_addrtoken <> ''
        AND s.k_addrtoken = x.k_addrtoken
    )

    AND NOT (
        s.k_addrprefix <> ''
        AND s.k_addrprefix = x.k_addrprefix
    )

    AND NOT (
        s.k_c2 <> ''
        AND s.k_c2 = x.k_c2
    )

    AND NOT (
        s.k_c3 <> ''
        AND s.k_c3 = x.k_c3
    )

    AND NOT (
        s.k_c4 <> ''
        AND s.k_c4 = x.k_c4
    )

LIMIT 20
"""

df = con.execute(query).fetchdf()

pd.set_option("display.max_colwidth", 100)

print("\n========== 20 UNREACHABLE TRUE PAIRS ==========\n")
print(df.to_string(index=False))

con.close()