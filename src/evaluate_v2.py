"""TRAIN-only evaluation for V2 blocking and the existing RF.

Blocking recall is measured against every TRAIN relationship. RF thresholds are
tested on source1-grouped holdout rows from training_features.tsv, with labels
rechecked against full ground truth. This deliberately does not touch TEST or
the existing submission files.
"""
from pathlib import Path
import json
import time
import re
from difflib import SequenceMatcher

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN = DATA / "dataset" / "train"
DB_PATH = DATA / "v2_train_eval_v2.duckdb"
TEMP_DIR = DATA / "v2_train_temp"
MODEL_PATH = DATA / "rf_model.joblib"
FEATURES_PATH = DATA / "training_features.tsv"
BLOCK_METRICS = DATA / "v2_train_blocking_metrics.tsv"
RF_METRICS = DATA / "v2_train_rf_metrics.tsv"
GENERATED_METRICS = DATA / "v2_train_generated_candidate_metrics.tsv"
SAMPLE_SOURCE1 = 1_000
MAX_PER_BLOCK = 50
FEATURE_NORMALIZER = re.compile(r"[^a-z0-9\s]+")
MEMORY_LIMIT = "2GB"


def qpath(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/").replace("'", "''")


def normalized(expr: str) -> str:
    value = f"lower(coalesce(cast({expr} AS VARCHAR), ''))"
    value = f"replace({value}, '&', ' and ')"
    value = f"regexp_replace({value}, '[^[:alnum:][:space:]]', ' ', 'g')"
    return f"trim(regexp_replace({value}, '[[:space:]]+', ' ', 'g'))"


def sql_source(path: Path) -> str:
    return f"read_parquet('{qpath(path)}')"


def create_train_tables(con: duckdb.DuckDBPyConnection) -> None:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "s1" not in tables:
        print("Preparing normalized TRAIN Source 1...", flush=True)
        con.execute(f"""
            CREATE TABLE s1 AS
            WITH n AS (
                SELECT entity_id, business_name, business_address, country,
                       {normalized('business_name')} AS n_name,
                       {normalized('business_address')} AS n_address,
                       {normalized('country')} AS n_country
                FROM {sql_source(DATA / 'processed' / 'train_source1.parquet')}
            ), t AS (
                SELECT *,
                    regexp_extract(regexp_replace(n_name, '^(the|a|an|and|of|company|co|inc|ltd|llc) +', ''), '^[^ ]+', 0) AS name_token,
                    regexp_extract(n_address, '^[^ ]+', 0) AS address_token,
                    regexp_extract(n_address, '[0-9]{5,6}', 0) AS postal
                FROM n
            )
            SELECT *, n_name AS k_exact,
                n_country || '|' || substr(n_name,1,1) AS k_c1,
                n_country || '|' || substr(n_name,1,2) AS k_c2,
                n_country || '|' || substr(n_name,1,3) AS k_c3,
                n_country || '|' || substr(n_name,1,4) AS k_c4,
                n_country || '|' || name_token AS k_token,
                n_country || '|' || substr(n_address,1,10) AS k_addrprefix,
                n_country || '|' || address_token AS k_addrtoken,
                n_country || '|' || postal AS k_postal
            FROM t
        """)

    if "target" not in tables:
        print("Preparing normalized TRAIN Sources 2+3...", flush=True)
        s2 = DATA / "processed" / "train_source2.parquet"
        s3 = DATA / "processed" / "train_source3.parquet"
        con.execute(f"""
            CREATE TABLE target AS
            WITH raw AS (
                SELECT entity_id, business_name, business_address, country FROM {sql_source(s2)}
                UNION ALL
                SELECT entity_id, business_name, business_address, country FROM {sql_source(s3)}
            ), n AS (
                SELECT entity_id, business_name, business_address, country,
                       {normalized('business_name')} AS n_name,
                       {normalized('business_address')} AS n_address,
                       {normalized('country')} AS n_country
                FROM raw
            ), t AS (
                SELECT *,
                    regexp_extract(regexp_replace(n_name, '^(the|a|an|and|of|company|co|inc|ltd|llc) +', ''), '^[^ ]+', 0) AS name_token,
                    regexp_extract(n_address, '^[^ ]+', 0) AS address_token,
                    regexp_extract(n_address, '[0-9]{5,6}', 0) AS postal
                FROM n
            )
            SELECT *, n_name AS k_exact,
                n_country || '|' || substr(n_name,1,1) AS k_c1,
                n_country || '|' || substr(n_name,1,2) AS k_c2,
                n_country || '|' || substr(n_name,1,3) AS k_c3,
                n_country || '|' || substr(n_name,1,4) AS k_c4,
                n_country || '|' || name_token AS k_token,
                n_country || '|' || substr(n_address,1,10) AS k_addrprefix,
                n_country || '|' || address_token AS k_addrtoken,
                n_country || '|' || postal AS k_postal
            FROM t
        """)

    if "truth" not in tables:
        print("Joining all TRAIN relationships to normalized records...", flush=True)
        con.execute(f"""
            CREATE TABLE truth AS
            SELECT r.source1_entity_id AS s1_id,
                   r.matched_entity_id AS target_id,
                   s.k_exact AS s_exact, t.k_exact AS t_exact,
                   s.k_c1 AS s_c1, t.k_c1 AS t_c1,
                   s.k_c2 AS s_c2, t.k_c2 AS t_c2,
                   s.k_c3 AS s_c3, t.k_c3 AS t_c3,
                   s.k_c4 AS s_c4, t.k_c4 AS t_c4,
                   s.k_token AS s_token, t.k_token AS t_token,
                   s.k_addrprefix AS s_ap, t.k_addrprefix AS t_ap,
                   s.k_addrtoken AS s_at, t.k_addrtoken AS t_at,
                   s.k_postal AS s_postal, t.k_postal AS t_postal
            FROM {sql_source(DATA / 'processed' / 'train_relationships.parquet')} r
            JOIN s1 s ON s.entity_id=r.source1_entity_id
            JOIN target t ON t.entity_id=r.matched_entity_id
        """)


def eval_blocking(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    n = con.execute("SELECT count(*) FROM truth").fetchone()[0]
    exprs = [
        ("exact_name", "s_exact=t_exact AND s_exact<>''"),
        ("country_name_1", "s_c1=t_c1 AND s_c1 NOT LIKE '%|'"),
        ("country_name_2", "s_c2=t_c2 AND s_c2 NOT LIKE '%|'"),
        ("country_name_3", "s_c3=t_c3 AND s_c3 NOT LIKE '%|'"),
        ("country_name_4", "s_c4=t_c4 AND s_c4 NOT LIKE '%|'"),
        ("country_first_meaningful_token", "s_token=t_token AND s_token NOT LIKE '%|'"),
        ("country_address_prefix_10", "s_ap=t_ap AND s_ap NOT LIKE '%|'"),
        ("country_address_first_token", "s_at=t_at AND s_at NOT LIKE '%|'"),
        ("country_postal_5_6_digits", "s_postal=t_postal AND s_postal NOT LIKE '%|'"),
    ]
    where = [f"({x[1]})" for x in exprs]
    sql = "SELECT " + ",".join(f"count(*) FILTER (WHERE {e}) AS \"{name}\"" for name,e in exprs)
    sql += ",count(*) FILTER (WHERE " + " OR ".join(where) + ") AS union_all"
    sql += " FROM truth"
    row = con.execute(sql).fetchone()
    out = []
    for (name, _), captured in zip(exprs, row):
        out.append({"strategy": name, "true_pairs": n, "captured_pairs": captured,
                    "blocking_recall": captured/n if n else 0.0})
    out.append({"strategy": "union_all_strategies", "true_pairs": n,
                "captured_pairs": row[-1], "blocking_recall": row[-1]/n if n else 0.0})
    result = pd.DataFrame(out).sort_values("blocking_recall", ascending=False)
    result.to_csv(BLOCK_METRICS, sep="\t", index=False)
    print("\nTRAIN blocking recall (all ground-truth pairs):", flush=True)
    for r in result.itertuples(index=False):
        print(f"  {r.strategy:36s} {r.blocking_recall:.4%} ({r.captured_pairs:,}/{n:,})", flush=True)
    return result


def source_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                   truth_counts: pd.Series, universe=None) -> dict:
    frame = pd.DataFrame({"sid": groups,
                          "tp": (y_true.astype(np.int8) & y_pred.astype(np.int8)),
                          "pred": y_pred.astype(np.int8)})
    grouped = frame.groupby("sid", sort=False).agg(tp=("tp", "sum"), predicted=("pred", "sum"))
    if universe is not None:
        grouped = grouped.reindex(pd.Index(universe, name="sid"), fill_value=0)
    grouped["actual"] = truth_counts.reindex(grouped.index).fillna(0).astype(int)
    grouped["fn"] = (grouped["actual"] - grouped["tp"]).clip(lower=0)
    denom = 1.25*grouped["tp"] + 0.25*grouped["fn"] + (grouped["predicted"]-grouped["tp"])
    per = np.divide(1.25*grouped["tp"], denom, out=np.zeros(len(denom), dtype=float), where=denom.to_numpy()!=0)
    empty_perfect = (grouped["actual"].to_numpy() == 0) & (grouped["predicted"].to_numpy() == 0)
    per[empty_perfect] = 1.0
    per_zero_div0 = per.copy()
    per_zero_div0[empty_perfect] = 0.0
    return {"macro_f0_5": float(per.mean()) if len(per) else 0.0,
            "macro_f0_5_zero_division_0": float(per_zero_div0.mean()) if len(per_zero_div0) else 0.0,
            "precision": float(grouped["tp"].sum()/grouped["predicted"].sum()) if grouped["predicted"].sum() else 0.0,
            "recall": float(grouped["tp"].sum()/grouped["actual"].sum()) if grouped["actual"].sum() else 0.0,
            "avg_predictions_per_source1": float(grouped["predicted"].mean()) if len(grouped) else 0.0,
            "source1_count": int(len(grouped)), "zero_truth_source1_evaluated": int((grouped["actual"]==0).sum()),
            "zero_truth_with_predictions": int(((grouped["actual"]==0) & (grouped["predicted"]>0)).sum()),
            "zero_truth_no_predictions": int(((grouped["actual"]==0) & (grouped["predicted"]==0)).sum()),
            "zero_pred_source1": int((grouped["predicted"]==0).sum())}


def eval_rf(con: duckdb.DuckDBPyConnection):
    print("\nEvaluating RF on a source1-grouped TRAIN holdout...", flush=True)
    frame = pd.read_csv(FEATURES_PATH, sep="\t")
    features = ["name_similarity", "address_similarity", "name_token_similarity",
                "address_token_similarity", "country_match"]
    hold = (pd.util.hash_pandas_object(frame["source1_entity_id"], index=False).to_numpy(dtype="uint64") % 5) == 0
    train = frame.loc[~hold]
    test = frame.loc[hold].copy()
    print(f"Pairs={len(frame):,}; train={len(train):,}; holdout={len(test):,}; "
          f"holdout Source1={test.source1_entity_id.nunique():,}", flush=True)

    old_model = joblib.load(MODEL_PATH)
    new_model = clone(old_model)
    new_model.fit(train[features], train["label"])

    # Ground truth membership is checked against every TRAIN relation, not the
    # 200k-positive capped sample used to make training_features.tsv.
    con.register("holdout_pairs", test[["source1_entity_id", "candidate_entity_id"]])
    test["score_existing"] = old_model.predict_proba(test[features])[:, 1]
    test["score_group_holdout"] = new_model.predict_proba(test[features])[:, 1]
    con.unregister("holdout_pairs")
    con.register("scored_holdout", test[["source1_entity_id", "candidate_entity_id", "score_existing", "score_group_holdout"]])
    con.execute("CREATE OR REPLACE TEMP TABLE holdout_truth AS SELECT h.source1_entity_id sid, h.candidate_entity_id cid, h.score_existing, h.score_group_holdout, (t.target_id IS NOT NULL)::INT truth FROM scored_holdout h LEFT JOIN truth t ON t.s1_id=h.source1_entity_id AND t.target_id=h.candidate_entity_id")
    truth_rows = con.execute("SELECT sid,cid,truth,score_existing,score_group_holdout FROM holdout_truth").fetchdf()
    truth_counts = con.execute("SELECT s1_id, count(*) AS n FROM truth GROUP BY s1_id").fetchdf().set_index("s1_id")["n"]
    group_ids = truth_rows["sid"].to_numpy()
    y = truth_rows["truth"].to_numpy(dtype=np.int8)

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
    rows = []
    for model_name in ("score_existing", "score_group_holdout"):
        scores = truth_rows[model_name].to_numpy()
        for threshold in thresholds:
            met = source_metrics(y, scores >= threshold, group_ids, truth_counts)
            rows.append({"model": model_name, "strategy": "threshold", "threshold": threshold, "top_k": "", **met})
        order = truth_rows.assign(_score=scores).groupby("sid", sort=False)["_score"].rank(method="first", ascending=False).to_numpy()
        for k in (1, 2, 3, 5, 10):
            met = source_metrics(y, order <= k, group_ids, truth_counts)
            rows.append({"model": model_name, "strategy": "top_k", "threshold": "", "top_k": k, **met})

    results = pd.DataFrame(rows)
    results.to_csv(RF_METRICS, sep="\t", index=False)
    honest = results[results.model == "score_group_holdout"]
    best = honest.sort_values("macro_f0_5", ascending=False).iloc[0]
    print("\nBest grouped-holdout setting (TRAIN only):", flush=True)
    print(best.to_string(), flush=True)
    print("\nAll grouped-holdout threshold/top-k settings:", flush=True)
    print(honest[["strategy", "threshold", "top_k", "macro_f0_5", "precision", "recall", "avg_predictions_per_source1"]].to_string(index=False), flush=True)
    print("\nNote: the pair-training holdout has random negatives rather than blocked hard negatives; "
          "use generated-candidate results below for matcher selection.", flush=True)
    return results, new_model, old_model, frame["source1_entity_id"].drop_duplicates().to_numpy()


def feature_normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", FEATURE_NORMALIZER.sub(" ", str(value).lower())).strip()


def exact_training_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Reproduce src/features.py's feature definitions for a bounded sample."""
    cache = {}

    def norm(v):
        key = "" if v is None or pd.isna(v) else str(v)
        if key not in cache:
            cache[key] = feature_normalize(v)
        return cache[key]

    out = pd.DataFrame(index=pairs.index)
    name_a = [norm(v) for v in pairs.s1_name]
    name_b = [norm(v) for v in pairs.candidate_name]
    addr_a = [norm(v) for v in pairs.s1_address]
    addr_b = [norm(v) for v in pairs.candidate_address]
    out["name_similarity"] = [SequenceMatcher(None, a, b).ratio() for a,b in zip(name_a,name_b)]
    out["address_similarity"] = [SequenceMatcher(None, a, b).ratio() for a,b in zip(addr_a,addr_b)]

    def jaccard(a, b):
        aa, bb = set(a.split()), set(b.split())
        return len(aa & bb) / len(aa | bb) if aa and bb else 0.0

    out["name_token_similarity"] = [jaccard(a,b) for a,b in zip(name_a,name_b)]
    out["address_token_similarity"] = [jaccard(a,b) for a,b in zip(addr_a,addr_b)]
    out["country_match"] = (pairs.s1_country.fillna("").str.lower() == pairs.candidate_country.fillna("").str.lower()).astype("int8")
    return out


def eval_generated_candidates(con: duckdb.DuckDBPyConnection, models, seen_source_ids) -> pd.DataFrame:
    """Generate candidates for unseen TRAIN entities, including zero-match rows."""
    new_model, old_model = models
    print(f"\nSelecting {SAMPLE_SOURCE1:,} TRAIN Source1 entities excluded from all RF training pairs...", flush=True)
    seen = pd.DataFrame({"entity_id": pd.Series(seen_source_ids, dtype="string")})
    con.register("seen_source_ids", seen)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sample_source AS
        SELECT s.* FROM s1 s ANTI JOIN seen_source_ids x USING(entity_id)
        ORDER BY hash(s.entity_id)
        LIMIT {SAMPLE_SOURCE1}
    """)
    con.unregister("seen_source_ids")
    sample_ids = con.execute("SELECT entity_id FROM sample_source ORDER BY entity_id").fetchdf()
    if len(sample_ids) < SAMPLE_SOURCE1:
        raise RuntimeError(f"Only {len(sample_ids)} unseen TRAIN Source1 rows available")
    print(f"Generating V2 blocked candidates for {len(sample_ids):,} unseen TRAIN entities...", flush=True)

    # Pre-aggregate a deterministic top-50 ID sample for every blocking key.
    # This bounds even broad first-letter/address blocks and allows a measured
    # 5/10/20/50 per-block recall curve on the same sample.
    con.execute("DROP TABLE IF EXISTS v2_block_candidates")
    blocks = [
        ("exact_name", "k_exact"), ("country_name_1", "k_c1"),
        ("country_name_2", "k_c2"), ("country_name_3", "k_c3"),
        ("country_name_4", "k_c4"), ("country_first_meaningful_token", "k_token"),
        ("country_address_prefix_10", "k_addrprefix"),
        ("country_address_first_token", "k_addrtoken"),
        ("country_postal_5_6_digits", "k_postal"),
    ]
    con.execute("DROP TABLE IF EXISTS v2_block_candidates")
    con.execute("CREATE TABLE v2_block_candidates(strategy VARCHAR, block_key VARCHAR, ids VARCHAR[])")
    for label, key in blocks[1:]:
        print(f"  Summarizing {label}...", flush=True)
        con.execute(f"""
            INSERT INTO v2_block_candidates
            SELECT '{label}', {key}, max_by(entity_id, entity_id, {MAX_PER_BLOCK})
            FROM target WHERE {key}<>'' AND {key} NOT LIKE '%|'
            GROUP BY {key}
        """)

    key_rows = []
    for label, key in blocks[1:]:
        key_rows.append(f"SELECT entity_id s1_id, '{label}' strategy, {key} block_key FROM sample_source WHERE {key}<>'' AND {key} NOT LIKE '%|'")
    con.execute("CREATE OR REPLACE TEMP TABLE sample_block_lists AS " +
                " SELECT s.entity_id s1_id,'exact_name' strategy,s.k_exact block_key,list(t.entity_id ORDER BY t.entity_id) ids "
                "FROM sample_source s JOIN target t ON s.k_exact=t.k_exact AND s.k_exact<>'' "
                "GROUP BY s.entity_id,s.k_exact UNION ALL " +
                "SELECT k.s1_id,k.strategy,k.block_key,b.ids FROM (" + " UNION ALL ".join(key_rows) +
                ") k JOIN v2_block_candidates b USING(strategy,block_key)")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE sample_truth AS
        SELECT t.* FROM truth t JOIN sample_source s ON s.entity_id=t.s1_id
    """)
    recall_rows = []
    for cap in (5, 10, 20, 50):
        hit_exprs = []
        for label, keycol in [
            ("exact_name", "s_exact"), ("country_name_1", "s_c1"),
            ("country_name_2", "s_c2"), ("country_name_3", "s_c3"),
            ("country_name_4", "s_c4"), ("country_first_meaningful_token", "s_token"),
            ("country_address_prefix_10", "s_ap"), ("country_address_first_token", "s_at"),
            ("country_postal_5_6_digits", "s_postal"),
        ]:
            hit_exprs.append(f"EXISTS (SELECT 1 FROM sample_block_lists b WHERE b.s1_id=t.s1_id AND b.strategy='{label}' AND b.block_key=t.{keycol} AND list_position(list_sort(b.ids),t.target_id)>len(b.ids)-{cap})")
        captured = con.execute("SELECT count(*) FROM sample_truth t WHERE " + " OR ".join(hit_exprs)).fetchone()[0]
        total = con.execute("SELECT count(*) FROM sample_truth").fetchone()[0]
        candidate_stats = con.execute(f"""
            WITH c AS (
                SELECT DISTINCT b.s1_id,u.candidate_id
                FROM sample_block_lists b
                CROSS JOIN UNNEST(list_slice(list_sort(b.ids), -{cap}, -1)) u(candidate_id)
            ), per_source AS (SELECT s1_id,count(*) n FROM c GROUP BY s1_id)
            SELECT (SELECT count(*) FROM c),
                   coalesce((SELECT avg(n) FROM per_source),0),
                   coalesce((SELECT quantile_cont(n,0.95) FROM per_source),0)
        """).fetchone()
        recall_rows.append({"per_block_cap": cap, "source1_sample": len(sample_ids),
                            "true_pairs": total, "captured_pairs": captured,
                            "candidate_recall": captured/total if total else 0.0,
                            "unique_candidate_pairs": candidate_stats[0],
                            "avg_candidates_per_source1": float(candidate_stats[1]),
                            "p95_candidates_per_source1": float(candidate_stats[2])})
    recall_curve = pd.DataFrame(recall_rows)
    print("Bounded candidate-generation recall curve:", flush=True)
    print(recall_curve.to_string(index=False), flush=True)

    metrics = []
    for cap in (20, 50):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE v2_sample_candidates AS
            SELECT DISTINCT b.s1_id, u.candidate_id
            FROM sample_block_lists b
            CROSS JOIN UNNEST(list_slice(list_sort(b.ids), -{cap}, -1)) u(candidate_id)
        """)
        count = con.execute("SELECT count(*) FROM v2_sample_candidates").fetchone()[0]
        print(f"Scoring {count:,} unique candidates at {cap}/block...", flush=True)
        rows = con.execute("""
        SELECT c.s1_id, c.candidate_id,
               s.business_name AS s1_name, s.business_address AS s1_address, s.country AS s1_country,
               t.business_name AS candidate_name, t.business_address AS candidate_address,
               t.country AS candidate_country,
               (tr.target_id IS NOT NULL)::INT AS truth
        FROM v2_sample_candidates c
        JOIN sample_source s ON s.entity_id=c.s1_id
        JOIN target t ON t.entity_id=c.candidate_id
        LEFT JOIN truth tr ON tr.s1_id=c.s1_id AND tr.target_id=c.candidate_id
        """).fetchdf()
        rows.sort_values(["s1_id", "candidate_id"], inplace=True, ignore_index=True)
        truth_counts = con.execute("SELECT s.entity_id AS s1_id,count(t.target_id) n FROM sample_source s LEFT JOIN truth t ON t.s1_id=s.entity_id GROUP BY s.entity_id").fetchdf().set_index("s1_id")["n"]
        all_sample_ids = sample_ids.entity_id.to_numpy()
        X = exact_training_features(rows)
        for feature in X.columns:
            rows[feature] = X[feature].to_numpy()
        rows["score_group_holdout"] = new_model.predict_proba(X)[:, 1]
        rows["score_existing"] = old_model.predict_proba(X)[:, 1]
        group_ids = rows["s1_id"].to_numpy()
        y = rows["truth"].to_numpy(dtype=np.int8)
        for model_name in ("score_group_holdout", "score_existing"):
            scores = rows[model_name].to_numpy()
            for threshold in [round(x,2) for x in np.arange(0.05,0.951,0.05)] + [0.98,0.99,0.995,0.999]:
                metric = source_metrics(y, scores>=threshold, group_ids, truth_counts, all_sample_ids)
                metrics.append({"record_type":"matching_metric","model":model_name,"strategy":"threshold","threshold":threshold,"top_k":"","per_block_cap":cap,**metric})
            ranked = rows.assign(_score=scores).sort_values(
                ["s1_id", "_score", "name_similarity", "address_similarity",
                 "name_token_similarity", "address_token_similarity", "candidate_id"],
                ascending=[True, False, False, False, False, False, True],
                kind="mergesort",
            )
            ranked["_rank"] = ranked.groupby("s1_id", sort=False).cumcount() + 1
            order = ranked["_rank"].reindex(rows.index).to_numpy()
            for k in (1,2,3,5,10):
                metric=source_metrics(y,order<=k,group_ids,truth_counts,all_sample_ids)
                metrics.append({"record_type":"matching_metric","model":model_name,"strategy":"top_k","threshold":"","top_k":k,"per_block_cap":cap,**metric})
            for threshold in (0.50,0.75,0.85,0.90,0.95,0.98,0.99,0.995,0.999):
                for k in (1,2,3,5):
                    metric=source_metrics(y,(scores>=threshold)&(order<=k),group_ids,truth_counts,all_sample_ids)
                    metrics.append({"record_type":"matching_metric","model":model_name,"strategy":"threshold_and_top_k","threshold":threshold,"top_k":k,"per_block_cap":cap,**metric})
    result=pd.DataFrame(metrics)
    recall_curve.insert(0,"record_type","blocking_recall")
    combined=pd.concat([recall_curve,result],ignore_index=True,sort=False)
    combined.to_csv(GENERATED_METRICS,sep="\t",index=False)
    honest=result[result.model=="score_group_holdout"]
    best=honest.sort_values("macro_f0_5",ascending=False).iloc[0]
    print("\nGenerated-candidate TRAIN holdout best setting:",flush=True)
    print(best.to_string(),flush=True)
    print("Generated-candidate threshold/top-k results:",flush=True)
    print(honest[["strategy","threshold","top_k","macro_f0_5","precision","recall","avg_predictions_per_source1","zero_pred_source1"]].to_string(index=False),flush=True)
    return result, recall_curve, best.to_dict()


def main() -> None:
    started = time.time()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{qpath(TEMP_DIR)}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")
    create_train_tables(con)
    blocking = eval_blocking(con)
    rf_results, new_model, old_model, seen_source_ids = eval_rf(con)
    generated, cap_curve, best = eval_generated_candidates(con, (new_model, old_model), seen_source_ids)
    report = {
        "train_relationship_pairs": int(con.execute("SELECT count(*) FROM truth").fetchone()[0]),
        "train_source1": int(con.execute("SELECT count(*) FROM s1").fetchone()[0]),
        "train_zero_match_source1": int(con.execute("SELECT count(*) FROM (SELECT s.entity_id FROM s1 s LEFT JOIN truth t ON t.s1_id=s.entity_id GROUP BY s.entity_id HAVING count(t.target_id)=0)").fetchone()[0]),
        "blocking_union_recall": float(blocking.loc[blocking.strategy == "union_all_strategies", "blocking_recall"].iloc[0]),
        "best_generated_candidate_holdout": {k: (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else str(v))) for k,v in best.items()},
        "elapsed_seconds": round(time.time()-started, 1),
    }
    report_path = DATA / "v2_train_evaluation_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSummary: {report_path}", flush=True)
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s", flush=True)
    con.close()


if __name__ == "__main__":
    main()
