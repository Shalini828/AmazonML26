"""TRAIN-only memory-safe candidate selection experiment for V2.

Processes each blocking strategy independently so the full block UNION is
never materialized at once.
"""

from pathlib import Path
import time

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "v2_train_eval_v2.duckdb"
TEMP = DATA / "v2_train_temp"
FEATURES = DATA / "training_features.tsv"
OUT = DATA / "v2_train_candidate_selection_metrics.tsv"

SAMPLE_N = 1_000

# High-precision blocks first, then broader name-prefix blocks.
BLOCKS = [
    ("exact_name", "k_exact", 0),
    ("country_first_meaningful_token", "k_token", 200),
    ("country_postal_5_6_digits", "k_postal", 20),
    ("country_address_first_token", "k_addrtoken", 100),
    ("country_address_prefix_10", "k_addrprefix", 20),
    ("country_name_2", "k_c2", 20),
    ("country_name_3", "k_c3", 20),
    ("country_name_4", "k_c4", 20),
]


def main():
    con = duckdb.connect(str(DB))

    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory='{TEMP.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")

    # ------------------------------------------------------------
    # Fixed TRAIN-only holdout
    # ------------------------------------------------------------

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

    sample_count = con.execute(
        "SELECT count(*) FROM sample_source"
    ).fetchone()[0]

    if sample_count != SAMPLE_N:
        raise RuntimeError(
            f"Expected {SAMPLE_N} sample rows, got {sample_count}"
        )

    sample_truth = con.execute("""
        SELECT count(*)
        FROM truth t
        JOIN sample_source s
          ON s.entity_id=t.s1_id
    """).fetchone()[0]

    print(
        f"Fixed TRAIN sample: {sample_count:,} Source1; "
        f"{sample_truth:,} true pairs",
        flush=True,
    )

    # ------------------------------------------------------------
    # Process ONE blocking strategy at a time
    # ------------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE TEMP TABLE selected_candidates (
            s1_id VARCHAR,
            candidate_id VARCHAR
        )
    """)

    strategy_stats = []

    for strategy, col, cap in BLOCKS:
        started = time.perf_counter()

        print(
            f"\nProcessing block: {strategy} ({col})",
            flush=True,
        )

        # Exact-name block:
        # keep every candidate because this is a high precision signal.
        if strategy == "exact_name":
            con.execute(f"""
                INSERT INTO selected_candidates
                SELECT DISTINCT
                    s.entity_id,
                    t.entity_id
                FROM sample_source s
                JOIN target t
                  ON s.{col}=t.{col}
                WHERE s.{col}<>''
                  AND s.{col} NOT LIKE '%|'
            """)

        else:
            # For broader blocks, rank candidates INSIDE THIS BLOCK ONLY.
            #
            # No giant UNION.
            # No SequenceMatcher.
            # Only cheap SQL signals.
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE block_candidates AS
                WITH raw AS (
                    SELECT
                        s.entity_id AS s1_id,
                        t.entity_id AS candidate_id,

                        (
                            s.n_name <> ''
                            AND s.n_name = t.n_name
                        )::INT AS exact_name,

                        (
                            length(s.n_name) >= 1
                            AND substr(s.n_name,1,1)
                                = substr(t.n_name,1,1)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 2
                            AND substr(s.n_name,1,2)
                                = substr(t.n_name,1,2)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 3
                            AND substr(s.n_name,1,3)
                                = substr(t.n_name,1,3)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 4
                            AND substr(s.n_name,1,4)
                                = substr(t.n_name,1,4)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 8
                            AND substr(s.n_name,1,8)
                                = substr(t.n_name,1,8)
                        )::INT
                        AS name_prefix_agreement,

                        (
                            s.n_country <> ''
                            AND s.n_country = t.n_country
                        )::INT AS country_equal,

                        (
                            s.n_address <> ''
                            AND s.n_address = t.n_address
                        )::INT AS address_exact,

                        (
                            s.n_address <> ''
                            AND t.n_address <> ''
                            AND (
                                starts_with(s.n_address,t.n_address)
                                OR starts_with(t.n_address,s.n_address)
                                OR contains(s.n_address,t.n_address)
                                OR contains(t.n_address,s.n_address)
                            )
                        )::INT AS address_partial,

                        CASE
                            WHEN length(s.n_name)+length(t.n_name)>0
                            THEN
                                length(
                                    list_intersect(
                                        list_distinct(
                                            string_split(s.n_name,' ')
                                        ),
                                        list_distinct(
                                            string_split(t.n_name,' ')
                                        )
                                    )
                                )::DOUBLE
                                /
                                greatest(
                                    1,
                                    length(
                                        list_distinct(
                                            string_split(s.n_name,' ')
                                        )
                                    )
                                    +
                                    length(
                                        list_distinct(
                                            string_split(t.n_name,' ')
                                        )
                                    )
                                    -
                                    length(
                                        list_intersect(
                                            list_distinct(
                                                string_split(s.n_name,' ')
                                            ),
                                            list_distinct(
                                                string_split(t.n_name,' ')
                                            )
                                        )
                                    )
                                )
                            ELSE 0
                        END AS name_token_overlap,

                        CASE
                            WHEN length(s.n_address)+length(t.n_address)>0
                            THEN
                                length(
                                    list_intersect(
                                        list_distinct(
                                            string_split(s.n_address,' ')
                                        ),
                                        list_distinct(
                                            string_split(t.n_address,' ')
                                        )
                                    )
                                )::DOUBLE
                                /
                                greatest(
                                    1,
                                    length(
                                        list_distinct(
                                            string_split(s.n_address,' ')
                                        )
                                    )
                                    +
                                    length(
                                        list_distinct(
                                            string_split(t.n_address,' ')
                                        )
                                    )
                                    -
                                    length(
                                        list_intersect(
                                            list_distinct(
                                                string_split(s.n_address,' ')
                                            ),
                                            list_distinct(
                                                string_split(t.n_address,' ')
                                            )
                                        )
                                    )
                                )
                            ELSE 0
                        END AS address_token_overlap

                    FROM sample_source s
                    JOIN target t
                      ON s.{col}=t.{col}
                    WHERE s.{col}<>''
                      AND s.{col} NOT LIKE '%|'
                ),

                ranked AS (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY s1_id
                            ORDER BY
                                exact_name DESC,
                                address_exact DESC,
                                address_partial DESC,
                                name_prefix_agreement DESC,
                                name_token_overlap DESC,
                                address_token_overlap DESC,
                                country_equal DESC,
                                candidate_id
                        ) AS rn
                    FROM raw
                )

                SELECT
                    s1_id,
                    candidate_id
                FROM ranked
                WHERE rn <= {cap}
            """)

            con.execute("""
                INSERT INTO selected_candidates
                SELECT DISTINCT
                    s1_id,
                    candidate_id
                FROM block_candidates
            """)

            con.execute("DROP TABLE block_candidates")

        elapsed = time.perf_counter() - started

        block_count = con.execute("""
            SELECT count(*)
            FROM selected_candidates
        """).fetchone()[0]

        print(
            f"  cumulative candidates: {block_count:,}"
            f" | {elapsed:.1f}s",
            flush=True,
        )

        strategy_stats.append(
            {
                "strategy": strategy,
                "cap": cap,
                "elapsed_seconds": round(elapsed, 3),
                "cumulative_candidates": int(block_count),
            }
        )

    # ------------------------------------------------------------
    # Evaluate final union of already-capped candidates
    # ------------------------------------------------------------

    print("\nEvaluating final candidate union...", flush=True)

    unique_count = con.execute("""
        SELECT count(*)
        FROM (
            SELECT DISTINCT s1_id,candidate_id
            FROM selected_candidates
        )
    """).fetchone()[0]

    retained = con.execute("""
        SELECT count(*)
        FROM truth t
        JOIN sample_source s
          ON s.entity_id=t.s1_id
        JOIN (
            SELECT DISTINCT s1_id, candidate_id
            FROM selected_candidates
        ) c
          ON c.s1_id=t.s1_id
         AND c.candidate_id=t.target_id
    """).fetchone()[0]

    counts = con.execute("""
        WITH per AS (
            SELECT
                s.entity_id AS s1_id,
                count(c.candidate_id) AS n
            FROM sample_source s
            LEFT JOIN selected_candidates c
              ON c.s1_id=s.entity_id
            GROUP BY s.entity_id
        )
        SELECT
            avg(n),
            quantile_cont(n,0.95),
            max(n)
        FROM per
    """).fetchone()

    recall = retained / sample_truth if sample_truth else 0.0

        # ------------------------------------------------------------
    # Per-block diagnostic: how many true pairs does each block
    # retain independently?
    # ------------------------------------------------------------

    print("\n========== PER-BLOCK RECALL ==========", flush=True)

    for strategy, col, cap in BLOCKS:
        if strategy == "exact_name":
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE diagnostic_candidates AS
                SELECT DISTINCT
                    s.entity_id AS s1_id,
                    t.entity_id AS candidate_id
                FROM sample_source s
                JOIN target t
                  ON s.{col}=t.{col}
                WHERE s.{col}<>''
                  AND s.{col} NOT LIKE '%|'
            """)
        else:
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE diagnostic_candidates AS
                WITH raw AS (
                    SELECT
                        s.entity_id AS s1_id,
                        t.entity_id AS candidate_id,

                        (
                            s.n_name <> ''
                            AND s.n_name = t.n_name
                        )::INT AS exact_name,

                        (
                            length(s.n_name) >= 1
                            AND substr(s.n_name,1,1)
                                = substr(t.n_name,1,1)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 2
                            AND substr(s.n_name,1,2)
                                = substr(t.n_name,1,2)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 3
                            AND substr(s.n_name,1,3)
                                = substr(t.n_name,1,3)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 4
                            AND substr(s.n_name,1,4)
                                = substr(t.n_name,1,4)
                        )::INT
                        +
                        (
                            length(s.n_name) >= 8
                            AND substr(s.n_name,1,8)
                                = substr(t.n_name,1,8)
                        )::INT
                        AS name_prefix_agreement,

                        (
                            s.n_country <> ''
                            AND s.n_country = t.n_country
                        )::INT AS country_equal,

                        (
                            s.n_address <> ''
                            AND s.n_address = t.n_address
                        )::INT AS address_exact,

                        (
                            s.n_address <> ''
                            AND t.n_address <> ''
                            AND (
                                starts_with(s.n_address,t.n_address)
                                OR starts_with(t.n_address,s.n_address)
                                OR contains(s.n_address,t.n_address)
                                OR contains(t.n_address,s.n_address)
                            )
                        )::INT AS address_partial,

                        CASE
                            WHEN length(s.n_name)+length(t.n_name)>0
                            THEN
                                length(
                                    list_intersect(
                                        list_distinct(string_split(s.n_name,' ')),
                                        list_distinct(string_split(t.n_name,' '))
                                    )
                                )::DOUBLE
                                /
                                greatest(
                                    1,
                                    length(list_distinct(string_split(s.n_name,' ')))
                                    +
                                    length(list_distinct(string_split(t.n_name,' ')))
                                    -
                                    length(
                                        list_intersect(
                                            list_distinct(string_split(s.n_name,' ')),
                                            list_distinct(string_split(t.n_name,' '))
                                        )
                                    )
                                )
                            ELSE 0
                        END AS name_token_overlap,

                        CASE
                            WHEN length(s.n_address)+length(t.n_address)>0
                            THEN
                                length(
                                    list_intersect(
                                        list_distinct(string_split(s.n_address,' ')),
                                        list_distinct(string_split(t.n_address,' '))
                                    )
                                )::DOUBLE
                                /
                                greatest(
                                    1,
                                    length(list_distinct(string_split(s.n_address,' ')))
                                    +
                                    length(list_distinct(string_split(t.n_address,' ')))
                                    -
                                    length(
                                        list_intersect(
                                            list_distinct(string_split(s.n_address,' ')),
                                            list_distinct(string_split(t.n_address,' '))
                                        )
                                    )
                                )
                            ELSE 0
                        END AS address_token_overlap
                    FROM sample_source s
                    JOIN target t
                      ON s.{col}=t.{col}
                    WHERE s.{col}<>''
                      AND s.{col} NOT LIKE '%|'
                ),
                ranked AS (
                    SELECT *,
                        row_number() OVER (
                            PARTITION BY s1_id
                            ORDER BY
                                exact_name DESC,
                                address_exact DESC,
                                address_partial DESC,
                                name_prefix_agreement DESC,
                                name_token_overlap DESC,
                                address_token_overlap DESC,
                                country_equal DESC,
                                candidate_id
                        ) AS rn
                    FROM raw
                )
                SELECT
                    s1_id,
                    candidate_id
                FROM ranked
                WHERE rn <= {cap}
            """)

        block_true = con.execute("""
            SELECT count(*)
            FROM truth t
            JOIN diagnostic_candidates c
              ON c.s1_id=t.s1_id
             AND c.candidate_id=t.target_id
            JOIN sample_source s
              ON s.entity_id=t.s1_id
        """).fetchone()[0]

        block_recall = (
            block_true / sample_truth
            if sample_truth else 0.0
        )

        print(
            f"{strategy:35s} "
            f"{block_true:5,}/{sample_truth:,} "
            f"({block_recall:.2%})",
            flush=True,
        )
    
    print("\n========== RESULT ==========", flush=True)
    print(
        f"True pairs retained : {retained:,}/{sample_truth:,}",
        flush=True,
    )
    print(
        f"Candidate recall    : {recall:.4%}",
        flush=True,
    )
    print(
        f"Unique candidates   : {unique_count:,}",
        flush=True,
    )
    print(
        f"Avg candidates/S1   : {counts[0]:.1f}",
        flush=True,
    )
    print(
        f"P95 candidates/S1  : {counts[1]:.0f}",
        flush=True,
    )
    print(
        f"Max candidates/S1  : {counts[2]:.0f}",
        flush=True,
    )

    result = pd.DataFrame(
        [
            {
                "setting": "memory_safe_capped_union",
                "source1_sample": sample_count,
                "true_pairs": sample_truth,
                "true_pairs_retained": retained,
                "candidate_recall": recall,
                "unique_candidate_pairs": unique_count,
                "avg_candidates_per_source1": float(counts[0]),
                "p95_candidates_per_source1": float(counts[1]),
                "max_candidates_per_source1": int(counts[2]),
            }
        ]
    )

    result.to_csv(
        OUT,
        sep="\t",
        index=False,
    )

    print(f"\nWrote {OUT}", flush=True)

        # ------------------------------------------------------------
    # Blocking ceiling diagnostic
    #
    # For every true pair, check whether Source1 and the true target
    # share ANY of our blocking keys.
    #
    # This ignores per-block caps. Therefore it tells us whether
    # missing pairs are lost because of ranking/caps or because our
    # blocking keys cannot reach them at all.
    # ------------------------------------------------------------

    print("\n========== BLOCKING CEILING ==========", flush=True)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE true_pair_reachability AS
        SELECT
            t.s1_id,
            t.target_id,

            (
                s.k_exact <> ''
                AND s.k_exact = x.k_exact
            )::INT AS exact_hit,

            (
                s.k_token <> ''
                AND s.k_token = x.k_token
            )::INT AS token_hit,

            (
                s.k_addrtoken <> ''
                AND s.k_addrtoken = x.k_addrtoken
            )::INT AS addrtoken_hit,

            (
                s.k_addrprefix <> ''
                AND s.k_addrprefix = x.k_addrprefix
            )::INT AS addrprefix_hit,

            (
                s.k_c2 <> ''
                AND s.k_c2 = x.k_c2
            )::INT AS c2_hit,

            (
                s.k_c3 <> ''
                AND s.k_c3 = x.k_c3
            )::INT AS c3_hit,

            (
                s.k_c4 <> ''
                AND s.k_c4 = x.k_c4
            )::INT AS c4_hit

        FROM truth t
        JOIN sample_source s
          ON s.entity_id=t.s1_id
        JOIN target x
          ON x.entity_id=t.target_id
    """)

    # True pairs reachable by at least one existing key but absent from the
    # final, per-block-capped candidate union.
    print("\n========== MISSING REACHABLE TRUE PAIRS ==========", flush=True)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE missing_reachable_true_pairs AS
        SELECT
            r.*,
            concat_ws(',',
                CASE WHEN exact_hit=1 THEN 'k_exact' END,
                CASE WHEN token_hit=1 THEN 'k_token' END,
                CASE WHEN addrtoken_hit=1 THEN 'k_addrtoken' END,
                CASE WHEN addrprefix_hit=1 THEN 'k_addrprefix' END,
                CASE WHEN c2_hit=1 THEN 'k_c2' END,
                CASE WHEN c3_hit=1 THEN 'k_c3' END,
                CASE WHEN c4_hit=1 THEN 'k_c4' END
            ) AS reachable_key_set
        FROM true_pair_reachability r
        LEFT JOIN (
            SELECT DISTINCT s1_id, candidate_id
            FROM selected_candidates
        ) c
          ON c.s1_id=r.s1_id
         AND c.candidate_id=r.target_id
        WHERE c.candidate_id IS NULL
          AND exact_hit + token_hit + addrtoken_hit + addrprefix_hit
              + c2_hit + c3_hit + c4_hit > 0
    """)

    missing_reachable_count = con.execute("""
        SELECT count(*) FROM missing_reachable_true_pairs
    """).fetchone()[0]
    print(
        f"missing reachable true pairs: {missing_reachable_count:,}",
        flush=True,
    )

    missing_key_counts = con.execute("""
        SELECT
            sum(exact_hit),
            sum(token_hit),
            sum(addrtoken_hit),
            sum(addrprefix_hit),
            sum(c2_hit),
            sum(c3_hit),
            sum(c4_hit)
        FROM missing_reachable_true_pairs
    """).fetchone()
    for key_name, key_count in zip(
        [
            "k_exact", "k_token", "k_addrtoken", "k_addrprefix",
            "k_c2", "k_c3", "k_c4",
        ],
        missing_key_counts,
    ):
        print(f"reachable via {key_name}: {int(key_count or 0):,}", flush=True)

    print("Combined reachable-key sets:", flush=True)
    for key_set, key_set_count in con.execute("""
        SELECT reachable_key_set, count(*)
        FROM missing_reachable_true_pairs
        GROUP BY reachable_key_set
        ORDER BY count(*) DESC, reachable_key_set
    """).fetchall():
        print(f"  {key_set}: {key_set_count:,}", flush=True)

    ceiling = con.execute("""
        SELECT
            count(*) AS total,

            sum(
                CASE WHEN
                    exact_hit
                    + token_hit
                    + addrtoken_hit
                    + addrprefix_hit
                    + c2_hit
                    + c3_hit
                    + c4_hit > 0
                THEN 1 ELSE 0 END
            ) AS reachable_any,

            sum(
                CASE WHEN
                    exact_hit
                    + token_hit
                    + addrtoken_hit
                    + addrprefix_hit
                    + c2_hit
                    + c3_hit
                    + c4_hit = 0
                THEN 1 ELSE 0 END
            ) AS unreachable
        FROM true_pair_reachability
    """).fetchone()

    total = int(ceiling[0])
    reachable = int(ceiling[1])
    unreachable = int(ceiling[2])

    print(
        f"Total true pairs       : {total:,}",
        flush=True,
    )
    print(
        f"Reachable by any key   : {reachable:,} "
        f"({reachable/total:.2%})",
        flush=True,
    )
    print(
        f"Unreachable by all keys : {unreachable:,} "
        f"({unreachable/total:.2%})",
        flush=True,
    )

    print("\nIndividual blocking-key coverage:", flush=True)

    key_stats = con.execute("""
        SELECT
            sum(exact_hit),
            sum(token_hit),
            sum(addrtoken_hit),
            sum(addrprefix_hit),
            sum(c2_hit),
            sum(c3_hit),
            sum(c4_hit)
        FROM true_pair_reachability
    """).fetchone()

    key_names = [
        "k_exact",
        "k_token",
        "k_addrtoken",
        "k_addrprefix",
        "k_c2",
        "k_c3",
        "k_c4",
    ]

    for name, value in zip(key_names, key_stats):
        print(
            f"{name:15s}: {int(value):,}/{total:,} "
            f"({int(value)/total:.2%})",
            flush=True,
        )

        for name, value in zip(key_names, key_stats):
         print(
            f"{name:15s}: {int(value):,}/{total:,} "
            f"({int(value)/total:.2%})",
            flush=True,
        )

    # ------------------------------------------------------------
    # Inspect unreachable true pairs
    # ------------------------------------------------------------

    print("\n========== UNREACHABLE PAIR DIAGNOSTIC ==========", flush=True)

    unreachable_stats = con.execute("""
        WITH unreachable AS (
            SELECT
                t.s1_id,
                t.target_id,
                s.n_country AS s_country,
                x.n_country AS t_country,
                s.n_name AS s_name,
                x.n_name AS t_name,
                s.n_address AS s_address,
                x.n_address AS t_address,
                s.k_postal AS s_postal,
                x.k_postal AS t_postal
            FROM truth t
            JOIN sample_source s
              ON s.entity_id=t.s1_id
            JOIN target x
              ON x.entity_id=t.target_id
            JOIN true_pair_reachability r
              ON r.s1_id=t.s1_id
             AND r.target_id=t.target_id
            WHERE
                r.exact_hit
                + r.token_hit
                + r.addrtoken_hit
                + r.addrprefix_hit
                + r.c2_hit
                + r.c3_hit
                + r.c4_hit = 0
        )
        SELECT
            count(*) AS total,

            sum(
                CASE WHEN
                    s_country <> ''
                    AND s_country = t_country
                THEN 1 ELSE 0 END
            ) AS same_country,

            sum(
                CASE WHEN
                    s_name <> ''
                    AND s_name = t_name
                THEN 1 ELSE 0 END
            ) AS exact_name,

            sum(
                CASE WHEN
                    length(s_name) >= 2
                    AND length(t_name) >= 2
                    AND substr(s_name,1,2)=substr(t_name,1,2)
                THEN 1 ELSE 0 END
            ) AS name_prefix_2,

            sum(
                CASE WHEN
                    length(s_name) >= 4
                    AND length(t_name) >= 4
                    AND substr(s_name,1,4)=substr(t_name,1,4)
                THEN 1 ELSE 0 END
            ) AS name_prefix_4,

            sum(
                CASE WHEN
                    s_address <> ''
                    AND s_address = t_address
                THEN 1 ELSE 0 END
            ) AS exact_address,

            sum(
                CASE WHEN
                    length(s_address) >= 10
                    AND length(t_address) >= 10
                    AND substr(s_address,1,10)=substr(t_address,1,10)
                THEN 1 ELSE 0 END
            ) AS address_prefix_10,

            sum(
    CASE WHEN
        s_postal <> ''
        AND t_postal <> ''
        AND s_postal = t_postal
        AND s_postal NOT LIKE '%|'
    THEN 1 ELSE 0 END
) AS exact_postal

        FROM unreachable
    """).fetchone()

    print(
        f"{'total':20s}: {int(unreachable_stats[0]):,}",
        flush=True,
    )

    labels = [
        "same_country",
        "exact_name",
        "name_prefix_2",
        "name_prefix_4",
        "exact_address",
        "address_prefix_10",
        "exact_postal",
    ]

    for label, value in zip(labels, unreachable_stats[1:]):
        print(
            f"{label:20s}: {int(value):,}",
            flush=True,
        )
    con.close()


if __name__ == "__main__":
    main()
