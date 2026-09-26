from pathlib import Path
import re
import duckdb
import joblib
import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "output" / "matcher_work_v2.duckdb"
MODEL_PATH = PROJECT_ROOT / "data" / "rf_model.joblib"

OUTPUT_DIR = PROJECT_ROOT / "output" / "model_checkpoints"
FINAL_OUTPUT = PROJECT_ROOT / "output" / "matching_results_model.tsv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MEMORY_LIMIT = "2GB"
THREADS = 4

BATCH_SIZE = 100_000

MODEL_THRESHOLD = 0.50


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# FAST SIMILARITY
# ============================================================

def address_similarity(a, b):

    return fuzz.ratio(
        normalize_text(a),
        normalize_text(b)
    ) / 100.0


def address_token_similarity(a, b):

    a_tokens = set(
        normalize_text(a).split()
    )

    b_tokens = set(
        normalize_text(b).split()
    )

    if not a_tokens or not b_tokens:
        return 0.0

    union = a_tokens | b_tokens

    if not union:
        return 0.0

    return len(a_tokens & b_tokens) / len(union)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def checkpoint_path(batch_number):

    return OUTPUT_DIR / (
        f"batch_{batch_number:05d}.tsv"
    )


def get_completed_batches():

    completed = set()

    for file in OUTPUT_DIR.glob(
        "batch_*.tsv"
    ):

        try:

            number = int(
                file.stem.split("_")[1]
            )

            completed.add(number)

        except Exception:

            pass

    return completed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESUMABLE FAST MODEL MATCHING")
    print("=" * 70)

    print()
    print("Database :", DB_PATH)
    print("Model    :", MODEL_PATH)
    print("Output   :", FINAL_OUTPUT)
    print()

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not DB_PATH.exists():

        print("ERROR: Database not found:")
        print(DB_PATH)

        return

    if not MODEL_PATH.exists():

        print("ERROR: Model not found:")
        print(MODEL_PATH)

        return

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading Random Forest model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded.")
    print()

    feature_columns = [
        "name_similarity",
        "address_similarity",
        "name_token_similarity",
        "address_token_similarity",
        "country_match"
    ]

    # --------------------------------------------------------
    # OPEN DATABASE
    # --------------------------------------------------------

    print("Opening DuckDB...")

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True
    )

    con.execute(
        f"SET memory_limit='{MEMORY_LIMIT}'"
    )

    con.execute(
        f"SET threads={THREADS}"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    # --------------------------------------------------------
    # TABLE CHECK
    # --------------------------------------------------------

    tables = {
        row[0]
        for row in con.execute(
            "SHOW TABLES"
        ).fetchall()
    }

    print(
        "Tables:",
        sorted(tables)
    )

    required = {
        "source1",
        "target",
        "unique_name_matches",
        "exact_address_matches"
    }

    missing = required - tables

    if missing:

        print()
        print(
            "ERROR: Missing tables:",
            sorted(missing)
        )

        con.close()

        return

    # --------------------------------------------------------
    # SOURCE COUNT
    # --------------------------------------------------------

    source_count = con.execute(
        "SELECT COUNT(*) FROM source1"
    ).fetchone()[0]

    # --------------------------------------------------------
    # CANDIDATE COUNT
    # --------------------------------------------------------

    candidate_count = con.execute("""
        SELECT COUNT(*)

        FROM source1 s

        JOIN target t

            ON s.n_country = t.n_country
            AND s.n_name = t.n_name

        WHERE

            s.n_country <> ''
            AND s.n_name <> ''

            AND NOT EXISTS (

                SELECT 1

                FROM unique_name_matches u

                WHERE
                    u.source1_entity_id =
                    s.entity_id
            )

            AND NOT EXISTS (

                SELECT 1

                FROM exact_address_matches a

                WHERE
                    a.source1_entity_id =
                    s.entity_id
            )
    """).fetchone()[0]

    print()
    print(
        f"Source records   : {source_count:,}"
    )

    print(
        f"Candidate pairs   : {candidate_count:,}"
    )

    # --------------------------------------------------------
    # COMPLETED CHECKPOINTS
    # --------------------------------------------------------

    completed = get_completed_batches()

    print()

    if completed:

        print(
            "Existing checkpoints:",
            len(completed)
        )

        print(
            "Completed batches:",
            sorted(completed)
        )

    else:

        print(
            "No previous checkpoints found."
        )

    print()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Candidate ordering is deterministic.
    #
    # We use ROW_NUMBER() to divide the candidate stream
    # into fixed batches.
    # --------------------------------------------------------

    print(
        "Preparing candidate stream..."
    )

    query = """

        SELECT

            s.entity_id AS source1_entity_id,

            t.entity_id AS candidate_entity_id,

            s.n_address AS s1_address,

            t.n_address AS candidate_address

        FROM source1 s

        JOIN target t

            ON s.n_country = t.n_country
            AND s.n_name = t.n_name

        WHERE

            s.n_country <> ''
            AND s.n_name <> ''

            AND NOT EXISTS (

                SELECT 1

                FROM unique_name_matches u

                WHERE
                    u.source1_entity_id =
                    s.entity_id
            )

            AND NOT EXISTS (

                SELECT 1

                FROM exact_address_matches a

                WHERE
                    a.source1_entity_id =
                    s.entity_id
            )

        ORDER BY

            s.entity_id,

            t.entity_id
    """

    cursor = con.execute(query)

    total_batches = (
        candidate_count + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"Total batches: {total_batches}"
    )

    print()

    # --------------------------------------------------------
    # PROCESS BATCHES
    # --------------------------------------------------------

    processed_candidates = 0

    for batch_number in range(
        total_batches
    ):

        start = (
            batch_number * BATCH_SIZE
        )

        end = min(
            start + BATCH_SIZE,
            candidate_count
        )

        output_file = checkpoint_path(
            batch_number
        )

        # ----------------------------------------------------
        # SKIP ALREADY COMPLETED BATCH
        # ----------------------------------------------------

        if batch_number in completed:

            print(
                f"[SKIP] Batch "
                f"{batch_number + 1}/"
                f"{total_batches} "
                f"already completed."
            )

            continue

        print()
        print(
            "=" * 70
        )

        print(
            f"PROCESSING BATCH "
            f"{batch_number + 1}/"
            f"{total_batches}"
        )

        print(
            f"Candidates "
            f"{start:,} - {end:,}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # FETCH EXACT BATCH
        #
        # We consume the cursor sequentially.
        # No OFFSET is used.
        # ----------------------------------------------------

        batch_rows = cursor.fetchmany(
            end - start
        )

        if not batch_rows:

            break

        # ----------------------------------------------------
        # BEST MATCH PER SOURCE IN THIS BATCH
        # ----------------------------------------------------

        best = {}

        for row in batch_rows:

            source_id = row[0]
            target_id = row[1]

            source_address = row[2]
            target_address = row[3]

            addr_sim = address_similarity(
                source_address,
                target_address
            )

            addr_token_sim = (
                address_token_similarity(
                    source_address,
                    target_address
                )
            )

            # ------------------------------------------------
            # Since candidate generation already requires:
            #
            # n_country == n_country
            # n_name == n_name
            #
            # these are effectively exact.
            # ------------------------------------------------

            X = pd.DataFrame(
                [[
                    1.0,
                    addr_sim,
                    1.0,
                    addr_token_sim,
                    1
                ]],
                columns=feature_columns
            )

            probability = float(
                model.predict_proba(X)[0][
                    list(model.classes_).index(1)
                ]
            )

            current = best.get(
                source_id
            )

            candidate = (
                probability,
                addr_sim,
                addr_token_sim,
                str(target_id),
                target_id
            )

            if (
                current is None
                or candidate[:4]
                > current[:4]
            ):

                best[source_id] = candidate

        # ----------------------------------------------------
        # WRITE CHECKPOINT
        # ----------------------------------------------------

        rows_out = []

        for source_id, value in best.items():

            probability = value[0]
            addr_sim = value[1]
            addr_token_sim = value[2]
            target_id = value[4]

            if probability >= MODEL_THRESHOLD:

                rows_out.append(
                    (
                        source_id,
                        target_id,
                        probability,
                        addr_sim,
                        addr_token_sim
                    )
                )

        checkpoint_df = pd.DataFrame(
            rows_out,
            columns=[
                "source1_entity_id",
                "target_id",
                "probability",
                "address_similarity",
                "address_token_similarity"
            ]
        )

        checkpoint_df.to_csv(
            output_file,
            sep="\t",
            index=False
        )

        processed_candidates = end

        print()
        print(
            f"Batch complete."
        )

        print(
            f"Accepted matches in batch: "
            f"{len(checkpoint_df):,}"
        )

        print(
            f"Progress: "
            f"{processed_candidates:,}/"
            f"{candidate_count:,}"
        )

        print(
            f"Checkpoint saved:"
        )

        print(
            output_file
        )

    # --------------------------------------------------------
    # COMBINE CHECKPOINTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMBINING CHECKPOINTS")
    print("=" * 70)

    checkpoint_files = sorted(
        OUTPUT_DIR.glob(
            "batch_*.tsv"
        )
    )

    if not checkpoint_files:

        print(
            "ERROR: No checkpoint files found."
        )

        con.close()

        return

    frames = []

    for file in checkpoint_files:

        try:

            df = pd.read_csv(
                file,
                sep="\t"
            )

            if not df.empty:

                frames.append(df)

        except Exception as e:

            print(
                f"WARNING: Could not read "
                f"{file}: {e}"
            )

    if frames:

        model_matches = pd.concat(
            frames,
            ignore_index=True
        )

    else:

        model_matches = pd.DataFrame(
            columns=[
                "source1_entity_id",
                "target_id",
                "probability",
                "address_similarity",
                "address_token_similarity"
            ]
        )

    # --------------------------------------------------------
    # ONE BEST MODEL MATCH PER SOURCE
    # --------------------------------------------------------

    if not model_matches.empty:

        model_matches = (
            model_matches
            .sort_values(
                [
                    "source1_entity_id",
                    "probability",
                    "address_similarity",
                    "address_token_similarity",
                    "target_id"
                ],
                ascending=[
                    True,
                    False,
                    False,
                    False,
                    True
                ]
            )
            .drop_duplicates(
                subset=[
                    "source1_entity_id"
                ],
                keep="first"
            )
        )

    print()
    print(
        f"Final model matches: "
        f"{len(model_matches):,}"
    )

    # --------------------------------------------------------
    # REGISTER MODEL MATCHES
    # --------------------------------------------------------

    con.register(
        "accepted_model_matches",
        model_matches[
            [
                "source1_entity_id",
                "target_id"
            ]
        ]
    )

    # --------------------------------------------------------
    # WRITE FINAL TSV
    # --------------------------------------------------------

    print()
    print(
        "Writing final matching_results_model.tsv..."
    )

    con.execute(
        f"""
        COPY (

            SELECT

                s.entity_id
                AS source1_entity_id,

                COALESCE(

                    CAST(
                        u.matched_entity_id
                        AS VARCHAR
                    ),

                    CAST(
                        a.target_id
                        AS VARCHAR
                    ),

                    CAST(
                        m.target_id
                        AS VARCHAR
                    ),

                    ''

                ) AS matched_entity_ids

            FROM source1 s

            LEFT JOIN unique_name_matches u

                ON s.entity_id =
                   u.source1_entity_id

            LEFT JOIN exact_address_matches a

                ON s.entity_id =
                   a.source1_entity_id

            LEFT JOIN accepted_model_matches m

                ON s.entity_id =
                   m.source1_entity_id

            ORDER BY
                s.entity_id

        )

        TO '{FINAL_OUTPUT.as_posix()}'

        (
            DELIMITER '\\t',
            HEADER true,
            QUOTE '"',
            ESCAPE '"'
        )
        """
    )

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    unique_count = con.execute("""
        SELECT COUNT(*)
        FROM unique_name_matches
    """).fetchone()[0]

    address_count = con.execute("""
        SELECT COUNT(*)
        FROM exact_address_matches
    """).fetchone()[0]

    model_count = len(
        model_matches
    )

    matched_count = con.execute("""
        SELECT COUNT(*)

        FROM (

            SELECT

                s.entity_id,

                COALESCE(

                    u.matched_entity_id,

                    a.target_id,

                    m.target_id

                ) AS matched_id

            FROM source1 s

            LEFT JOIN unique_name_matches u

                ON s.entity_id =
                   u.source1_entity_id

            LEFT JOIN exact_address_matches a

                ON s.entity_id =
                   a.source1_entity_id

            LEFT JOIN accepted_model_matches m

                ON s.entity_id =
                   m.source1_entity_id
        )

        WHERE

            matched_id IS NOT NULL

            AND matched_id <> ''
    """).fetchone()[0]

    unmatched_count = (
        source_count
        - matched_count
    )

    con.close()

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MATCHING COMPLETE")
    print("=" * 70)

    print(
        f"Source records        : "
        f"{source_count:,}"
    )

    print(
        f"Unique-name matches   : "
        f"{unique_count:,}"
    )

    print(
        f"Exact-address matches : "
        f"{address_count:,}"
    )

    print(
        f"Model matches         : "
        f"{model_count:,}"
    )

    print(
        f"Total matched         : "
        f"{matched_count:,}"
    )

    print(
        f"Total unmatched       : "
        f"{unmatched_count:,}"
    )

    print()
    print(
        "FINAL FILE:"
    )

    print(
        FINAL_OUTPUT
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()