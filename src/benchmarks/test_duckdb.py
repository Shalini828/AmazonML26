import duckdb

con = duckdb.connect("amazon_ml.duckdb")

files = {
    "Source 1": "data/dataset/train/train_source1.tsv",
    "Source 2": "data/dataset/train/train_source2.tsv",
    "Source 3": "data/dataset/train/train_source3.tsv",
    "Ground Truth": "data/dataset/train/train_ground_truth.tsv",
}

for name, path in files.items():
    result = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv_auto(
            '{path}',
            delim='\t'
        )
    """).fetchone()[0]

    print(f"{name}: {result:,} records")

con.close()