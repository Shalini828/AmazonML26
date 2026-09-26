import duckdb

con = duckdb.connect("output/matcher_work_v2.duckdb", read_only=True)

print("TABLES:")
print(con.execute("SHOW TABLES").fetchall())

print("SOURCE1 ROWS:")
print(con.execute("SELECT COUNT(*) FROM source1").fetchone()[0])

con.close()
