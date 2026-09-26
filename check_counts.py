import duckdb

con = duckdb.connect("output/matcher_work_v2.duckdb", read_only=True)

print("SOURCE1:", con.execute("SELECT COUNT(*) FROM source1").fetchone()[0])
print("TARGET :", con.execute("SELECT COUNT(*) FROM target").fetchone()[0])

con.close()
