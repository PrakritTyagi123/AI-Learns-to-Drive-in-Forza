import sqlite3
c = sqlite3.connect('data/forzatek.db')
c.executescript("""
    DELETE FROM labels WHERE provenance='auto_trusted';
    DELETE FROM proposals;
    DELETE FROM active_queue;
    UPDATE frames SET label_status='unlabeled'
    WHERE label_status IN ('queued','labeled','skipped')
      AND id NOT IN (SELECT DISTINCT frame_id FROM labels);
""")
c.commit()
print("After reset:")
for status, n in c.execute("SELECT label_status, COUNT(*) FROM frames GROUP BY label_status"):
    print(f"  {status:12s} {n}")
c.close()