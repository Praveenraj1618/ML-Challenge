"""Synthetic write-path benchmark; not an estimate for the actual dataset."""
import hashlib
import sqlite3
import tempfile
import time
from pathlib import Path

from code.business_entity_resolution.src.fast_pipeline import Index


def chunk(start, stop):
    records = [(i, f"S2-{i:09d}", f"Business {i}", "12 Example Road", "US") for i in range(start,stop)]
    postings = [(hashlib.blake2b(f"{i}:{j}".encode(),digest_size=12).digest(),i)
                for i in range(start,stop) for j in range(20)]
    return records, postings


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        base = Index(root / "base.sqlite")
        try:
            for start in range(1,150001,10000):
                records, postings = chunk(start,start+10000)
                base._flush(records,postings,start+9999)
            for name in ("old","tuned"):
                connection=sqlite3.connect(root / f"{name}.sqlite")
                base.db.backup(connection)
                connection.close()
        finally:
            base.db.close()
        for name in ("old","tuned"):
            index = Index(root / f"{name}.sqlite",cache_mb=64 if name=="old" else 512)
            if name=="old":
                index.db.execute("PRAGMA wal_autocheckpoint=1000")
            elapsed=0.
            try:
                size=5000 if name=="old" else 10000
                for start in range(150001,180001,size):
                    records,postings=chunk(start,start+size)
                    before=time.monotonic()
                    if name=="old":
                        with index.db:
                            index.db.executemany("INSERT INTO records VALUES (?,?,?,?,?)",records)
                            index.db.executemany("INSERT INTO blocks VALUES (?,?)",postings)
                            index.put("done",start+size-1)
                    else:
                        index._flush(records,postings,start+size-1)
                    elapsed+=time.monotonic()-before
                before=time.monotonic()
                index.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                elapsed+=time.monotonic()-before
                assert index.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]==180000
                assert index.db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]==3600000
                print(f"{name}: 30,000 rows appended in {elapsed:.2f}s ({30000/elapsed:.0f} rows/s), including final checkpoint",flush=True)
            finally:
                index.db.close()


if __name__=="__main__":
    main()
