import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from code.business_entity_resolution.src.fast_pipeline import Index, FIELDS, digest_files, keys


class FastIndexTests(unittest.TestCase):
    def test_resume_reuse_and_country_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "targets.tsv"
            records = [dict(zip(FIELDS,r)) for r in [
                ("S2-1","Alpha Bakery","12 Main Street","India"),
                ("S3-1","Alpha Bakery","12 Main Street","France"),
                ("S2-2","Beta Cafe","17 Park Road","India")]]
            with path.open("w",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=FIELDS,delimiter="\t")
                writer.writeheader()
                writer.writerows(records)
            index = Index(root / "index.sqlite")
            try:
                # Create the checkpoint using the ORIGINAL unsorted SQL writes.
                index.put("fingerprint",digest_files([path]))
                first = records[0]
                with index.db:
                    index.db.execute("INSERT INTO records VALUES (?,?,?,?,?)",(1,*(first[f] for f in FIELDS)))
                    index.db.executemany("INSERT INTO blocks VALUES (?,?)",
                        [(k,1) for k in keys(first["business_name"],first["business_address"],first["country"])])
                    index.put("done",1)
                index.db.close()
                index = Index(root / "index.sqlite",index_batch=1)
                index.build([path])
                found, _ = index.retrieve(dict(first, entity_id="S1-1"))
                self.assertIn("S2-1",[r[1] for r in found])
                self.assertNotIn("S3-1",[r[1] for r in found])
                index.build([path])
                self.assertEqual(index.db.execute("SELECT COUNT(*) FROM records").fetchone()[0],3)
                self.assertEqual(index.meta("done"),3)
                # Failed transactions must not advance the restart offset.
                with self.assertRaises(sqlite3.IntegrityError):
                    index._flush([(4,"S2-1","duplicate","","India")],[],4)
                self.assertEqual(index.meta("done"),3)
                with path.open("a") as stream:
                    stream.write("S2-9\tChanged\tAddress\tIndia\n")
                with self.assertRaisesRegex(ValueError,"changed"):
                    index.build([path])
            finally:
                index.db.close()

    def test_empty_text_has_no_blocks(self):
        self.assertEqual(keys("","","India"),[])

    def test_country_case_and_suffix_normalization(self):
        self.assertEqual(keys("Alpha Bakery Ltd","12 Main Road","INDIA"),
                         keys("Alpha Bakery","12 Main Rd","india"))


if __name__ == "__main__":
    unittest.main()
