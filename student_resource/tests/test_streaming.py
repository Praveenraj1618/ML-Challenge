import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from code.business_entity_resolution.src.retrieval import RetrievalConfig, iter_candidates
from code.business_entity_resolution.src.features import build_features


def records(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


class StreamingTests(unittest.TestCase):
    def test_shard_and_query_sizes_do_not_change_scores(self):
        targets = records([
            ("S2-1", "Alpha Bakery", "12 High Street", "IN"),
            ("S2-2", "Beta Bakery", "90 Low Street", "IN"),
            ("S3-1", "Alpha Bakery", "12 High Street", "IN"),
            ("S3-2", "Gamma Bank", "4 Park Road", "IN"),
        ])
        queries = records([("S1-1", "Alpha Bakery", "12 High Street", "in"),
                           ("S1-2", "Beta Bakery", "90 Low Street", "IN")])
        outputs = []
        for shard, batch in [(1, 1), (100, 100)]:
            config = RetrievalConfig(document_batch=shard, query_batch=batch,
                                     char_name_k=4, char_address_k=4, bm25_k=4)
            frames = [p for _, p in iter_candidates(queries, targets, config)]
            outputs.append(pd.concat(frames).sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True))
        pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=False, atol=1e-6, rtol=1e-6)
        self.assertIn("S2-1", outputs[0].candidate_entity_id.tolist())
        features = build_features(outputs[0], queries, targets)
        self.assertTrue(np.isfinite(features.select_dtypes("number").to_numpy()).all())
        # At a tied K boundary any equally-scored ID is valid, but the cap holds.
        capped = pd.concat([p for _, p in iter_candidates(queries, targets,
            RetrievalConfig(document_batch=1, char_name_k=1, char_address_k=1, bm25_k=1))])
        self.assertLessEqual(capped.groupby("source1_entity_id").size().max(), 3)

    def test_empty_country_and_empty_text(self):
        queries = records([("S1-1", "", "", "IN"), ("S1-2", "a", "", "XX")])
        targets = records([("S2-1", "", "", "IN")])
        with tempfile.TemporaryDirectory() as directory:
            results = list(iter_candidates(queries, targets, RetrievalConfig(), directory))
            self.assertEqual(sum(len(q) for q, _ in results), 2)
            self.assertTrue(all(p.empty for _, p in results))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_batch(self):
        with self.assertRaises(ValueError):
            list(iter_candidates(records([]), records([]), RetrievalConfig(query_batch=0)))


if __name__ == "__main__":
    unittest.main()
