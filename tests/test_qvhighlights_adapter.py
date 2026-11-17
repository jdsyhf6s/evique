from __future__ import annotations

import unittest

try:
    from evaluation.qvhighlights.adapter import query_to_prediction, validate_submission
except ModuleNotFoundError:
    query_to_prediction = None
    validate_submission = None


class QVHighlightsAdapterTests(unittest.TestCase):
    def test_explicit_same_video_interval_is_preserved(self) -> None:
        self.assertIsNotNone(query_to_prediction, "adapter implementation is missing")
        query = {"qid": 7, "query": "synthetic event", "vid": "video-1", "duration": 10.0}
        retrieval = {
            "evidence": [
                {
                    "score": 0.75,
                    "provenance": {
                        "video_id": "video-1",
                        "start_time": 2.0,
                        "end_time": 6.0,
                    },
                }
            ]
        }
        row = query_to_prediction(query, retrieval)
        self.assertEqual(row["qid"], 7)
        self.assertEqual(row["vid"], "video-1")
        self.assertEqual(row["pred_relevant_windows"], [[2.0, 6.0, 0.75]])
        validate_submission([row])

    def test_cross_video_evidence_fails_closed(self) -> None:
        self.assertIsNotNone(query_to_prediction, "adapter implementation is missing")
        query = {"qid": 8, "query": "synthetic event", "vid": "video-1", "duration": 10.0}
        retrieval = {
            "evidence": [
                {
                    "score": 0.5,
                    "provenance": {
                        "video_id": "video-2",
                        "start_time": 1.0,
                        "end_time": 2.0,
                    },
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "video identity"):
            query_to_prediction(query, retrieval)

    def test_missing_explicit_interval_is_not_invented(self) -> None:
        self.assertIsNotNone(query_to_prediction, "adapter implementation is missing")
        query = {"qid": 9, "query": "synthetic event", "vid": "video-1", "duration": 10.0}
        row = query_to_prediction(query, {"evidence": [{"score": 0.4, "timestamp": 3.0}]})
        self.assertEqual(row["pred_relevant_windows"], [])


if __name__ == "__main__":
    unittest.main()
