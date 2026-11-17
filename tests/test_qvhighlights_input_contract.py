from __future__ import annotations

import unittest

try:
    from experiments.qvhighlights.input_contract import normalize_inference_row
except ModuleNotFoundError:
    normalize_inference_row = None


class QVHighlightsInputContractTests(unittest.TestCase):
    def test_ground_truth_fields_are_removed_before_inference(self) -> None:
        self.assertIsNotNone(normalize_inference_row, "input contract implementation is missing")
        row = normalize_inference_row(
            {
                "qid": 11,
                "query": "synthetic query",
                "vid": "video-1",
                "duration": 12.0,
                "relevant_windows": [[1.0, 2.0]],
                "relevant_clip_ids": [1],
                "saliency_scores": [[1, 2, 3]],
                "answer": "not for inference",
            }
        )
        self.assertEqual(
            row,
            {
                "dataset": "qvhighlights",
                "query_id": "11",
                "question": "synthetic query",
                "video_id": "video-1",
                "duration": 12.0,
            },
        )

    def test_required_identity_is_enforced(self) -> None:
        self.assertIsNotNone(normalize_inference_row, "input contract implementation is missing")
        with self.assertRaisesRegex(ValueError, "requires query ID"):
            normalize_inference_row({"query": "synthetic query", "vid": "video-1"})


if __name__ == "__main__":
    unittest.main()
