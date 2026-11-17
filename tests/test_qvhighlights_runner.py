from __future__ import annotations

import unittest

try:
    from experiments.qvhighlights.run import run_queries
except ModuleNotFoundError:
    run_queries = None


class QVHighlightsRunnerTests(unittest.TestCase):
    def test_retriever_receives_no_ground_truth(self) -> None:
        self.assertIsNotNone(run_queries, "runner implementation is missing")
        observed = []

        def retrieve(question, metadata):
            observed.append((question, dict(metadata)))
            return {
                "evidence": [
                    {
                        "score": 0.8,
                        "provenance": {
                            "video_id": metadata["video_id"],
                            "start_time": 1.0,
                            "end_time": 4.0,
                        },
                    }
                ]
            }

        rows = [
            {
                "qid": 15,
                "query": "synthetic query",
                "vid": "video-1",
                "duration": 10.0,
                "relevant_windows": [[1.0, 4.0]],
                "saliency_scores": [[1, 1, 1]],
            }
        ]
        predictions = run_queries(rows, retrieve)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(observed[0][0], "synthetic query")
        self.assertNotIn("relevant_windows", observed[0][1])
        self.assertNotIn("saliency_scores", observed[0][1])


if __name__ == "__main__":
    unittest.main()
