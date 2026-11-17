from __future__ import annotations

import json
from pathlib import Path
import tempfile
import textwrap
import unittest

try:
    from evaluation.qvhighlights.run_official import run_official_evaluator
except ModuleNotFoundError:
    run_official_evaluator = None


class OfficialEvaluatorWrapperTests(unittest.TestCase):
    def test_official_cli_arguments_are_forwarded(self) -> None:
        self.assertIsNotNone(run_official_evaluator, "official wrapper implementation is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "eval.py"
            submission = root / "submission.jsonl"
            ground_truth = root / "gt.jsonl"
            output = root / "metrics.json"
            evaluator.write_text(
                textwrap.dedent(
                    """
                    import argparse, json
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--submission_path', required=True)
                    parser.add_argument('--gt_path', required=True)
                    parser.add_argument('--save_path', required=True)
                    parser.add_argument('--not_verbose', action='store_true')
                    args = parser.parse_args()
                    with open(args.save_path, 'w', encoding='utf-8') as handle:
                        json.dump({'submission': args.submission_path,
                                   'gt': args.gt_path,
                                   'not_verbose': args.not_verbose}, handle)
                    """
                ),
                encoding="utf-8",
            )
            submission.write_text(
                json.dumps({"qid": 1, "query": "synthetic", "vid": "v1", "pred_relevant_windows": []}) + "\n",
                encoding="utf-8",
            )
            ground_truth.write_text("{}\n", encoding="utf-8")
            result = run_official_evaluator(evaluator, submission, ground_truth, output)
            self.assertTrue(result["not_verbose"])
            self.assertEqual(Path(result["submission"]), submission.resolve())
            self.assertEqual(Path(result["gt"]), ground_truth.resolve())

    def test_existing_output_is_not_overwritten(self) -> None:
        self.assertIsNotNone(run_official_evaluator, "official wrapper implementation is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("eval.py", "submission.jsonl", "gt.jsonl", "metrics.json"):
                (root / name).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                run_official_evaluator(
                    root / "eval.py",
                    root / "submission.jsonl",
                    root / "gt.jsonl",
                    root / "metrics.json",
                )


if __name__ == "__main__":
    unittest.main()
