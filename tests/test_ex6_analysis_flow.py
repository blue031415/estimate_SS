import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ex6_analysis


class Ex6AnalysisFlowTest(unittest.TestCase):
    @patch("ex6_analysis._run_ex2_analyses")
    def test_saved_npz_is_analyzed_without_pose_estimation(self, run_analyses):
        input_dir = Path("out/test_ex6_analysis_input")
        output_dir = Path("out/test_ex6_analysis_output")
        input_dir.mkdir(parents=True, exist_ok=True)
        points = np.asarray(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        np.savez_compressed(input_dir / "motion.npz", world_landmarks=points)
        config = dict(ex6_analysis.DEFAULT_CONFIG)
        config.update({"input_dir": input_dir, "output_dir": output_dir})

        with patch("ex6_analysis.load_experiment_config", return_value=config):
            ex6_analysis.main("ignored.yaml")

        run_analyses.assert_called_once()
        np.testing.assert_array_equal(
            run_analyses.call_args.args[0],
            [
                [[1.0, 3.0, -2.0], [4.0, 6.0, -5.0]],
                [[7.0, 9.0, -8.0], [10.0, 12.0, -11.0]],
            ],
        )
        self.assertEqual(run_analyses.call_args.args[1], output_dir)
        self.assertEqual(run_analyses.call_args.args[2], "motion")


if __name__ == "__main__":
    unittest.main()
