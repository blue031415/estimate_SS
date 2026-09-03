import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ex12


class Ex12FlowTest(unittest.TestCase):
    @patch("ex12.save_score_comparison")
    def test_comparison_graphs_use_matching_dimension_and_interval(self, save_graph):
        scores6 = {"first": [1.0, 2.0], "second": [3.0], "along_geodestic": [4.0],
                   "orth_geodestic": [5.0]}
        scores11 = {"first": [1.5, 2.5], "second": [3.5], "along_geodestic": [4.5],
                    "orth_geodestic": [5.5]}
        with tempfile.TemporaryDirectory() as directory:
            ex12._save_comparisons(scores6, scores11, Path(directory), "motion", 4, 2)

        self.assertEqual(save_graph.call_count, 5)
        self.assertEqual(list(save_graph.call_args_list[0].args[0]), [0, 1])
        self.assertEqual(list(save_graph.call_args_list[1].args[0]), [2])
        self.assertTrue(str(save_graph.call_args_list[0].args[3]).endswith("_d4_fi2.png"))


if __name__ == "__main__":
    unittest.main()
