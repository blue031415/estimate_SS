import unittest
from pathlib import Path

from ex3 import DEFAULT_CONFIG
from ex4 import DEFAULT_CONFIG as EX4_DEFAULT_CONFIG
from ex5 import DEFAULT_CONFIG as EX5_DEFAULT_CONFIG
from experiments.config import load_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def test_loads_ex3_yaml_and_resolves_paths_from_yaml_directory(self):
        config = load_experiment_config(DEFAULT_CONFIG, Path("configs/ex3.yaml"))

        project_root = Path.cwd().resolve()
        self.assertEqual(config["data_path"], project_root / "data/07_01.c3d")
        self.assertEqual(config["output_path"], project_root / "out/ex3/eigen_val_noise.png")
        self.assertEqual(config["contribution_threshold"], 0.999)
        self.assertEqual(config["noise"]["scale"], 30)
        self.assertEqual(config["noise"]["body_parts"], ["left_arm"])

    def test_omitted_yaml_keeps_original_defaults(self):
        defaults = {"data_path": Path("data/input.c3d"), "dimension": 5}

        config = load_experiment_config(defaults)

        self.assertEqual(config, defaults)
        self.assertIsNot(config, defaults)

    def test_rejects_unknown_keys(self):
        defaults = {"data_path": Path("data/input.c3d")}

        with self.assertRaisesRegex(ValueError, "unknown_key"):
            load_experiment_config(defaults, Path("tests/fixtures/unknown_key.yaml"))

    def test_ex4_video_and_model_paths_are_resolved_from_yaml_directory(self):
        config = load_experiment_config(EX4_DEFAULT_CONFIG, Path("configs/ex4.yaml"))

        project_root = Path.cwd().resolve()
        self.assertEqual(
            config["input_videos"]["s1_seq1_cam0"],
            project_root / "data/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi",
        )
        self.assertEqual(config["model_path"], project_root / "data/models/pose_landmarker_lite.task")
        self.assertEqual(config["output_dir"], project_root / "out/ex4")
        self.assertEqual(config["estimator"], "mediapipe")

    def test_ex5_gahu_paths_are_resolved_from_yaml_directory(self):
        config = load_experiment_config(EX5_DEFAULT_CONFIG, Path("configs/ex5.yaml"))

        project_root = Path.cwd().resolve()
        self.assertEqual(config["input_path"], project_root / "data/GaHu-Video/S001.avi")
        self.assertEqual(config["model_path"], project_root / "data/models/pose_landmarker_lite.task")
        self.assertEqual(config["npz_path"], project_root / "out/ex5/S001.npz")
        self.assertEqual(config["gif_path"], project_root / "out/ex5/S001.gif")
        self.assertEqual(config["estimator"], "mediapipe")


if __name__ == "__main__":
    unittest.main()
