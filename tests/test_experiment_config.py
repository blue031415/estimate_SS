import unittest
from pathlib import Path

from ex3 import DEFAULT_CONFIG
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


if __name__ == "__main__":
    unittest.main()
