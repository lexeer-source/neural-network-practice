"""Check analytic gradients, leakage, persistence, CLI and worker shutdown."""

import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image

from part1_mlp.app.exceptions import (
    DataFileError, InvalidLayerSizeError, MismatchedDataError, ModelFileError,
)
from part1_mlp.app.manager import DatasetManager
from part1_mlp.app.models import NeuralNetwork
from part2_prepare.app.config import PreparationError
from part2_prepare.app.scanner import prepare
from part3_parallel.app.producer_consumer import run_parallel, run_sequential

ROOT = Path(__file__).resolve().parents[1]


class ProjectChecks(unittest.TestCase):
    """Use temporary fixtures and actual subprocesses for public interfaces."""

    def setUp(self):
        """Create a private temporary directory for each independent check."""
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        """Remove generated temporary files after workers have joined."""
        self.temporary.cleanup()

    def write_csv(self, name, header, rows):
        """Return a UTF-8 numeric CSV fixture with the requested header."""
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_gradients_by_finite_differences(self):
        """Compare every parameter's backprop gradient to central differences.
        """
        features = np.array([[0.2, -0.7], [0.8, 0.4], [-0.4, 0.5]])
        targets = np.array([[1.0], [0.0], [1.0]])
        for activation in ["sigmoid", "relu"]:
            model = NeuralNetwork([2, 3, 1], activation, seed=7)
            _, gradients = model.loss_and_gradients(features, targets)
            for layer, expected in zip(model.layers, gradients):
                for attribute, analytic in zip(
                        ["weights", "biases"], expected):
                    parameter = getattr(layer, attribute)
                    for index in np.ndindex(parameter.shape):
                        original = parameter[index]
                        parameter[index] = original + 1e-6
                        plus = np.mean(
                            (model.forward(features) - targets) ** 2)
                        parameter[index] = original - 1e-6
                        minus = np.mean(
                            (model.forward(features) - targets) ** 2)
                        parameter[index] = original
                        self.assertAlmostEqual(
                            analytic[index], (plus - minus) / 2e-6, places=7)

    def test_split_and_scaler_use_train_only(self):
        """An extreme held-out value must not alter training normalization."""
        path = self.write_csv("samples.csv", ["value", "constant", "target"],
                              [[index, 3, index % 2] for index in range(20)])
        manager = DatasetManager().load_csv(path)
        training, _, _, _ = manager.split(seed=42)
        self.assertFalse(set(manager.train_indices)
                         & set(manager.test_indices))
        np.testing.assert_allclose(training.mean(axis=0), 0, atol=1e-14)
        center = manager.center.copy()
        scale = manager.scale.copy()
        manager.transform(np.array([[1e9, 3]]))
        np.testing.assert_array_equal(manager.center, center)
        np.testing.assert_array_equal(manager.scale, scale)
        self.assertEqual(manager.scale[1], 1)

    def test_persistence_preserves_raw_predictions(self):
        """Check weights, activation, scaler and history after a JSON round-
        trip.
        """
        model = NeuralNetwork([2, 5, 1], "relu")
        model.preprocessing = {"center": [1, 20], "scale": [2, 4]}
        model.history = [0.3, 0.2]
        expected = model.predict_raw([[3, 24], [-1, 16]])
        path = self.directory / "model.json"
        model.save(path)
        restored = NeuralNetwork.load(path)
        np.testing.assert_array_equal(
            expected, restored.predict_raw([[3, 24], [-1, 16]]))
        self.assertEqual(restored.history, model.history)

    def test_bad_inputs_have_specific_errors(self):
        """Reject malformed model/CSV, nonfinite data and incompatible sizes.
        """
        with self.assertRaises(InvalidLayerSizeError):
            NeuralNetwork([2, 0, 1])
        model = NeuralNetwork([2, 3, 1])
        for features in [[[1]], [[1, 2], [3]], [[1, float("nan")]]]:
            with self.assertRaises(MismatchedDataError):
                model.forward(features)
        bad = self.write_csv("bad.csv", ["feature", "target"], [[1, 4]])
        with self.assertRaises(DataFileError):
            DatasetManager().load_csv(bad)
        path = self.directory / "model.json"
        path.write_text('{"sizes": []}', encoding="utf-8")
        with self.assertRaises(ModelFileError):
            NeuralNetwork.load(path)

    def test_console_menu_all_actions(self):
        """Exercise create, CSV, train, manual/file predict, save/load and
        plot.
        """
        source = ROOT / "data/input/xor.csv"
        features = self.write_csv("predict.csv", ["left", "right"],
                                   [[0, 1], [1, 1]])
        model = self.directory / "saved.json"
        plot = self.directory / "loss.png"
        commands = ["bad", "1", "2 8 1", "sigmoid", "2", str(source), "0.2",
                    "3", "1", "200", "16", "4", "1", "0 1;1 1", "4", "2",
                    str(features), "5", str(model), "6", str(model), "7",
                    str(plot), "8", "relu", "0"]
        result = subprocess.run(
            [sys.executable, "-m", "part1_mlp.app.main"], cwd=ROOT,
            input="\n".join(commands) + "\n", capture_output=True,
            text=True, encoding="utf-8", timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)
        self.assertIn("Неизвестный пункт меню", result.stdout)
        self.assertIn("класс=", result.stdout)
        self.assertTrue(model.is_file())
        self.assertTrue(plot.is_file())
        (ROOT / "logs/menu_check.txt").write_text(
            result.stdout, encoding="utf-8")

    def test_prepare_csv_and_images(self):
        """Verify a recursive batch of images and shared CSV min-max scaling.
        """
        image_dir = self.directory / "images/nested"
        image_dir.mkdir(parents=True)
        Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(
            image_dir / "white.png")
        output = self.directory / "images.npy"
        stats = prepare(image_dir.parent, {".png"}, output, 16, 2)
        image_array = np.load(output, allow_pickle=False)
        self.assertEqual(stats["file_count"], 1)
        np.testing.assert_array_equal(image_array, np.ones((1, 16, 16)))
        table_dir = self.directory / "tables"
        table_dir.mkdir()
        for name, values in [("first.csv", "value\n10\n20\n"),
                             ("second.csv", "value\n30\n40\n")]:
            (table_dir / name).write_text(values, encoding="utf-8")
        output = self.directory / "tables.npy"
        prepare(table_dir, {".csv"}, output, chunk_size=1)
        np.testing.assert_allclose(np.load(output).ravel(), [0, 1/3, 2/3, 1])

    def test_env_defaults_and_cli_error(self):
        """Read a valid ENV configuration, then reject an absent input
        directory.
        """
        image_dir = self.directory / "images"
        image_dir.mkdir()
        Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(
            image_dir / "black.png")
        environment = os.environ.copy()
        environment.update(NN_DATA_PATH=str(image_dir), NN_EXT="png",
                           NN_OUTPUT_DIR=str(self.directory / "result"),
                           NN_IMAGE_SIZE="12", NN_LOG_LEVEL="ERROR")
        result = subprocess.run(
            [sys.executable, "-m", "part2_prepare.app.cli", "prepare"],
            cwd=ROOT, env=environment, text=True, capture_output=True,
            timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["shape"], [1, 12, 12])
        result = subprocess.run(
            [sys.executable, "-m", "part2_prepare.app.cli", "prepare",
             "--path", str(self.directory / "missing")], cwd=ROOT,
            text=True, capture_output=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)

    def test_parallel_equality_and_shutdown(self):
        """Check deterministic outputs, locked counts and no remaining
        workers.
        """
        image_dir = self.directory / "images"
        image_dir.mkdir()
        for index in range(5):
            pixels = np.full((12, 12), index * 40, dtype=np.uint8)
            Image.fromarray(pixels).save(image_dir / f"image_{index}.png")
        sequential = self.directory / "sequential.npy"
        parallel = self.directory / "parallel.npy"
        run_sequential(image_dir, {".png"}, sequential, 16, 2, 42)
        result = run_parallel(image_dir, {".png"}, parallel, size=16,
                              variants=2, readers=2, processes=2, queue_size=2)
        np.testing.assert_array_equal(np.load(sequential), np.load(parallel))
        self.assertEqual(result["counters"], {"read": 5, "completed": 5})
        self.assertTrue(result["threads_joined"])
        self.assertFalse(mp.active_children())
        self.assertFalse(any(thread.name.startswith("NN-")
                             for thread in threading.enumerate()))

    def test_corrupt_image_cleans_workers_and_partial_output(self):
        """A reader failure must not deadlock or leave an apparently valid
        NPY.
        """
        (self.directory / "broken.png").write_bytes(b"not an image")
        output = self.directory / "failed.npy"
        with self.assertRaises(PreparationError):
            run_parallel(self.directory, {".png"}, output, readers=2,
                         processes=2, queue_size=1)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob("*.partial")))
        self.assertFalse(mp.active_children())
        self.assertFalse(any(thread.name.startswith("NN-")
                             for thread in threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
