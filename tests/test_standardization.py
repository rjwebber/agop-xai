from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from zc_xai.io import sha256_file
from zc_xai.standardization import load_recorded_standardizer


def _normalization_bundle(root: Path, *, count: int = 3) -> tuple[str, str, dict]:
    relative = "models/cnn/lead-10m/years-9000/seed-000042"
    directory = root / relative
    directory.mkdir(parents=True)
    normalization = directory / "normalization.npz"
    indices = directory / "indices.npz"
    np.savez(
        normalization,
        mean=np.zeros((2, 2, 3), dtype=np.float32),
        scale=np.full((2, 2, 3), 2.0, dtype=np.float32),
        scale_floor=np.asarray(1.0e-6),
        count=np.asarray(count),
    )
    np.savez(indices, fit_inputs=np.arange(3, dtype=np.int64))
    spec = {
        "architecture": "cnn",
        "lead_months": 10,
        "train_years": 9000.0,
        "seed": 42,
    }
    completion = {
        "schema_version": 5,
        "generation_id": "test-generation",
        "data_metadata_sha256": "a" * 64,
        "spec": spec,
        "files": {
            "normalization.npz": sha256_file(normalization),
            "indices.npz": sha256_file(indices),
        },
    }
    (directory / "completed.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    return relative, completion["files"]["normalization.npz"], spec


def _add_distinct_standardization_population(root: Path, relative: str) -> None:
    directory = root / relative
    indices = directory / "indices.npz"
    np.savez(
        indices,
        standardization_inputs=np.arange(3, dtype=np.int64),
        fit_inputs=np.arange(2, dtype=np.int64),
    )
    completion_path = directory / "completed.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files"]["indices.npz"] = sha256_file(indices)
    completion_path.write_text(json.dumps(completion), encoding="utf-8")


class RecordedStandardizerTests(unittest.TestCase):
    def test_validates_fit_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative, digest, spec = _normalization_bundle(root)
            loaded = load_recorded_standardizer(
                root,
                relative,
                expected_data_metadata_sha256="a" * 64,
                expected_input_shape=(2, 2, 3),
                expected_normalization_sha256=digest,
                expected_spec=spec,
            )
            self.assertEqual(loaded.standardizer.count, 3)
            self.assertEqual(
                loaded.completion_sha256,
                sha256_file(loaded.completion_path),
            )
            self.assertTrue(
                np.array_equal(
                    loaded.standardizer.scale,
                    np.full((2, 2, 3), 2.0),
                )
            )

    def test_fresh_artifact_validates_distinct_standardization_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative, digest, spec = _normalization_bundle(root)
            _add_distinct_standardization_population(root, relative)
            loaded = load_recorded_standardizer(
                root,
                relative,
                expected_data_metadata_sha256="a" * 64,
                expected_input_shape=(2, 2, 3),
                expected_normalization_sha256=digest,
                expected_spec=spec,
            )
            self.assertEqual(loaded.standardizer.count, 3)

    def test_rejects_wrong_fit_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative, digest, spec = _normalization_bundle(root, count=4)
            with self.assertRaisesRegex(ValueError, "optimization-fit indices"):
                load_recorded_standardizer(
                    root,
                    relative,
                    expected_data_metadata_sha256="a" * 64,
                    expected_input_shape=(2, 2, 3),
                    expected_normalization_sha256=digest,
                    expected_spec=spec,
                )

    def test_rejects_noninteger_or_duplicate_fit_indices(self) -> None:
        for fit_inputs, message in (
            (np.asarray([0.0, 1.0, 2.0]), "not integers"),
            (np.asarray([0, 1, 1], dtype=np.int64), "sorted, and unique"),
        ):
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                relative, digest, spec = _normalization_bundle(root)
                directory = root / relative
                indices = directory / "indices.npz"
                np.savez(indices, fit_inputs=fit_inputs)
                completion_path = directory / "completed.json"
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                completion["files"]["indices.npz"] = sha256_file(indices)
                completion_path.write_text(json.dumps(completion), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_recorded_standardizer(
                        root,
                        relative,
                        expected_data_metadata_sha256="a" * 64,
                        expected_input_shape=(2, 2, 3),
                        expected_normalization_sha256=digest,
                        expected_spec=spec,
                    )


if __name__ == "__main__":
    unittest.main()
