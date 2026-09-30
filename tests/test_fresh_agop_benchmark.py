"""Numerical and cache-safety tests for the fresh exact-AGOP benchmark."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.benchmark_fresh_agop import (
    _validated_array_cache,
    accumulate_dense_from_gradient_cache,
    build_eigensystem_cache,
    build_matrix_cache,
    build_parser,
    eigendecompose_dense_matrix,
    generate_gradient_cache,
)
from zc_xai.data import Standardizer
from zc_xai.io import load_json
from zc_xai.xai import _accumulate_dense_agop, input_gradients


class TinyFlatData:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.n_time_steps, self.n_features = self.values.shape
        self.input_shape = (self.n_features,)
        self.metadata_sha256 = "tiny-fresh-agop-benchmark"
        self.input_profile = "core4"

    def load_inputs(
        self,
        indices: np.ndarray,
        *,
        standardizer: Standardizer,
    ) -> np.ndarray:
        return standardizer.transform(self.values[indices]).astype(np.float32)


class QuadraticModel(nn.Module):
    def __init__(self, coefficients: np.ndarray) -> None:
        super().__init__()
        self.register_buffer(
            "coefficients",
            torch.as_tensor(coefficients, dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.coefficients * inputs.square()).sum(dim=1)


def _standardizer(n_features: int, count: int) -> Standardizer:
    return Standardizer(
        mean=np.linspace(-0.2, 0.2, n_features, dtype=np.float32),
        scale=np.linspace(0.8, 1.2, n_features, dtype=np.float32),
        scale_floor=1.0e-6,
        count=count,
    )


class FreshAgopBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(31)
        self.data = TinyFlatData(rng.normal(size=(13, 5)))
        self.standardizer = _standardizer(5, 13)
        self.model = QuadraticModel(
            np.asarray([0.5, -1.0, 2.0, 0.25, 1.25], dtype=np.float32)
        )
        self.references = np.arange(11, dtype=np.int64)

    def test_staged_result_matches_streamed_production_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gradient_path = root / "input_gradients.npy"
            gradients, gradient_record = generate_gradient_cache(
                self.model,
                self.data,  # type: ignore[arg-type]
                self.standardizer,
                self.references,
                batch_size=3,
                device="cpu",
                path=gradient_path,
                identity={"fixture": "gradient-v1"},
                overwrite=False,
            )
            self.assertFalse(gradient_record["cache_hit"])
            expected_gradients = input_gradients(
                self.model,
                self.data.load_inputs(
                    self.references,
                    standardizer=self.standardizer,
                ),
                batch_size=3,
                device="cpu",
            ).reshape(11, 5)
            np.testing.assert_array_equal(gradients, expected_gradients)

            staged_matrix, timing = accumulate_dense_from_gradient_cache(
                gradients,
                block_rows=3,
            )
            streamed_matrix = _accumulate_dense_agop(
                self.model,
                self.data,  # type: ignore[arg-type]
                self.standardizer,
                self.references,
                gradient_batch_size=3,
                device="cpu",
            )
            np.testing.assert_array_equal(staged_matrix, streamed_matrix)
            self.assertGreater(timing["dense_accumulation_seconds"], 0.0)

            basis, root_eigenvalues, decomposition = eigendecompose_dense_matrix(
                staged_matrix
            )
            reconstructed = (
                basis * np.square(root_eigenvalues)[None, :]
            ) @ basis.T
            np.testing.assert_allclose(
                reconstructed,
                staged_matrix,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            self.assertGreater(
                decomposition["full_eigendecomposition_seconds"], 0.0
            )

    def test_stage_caches_resume_and_reject_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gradient_path = root / "input_gradients.npy"
            identity = {"fixture": "resume-v1"}
            first, first_record = generate_gradient_cache(
                self.model,
                self.data,  # type: ignore[arg-type]
                self.standardizer,
                self.references,
                batch_size=4,
                device="cpu",
                path=gradient_path,
                identity=identity,
                overwrite=False,
            )
            second, second_record = generate_gradient_cache(
                self.model,
                self.data,  # type: ignore[arg-type]
                self.standardizer,
                self.references,
                batch_size=4,
                device="cpu",
                path=gradient_path,
                identity=identity,
                overwrite=False,
            )
            self.assertFalse(first_record["cache_hit"])
            self.assertTrue(second_record["cache_hit"])
            np.testing.assert_array_equal(first, second)

            matrix_path = root / "dense_agop_matrix.npy"
            matrix, matrix_record = build_matrix_cache(
                second,
                block_rows=4,
                path=matrix_path,
                identity={"fixture": "matrix-v1"},
                overwrite=False,
            )
            loaded_matrix, loaded_matrix_record = build_matrix_cache(
                second,
                block_rows=4,
                path=matrix_path,
                identity={"fixture": "matrix-v1"},
                overwrite=False,
            )
            self.assertFalse(matrix_record["cache_hit"])
            self.assertTrue(loaded_matrix_record["cache_hit"])
            np.testing.assert_array_equal(matrix, loaded_matrix)

            eigensystem_path = root / "full_eigensystem.npz"
            first_eigensystem = build_eigensystem_cache(
                matrix,
                path=eigensystem_path,
                identity={"fixture": "eigensystem-v1"},
                overwrite=False,
            )
            second_eigensystem = build_eigensystem_cache(
                matrix,
                path=eigensystem_path,
                identity={"fixture": "eigensystem-v1"},
                overwrite=False,
            )
            self.assertFalse(first_eigensystem["cache_hit"])
            self.assertTrue(second_eigensystem["cache_hit"])

            with gradient_path.open("r+b") as stream:
                stream.seek(-1, 2)
                final_byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([final_byte[0] ^ 1]))
            manifest = load_json(
                gradient_path.with_name(gradient_path.name + ".json")
            )
            with self.assertRaisesRegex(ValueError, "corrupted"):
                _validated_array_cache(
                    gradient_path,
                    stage="input_gradients",
                    identity_sha256=manifest["identity_sha256"],
                    expected_shape=(11, 5),
                    expected_dtype=np.dtype(np.float32),
                )

    def test_default_cli_targets_final_core4_cnn(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.input_profile, "core4")
        self.assertEqual(args.architecture, "cnn")
        self.assertEqual(args.lead_months, 10)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.reference_count, 0)
        self.assertEqual(args.gradient_batch_size, 256)


if __name__ == "__main__":
    unittest.main()
