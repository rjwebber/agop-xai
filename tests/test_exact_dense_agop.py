"""Numerical tests for the streamed exact-dense AGOP implementation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zc_xai.data import Standardizer
from zc_xai.xai import (
    _clip_numerical_negative_eigenvalues,
    build_exact_dense_agop_factor,
    input_gradients,
    sample_empirical_neighbors,
)


class TinyFlatData:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.n_time_steps, self.n_features = self.values.shape
        self.input_shape = (self.n_features,)
        self.metadata_sha256 = "tiny-flat-data"
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


def _standardizer(n_features: int) -> Standardizer:
    return Standardizer(
        mean=np.zeros(n_features, dtype=np.float32),
        scale=np.ones(n_features, dtype=np.float32),
        scale_floor=1.0e-6,
        count=11,
    )


def _matrix_from_factor(factor) -> np.ndarray:
    basis = np.asarray(factor.basis, dtype=np.float64)
    eigenvalues = np.square(factor.root_eigenvalues, dtype=np.float64)
    return (basis * eigenvalues[None, :]) @ basis.T


class ExactDenseAgopTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(27)
        self.data = TinyFlatData(rng.standard_normal((11, 4)))
        self.standardizer = _standardizer(4)
        self.model = QuadraticModel(np.asarray([0.5, -1.0, 2.0, 0.25]))
        self.references = np.arange(11, dtype=np.int64)

    def test_matches_direct_gradient_gram_and_exact_eigh(self) -> None:
        gradients = input_gradients(
            self.model,
            self.data.load_inputs(
                self.references,
                standardizer=self.standardizer,
            ),
            batch_size=11,
            device="cpu",
        ).astype(np.float64)
        expected = gradients.T @ gradients / gradients.shape[0]
        factor = build_exact_dense_agop_factor(
            self.model,
            self.data,  # type: ignore[arg-type]
            self.standardizer,
            self.references,
            gradient_batch_size=3,
            device="cpu",
        )
        np.testing.assert_allclose(
            _matrix_from_factor(factor),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        expected_eigenvalues = np.linalg.eigvalsh(expected)[::-1]
        np.testing.assert_allclose(
            np.square(factor.root_eigenvalues),
            expected_eigenvalues,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertEqual(factor.approximation_rank, 4)
        self.assertEqual(factor.basis.dtype, np.float64)
        self.assertEqual(
            factor.solver_metadata["method"],  # type: ignore[index]
            "exact_dense_empirical_agop",
        )
        self.assertEqual(
            factor.solver_metadata["accumulation_kernel"],  # type: ignore[index]
            "scipy.linalg.blas.dsyrk upper triangle",
        )
        self.assertEqual(
            factor.solver_metadata[  # type: ignore[index]
                "eigendecomposition_driver"
            ],
            "scipy.linalg.eigh driver=evd",
        )

    def test_batching_invariance_and_matrix_cache_round_trip(self) -> None:
        first = build_exact_dense_agop_factor(
            self.model,
            self.data,  # type: ignore[arg-type]
            self.standardizer,
            self.references,
            gradient_batch_size=1,
            device="cpu",
        )
        second = build_exact_dense_agop_factor(
            self.model,
            self.data,  # type: ignore[arg-type]
            self.standardizer,
            self.references[::-1],
            gradient_batch_size=7,
            device="cpu",
        )
        np.testing.assert_allclose(
            _matrix_from_factor(first),
            _matrix_from_factor(second),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        np.testing.assert_array_equal(second.reference_indices, self.references)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dense_agop.npy"
            written = build_exact_dense_agop_factor(
                self.model,
                self.data,  # type: ignore[arg-type]
                self.standardizer,
                self.references,
                gradient_batch_size=4,
                device="cpu",
                matrix_cache_path=path,
            )
            loaded = build_exact_dense_agop_factor(
                self.model,
                self.data,  # type: ignore[arg-type]
                self.standardizer,
                self.references,
                gradient_batch_size=2,
                device="cpu",
                matrix_cache_path=path,
            )
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_name(path.name + ".json").is_file())
            self.assertTrue(
                loaded.solver_metadata["matrix_cache"]["loaded"]  # type: ignore[index]
            )
            np.testing.assert_allclose(
                _matrix_from_factor(written),
                _matrix_from_factor(loaded),
                rtol=0.0,
                atol=0.0,
            )

    def test_bad_inputs_and_negative_eigenvalues_are_rejected(self) -> None:
        invalid_indices = (
            np.asarray([], dtype=np.int64),
            np.asarray([[0]], dtype=np.int64),
            np.asarray([0.0, 1.0]),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([11], dtype=np.int64),
        )
        for indices in invalid_indices:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                build_exact_dense_agop_factor(
                    self.model,
                    self.data,  # type: ignore[arg-type]
                    self.standardizer,
                    indices,
                    gradient_batch_size=2,
                    device="cpu",
                )
        with self.assertRaisesRegex(ValueError, "gradient_batch_size"):
            build_exact_dense_agop_factor(
                self.model,
                self.data,  # type: ignore[arg-type]
                self.standardizer,
                self.references,
                gradient_batch_size=0,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "materially negative"):
            _clip_numerical_negative_eigenvalues(np.asarray([-1.0, 2.0]))

    def test_flat_neighbor_distances_use_all_nonbatch_coordinates(self) -> None:
        values = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0]],
            dtype=np.float32,
        )
        data = TinyFlatData(values)
        sample = sample_empirical_neighbors(
            data,  # type: ignore[arg-type]
            _standardizer(2),
            query_index=0,
            candidate_indices=np.arange(4),
            neighbor_percent=100.0,
            n_samples=3,
            seed=3,
            distance_batch_size=2,
        )
        np.testing.assert_array_equal(sample.neighborhood_indices, [1, 2, 3])
        np.testing.assert_allclose(
            sample.neighborhood_rms_distances,
            np.asarray([1.0, 2.0, 3.0]) / np.sqrt(2.0),
        )


if __name__ == "__main__":
    unittest.main()
