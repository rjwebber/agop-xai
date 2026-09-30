from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from zc_xai.io import sha256_array, sha256_file, write_npz
from zc_xai.native_covariance import (
    DENSE_NATIVE_COVARIANCE_SCHEMA_VERSION,
    DENSE_NATIVE_COVARIANCE_STATUS,
    NATIVE_COVARIANCE_SCHEMA_VERSION,
    NATIVE_COVARIANCE_STATUS,
    CenteredSampleCovarianceFactor,
    DenseNativeCovarianceOperator,
    NativeCovarianceBundle,
    PooledCenteredSampleCovarianceFactor,
    solve_minimum_action,
)


class NativeCovarianceTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(91)
        self.x1 = generator.normal(size=(8, 5))
        self.x2 = generator.normal(size=(8, 5)) + np.arange(5)
        self.f1 = CenteredSampleCovarianceFactor(
            self.x1, self.x1.mean(axis=0), 26
        )
        self.f2 = CenteredSampleCovarianceFactor(
            self.x2, self.x2.mean(axis=0), 27
        )

    def test_phase_factor_and_transpose_are_exact(self) -> None:
        centered = self.x1 - self.x1.mean(axis=0)
        dense_factor = centered.T / np.sqrt(self.x1.shape[0] - 1)
        coefficients = np.arange(8, dtype=np.float64) - 2.0
        covector = np.linspace(-1.0, 1.0, 5)
        np.testing.assert_allclose(
            self.f1.apply(coefficients), dense_factor @ coefficients
        )
        np.testing.assert_allclose(
            self.f1.transpose(covector), dense_factor.T @ covector
        )
        self.assertAlmostEqual(
            float(covector @ self.f1.apply(coefficients)),
            float(coefficients @ self.f1.transpose(covector)),
        )

    def test_phase_batch_covariance_matches_dense(self) -> None:
        generator = np.random.default_rng(12)
        vectors = generator.normal(size=(5, 4))
        dense = np.cov(self.x1, rowvar=False, ddof=1)
        np.testing.assert_allclose(
            self.f1.covariance_apply_matrix(vectors, block_rows=3),
            dense @ vectors,
            rtol=2.0e-14,
            atol=2.0e-14,
        )

    def test_phase_workspace_apply_matches_legacy_blocks_bitwise(self) -> None:
        generator = np.random.default_rng(1208)
        samples = generator.normal(size=(17, 11)).astype("<f4")
        mean = samples.mean(axis=0, dtype=np.float64)
        vectors = generator.normal(size=(11, 7))
        vectors_before = vectors.copy()
        factor = CenteredSampleCovarianceFactor(samples, mean, 19)

        expected = np.zeros((11, 7), dtype=np.float64)
        for start in range(0, samples.shape[0], 5):
            centered = np.asarray(samples[start : start + 5], dtype=np.float64)
            centered -= mean
            expected += centered.T @ (centered @ vectors)
        expected /= samples.shape[0] - 1

        actual = factor.covariance_apply_matrix(vectors, block_rows=5)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(vectors, vectors_before)
        self.assertEqual(actual.dtype, np.dtype(np.float64))

    def test_pooled_covariance_is_equal_phase_average(self) -> None:
        pooled = PooledCenteredSampleCovarianceFactor((self.f1, self.f2))
        dense = (
            np.cov(self.x1, rowvar=False, ddof=1)
            + np.cov(self.x2, rowvar=False, ddof=1)
        ) / 2.0
        generator = np.random.default_rng(8)
        vector = generator.normal(size=5)
        vectors = generator.normal(size=(5, 3))
        np.testing.assert_allclose(pooled.covariance_apply(vector), dense @ vector)
        np.testing.assert_allclose(
            pooled.covariance_apply_matrix(vectors, block_rows=3), dense @ vectors
        )

    def test_fused_pooled_apply_matches_separate_legacy_phase_passes(self) -> None:
        generator = np.random.default_rng(1209)
        samples_by_phase = (
            generator.normal(size=(19, 13)).astype("<f4"),
            generator.normal(size=(14, 13)).astype("<f4"),
            generator.normal(size=(23, 13)).astype("<f4"),
        )
        factors = tuple(
            CenteredSampleCovarianceFactor(
                samples,
                samples.mean(axis=0, dtype=np.float64),
                phase_offset,
            )
            for samples, phase_offset in zip(
                samples_by_phase, (17, 18, 19), strict=True
            )
        )
        vectors = generator.normal(size=(13, 6))
        expected = np.zeros((13, 6), dtype=np.float64)
        for samples, factor in zip(samples_by_phase, factors, strict=True):
            phase_result = np.zeros((13, 6), dtype=np.float64)
            for start in range(0, samples.shape[0], 4):
                centered = np.asarray(samples[start : start + 4], dtype=np.float64)
                centered -= factor.mean
                phase_result += centered.T @ (centered @ vectors)
            expected += phase_result / (samples.shape[0] - 1)
        expected /= len(factors)

        pooled = PooledCenteredSampleCovarianceFactor(factors)
        actual = pooled.covariance_apply_matrix(vectors, block_rows=4)
        np.testing.assert_array_equal(actual, expected)

    def test_exact_centered_cache_is_budgeted_and_numerically_identical(self) -> None:
        generator = np.random.default_rng(1210)
        samples = generator.normal(size=(16, 9)).astype("<f4")
        samples_before = samples.copy()
        factor = CenteredSampleCovarianceFactor(
            samples,
            samples.mean(axis=0, dtype=np.float64),
            22,
            source_path=Path("phase22.npy"),
        )
        required = samples.shape[0] * samples.shape[1] * 8
        self.assertEqual(factor.estimated_centered_cache_bytes, required)
        self.assertEqual(factor.centered_cache_bytes, 0)
        with self.assertRaisesRegex(MemoryError, "exceeding"):
            factor.with_contiguous_float64_cache(maximum_bytes=required - 1)

        cached = factor.with_contiguous_float64_cache(maximum_bytes=required)
        self.assertEqual(cached.centered_cache_bytes, required)
        self.assertEqual(cached.source_path, factor.source_path)
        self.assertFalse(cached._centered_float64_samples.flags.writeable)
        vectors = generator.normal(size=(9, 5))
        np.testing.assert_array_equal(
            cached.covariance_apply_matrix(vectors, block_rows=6),
            factor.covariance_apply_matrix(vectors, block_rows=6),
        )
        np.testing.assert_array_equal(samples, samples_before)

    def test_pooled_exact_cache_checks_total_budget_before_allocation(self) -> None:
        pooled = PooledCenteredSampleCovarianceFactor((self.f1, self.f2))
        required = sum(
            factor.estimated_centered_cache_bytes
            for factor in pooled.phase_factors
        )
        self.assertEqual(pooled.estimated_centered_cache_bytes, required)
        with self.assertRaisesRegex(MemoryError, "pooled"):
            pooled.with_contiguous_float64_cache(maximum_bytes=required - 1)
        cached = pooled.with_contiguous_float64_cache(maximum_bytes=required)
        self.assertEqual(cached.centered_cache_bytes, required)
        self.assertEqual(cached.estimated_centered_cache_bytes, 0)
        generator = np.random.default_rng(1211)
        vectors = generator.normal(size=(5, 4))
        np.testing.assert_array_equal(
            cached.covariance_apply_matrix(vectors, block_rows=3),
            pooled.covariance_apply_matrix(vectors, block_rows=3),
        )

        cached_first = self.f1.with_contiguous_float64_cache(
            maximum_bytes=self.f1.estimated_centered_cache_bytes
        )
        mixed = PooledCenteredSampleCovarianceFactor((cached_first, self.f2))
        np.testing.assert_array_equal(
            mixed.covariance_apply_matrix(vectors, block_rows=3),
            pooled.covariance_apply_matrix(vectors, block_rows=3),
        )

    def test_workspace_preserves_legacy_extended_float_conversion(self) -> None:
        if np.dtype(np.longdouble).itemsize <= np.dtype(np.float64).itemsize:
            self.skipTest("platform long double is not wider than float64")
        generator = np.random.default_rng(1212)
        samples = generator.normal(size=(12, 7)).astype(np.longdouble)
        mean = samples.mean(axis=0, dtype=np.longdouble).astype(np.float64)
        vectors = generator.normal(size=(7, 3))
        factor = CenteredSampleCovarianceFactor(samples, mean, 14)
        expected = np.asarray(samples, dtype=np.float64)
        expected -= mean
        expected = expected.T @ (expected @ vectors) / (samples.shape[0] - 1)
        np.testing.assert_array_equal(
            factor.covariance_apply_matrix(vectors, block_rows=12),
            expected,
        )

    def test_minimum_action_matches_covariance_pseudoinverse(self) -> None:
        # Use d > n-1 to exercise the singular empirical covariance case.
        generator = np.random.default_rng(31)
        samples = generator.normal(size=(5, 8))
        factor = CenteredSampleCovarianceFactor(samples, samples.mean(axis=0), 30)
        source_coefficients = generator.normal(size=5)
        target = factor.apply(source_coefficients)
        result = solve_minimum_action(
            factor, target, required_relative_residual=1.0e-9
        )
        dense_factor = (samples - samples.mean(axis=0)).T / 2.0
        covariance = dense_factor @ dense_factor.T
        expected_action = float(target @ np.linalg.pinv(covariance) @ target)
        np.testing.assert_allclose(result.reconstructed_increment, target, atol=1.0e-9)
        self.assertAlmostEqual(result.squared_action, expected_action, places=8)
        self.assertAlmostEqual(float(result.coefficients.mean()), 0.0, places=13)

    def test_out_of_range_increment_is_rejected(self) -> None:
        samples = np.column_stack([np.arange(5.0), np.zeros((5, 2))])
        factor = CenteredSampleCovarianceFactor(samples, samples.mean(axis=0), 26)
        with self.assertRaisesRegex(
            ValueError, "outside the certified covariance range"
        ):
            solve_minimum_action(factor, np.array([0.0, 1.0, 0.0]))

    def test_bundle_loads_relative_memmapped_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples_path = root / "samples.npy"
            np.save(samples_path, self.x1.astype("<f4"), allow_pickle=False)
            means = self.x1.astype("<f4").mean(axis=0, dtype=np.float64)[None, :]
            variances = self.x1.astype("<f4").var(
                axis=0, ddof=1, dtype=np.float64
            )[None, :]
            moments_path = root / "moments.npz"
            write_npz(
                moments_path,
                overwrite=False,
                phase_offsets=np.array([26], dtype=np.int64),
                means=means,
                coordinate_variances=variances,
                pooled_coordinate_variance=variances[0],
            )
            manifest = {
                "schema_version": NATIVE_COVARIANCE_SCHEMA_VERSION,
                "status": NATIVE_COVARIANCE_STATUS,
                "phase_factors": [
                    {
                        "phase_offset": 26,
                        "file": samples_path.name,
                        "sha256": sha256_file(samples_path),
                        "shape": list(self.x1.shape),
                        "dtype": np.dtype("<f4").str,
                    }
                ],
                "means_and_variances": {
                    "file": moments_path.name,
                    "sha256": sha256_file(moments_path),
                    "means_sha256": sha256_array(means),
                    "coordinate_variances_sha256": sha256_array(variances),
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            bundle = NativeCovarianceBundle.load(manifest_path)
            self.assertEqual(bundle.phase_factor(26).samples.shape, self.x1.shape)
            self.assertIsInstance(bundle.phase_factor(26).samples, np.memmap)
            np.testing.assert_allclose(
                bundle.phase_factor(26).covariance_apply(np.ones(5)),
                np.cov(self.x1.astype("<f4"), rowvar=False) @ np.ones(5),
            )


class DenseNativeCovarianceOperatorTest(unittest.TestCase):
    @staticmethod
    def _write_bundle(
        root: Path,
        covariance: np.ndarray,
        *,
        order: str,
    ) -> Path:
        covariance_path = root / "covariance.npy"
        stored = np.array(covariance, dtype="<f8", order=order, copy=True)
        np.save(covariance_path, stored, allow_pickle=False)
        manifest = {
            "schema_version": DENSE_NATIVE_COVARIANCE_SCHEMA_VERSION,
            "status": DENSE_NATIVE_COVARIANCE_STATUS,
            "covariance": {
                "file": covariance_path.name,
                "sha256": sha256_file(covariance_path),
                "shape": list(stored.shape),
                "dtype": np.dtype("<f8").str,
                "order": order,
            },
            "provenance": {"phase_offsets": list(range(36))},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_loads_read_only_c_and_fortran_order_memmaps(self) -> None:
        covariance = np.array(
            [[3.0, -1.0, 0.5], [-1.0, 2.0, 0.25], [0.5, 0.25, 1.5]]
        )
        vector = np.array([0.25, -2.0, 1.5])
        vectors = np.column_stack((vector, np.ones(3)))
        for order in ("C", "F"):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as temporary:
                manifest_path = self._write_bundle(
                    Path(temporary), covariance, order=order
                )
                operator = DenseNativeCovarianceOperator.load(manifest_path)

                self.assertEqual(operator.state_size, 3)
                self.assertEqual(operator.order, order)
                self.assertIsInstance(operator.covariance, np.memmap)
                self.assertFalse(operator.covariance.flags.writeable)
                self.assertEqual(operator.covariance.mode, "r")
                self.assertTrue(
                    operator.covariance.flags.c_contiguous
                    if order == "C"
                    else operator.covariance.flags.f_contiguous
                )
                np.testing.assert_array_equal(
                    operator.covariance_apply(vector), covariance @ vector
                )
                np.testing.assert_array_equal(
                    operator.covariance_apply_matrix(vectors, block_rows=1),
                    covariance @ vectors,
                )

    def test_rejects_hash_shape_and_storage_order_mismatches(self) -> None:
        covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._write_bundle(root, covariance, order="C")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            manifest["covariance"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                DenseNativeCovarianceOperator.load(manifest_path)

            manifest["covariance"]["sha256"] = sha256_file(
                root / "covariance.npy"
            )
            manifest["covariance"]["shape"] = [3, 3]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shape does not match"):
                DenseNativeCovarianceOperator.load(manifest_path)

            manifest["covariance"]["shape"] = [2, 2]
            manifest["covariance"]["order"] = "F"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declared order"):
                DenseNativeCovarianceOperator.load(manifest_path)

    def test_rejects_non_float64_schema_and_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            covariance_path = root / "covariance.npy"
            np.save(covariance_path, np.eye(2, dtype="<f4"), allow_pickle=False)
            manifest = {
                "schema_version": DENSE_NATIVE_COVARIANCE_SCHEMA_VERSION,
                "status": DENSE_NATIVE_COVARIANCE_STATUS,
                "covariance": {
                    "file": covariance_path.name,
                    "sha256": sha256_file(covariance_path),
                    "shape": [2, 2],
                    "dtype": "<f4",
                    "order": "C",
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dtype must be <f8"):
                DenseNativeCovarianceOperator.load(manifest_path)

            manifest_path = self._write_bundle(root, np.eye(2), order="C")
            operator = DenseNativeCovarianceOperator.load(manifest_path)
            with self.assertRaisesRegex(ValueError, "native_vector must have shape"):
                operator.covariance_apply(np.ones(3))
            with self.assertRaisesRegex(ValueError, "at least one column"):
                operator.covariance_apply_matrix(np.empty((2, 0)))
            with self.assertRaisesRegex(ValueError, "block_rows must be positive"):
                operator.covariance_apply_matrix(np.eye(2), block_rows=0)
            with self.assertRaisesRegex(TypeError, "block_rows must be an integer"):
                operator.covariance_apply_matrix(np.eye(2), block_rows=1.5)


if __name__ == "__main__":
    unittest.main()
