"""Focused tests for the all-phase dense native covariance compiler."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scripts.build_zc_all_phase_dense_covariance as dense_builder
from zc_xai.io import sha256_array, sha256_file, write_json, write_npz
from zc_xai.native_covariance import (
    NATIVE_COVARIANCE_SCHEMA_VERSION,
    NATIVE_COVARIANCE_STATUS,
    DenseNativeCovarianceOperator,
)


def _write_factor_artifact(
    root: Path, *, phase_count: int = 3, sample_count: int = 6, state_size: int = 4
) -> tuple[Path, tuple[np.ndarray, ...]]:
    factor_dir = root / "factors"
    factor_dir.mkdir()
    generator = np.random.default_rng(4471)
    samples = tuple(
        (
            generator.normal(size=(sample_count, state_size))
            + 3.0 * phase
            + np.arange(state_size)
        ).astype("<f4")
        for phase in range(phase_count)
    )
    means = np.stack(
        [np.sum(values, axis=0, dtype=np.float64) / sample_count for values in samples]
    )
    variances = np.stack(
        [
            np.sum(
                (np.asarray(values, dtype=np.float64) - mean) ** 2,
                axis=0,
                dtype=np.float64,
            )
            / (sample_count - 1)
            for values, mean in zip(samples, means, strict=True)
        ]
    )
    pooled_variance = np.mean(variances, axis=0, dtype=np.float64)
    moments_path = factor_dir / "phase_moments.npz"
    write_npz(
        moments_path,
        overwrite=False,
        phase_offsets=np.arange(phase_count, dtype=np.int64),
        means=means,
        coordinate_variances=variances,
        pooled_coordinate_variance=pooled_variance,
    )

    records = []
    for phase, (values, mean, variance) in enumerate(
        zip(samples, means, variances, strict=True)
    ):
        path = factor_dir / f"native_phase_{phase:02d}.npy"
        np.save(path, values, allow_pickle=False)
        records.append(
            {
                "phase_offset": phase,
                "file": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "shape": list(values.shape),
                "dtype": values.dtype.str,
                "mean_sha256": sha256_array(mean),
                "coordinate_variance_sha256": sha256_array(variance),
            }
        )
    manifest = {
        "schema_version": NATIVE_COVARIANCE_SCHEMA_VERSION,
        "status": NATIVE_COVARIANCE_STATUS,
        "dimensions": {
            "state_size": state_size,
            "sample_count_per_phase": sample_count,
            "phase_count": phase_count,
            "phase_offsets": list(range(phase_count)),
        },
        "phase_factors": records,
        "means_and_variances": {
            "file": moments_path.name,
            "sha256": sha256_file(moments_path),
            "means_sha256": sha256_array(means),
            "coordinate_variances_sha256": sha256_array(variances),
            "pooled_coordinate_variance_sha256": sha256_array(pooled_variance),
        },
        "source_capture": {"manifest_sha256": "c" * 64},
    }
    manifest_path = factor_dir / "manifest.json"
    write_json(manifest_path, manifest, overwrite=False)
    return manifest_path, samples


def _arguments(manifest: Path, output: Path, *, resume: bool = False) -> list[str]:
    result = [
        "--factor-manifest",
        str(manifest),
        "--output-dir",
        str(output),
        "--expected-state-size",
        "4",
        "--expected-samples",
        "6",
        "--expected-phase-count",
        "3",
        "--block-rows",
        "6",
        "--audit-vectors",
        "2",
        "--audit-block-columns",
        "2",
    ]
    if resume:
        result.append("--resume")
    return result


def _expected_covariance(samples: tuple[np.ndarray, ...]) -> np.ndarray:
    covariances = []
    for values in samples:
        values64 = np.asarray(values, dtype=np.float64)
        centered = values64 - np.mean(values64, axis=0, dtype=np.float64)
        covariances.append(centered.T @ centered / (values.shape[0] - 1))
    return np.mean(covariances, axis=0, dtype=np.float64)


class AllPhaseDenseCovarianceTests(unittest.TestCase):
    def test_compiles_exact_equal_phase_covariance_and_loader_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, samples = _write_factor_artifact(root)
            output = root / "dense"
            self.assertEqual(dense_builder.main(_arguments(manifest, output)), 0)

            operator = DenseNativeCovarianceOperator.load(
                output / "dense_manifest.json"
            )
            self.assertEqual(operator.order, "F")
            self.assertTrue(operator.covariance.flags.f_contiguous)
            self.assertFalse(operator.covariance.flags.writeable)
            np.testing.assert_allclose(
                operator.covariance,
                _expected_covariance(samples),
                rtol=2.0e-13,
                atol=2.0e-13,
            )
            self.assertFalse((output / dense_builder.WORK_DIRECTORY_NAME).exists())

    def test_interrupted_phase_is_not_published_or_double_counted_on_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, samples = _write_factor_artifact(root)
            output = root / "dense"
            real_update = dense_builder._dsyrk_update
            calls = 0

            def fail_on_second_phase(
                covariance: np.ndarray,
                centered_samples: np.ndarray,
                *,
                alpha: float,
            ) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("intentional interruption")
                real_update(covariance, centered_samples, alpha=alpha)

            with (
                patch.object(
                    dense_builder,
                    "_dsyrk_update",
                    side_effect=fail_on_second_phase,
                ),
                self.assertRaisesRegex(RuntimeError, "intentional interruption"),
            ):
                dense_builder.main(_arguments(manifest, output))

            self.assertFalse((output / "dense_manifest.json").exists())
            self.assertFalse((output / "covariance.npy").exists())
            progress = dense_builder.load_json(
                output / dense_builder.WORK_DIRECTORY_NAME / "progress.json"
            )
            self.assertEqual(progress["completed_phase_count"], 1)

            self.assertEqual(
                dense_builder.main(_arguments(manifest, output, resume=True)), 0
            )
            operator = DenseNativeCovarianceOperator.load(
                output / "dense_manifest.json"
            )
            np.testing.assert_allclose(
                operator.covariance,
                _expected_covariance(samples),
                rtol=2.0e-13,
                atol=2.0e-13,
            )


if __name__ == "__main__":
    unittest.main()
