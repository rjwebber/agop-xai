"""Fast algebra and packed-I/O tests for the empirical balanced control map."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adjoint.balanced_control_map import (
    BalancedControlMap,
    ControlSegment,
    IndependentControlLayout,
    fit_balanced_control_map,
    load_independent_control_layout,
    paired_reconstruction_diagnostics,
    read_restart_sample,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def small_layout() -> IndependentControlLayout:
    return IndependentControlLayout(
        packed_size=9,
        manifest_schema_version=2,
        segments=(
            ControlSegment(
                name="ocean",
                packed_start=1,
                packed_stop=4,
                compact_start=0,
                compact_stop=3,
                shape=(3,),
                order="F",
                role="synthetic ocean",
                restart_record=2,
            ),
            ControlSegment(
                name="sst",
                packed_start=7,
                packed_stop=8,
                compact_start=3,
                compact_stop=4,
                shape=(1,),
                order="F",
                role="synthetic SST",
                restart_record=3,
            ),
        ),
    )


def fitted_map() -> tuple[BalancedControlMap, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(81)
    latent = rng.standard_normal((300, 3))
    native_loading = rng.standard_normal((4, 3))
    observation_loading = rng.standard_normal((8, 3))
    native = latent @ native_loading.T + np.asarray((2.0, -1.0, 4.0, 0.5))
    observation = latent @ observation_loading.T + np.arange(8) / 10.0
    model = fit_balanced_control_map(
        native,
        observation,
        layout=small_layout(),
        rank=3,
    )
    return model, native, observation


class BalancedMapAlgebraTests(unittest.TestCase):
    def test_explicit_B_and_BT_satisfy_packed_dot_identity(self) -> None:
        model, _, _ = fitted_map()
        rng = np.random.default_rng(3)
        packed_matrix = model.packed_control_matrix()
        self.assertEqual(packed_matrix.shape, (model.layout.packed_size, model.rank))
        for _ in range(12):
            control = rng.standard_normal(model.rank)
            packed_seed = rng.standard_normal(model.layout.packed_size)
            left = model.apply_B(control) @ packed_seed
            right = control @ model.apply_BT(packed_seed)
            self.assertAlmostEqual(float(left), float(right), places=13)
            # BLAS implementations may associate the two equivalent matrix
            # products differently, so require floating-point agreement rather
            # than bitwise identity.
            np.testing.assert_allclose(
                packed_matrix @ control,
                model.apply_B(control),
                rtol=2e-14,
                atol=2e-14,
            )
            outside = np.ones(model.layout.packed_size, dtype=bool)
            outside[model.layout.packed_indices] = False
            np.testing.assert_array_equal(model.apply_B(control)[outside], 0.0)

    def test_explicit_G_and_GT_satisfy_dot_identity_and_eof_gram(self) -> None:
        model, _, _ = fitted_map()
        rng = np.random.default_rng(4)
        control = rng.standard_normal(model.rank)
        seed = rng.standard_normal(model.observation_size)
        self.assertAlmostEqual(
            float(model.apply_G(control) @ seed),
            float(control @ model.apply_GT(seed)),
            places=13,
        )
        np.testing.assert_allclose(
            model.G.T @ model.G,
            np.diag(model.eigenvalues),
            rtol=2e-14,
            atol=2e-14,
        )

    def test_unregularized_lift_is_exact_only_inside_retained_range(self) -> None:
        model, _, _ = fitted_map()
        target_control = np.asarray((0.3, -0.5, 0.7))
        target = model.apply_G(target_control)
        solution = model.lift_observation(target, ridge_fraction=0.0)
        np.testing.assert_allclose(solution.control, target_control, atol=2e-14)
        np.testing.assert_allclose(
            solution.predicted_observation, target, atol=2e-14
        )
        self.assertAlmostEqual(
            solution.diagnostics["representable_squared_fraction"], 1.0, places=13
        )

        rng = np.random.default_rng(5)
        outside = rng.standard_normal(model.observation_size)
        outside -= model.G @ np.linalg.solve(model.G.T @ model.G, model.G.T @ outside)
        unresolved = model.lift_observation(outside, ridge_fraction=0.0)
        self.assertLess(np.linalg.norm(unresolved.predicted_observation), 2e-14)
        self.assertLess(
            abs(unresolved.diagnostics["representable_squared_fraction"]), 2e-14
        )

    def test_serialization_round_trip_preserves_factors_without_pickle(self) -> None:
        model, _, _ = fitted_map()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            model.save(path, metadata={"purpose": "unit test"})
            loaded, metadata = BalancedControlMap.load(path)
        self.assertEqual(metadata, {"purpose": "unit test"})
        np.testing.assert_array_equal(loaded.B, model.B)
        np.testing.assert_array_equal(loaded.G, model.G)
        np.testing.assert_array_equal(loaded.layout.packed_indices, [1, 2, 3, 7])

    def test_paired_reconstruction_reports_observation_and_native_errors(self) -> None:
        model, native, observation = fitted_map()
        result = paired_reconstruction_diagnostics(
            model,
            native[-20:],
            observation[-20:],
            ridge_fraction=0.0,
        )
        self.assertLess(
            result.summary["median_observation_standardized_rms_residual"],
            2e-14,
        )
        self.assertAlmostEqual(
            result.summary["median_observation_cosine"], 1.0, places=13
        )
        self.assertEqual(result.controls.shape, (20, 3))


class PackedRestartTests(unittest.TestCase):
    def test_real_manifest_layout_has_only_declared_independent_controls(self) -> None:
        layout = load_independent_control_layout(
            REPOSITORY_ROOT / "adjoint/fortran_kernel/state_manifest.json"
        )
        self.assertEqual(layout.packed_size, 59_148)
        self.assertEqual(layout.compact_size, 28_591)
        self.assertEqual(
            [segment.name for segment in layout.segments],
            ["AKB", "UB", "HB", "V", "TO"],
        )
        compact = np.arange(layout.compact_size, dtype=np.float64)
        packed = layout.compact_to_packed(compact)
        np.testing.assert_array_equal(layout.packed_to_compact(packed), compact)
        self.assertEqual(np.count_nonzero(packed), layout.compact_size - 1)

    def test_fresh_restart_parser_respects_records_and_manifest_offsets(self) -> None:
        layout = load_independent_control_layout(
            REPOSITORY_ROOT / "adjoint/fortran_kernel/state_manifest.json"
        )
        record1 = bytearray(172)
        struct.pack_into("<f", record1, 0, 123.5)
        struct.pack_into("<i", record1, 4, 4321)
        record2_values = np.arange(28_041, dtype="<f4")
        record2 = b"H" * 80 + record2_values.tobytes()
        record3_values = (100_000 + np.arange(8_160)).astype("<f4")
        record3 = record3_values.tobytes()
        record4 = b"diagnostic payload is deliberately opaque"

        def frame(payload: bytes | bytearray) -> bytes:
            marker = struct.pack("<I", len(payload))
            return marker + payload + marker

        restart = b"".join(
            frame(record) for record in (record1, record2, record3, record4)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.hst"
            path.write_bytes(restart)
            sample = read_restart_sample(path, layout)
        self.assertEqual(sample.nt, 4321)
        self.assertEqual(sample.time_months, 123.5)
        packed_prefix = np.concatenate((record2_values, record3_values))
        np.testing.assert_array_equal(
            sample.native_controls,
            packed_prefix[layout.packed_indices],
        )


if __name__ == "__main__":
    unittest.main()
