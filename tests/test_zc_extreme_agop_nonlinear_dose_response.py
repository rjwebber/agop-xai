"""Focused tests for the nonlinear extreme-event dose-response orchestrator."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_zc_extreme_agop_nonlinear_dose_response as dose  # noqa: E402


class NonlinearDoseResponseTests(unittest.TestCase):
    def test_case_labels_and_coefficient_order_are_stable(self) -> None:
        self.assertEqual(
            [dose._case_label(value) for value in dose.COMMON_COEFFICIENTS],
            [
                "minus_1",
                "minus_0p9",
                "minus_0p8",
                "minus_0p7",
                "minus_0p6",
                "minus_0p5",
                "minus_0p4",
                "minus_0p3",
                "minus_0p2",
                "minus_0p1",
                "plus_0p1",
                "plus_0p2",
                "plus_0p3",
            ],
        )
        self.assertEqual(
            dose.COEFFICIENTS_BY_EVENT["extreme_el_nino"],
            dose.COEFFICIENTS_BY_EVENT["extreme_la_nina"],
        )

    def test_event_filtering_preserves_canonical_order_and_skips_unsupported(
        self,
    ) -> None:
        requested = (0.5, 0.1, -1.0, 0.3)
        self.assertEqual(
            dose._selected_coefficients("extreme_el_nino", requested),
            (-1.0, 0.1, 0.3),
        )
        self.assertEqual(
            dose._selected_coefficients("extreme_la_nina", requested),
            (-1.0, 0.1, 0.3),
        )
        self.assertEqual(dose._selected_coefficients("extreme_la_nina", (0.5,)), ())

    def test_execution_order_moves_outward_from_zero(self) -> None:
        self.assertEqual(
            dose._solve_order((-1.0, -0.3, -0.1, 0.5, 0.1, 0.3)),
            (-0.1, -0.3, -1.0, 0.1, 0.3, 0.5),
        )
        self.assertEqual(dose._adjacent_coefficient(-0.4), -0.3)
        self.assertEqual(dose._adjacent_coefficient(0.5), 0.4)
        self.assertIsNone(dose._adjacent_coefficient(-0.1))

    def test_fine_homotopy_relation_is_strictly_between_doses(self) -> None:
        self.assertEqual(
            dose._warm_start_relation(-0.8, -0.9, -0.8),
            "verified_adjacent_dose",
        )
        self.assertEqual(
            dose._warm_start_relation(-0.8, -0.9, -0.875),
            "verified_fine_homotopy_bridge",
        )
        self.assertIsNone(dose._warm_start_relation(-0.8, -0.9, -0.75))
        self.assertIsNone(dose._warm_start_relation(-0.8, -0.9, 0.875))

    def test_fine_bridge_resolver_checks_identity_direction_and_file_hash(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "dose-response"
            source_dir = root / "retries" / "fine_bridge"
            current_dir = root / "canonical" / "extreme_la_nina" / "minus_0p9"
            source_dir.mkdir(parents=True)
            current_dir.mkdir(parents=True)
            event_index = 123
            source_report, source_result = dose._result_paths(
                source_dir, event_index
            )
            source_report.parent.mkdir(parents=True)
            source_report.write_text("{}\n", encoding="utf-8")
            source_dual = np.asarray([[1.0, 2.0]], dtype=np.float64)
            dose.write_npz(
                source_result,
                overwrite=True,
                dual_variables=source_dual,
            )
            source_hash = dose.sha256_file(source_result)
            source_dual_hash = dose.sha256_array(source_dual)
            current_identity = {
                "target_event_label": "extreme_la_nina",
                "extreme_event_input_index": event_index,
                "trajectory": "authentic-extreme",
                "data_metadata_sha256": "data",
                "model_checkpoint_sha256": "model",
                "base_agop_direction_sha256": "base-direction",
                "covariance_manifest_sha256": "covariance",
            }
            current_target = {
                "base_direction_sha256": "base-direction",
                "direction_sha256": "oriented-direction",
                "event_phase_modulo_36": 26,
            }

            def write_candidate(coefficient: float) -> None:
                dose.write_json(
                    source_dir / "run_manifest.json",
                    {
                        "status": "complete",
                        "run_identity": current_identity
                        | {"projection_change_multiple": coefficient},
                        "agop_target": current_target,
                    },
                    overwrite=True,
                )

            write_candidate(-0.875)
            with patch.object(dose, "_validated_case", return_value={"ok": True}):
                resolved = dose._find_verified_warm_start_ancestor(
                    current_dir,
                    -0.9,
                    initial_dual_npz_sha256=source_hash,
                    initial_dual_raw_variables_sha256=source_dual_hash,
                    initial_dual_scale=1.0,
                    initial_dual_scaled_variables_sha256=source_dual_hash,
                    current_identity=current_identity,
                    current_target=current_target,
                )
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(
                resolved["source_kind"], "verified_fine_homotopy_bridge"
            )
            self.assertEqual(resolved["source_coefficient"], -0.875)
            self.assertEqual(resolved["source_result_sha256"], source_hash)

            for label, coefficient, requested_hash in (
                ("unrelated", -0.75, source_hash),
                ("wrong_direction", 0.875, source_hash),
                ("wrong_hash", -0.875, "0" * 64),
            ):
                with self.subTest(label=label):
                    write_candidate(coefficient)
                    with patch.object(
                        dose, "_validated_case", return_value={"ok": True}
                    ):
                        rejected = dose._find_verified_warm_start_ancestor(
                            current_dir,
                            -0.9,
                            initial_dual_npz_sha256=requested_hash,
                            initial_dual_raw_variables_sha256=source_dual_hash,
                            initial_dual_scale=1.0,
                            initial_dual_scaled_variables_sha256=source_dual_hash,
                            current_identity=current_identity,
                            current_target=current_target,
                        )
                    self.assertIsNone(rejected)

            write_candidate(-0.3)
            scaled_dual_hash = dose.sha256_array((2.0 / 3.0) * source_dual)
            with patch.object(dose, "_validated_case", return_value={"ok": True}):
                scaled = dose._find_verified_warm_start_ancestor(
                    current_dir,
                    -0.2,
                    initial_dual_npz_sha256=source_hash,
                    initial_dual_raw_variables_sha256=source_dual_hash,
                    initial_dual_scale=2.0 / 3.0,
                    initial_dual_scaled_variables_sha256=scaled_dual_hash,
                    current_identity=current_identity,
                    current_target=current_target,
                )
            self.assertIsNotNone(scaled)
            assert scaled is not None
            self.assertEqual(
                scaled["source_kind"], "verified_scaled_outer_dose"
            )
            self.assertEqual(scaled["source_coefficient"], -0.3)

    def test_scaled_outer_warm_start_relation_is_exact_and_same_sign(self) -> None:
        self.assertEqual(
            dose._warm_start_relation(
                -0.1,
                -0.2,
                -0.3,
                initial_dual_scale=2.0 / 3.0,
            ),
            "verified_scaled_outer_dose",
        )
        for source, scale in ((-0.3, 1.0), (-0.3, 0.5), (0.3, 2.0 / 3.0)):
            with self.subTest(source=source, scale=scale):
                self.assertIsNone(
                    dose._warm_start_relation(
                        -0.1,
                        -0.2,
                        source,
                        initial_dual_scale=scale,
                    )
                )

    def test_continuation_uses_tenth_dose_increments(self) -> None:
        self.assertEqual(dose._continuation_fractions(0.1), (1.0,))
        self.assertEqual(
            dose._continuation_fractions(-0.3), (1 / 3, 2 / 3, 1.0)
        )

    def test_direct_command_requests_authentic_pooled_nonlinear_problem(self) -> None:
        args = argparse.Namespace(
            data_dir=Path("data/processed/zc-v3"),
            artifacts_dir=Path("artifacts/zc-v3"),
            agop_benchmark_dir=Path("outputs/agop"),
            generation_workspace=Path("outputs/generation"),
            adjoint_validation_dir=Path("outputs/validation"),
            maximum_iterations=30,
            constraint_tolerance=1.0e-5,
            stationarity_tolerance=1.0e-5,
            complementarity_tolerance=1.0e-5,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-4,
            initial_trust_radius=10.0,
            maximum_trust_radius=100.0,
            covariance_block_rows=1024,
            timeout_seconds=120.0,
            skip_data_checksums=False,
        )
        command = dose._direct_command(
            args,
            event="extreme_la_nina",
            coefficient=-0.3,
            output_dir=Path("/tmp/one-case"),
        )
        joined = " ".join(command)
        self.assertIn("--trajectory authentic-extreme", joined)
        self.assertIn("--target-event extreme_la_nina", joined)
        self.assertIn("--projection-change-multiple -0.3", joined)
        self.assertIn("--case pooled", joined)
        self.assertIn("--continuation-fractions 0.333333,0.666667,1", joined)
        self.assertIn("--scale-continuation-warm-start", command)

        warm_started = dose._direct_command(
            args,
            event="extreme_la_nina",
            coefficient=-0.4,
            output_dir=Path("/tmp/one-case"),
            initial_dual_npz=Path("/tmp/minus_0p3/result.npz"),
        )
        warm_joined = " ".join(warm_started)
        self.assertIn("--continuation-fractions 1", warm_joined)
        self.assertIn(
            "--initial-dual-npz /tmp/minus_0p3/result.npz", warm_joined
        )
        self.assertNotIn("--scale-continuation-warm-start", warm_started)
        scaled_outer = dose._direct_command(
            args,
            event="extreme_la_nina",
            coefficient=-0.2,
            output_dir=Path("/tmp/one-scaled-case"),
            initial_dual_npz=Path("/tmp/minus_0p3/result.npz"),
            initial_dual_scale=2.0 / 3.0,
        )
        self.assertEqual(
            scaled_outer[scaled_outer.index("--initial-dual-scale") + 1],
            "0.66666666666666663",
        )

    def test_combined_schema_uses_the_common_event_coefficient_family(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for event_index, event in enumerate(dose.EVENT_ORDER):
                event_dir = output_root / event
                event_dir.mkdir()
                coefficients = dose.COEFFICIENTS_BY_EVENT[event]
                arrays: dict[str, np.ndarray] = {
                    "coefficients": np.asarray(coefficients, dtype=np.float64),
                    "nino3_baseline": np.linspace(
                        -1.0, 1.0, dose.BOUNDARY_COUNT, dtype=np.float64
                    ),
                }
                for coefficient_index, coefficient in enumerate(coefficients):
                    label = dose._case_label(coefficient)
                    arrays[f"requested_release_projection__{label}"] = np.asarray(
                        coefficient, dtype=np.float64
                    )
                    arrays[f"realized_release_projection__{label}"] = np.asarray(
                        coefficient + 1.0e-8, dtype=np.float64
                    )
                    arrays[f"nino3__{label}"] = np.full(
                        dose.BOUNDARY_COUNT,
                        coefficient_index + event_index,
                        dtype=np.float64,
                    )
                archive_path = event_dir / "trajectories.npz"
                dose.write_npz(archive_path, overwrite=True, **arrays)
                dose.write_json(
                    event_dir / "report.json",
                    {
                        "schema_version": dose.SCHEMA_VERSION,
                        "status": "complete",
                        "experiment_type": dose.EXPERIMENT_TYPE,
                        "event_label": event,
                        "event_input_index": 100 + event_index,
                        "event_target_index": 130 + event_index,
                        "natural_projection_q": 2.0 + event_index,
                        "baseline_original_e_projection": 2.0 + event_index,
                        "coefficient_order": list(coefficients),
                        "completed_coefficients": list(coefficients),
                        "cases": {},
                        "outputs": {
                            "trajectories_file": archive_path.name,
                            "trajectories_sha256": dose.sha256_file(archive_path),
                        },
                    },
                    overwrite=True,
                )

            report = dose._combine_if_complete(output_root)
            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(
                report["coefficients_by_event"]["extreme_el_nino"],
                list(dose.COMMON_COEFFICIENTS),
            )
            self.assertEqual(
                report["coefficients_by_event"]["extreme_la_nina"],
                list(dose.COMMON_COEFFICIENTS),
            )
            with np.load(
                output_root / "trajectories.npz", allow_pickle=False
            ) as archive:
                self.assertNotIn("coefficients", archive.files)
                for event, coefficients in dose.COEFFICIENTS_BY_EVENT.items():
                    np.testing.assert_array_equal(
                        archive[f"coefficients__{event}"], coefficients
                    )
                    self.assertEqual(
                        archive[f"nino3_perturbed__{event}"].shape,
                        (len(coefficients), dose.BOUNDARY_COUNT),
                    )
                    self.assertTrue(
                        np.isfinite(archive[f"nino3_perturbed__{event}"]).all()
                    )


if __name__ == "__main__":
    unittest.main()
