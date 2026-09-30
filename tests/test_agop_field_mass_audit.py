"""Focused tests for the reproducible AGOP field-mass audit."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import analyze_agop_field_mass as audit  # noqa: E402

from zc_xai.fresh_figure7 import load_or_build_exact_factor  # noqa: E402
from zc_xai.training import ExperimentSpec  # noqa: E402
from zc_xai.xai import AgopFactor  # noqa: E402


class AgopFieldMassAuditTests(unittest.TestCase):
    def test_channel_order_and_phase_are_explicit(self) -> None:
        self.assertEqual(
            audit.SPATIAL_FIELDS,
            (
                "sst_anomaly",
                "thermocline_depth",
                "zonal_ocean_current",
                "meridional_ocean_current",
            ),
        )
        self.assertEqual(
            audit.PHASE_FIELDS,
            ("annual_phase_sin", "annual_phase_cos"),
        )

    def test_field_mass_keeps_phase_in_denominator(self) -> None:
        vector = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 12.0])
        result = audit.field_mass(vector, (4, 1, 1))
        total = 199.0
        self.assertAlmostEqual(result["sst_anomaly"], 1.0 / total)
        self.assertAlmostEqual(result["thermocline_depth"], 4.0 / total)
        self.assertAlmostEqual(result["zonal_ocean_current"], 9.0 / total)
        self.assertAlmostEqual(result["meridional_ocean_current"], 16.0 / total)
        self.assertAlmostEqual(result["ocean_currents"], 25.0 / total)
        self.assertAlmostEqual(result["thermocline_and_currents"], 29.0 / total)
        self.assertAlmostEqual(result["phase"], 169.0 / total)
        self.assertAlmostEqual(result["spatial"] + result["phase"], 1.0)
        self.assertAlmostEqual(result["l2_norm"], np.sqrt(total))

    def test_exact_agop_action_is_normalized(self) -> None:
        basis = np.eye(3)
        roots = np.asarray([1.0, 2.0, 3.0])
        actual = audit._agop_explanation(np.asarray([1.0, 1.0, 1.0]), basis, roots)
        expected = np.asarray([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(float(np.linalg.norm(actual)), 1.0)

    def test_crossover_reports_first_overtake_and_near_tie(self) -> None:
        rows = []
        for lead in audit.EXPECTED_LEADS:
            sst = 0.6 if lead < 9 else 0.4
            thermocline = 0.4 if lead < 9 else 0.6
            if lead == 11:
                sst, thermocline = 0.50005, 0.49995
            rows.append(
                {
                    "lead_months": lead,
                    "mass_fraction": {
                        "sst_anomaly": sst,
                        "thermocline_depth": thermocline,
                    },
                }
            )
        result = audit.crossover_summary(rows)
        self.assertEqual(result["first_lead_thermocline_exceeds_sst"], 9)
        self.assertEqual(result["leads_thermocline_exceeds_sst"], [9, 10, 12])
        self.assertEqual(
            result["leads_sst_exceeds_thermocline"], list(range(1, 9)) + [11]
        )
        self.assertEqual(
            result["near_ties_within_0.1_percentage_point"][0]["lead_months"],
            11,
        )

    def test_mass_rejects_missing_phase_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "four spatial fields"):
            audit.field_mass(np.ones(5), (4, 1, 1))

    def test_release_defaults_use_staged_factor_cache(self) -> None:
        args = audit.build_parser().parse_args([])
        self.assertEqual(
            args.factor_dir,
            Path("artifacts/zc-v3/manuscript/field_mass_crosslead"),
        )
        self.assertFalse(args.build_missing_factors)
        self.assertEqual(args.gradient_batch_size, 1024)

    def test_compact_factor_build_omits_dense_matrix(self) -> None:
        data = SimpleNamespace(
            metadata_sha256="data-sha256",
            input_profile="core4",
            n_features=2,
        )
        standardizer = SimpleNamespace(
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
        )
        experiment = SimpleNamespace(
            checkpoint_sha256="checkpoint-sha256",
            spec=ExperimentSpec(
                architecture="cnn",
                lead_months=1,
                train_years=10_000.0,
                common_period_max_lead_months=12,
                input_profile="core4",
            ),
            fit_inputs=np.asarray([1, 2, 3], dtype=np.int64),
            standardizer=standardizer,
            model=object(),
        )
        expected = AgopFactor(
            basis=np.eye(2, dtype=np.float64),
            root_eigenvalues=np.ones(2, dtype=np.float64),
            center=np.zeros(2, dtype=np.float64),
            reference_indices=experiment.fit_inputs,
            approximation_rank=2,
            solver_metadata={"method": "test"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            with patch(
                "zc_xai.fresh_figure7.build_exact_dense_agop_factor",
                return_value=expected,
            ) as build:
                actual, record = load_or_build_exact_factor(
                    data,
                    experiment,
                    cache,
                    gradient_batch_size=1024,
                    device="cpu",
                    retain_matrix_cache=False,
                )
            self.assertIs(actual, expected)
            self.assertFalse(record["cache_hit"])
            self.assertIsNone(build.call_args.kwargs["matrix_cache_path"])
            self.assertFalse((cache / "dense_agop_matrix.npy").exists())
            manifest = (cache / "full_eigensystem.npz.json").read_text()
            self.assertNotIn("matrix_file", manifest)
            self.assertNotIn("matrix_sha256", manifest)
            loaded, loaded_record = load_or_build_exact_factor(
                data,
                experiment,
                cache,
                gradient_batch_size=1024,
                device="cpu",
                retain_matrix_cache=False,
            )
            self.assertTrue(loaded_record["cache_hit"])
            np.testing.assert_array_equal(loaded.basis, expected.basis)
            np.testing.assert_array_equal(
                loaded.reference_indices,
                expected.reference_indices,
            )


if __name__ == "__main__":
    unittest.main()
