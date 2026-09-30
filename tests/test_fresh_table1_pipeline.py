"""Contract tests for the fresh core4 Table I and Figure 6/7 pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from scripts.generate_fresh_table1_data import (
    _agop_directory,
    _validate_agop_gradient_batch,
    build_parser,
)
from zc_xai.fresh_xai_outputs import (
    FRESH_EXPECTED_GRADIENT_COUNT,
    FRESH_IG_GRADIENT_COUNT,
    FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
    FRESH_XAI_GRADIENT_BATCH_SIZE,
    exact_robustness_summary,
    phase_coefficient_summary,
    spatial_coherence,
    split_spatial_phase,
    validate_unit_explanation,
)


class FreshTable1PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = SimpleNamespace(
            n_phase_features=2,
            spatial_input_shape=(4, 2, 3),
        )

    def test_scientific_configuration_is_fixed_at_production_values(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.lead_months, 10)
        self.assertEqual(args.architectures, ("mlp", "cnn", "vit"))
        self.assertEqual(FRESH_XAI_GRADIENT_BATCH_SIZE, 1024)
        self.assertEqual(FRESH_IG_GRADIENT_COUNT, 1024)
        self.assertEqual(FRESH_EXPECTED_GRADIENT_COUNT, 1024)
        self.assertEqual(FRESH_ROBUSTNESS_NEIGHBOR_PERCENT, 1.0)

    def test_agop_cache_prefers_explicit_production_batch_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "core4-cnn-lead-10m-seed-000042-refs-all"
            explicit = root / (
                "core4-cnn-lead-10m-seed-000042-refs-all-batch-1024"
            )
            legacy.mkdir()
            self.assertEqual(_agop_directory(root, "cnn", 10, 42), legacy)
            explicit.mkdir()
            self.assertEqual(_agop_directory(root, "cnn", 10, 42), explicit)

    def test_agop_cache_rejects_a_mismatched_recorded_batch(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report_path = directory / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "run_identity": {
                            "numeric_method": {"gradient_batch_size": 256}
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "batch size 1024"):
                _validate_agop_gradient_batch(directory)
            report_path.write_text(
                json.dumps(
                    {
                        "run_identity": {
                            "numeric_method": {"gradient_batch_size": 1024}
                        }
                    }
                ),
                encoding="utf-8",
            )
            _validate_agop_gradient_batch(directory)

    def test_phase_is_separated_reported_and_excluded_from_coherence(self) -> None:
        values = np.concatenate((np.ones(24), np.asarray([3.0, -4.0])))
        spatial, phase = split_spatial_phase(values, self.data)
        self.assertEqual(spatial.shape, (4, 2, 3))
        np.testing.assert_array_equal(phase, np.asarray([3.0, -4.0]))
        summary = phase_coefficient_summary(values, self.data)
        self.assertEqual(summary["annual_phase_sin_coefficient"], 3.0)
        self.assertEqual(summary["annual_phase_cos_coefficient"], -4.0)
        self.assertAlmostEqual(
            summary["annual_phase_squared_mass_fraction"], 25.0 / 49.0
        )
        self.assertAlmostEqual(spatial_coherence(values, self.data), 1.0)
        changed_phase = values.copy()
        changed_phase[-2:] = (3000.0, -4000.0)
        self.assertEqual(
            spatial_coherence(changed_phase, self.data),
            spatial_coherence(values, self.data),
        )

    def test_exact_robustness_has_zero_sampling_error(self) -> None:
        summary = exact_robustness_summary(np.asarray([0.2, 0.4, 0.6]))
        self.assertAlmostEqual(summary["robustness"], 0.4)
        self.assertAlmostEqual(summary["explanation_change"], 0.6)
        self.assertEqual(summary["finite_population_standard_error"], 0.0)
        self.assertEqual(summary["evaluated_count"], 3)
        self.assertTrue(summary["exhaustive"])

    def test_unit_explanation_validation_includes_phase_dimensions(self) -> None:
        values = np.zeros(26)
        values[-1] = 1.0
        validated = validate_unit_explanation(values, 26)
        np.testing.assert_array_equal(validated, values)
        with self.assertRaisesRegex(ValueError, "not unit norm"):
            validate_unit_explanation(2.0 * values, 26)


if __name__ == "__main__":
    unittest.main()
