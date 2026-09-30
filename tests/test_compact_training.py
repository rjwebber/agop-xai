"""Round-trip checks for compact models on the fresh fixed-block contract."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from tests.test_fresh_zc_pipeline import FreshZCDataTests
from zc_xai.data import ZCData
from zc_xai.training import (
    FIXED_STANDARDIZATION_POPULATION,
    FIXED_VALIDATION_STRATEGY,
    ExperimentSpec,
    TrainingConfig,
    load_experiment,
    train_experiment,
)


class CompactTrainingRoundTripTests(unittest.TestCase):
    def test_fresh_fixed_blocks_and_all_training_state_normalization(self) -> None:
        fixture = FreshZCDataTests()
        fixture.setUp()
        try:
            data = ZCData(fixture.directory, input_profile="core4")
            spec = ExperimentSpec(
                "mlp",
                lead_months=1,
                train_years=10_000,
                seed=7,
                input_profile="core4",
            )
            config = TrainingConfig(
                batch_size=2,
                maximum_epochs=1,
                patience=1,
                statistics_batch_size=2,
            )
            with tempfile.TemporaryDirectory() as artifact_directory:
                trained = train_experiment(
                    data,
                    Path(artifact_directory),
                    spec,
                    config,
                    device="cpu",
                )
                loaded = load_experiment(
                    data,
                    Path(artifact_directory),
                    spec,
                    device="cpu",
                )
            np.testing.assert_array_equal(trained.fit_inputs, np.arange(5))
            np.testing.assert_array_equal(trained.validation_inputs, np.arange(6, 8))
            np.testing.assert_array_equal(trained.test_inputs, np.arange(9, 11))
            np.testing.assert_array_equal(
                loaded.standardization_inputs,
                np.arange(6),
            )
            self.assertEqual(loaded.standardizer.count, 6)
            self.assertEqual(
                loaded.metrics["standardization_population"],
                FIXED_STANDARDIZATION_POPULATION,
            )
            self.assertEqual(
                loaded.metrics["validation_strategy"],
                FIXED_VALIDATION_STRATEGY,
            )
            self.assertIn("core4", loaded.artifact_dir.parts)
        finally:
            fixture.tearDown()

    def test_fresh_training_rejects_ignored_fractional_options(self) -> None:
        fixture = FreshZCDataTests()
        fixture.setUp()
        try:
            data = ZCData(fixture.directory, input_profile="core4")
            base = ExperimentSpec(
                "mlp",
                lead_months=1,
                train_years=10_000,
                input_profile="core4",
            )
            invalid = (
                replace(base, train_fraction=0.8),
                replace(base, validation_fraction=0.1),
                replace(base, common_period_max_lead_months=2),
            )
            with tempfile.TemporaryDirectory() as artifact_directory:
                for spec in invalid:
                    with self.subTest(spec=spec), self.assertRaises(ValueError):
                        train_experiment(
                            data,
                            Path(artifact_directory),
                            spec,
                            TrainingConfig(maximum_epochs=1),
                            device="cpu",
                        )
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
