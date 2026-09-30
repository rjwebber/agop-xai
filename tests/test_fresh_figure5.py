"""Tests for the fixed-block fresh Figure 5 training contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_figure5_experiments import build_parser
from tests.test_fresh_zc_pipeline import FreshZCDataTests
from zc_xai.data import ZCData
from zc_xai.figure5 import select_figure5_inputs, train_figure5_experiment
from zc_xai.training import ExperimentSpec, TrainingConfig


class FreshFigure5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = FreshZCDataTests()
        self.fixture.setUp()
        self.data = ZCData(self.fixture.directory, input_profile="core4")

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def test_runner_exposes_canonical_learning_rate(self) -> None:
        arguments = build_parser().parse_args(
            ["--data-dir", "data", "--output", "figure5.csv"]
        )
        self.assertEqual(arguments.learning_rate, 1.0e-3)

    def test_common_lead_uses_identical_target_dates_in_every_block(self) -> None:
        lead_one = ExperimentSpec(
            "mlp",
            lead_months=1,
            train_years=10_000,
            seed=7,
            common_period_max_lead_months=2,
            input_profile="core4",
        )
        lead_two = ExperimentSpec(
            "mlp",
            lead_months=2,
            train_years=10_000,
            seed=7,
            common_period_max_lead_months=2,
            input_profile="core4",
        )
        first = select_figure5_inputs(self.data, lead_one)
        second = select_figure5_inputs(self.data, lead_two)
        for first_inputs, second_inputs in (
            (first.fit_inputs, second.fit_inputs),
            (first.validation_inputs, second.validation_inputs),
            (first.test_inputs, second.test_inputs),
        ):
            np.testing.assert_array_equal(first_inputs + 1, second_inputs + 2)
        np.testing.assert_array_equal(first.fit_inputs, np.arange(1, 5))
        np.testing.assert_array_equal(first.standardization_inputs, np.arange(6))
        np.testing.assert_array_equal(first.validation_inputs, np.asarray([7]))
        np.testing.assert_array_equal(first.test_inputs, np.asarray([10]))

    def test_subset_training_uses_fixed_validation_and_test_and_resumes(self) -> None:
        spec = ExperimentSpec(
            "mlp",
            lead_months=1,
            train_years=0.25,
            seed=3,
            input_profile="core4",
        )
        config = TrainingConfig(
            batch_size=2,
            maximum_epochs=1,
            patience=1,
            statistics_batch_size=2,
        )
        selection = select_figure5_inputs(self.data, spec)
        self.assertEqual(selection.fit_inputs.size, 3)
        np.testing.assert_array_equal(
            selection.standardization_inputs,
            selection.fit_inputs,
        )
        np.testing.assert_array_equal(selection.validation_inputs, np.arange(6, 8))
        np.testing.assert_array_equal(selection.test_inputs, np.arange(9, 11))

        with tempfile.TemporaryDirectory() as artifact_directory:
            root = Path(artifact_directory)
            first = train_figure5_experiment(
                self.data,
                root,
                spec,
                config,
                device="cpu",
            )
            second = train_figure5_experiment(
                self.data,
                root,
                spec,
                config,
                device="cpu",
            )
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.checkpoint_sha256, second.checkpoint_sha256)
        self.assertEqual(first.metrics["selection"]["validation_years"], 2 / 12)
        self.assertEqual(first.metrics["selection"]["test_years"], 2 / 12)
        self.assertEqual(first.metrics["selection"]["window_offset"], 2)


if __name__ == "__main__":
    unittest.main()
