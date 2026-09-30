"""Lightweight invariants for fresh XAI/Figures 7--9 orchestration."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests import test_fresh_zc_pipeline
from zc_xai.data import ZCData
from zc_xai.fresh_case_studies import (
    FreshXAISettings,
    build_explainers,
    common_fixed_test_targets,
    exhaustive_nearest_neighbors,
    expected_gradient_references,
    fixed_test_extreme,
    spatial_part,
    target_locked_event,
)
from zc_xai.fresh_figure7 import load_common_target_cnn
from zc_xai.xai import AgopFactor

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FreshCaseOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = test_fresh_zc_pipeline.FreshZCDataTests()
        self.fixture.setUp()
        self.data = ZCData(self.fixture.directory, input_profile="core4")

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def test_spatial_prefix_excludes_two_phase_scalars(self) -> None:
        values = self.data.load_inputs(np.asarray([4], dtype=np.int64))[0]
        spatial = spatial_part(self.data, values)
        self.assertEqual(spatial.shape, (4, 20, 27))
        self.assertEqual(spatial.size + 2, values.size)
        self.assertFalse(np.any(spatial == values[-1]))

    def test_fixed_extremes_and_target_lock_use_only_fixed_test_pairs(self) -> None:
        warm = fixed_test_extreme(self.data, lead_months=1, kind="maximum")
        cold = fixed_test_extreme(self.data, lead_months=1, kind="minimum")
        self.assertEqual((warm.input_step, warm.target_step), (10, 11))
        self.assertEqual((cold.input_step, cold.target_step), (9, 10))
        locked = target_locked_event(
            self.data,
            case_id="locked",
            lead_months=2,
            target_step=11,
            selection_rule="test",
        )
        self.assertEqual((locked.input_step, locked.target_step), (9, 11))
        with self.assertRaisesRegex(ValueError, "not eligible"):
            target_locked_event(
                self.data,
                case_id="bad",
                lead_months=2,
                target_step=10,
                selection_rule="test",
            )

    def test_common_target_dates_are_the_intersection_not_shifted_inputs(self) -> None:
        targets = common_fixed_test_targets(self.data, [1, 2])
        np.testing.assert_array_equal(targets, np.asarray([11], dtype=np.int64))

    def test_exhaustive_neighbor_population_is_not_randomly_subsampled(self) -> None:
        standardizer = self.data.compute_standardizer(np.arange(6), batch_size=3)
        experiment = SimpleNamespace(
            standardizer=standardizer,
            fit_inputs=np.arange(6, dtype=np.int64),
            development_inputs=np.arange(9, dtype=np.int64),
        )
        settings = FreshXAISettings(neighbor_percent=50.0, distance_batch_size=3)
        sample = exhaustive_nearest_neighbors(
            self.data,
            experiment,
            query_index=10,
            settings=settings,
        )
        self.assertEqual(sample.neighborhood_count, 3)
        self.assertEqual(sample.sampled_indices.size, sample.neighborhood_count)
        self.assertTrue(np.all(np.isin(sample.sampled_indices, experiment.fit_inputs)))
        self.assertFalse(np.any(np.isin(sample.sampled_indices, np.arange(6, 9))))
        np.testing.assert_array_equal(
            sample.sampled_indices,
            sample.neighborhood_indices,
        )
        np.testing.assert_array_equal(
            sample.sampled_rms_distances,
            sample.neighborhood_rms_distances,
        )

    def test_expected_gradient_references_are_distinct_and_reproducible(self) -> None:
        candidates = np.arange(20, dtype=np.int64)
        first_indices, first_alphas = expected_gradient_references(
            candidates, count=8, seed=42
        )
        second_indices, second_alphas = expected_gradient_references(
            candidates, count=8, seed=42
        )
        np.testing.assert_array_equal(first_indices, second_indices)
        np.testing.assert_array_equal(first_alphas, second_alphas)
        self.assertEqual(np.unique(first_indices).size, 8)
        self.assertTrue(np.all((first_alphas >= 0.0) & (first_alphas <= 1.0)))

    def test_explainers_draw_expected_gradient_baselines_from_training_only(
        self,
    ) -> None:
        fit_inputs = np.arange(6, dtype=np.int64)
        experiment = SimpleNamespace(
            fit_inputs=fit_inputs,
            development_inputs=np.arange(9, dtype=np.int64),
            standardizer=self.data.compute_standardizer(fit_inputs, batch_size=3),
            model=object(),
        )
        factor = AgopFactor(
            basis=np.empty((0, 0)),
            root_eigenvalues=np.empty(0),
            center=np.empty(0),
            reference_indices=fit_inputs,
            approximation_rank=self.data.n_features,
        )
        _explainers, references = build_explainers(
            self.data,
            experiment,
            factor,
            FreshXAISettings(expected_gradients_samples=4),
            device="cpu",
        )
        chosen = references["expected_gradient_background_indices"]
        self.assertEqual(np.unique(chosen).size, 4)
        self.assertTrue(np.all(np.isin(chosen, fit_inputs)))
        self.assertFalse(np.any(np.isin(chosen, np.arange(6, 9))))

    def test_command_defaults_fix_final_numerical_choices(self) -> None:
        figures8_9 = _load_script("generate_fresh_figures8_9_data.py")
        figure7 = _load_script("generate_fresh_figure7_data.py")
        figures8_9_args = figures8_9.build_parser().parse_args([])
        figure7_args = figure7.build_parser().parse_args([])
        self.assertEqual(figures8_9_args.data_dir, Path("data/processed/zc-v3"))
        self.assertEqual(figure7_args.lead_months, list(range(1, 13)))
        settings = FreshXAISettings()
        self.assertEqual(settings.neighbor_percent, 1.0)
        self.assertEqual(settings.integrated_gradients_steps, 1024)
        self.assertEqual(settings.expected_gradients_samples, 1024)
        self.assertEqual(settings.gradient_batch_size, 1024)

    def test_figure7_loader_never_silently_trains_a_missing_model(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Missing full-size"):
            load_common_target_cnn(
                self.data,
                self.fixture.directory / "missing-artifacts",
                lead_months=1,
                seed=42,
                maximum_lead_months=12,
            )


if __name__ == "__main__":
    unittest.main()
