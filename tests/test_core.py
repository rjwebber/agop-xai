"""Fast checks for model definitions and revised XAI score calculations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zc_xai.artifacts import load_completed_bundle
from zc_xai.data import Standardizer, ZCData
from zc_xai.io import sha256_file, write_json
from zc_xai.models import build_model, parameter_count
from zc_xai.training import (
    ExperimentSpec,
    TrainingConfig,
    _comparison_inputs,
    experiment_directory,
    r2_score,
    resolved_patience,
)
from zc_xai.xai import (
    DEFAULT_ROBUSTNESS_NEIGHBOR_PERCENT,
    attribution_score,
    build_agop_factor,
    coherence_score,
    input_gradients,
    robustness_score,
    sample_empirical_neighbors,
    sensitivity_score,
)


class LinearSum(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.flatten(start_dim=1).sum(dim=1)


class TinyCoupledRegressor(nn.Module):
    """Small nonlinear model whose output couples every input coordinate."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mixing",
            torch.tensor(
                [
                    [0.3, -0.8, 0.1, 0.6],
                    [-0.4, 0.2, 0.9, -0.1],
                    [0.7, 0.5, -0.3, 0.4],
                    [-0.2, 0.6, 0.8, 0.3],
                    [0.9, -0.7, 0.2, 0.5],
                    [0.1, 0.4, -0.6, 0.8],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "readout",
            torch.tensor([0.5, -0.7, 0.2, 0.9], dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.flatten(start_dim=1)
        hidden = torch.tanh(flattened @ self.mixing)
        return (hidden * self.readout).sum(dim=1)


class TinyNeighborData:
    def __init__(self) -> None:
        self.values = np.arange(20, dtype=np.float32).reshape(5, 1, 2, 2)
        self.n_features = 4
        self.metadata_sha256 = "tiny-data"

    def load_inputs(
        self,
        indices: np.ndarray,
        *,
        standardizer: Standardizer,
    ) -> np.ndarray:
        return standardizer.transform(self.values[indices]).astype(np.float32)


class TinySplitData:
    n_time_steps = 100
    steps_per_month = 1

    def supervised_split(self, lead_months: int, *, train_fraction: float = 0.9):
        from zc_xai.data import ZCData

        return ZCData.supervised_split(  # type: ignore[arg-type]
            self,
            lead_months,
            train_fraction=train_fraction,
        )


class TinyDevelopmentData:
    steps_per_year = 10

    def development_selection(self, *args, **kwargs):
        return ZCData.development_selection(self, *args, **kwargs)  # type: ignore[arg-type]


class ModelTests(unittest.TestCase):
    def test_compact_channel_variant_counts_and_output_shapes(self) -> None:
        expected = {
            "core4": {"mlp": 236_601, "cnn": 217_917, "vit": 211_777},
            "core4_wp": {"mlp": 229_281, "cnn": 218_367, "vit": 212_161},
            "legacy10": {"mlp": 219_441, "cnn": 220_617, "vit": 214_081},
        }
        spatial_channels = {"core4": 4, "core4_wp": 5, "legacy10": 10}
        for profile, counts in expected.items():
            channels = spatial_channels[profile]
            inputs = torch.zeros(2, channels * 20 * 27 + 2)
            for architecture, count in counts.items():
                with self.subTest(profile=profile, architecture=architecture):
                    model = build_model(
                        architecture,
                        (channels, 20, 27),
                        phase_features=2,
                    )
                    self.assertEqual(parameter_count(model), count)
                    self.assertEqual(tuple(model(inputs).shape), (2,))

    def test_cnn_pooling_covers_grid_exactly(self) -> None:
        model = build_model("cnn", (4, 20, 27), phase_features=2)
        pooling_layers = [
            layer for layer in model.features if isinstance(layer, nn.AvgPool2d)
        ]
        self.assertEqual(len(pooling_layers), 2)
        first = pooling_layers[0](torch.ones(1, 1, 20, 27))
        second = pooling_layers[1](first)
        self.assertEqual(tuple(first.shape), (1, 1, 10, 9))
        self.assertEqual(tuple(second.shape), (1, 1, 5, 9))
        self.assertTrue(torch.equal(second, torch.ones_like(second)))

    def test_vit_patches_are_an_exact_permutation_of_the_grid(self) -> None:
        model = build_model("vit", (4, 20, 27), phase_features=2)
        fields = torch.arange(4 * 20 * 27, dtype=torch.float32).reshape(
            1, 4, 20, 27
        )
        patches = model.patches(fields)
        self.assertEqual(tuple(patches.shape), (1, 90, 24))
        self.assertEqual(
            torch.sort(patches.flatten()).values.tolist(),
            torch.sort(fields.flatten()).values.tolist(),
        )

    def test_phase_features_are_two_coordinates_with_finite_gradients(self) -> None:
        inputs = torch.randn(3, 4 * 20 * 27 + 2, requires_grad=True)
        self.assertEqual(inputs.shape[1], 2_162)
        for architecture in ("mlp", "cnn", "vit"):
            with self.subTest(architecture=architecture):
                model = build_model(
                    architecture,
                    (4, 20, 27),
                    phase_features=2,
                )
                gradient = torch.autograd.grad(model(inputs).sum(), inputs)[0]
                self.assertEqual(tuple(gradient[:, -2:].shape), (3, 2))
                self.assertTrue(torch.isfinite(gradient[:, -2:]).all())

    def test_phase_and_spatial_coordinates_have_equal_distance_weight(self) -> None:
        reference = torch.zeros(2_162)
        spatial_change = reference.clone()
        phase_change = reference.clone()
        spatial_change[0] = 1.0
        phase_change[-1] = 1.0
        self.assertEqual(
            torch.linalg.vector_norm(spatial_change - reference),
            torch.linalg.vector_norm(phase_change - reference),
        )

    def test_shared_default_patience(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.learning_rate, 1.0e-3)
        self.assertEqual(config.weight_decay, 1.0e-4)
        self.assertEqual(config.maximum_epochs, 100)
        for architecture in ("mlp", "cnn", "vit"):
            spec = ExperimentSpec(architecture, lead_months=10, train_years=50)
            self.assertEqual(resolved_patience(spec, config), 10)
        override = TrainingConfig(patience=7)
        spec = ExperimentSpec("vit", lead_months=10, train_years=50)
        self.assertEqual(resolved_patience(spec, override), 7)

    def test_input_profiles_have_distinct_artifact_paths(self) -> None:
        root = Path("artifacts")
        core = ExperimentSpec(
            "cnn", 10, 10_000, input_profile="core4"
        )
        legacy = ExperimentSpec(
            "cnn", 10, 10_000, input_profile="legacy10"
        )
        self.assertNotEqual(
            experiment_directory(root, core),
            experiment_directory(root, legacy),
        )
        self.assertIn("core4", experiment_directory(root, core).parts)

    def test_common_period_uses_identical_target_dates(self) -> None:
        data = TinySplitData()
        short_spec = ExperimentSpec(
            "cnn",
            lead_months=2,
            train_years=1,
            common_period_max_lead_months=5,
        )
        long_spec = ExperimentSpec(
            "cnn",
            lead_months=5,
            train_years=1,
            common_period_max_lead_months=5,
        )
        short_split, short_development, short_test = _comparison_inputs(
            data,
            short_spec,  # type: ignore[arg-type]
        )
        long_split, long_development, long_test = _comparison_inputs(
            data,
            long_spec,  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(
            short_development + short_split.lead_steps,
            long_development + long_split.lead_steps,
        )
        np.testing.assert_array_equal(
            short_test + short_split.lead_steps,
            long_test + long_split.lead_steps,
        )

    def test_validation_is_a_chronological_block_after_an_embargo(self) -> None:
        selection = TinyDevelopmentData().development_selection(
            np.arange(100, dtype=np.int64),
            train_years=5,
            validation_fraction=0.2,
            validation_embargo_steps=4,
            seed=7,
        )
        self.assertEqual(selection.development_inputs.size, 50)
        self.assertEqual(selection.fit_inputs.size, 36)
        self.assertEqual(selection.embargo_inputs.size, 4)
        self.assertEqual(selection.validation_inputs.size, 10)
        np.testing.assert_array_equal(
            np.concatenate(
                (
                    selection.fit_inputs,
                    selection.embargo_inputs,
                    selection.validation_inputs,
                )
            ),
            selection.development_inputs,
        )
        self.assertEqual(
            int(selection.validation_inputs[0] - selection.fit_inputs[-1]),
            5,
        )
        self.assertLess(
            int(selection.fit_inputs[-1] + selection.validation_embargo_steps),
            int(selection.validation_inputs[0]),
        )


class ArtifactTests(unittest.TestCase):
    def test_completed_bundle_validates_both_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            payload = directory / "values.dat"
            payload.write_bytes(b"validated payload")
            payload_hash = sha256_file(payload)
            metadata = {
                "artifact": "test bundle",
                "output_files": {
                    "values": {"file": payload.name, "sha256": payload_hash}
                },
            }
            metadata_path = directory / "metadata.json"
            write_json(metadata_path, metadata, overwrite=False)
            write_json(
                directory / "completed.json",
                {
                    "artifact": "test bundle",
                    "files": {
                        "values": payload_hash,
                        "metadata": sha256_file(metadata_path),
                    },
                },
                overwrite=False,
            )
            bundle = load_completed_bundle(
                directory,
                expected_artifact="test bundle",
                required_files=("values",),
            )
            self.assertEqual(bundle.files["values"], payload.resolve())
            payload.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_completed_bundle(
                    directory,
                    expected_artifact="test bundle",
                    required_files=("values",),
                )


class ScoreTests(unittest.TestCase):
    def test_canonical_robustness_neighborhood_is_one_percent(self) -> None:
        self.assertEqual(DEFAULT_ROBUSTNESS_NEIGHBOR_PERCENT, 1.0)
        sample = sample_empirical_neighbors(
            TinyNeighborData(),  # type: ignore[arg-type]
            Standardizer(
                mean=np.zeros((1, 2, 2), dtype=np.float32),
                scale=np.ones((1, 2, 2), dtype=np.float32),
                scale_floor=1.0e-6,
                count=5,
            ),
            query_index=2,
            candidate_indices=np.arange(5),
            n_samples=2,
            seed=7,
            distance_batch_size=2,
        )
        self.assertEqual(sample.neighbor_percent, 1.0)
        self.assertEqual(sample.neighborhood_count, 1)

    def test_sensitivity_and_attribution_for_linear_sum(self) -> None:
        query = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        direction = np.ones(4, dtype=np.float64) / 2.0
        gradient = np.ones(4, dtype=np.float64)
        self.assertAlmostEqual(sensitivity_score(gradient, direction), 1.0)
        score, ratio, _ = attribution_score(
            LinearSum(),
            query,
            direction,
            device="cpu",
        )
        self.assertAlmostEqual(score, 1.0, places=6)
        self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_coherence_extremes(self) -> None:
        constant = np.ones((1, 2, 2), dtype=np.float64)
        checkerboard = np.asarray([[[1.0, -1.0], [-1.0, 1.0]]])
        self.assertAlmostEqual(coherence_score(constant, constant.shape), 1.0)
        self.assertAlmostEqual(
            coherence_score(checkerboard, checkerboard.shape),
            -1.0,
        )

    def test_finite_population_robustness(self) -> None:
        target = np.asarray([1.0, 0.0])
        neighbors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        summary = robustness_score(target, neighbors, neighborhood_count=4)
        self.assertAlmostEqual(summary.score, 0.5)
        self.assertAlmostEqual(summary.sample_standard_deviation, 2**-0.5)
        self.assertAlmostEqual(summary.finite_population_standard_error, 8**-0.5)

    def test_one_draw_from_larger_population_has_undefined_uncertainty(self) -> None:
        summary = robustness_score(
            np.asarray([1.0, 0.0]),
            np.asarray([[1.0, 0.0]]),
            neighborhood_count=4,
        )
        self.assertTrue(np.isnan(summary.sample_standard_deviation))
        self.assertTrue(np.isnan(summary.finite_population_standard_error))

    def test_r2_definition(self) -> None:
        targets = np.asarray([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r2_score(targets, targets), 1.0)

    def test_neighbor_pool_and_sampling_are_deterministic(self) -> None:
        data = TinyNeighborData()
        standardizer = Standardizer(
            mean=np.zeros((1, 2, 2), dtype=np.float32),
            scale=np.ones((1, 2, 2), dtype=np.float32),
            scale_floor=1.0e-6,
            count=5,
        )
        first = sample_empirical_neighbors(
            data,  # type: ignore[arg-type]
            standardizer,
            query_index=2,
            candidate_indices=np.arange(5),
            neighbor_percent=50.0,
            n_samples=2,
            seed=7,
            distance_batch_size=2,
        )
        second = sample_empirical_neighbors(
            data,  # type: ignore[arg-type]
            standardizer,
            query_index=2,
            candidate_indices=np.arange(5),
            neighbor_percent=50.0,
            n_samples=2,
            seed=7,
            distance_batch_size=2,
        )
        np.testing.assert_array_equal(first.sampled_indices, second.sampled_indices)
        self.assertEqual(first.neighborhood_count, 2)
        self.assertNotIn(2, first.neighborhood_indices.tolist())

    def test_agop_gradient_cache_is_identity_bound(self) -> None:
        data = TinyNeighborData()
        standardizer = Standardizer(
            mean=np.zeros((1, 2, 2), dtype=np.float32),
            scale=np.ones((1, 2, 2), dtype=np.float32),
            scale_floor=1.0e-6,
            count=3,
        )
        references = np.asarray([0, 1, 2], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gradients.npy"
            build_agop_factor(
                LinearSum(),
                data,  # type: ignore[arg-type]
                standardizer,
                references,
                rank=2,
                gradient_batch_size=2,
                device="cpu",
                gradient_matrix_path=path,
                gradient_cache_identity={"checkpoint_sha256": "first"},
            )
            self.assertTrue(path.with_name(path.name + ".json").is_file())
            with self.assertRaisesRegex(ValueError, "identity does not match"):
                build_agop_factor(
                    LinearSum(),
                    data,  # type: ignore[arg-type]
                    standardizer,
                    references,
                    rank=2,
                    gradient_batch_size=2,
                    device="cpu",
                    gradient_matrix_path=path,
                    gradient_cache_identity={"checkpoint_sha256": "second"},
                )

    def test_agop_power_iterations_must_be_nonnegative(self) -> None:
        with self.assertRaisesRegex(ValueError, "power_iterations"):
            build_agop_factor(
                LinearSum(),
                TinyNeighborData(),  # type: ignore[arg-type]
                Standardizer(
                    mean=np.zeros((1, 2, 2), dtype=np.float32),
                    scale=np.ones((1, 2, 2), dtype=np.float32),
                    scale_floor=1.0e-6,
                    count=3,
                ),
                np.asarray([0, 1, 2], dtype=np.int64),
                rank=2,
                gradient_batch_size=2,
                device="cpu",
                power_iterations=-1,
            )


class GradientTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> np.ndarray:
        rng = np.random.default_rng(1729)
        return rng.normal(size=(17, 1, 2, 3)).astype(np.float32)

    def test_batched_cpu_gradients_match_single_example_gradients(self) -> None:
        inputs = self._inputs()
        model = TinyCoupledRegressor()
        batched = input_gradients(model, inputs, batch_size=17, device="cpu")
        one_at_a_time = np.concatenate(
            [
                input_gradients(
                    model,
                    inputs[position : position + 1],
                    batch_size=1,
                    device="cpu",
                )
                for position in range(inputs.shape[0])
            ],
            axis=0,
        )
        np.testing.assert_allclose(batched, one_at_a_time, rtol=1.0e-6, atol=1.0e-7)

    @unittest.skipUnless(
        torch.backends.mps.is_available(),
        "Apple MPS is not available on this machine.",
    )
    def test_batched_mps_gradients_match_cpu(self) -> None:
        inputs = self._inputs()
        model = TinyCoupledRegressor()
        cpu = input_gradients(model, inputs, batch_size=17, device="cpu")
        mps = input_gradients(model, inputs, batch_size=17, device="mps")
        np.testing.assert_allclose(mps, cpu, rtol=5.0e-5, atol=5.0e-6)


if __name__ == "__main__":
    unittest.main()
