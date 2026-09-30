"""Focused tests for fresh XAI workload and fused streaming benchmarks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.benchmark_fresh_xai_methods import (
    FusedGradientShapExplainer,
    FusedIntegratedGradientsExplainer,
    _canonicalize_execution_modes,
    _fixed_expected_gradient_references,
    build_parser,
    fused_expected_gradients,
    fused_integrated_gradients,
    load_or_compute_npz,
    method_variants,
    method_workload,
)
from zc_xai.xai import GradientShapExplainer, IntegratedGradientsExplainer


class SmoothModel(nn.Module):
    def __init__(self, coefficients: np.ndarray) -> None:
        super().__init__()
        self.register_buffer(
            "coefficients",
            torch.as_tensor(coefficients, dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (
            self.coefficients * inputs.square()
            + 0.3 * self.coefficients * inputs
        ).sum(dim=1)


class FreshXaiMethodBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(51)
        self.inputs = rng.normal(size=(4, 6)).astype(np.float32)
        self.backgrounds = rng.normal(size=(7, 6)).astype(np.float32)
        self.alphas = rng.uniform(size=7).astype(np.float32)
        self.model = SmoothModel(
            np.asarray([0.5, -1.0, 2.0, 0.25, 1.25, -0.75])
        )

    def test_fused_ig_matches_current_per_query_explainer(self) -> None:
        current = IntegratedGradientsExplainer(
            self.model,
            n_steps=11,
            gradient_batch_size=3,
            device="cpu",
        ).explain(self.inputs)
        fused = FusedIntegratedGradientsExplainer(
            self.model,
            n_steps=11,
            pair_batch_size=5,
            device="cpu",
        ).explain(self.inputs)
        np.testing.assert_allclose(fused, current, rtol=2.0e-7, atol=2.0e-7)

    def test_ig_uses_all_1024_right_endpoints(self) -> None:
        query = np.asarray([[2.0, -1.0]], dtype=np.float32)
        seen: list[np.ndarray] = []

        def gradients(
            _model: nn.Module,
            inputs: np.ndarray,
            *,
            batch_size: int,
            device: str,
        ) -> np.ndarray:
            self.assertEqual(batch_size, 1024)
            self.assertEqual(device, "cpu")
            seen.append(inputs.copy())
            return np.ones_like(inputs)

        with patch("zc_xai.xai.input_gradients", side_effect=gradients):
            IntegratedGradientsExplainer(
                self.model,
                n_steps=1024,
                gradient_batch_size=1024,
                device="cpu",
            ).explain(query)
        path = seen[0]
        self.assertEqual(path.shape, (1024, 2))
        np.testing.assert_allclose(path[0], query[0] / 1024.0)
        np.testing.assert_array_equal(path[-1], query[0])

    def test_fused_expected_gradients_matches_current_explainer(self) -> None:
        current = GradientShapExplainer(
            self.model,
            backgrounds=self.backgrounds,
            alphas=self.alphas,
            gradient_batch_size=3,
            device="cpu",
        ).explain(self.inputs)
        fused = FusedGradientShapExplainer(
            self.model,
            backgrounds=self.backgrounds,
            alphas=self.alphas,
            pair_batch_size=5,
            device="cpu",
        ).explain(self.inputs)
        np.testing.assert_allclose(fused, current, rtol=2.0e-7, atol=2.0e-7)

    def test_fused_implementations_never_materialize_cartesian_population(
        self,
    ) -> None:
        seen_batch_sizes: list[int] = []

        def gradients(
            _model: nn.Module,
            inputs: np.ndarray,
            *,
            batch_size: int,
            device: str,
        ) -> np.ndarray:
            self.assertEqual(batch_size, 5)
            self.assertEqual(device, "cpu")
            seen_batch_sizes.append(inputs.shape[0])
            return np.ones_like(inputs)

        with patch(
            "scripts.benchmark_fresh_xai_methods.input_gradients",
            side_effect=gradients,
        ):
            fused_integrated_gradients(
                self.model,
                self.inputs[:3],
                n_steps=7,
                pair_batch_size=5,
                device="cpu",
            )
        self.assertEqual(len(seen_batch_sizes), 5)
        self.assertLessEqual(max(seen_batch_sizes), 5)
        self.assertEqual(sum(seen_batch_sizes), 21)

        seen_batch_sizes.clear()
        with patch(
            "scripts.benchmark_fresh_xai_methods.input_gradients",
            side_effect=gradients,
        ):
            fused_expected_gradients(
                self.model,
                self.inputs[:3],
                backgrounds=self.backgrounds,
                alphas=self.alphas,
                pair_batch_size=5,
                device="cpu",
            )
        self.assertEqual(len(seen_batch_sizes), 5)
        self.assertLessEqual(max(seen_batch_sizes), 5)
        self.assertEqual(sum(seen_batch_sizes), 21)

    def test_workloads_count_true_per_query_and_fused_batches(self) -> None:
        common = {
            "query_count": 256,
            "gradient_batch_size": 64,
            "fused_pair_batch_size": 256,
            "ig_steps": 1024,
            "gradient_shap_samples": 1024,
        }
        ig_current = method_workload("IG", execution_mode="current", **common)
        ig_fused = method_workload("IG", execution_mode="fused", **common)
        shap_current = method_workload(
            "GradientSHAP", execution_mode="current", **common
        )
        shap_fused = method_workload(
            "GradientSHAP", execution_mode="fused", **common
        )
        self.assertEqual(ig_current["gradient_rows"], 262_144)
        self.assertEqual(ig_current["gradient_batches"], 4_096)
        self.assertEqual(ig_current["input_gradients_calls"], 256)
        self.assertEqual(ig_fused["gradient_rows"], 262_144)
        self.assertEqual(ig_fused["gradient_batches"], 1_024)
        self.assertEqual(ig_fused["nominal_gradient_batches_per_query"], 4)
        self.assertEqual(shap_current["gradient_rows"], 262_144)
        self.assertEqual(shap_current["gradient_batches"], 4_096)
        self.assertEqual(shap_fused["gradient_rows"], 262_144)
        self.assertEqual(shap_fused["gradient_batches"], 1_024)
        self.assertEqual(shap_fused["nominal_gradient_batches_per_query"], 4)

        target_common = {**common, "query_count": 1}
        for method in ("IG", "GradientSHAP"):
            neighbor = method_workload(
                method, execution_mode="fused", **common
            )
            target = method_workload(
                method, execution_mode="fused", **target_common
            )
            self.assertEqual(
                target["gradient_rows"] + neighbor["gradient_rows"],
                263_168,
            )
            self.assertEqual(
                target["gradient_batches"] + neighbor["gradient_batches"],
                1_028,
            )

    def test_production_defaults_are_exact_and_custom_counts_remain_available(
        self,
    ) -> None:
        production = build_parser().parse_args([])
        _canonicalize_execution_modes(production)
        self.assertEqual(production.ig_steps, 1024)
        self.assertEqual(production.gradient_shap_samples, 1024)
        self.assertEqual(production.fused_pair_batch_size, 256)
        self.assertEqual(
            method_variants(production.execution_modes),
            (
                "GRAD",
                "IG-current",
                "IG-fused",
                "GradientSHAP-current",
                "GradientSHAP-fused",
                "AGOP",
            ),
        )

        custom = build_parser().parse_args(
            [
                "--ig-steps",
                "300",
                "--gradient-shap-samples",
                "1000",
                "--execution-modes",
                "fused",
            ]
        )
        _canonicalize_execution_modes(custom)
        self.assertEqual(custom.ig_steps, 300)
        self.assertEqual(custom.gradient_shap_samples, 1000)
        self.assertEqual(custom.execution_modes, ("fused",))

        high_sample = build_parser().parse_args(
            [
                "--ig-steps",
                "1000",
                "--gradient-shap-samples",
                "10000",
            ]
        )
        self.assertEqual(high_sample.ig_steps, 1000)
        self.assertEqual(high_sample.gradient_shap_samples, 10000)

    def test_expected_gradient_pairs_are_distinct_fixed_common_random_numbers(
        self,
    ) -> None:
        candidates = np.arange(5000, dtype=np.int64)
        first = _fixed_expected_gradient_references(
            candidates,
            count=1024,
            seed=42,
        )
        second = _fixed_expected_gradient_references(
            candidates,
            count=1024,
            seed=42,
        )
        indices = first["background_indices"]
        alphas = first["alphas"]
        self.assertEqual(indices.size, 1024)
        self.assertEqual(np.unique(indices).size, 1024)
        self.assertTrue(np.all((alphas >= 0.0) & (alphas <= 1.0)))
        np.testing.assert_array_equal(indices, second["background_indices"])
        np.testing.assert_array_equal(alphas, second["alphas"])

    def test_atomic_stage_cache_resumes_and_rejects_wrong_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stage.npz"
            calls = 0

            def compute() -> dict[str, np.ndarray]:
                nonlocal calls
                calls += 1
                return {"values": np.arange(6, dtype=np.float32).reshape(2, 3)}

            first, first_record = load_or_compute_npz(
                path,
                stage="fixture",
                identity={"version": 1},
                compute=compute,
                overwrite=False,
            )
            second, second_record = load_or_compute_npz(
                path,
                stage="fixture",
                identity={"version": 1},
                compute=compute,
                overwrite=False,
            )
            self.assertEqual(calls, 1)
            self.assertFalse(first_record["cache_hit"])
            self.assertTrue(second_record["cache_hit"])
            np.testing.assert_array_equal(first["values"], second["values"])
            with self.assertRaisesRegex(ValueError, "another identity"):
                load_or_compute_npz(
                    path,
                    stage="fixture",
                    identity={"version": 2},
                    compute=compute,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
