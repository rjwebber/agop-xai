"""Focused tests for method-specific direct ZC XAI targets and wrapper policy."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from zc_xai.direct_xai_target import (
    GRADIENT_SHAP_ALPHA_SEED,
    GRADIENT_SHAP_BACKGROUND_SEED,
    GRADIENT_SHAP_SAMPLES,
    _method_explainer,
    build_direct_xai_target,
)

sys.path.insert(0, "scripts")
import run_zc_pooled_xai_covariance_action as wrapper  # noqa: E402


class _DummyExplainer:
    def __init__(self, explanation: np.ndarray) -> None:
        self.explanation = np.asarray(explanation, dtype=np.float64)

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        return np.repeat(self.explanation[None], inputs.shape[0], axis=0)


class _LinearForecast(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.arange(1, values.shape[-1] + 1, device=values.device)
        return (values * weights).sum(dim=-1)


class _DummyData:
    input_shape = (4,)
    n_phase_features = 2

    def __init__(self) -> None:
        self.requested: np.ndarray | None = None

    def load_inputs(self, indices: np.ndarray, *, standardizer: object) -> np.ndarray:
        del standardizer
        self.requested = np.asarray(indices, dtype=np.int64).copy()
        values = self.requested.astype(np.float32)
        return np.stack((values, values + 1, values + 2, values + 3), axis=1)


class DirectXAITargetTests(unittest.TestCase):
    def test_phase_is_removed_and_negative_event_projection_is_oriented(self) -> None:
        data = _DummyData()
        event = np.asarray((2.0, 0.0, 5.0, -5.0))
        raw = np.asarray((-1.0, 0.0, 100.0, -100.0))
        with patch(
            "zc_xai.direct_xai_target._method_explainer",
            return_value=(_DummyExplainer(raw), {"definition": "test"}, {}),
        ):
            target = build_direct_xai_target(
                "GRAD",
                data=data,  # type: ignore[arg-type]
                experiment=SimpleNamespace(),  # type: ignore[arg-type]
                event_standardized=event,
            )
        np.testing.assert_array_equal(target.fixed_phase_explanation[-2:], 0.0)
        np.testing.assert_allclose(target.raw_spatial_direction, (-1.0, 0.0))
        np.testing.assert_allclose(target.oriented_spatial_direction, (1.0, 0.0))
        self.assertEqual(target.orientation_multiplier, -1.0)
        self.assertEqual(target.raw_event_projection, -2.0)
        self.assertEqual(target.oriented_event_projection, 2.0)

    def test_gradientshap_references_are_distinct_training_only_seed_42(self) -> None:
        data = _DummyData()
        fit_inputs = np.arange(100, 1300, dtype=np.int64)
        experiment = SimpleNamespace(
            fit_inputs=fit_inputs,
            standardizer=object(),
            model=_LinearForecast(),
        )
        _, provenance, arrays = _method_explainer(
            "GradientSHAP",
            data=data,  # type: ignore[arg-type]
            experiment=experiment,  # type: ignore[arg-type]
            agop_factor=None,
            device="cpu",
        )
        indices = arrays["gradientshap_background_indices"]
        alphas = arrays["gradientshap_alphas"]
        expected_indices = np.random.default_rng(GRADIENT_SHAP_BACKGROUND_SEED).choice(
            fit_inputs, size=GRADIENT_SHAP_SAMPLES, replace=False
        )
        expected_alphas = (
            np.random.default_rng(GRADIENT_SHAP_ALPHA_SEED)
            .uniform(0.0, 1.0, size=GRADIENT_SHAP_SAMPLES)
            .astype(np.float32)
        )
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(alphas, expected_alphas)
        self.assertEqual(np.unique(indices).size, GRADIENT_SHAP_SAMPLES)
        self.assertTrue(np.isin(indices, fit_inputs).all())
        self.assertEqual(
            provenance["reference_population"], "experiment.fit_inputs only"
        )
        self.assertEqual(provenance["background_seed"], 42)
        self.assertEqual(provenance["alpha_seed"], 43)

    def test_wrapper_fixes_pooled_seed42_cohort_and_output_location(self) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--xai-method",
                    "GRAD",
                    "--target-event",
                    "extreme_la_nina",
                    "--member",
                    "399086",
                    "--maximum-iterations",
                    "1",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        self.assertEqual(values[values.index("--case") + 1], "pooled")
        self.assertEqual(values[values.index("--selection-seed") + 1], "42")
        self.assertEqual(values[values.index("--trajectory") + 1], "uniform-cohort")
        parsed = wrapper.direct.parser().parse_args(values)
        self.assertEqual(
            parsed.covariance_policy,
            wrapper.direct.ANNUAL_SHARED_COVARIANCE_POLICY,
        )
        self.assertIsNone(parsed.covariance_manifest)
        output = values[values.index("--output-dir") + 1]
        self.assertTrue(output.endswith("/grad/extreme-la-nina"))
        self.assertIn("--maximum-iterations", values)

    def test_wrapper_rejects_scientific_policy_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed by the pooled"):
            wrapper.main(
                [
                    "--xai-method",
                    "IG",
                    "--target-event",
                    "extreme_el_nino",
                    "--selection-seed=7",
                ]
            )

    def test_wrapper_coherent_adaptive_preset_is_explicit_and_exact(self) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--xai-method",
                    "IG",
                    "--target-event",
                    "extreme_el_nino",
                    "--coherent-adaptive",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        expected = wrapper.COHERENT_ADAPTIVE_PRESET
        self.assertEqual(
            values[values.index("--maximum-iterations") + 1],
            str(expected["maximum_iterations"]),
        )
        self.assertEqual(
            values[values.index("--continuation-fractions") + 1],
            expected["continuation_fractions"],
        )
        self.assertIn("--scale-continuation-warm-start", values)
        self.assertIn("--adaptive-continuation", values)
        self.assertIn("--zero-dual-restart-after-warm-rejection", values)
        self.assertNotIn("--direct-exact-target-first", values)
        self.assertEqual(
            values[values.index("--adaptive-minimum-fraction-step") + 1],
            str(expected["adaptive_minimum_fraction_step"]),
        )
        self.assertEqual(
            values[values.index("--maximum-adaptive-subdivisions") + 1],
            str(expected["maximum_adaptive_subdivisions"]),
        )

    def test_wrapper_coherent_preset_forwards_exact_cache_budget(self) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--xai-method",
                    "AGOP",
                    "--target-event",
                    "extreme_el_nino",
                    "--coherent-adaptive",
                    "--covariance-centered-cache-gib",
                    "24",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        position = values.index("--covariance-centered-cache-gib")
        self.assertEqual(values[position + 1], "24")

    def test_wrapper_coherent_adaptive_rejects_partial_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            wrapper.main(
                [
                    "--xai-method",
                    "GradientSHAP",
                    "--target-event",
                    "extreme_la_nina",
                    "--coherent-adaptive",
                    "--maximum-iterations",
                    "41",
                ]
            )

    def test_wrapper_coherent_direct_first_adaptive_preset_expands_exactly(
        self,
    ) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--xai-method",
                    "GRAD",
                    "--target-event",
                    "extreme_la_nina",
                    "--output-root",
                    "outputs/fresh-direct-first",
                    "--coherent-direct-first-adaptive",
                    "--covariance-centered-cache-gib",
                    "24",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        expected = wrapper.COHERENT_DIRECT_FIRST_ADAPTIVE_PRESET
        self.assertEqual(
            values[values.index("--maximum-iterations") + 1],
            str(expected["maximum_iterations"]),
        )
        self.assertEqual(
            values[values.index("--continuation-fractions") + 1],
            expected["continuation_fractions"],
        )
        self.assertIn("--scale-continuation-warm-start", values)
        self.assertIn("--adaptive-continuation", values)
        self.assertIn("--zero-dual-restart-after-warm-rejection", values)
        self.assertIn("--direct-exact-target-first", values)
        self.assertEqual(
            values[values.index("--adaptive-minimum-fraction-step") + 1],
            str(expected["adaptive_minimum_fraction_step"]),
        )
        self.assertEqual(
            values[values.index("--maximum-adaptive-subdivisions") + 1],
            str(expected["maximum_adaptive_subdivisions"]),
        )
        self.assertEqual(
            values[values.index("--covariance-centered-cache-gib") + 1], "24"
        )
        output = values[values.index("--output-dir") + 1]
        self.assertTrue(output.startswith("outputs/fresh-direct-first/"))

    def test_wrapper_coherent_presets_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            wrapper.main(
                [
                    "--xai-method",
                    "IG",
                    "--target-event",
                    "extreme_el_nino",
                    "--coherent-adaptive",
                    "--coherent-direct-first-adaptive",
                ]
            )

    def test_wrapper_fixed_batch_rescue_preset_is_bounded_and_exact(self) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--xai-method",
                    "GradientSHAP",
                    "--target-event",
                    "extreme_el_nino",
                    "--fixed-batch-rescue-policy",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        expected = wrapper.FIXED_BATCH_RESCUE_PRESET
        self.assertEqual(
            values[values.index("--maximum-iterations") + 1],
            str(expected["maximum_iterations"]),
        )
        self.assertEqual(
            values[values.index("--continuation-fractions") + 1], "0.5,1.0"
        )
        self.assertEqual(
            values[values.index("--adaptive-minimum-fraction-step") + 1],
            "0.00625",
        )
        self.assertEqual(
            values[values.index("--maximum-adaptive-subdivisions") + 1], "3"
        )
        # The supervisor already spent the single 40-iteration primary solve.
        # This command is the bounded rescue only, so it must not repeat it.
        self.assertNotIn("--direct-exact-target-first", values)
        self.assertIn("--scale-continuation-warm-start", values)
        self.assertIn("--adaptive-continuation", values)
        self.assertNotIn("--zero-dual-restart-after-warm-rejection", values)
        # Publication-gate tolerances must remain the direct-driver defaults.
        self.assertNotIn("--constraint-tolerance", values)
        self.assertNotIn("--relative-stationarity-tolerance", values)
        self.assertNotIn("--relative-complementarity-tolerance", values)

    def test_wrapper_fixed_batch_rescue_rejects_other_presets_and_overrides(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            wrapper.main(
                [
                    "--xai-method",
                    "IG",
                    "--target-event",
                    "extreme_el_nino",
                    "--fixed-batch-rescue-policy",
                    "--coherent-adaptive",
                ]
            )

    def test_wrapper_primary_exact_only_is_one_40_iteration_solve(self) -> None:
        with patch.object(wrapper.direct, "main", return_value=1) as delegated:
            status = wrapper.main(
                [
                    "--xai-method",
                    "GRAD",
                    "--target-event",
                    "extreme_la_nina",
                    "--primary-exact-only",
                ]
            )
        self.assertEqual(status, 1)
        values = delegated.call_args.args[0]
        self.assertEqual(values[values.index("--maximum-iterations") + 1], "40")
        self.assertEqual(values[values.index("--continuation-fractions") + 1], "1")
        for forbidden in (
            "--direct-exact-target-first",
            "--scale-continuation-warm-start",
            "--adaptive-continuation",
            "--zero-dual-restart-after-warm-rejection",
        ):
            self.assertNotIn(forbidden, values)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            wrapper.main(
                [
                    "--xai-method",
                    "GRAD",
                    "--target-event",
                    "extreme_la_nina",
                    "--primary-exact-only",
                    "--fixed-batch-rescue-policy",
                ]
            )

        for override in (
            ("--maximum-iterations", "41"),
            ("--continuation-fractions", "0.5,1.0"),
            ("--adaptive-continuation",),
        ):
            with (
                self.subTest(primary_override=override),
                self.assertRaisesRegex(ValueError, "cannot be combined"),
            ):
                wrapper.main(
                    [
                        "--xai-method",
                        "GRAD",
                        "--target-event",
                        "extreme_la_nina",
                        "--primary-exact-only",
                        *override,
                    ]
                )

    def test_wrapper_fixed_batch_rescue_rejects_optimizer_overrides(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            wrapper.main(
                [
                    "--xai-method",
                    "IG",
                    "--target-event",
                    "extreme_el_nino",
                    "--fixed-batch-rescue-policy",
                    "--maximum-adaptive-subdivisions",
                    "4",
                ]
            )

    def test_wrapper_hard_batch_rescue_is_exactly_twelve_stage_attempts(
        self,
    ) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--xai-method",
                    "AGOP",
                    "--target-event",
                    "extreme_la_nina",
                    "--hard-batch-rescue-policy",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        self.assertEqual(values[values.index("--maximum-iterations") + 1], "55")
        self.assertEqual(
            values[values.index("--continuation-fractions") + 1], "0.5,1.0"
        )
        self.assertEqual(
            values[values.index("--maximum-adaptive-subdivisions") + 1], "10"
        )
        self.assertEqual(
            values[values.index("--adaptive-minimum-fraction-step") + 1],
            "0.00625",
        )
        self.assertIn("--scale-continuation-warm-start", values)
        self.assertIn("--adaptive-continuation", values)
        self.assertNotIn("--zero-dual-restart-after-warm-rejection", values)
        self.assertNotIn("--direct-exact-target-first", values)
        for gate_override in (
            "--constraint-tolerance",
            "--relative-stationarity-tolerance",
            "--relative-complementarity-tolerance",
        ):
            self.assertNotIn(gate_override, values)

    def test_wrapper_direct_first_preset_rejects_partial_override(self) -> None:
        for override in (
            ("--direct-exact-target-first",),
            ("--maximum-iterations", "41"),
            ("--continuation-fractions", "0.5,1.0"),
        ):
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ValueError, "cannot be combined"),
            ):
                wrapper.main(
                    [
                        "--xai-method",
                        "GradientSHAP",
                        "--target-event",
                        "extreme_la_nina",
                        "--coherent-direct-first-adaptive",
                        *override,
                    ]
                )

    def test_wrapper_and_direct_driver_disable_option_abbreviation(self) -> None:
        self.assertFalse(wrapper.parser().allow_abbrev)
        self.assertFalse(wrapper.direct.parser().allow_abbrev)
        # Previously ``--sel`` could pass through this wrapper and abbreviate
        # the direct driver's fixed ``--selection-seed`` option.
        with self.assertRaises(SystemExit):
            wrapper.main(
                [
                    "--xai-method",
                    "GRAD",
                    "--target-event",
                    "extreme_el_nino",
                    "--sel=7",
                ]
            )


if __name__ == "__main__":
    unittest.main()
