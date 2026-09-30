"""Tests for the event-composite native ZC control target and wrapper."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from zc_xai.direct_composite_target import (
    COMPOSITE_METHOD,
    build_direct_composite_target,
)

sys.path.insert(0, "scripts")
import run_zc_pooled_composite_covariance_action as wrapper  # noqa: E402


class _DummyCompositeData:
    input_shape = (4,)
    n_phase_features = 2
    steps_per_year = 12

    def __init__(self) -> None:
        self.target = np.zeros(20, dtype=np.float64)
        self.target[3] = 1.5
        self.target[4] = 3.0
        self.target[7] = 2.0
        self.target[10] = 1.25
        self.metadata = {
            "event_restart_checkpoints": [
                {
                    "label": "extreme_el_nino",
                    "input_index": 2,
                    "target_index": 4,
                }
            ]
        }

    def fixed_supervised_split(self, lead_months: int) -> object:
        if lead_months != 10:
            raise AssertionError("unexpected lead")
        return SimpleNamespace(
            test_inputs=np.arange(12, dtype=np.int64),
            lead_steps=2,
        )

    def load_inputs(
        self,
        indices: np.ndarray,
        *,
        standardizer: object | None = None,
    ) -> np.ndarray:
        del standardizer
        indices = np.asarray(indices, dtype=np.float64)
        return np.stack(
            (
                indices,
                np.ones(indices.size),
                np.full(indices.size, 0.5),
                np.full(indices.size, -0.5),
            ),
            axis=1,
        )


class DirectCompositeTargetTests(unittest.TestCase):
    def test_event_equal_top_decile_drops_phase_and_targets_extreme_projection(
        self,
    ) -> None:
        data = _DummyCompositeData()
        event = np.asarray((4.0, 2.0, 0.0, 0.0))
        target = build_direct_composite_target(
            COMPOSITE_METHOD,
            data=data,  # type: ignore[arg-type]
            experiment=SimpleNamespace(
                standardizer=SimpleNamespace(
                    mean=np.zeros(4),
                    scale=np.ones(4),
                )
            ),  # type: ignore[arg-type]
            event_standardized=event,
            target_event_label="extreme_el_nino",
        )
        np.testing.assert_array_equal(
            target.reference_arrays["composite_selected_peak_steps_ranked"],
            (4,),
        )
        np.testing.assert_allclose(
            target.raw_spatial_direction,
            np.asarray((2.0, 1.0)) / np.sqrt(5.0),
        )
        np.testing.assert_array_equal(target.fixed_phase_explanation[-2:], 0.0)
        self.assertAlmostEqual(target.oriented_event_projection, np.sqrt(20.0))
        self.assertEqual(target.orientation_multiplier, 1.0)
        self.assertEqual(target.provenance["parameters"]["selected_event_count"], 1)

    def test_wrapper_fixes_the_matched_controller_and_restores_registration(
        self,
    ) -> None:
        original_methods = wrapper.direct.XAI_METHODS
        original_builder = wrapper.direct.build_direct_xai_target
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--target-event",
                    "extreme_la_nina",
                    "--member",
                    "399086",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        self.assertEqual(values[values.index("--xai-method") + 1], COMPOSITE_METHOD)
        self.assertEqual(values[values.index("--case") + 1], "pooled")
        self.assertEqual(values[values.index("--selection-seed") + 1], "42")
        self.assertEqual(values[values.index("--trajectory") + 1], "uniform-cohort")
        self.assertNotIn("--covariance-policy", values)
        self.assertEqual(
            wrapper.direct.parser().parse_args([]).covariance_policy,
            wrapper.direct.ANNUAL_SHARED_COVARIANCE_POLICY,
        )
        self.assertIn("--direct-exact-target-first", values)
        self.assertEqual(
            values[values.index("--covariance-centered-cache-gib") + 1], "24.0"
        )
        self.assertTrue(
            values[values.index("--output-dir") + 1].endswith("/extreme-la-nina")
        )
        self.assertEqual(wrapper.direct.XAI_METHODS, original_methods)
        self.assertIs(wrapper.direct.build_direct_xai_target, original_builder)

    def test_fixed_batch_rescue_policy_is_bounded_and_keeps_strict_gates(
        self,
    ) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--target-event",
                    "extreme_el_nino",
                    "--fixed-batch-rescue-policy",
                ]
            )
        self.assertEqual(status, 0)
        values = delegated.call_args.args[0]
        self.assertEqual(values[values.index("--maximum-iterations") + 1], "55")
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
        self.assertNotIn("--direct-exact-target-first", values)
        self.assertIn("--scale-continuation-warm-start", values)
        self.assertIn("--adaptive-continuation", values)
        self.assertNotIn("--zero-dual-restart-after-warm-rejection", values)
        self.assertNotIn("--constraint-tolerance", values)
        self.assertNotIn("--relative-stationarity-tolerance", values)
        self.assertNotIn("--relative-complementarity-tolerance", values)

    def test_primary_exact_only_is_one_40_iteration_solve(self) -> None:
        with patch.object(wrapper.direct, "main", return_value=1) as delegated:
            status = wrapper.main(
                [
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
                    "--target-event",
                    "extreme_la_nina",
                    "--primary-exact-only",
                    "--fixed-batch-rescue-policy",
                ]
            )

    def test_hard_batch_rescue_is_exactly_twelve_stage_attempts(self) -> None:
        with patch.object(wrapper.direct, "main", return_value=0) as delegated:
            status = wrapper.main(
                [
                    "--target-event",
                    "extreme_el_nino",
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


if __name__ == "__main__":
    unittest.main()
