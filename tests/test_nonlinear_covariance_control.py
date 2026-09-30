from __future__ import annotations

import math
import unittest

import numpy as np

from zc_xai.nonlinear_covariance_control import (
    _polish_positive_constraint_along_native_ray,
    covariance_action,
    pull_back_native_covectors,
    solve_minimum_covariance_action_sqp,
    solve_minimum_native_covariance_action_sqp,
)


class _DenseCovariance:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.vector_calls = 0
        self.matrix_calls = 0

    @property
    def state_size(self) -> int:
        return self.matrix.shape[0]

    def covariance_apply(self, vector: np.ndarray) -> np.ndarray:
        self.vector_calls += 1
        return self.matrix @ vector

    def covariance_apply_matrix(
        self, vectors: np.ndarray, *, block_rows: int = 128
    ) -> np.ndarray:
        del block_rows
        self.matrix_calls += 1
        return self.matrix @ vectors


class CovarianceAlgebraTests(unittest.TestCase):
    def test_covariance_action_uses_minimum_norm_representation(self) -> None:
        # The duplicate columns emulate the null direction of centered samples.
        factor = np.asarray([[1.0, 1.0], [0.0, 0.0]])
        increment = np.asarray([2.0, 0.0])
        self.assertAlmostEqual(covariance_action(increment, factor), 2.0)

    def test_control_pullback_is_exact_transpose(self) -> None:
        rng = np.random.default_rng(943)
        factors = rng.normal(size=(3, 5, 4))
        controls = rng.normal(size=(3, 4))
        covectors = rng.normal(size=(3, 5))
        native = np.einsum("snc,sc->sn", factors, controls)
        pulled = pull_back_native_covectors(factors, covectors)
        self.assertAlmostEqual(
            float(native.reshape(-1) @ covectors.reshape(-1)),
            float(controls.reshape(-1) @ pulled),
        )


class ScalarSQPTests(unittest.TestCase):
    def test_affine_constraint_converges_to_exact_minimum(self) -> None:
        normal = np.asarray([1.0, -2.0, 0.5])
        threshold = 3.0

        def value(control: np.ndarray) -> float:
            return float(normal @ control - threshold)

        def value_gradient(control: np.ndarray) -> tuple[float, np.ndarray]:
            return value(control), normal.copy()

        result = solve_minimum_covariance_action_sqp(
            value,
            value_gradient,
            dimension=3,
            initial_trust_radius=10.0,
        )
        expected = threshold * normal / float(normal @ normal)
        self.assertTrue(result.success, result.status)
        np.testing.assert_allclose(result.control, expected, rtol=1e-12, atol=1e-12)
        self.assertLessEqual(result.primal_violation, 1e-12)
        self.assertLess(result.stationarity_norm, 1e-12)
        self.assertGreaterEqual(result.lagrange_multiplier, 0.0)
        self.assertGreaterEqual(len(result.iterations), 2)

    def test_nonlinear_constraint_reports_updates_and_kkt(self) -> None:
        # Feasible set is x + x^2/2 >= 1.  The closest point to zero is
        # sqrt(3)-1; the other boundary is much farther away.
        def value(control: np.ndarray) -> float:
            x = float(control[0])
            return x + 0.5 * x * x - 1.0

        def value_gradient(control: np.ndarray) -> tuple[float, np.ndarray]:
            x = float(control[0])
            return value(control), np.asarray([1.0 + x])

        result = solve_minimum_covariance_action_sqp(
            value,
            value_gradient,
            dimension=1,
            initial_trust_radius=0.5,
            constraint_tolerance=1e-10,
            stationarity_tolerance=1e-10,
            complementarity_tolerance=1e-10,
        )
        self.assertTrue(result.success, result.status)
        self.assertAlmostEqual(result.control[0], math.sqrt(3.0) - 1.0, places=9)
        self.assertLessEqual(result.primal_violation, 1e-10)
        self.assertLessEqual(result.stationarity_norm, 1e-10)
        self.assertGreater(result.value_evaluations, 0)
        self.assertGreater(result.gradient_evaluations, 0)
        self.assertTrue(
            any(item.accepted_step_norm > 0.0 for item in result.iterations)
        )

    def test_rejects_vanishing_infeasible_constraint_gradient(self) -> None:
        def value(control: np.ndarray) -> float:
            return -1.0

        def value_gradient(control: np.ndarray) -> tuple[float, np.ndarray]:
            return value(control), np.zeros_like(control)

        result = solve_minimum_covariance_action_sqp(value, value_gradient, dimension=2)
        self.assertFalse(result.success)
        self.assertEqual(
            result.status, "constraint_gradient_vanished_before_feasibility"
        )


class NativeScalarSQPTests(unittest.TestCase):
    def test_common_covariance_affine_solution_uses_batched_passes(self) -> None:
        covariance = _DenseCovariance(np.diag([4.0, 1.0]))
        normal = np.asarray([[1.0, 2.0], [-1.0, 1.0], [0.5, -2.0]])
        threshold = 3.0

        def value(native: np.ndarray) -> float:
            return float(np.sum(normal * native) - threshold)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), normal.copy()

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=3,
            initial_trust_radius=10.0,
        )
        covariance_normal = normal @ covariance.matrix
        denominator = float(np.sum(normal * covariance_normal))
        expected_dual = threshold * normal / denominator
        expected_native = expected_dual @ covariance.matrix
        self.assertTrue(result.success, result.status)
        np.testing.assert_allclose(
            result.dual_variables, expected_dual, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            result.native_interventions,
            expected_native,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(covariance.vector_calls, 0)
        self.assertEqual(covariance.matrix_calls, result.gradient_evaluations)
        self.assertEqual(result.covariance_passes, result.gradient_evaluations)
        self.assertLess(result.covariance_stationarity_norm, 1e-12)

    def test_phase_specific_covariances_match_closed_form(self) -> None:
        matrices = (
            np.diag([1.0, 2.0]),
            np.diag([3.0, 0.5]),
        )
        covariances = tuple(_DenseCovariance(matrix) for matrix in matrices)
        normal = np.asarray([[2.0, -1.0], [0.5, 3.0]])
        threshold = 2.0

        def value(native: np.ndarray) -> float:
            return float(np.sum(normal * native) - threshold)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), normal.copy()

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariances,
            control_steps=2,
            initial_trust_radius=10.0,
        )
        covariance_normal = np.vstack(
            [matrix @ row for matrix, row in zip(matrices, normal, strict=True)]
        )
        denominator = float(np.sum(normal * covariance_normal))
        expected_native = threshold * covariance_normal / denominator
        self.assertTrue(result.success, result.status)
        np.testing.assert_allclose(
            result.native_interventions,
            expected_native,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(
            result.covariance_passes,
            2 * result.gradient_evaluations,
        )
        self.assertTrue(all(item.matrix_calls == 0 for item in covariances))
        self.assertTrue(
            all(
                item.vector_calls == result.gradient_evaluations for item in covariances
            )
        )

    def test_relative_kkt_tolerances_stop_on_dimensionless_residuals(self) -> None:
        covariance = _DenseCovariance(np.eye(2))

        def value(native: np.ndarray) -> float:
            x = native[0]
            return float(x[0] + x[1] + 0.5 * x[0] ** 2 - 1.0)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            x = native[0]
            return value(native), np.asarray([[1.0 + x[0], 1.0]])

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            maximum_iterations=5,
            stationarity_tolerance=1e-20,
            complementarity_tolerance=1e-20,
            relative_stationarity_tolerance=1e-2,
            relative_complementarity_tolerance=1e-2,
            initial_trust_radius=10.0,
        )
        self.assertTrue(result.success, result.status)
        self.assertEqual(result.status, "first_order_kkt_satisfied")
        self.assertLessEqual(result.covariance_stationarity_relative, 1e-2)
        self.assertLessEqual(result.complementarity_relative, 1e-2)
        self.assertGreater(result.covariance_stationarity_norm, 1e-20)
        self.assertGreater(result.complementarity_absolute, 1e-20)
        self.assertAlmostEqual(
            result.covariance_stationarity_relative,
            result.covariance_stationarity_norm / result.stationarity_reference_norm,
        )

    def test_infeasible_origin_uses_active_boundary_penalty(self) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            x = float(native[0, 0])
            return x + 20.0 * x**2 - 1.0

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            x = float(native[0, 0])
            return value(native), np.asarray([[1.0 + 40.0 * x]])

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            maximum_iterations=10,
            initial_trust_radius=10.0,
            stationarity_tolerance=1.0e-10,
            complementarity_tolerance=1.0e-10,
        )
        self.assertTrue(result.success, result.status)
        # The first accepted trial crosses from the infeasible origin into
        # h > 0.  Because the origin is infeasible, the optimum is known to
        # be on h = 0 and the exact merit continues to penalize |h|.
        overshoot = result.iterations[1]
        self.assertGreater(overshoot.constraint_value, 0.0)
        self.assertAlmostEqual(
            overshoot.merit_value,
            overshoot.objective_value
            + overshoot.merit_penalty * abs(overshoot.constraint_value),
        )
        self.assertLess(abs(result.constraint_value), 1.0e-10)

    def test_known_active_boundary_preserves_penalty_after_warm_start_overshoot(
        self,
    ) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            x = float(native[0, 0])
            return x + 20.0 * x**2 - 1.0

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            x = float(native[0, 0])
            return value(native), np.asarray([[1.0 + 40.0 * x]])

        kwargs = {
            "control_steps": 1,
            "initial_dual_variables": np.asarray([[0.1]]),
            "maximum_iterations": 10,
            "initial_trust_radius": 10.0,
            "stationarity_tolerance": 1.0e-10,
            "complementarity_tolerance": 1.0e-10,
        }
        general = solve_minimum_native_covariance_action_sqp(
            value, value_gradient, covariance, **kwargs
        )
        active = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            known_active_boundary=True,
            **kwargs,
        )
        self.assertTrue(active.success, active.status)
        general_overshoot = general.iterations[1]
        active_overshoot = active.iterations[1]
        self.assertGreater(active_overshoot.constraint_value, 0.0)
        self.assertAlmostEqual(
            active_overshoot.merit_value,
            active_overshoot.objective_value
            + active_overshoot.merit_penalty * abs(active_overshoot.constraint_value),
        )
        self.assertAlmostEqual(
            general_overshoot.merit_value,
            general_overshoot.objective_value,
        )
        self.assertEqual(active_overshoot.trust_radius, 10.0)
        self.assertGreaterEqual(active.constraint_value, 0.0)
        self.assertLess(active.constraint_value, 1.0e-8)

    def test_positive_active_boundary_polish_rescues_stalled_small_trust(self) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            return float(native[0, 0] ** 3 - 1.0)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            x = float(native[0, 0])
            return value(native), np.asarray([[3.0 * x**2]])

        kwargs = {
            "control_steps": 1,
            "initial_dual_variables": np.asarray([[1.01]]),
            "maximum_iterations": 1,
            "initial_trust_radius": 1.0e-5,
            "relative_stationarity_tolerance": 1.0e-10,
            "relative_complementarity_tolerance": 1.0e-10,
        }
        stalled = solve_minimum_native_covariance_action_sqp(
            value, value_gradient, covariance, **kwargs
        )
        polished = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            known_active_boundary=True,
            **kwargs,
        )
        self.assertFalse(stalled.success)
        self.assertEqual(stalled.status, "maximum_iterations_reached")
        self.assertTrue(polished.success, polished.status)
        self.assertEqual(
            polished.status,
            "first_order_kkt_satisfied_after_active_boundary_radial_polishing",
        )
        self.assertTrue(polished.radial_restoration_applied)
        self.assertLess(polished.radial_restoration_scale, 1.0)
        self.assertGreaterEqual(polished.constraint_value, 0.0)
        self.assertLessEqual(polished.covariance_stationarity_relative, 1.0e-10)
        self.assertLessEqual(polished.complementarity_relative, 1.0e-10)
        np.testing.assert_allclose(
            polished.native_interventions,
            covariance.matrix @ polished.dual_variables.T,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_positive_polish_may_start_just_above_stationarity_gate(self) -> None:
        covariance = _DenseCovariance(np.eye(2))
        curvature = 0.52313
        boundary_scale = 0.96
        target = boundary_scale + curvature * boundary_scale**2

        def value(native: np.ndarray) -> float:
            x, y = (float(item) for item in native[0])
            return x + curvature * y**2 - target

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            _, y = (float(item) for item in native[0])
            return value(native), np.asarray([[1.0, 2.0 * curvature * y]])

        kwargs = {
            "control_steps": 1,
            "initial_dual_variables": np.ones((1, 2)),
            "maximum_iterations": 1,
            "initial_trust_radius": 1.0e-8,
            "relative_stationarity_tolerance": 2.0e-2,
            "relative_complementarity_tolerance": 1.0e-8,
        }
        unpolished = solve_minimum_native_covariance_action_sqp(
            value, value_gradient, covariance, **kwargs
        )
        polished = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            known_active_boundary=True,
            **kwargs,
        )
        initial_stationarity = unpolished.iterations[0].covariance_stationarity_relative
        self.assertAlmostEqual(initial_stationarity, 0.0226, places=4)
        self.assertGreater(initial_stationarity, 2.0e-2)
        self.assertLessEqual(initial_stationarity, 1.5 * 2.0e-2)
        self.assertTrue(polished.success, polished.status)
        self.assertTrue(polished.radial_restoration_applied)
        self.assertLessEqual(polished.covariance_stationarity_relative, 2.0e-2)
        self.assertLessEqual(polished.complementarity_relative, 1.0e-8)

    def test_positive_boundary_candidate_can_continue_as_sqp_iterate(self) -> None:
        covariance = _DenseCovariance(np.eye(2))
        cross_weight = 0.935
        curvature = 0.1
        boundary_scale = 0.9682
        target = (
            boundary_scale * (1.0 + cross_weight)
            + curvature * (boundary_scale - 1.0) ** 2
        )

        def value(native: np.ndarray) -> float:
            x, y = (float(item) for item in native[0])
            return x + cross_weight * y + curvature * (y - 1.0) ** 2 - target

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            _, y = (float(item) for item in native[0])
            return value(native), np.asarray(
                [[1.0, cross_weight + 2.0 * curvature * (y - 1.0)]]
            )

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            initial_dual_variables=np.ones((1, 2)),
            known_active_boundary=True,
            maximum_iterations=10,
            initial_trust_radius=1.0e-8,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-4,
        )

        radial_iterates = [
            item
            for item in result.iterations
            if item.radial_restoration_reason is not None
        ]
        self.assertEqual(len(radial_iterates), 1)
        self.assertGreater(radial_iterates[0].covariance_stationarity_relative, 2.0e-2)
        self.assertTrue(result.success, result.status)
        self.assertEqual(
            result.status,
            "first_order_kkt_satisfied_after_active_boundary_radial_iterate",
        )
        self.assertLessEqual(result.covariance_stationarity_relative, 2.0e-2)
        self.assertLessEqual(result.complementarity_relative, 1.0e-4)
        np.testing.assert_allclose(
            result.native_interventions,
            (covariance.matrix @ result.dual_variables.T).T,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_partial_positive_boundary_iterate_is_not_false_success(self) -> None:
        covariance = _DenseCovariance(np.eye(2))
        cross_weight = 0.935
        curvature = 0.1
        boundary_scale = 0.9682
        target = (
            boundary_scale * (1.0 + cross_weight)
            + curvature * (boundary_scale - 1.0) ** 2
        )

        def value(native: np.ndarray) -> float:
            x, y = (float(item) for item in native[0])
            return x + cross_weight * y + curvature * (y - 1.0) ** 2 - target

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            _, y = (float(item) for item in native[0])
            return value(native), np.asarray(
                [[1.0, cross_weight + 2.0 * curvature * (y - 1.0)]]
            )

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            initial_dual_variables=np.ones((1, 2)),
            known_active_boundary=True,
            maximum_iterations=1,
            initial_trust_radius=1.0e-8,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-4,
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.status,
            "maximum_iterations_reached_after_active_boundary_radial_iterate",
        )
        self.assertTrue(result.radial_restoration_applied)
        self.assertGreaterEqual(result.constraint_value, 0.0)
        self.assertGreater(result.covariance_stationarity_relative, 2.0e-2)
        self.assertIn(
            "candidate_accepted_as_sqp_iterate", result.radial_restoration_reason
        )
        np.testing.assert_allclose(
            result.native_interventions,
            (covariance.matrix @ result.dual_variables.T).T,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_negative_boundary_candidate_can_continue_as_sqp_iterate(self) -> None:
        covariance = _DenseCovariance(np.eye(2))
        cross_weight = 0.9402
        radial_derivative = 34.0013
        normal_scale = radial_derivative / (1.0 + cross_weight)
        normal = normal_scale * np.asarray([[1.0, cross_weight]])
        initial_gap = 1.47077e-3
        target = float(np.sum(normal) + initial_gap)

        def value(native: np.ndarray) -> float:
            return float(np.sum(normal * native) - target)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), normal.copy()

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            initial_dual_variables=np.ones((1, 2)),
            known_active_boundary=True,
            maximum_iterations=2,
            initial_trust_radius=1.0e-10,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-4,
        )

        radial_iterates = [
            item
            for item in result.iterations
            if item.radial_restoration_reason is not None
        ]
        self.assertEqual(len(radial_iterates), 1)
        self.assertAlmostEqual(initial_gap / radial_derivative, 4.32563e-5, places=10)
        self.assertGreater(radial_iterates[0].radial_restoration_scale, 1.0)
        self.assertGreater(radial_iterates[0].covariance_stationarity_relative, 2.0e-2)
        self.assertTrue(result.success, result.status)
        self.assertEqual(
            result.status,
            "first_order_kkt_satisfied_after_active_boundary_radial_iterate",
        )
        self.assertGreaterEqual(result.constraint_value, 0.0)
        self.assertLessEqual(result.covariance_stationarity_relative, 2.0e-2)
        self.assertLessEqual(result.complementarity_relative, 1.0e-4)
        np.testing.assert_allclose(
            result.native_interventions,
            (covariance.matrix @ result.dual_variables.T).T,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_partial_negative_boundary_iterate_is_not_false_success(self) -> None:
        covariance = _DenseCovariance(np.eye(2))
        cross_weight = 0.9402
        radial_derivative = 34.0013
        normal_scale = radial_derivative / (1.0 + cross_weight)
        normal = normal_scale * np.asarray([[1.0, cross_weight]])
        target = float(np.sum(normal) + 1.47077e-3)

        def value(native: np.ndarray) -> float:
            return float(np.sum(normal * native) - target)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), normal.copy()

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            initial_dual_variables=np.ones((1, 2)),
            known_active_boundary=True,
            maximum_iterations=1,
            initial_trust_radius=1.0e-10,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-4,
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.status,
            "maximum_iterations_reached_after_active_boundary_radial_iterate",
        )
        self.assertTrue(result.radial_restoration_applied)
        self.assertGreater(result.radial_restoration_scale, 1.0)
        self.assertGreaterEqual(result.constraint_value, 0.0)
        self.assertGreater(result.covariance_stationarity_relative, 2.0e-2)
        self.assertIn(
            "candidate_accepted_as_sqp_iterate", result.radial_restoration_reason
        )
        np.testing.assert_allclose(
            result.native_interventions,
            (covariance.matrix @ result.dual_variables.T).T,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_active_boundary_second_order_correction_avoids_maratos_stall(
        self,
    ) -> None:
        covariance = _DenseCovariance(np.eye(2))
        cross_weight = 0.94
        curvature = 1.0
        initial_positive_residual = 5.0e-6
        target = 1.0 + cross_weight - initial_positive_residual

        def value(native: np.ndarray) -> float:
            x, y = (float(item) for item in native[0])
            return x + cross_weight * y + curvature * (y - 1.0) ** 2 - target

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            _, y = (float(item) for item in native[0])
            return value(native), np.asarray(
                [[1.0, cross_weight + 2.0 * curvature * (y - 1.0)]]
            )

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            initial_dual_variables=np.ones((1, 2)),
            known_active_boundary=True,
            maximum_iterations=20,
            initial_trust_radius=10.0,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-4,
        )

        corrected = [
            item for item in result.iterations if item.second_order_correction_applied
        ]
        self.assertGreaterEqual(len(corrected), 1)
        first = corrected[0]
        self.assertEqual(first.accepted_step_scale, 1.0)
        self.assertEqual(first.line_search_evaluations, 2)
        self.assertGreater(first.second_order_correction_action_norm, 0.0)
        self.assertLessEqual(
            first.second_order_correction_action_norm,
            first.proposed_step_action_norm,
        )
        self.assertTrue(result.success, result.status)
        self.assertGreaterEqual(result.constraint_value, 0.0)
        self.assertLessEqual(result.covariance_stationarity_relative, 2.0e-2)
        self.assertLessEqual(result.complementarity_relative, 1.0e-4)
        np.testing.assert_allclose(
            result.native_interventions,
            (covariance.matrix @ result.dual_variables.T).T,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_second_order_correction_cannot_declare_false_success(self) -> None:
        covariance = _DenseCovariance(np.eye(2))
        cross_weight = 0.94
        curvature = 1.0
        target = 1.0 + cross_weight - 5.0e-6

        def value(native: np.ndarray) -> float:
            x, y = (float(item) for item in native[0])
            return x + cross_weight * y + curvature * (y - 1.0) ** 2 - target

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            _, y = (float(item) for item in native[0])
            return value(native), np.asarray(
                [[1.0, cross_weight + 2.0 * curvature * (y - 1.0)]]
            )

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            initial_dual_variables=np.ones((1, 2)),
            known_active_boundary=True,
            maximum_iterations=1,
            initial_trust_radius=10.0,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-4,
        )

        self.assertTrue(result.iterations[0].second_order_correction_applied)
        self.assertFalse(result.success)
        self.assertEqual(result.status, "maximum_iterations_reached")
        self.assertGreater(result.covariance_stationarity_relative, 2.0e-2)
        np.testing.assert_allclose(
            result.native_interventions,
            (covariance.matrix @ result.dual_variables.T).T,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_positive_polish_does_not_start_beyond_stationarity_attempt_band(
        self,
    ) -> None:
        covariance = _DenseCovariance(np.eye(2))
        curvature = 0.55
        boundary_scale = 0.96
        target = boundary_scale + curvature * boundary_scale**2

        def value(native: np.ndarray) -> float:
            x, y = (float(item) for item in native[0])
            return x + curvature * y**2 - target

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            _, y = (float(item) for item in native[0])
            return value(native), np.asarray([[1.0, 2.0 * curvature * y]])

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            initial_dual_variables=np.ones((1, 2)),
            known_active_boundary=True,
            maximum_iterations=1,
            initial_trust_radius=1.0e-8,
            relative_stationarity_tolerance=2.0e-2,
            relative_complementarity_tolerance=1.0e-8,
        )
        self.assertGreater(
            result.iterations[0].covariance_stationarity_relative,
            2.0 * 2.0e-2,
        )
        self.assertFalse(result.radial_restoration_applied)
        self.assertEqual(result.radial_restoration_value_evaluations, 0)
        self.assertIsNone(result.radial_restoration_reason)

    def test_positive_radial_polish_uses_nearest_sampled_nonmonotone_crossing(
        self,
    ) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            x = float(native[0, 0])
            return (x - 0.99) * (x - 0.97)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            x = float(native[0, 0])
            return value(native), np.asarray([[2.0 * x - 1.96]])

        native = np.ones((1, 1))
        attempt = _polish_positive_constraint_along_native_ray(
            value,
            value_gradient,
            covariance,
            (),
            dual=native,
            native=native,
            value=value(native),
            gradient=np.asarray([[0.04]]),
            covariance_block_rows=1,
            target_positive_value=1.0e-12,
        )
        self.assertEqual(attempt.reason, "candidate_resolved")
        self.assertIsNotNone(attempt.candidate)
        assert attempt.candidate is not None
        self.assertAlmostEqual(attempt.candidate.scale, 0.99, places=8)
        self.assertGreater(attempt.candidate.scale, 0.98)
        self.assertGreaterEqual(attempt.candidate.value, 0.0)
        self.assertLessEqual(attempt.candidate.value, 1.0e-12)

    def test_positive_radial_polish_aborts_at_nonfinite_bracket_gap(self) -> None:
        covariance = _DenseCovariance(np.eye(1))
        native = np.ones((1, 1))

        def value(candidate: np.ndarray) -> float:
            x = float(candidate[0, 0])
            if 0.992 <= x <= 0.993:
                return math.nan
            return (x - 0.99) * (x - 0.97)

        def value_gradient(candidate: np.ndarray) -> tuple[float, np.ndarray]:
            x = float(candidate[0, 0])
            return value(candidate), np.asarray([[2.0 * x - 1.96]])

        attempt = _polish_positive_constraint_along_native_ray(
            value,
            value_gradient,
            covariance,
            (),
            dual=native,
            native=native,
            value=3.0e-4,
            gradient=np.asarray([[0.04]]),
            covariance_block_rows=1,
            target_positive_value=1.0e-12,
        )
        self.assertIsNone(attempt.candidate)
        self.assertEqual(attempt.reason, "nonfinite_value_before_sign_bracket")
        self.assertEqual(attempt.value_evaluations, 1)
        self.assertEqual(attempt.gradient_evaluations, 0)

    def test_known_active_boundary_requires_a_bool(self) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            return float(native[0, 0] - 1.0)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), np.ones_like(native)

        with self.assertRaisesRegex(ValueError, "known_active_boundary"):
            solve_minimum_native_covariance_action_sqp(
                value,
                value_gradient,
                covariance,
                control_steps=1,
                known_active_boundary=1,  # type: ignore[arg-type]
            )

    def test_radial_restoration_repairs_stationary_infeasible_termination(self) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            x = float(native[0, 0])
            return x - 0.01 * x**2 - 1.0

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            x = float(native[0, 0])
            return value(native), np.asarray([[1.0 - 0.02 * x]])

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            maximum_iterations=1,
            initial_trust_radius=10.0,
            stationarity_tolerance=1.0e-10,
            complementarity_tolerance=1.0e-10,
        )
        expected = 50.0 * (1.0 - math.sqrt(0.96))
        self.assertTrue(result.success, result.status)
        self.assertEqual(
            result.status, "first_order_kkt_satisfied_after_radial_restoration"
        )
        self.assertTrue(result.radial_restoration_applied)
        self.assertEqual(
            result.radial_restoration_reason,
            "negative_constraint_at_otherwise_stationary_sqp_termination",
        )
        self.assertIsNotNone(result.radial_restoration_scale)
        self.assertGreater(result.radial_restoration_value_evaluations, 0)
        self.assertEqual(result.radial_restoration_gradient_evaluations, 1)
        self.assertEqual(result.radial_restoration_covariance_passes, 1)
        self.assertGreaterEqual(result.constraint_value, 0.0)
        self.assertAlmostEqual(result.native_interventions[0, 0], expected, places=9)
        np.testing.assert_allclose(
            result.native_interventions,
            covariance.matrix @ result.dual_variables.T,
            rtol=0.0,
            atol=1.0e-14,
        )
        restored_record = result.iterations[-1]
        self.assertEqual(
            restored_record.radial_restoration_reason,
            result.radial_restoration_reason,
        )
        self.assertEqual(
            restored_record.radial_restoration_scale,
            result.radial_restoration_scale,
        )
        self.assertEqual(
            restored_record.radial_restoration_value_evaluations,
            result.radial_restoration_value_evaluations,
        )

    def test_radial_restoration_enforces_literal_nonnegative_constraint(self) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            x = float(native[0, 0])
            if x == 1.0:
                return -5.0e-6
            return x - 1.0

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), np.ones_like(native)

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
            maximum_iterations=2,
            initial_trust_radius=10.0,
            relative_stationarity_tolerance=1.0e-8,
            relative_complementarity_tolerance=1.0e-4,
        )
        self.assertTrue(result.success, result.status)
        self.assertTrue(result.radial_restoration_applied)
        self.assertGreaterEqual(result.constraint_value, 0.0)
        self.assertLess(result.constraint_value, 1.0e-5)
        self.assertGreater(result.native_interventions[0, 0], 1.0)
        self.assertLessEqual(result.covariance_stationarity_relative, 1.0e-8)
        self.assertLessEqual(result.complementarity_relative, 1.0e-4)

    def test_feasible_origin_retains_inequality_solution(self) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            return float(1.0 - native[0, 0])

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), np.asarray([[-1.0]])

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=1,
        )
        self.assertTrue(result.success, result.status)
        np.testing.assert_array_equal(result.native_interventions, np.zeros((1, 1)))
        self.assertEqual(result.iterations[0].constraint_value, 1.0)
        self.assertEqual(result.iterations[0].merit_value, 0.0)

    def test_rejects_invalid_relative_kkt_tolerance(self) -> None:
        covariance = _DenseCovariance(np.eye(1))

        def value(native: np.ndarray) -> float:
            return float(native[0, 0] - 1.0)

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), np.ones_like(native)

        with self.assertRaisesRegex(ValueError, "relative_stationarity_tolerance"):
            solve_minimum_native_covariance_action_sqp(
                value,
                value_gradient,
                covariance,
                control_steps=1,
                relative_stationarity_tolerance=0.0,
            )

    def test_native_solver_reports_covariance_range_gradient_failure(self) -> None:
        covariance = _DenseCovariance(np.zeros((2, 2)))

        def value(native: np.ndarray) -> float:
            return -1.0

        def value_gradient(native: np.ndarray) -> tuple[float, np.ndarray]:
            return value(native), np.ones_like(native)

        result = solve_minimum_native_covariance_action_sqp(
            value,
            value_gradient,
            covariance,
            control_steps=2,
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.status,
            "constraint_gradient_vanished_in_covariance_range",
        )


if __name__ == "__main__":
    unittest.main()
