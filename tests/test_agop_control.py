from __future__ import annotations

import unittest

import numpy as np

from zc_xai.agop_control import (
    control_space_forecast_gradient,
    half_cosine_iau_weights,
    solve_linearized_agop_alignment,
    unit_fixed_phase_direction,
)


class AGOPControlTests(unittest.TestCase):
    def test_fixed_phase_direction_is_spatial_and_unit_norm(self) -> None:
        direction = np.asarray([3.0, 4.0, 9.0, -2.0])
        result = unit_fixed_phase_direction(direction, phase_features=2)
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_half_cosine_iau_is_positive_smooth_and_normalized(self) -> None:
        weights = half_cosine_iau_weights(9)
        self.assertEqual(weights.shape, (9,))
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        np.testing.assert_allclose(weights, weights[::-1], rtol=0.0, atol=2e-16)

    def test_closed_form_solution_satisfies_constraint_and_kkt_system(self) -> None:
        rng = np.random.default_rng(20260906)
        response = rng.normal(size=(17, 6))
        direction = rng.normal(size=17)
        direction /= np.linalg.norm(direction)
        solution = solve_linearized_agop_alignment(
            response,
            direction,
            projection=1.75,
            perpendicular_penalty=12.0,
        )
        self.assertAlmostEqual(solution.realized_projection, 1.75, places=12)
        self.assertLess(solution.kkt_stationarity_norm, 1.0e-11)

        along = direction @ response
        perpendicular = response - np.outer(direction, along)
        quadratic = np.eye(response.shape[1]) + 12.0 * (
            perpendicular.T @ perpendicular
        )
        kkt = np.block(
            [
                [quadratic, along[:, None]],
                [along[None, :], np.zeros((1, 1))],
            ]
        )
        rhs = np.concatenate((np.zeros(response.shape[1]), [1.75]))
        direct = np.linalg.solve(kkt, rhs)[:-1]
        np.testing.assert_allclose(solution.control, direct, rtol=2e-13, atol=2e-13)

    def test_perpendicular_penalty_reduces_leakage(self) -> None:
        response = np.asarray(
            [
                [1.0, 0.2],
                [1.0, -1.0],
                [0.4, 1.5],
            ]
        )
        direction = np.asarray([1.0, 0.0, 0.0])
        unpenalized = solve_linearized_agop_alignment(
            response,
            direction,
            projection=1.0,
            perpendicular_penalty=0.0,
        )
        penalized = solve_linearized_agop_alignment(
            response,
            direction,
            projection=1.0,
            perpendicular_penalty=1.0e4,
        )
        self.assertLess(
            penalized.perpendicular_norm,
            unpenalized.perpendicular_norm,
        )

    def test_unreachable_direction_is_rejected(self) -> None:
        response = np.asarray([[0.0, 0.0], [1.0, 2.0]])
        direction = np.asarray([1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "cannot produce"):
            solve_linearized_agop_alignment(
                response,
                direction,
                projection=1.0,
                perpendicular_penalty=1.0,
            )

    def test_control_gradient_is_exact_transpose(self) -> None:
        rng = np.random.default_rng(11)
        basis = rng.normal(size=(23, 5))
        state_gradient = rng.normal(size=23)
        control = rng.normal(size=5)
        projected = control_space_forecast_gradient(basis, state_gradient)
        self.assertAlmostEqual(
            float(state_gradient @ (basis @ control)),
            float(projected @ control),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
