"""Tests for the boundary-level controlled ZC tangent/adjoint derivation."""

from __future__ import annotations

import unittest

import numpy as np

from zc_xai.adjoint_chain import (
    DistributedControlOperator,
    form_release_response_matrix,
    propagate_distributed_control_adjoint,
    propagate_distributed_control_tangent,
    three_mode_half_cosine_iau_basis,
)


class TemporalBasisTests(unittest.TestCase):
    def test_three_mode_basis_has_declared_action_and_dose(self) -> None:
        basis = three_mode_half_cosine_iau_basis(9)
        scale = float(basis[:, 0] @ basis[:, 0])
        np.testing.assert_allclose(basis.T @ basis, scale * np.eye(3), atol=1.0e-15)
        self.assertAlmostEqual(float(np.sum(basis[:, 0])), 1.0, places=15)
        np.testing.assert_allclose(np.sum(basis[:, 1:], axis=0), 0.0, atol=1.0e-14)
        self.assertTrue(np.all(basis[:, 0] > 0.0))


class DistributedControlTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(8301)
        self.spatial = rng.normal(size=(7, 3))
        self.temporal = three_mode_half_cosine_iau_basis(4)
        self.operator = DistributedControlOperator(self.spatial, self.temporal)

    def test_each_step_map_has_exact_transpose(self) -> None:
        rng = np.random.default_rng(8302)
        control = rng.normal(size=self.operator.control_size)
        seed = rng.normal(size=self.operator.state_size)
        for step in range(self.operator.control_steps):
            left = float(self.operator.apply_at_step(control, step) @ seed)
            right = float(control @ self.operator.transpose_at_step(seed, step))
            self.assertAlmostEqual(left, right, places=13)

    def test_phase_local_spatial_maps_have_exact_transposes(self) -> None:
        rng = np.random.default_rng(8305)
        spatial = rng.normal(size=(4, 7, 3))
        operator = DistributedControlOperator(spatial, self.temporal)
        control = rng.normal(size=operator.control_size)
        seed = rng.normal(size=operator.state_size)
        for step in range(operator.control_steps):
            left = float(operator.apply_at_step(control, step) @ seed)
            right = float(control @ operator.transpose_at_step(seed, step))
            self.assertAlmostEqual(left, right, places=13)

    def test_full_tangent_and_adjoint_are_transposes(self) -> None:
        rng = np.random.default_rng(8303)
        matrices = [rng.normal(size=(7, 7)) / 3.0 for _ in range(8)]
        previous_observation = rng.normal(size=(5, 7))
        post_observation = rng.normal(size=(5, 7))

        def tangent_step(step: int, vector: np.ndarray) -> np.ndarray:
            return matrices[step] @ vector

        def transpose_step(step: int, vector: np.ndarray) -> np.ndarray:
            return matrices[step].T @ vector

        def observation_tangent(previous: np.ndarray, post: np.ndarray) -> np.ndarray:
            return previous_observation @ previous + post_observation @ post

        def observation_transpose(seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return previous_observation.T @ seed, post_observation.T @ seed

        control = rng.normal(size=self.operator.control_size)
        initial = rng.normal(size=7)
        release_seed = rng.normal(size=5)
        terminal_seed = rng.normal(size=7)
        tangent = propagate_distributed_control_tangent(
            self.operator,
            control,
            total_steps=8,
            release_previous_boundary=4,
            release_post_boundary=5,
            step_tangent=tangent_step,
            observation_tangent=observation_tangent,
            initial_state_tangent=initial,
        )
        adjoint = propagate_distributed_control_adjoint(
            self.operator,
            total_steps=8,
            release_previous_boundary=4,
            release_post_boundary=5,
            step_transpose=transpose_step,
            observation_transpose=observation_transpose,
            release_observation_covector=release_seed,
            terminal_state_covector=terminal_seed,
        )
        left = float(
            release_seed @ tangent.release_observation
            + terminal_seed @ tangent.terminal_state
        )
        right = float(
            control @ adjoint.control_gradient
            + initial @ adjoint.initial_state_covector
        )
        self.assertAlmostEqual(left, right, places=11)

    def test_response_matrix_matches_direct_tangent(self) -> None:
        rng = np.random.default_rng(8304)
        matrices = [rng.normal(size=(7, 7)) / 4.0 for _ in range(6)]
        previous_observation = rng.normal(size=(4, 7))
        post_observation = rng.normal(size=(4, 7))

        arguments = {
            "total_steps": 6,
            "release_previous_boundary": 4,
            "release_post_boundary": 5,
            "step_tangent": lambda step, vector: matrices[step] @ vector,
            "observation_tangent": lambda previous, post: (
                previous_observation @ previous + post_observation @ post
            ),
        }
        response = form_release_response_matrix(self.operator, **arguments)
        direction = rng.normal(size=self.operator.control_size)
        direct = propagate_distributed_control_tangent(
            self.operator, direction, **arguments
        )
        np.testing.assert_allclose(
            response @ direction, direct.release_observation, rtol=2.0e-13
        )


if __name__ == "__main__":
    unittest.main()
