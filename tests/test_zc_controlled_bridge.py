"""Tests for the file-safe controlled ZC one-step bridge."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adjoint.balanced_control_map import BalancedControlMap
from zc_xai.adjoint_chain import DistributedControlOperator
from zc_xai.native_observation import (
    FrozenCore4ObservationChain,
    PackedZCState,
    read_packed_zc_state,
)
from zc_xai.zc_controlled_bridge import (
    PrimalStepResult,
    TangentStepResult,
    _build_zero_control_path_with_runner,
    _form_authentic_release_response_with_runner,
    _replay_control_with_runner,
    build_zero_control_path,
    form_authentic_release_response,
    packed_states_bitwise_equal,
    replay_control,
    verify_zero_control_replay,
)

ROOT = Path(__file__).resolve().parents[1]


def small_state(values: np.ndarray, boundary: int = 0) -> PackedZCState:
    return PackedZCState(
        real32=np.asarray(values, dtype=np.float32),
        complex64=np.asarray((1.0 + 2.0j,), dtype=np.complex64),
        real64=np.asarray((3.0,), dtype=np.float64),
        integers=np.asarray((boundary,), dtype=np.int32),
        passive_time=np.asarray((float(boundary),), dtype=np.float32),
    )


class LinearStepRunner:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def _result(self, state: PackedZCState) -> tuple[PackedZCState, np.ndarray]:
        boundary = int(state.integers[0])
        output = small_state(self.matrix @ state.real32, boundary + 1)
        tape = np.asarray((boundary % 2, boundary + 3, 7), dtype=np.int32)
        return output, tape

    def advance(self, state: PackedZCState) -> PrimalStepResult:
        output, tape = self._result(state)
        return PrimalStepResult(output, tape, 0.01)

    def tangent(
        self, state: PackedZCState, direction: np.ndarray
    ) -> TangentStepResult:
        output, tape = self._result(state)
        return TangentStepResult(
            output,
            self.matrix @ np.asarray(direction, dtype=np.float64),
            tape,
            0.02,
        )


class SmallObservationChain:
    def __init__(self, state_size: int) -> None:
        self.layout = SimpleNamespace(real32_length=state_size)

    @staticmethod
    def native_time_months(state: PackedZCState) -> float:
        return float(state.passive_time[0])

    @staticmethod
    def tangent(
        previous: np.ndarray,
        post: np.ndarray,
        *,
        post_step_native_time_months: float,
        post_step_td_tangent: float,
    ) -> np.ndarray:
        del post_step_native_time_months, post_step_td_tangent
        return np.asarray(
            (
                previous[0] + 2.0 * post[0],
                previous[1] - post[1],
                0.0,
                0.0,
            )
        )


class ControlledBridgeUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.asarray(
            ((1.2, -0.1, 0.0), (0.2, 0.8, 0.1), (0.0, -0.2, 0.9))
        )
        self.runner = LinearStepRunner(self.matrix)
        self.initial = small_state(np.asarray((0.4, -0.2, 0.7)))

    def test_zero_path_and_explicit_zero_control_are_bitwise_equal(self) -> None:
        baseline = _build_zero_control_path_with_runner(
            self.initial, self.runner, total_steps=5
        )
        operator = DistributedControlOperator(
            np.asarray(((1.0,), (0.5,), (-0.25,))),
            np.asarray(((0.4,), (0.6,))),
        )
        replay = _replay_control_with_runner(
            self.initial,
            operator,
            np.zeros(1),
            self.runner,
            total_steps=5,
        )
        verify_zero_control_replay(baseline, replay)
        self.assertTrue(
            all(
                packed_states_bitwise_equal(actual, expected)
                for actual, expected in zip(
                    replay.step_inputs, baseline.states[:-1], strict=True
                )
            )
        )

    def test_release_response_matches_manual_distributed_tangent(self) -> None:
        baseline = _build_zero_control_path_with_runner(
            self.initial, self.runner, total_steps=3
        )
        spatial = np.asarray(
            ((1.0, 0.2), (0.5, -0.3), (-0.25, 0.4)), dtype=np.float64
        )
        temporal = np.asarray(((0.25,), (0.75,)), dtype=np.float64)
        operator = DistributedControlOperator(spatial, temporal)
        response = _form_authentic_release_response_with_runner(
            baseline,
            operator,
            SmallObservationChain(3),  # type: ignore[arg-type]
            self.runner,
            release=(2, 3),
            drop_phase=True,
        )

        expected_columns = []
        for column in range(operator.control_size):
            control = np.zeros(operator.control_size)
            control[column] = 1.0
            tangent = np.zeros(3)
            boundaries = [tangent.copy()]
            for step in range(3):
                if step < operator.control_steps:
                    tangent += operator.apply_at_step(control, step)
                tangent = self.matrix @ tangent
                boundaries.append(tangent.copy())
            expected_columns.append(
                SmallObservationChain.tangent(
                    boundaries[2],
                    boundaries[3],
                    post_step_native_time_months=3.0,
                    post_step_td_tangent=0.0,
                )[:-2]
            )
        np.testing.assert_allclose(
            response, np.column_stack(expected_columns), rtol=1.0e-14, atol=1.0e-14
        )

    def test_nonlinear_replay_inserts_before_each_controlled_step(self) -> None:
        spatial = np.asarray(((1.0,), (-0.5,), (0.25,)))
        temporal = np.asarray(((0.2,), (0.8,)))
        operator = DistributedControlOperator(spatial, temporal)
        replay = _replay_control_with_runner(
            self.initial,
            operator,
            np.asarray((1.5,)),
            self.runner,
            total_steps=4,
        )

        expected = self.initial.real32.astype(np.float64)
        for step in range(4):
            increment = (
                operator.apply_at_step(np.asarray((1.5,)), step)
                if step < 2
                else np.zeros(3)
            )
            expected_input = (expected + increment).astype(np.float32)
            np.testing.assert_array_equal(
                replay.step_inputs[step].real32, expected_input
            )
            expected = (self.matrix @ expected_input).astype(np.float32)
            np.testing.assert_array_equal(replay.states[step + 1].real32, expected)


class ControlledBridgeAuthenticSmokeTest(unittest.TestCase):
    def test_two_step_path_response_and_zero_replay(self) -> None:
        runtime = (
            ROOT
            / "outputs/zc_adjoint/kernel_replay_validation_run23_final3/"
            "neutral_member_04_040step/kernel"
        )
        primal = ROOT / "adjoint/controlled_window/build/zc_one_step"
        tangent = ROOT / "adjoint/controlled_window/build/zc_one_step_tangent"
        map_path = (
            ROOT
            / "outputs/zc_balanced_control_map/training-phase26-rank36/"
            "balanced_control_map.npz"
        )
        required = (
            runtime / "kernel_initial_state.bin",
            runtime / "fc.data",
            runtime / "zeq9fsu.hst",
            primal,
            tangent,
            map_path,
            ROOT / "data/processed/zc-v3/metadata.json",
        )
        if not all(path.exists() for path in required) or not all(
            os.access(path, os.X_OK) for path in (primal, tangent)
        ):
            self.skipTest("local authentic ZC bridge inputs are unavailable")

        initial = read_packed_zc_state(runtime / "kernel_initial_state.bin")
        baseline = build_zero_control_path(initial, runtime, primal, 2)
        chain = FrozenCore4ObservationChain.from_recorded_artifact(
            ROOT / "data/processed/zc-v3",
            ROOT / "artifacts/zc-v3",
            "models/core4/cnn/lead-10m/years-10000/seed-000042",
        )
        control_map, _ = BalancedControlMap.load(map_path)
        packed_B = control_map.packed_control_matrix()[:, :1]
        response = form_authentic_release_response(
            baseline,
            packed_B,
            np.ones((1, 1)),
            tangent,
            runtime,
            observation_chain=chain,
            control_steps=1,
            release=(1, 2),
        )
        self.assertEqual(response.shape, (2160, 1))
        self.assertTrue(np.isfinite(response).all())
        self.assertGreater(float(np.linalg.norm(response)), 0.0)

        operator = DistributedControlOperator(packed_B, np.ones((1, 1)))
        zero = replay_control(
            initial,
            operator,
            np.zeros(1),
            runtime,
            primal,
            total_steps=2,
        )
        verify_zero_control_replay(baseline, zero)


if __name__ == "__main__":
    unittest.main()
