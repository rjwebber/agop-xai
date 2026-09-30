from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from zc_xai.adjoint_chain import DistributedControlOperator
from zc_xai.data import Standardizer
from zc_xai.native_observation import (
    CORE4_FEATURES,
    FrozenCore4ObservationChain,
    PackedZCState,
    read_packed_zc_state,
)
from zc_xai.zc_controlled_adjoint_bridge import (
    AdjointStepResult,
    FortranOneStepAdjointRunner,
    reduced_control_gradient,
    reverse_controlled_release_scalar_with_runner,
)
from zc_xai.zc_controlled_bridge import (
    ControlledReplay,
    FortranOneStepRunner,
    _verify_tangent_primal_replay,
    packed_states_bitwise_equal,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    REPOSITORY_ROOT / "outputs/zc_adjoint/kernel_replay_validation_run23_final3/"
    "neutral_member_04_040step/kernel"
)
MATCHED_BUILD = REPOSITORY_ROOT / "adjoint/controlled_reverse/build-v5"
PRIMAL = MATCHED_BUILD / "zc_one_step_primal"
TANGENT = MATCHED_BUILD / "zc_one_step_tangent"
ADJOINT = MATCHED_BUILD / "zc_one_step_adjoint"
AUTHENTIC_STATE = RUNTIME / "kernel_initial_state.bin"


def _chain() -> FrozenCore4ObservationChain:
    return FrozenCore4ObservationChain(
        Standardizer(
            mean=np.zeros(CORE4_FEATURES, dtype=np.float32),
            scale=np.ones(CORE4_FEATURES, dtype=np.float32),
            scale_floor=1.0e-6,
            count=2,
        )
    )


def _state(chain: FrozenCore4ObservationChain, boundary: int) -> PackedZCState:
    layout = chain.layout
    integers = np.zeros(layout.integer_length, dtype=np.int32)
    integers[0] = boundary
    passive_time = np.zeros(layout.passive_time_length, dtype=np.float32)
    passive_time[0] = float(boundary)
    return PackedZCState(
        real32=np.zeros(layout.real32_length, dtype=np.float32),
        complex64=np.zeros(layout.complex64_length, dtype=np.complex64),
        real64=np.zeros(layout.real64_length, dtype=np.float64),
        integers=integers,
        passive_time=passive_time,
    )


class _DiagonalStepRunner:
    def __init__(
        self,
        outputs: tuple[PackedZCState, ...],
        tapes: tuple[np.ndarray, ...],
        scales: np.ndarray,
    ) -> None:
        self.outputs = outputs
        self.tapes = tapes
        self.scales = scales
        self.steps: list[int] = []

    def transpose(
        self,
        state: PackedZCState,
        expected_output_state: PackedZCState,
        output_covector: np.ndarray,
    ) -> AdjointStepResult:
        step = int(state.integers[0])
        self.steps.append(step)
        if expected_output_state is not self.outputs[step]:
            raise AssertionError("test passed the wrong reference output")
        return AdjointStepResult(
            state=self.outputs[step],
            input_covector=self.scales[step] * output_covector,
            tape=self.tapes[step],
            wall_seconds=0.01,
        )


class ControlledReleaseCompositionTests(unittest.TestCase):
    def test_one_reverse_sweep_returns_all_nine_control_gradients(self) -> None:
        chain = _chain()
        states = tuple(_state(chain, boundary) for boundary in range(11))
        tapes = tuple(np.asarray([2, 3, 1], dtype=np.int32) for _ in range(10))
        zeros = tuple(np.zeros(chain.layout.real32_length) for _ in range(10))
        replay = ControlledReplay(
            states=states,
            step_inputs=states[:-1],
            tapes=tapes,
            applied_real32_increments=zeros,
            wall_seconds=0.0,
        )
        scales = np.linspace(0.8, 1.2, 10)
        runner = _DiagonalStepRunner(states[1:], tapes, scales)
        rng = np.random.default_rng(122)
        release_seed = rng.normal(size=CORE4_FEATURES - 2)

        result = reverse_controlled_release_scalar_with_runner(
            replay,
            chain,
            release_seed,
            runner,
        )

        padded = np.concatenate((release_seed, np.zeros(2)))
        transpose = chain.transpose(
            padded,
            post_step_native_time_months=chain.native_time_months(states[10]),
        )
        previous_seed, post_seed = transpose.scientific_control_pair()
        expected = np.zeros_like(result.native_control_covectors)
        covector = post_seed.copy()
        for step in range(9, -1, -1):
            covector = scales[step] * covector
            if step < 9:
                expected[step] = covector
            if step == 9:
                covector += previous_seed
        np.testing.assert_allclose(result.native_control_covectors, expected)
        np.testing.assert_allclose(result.initial_state_covector, covector)
        self.assertEqual(runner.steps, list(range(9, -1, -1)))
        self.assertAlmostEqual(result.wall_seconds, 0.1)

        spatial = np.zeros((9, chain.layout.real32_length, 1))
        selected = rng.choice(chain.layout.real32_length, 16, replace=False)
        spatial[:, selected, 0] = rng.normal(size=(9, selected.size))
        operator = DistributedControlOperator(spatial, np.eye(9))
        control = rng.normal(size=9)
        gradient = reduced_control_gradient(result.native_control_covectors, operator)
        left = 0.0
        for step in range(9):
            left += float(
                result.native_control_covectors[step]
                @ operator.apply_at_step(control, step)
            )
        self.assertAlmostEqual(left, float(gradient @ control), places=11)

    def test_rejects_active_phase_cotangent(self) -> None:
        chain = _chain()
        states = tuple(_state(chain, boundary) for boundary in range(11))
        tapes = tuple(np.zeros(3, dtype=np.int32) for _ in range(10))
        replay = ControlledReplay(
            states,
            states[:-1],
            tapes,
            tuple(np.zeros(chain.layout.real32_length) for _ in range(10)),
            0.0,
        )
        runner = _DiagonalStepRunner(states[1:], tapes, np.ones(10))
        seed = np.zeros(CORE4_FEATURES)
        seed[-1] = 1.0
        with self.assertRaisesRegex(ValueError, "phase cotangents are passive"):
            reverse_controlled_release_scalar_with_runner(replay, chain, seed, runner)


@unittest.skipUnless(
    RUNTIME.is_dir()
    and PRIMAL.is_file()
    and TANGENT.is_file()
    and ADJOINT.is_file()
    and AUTHENTIC_STATE.is_file(),
    "authentic one-step derivative artifacts are unavailable",
)
class AuthenticOneStepAdjointTests(unittest.TestCase):
    def test_tangent_adjoint_dot_product(self) -> None:
        state = read_packed_zc_state(AUTHENTIC_STATE)
        rng = np.random.default_rng(20260907)
        direction = np.zeros(state.real32.size)
        seed = np.zeros(state.real32.size)
        direction_indices = rng.choice(direction.size, 256, replace=False)
        seed_indices = rng.choice(seed.size, 256, replace=False)
        direction[direction_indices] = rng.normal(size=direction_indices.size)
        seed[seed_indices] = rng.normal(size=seed_indices.size)
        with FortranOneStepRunner(
            RUNTIME,
            primal_executable=PRIMAL,
            tangent_executable=TANGENT,
        ) as tangent_runner:
            primal = tangent_runner.advance(state)
            tangent = tangent_runner.tangent(state, direction)
        with FortranOneStepAdjointRunner(RUNTIME, ADJOINT) as adjoint_runner:
            adjoint = adjoint_runner.transpose(state, primal.state, seed)
        _verify_tangent_primal_replay(tangent.state, primal.state, step=0)
        self.assertTrue(packed_states_bitwise_equal(adjoint.state, primal.state))
        self.assertTrue(np.array_equal(adjoint.tape, primal.tape))
        self.assertTrue(np.array_equal(tangent.tape, primal.tape))
        left = float(tangent.tangent @ seed)
        right = float(direction @ adjoint.input_covector)
        relative = abs(left - right) / max(abs(left), abs(right), 1.0e-30)
        # Both differentiated executables use the model's native REAL
        # (float32) arithmetic; this is a roundoff-level transpose check.
        self.assertLess(relative, 5.0e-6)


if __name__ == "__main__":
    unittest.main()
