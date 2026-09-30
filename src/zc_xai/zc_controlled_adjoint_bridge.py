"""Exact reverse bridge for controls inserted along an authentic ZC replay.

The Tapenade kernel differentiates one explicit coupled transition.  This
module composes those one-step transposes backward over an already generated
``ControlledReplay`` and exposes the native cotangent at every intervention
boundary.  Thus one reverse sweep returns the derivative with respect to all
nine independent interventions; covariance factors can subsequently pull
those native cotangents into coefficient space without another model run.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .adjoint_chain import DistributedControlOperator
from .native_observation import (
    CORE4_FEATURES,
    DEFAULT_STATE_MANIFEST,
    FrozenCore4ObservationChain,
    PackedZCState,
    read_packed_zc_state,
    write_packed_zc_state,
)
from .zc_controlled_bridge import (
    ControlledReplay,
    FortranOneStepRunner,
    _verify_tangent_primal_replay,
)


@dataclass(frozen=True)
class AdjointStepResult:
    """One exact one-step transpose and its simultaneously replayed primal."""

    state: PackedZCState
    input_covector: np.ndarray
    tape: np.ndarray
    wall_seconds: float


@dataclass(frozen=True)
class ControlledReleaseAdjoint:
    """Native gradients of a release scalar along a controlled trajectory."""

    native_control_covectors: np.ndarray
    initial_state_covector: np.ndarray
    wall_seconds: float


class OneStepTransposeRunner(Protocol):
    """Interface needed by the controlled-path reverse composition."""

    def transpose(
        self,
        state: PackedZCState,
        expected_output_state: PackedZCState,
        output_covector: np.ndarray,
    ) -> AdjointStepResult:
        """Apply the transpose of one model transition at ``state``."""


class FortranOneStepAdjointRunner(FortranOneStepRunner):
    """Run the isolated one-step Tapenade reverse in a private runtime copy."""

    def __init__(
        self,
        runtime_dir: Path,
        adjoint_executable: Path,
        *,
        manifest_path: Path = DEFAULT_STATE_MANIFEST,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            runtime_dir,
            manifest_path=manifest_path,
            timeout_seconds=timeout_seconds,
        )
        self.adjoint_executable = self._executable(adjoint_executable, "adjoint")
        assert self.adjoint_executable is not None

    def transpose(
        self,
        state: PackedZCState,
        expected_output_state: PackedZCState,
        output_covector: np.ndarray,
    ) -> AdjointStepResult:
        vector = np.asarray(output_covector, dtype=np.float64)
        expected_shape = (self.layout.real32_length,)
        if vector.shape != expected_shape:
            raise ValueError(
                f"output_covector must have shape ({self.layout.real32_length},)"
            )
        if not np.isfinite(vector).all():
            raise ValueError("output_covector must be finite")
        float32_vector = vector.astype(np.float32)
        if not np.isfinite(float32_vector).all():
            raise ValueError("output_covector is outside finite float32 range")

        with tempfile.TemporaryDirectory(
            prefix="call-adjoint-", dir=self.runtime
        ) as raw:
            call = Path(raw)
            input_path = call / "input_state.bin"
            seed_path = call / "output_cotangent.bin"
            reference_path = call / "reference_output_state.bin"
            output_path = call / "output_state.bin"
            vjp_path = call / "input_cotangent.bin"
            tape_path = call / "tape.bin"
            write_packed_zc_state(input_path, state, layout=self.layout)
            write_packed_zc_state(
                reference_path, expected_output_state, layout=self.layout
            )
            seed_path.write_bytes(
                np.asarray(float32_vector, dtype="<f4").tobytes(order="C")
            )
            elapsed = self._invoke(
                [
                    str(self.adjoint_executable),
                    str(input_path),
                    str(seed_path),
                    str(reference_path),
                    str(output_path),
                    str(vjp_path),
                    str(tape_path),
                ]
            )
            output = read_packed_zc_state(output_path, layout=self.layout)
            payload = vjp_path.read_bytes()
            expected_bytes = 4 * self.layout.real32_length
            if len(payload) != expected_bytes:
                raise RuntimeError(
                    f"one-step adjoint has {len(payload)} bytes; "
                    f"expected {expected_bytes}"
                )
            input_covector = np.frombuffer(payload, dtype="<f4").astype(
                np.float64, copy=True
            )
            if not np.isfinite(input_covector).all():
                raise RuntimeError("one-step adjoint output is nonfinite")
            tape = self._read_tape(tape_path)
        return AdjointStepResult(
            state=output,
            input_covector=input_covector,
            tape=tape,
            wall_seconds=elapsed,
        )


def _scientific_release_seed(
    cotangent: np.ndarray,
    observation_chain: FrozenCore4ObservationChain,
    *,
    post_step_native_time_months: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(cotangent, dtype=np.float64)
    scientific_size = CORE4_FEATURES - 2
    if values.shape == (scientific_size,):
        standardized = np.concatenate((values, np.zeros(2, dtype=np.float64)))
    elif values.shape == (CORE4_FEATURES,):
        if np.any(values[-2:] != 0.0):
            raise ValueError(
                "annual-phase cotangents are passive; the final two entries "
                "must be zero"
            )
        standardized = values
    else:
        raise ValueError(
            f"release cotangent must have shape ({scientific_size},) or "
            f"({CORE4_FEATURES},)"
        )
    transpose = observation_chain.transpose(
        standardized,
        post_step_native_time_months=post_step_native_time_months,
    )
    return transpose.scientific_control_pair()


def reverse_controlled_release_scalar_with_runner(
    replay: ControlledReplay,
    observation_chain: FrozenCore4ObservationChain,
    release_observation_covector: np.ndarray,
    runner: OneStepTransposeRunner,
    *,
    control_steps: int = 9,
    release: tuple[int, int] = (9, 10),
) -> ControlledReleaseAdjoint:
    """Differentiate one release scalar with respect to every intervention.

    Controls are additive updates to the REAL packed state immediately before
    transitions ``0, ..., control_steps - 1``.  The returned row ``k`` is the
    exact native-state gradient with respect to the complete intervention at
    step ``k``.  The default performs ten one-step reverse calls and returns
    all nine intervention gradients.
    """

    previous_boundary, post_boundary = release
    if (
        isinstance(previous_boundary, bool)
        or isinstance(post_boundary, bool)
        or not isinstance(previous_boundary, int)
        or not isinstance(post_boundary, int)
        or previous_boundary < 0
        or post_boundary != previous_boundary + 1
    ):
        raise ValueError("release must name adjacent nonnegative boundaries")
    if isinstance(control_steps, bool) or not isinstance(control_steps, int):
        raise TypeError("control_steps must be an integer")
    if not 1 <= control_steps <= previous_boundary:
        raise ValueError("controls must end no later than the release boundary")
    if replay.total_steps < post_boundary:
        raise ValueError("controlled replay does not reach the release")
    if (
        len(replay.states) != replay.total_steps + 1
        or len(replay.step_inputs) != replay.total_steps
        or len(replay.tapes) != replay.total_steps
    ):
        raise ValueError("controlled replay arrays have inconsistent lengths")
    state_size = observation_chain.layout.real32_length
    if any(np.asarray(state.real32).shape != (state_size,) for state in replay.states):
        raise ValueError("controlled replay uses a different native-state layout")

    post_time = observation_chain.native_time_months(replay.states[post_boundary])
    previous_seed, post_seed = _scientific_release_seed(
        release_observation_covector,
        observation_chain,
        post_step_native_time_months=post_time,
    )
    state_covector = np.array(post_seed, dtype=np.float64, copy=True)
    native = np.zeros((control_steps, state_size), dtype=np.float64)
    elapsed = 0.0
    for step in range(post_boundary - 1, -1, -1):
        result = runner.transpose(
            replay.step_inputs[step], replay.states[step + 1], state_covector
        )
        _verify_tangent_primal_replay(
            result.state,
            replay.states[step + 1],
            step=step,
        )
        if not np.array_equal(result.tape, replay.tapes[step]):
            raise RuntimeError(f"adjoint branch tape differs at step {step}")
        state_covector = np.array(
            result.input_covector, dtype=np.float64, copy=True
        )
        if step < control_steps:
            native[step] = state_covector
        if step == previous_boundary:
            state_covector += previous_seed
        elapsed += float(result.wall_seconds)

    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("invalid accumulated adjoint wall time")
    return ControlledReleaseAdjoint(
        native_control_covectors=native,
        initial_state_covector=state_covector,
        wall_seconds=elapsed,
    )


def reverse_controlled_release_scalar(
    replay: ControlledReplay,
    observation_chain: FrozenCore4ObservationChain,
    release_observation_covector: np.ndarray,
    runtime_dir: Path,
    adjoint_executable: Path,
    *,
    control_steps: int = 9,
    release: tuple[int, int] = (9, 10),
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
    timeout_seconds: float = 120.0,
) -> ControlledReleaseAdjoint:
    """Run the authentic process-isolated reverse over a controlled replay."""

    with FortranOneStepAdjointRunner(
        runtime_dir,
        adjoint_executable,
        manifest_path=manifest_path,
        timeout_seconds=timeout_seconds,
    ) as runner:
        return reverse_controlled_release_scalar_with_runner(
            replay,
            observation_chain,
            release_observation_covector,
            runner,
            control_steps=control_steps,
            release=release,
        )


def reduced_control_gradient(
    native_control_covectors: np.ndarray,
    operator: DistributedControlOperator,
) -> np.ndarray:
    """Pull native intervention gradients through a reduced control map."""

    native = np.asarray(native_control_covectors, dtype=np.float64)
    if native.shape != (operator.control_steps, operator.state_size):
        raise ValueError(
            "native_control_covectors must match the operator's steps and state"
        )
    if not np.isfinite(native).all():
        raise ValueError("native_control_covectors must be finite")
    result = np.zeros(operator.control_size, dtype=np.float64)
    for step in range(operator.control_steps):
        result += operator.transpose_at_step(native[step], step)
    return result
