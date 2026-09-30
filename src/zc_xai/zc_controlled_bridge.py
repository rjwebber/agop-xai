"""File-safe Python bridge to the differentiated Zebiak--Cane one-step tools.

The certified Fortran build remains immutable.  This module runs its existing
one-step primal and tangent executables in a private copy of a validated
runtime directory, composes distributed native-state controls at model-step
boundaries, and applies the already validated staggered core4 observation.

The bridge deliberately forms the release response in tangent mode.  For the
intended ranks the control dimension is at most ``3 * 36 = 108``, so forming
all response columns is cheaper and simpler than introducing another reverse
driver for the 2,160-dimensional release observation.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .adjoint_chain import (
    DistributedControlOperator,
    form_release_response_matrix,
)
from .native_observation import (
    DEFAULT_STATE_MANIFEST,
    FrozenCore4ObservationChain,
    PackedZCState,
    load_native_state_layout,
    read_packed_zc_state,
    write_packed_zc_state,
)

TAPE_WIDTH = 3
_REAL_REPLAY_MAX_TOLERANCE = 1.0e-3
_REAL_REPLAY_RELATIVE_L2_TOLERANCE = 1.0e-6
_COMPLEX_REPLAY_MAX_TOLERANCE = 1.0e-4
_COMPLEX_REPLAY_RELATIVE_L2_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class PrimalStepResult:
    """One authentic nonlinear step and its recorded branch tape."""

    state: PackedZCState
    tape: np.ndarray
    wall_seconds: float


@dataclass(frozen=True)
class TangentStepResult:
    """One tangent step, including the simultaneously replayed primal state."""

    state: PackedZCState
    tangent: np.ndarray
    tape: np.ndarray
    wall_seconds: float


@dataclass(frozen=True)
class ZeroControlPath:
    """Boundary states and branch tapes for an unforced reference path.

    ``states[k]`` is boundary ``s_k`` and ``tapes[k]`` belongs to the step
    from ``s_k`` to ``s_{k+1}``.
    """

    states: tuple[PackedZCState, ...]
    tapes: tuple[np.ndarray, ...]
    wall_seconds: float

    @property
    def total_steps(self) -> int:
        return len(self.tapes)


@dataclass(frozen=True)
class ControlledReplay:
    """A nonlinear path with explicit pre-step control insertions.

    ``states`` uses the same boundary convention as :class:`ZeroControlPath`.
    ``step_inputs[k]`` is the state actually supplied to the model after the
    optional increment at boundary ``k``.  Consequently it differs from
    ``states[k]`` only during the controlled window.
    """

    states: tuple[PackedZCState, ...]
    step_inputs: tuple[PackedZCState, ...]
    tapes: tuple[np.ndarray, ...]
    applied_real32_increments: tuple[np.ndarray, ...]
    wall_seconds: float

    @property
    def total_steps(self) -> int:
        return len(self.tapes)


class OneStepRunner(Protocol):
    """Small interface used by the algebraic composition and its tests."""

    def advance(self, state: PackedZCState) -> PrimalStepResult:
        """Advance one nonlinear coupled step."""

    def tangent(
        self, state: PackedZCState, direction: np.ndarray
    ) -> TangentStepResult:
        """Advance one primal state and one real-state tangent together."""


def _positive_steps(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _copy_state(state: PackedZCState) -> PackedZCState:
    return PackedZCState(
        real32=np.array(state.real32, copy=True),
        complex64=np.array(state.complex64, copy=True),
        real64=np.array(state.real64, copy=True),
        integers=np.array(state.integers, copy=True),
        passive_time=np.array(state.passive_time, copy=True),
    )


def packed_states_bitwise_equal(left: PackedZCState, right: PackedZCState) -> bool:
    """Return whether two typed state streams have identical array bytes."""

    return all(
        np.asarray(first).dtype == np.asarray(second).dtype
        and np.asarray(first).shape == np.asarray(second).shape
        and np.asarray(first).tobytes(order="C")
        == np.asarray(second).tobytes(order="C")
        for first, second in zip(
            (
                left.real32,
                left.complex64,
                left.real64,
                left.integers,
                left.passive_time,
            ),
            (
                right.real32,
                right.complex64,
                right.real64,
                right.integers,
                right.passive_time,
            ),
            strict=True,
        )
    )


def _relative_l2_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual) - np.asarray(expected)
    numerator = float(np.linalg.norm(difference.reshape(-1)))
    denominator = max(float(np.linalg.norm(np.asarray(expected).reshape(-1))), 1.0e-300)
    return numerator / denominator


def _verify_tangent_primal_replay(
    actual: PackedZCState,
    expected: PackedZCState,
    *,
    step: int,
) -> None:
    """Apply the same replay tolerances as the curated final3 tangent audit."""

    if not np.array_equal(actual.integers, expected.integers):
        raise RuntimeError(f"tangent primal integer replay differs at step {step}")
    if not np.array_equal(actual.passive_time, expected.passive_time):
        raise RuntimeError(f"tangent primal clock replay differs at step {step}")
    if not np.array_equal(actual.real64, expected.real64):
        raise RuntimeError(f"tangent primal real64 replay differs at step {step}")

    real_difference = np.asarray(actual.real32, dtype=np.float64) - np.asarray(
        expected.real32, dtype=np.float64
    )
    real_max = float(np.max(np.abs(real_difference), initial=0.0))
    real_relative = _relative_l2_difference(
        np.asarray(actual.real32, dtype=np.float64),
        np.asarray(expected.real32, dtype=np.float64),
    )
    complex_difference = np.asarray(actual.complex64, dtype=np.complex128) - np.asarray(
        expected.complex64, dtype=np.complex128
    )
    complex_max = float(np.max(np.abs(complex_difference), initial=0.0))
    complex_relative = _relative_l2_difference(
        np.asarray(actual.complex64, dtype=np.complex128),
        np.asarray(expected.complex64, dtype=np.complex128),
    )
    if (
        real_max > _REAL_REPLAY_MAX_TOLERANCE
        or real_relative > _REAL_REPLAY_RELATIVE_L2_TOLERANCE
        or complex_max > _COMPLEX_REPLAY_MAX_TOLERANCE
        or complex_relative > _COMPLEX_REPLAY_RELATIVE_L2_TOLERANCE
    ):
        raise RuntimeError(
            "tangent primal replay exceeds final3 tolerances at step "
            f"{step}: real max={real_max:.6g}, real rel-L2={real_relative:.6g}, "
            f"complex max={complex_max:.6g}, "
            f"complex rel-L2={complex_relative:.6g}"
        )


class FortranOneStepRunner:
    """Run immutable one-step executables in a private runtime-directory copy."""

    def __init__(
        self,
        runtime_dir: Path,
        *,
        primal_executable: Path | None = None,
        tangent_executable: Path | None = None,
        manifest_path: Path = DEFAULT_STATE_MANIFEST,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.runtime_template = runtime_dir.expanduser().resolve()
        self.primal_executable = self._executable(primal_executable, "primal")
        self.tangent_executable = self._executable(tangent_executable, "tangent")
        self.layout = load_native_state_layout(manifest_path)
        if not self.runtime_template.is_dir():
            raise FileNotFoundError(self.runtime_template)
        for required in ("fc.data", "zeq9fsu.hst", "Data"):
            if not (self.runtime_template / required).exists():
                raise FileNotFoundError(self.runtime_template / required)
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._runtime: Path | None = None

    @staticmethod
    def _executable(path: Path | None, label: str) -> Path | None:
        if path is None:
            return None
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if not os.access(resolved, os.X_OK):
            raise PermissionError(f"{label} executable is not executable: {resolved}")
        return resolved

    def __enter__(self) -> FortranOneStepRunner:
        if self._temporary is not None:
            raise RuntimeError("FortranOneStepRunner is already active")
        self._temporary = tempfile.TemporaryDirectory(prefix="zc-controlled-runtime-")
        self._runtime = Path(self._temporary.name) / "runtime"
        shutil.copytree(self.runtime_template, self._runtime, symlinks=False)
        return self

    def __exit__(self, *_: object) -> None:
        assert self._temporary is not None
        self._temporary.cleanup()
        self._temporary = None
        self._runtime = None

    @property
    def runtime(self) -> Path:
        if self._runtime is None:
            raise RuntimeError("FortranOneStepRunner must be used as a context manager")
        return self._runtime

    def _invoke(self, arguments: list[str]) -> float:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                arguments,
                cwd=self.runtime,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"ZC one-step process exceeded {self.timeout_seconds:g} seconds"
            ) from error
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            stdout_tail = completed.stdout[-2_000:].strip()
            stderr_tail = completed.stderr[-2_000:].strip()
            raise RuntimeError(
                f"ZC one-step process failed with status {completed.returncode}; "
                f"stdout tail={stdout_tail!r}; stderr tail={stderr_tail!r}"
            )
        return elapsed

    @staticmethod
    def _read_tape(path: Path) -> np.ndarray:
        payload = path.read_bytes()
        if len(payload) != 4 * TAPE_WIDTH:
            raise RuntimeError(
                f"one-step tape has {len(payload)} bytes; expected {4 * TAPE_WIDTH}"
            )
        tape = np.frombuffer(payload, dtype="<i4").astype(np.int32, copy=True)
        tape.setflags(write=False)
        return tape

    def advance(self, state: PackedZCState) -> PrimalStepResult:
        if self.primal_executable is None:
            raise RuntimeError("no primal one-step executable was configured")
        with tempfile.TemporaryDirectory(
            prefix="call-primal-", dir=self.runtime
        ) as raw:
            call = Path(raw)
            input_path = call / "input_state.bin"
            output_path = call / "output_state.bin"
            tape_path = call / "tape.bin"
            write_packed_zc_state(input_path, state, layout=self.layout)
            elapsed = self._invoke(
                [
                    str(self.primal_executable),
                    str(input_path),
                    str(output_path),
                    str(tape_path),
                ]
            )
            output = read_packed_zc_state(output_path, layout=self.layout)
            tape = self._read_tape(tape_path)
        return PrimalStepResult(state=output, tape=tape, wall_seconds=elapsed)

    def tangent(
        self, state: PackedZCState, direction: np.ndarray
    ) -> TangentStepResult:
        if self.tangent_executable is None:
            raise RuntimeError("no tangent one-step executable was configured")
        vector = np.asarray(direction, dtype=np.float64)
        if vector.shape != (self.layout.real32_length,):
            raise ValueError(
                f"tangent direction must have shape ({self.layout.real32_length},)"
            )
        if not np.isfinite(vector).all():
            raise ValueError("tangent direction must be finite")
        float32_vector = vector.astype(np.float32)
        if not np.isfinite(float32_vector).all():
            raise ValueError("tangent direction is outside finite float32 range")

        with tempfile.TemporaryDirectory(
            prefix="call-tangent-", dir=self.runtime
        ) as raw:
            call = Path(raw)
            input_path = call / "input_state.bin"
            direction_path = call / "direction.bin"
            output_path = call / "output_state.bin"
            tangent_path = call / "tangent.bin"
            tape_path = call / "tape.bin"
            write_packed_zc_state(input_path, state, layout=self.layout)
            direction_path.write_bytes(
                np.asarray(float32_vector, dtype="<f4").tobytes(order="C")
            )
            elapsed = self._invoke(
                [
                    str(self.tangent_executable),
                    str(input_path),
                    str(direction_path),
                    str(output_path),
                    str(tangent_path),
                    str(tape_path),
                ]
            )
            output = read_packed_zc_state(output_path, layout=self.layout)
            tangent_payload = tangent_path.read_bytes()
            expected_bytes = 4 * self.layout.real32_length
            if len(tangent_payload) != expected_bytes:
                raise RuntimeError(
                    f"one-step tangent has {len(tangent_payload)} bytes; "
                    f"expected {expected_bytes}"
                )
            tangent = np.frombuffer(tangent_payload, dtype="<f4").astype(
                np.float64, copy=True
            )
            if not np.isfinite(tangent).all():
                raise RuntimeError("one-step tangent output is nonfinite")
            tape = self._read_tape(tape_path)
        return TangentStepResult(
            state=output,
            tangent=tangent,
            tape=tape,
            wall_seconds=elapsed,
        )


def _build_zero_control_path_with_runner(
    initial_state: PackedZCState,
    runner: OneStepRunner,
    *,
    total_steps: int,
) -> ZeroControlPath:
    _positive_steps(total_steps, "total_steps")
    states = [_copy_state(initial_state)]
    tapes: list[np.ndarray] = []
    elapsed = 0.0
    state = states[0]
    for _ in range(total_steps):
        result = runner.advance(state)
        state = _copy_state(result.state)
        states.append(state)
        tapes.append(np.array(result.tape, dtype=np.int32, copy=True))
        elapsed += float(result.wall_seconds)
    return ZeroControlPath(
        states=tuple(states), tapes=tuple(tapes), wall_seconds=elapsed
    )


def build_zero_control_path(
    initial_state: PackedZCState,
    runtime_dir: Path,
    primal_executable: Path,
    total_steps: int,
    *,
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
    timeout_seconds: float = 120.0,
) -> ZeroControlPath:
    """Build an isolated authentic unforced path of ``total_steps`` steps."""

    with FortranOneStepRunner(
        runtime_dir,
        primal_executable=primal_executable,
        manifest_path=manifest_path,
        timeout_seconds=timeout_seconds,
    ) as runner:
        return _build_zero_control_path_with_runner(
            initial_state, runner, total_steps=total_steps
        )


def _form_authentic_release_response_with_runner(
    path: ZeroControlPath,
    operator: DistributedControlOperator,
    observation_chain: FrozenCore4ObservationChain,
    runner: OneStepRunner,
    *,
    release: tuple[int, int],
    drop_phase: bool,
) -> np.ndarray:
    previous_boundary, post_boundary = release
    if (
        isinstance(previous_boundary, bool)
        or isinstance(post_boundary, bool)
        or not isinstance(previous_boundary, int)
        or not isinstance(post_boundary, int)
        or previous_boundary < 0
        or post_boundary != previous_boundary + 1
    ):
        raise ValueError("release must name two consecutive nonnegative boundaries")
    if post_boundary > path.total_steps:
        raise ValueError("reference path does not reach the requested release")
    if operator.control_steps > previous_boundary:
        raise ValueError(
            "all controls must end no later than the previous release boundary"
        )
    if operator.state_size != observation_chain.layout.real32_length:
        raise ValueError("control state size differs from the observation layout")
    if len(path.states) != path.total_steps + 1:
        raise ValueError("reference path boundary count is inconsistent")

    post_time = observation_chain.native_time_months(path.states[post_boundary])

    def tangent_step(step: int, direction: np.ndarray) -> np.ndarray:
        result = runner.tangent(path.states[step], direction)
        _verify_tangent_primal_replay(
            result.state, path.states[step + 1], step=step
        )
        if not np.array_equal(result.tape, path.tapes[step]):
            raise RuntimeError(f"tangent branch tape differs at step {step}")
        return result.tangent

    def observe(previous: np.ndarray, post: np.ndarray) -> np.ndarray:
        values = observation_chain.tangent(
            previous,
            post,
            post_step_native_time_months=post_time,
            post_step_td_tangent=0.0,
        )
        return values[:-2] if drop_phase else values

    response = form_release_response_matrix(
        operator,
        total_steps=post_boundary,
        release_previous_boundary=previous_boundary,
        release_post_boundary=post_boundary,
        step_tangent=tangent_step,
        observation_tangent=observe,
    )
    if not np.isfinite(response).all():
        raise RuntimeError("release response contains nonfinite values")
    return response


def form_authentic_release_response(
    path: ZeroControlPath,
    packed_B: np.ndarray,
    temporal_basis: np.ndarray,
    tangent_executable: Path,
    runtime_dir: Path,
    *,
    observation_chain: FrozenCore4ObservationChain,
    control_steps: int = 9,
    release: tuple[int, int] = (9, 10),
    drop_phase: bool = True,
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
    timeout_seconds: float = 120.0,
) -> np.ndarray:
    """Form the authentic standardized release Jacobian ``A``.

    ``packed_B`` may be one constant ``(59148, r)`` map or a coherent
    phase-local stack ``(control_steps, 59148, r)``.  Independently fitted PCA
    maps must not be stacked unless their reduced coordinates have first been
    aligned.  The default removes the two passive annual-phase rows, yielding
    the scientific ``2160 x (temporal_rank * r)`` response.
    """

    _positive_steps(control_steps, "control_steps")
    temporal = np.asarray(temporal_basis, dtype=np.float64)
    if temporal.ndim != 2 or temporal.shape[0] != control_steps:
        raise ValueError("temporal_basis row count must equal control_steps")
    operator = DistributedControlOperator(
        spatial_map=np.asarray(packed_B, dtype=np.float64),
        temporal_basis=temporal,
    )
    with FortranOneStepRunner(
        runtime_dir,
        tangent_executable=tangent_executable,
        manifest_path=manifest_path,
        timeout_seconds=timeout_seconds,
    ) as runner:
        return _form_authentic_release_response_with_runner(
            path,
            operator,
            observation_chain,
            runner,
            release=release,
            drop_phase=drop_phase,
        )


def _controlled_input_state(
    state: PackedZCState,
    increment: np.ndarray,
) -> PackedZCState:
    values = np.asarray(increment, dtype=np.float64)
    real = np.asarray(state.real32)
    if values.shape != real.shape:
        raise ValueError("native control increment has the wrong packed shape")
    if not np.isfinite(values).all():
        raise ValueError("native control increment must be finite")
    controlled64 = real.astype(np.float64) + values
    if not np.isfinite(controlled64).all():
        raise ValueError("controlled native state is nonfinite")
    controlled = controlled64.astype(np.float32)
    if not np.isfinite(controlled).all():
        raise ValueError("controlled native state is outside finite float32 range")
    return PackedZCState(
        real32=controlled,
        complex64=np.array(state.complex64, copy=True),
        real64=np.array(state.real64, copy=True),
        integers=np.array(state.integers, copy=True),
        passive_time=np.array(state.passive_time, copy=True),
    )


def _replay_control_with_runner(
    initial_state: PackedZCState,
    operator: DistributedControlOperator,
    control: np.ndarray,
    runner: OneStepRunner,
    *,
    total_steps: int,
) -> ControlledReplay:
    _positive_steps(total_steps, "total_steps")
    if total_steps < operator.control_steps:
        raise ValueError("total_steps cannot end inside the controlled window")
    coefficients = np.asarray(control, dtype=np.float64)
    operator.coefficient_matrix(coefficients)

    states = [_copy_state(initial_state)]
    step_inputs: list[PackedZCState] = []
    tapes: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    elapsed = 0.0
    state = states[0]
    for step in range(total_steps):
        if step < operator.control_steps:
            increment = operator.apply_at_step(coefficients, step)
        else:
            increment = np.zeros(operator.state_size, dtype=np.float64)
        step_input = _controlled_input_state(state, increment)
        result = runner.advance(step_input)
        step_inputs.append(step_input)
        increments.append(np.asarray(increment, dtype=np.float64).copy())
        tapes.append(np.array(result.tape, dtype=np.int32, copy=True))
        state = _copy_state(result.state)
        states.append(state)
        elapsed += float(result.wall_seconds)
    return ControlledReplay(
        states=tuple(states),
        step_inputs=tuple(step_inputs),
        tapes=tuple(tapes),
        applied_real32_increments=tuple(increments),
        wall_seconds=elapsed,
    )


def replay_control(
    initial_state: PackedZCState,
    operator: DistributedControlOperator,
    control: np.ndarray,
    runtime_dir: Path,
    primal_executable: Path,
    *,
    total_steps: int = 40,
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
    timeout_seconds: float = 120.0,
) -> ControlledReplay:
    """Apply a distributed control and replay the authentic nonlinear model."""

    with FortranOneStepRunner(
        runtime_dir,
        primal_executable=primal_executable,
        manifest_path=manifest_path,
        timeout_seconds=timeout_seconds,
    ) as runner:
        return _replay_control_with_runner(
            initial_state,
            operator,
            control,
            runner,
            total_steps=total_steps,
        )


def verify_zero_control_replay(
    baseline: ZeroControlPath,
    replay: ControlledReplay,
) -> None:
    """Require exact state and branch equality for an explicit zero control."""

    if replay.total_steps != baseline.total_steps:
        raise ValueError("baseline and replay lengths differ")
    for boundary, (actual, expected) in enumerate(
        zip(replay.states, baseline.states, strict=True)
    ):
        if not packed_states_bitwise_equal(actual, expected):
            raise RuntimeError(
                f"zero-control state is not bitwise identical at boundary {boundary}"
            )
    for step, (actual, expected) in enumerate(
        zip(replay.tapes, baseline.tapes, strict=True)
    ):
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"zero-control branch tape differs at step {step}")
