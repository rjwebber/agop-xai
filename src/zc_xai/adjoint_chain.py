"""Compose balanced native controls with a discrete model tangent and adjoint.

This module contains no Zebiak--Cane equations.  It encodes the chain rule at
the boundaries between already differentiated model steps.  Keeping this
composition separate makes the time staggering of the fresh core4 record and
the location of incremental-analysis updates explicit and independently
testable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .agop_control import half_cosine_iau_weights

StepLinearMap = Callable[[int, np.ndarray], np.ndarray]
ObservationTangent = Callable[[np.ndarray, np.ndarray], np.ndarray]
ObservationTranspose = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def three_mode_half_cosine_iau_basis(step_count: int) -> np.ndarray:
    """Return a smooth three-column temporal basis for gradual controls.

    Column zero is the positive half-cosine incremental-analysis schedule and
    sums to one.  The other columns are smooth early/late and
    middle-versus-edge timing contrasts.  They each sum to zero, are
    orthogonal to column zero and to one another, and have the same Euclidean
    norm as column zero.  Therefore

    ``W.T @ W / ||W[:, 0]||**2 == I``.

    With a `(3, spatial_rank)` coefficient array, ordinary coefficient norm is
    consequently the normalized time-integrated reduced-control action.  The
    two zero-sum columns can change when an increment is applied without
    silently changing its net prescribed dose.
    """

    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise TypeError("step_count must be an integer")
    if step_count < 4:
        raise ValueError("three temporal modes require at least four steps")

    leading = half_cosine_iau_weights(step_count)
    time = np.linspace(-1.0, 1.0, step_count, dtype=np.float64)
    constraints = np.column_stack((np.ones(step_count), leading))
    projector = np.eye(step_count) - constraints @ np.linalg.solve(
        constraints.T @ constraints, constraints.T
    )
    raw_candidates = (
        time,
        time * time,
        time * time * time,
        np.sin(np.pi * time),
        np.cos(np.pi * time),
    )

    contrasts: list[np.ndarray] = []
    for candidate in raw_candidates:
        vector = np.asarray(projector @ candidate, dtype=np.float64)
        for previous in contrasts:
            vector -= previous * float(previous @ vector) / float(previous @ previous)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm):
            raise RuntimeError("smooth temporal candidate is nonfinite")
        if norm <= 100.0 * np.finfo(np.float64).eps:
            continue
        contrasts.append(vector / norm)
        if len(contrasts) == 2:
            break
    if len(contrasts) != 2:
        raise RuntimeError("smooth temporal candidates are rank deficient")

    leading_norm = float(np.linalg.norm(leading))
    result = np.column_stack(
        (leading, leading_norm * contrasts[0], leading_norm * contrasts[1])
    )
    return result


@dataclass(frozen=True)
class DistributedControlOperator:
    """A covariance-scaled spatial map distributed over model steps.

    ``spatial_map`` is either one packed native matrix ``B`` with shape
    `(state_size, spatial_rank)` or a phase-local stack with shape
    `(control_steps, state_size, spatial_rank)`.  ``temporal_basis`` has shape
    `(control_steps, temporal_rank)`.  A flattened control is interpreted in
    temporal-major order as `(temporal_rank, spatial_rank)`.
    """

    spatial_map: np.ndarray
    temporal_basis: np.ndarray

    def __post_init__(self) -> None:
        spatial = np.asarray(self.spatial_map, dtype=np.float64)
        temporal = np.asarray(self.temporal_basis, dtype=np.float64)
        if spatial.ndim not in {2, 3} or any(length == 0 for length in spatial.shape):
            raise ValueError("spatial_map must be one matrix or a nonempty stack")
        if temporal.ndim != 2 or not temporal.shape[0] or not temporal.shape[1]:
            raise ValueError("temporal_basis must be a nonempty matrix")
        if not np.isfinite(spatial).all() or not np.isfinite(temporal).all():
            raise ValueError("distributed-control factors must be finite")
        if spatial.ndim == 3 and spatial.shape[0] != temporal.shape[0]:
            raise ValueError("phase-local spatial maps must match control_steps")
        object.__setattr__(self, "spatial_map", spatial)
        object.__setattr__(self, "temporal_basis", temporal)

    @property
    def state_size(self) -> int:
        return self.spatial_map.shape[-2]

    @property
    def spatial_rank(self) -> int:
        return self.spatial_map.shape[-1]

    @property
    def control_steps(self) -> int:
        return self.temporal_basis.shape[0]

    @property
    def temporal_rank(self) -> int:
        return self.temporal_basis.shape[1]

    @property
    def control_size(self) -> int:
        return self.spatial_rank * self.temporal_rank

    def coefficient_matrix(self, control: np.ndarray) -> np.ndarray:
        values = np.asarray(control, dtype=np.float64)
        if values.shape != (self.control_size,):
            raise ValueError(f"control must have shape ({self.control_size},)")
        if not np.isfinite(values).all():
            raise ValueError("control must be finite")
        return values.reshape(self.temporal_rank, self.spatial_rank)

    def apply_at_step(self, control: np.ndarray, step: int) -> np.ndarray:
        """Return the packed native increment inserted before one model step."""

        if not 0 <= step < self.control_steps:
            raise IndexError("control step is outside the temporal basis")
        coefficients = self.temporal_basis[step] @ self.coefficient_matrix(control)
        return self._spatial_map_at_step(step) @ coefficients

    def transpose_at_step(self, native_covector: np.ndarray, step: int) -> np.ndarray:
        """Apply the exact transpose of :meth:`apply_at_step`."""

        if not 0 <= step < self.control_steps:
            raise IndexError("control step is outside the temporal basis")
        covector = np.asarray(native_covector, dtype=np.float64)
        if covector.shape != (self.state_size,):
            raise ValueError(f"native_covector must have shape ({self.state_size},)")
        if not np.isfinite(covector).all():
            raise ValueError("native_covector must be finite")
        spatial_covector = self._spatial_map_at_step(step).T @ covector
        return np.outer(self.temporal_basis[step], spatial_covector).reshape(-1)

    def _spatial_map_at_step(self, step: int) -> np.ndarray:
        if self.spatial_map.ndim == 2:
            return self.spatial_map
        return self.spatial_map[step]


@dataclass(frozen=True)
class TangentChainResult:
    """Release observation and terminal state from one tangent propagation."""

    release_observation: np.ndarray
    terminal_state: np.ndarray


@dataclass(frozen=True)
class AdjointChainResult:
    """Control and initial-state covectors from one reverse propagation."""

    control_gradient: np.ndarray
    initial_state_covector: np.ndarray


def propagate_distributed_control_tangent(
    operator: DistributedControlOperator,
    control_tangent: np.ndarray,
    *,
    total_steps: int,
    release_previous_boundary: int,
    release_post_boundary: int,
    step_tangent: StepLinearMap,
    observation_tangent: ObservationTangent,
    initial_state_tangent: np.ndarray | None = None,
) -> TangentChainResult:
    """Apply the complete controlled-window tangent recurrence.

    Controls are inserted immediately before steps `0, ..., control_steps-1`.
    The observed release pair must occur after those insertions.  In the fresh
    three-month/ten-month experiment the values are respectively 9, 10, and
    40 for `control_steps`, the two release boundaries, and `total_steps`.
    """

    _validate_window(
        operator,
        total_steps=total_steps,
        release_previous_boundary=release_previous_boundary,
        release_post_boundary=release_post_boundary,
    )
    control = np.asarray(control_tangent, dtype=np.float64)
    operator.coefficient_matrix(control)
    if initial_state_tangent is None:
        tangent = np.zeros(operator.state_size, dtype=np.float64)
    else:
        tangent = _state_vector(
            initial_state_tangent, operator.state_size, "initial_state_tangent"
        ).copy()

    previous_release: np.ndarray | None = None
    post_release: np.ndarray | None = None
    if release_previous_boundary == 0:
        previous_release = tangent.copy()
    for step in range(total_steps):
        pre_step = tangent
        if step < operator.control_steps:
            pre_step = tangent + operator.apply_at_step(control, step)
        tangent = _state_vector(
            step_tangent(step, pre_step), operator.state_size, "step tangent output"
        ).copy()
        boundary = step + 1
        if boundary == release_previous_boundary:
            previous_release = tangent.copy()
        if boundary == release_post_boundary:
            post_release = tangent.copy()

    if previous_release is None or post_release is None:
        raise RuntimeError("release boundary tangents were not captured")
    release = np.asarray(
        observation_tangent(previous_release, post_release), dtype=np.float64
    )
    if release.ndim != 1 or not np.isfinite(release).all():
        raise ValueError("observation_tangent must return one finite vector")
    return TangentChainResult(
        release_observation=release,
        terminal_state=tangent,
    )


def propagate_distributed_control_adjoint(
    operator: DistributedControlOperator,
    *,
    total_steps: int,
    release_previous_boundary: int,
    release_post_boundary: int,
    step_transpose: StepLinearMap,
    observation_transpose: ObservationTranspose,
    release_observation_covector: np.ndarray,
    terminal_state_covector: np.ndarray,
) -> AdjointChainResult:
    """Apply the exact reverse recurrence for the controlled-window chain."""

    _validate_window(
        operator,
        total_steps=total_steps,
        release_previous_boundary=release_previous_boundary,
        release_post_boundary=release_post_boundary,
    )
    release_seed = np.asarray(release_observation_covector, dtype=np.float64)
    if release_seed.ndim != 1 or not np.isfinite(release_seed).all():
        raise ValueError("release_observation_covector must be one finite vector")
    previous_seed, post_seed = observation_transpose(release_seed)
    previous_seed = _state_vector(
        previous_seed, operator.state_size, "previous observation seed"
    )
    post_seed = _state_vector(post_seed, operator.state_size, "post observation seed")
    direct_seeds = {
        release_previous_boundary: previous_seed,
        release_post_boundary: post_seed,
    }

    state_covector = _state_vector(
        terminal_state_covector, operator.state_size, "terminal_state_covector"
    ).copy()
    if total_steps in direct_seeds:
        state_covector += direct_seeds[total_steps]
    control_gradient = np.zeros(operator.control_size, dtype=np.float64)
    for step in range(total_steps - 1, -1, -1):
        pre_step_covector = _state_vector(
            step_transpose(step, state_covector),
            operator.state_size,
            "step transpose output",
        )
        if step < operator.control_steps:
            control_gradient += operator.transpose_at_step(pre_step_covector, step)
        state_covector = pre_step_covector.copy()
        if step in direct_seeds:
            state_covector += direct_seeds[step]

    return AdjointChainResult(
        control_gradient=control_gradient,
        initial_state_covector=state_covector,
    )


def form_release_response_matrix(
    operator: DistributedControlOperator,
    **tangent_arguments: object,
) -> np.ndarray:
    """Form the small release Jacobian by tangent propagation of each control."""

    columns: list[np.ndarray] = []
    for index in range(operator.control_size):
        direction = np.zeros(operator.control_size, dtype=np.float64)
        direction[index] = 1.0
        result = propagate_distributed_control_tangent(
            operator,
            direction,
            **tangent_arguments,  # type: ignore[arg-type]
        )
        columns.append(result.release_observation)
    return np.column_stack(columns)


def _validate_window(
    operator: DistributedControlOperator,
    *,
    total_steps: int,
    release_previous_boundary: int,
    release_post_boundary: int,
) -> None:
    integers = (total_steps, release_previous_boundary, release_post_boundary)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("window lengths and boundaries must be integers")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if release_post_boundary != release_previous_boundary + 1:
        raise ValueError("the fresh observation requires adjacent state boundaries")
    if not 0 <= release_previous_boundary < release_post_boundary <= total_steps:
        raise ValueError("release boundaries lie outside the model window")
    if operator.control_steps > release_previous_boundary:
        raise ValueError("all gradual controls must end before the release pair")


def _state_vector(values: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result
