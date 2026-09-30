"""Matrix-free SQP for a nonlinear minimum-covariance-action control.

The scientific problem is expressed in covariance-whitened coordinates.  If
the native intervention at step ``k`` is ``F_k u_k`` and
``C_k = F_k F_k.T``, then the minimum coefficient norm over representations
of that intervention equals its Moore--Penrose covariance action:

``min ||u_k||^2 = (F_k u_k).T C_k^+ (F_k u_k)``.

For the scalar release constraint ``g(u) >= 0`` this module solves

``min 0.5 ||u||^2 subject to g(u) >= 0``

with a globally safeguarded, matrix-free SQP iteration.  Only the nonlinear
constraint value and its exact gradient are required.  No dense Hessian in
the potentially thousands-dimensional coefficient space is formed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

ConstraintValue = Callable[[np.ndarray], float]
ConstraintValueGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]
NativeConstraintValue = Callable[[np.ndarray], float]
NativeConstraintValueGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]
ACTIVE_BOUNDARY_POLISH_STATIONARITY_FACTOR = 2.0


class NativeCovarianceOperator(Protocol):
    """Matrix-free covariance action needed by the native SQP."""

    @property
    def state_size(self) -> int: ...

    def covariance_apply(self, native_vector: np.ndarray) -> np.ndarray: ...

    def covariance_apply_matrix(
        self, native_vectors: np.ndarray, *, block_rows: int = 128
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ScalarSQPIteration:
    """One accepted major iterate, including first-order KKT diagnostics."""

    iteration: int
    objective_value: float
    constraint_value: float
    primal_violation: float
    constraint_gradient_norm: float
    lagrange_multiplier: float
    stationarity_norm: float
    complementarity_absolute: float
    control_norm: float
    proposed_step_norm: float
    accepted_step_norm: float
    accepted_step_scale: float
    trust_radius: float
    merit_penalty: float
    merit_value: float
    line_search_evaluations: int


@dataclass(frozen=True)
class ScalarSQPResult:
    """Final control and complete reproducible convergence record."""

    control: np.ndarray
    success: bool
    status: str
    objective_value: float
    constraint_value: float
    primal_violation: float
    constraint_gradient: np.ndarray
    lagrange_multiplier: float
    stationarity_norm: float
    complementarity_absolute: float
    iterations: tuple[ScalarSQPIteration, ...]
    value_evaluations: int
    gradient_evaluations: int


@dataclass(frozen=True)
class NativeScalarSQPIteration:
    """One native/dual SQP iterate and covariance-metric diagnostics."""

    iteration: int
    objective_value: float
    constraint_value: float
    primal_violation: float
    native_gradient_l2_norm: float
    lagrange_multiplier: float
    covariance_stationarity_norm: float
    covariance_stationarity_relative: float
    complementarity_absolute: float
    complementarity_relative: float
    intervention_action_norm: float
    proposed_step_action_norm: float
    accepted_step_action_norm: float
    accepted_step_scale: float
    trust_radius: float
    merit_penalty: float
    merit_value: float
    line_search_evaluations: int
    radial_restoration_reason: str | None = None
    radial_restoration_scale: float | None = None
    radial_restoration_value_evaluations: int = 0
    second_order_correction_applied: bool = False
    second_order_correction_action_norm: float = 0.0
    second_order_correction_value_evaluations: int = 0


@dataclass(frozen=True)
class NativeScalarSQPResult:
    """Final native interventions, duals, and full SQP convergence record."""

    native_interventions: np.ndarray
    dual_variables: np.ndarray
    success: bool
    status: str
    objective_value: float
    squared_action: float
    constraint_value: float
    primal_violation: float
    native_constraint_gradient: np.ndarray
    lagrange_multiplier: float
    covariance_stationarity_norm: float
    covariance_stationarity_relative: float
    complementarity_absolute: float
    complementarity_relative: float
    stationarity_reference_norm: float
    constraint_gradient_covariance_norm: float
    iterations: tuple[NativeScalarSQPIteration, ...]
    value_evaluations: int
    gradient_evaluations: int
    covariance_passes: int
    radial_restoration_applied: bool
    radial_restoration_reason: str | None
    radial_restoration_scale: float | None
    radial_restoration_value_evaluations: int
    radial_restoration_gradient_evaluations: int
    radial_restoration_covariance_passes: int


def covariance_action(native_increment: np.ndarray, factor: np.ndarray) -> float:
    """Return ``a.T (F F.T)^+ a`` for an increment in ``range(F)``.

    The least-squares solve chooses the minimum-norm coefficient vector.  This
    remains correct when a centered-sample factor has a redundant all-ones
    coefficient direction.
    """

    matrix = np.asarray(factor, dtype=np.float64)
    increment = np.asarray(native_increment, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("factor must be a nonempty matrix")
    if increment.shape != (matrix.shape[0],):
        raise ValueError("native_increment does not match factor rows")
    if not np.isfinite(matrix).all() or not np.isfinite(increment).all():
        raise ValueError("factor and native_increment must be finite")
    coefficients, residuals, _, _ = np.linalg.lstsq(matrix, increment, rcond=None)
    tolerance = (
        100.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(increment)))
    )
    explicit_residual = float(np.linalg.norm(matrix @ coefficients - increment))
    if residuals.size and float(np.sqrt(np.sum(residuals))) > tolerance:
        raise ValueError("native_increment lies outside the covariance range")
    if explicit_residual > tolerance:
        raise ValueError("native_increment lies outside the covariance range")
    return float(coefficients @ coefficients)


def pull_back_native_covectors(
    factors: np.ndarray, native_covectors: np.ndarray
) -> np.ndarray:
    """Return all ``F_k.T lambda_k`` blocks as one coefficient gradient."""

    maps = np.asarray(factors, dtype=np.float64)
    covectors = np.asarray(native_covectors, dtype=np.float64)
    if maps.ndim != 3 or any(size <= 0 for size in maps.shape):
        raise ValueError("factors must have shape (steps, native, coefficients)")
    if covectors.shape != maps.shape[:2]:
        raise ValueError("native_covectors must have shape (steps, native)")
    if not np.isfinite(maps).all() or not np.isfinite(covectors).all():
        raise ValueError("factors and native_covectors must be finite")
    return np.einsum("snc,sn->sc", maps, covectors).reshape(-1)


def _validate_oracle_result(
    raw_value: float,
    raw_gradient: np.ndarray,
    dimension: int,
) -> tuple[float, np.ndarray]:
    value = float(raw_value)
    gradient = np.asarray(raw_gradient, dtype=np.float64)
    if not math.isfinite(value):
        raise ValueError("constraint oracle returned a nonfinite value")
    if gradient.shape != (dimension,):
        raise ValueError(f"constraint gradient must have shape ({dimension},)")
    if not np.isfinite(gradient).all():
        raise ValueError("constraint oracle returned a nonfinite gradient")
    return value, gradient


def _kkt_diagnostics(
    control: np.ndarray, constraint: float, gradient: np.ndarray
) -> tuple[float, float, float, float]:
    gradient_squared = float(gradient @ gradient)
    if gradient_squared <= np.finfo(np.float64).tiny:
        multiplier = 0.0
    else:
        multiplier = max(0.0, float(control @ gradient) / gradient_squared)
    stationarity = float(np.linalg.norm(control - multiplier * gradient))
    violation = max(0.0, -constraint)
    complementarity = abs(multiplier * constraint)
    return multiplier, stationarity, violation, complementarity


def solve_minimum_covariance_action_sqp(
    constraint_value: ConstraintValue,
    constraint_value_gradient: ConstraintValueGradient,
    *,
    dimension: int,
    initial_control: np.ndarray | None = None,
    maximum_iterations: int = 30,
    constraint_tolerance: float = 1.0e-5,
    stationarity_tolerance: float = 1.0e-5,
    complementarity_tolerance: float = 1.0e-5,
    initial_trust_radius: float = 1.0,
    maximum_trust_radius: float = 10.0,
    initial_merit_penalty: float = 1.0,
    armijo_fraction: float = 1.0e-4,
    backtrack_factor: float = 0.5,
    maximum_line_search_steps: int = 16,
) -> ScalarSQPResult:
    r"""Solve a scalar nonlinear inequality with matrix-free safeguarded SQP.

    At each major iterate, the untruncated SQP trial is the exact solution of

    ``min_y 0.5 ||y||^2`` subject to
    ``g(u) + grad(g(u)).T (y-u) >= 0``.

    A trust radius and an exact ``l1`` violation merit function globalize this
    local model.  Trial line-search evaluations require only a nonlinear
    forward value; one exact gradient (normally one reverse sweep) is used per
    major iteration.  The returned history is sufficient to report every
    accepted update and the final first-order KKT residuals.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations <= 0
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    if (
        isinstance(maximum_line_search_steps, bool)
        or not isinstance(maximum_line_search_steps, int)
        or maximum_line_search_steps <= 0
    ):
        raise ValueError("maximum_line_search_steps must be a positive integer")
    positive_parameters = {
        "constraint_tolerance": constraint_tolerance,
        "stationarity_tolerance": stationarity_tolerance,
        "complementarity_tolerance": complementarity_tolerance,
        "initial_trust_radius": initial_trust_radius,
        "maximum_trust_radius": maximum_trust_radius,
        "initial_merit_penalty": initial_merit_penalty,
        "armijo_fraction": armijo_fraction,
    }
    for name, raw in positive_parameters.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_trust_radius < initial_trust_radius:
        raise ValueError("maximum_trust_radius cannot be smaller than initial")
    if not math.isfinite(backtrack_factor) or not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must lie in (0, 1)")

    if initial_control is None:
        control = np.zeros(dimension, dtype=np.float64)
    else:
        control = np.asarray(initial_control, dtype=np.float64).copy()
        if control.shape != (dimension,) or not np.isfinite(control).all():
            raise ValueError("initial_control has the wrong shape or is nonfinite")

    trust_radius = float(initial_trust_radius)
    merit_penalty = float(initial_merit_penalty)
    history: list[ScalarSQPIteration] = []
    value_evaluations = 0
    gradient_evaluations = 0
    status = "maximum_iterations_reached"
    success = False
    final_value = math.nan
    final_gradient = np.empty(dimension, dtype=np.float64)

    for iteration in range(maximum_iterations + 1):
        raw_value, raw_gradient = constraint_value_gradient(control.copy())
        gradient_evaluations += 1
        value_evaluations += 1
        value, gradient = _validate_oracle_result(raw_value, raw_gradient, dimension)
        multiplier, stationarity, violation, complementarity = _kkt_diagnostics(
            control, value, gradient
        )
        final_value = value
        final_gradient = gradient
        objective = 0.5 * float(control @ control)
        merit = objective + merit_penalty * violation

        converged = (
            violation <= constraint_tolerance
            and stationarity <= stationarity_tolerance
            and complementarity <= complementarity_tolerance
        )
        if converged:
            history.append(
                ScalarSQPIteration(
                    iteration=iteration,
                    objective_value=objective,
                    constraint_value=value,
                    primal_violation=violation,
                    constraint_gradient_norm=float(np.linalg.norm(gradient)),
                    lagrange_multiplier=multiplier,
                    stationarity_norm=stationarity,
                    complementarity_absolute=complementarity,
                    control_norm=float(np.linalg.norm(control)),
                    proposed_step_norm=0.0,
                    accepted_step_norm=0.0,
                    accepted_step_scale=0.0,
                    trust_radius=trust_radius,
                    merit_penalty=merit_penalty,
                    merit_value=merit,
                    line_search_evaluations=0,
                )
            )
            status = "first_order_kkt_satisfied"
            success = True
            break
        if iteration == maximum_iterations:
            break

        gradient_squared = float(gradient @ gradient)
        if gradient_squared <= np.finfo(np.float64).tiny:
            status = "constraint_gradient_vanished_before_feasibility"
            break
        linear_boundary = float(gradient @ control) - value
        if linear_boundary <= 0.0:
            linear_solution = np.zeros(dimension, dtype=np.float64)
            linear_multiplier = 0.0
        else:
            linear_multiplier = linear_boundary / gradient_squared
            linear_solution = linear_multiplier * gradient
        step = linear_solution - control
        proposed_step_norm = float(np.linalg.norm(step))
        if proposed_step_norm > trust_radius:
            step *= trust_radius / proposed_step_norm

        # An exact penalty larger than the local multiplier makes the SQP
        # direction a descent direction for the first-order merit model.
        merit_penalty = max(
            merit_penalty,
            1.5 * linear_multiplier + 1.0e-12,
        )
        if value < 0.0:
            directional_merit = float(control @ step) - merit_penalty * float(
                gradient @ step
            )
        else:
            # At a strictly feasible point the local directional derivative
            # of max(0, -g) is zero.  A trial that actually crosses the
            # boundary is still charged its full violation below.
            directional_merit = float(control @ step)
        if directional_merit >= 0.0 and value < 0.0 and gradient @ step > 0.0:
            needed = float(control @ step) / float(gradient @ step)
            merit_penalty = max(merit_penalty, 1.5 * needed + 1.0e-12)
            directional_merit = float(control @ step) - merit_penalty * float(
                gradient @ step
            )
        if directional_merit >= 0.0:
            status = "no_exact_merit_descent_direction"
            break

        scale = 1.0
        accepted_control: np.ndarray | None = None
        accepted_value = math.nan
        line_evaluations = 0
        for _ in range(maximum_line_search_steps):
            trial = control + scale * step
            trial_value = float(constraint_value(trial.copy()))
            value_evaluations += 1
            line_evaluations += 1
            if not math.isfinite(trial_value):
                scale *= backtrack_factor
                continue
            trial_objective = 0.5 * float(trial @ trial)
            trial_merit = trial_objective + merit_penalty * max(0.0, -trial_value)
            armijo_bound = merit + armijo_fraction * scale * directional_merit
            if trial_merit <= armijo_bound:
                accepted_control = trial
                accepted_value = trial_value
                break
            scale *= backtrack_factor
        if accepted_control is None:
            trust_radius *= 0.25
            if trust_radius <= 10.0 * np.finfo(np.float64).eps:
                status = "line_search_failed_at_minimum_trust_radius"
                break
            # Record a rejected major trial, then retry from the same point.
            history.append(
                ScalarSQPIteration(
                    iteration=iteration,
                    objective_value=objective,
                    constraint_value=value,
                    primal_violation=violation,
                    constraint_gradient_norm=float(np.linalg.norm(gradient)),
                    lagrange_multiplier=multiplier,
                    stationarity_norm=stationarity,
                    complementarity_absolute=complementarity,
                    control_norm=float(np.linalg.norm(control)),
                    proposed_step_norm=proposed_step_norm,
                    accepted_step_norm=0.0,
                    accepted_step_scale=0.0,
                    trust_radius=trust_radius,
                    merit_penalty=merit_penalty,
                    merit_value=merit,
                    line_search_evaluations=line_evaluations,
                )
            )
            continue

        accepted_step = accepted_control - control
        history.append(
            ScalarSQPIteration(
                iteration=iteration,
                objective_value=objective,
                constraint_value=value,
                primal_violation=violation,
                constraint_gradient_norm=float(np.linalg.norm(gradient)),
                lagrange_multiplier=multiplier,
                stationarity_norm=stationarity,
                complementarity_absolute=complementarity,
                control_norm=float(np.linalg.norm(control)),
                proposed_step_norm=proposed_step_norm,
                accepted_step_norm=float(np.linalg.norm(accepted_step)),
                accepted_step_scale=scale,
                trust_radius=trust_radius,
                merit_penalty=merit_penalty,
                merit_value=merit,
                line_search_evaluations=line_evaluations,
            )
        )
        control = accepted_control
        final_value = accepted_value
        if scale == 1.0 and np.linalg.norm(accepted_step) >= 0.8 * trust_radius:
            trust_radius = min(maximum_trust_radius, 2.0 * trust_radius)
        elif scale < 0.5:
            trust_radius = max(10.0 * np.finfo(np.float64).eps, 0.5 * trust_radius)

    multiplier, stationarity, violation, complementarity = _kkt_diagnostics(
        control, final_value, final_gradient
    )
    return ScalarSQPResult(
        control=control,
        success=success,
        status=status,
        objective_value=0.5 * float(control @ control),
        constraint_value=final_value,
        primal_violation=violation,
        constraint_gradient=final_gradient.copy(),
        lagrange_multiplier=multiplier,
        stationarity_norm=stationarity,
        complementarity_absolute=complementarity,
        iterations=tuple(history),
        value_evaluations=value_evaluations,
        gradient_evaluations=gradient_evaluations,
    )


def _native_covariance_specification(
    covariance: NativeCovarianceOperator | Sequence[NativeCovarianceOperator],
    control_steps: int,
) -> tuple[NativeCovarianceOperator | None, tuple[NativeCovarianceOperator, ...], int]:
    if (
        isinstance(control_steps, bool)
        or not isinstance(control_steps, int)
        or control_steps <= 0
    ):
        raise ValueError("control_steps must be a positive integer")
    if hasattr(covariance, "covariance_apply") and hasattr(
        covariance, "covariance_apply_matrix"
    ):
        common = covariance  # type: ignore[assignment]
        state_size = int(common.state_size)
        if state_size <= 0:
            raise ValueError("covariance state_size must be positive")
        return common, (), state_size
    operators = tuple(covariance)  # type: ignore[arg-type]
    if len(operators) != control_steps:
        raise ValueError("phase-specific covariance count must equal control_steps")
    if not operators:
        raise ValueError("at least one covariance operator is required")
    state_sizes = {int(operator.state_size) for operator in operators}
    if len(state_sizes) != 1 or min(state_sizes) <= 0:
        raise ValueError("phase-specific covariance state sizes must agree")
    return None, operators, state_sizes.pop()


def _apply_covariance_blocks(
    common: NativeCovarianceOperator | None,
    phase_specific: tuple[NativeCovarianceOperator, ...],
    vectors: np.ndarray,
    *,
    block_rows: int,
) -> tuple[np.ndarray, int]:
    if common is not None:
        raw = common.covariance_apply_matrix(vectors.T, block_rows=block_rows)
        result = np.asarray(raw, dtype=np.float64).T
        passes = 1
    else:
        result = np.vstack(
            [
                np.asarray(operator.covariance_apply(vector), dtype=np.float64)
                for operator, vector in zip(phase_specific, vectors, strict=True)
            ]
        )
        passes = len(phase_specific)
    if result.shape != vectors.shape or not np.isfinite(result).all():
        raise ValueError("covariance operator returned an invalid result")
    return result, passes


def _validate_native_oracle_result(
    raw_value: float,
    raw_gradient: np.ndarray,
    shape: tuple[int, int],
) -> tuple[float, np.ndarray]:
    value = float(raw_value)
    gradient = np.asarray(raw_gradient, dtype=np.float64)
    if not math.isfinite(value):
        raise ValueError("constraint oracle returned a nonfinite value")
    if gradient.shape != shape:
        raise ValueError(f"native constraint gradient must have shape {shape}")
    if not np.isfinite(gradient).all():
        raise ValueError("constraint oracle returned a nonfinite gradient")
    return value, gradient


def _nonnegative_quadratic(value: float, label: str) -> float:
    tolerance = 1.0e-10 * max(1.0, abs(value))
    if value < -tolerance:
        raise ValueError(f"{label} is negative; covariance is not PSD")
    return max(0.0, value)


def _native_kkt_diagnostics(
    dual: np.ndarray,
    native: np.ndarray,
    constraint: float,
    gradient: np.ndarray,
    covariance_gradient: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    squared_action = _nonnegative_quadratic(
        float(np.sum(dual * native)), "squared covariance action"
    )
    gradient_metric_squared = _nonnegative_quadratic(
        float(np.sum(gradient * covariance_gradient)),
        "covariance gradient norm",
    )
    gradient_native_product = float(np.sum(gradient * native))
    if gradient_metric_squared <= np.finfo(np.float64).tiny:
        multiplier = 0.0
    else:
        multiplier = max(0.0, gradient_native_product / gradient_metric_squared)
    residual_dual = dual - multiplier * gradient
    residual_native = native - multiplier * covariance_gradient
    stationarity_squared = float(np.sum(residual_dual * residual_native))
    stationarity = math.sqrt(
        _nonnegative_quadratic(stationarity_squared, "covariance-metric stationarity")
    )
    violation = max(0.0, -constraint)
    complementarity = abs(multiplier * constraint)
    action_norm = math.sqrt(squared_action)
    gradient_covariance_norm = math.sqrt(gradient_metric_squared)
    stationarity_reference = max(
        action_norm,
        multiplier * gradient_covariance_norm,
        np.finfo(np.float64).tiny,
    )
    stationarity_relative = stationarity / stationarity_reference
    complementarity_relative = complementarity / max(
        squared_action, np.finfo(np.float64).tiny
    )
    return (
        multiplier,
        stationarity,
        stationarity_relative,
        violation,
        complementarity,
        complementarity_relative,
        squared_action,
        stationarity_reference,
        gradient_covariance_norm,
    )


@dataclass(frozen=True)
class _NativeRadialRestoration:
    """A covariance-range-preserving radial feasibility correction."""

    dual: np.ndarray
    native: np.ndarray
    value: float
    gradient: np.ndarray
    covariance_gradient: np.ndarray
    scale: float
    value_evaluations: int
    gradient_evaluations: int
    covariance_passes: int


@dataclass(frozen=True)
class _NativeRadialRestorationAttempt:
    """A radial candidate or a fail-closed account of the attempted work."""

    candidate: _NativeRadialRestoration | None
    reason: str
    value_evaluations: int
    gradient_evaluations: int
    covariance_passes: int


def _polish_positive_constraint_along_native_ray(
    constraint_value: NativeConstraintValue,
    constraint_value_gradient: NativeConstraintValueGradient,
    common_covariance: NativeCovarianceOperator | None,
    phase_covariances: tuple[NativeCovarianceOperator, ...],
    *,
    dual: np.ndarray,
    native: np.ndarray,
    value: float,
    gradient: np.ndarray,
    covariance_block_rows: int,
    target_positive_value: float,
    maximum_relative_contraction: float = 5.0e-2,
    maximum_bracket_steps: int = 16,
    maximum_bisection_steps: int = 40,
) -> _NativeRadialRestorationAttempt:
    """Contract to the nearest bracketed active boundary along a common ray.

    The current point must be strictly feasible. Finite positive samples are
    followed inward until the first sampled nonpositive value brackets a sign
    change. Bisection retains the positive side of that bracket. A nonfinite
    value aborts rather than stepping across an unaudited interval, which keeps
    this polishing operation safe even when the nonlinear replay is not
    monotone along the full ray.
    """

    if value <= 0.0:
        return _NativeRadialRestorationAttempt(
            None, "initial_constraint_not_positive", 0, 0, 0
        )
    radial_derivative = float(np.sum(gradient * native))
    if not math.isfinite(radial_derivative) or radial_derivative <= 0.0:
        return _NativeRadialRestorationAttempt(
            None, "nonpositive_radial_derivative", 0, 0, 0
        )
    predicted_contraction = value / radial_derivative
    if (
        not math.isfinite(predicted_contraction)
        or predicted_contraction <= 0.0
        or predicted_contraction > maximum_relative_contraction
    ):
        return _NativeRadialRestorationAttempt(
            None, "local_contraction_outside_limit", 0, 0, 0
        )
    if not math.isfinite(target_positive_value) or target_positive_value <= 0.0:
        return _NativeRadialRestorationAttempt(None, "invalid_positive_target", 0, 0, 0)

    minimum_contraction = 32.0 * np.finfo(np.float64).eps
    contraction = max(predicted_contraction, minimum_contraction)
    positive_scale = 1.0
    positive_value = value
    nonpositive_scale: float | None = None
    nonpositive_value = math.nan
    value_evaluations = 0
    for _ in range(maximum_bracket_steps):
        trial_scale = 1.0 - min(contraction, maximum_relative_contraction)
        raw_trial_value = float(constraint_value(trial_scale * native))
        value_evaluations += 1
        if not math.isfinite(raw_trial_value):
            return _NativeRadialRestorationAttempt(
                None,
                "nonfinite_value_before_sign_bracket",
                value_evaluations,
                0,
                0,
            )
        if raw_trial_value <= 0.0:
            nonpositive_scale = trial_scale
            nonpositive_value = raw_trial_value
            break
        positive_scale = trial_scale
        positive_value = raw_trial_value
        if contraction >= maximum_relative_contraction:
            break
        contraction = min(2.0 * contraction, maximum_relative_contraction)
    if nonpositive_scale is None:
        return _NativeRadialRestorationAttempt(
            None, "no_inward_sign_bracket", value_evaluations, 0, 0
        )

    if nonpositive_value == 0.0:
        candidate_scale = nonpositive_scale
        candidate_value = 0.0
    else:
        for _ in range(maximum_bisection_steps):
            if positive_value <= target_positive_value:
                break
            midpoint = 0.5 * (nonpositive_scale + positive_scale)
            if midpoint in (nonpositive_scale, positive_scale):
                break
            raw_midpoint_value = float(constraint_value(midpoint * native))
            value_evaluations += 1
            if not math.isfinite(raw_midpoint_value):
                return _NativeRadialRestorationAttempt(
                    None,
                    "nonfinite_value_inside_sign_bracket",
                    value_evaluations,
                    0,
                    0,
                )
            if raw_midpoint_value >= 0.0:
                positive_scale = midpoint
                positive_value = raw_midpoint_value
            else:
                nonpositive_scale = midpoint
                nonpositive_value = raw_midpoint_value
        del nonpositive_value
        if positive_value > target_positive_value:
            return _NativeRadialRestorationAttempt(
                None,
                "positive_boundary_not_resolved",
                value_evaluations,
                0,
                0,
            )
        candidate_scale = positive_scale
        candidate_value = positive_value

    restored_dual = candidate_scale * dual
    restored_native = candidate_scale * native
    raw_value, raw_gradient = constraint_value_gradient(restored_native.copy())
    value_evaluations += 1
    restored_value, restored_gradient = _validate_native_oracle_result(
        raw_value, raw_gradient, native.shape
    )
    if restored_value < 0.0 or restored_value > target_positive_value:
        return _NativeRadialRestorationAttempt(
            None,
            "exact_gradient_replay_left_positive_boundary_target",
            value_evaluations,
            1,
            0,
        )
    agreement_tolerance = (
        10.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(candidate_value), abs(restored_value))
    )
    if abs(restored_value - candidate_value) > agreement_tolerance:
        return _NativeRadialRestorationAttempt(
            None,
            "value_and_gradient_oracles_disagree",
            value_evaluations,
            1,
            0,
        )
    restored_covariance_gradient, covariance_passes = _apply_covariance_blocks(
        common_covariance,
        phase_covariances,
        restored_gradient,
        block_rows=covariance_block_rows,
    )
    candidate = _NativeRadialRestoration(
        dual=restored_dual,
        native=restored_native,
        value=restored_value,
        gradient=restored_gradient,
        covariance_gradient=restored_covariance_gradient,
        scale=candidate_scale,
        value_evaluations=value_evaluations,
        gradient_evaluations=1,
        covariance_passes=covariance_passes,
    )
    return _NativeRadialRestorationAttempt(
        candidate,
        "candidate_resolved",
        value_evaluations,
        1,
        covariance_passes,
    )


def _restore_nonnegative_constraint_along_native_ray(
    constraint_value: NativeConstraintValue,
    constraint_value_gradient: NativeConstraintValueGradient,
    common_covariance: NativeCovarianceOperator | None,
    phase_covariances: tuple[NativeCovarianceOperator, ...],
    *,
    dual: np.ndarray,
    native: np.ndarray,
    value: float,
    gradient: np.ndarray,
    covariance_block_rows: int,
    target_positive_value: float,
    maximum_relative_expansion: float = 5.0e-2,
    maximum_bracket_steps: int = 16,
    maximum_bisection_steps: int = 40,
) -> _NativeRadialRestoration | None:
    """Find the first resolved nonnegative point on ``alpha * native``.

    Scaling ``dual`` and ``native`` by the same positive scalar preserves the
    exact covariance relation ``native = C dual`` and increases the quadratic
    action monotonically.  This narrowly repairs a small negative feasibility
    residual left by finite-precision nonlinear replays; it is not a second
    optimization algorithm.  A safeguarded expanding bracket accommodates a
    locally nonmonotone replay before bisection finds the first sign change
    resolved by the oracle.
    """

    if value >= 0.0:
        return None
    radial_derivative = float(np.sum(gradient * native))
    if not math.isfinite(radial_derivative) or radial_derivative <= 0.0:
        return None
    predicted_increment = -value / radial_derivative
    if (
        not math.isfinite(predicted_increment)
        or predicted_increment <= 0.0
        or predicted_increment > maximum_relative_expansion
    ):
        return None

    minimum_increment = 32.0 * np.finfo(np.float64).eps
    increment = max(predicted_increment, minimum_increment)
    lower_scale = 1.0
    lower_value = value
    upper_scale: float | None = None
    upper_value = math.nan
    value_evaluations = 0
    for _ in range(maximum_bracket_steps):
        trial_scale = 1.0 + min(increment, maximum_relative_expansion)
        raw_trial_value = float(constraint_value(trial_scale * native))
        value_evaluations += 1
        if math.isfinite(raw_trial_value):
            if raw_trial_value >= 0.0:
                upper_scale = trial_scale
                upper_value = raw_trial_value
                break
            lower_scale = trial_scale
            lower_value = raw_trial_value
        if increment >= maximum_relative_expansion:
            break
        increment = min(2.0 * increment, maximum_relative_expansion)
    if upper_scale is None:
        return None

    # A positive value no larger than the original absolute feasibility
    # tolerance is already more accurate than the SQP gate.  Otherwise refine
    # the sign bracket.  The lower endpoint need only remain negative; local
    # monotonicity is not assumed outside the current bracket.
    for _ in range(maximum_bisection_steps):
        if upper_value <= target_positive_value:
            break
        midpoint = 0.5 * (lower_scale + upper_scale)
        if midpoint in (lower_scale, upper_scale):
            break
        raw_midpoint_value = float(constraint_value(midpoint * native))
        value_evaluations += 1
        if not math.isfinite(raw_midpoint_value):
            lower_scale = midpoint
            continue
        if raw_midpoint_value >= 0.0:
            upper_scale = midpoint
            upper_value = raw_midpoint_value
        else:
            lower_scale = midpoint
            lower_value = raw_midpoint_value
    del lower_value

    restored_dual = upper_scale * dual
    restored_native = upper_scale * native
    raw_value, raw_gradient = constraint_value_gradient(restored_native.copy())
    value_evaluations += 1
    restored_value, restored_gradient = _validate_native_oracle_result(
        raw_value, raw_gradient, native.shape
    )
    if restored_value < 0.0:
        return None
    restored_covariance_gradient, covariance_passes = _apply_covariance_blocks(
        common_covariance,
        phase_covariances,
        restored_gradient,
        block_rows=covariance_block_rows,
    )
    return _NativeRadialRestoration(
        dual=restored_dual,
        native=restored_native,
        value=restored_value,
        gradient=restored_gradient,
        covariance_gradient=restored_covariance_gradient,
        scale=upper_scale,
        value_evaluations=value_evaluations,
        gradient_evaluations=1,
        covariance_passes=covariance_passes,
    )


def solve_minimum_native_covariance_action_sqp(
    constraint_value: NativeConstraintValue,
    constraint_value_gradient: NativeConstraintValueGradient,
    covariance: NativeCovarianceOperator | Sequence[NativeCovarianceOperator],
    *,
    control_steps: int,
    initial_dual_variables: np.ndarray | None = None,
    known_active_boundary: bool = False,
    maximum_iterations: int = 30,
    constraint_tolerance: float = 1.0e-5,
    stationarity_tolerance: float = 1.0e-5,
    complementarity_tolerance: float = 1.0e-5,
    relative_stationarity_tolerance: float | None = None,
    relative_complementarity_tolerance: float | None = None,
    initial_trust_radius: float = 1.0,
    maximum_trust_radius: float = 10.0,
    initial_merit_penalty: float = 1.0,
    armijo_fraction: float = 1.0e-4,
    backtrack_factor: float = 0.5,
    maximum_line_search_steps: int = 16,
    covariance_block_rows: int = 128,
) -> NativeScalarSQPResult:
    r"""Solve the nonlinear action problem without sample coefficients.

    Each intervention is maintained as ``a_k = C_k v_k``.  At a major
    iterate, with nonlinear constraint value ``h`` and native gradients
    ``g_k``, the exact linearized minimum-action target is

    ``y_k = lambda * C_k g_k``, where
    ``lambda = (sum(g_k.T a_k) - h) / sum(g_k.T C_k g_k)``

    when that numerator is positive (otherwise ``y=0``).  The safeguarded
    direction is ``y-a``.  One covariance application per phase is needed at
    each major iteration; a shared covariance uses one batched application
    for all phases.  Nonlinear line-search trials reuse the native direction
    and therefore require no additional covariance pass.

    The objective is ``0.5 * sum(v_k.T a_k)``.  The reported stationarity is
    the covariance-metric KKT residual
    ``sqrt(sum((v_k-lambda*g_k).T C_k (v_k-lambda*g_k)))``.  Optional
    relative KKT tolerances scale stationarity by
    ``max(||a||_{C+}, lambda ||g||_C)`` and complementarity by the squared
    action.  These dimensionless tests are preferable when the physical
    covariance and constraint scales vary between experiments.

    With the default zero intervention, an initially violated constraint
    proves that the minimum-action solution must lie on ``h = 0``: the only
    unconstrained stationary point is the infeasible origin.  The caller may
    provide the same proof explicitly through ``known_active_boundary=True``
    when supplying a nonzero warm start.  In either case the line search uses
    the active-boundary exact penalty
    ``0.5 * ||a||^2_{C+} + rho * |h|``.  This is the standard equality-SQP
    globalization of the necessarily active inequality.  In particular, it
    prevents an SQP step that overshoots into ``h > 0`` from discarding all
    information about its distance from the boundary.  Otherwise, the general
    inequality merit ``0.5 * ||a||^2_{C+} + rho * max(0, -h)`` is retained.
    Near active-boundary stationarity, a safeguarded contraction from a small
    positive overshoot or expansion from a small negative residual may return
    ``h`` to its nearest bracketed boundary.  Both ``v`` and ``a=Cv`` are
    scaled along their common ray.  A point satisfying the original primal,
    stationarity, and complementarity tests terminates successfully.
    Otherwise, the verified boundary point becomes an intermediate SQP
    iterate, and the original tests must still pass at a later iterate before
    success is reported.  Near the active boundary, a rejected tangential SQP
    trial may also receive the standard second-order correction
    ``beta Cg``, with ``beta = -h(trial)/(g.T Cg)``.  This corrects the
    second-order constraint drift responsible for the Maratos effect while
    preserving ``a=Cv``.  The corrected trial must pass the original Armijo
    merit test; it cannot itself declare convergence.
    """

    common, phase_specific, state_size = _native_covariance_specification(
        covariance, control_steps
    )
    shape = (control_steps, state_size)
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations <= 0
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    if (
        isinstance(maximum_line_search_steps, bool)
        or not isinstance(maximum_line_search_steps, int)
        or maximum_line_search_steps <= 0
    ):
        raise ValueError("maximum_line_search_steps must be a positive integer")
    if (
        isinstance(covariance_block_rows, bool)
        or not isinstance(covariance_block_rows, int)
        or covariance_block_rows <= 0
    ):
        raise ValueError("covariance_block_rows must be a positive integer")
    if not isinstance(known_active_boundary, bool):
        raise ValueError("known_active_boundary must be a bool")
    positive_parameters = {
        "constraint_tolerance": constraint_tolerance,
        "stationarity_tolerance": stationarity_tolerance,
        "complementarity_tolerance": complementarity_tolerance,
        "initial_trust_radius": initial_trust_radius,
        "maximum_trust_radius": maximum_trust_radius,
        "initial_merit_penalty": initial_merit_penalty,
        "armijo_fraction": armijo_fraction,
    }
    for name, raw in positive_parameters.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    optional_positive_parameters = {
        "relative_stationarity_tolerance": relative_stationarity_tolerance,
        "relative_complementarity_tolerance": (relative_complementarity_tolerance),
    }
    for name, raw in optional_positive_parameters.items():
        if raw is not None and (not math.isfinite(raw) or raw <= 0.0):
            raise ValueError(f"{name} must be finite and positive when provided")
    if maximum_trust_radius < initial_trust_radius:
        raise ValueError("maximum_trust_radius cannot be smaller than initial")
    if not math.isfinite(backtrack_factor) or not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must lie in (0, 1)")

    covariance_passes = 0
    if initial_dual_variables is None:
        dual = np.zeros(shape, dtype=np.float64)
        native = np.zeros(shape, dtype=np.float64)
    else:
        dual = np.asarray(initial_dual_variables, dtype=np.float64).copy()
        if dual.shape != shape or not np.isfinite(dual).all():
            raise ValueError(
                "initial_dual_variables has the wrong shape or is nonfinite"
            )
        native, passes = _apply_covariance_blocks(
            common,
            phase_specific,
            dual,
            block_rows=covariance_block_rows,
        )
        covariance_passes += passes

    trust_radius = float(initial_trust_radius)
    merit_penalty = float(initial_merit_penalty)
    history: list[NativeScalarSQPIteration] = []
    value_evaluations = 0
    gradient_evaluations = 0
    status = "maximum_iterations_reached"
    success = False
    final_value = math.nan
    final_gradient = np.empty(shape, dtype=np.float64)
    final_covariance_gradient = np.empty(shape, dtype=np.float64)
    active_boundary_merit = known_active_boundary
    radial_restoration_applied = False
    radial_restoration_reason: str | None = None
    radial_restoration_scale: float | None = None
    radial_restoration_value_evaluations = 0
    radial_restoration_gradient_evaluations = 0
    radial_restoration_covariance_passes = 0
    positive_polish_attempted_native: np.ndarray | None = None
    negative_restoration_attempted_native: np.ndarray | None = None

    for iteration in range(maximum_iterations + 1):
        raw_value, raw_gradient = constraint_value_gradient(native.copy())
        gradient_evaluations += 1
        value_evaluations += 1
        value, gradient = _validate_native_oracle_result(raw_value, raw_gradient, shape)
        covariance_gradient, passes = _apply_covariance_blocks(
            common,
            phase_specific,
            gradient,
            block_rows=covariance_block_rows,
        )
        covariance_passes += passes
        (
            multiplier,
            stationarity,
            stationarity_relative,
            violation,
            complementarity,
            complementarity_relative,
            squared_action,
            stationarity_reference,
            gradient_covariance_norm,
        ) = _native_kkt_diagnostics(dual, native, value, gradient, covariance_gradient)
        final_value = value
        final_gradient = gradient
        final_covariance_gradient = covariance_gradient
        objective = 0.5 * squared_action
        if iteration == 0 and initial_dual_variables is None and value < 0.0:
            active_boundary_merit = True
        merit_residual = abs(value) if active_boundary_merit else violation
        merit = objective + merit_penalty * merit_residual

        stationarity_converged = (
            stationarity <= stationarity_tolerance
            if relative_stationarity_tolerance is None
            else stationarity_relative <= relative_stationarity_tolerance
        )
        stationarity_near_threshold = (
            stationarity
            <= ACTIVE_BOUNDARY_POLISH_STATIONARITY_FACTOR * stationarity_tolerance
            if relative_stationarity_tolerance is None
            else stationarity_relative
            <= ACTIVE_BOUNDARY_POLISH_STATIONARITY_FACTOR
            * relative_stationarity_tolerance
        )
        complementarity_converged = (
            complementarity <= complementarity_tolerance
            if relative_complementarity_tolerance is None
            else complementarity_relative <= relative_complementarity_tolerance
        )
        radial_derivative = float(np.sum(gradient * native))
        predicted_radial_contraction = (
            value / radial_derivative
            if radial_derivative > 0.0 and math.isfinite(radial_derivative)
            else math.inf
        )
        should_try_positive_polish = (
            known_active_boundary
            and value > 0.0
            and stationarity_near_threshold
            and not complementarity_converged
            and not radial_restoration_applied
            and squared_action > np.finfo(np.float64).tiny
            and math.isfinite(predicted_radial_contraction)
            and 0.0 < predicted_radial_contraction <= 5.0e-2
            and (
                positive_polish_attempted_native is None
                or not np.array_equal(native, positive_polish_attempted_native)
            )
        )
        if should_try_positive_polish:
            positive_polish_attempted_native = native.copy()
            radial_restoration_reason = (
                "positive_constraint_at_otherwise_stationary_active_boundary"
            )
            if multiplier > np.finfo(np.float64).tiny:
                if relative_complementarity_tolerance is None:
                    complementarity_value_limit = complementarity_tolerance / multiplier
                else:
                    complementarity_value_limit = (
                        relative_complementarity_tolerance * squared_action / multiplier
                    )
                target_positive_value = min(
                    constraint_tolerance,
                    0.5 * complementarity_value_limit,
                )
            else:
                target_positive_value = constraint_tolerance
            attempt = _polish_positive_constraint_along_native_ray(
                constraint_value,
                constraint_value_gradient,
                common,
                phase_specific,
                dual=dual,
                native=native,
                value=value,
                gradient=gradient,
                covariance_block_rows=covariance_block_rows,
                target_positive_value=max(
                    target_positive_value, np.finfo(np.float64).tiny
                ),
            )
            radial_restoration_value_evaluations += attempt.value_evaluations
            radial_restoration_gradient_evaluations += attempt.gradient_evaluations
            radial_restoration_covariance_passes += attempt.covariance_passes
            value_evaluations += attempt.value_evaluations
            gradient_evaluations += attempt.gradient_evaluations
            covariance_passes += attempt.covariance_passes
            restoration = attempt.candidate
            if restoration is None:
                radial_restoration_reason += f":{attempt.reason}"
            else:
                radial_restoration_scale = restoration.scale
                (
                    restored_multiplier,
                    restored_stationarity,
                    restored_stationarity_relative,
                    restored_violation,
                    restored_complementarity,
                    restored_complementarity_relative,
                    restored_squared_action,
                    restored_stationarity_reference,
                    restored_gradient_covariance_norm,
                ) = _native_kkt_diagnostics(
                    restoration.dual,
                    restoration.native,
                    restoration.value,
                    restoration.gradient,
                    restoration.covariance_gradient,
                )
                restored_stationarity_converged = (
                    restored_stationarity <= stationarity_tolerance
                    if relative_stationarity_tolerance is None
                    else restored_stationarity_relative
                    <= relative_stationarity_tolerance
                )
                restored_complementarity_converged = (
                    restored_complementarity <= complementarity_tolerance
                    if relative_complementarity_tolerance is None
                    else restored_complementarity_relative
                    <= relative_complementarity_tolerance
                )
                restoration_succeeded = (
                    restoration.value >= 0.0
                    and restored_violation <= constraint_tolerance
                    and restored_stationarity_converged
                    and restored_complementarity_converged
                )
                step_squared = _nonnegative_quadratic(
                    float(
                        np.sum(
                            (restoration.dual - dual) * (restoration.native - native)
                        )
                    ),
                    "radial polishing covariance action",
                )
                step_norm = math.sqrt(step_squared)
                dual = restoration.dual
                native = restoration.native
                final_value = restoration.value
                final_gradient = restoration.gradient
                final_covariance_gradient = restoration.covariance_gradient
                multiplier = restored_multiplier
                stationarity = restored_stationarity
                stationarity_relative = restored_stationarity_relative
                violation = restored_violation
                complementarity = restored_complementarity
                complementarity_relative = restored_complementarity_relative
                squared_action = restored_squared_action
                stationarity_reference = restored_stationarity_reference
                gradient_covariance_norm = restored_gradient_covariance_norm
                radial_restoration_applied = True
                if not restoration_succeeded:
                    radial_restoration_reason += ":candidate_accepted_as_sqp_iterate"
                history.append(
                    NativeScalarSQPIteration(
                        iteration=iteration,
                        objective_value=0.5 * squared_action,
                        constraint_value=final_value,
                        primal_violation=violation,
                        native_gradient_l2_norm=float(np.linalg.norm(final_gradient)),
                        lagrange_multiplier=multiplier,
                        covariance_stationarity_norm=stationarity,
                        covariance_stationarity_relative=stationarity_relative,
                        complementarity_absolute=complementarity,
                        complementarity_relative=complementarity_relative,
                        intervention_action_norm=math.sqrt(squared_action),
                        proposed_step_action_norm=step_norm,
                        accepted_step_action_norm=step_norm,
                        accepted_step_scale=1.0,
                        trust_radius=trust_radius,
                        merit_penalty=merit_penalty,
                        merit_value=(
                            0.5 * squared_action + merit_penalty * abs(final_value)
                        ),
                        line_search_evaluations=0,
                        radial_restoration_reason=radial_restoration_reason,
                        radial_restoration_scale=radial_restoration_scale,
                        radial_restoration_value_evaluations=(
                            attempt.value_evaluations
                        ),
                    )
                )
                if restoration_succeeded:
                    status = (
                        "first_order_kkt_satisfied_after_active_boundary_"
                        "radial_polishing"
                    )
                    success = True
                    break
                # The old local model collapsed the trust radius while trying
                # to reduce |h| from the positive side.  The exact radial
                # replay supplies a new model center on h=0.  At this center,
                # the covariance-metric stationarity norm is the action norm
                # of the full linearized KKT correction, so admitting that
                # radius permits one such correction without exceeding the
                # configured maximum trust radius.
                trust_radius = min(
                    maximum_trust_radius,
                    max(trust_radius, restored_stationarity),
                )
                positive_polish_attempted_native = native.copy()
                status = "active_boundary_radial_iterate_accepted"
                continue
        predicted_radial_expansion = (
            -value / radial_derivative
            if radial_derivative > 0.0 and math.isfinite(radial_derivative)
            else math.inf
        )
        should_try_negative_restoration = (
            known_active_boundary
            and value < 0.0
            and not stationarity_converged
            and stationarity_near_threshold
            and not radial_restoration_applied
            and squared_action > np.finfo(np.float64).tiny
            and math.isfinite(predicted_radial_expansion)
            and 0.0 < predicted_radial_expansion <= 5.0e-2
            and (
                negative_restoration_attempted_native is None
                or not np.array_equal(native, negative_restoration_attempted_native)
            )
        )
        if should_try_negative_restoration:
            negative_restoration_attempted_native = native.copy()
            radial_restoration_reason = (
                "negative_constraint_near_stationary_active_boundary"
            )
            if multiplier > np.finfo(np.float64).tiny:
                if relative_complementarity_tolerance is None:
                    complementarity_value_limit = complementarity_tolerance / multiplier
                else:
                    complementarity_value_limit = (
                        relative_complementarity_tolerance * squared_action / multiplier
                    )
                target_positive_value = min(
                    constraint_tolerance,
                    0.5 * complementarity_value_limit,
                )
            else:
                target_positive_value = constraint_tolerance
            restoration = _restore_nonnegative_constraint_along_native_ray(
                constraint_value,
                constraint_value_gradient,
                common,
                phase_specific,
                dual=dual,
                native=native,
                value=value,
                gradient=gradient,
                covariance_block_rows=covariance_block_rows,
                target_positive_value=max(
                    target_positive_value, np.finfo(np.float64).tiny
                ),
            )
            if restoration is None:
                radial_restoration_reason += ":no_nonnegative_radial_bracket"
            else:
                radial_restoration_scale = restoration.scale
                radial_restoration_value_evaluations += restoration.value_evaluations
                radial_restoration_gradient_evaluations += (
                    restoration.gradient_evaluations
                )
                radial_restoration_covariance_passes += restoration.covariance_passes
                value_evaluations += restoration.value_evaluations
                gradient_evaluations += restoration.gradient_evaluations
                covariance_passes += restoration.covariance_passes
                (
                    restored_multiplier,
                    restored_stationarity,
                    restored_stationarity_relative,
                    restored_violation,
                    restored_complementarity,
                    restored_complementarity_relative,
                    restored_squared_action,
                    restored_stationarity_reference,
                    restored_gradient_covariance_norm,
                ) = _native_kkt_diagnostics(
                    restoration.dual,
                    restoration.native,
                    restoration.value,
                    restoration.gradient,
                    restoration.covariance_gradient,
                )
                restored_stationarity_converged = (
                    restored_stationarity <= stationarity_tolerance
                    if relative_stationarity_tolerance is None
                    else restored_stationarity_relative
                    <= relative_stationarity_tolerance
                )
                restored_complementarity_converged = (
                    restored_complementarity <= complementarity_tolerance
                    if relative_complementarity_tolerance is None
                    else restored_complementarity_relative
                    <= relative_complementarity_tolerance
                )
                restoration_succeeded = (
                    restoration.value >= 0.0
                    and restored_violation <= constraint_tolerance
                    and restored_stationarity_converged
                    and restored_complementarity_converged
                )
                step_squared = _nonnegative_quadratic(
                    float(
                        np.sum(
                            (restoration.dual - dual) * (restoration.native - native)
                        )
                    ),
                    "radial restoration covariance action",
                )
                step_norm = math.sqrt(step_squared)
                dual = restoration.dual
                native = restoration.native
                final_value = restoration.value
                final_gradient = restoration.gradient
                final_covariance_gradient = restoration.covariance_gradient
                multiplier = restored_multiplier
                stationarity = restored_stationarity
                stationarity_relative = restored_stationarity_relative
                violation = restored_violation
                complementarity = restored_complementarity
                complementarity_relative = restored_complementarity_relative
                squared_action = restored_squared_action
                stationarity_reference = restored_stationarity_reference
                gradient_covariance_norm = restored_gradient_covariance_norm
                radial_restoration_applied = True
                if not restoration_succeeded:
                    radial_restoration_reason += ":candidate_accepted_as_sqp_iterate"
                history.append(
                    NativeScalarSQPIteration(
                        iteration=iteration,
                        objective_value=0.5 * squared_action,
                        constraint_value=final_value,
                        primal_violation=violation,
                        native_gradient_l2_norm=float(np.linalg.norm(final_gradient)),
                        lagrange_multiplier=multiplier,
                        covariance_stationarity_norm=stationarity,
                        covariance_stationarity_relative=stationarity_relative,
                        complementarity_absolute=complementarity,
                        complementarity_relative=complementarity_relative,
                        intervention_action_norm=math.sqrt(squared_action),
                        proposed_step_action_norm=step_norm,
                        accepted_step_action_norm=step_norm,
                        accepted_step_scale=1.0,
                        trust_radius=trust_radius,
                        merit_penalty=merit_penalty,
                        merit_value=(
                            0.5 * squared_action + merit_penalty * abs(final_value)
                        ),
                        line_search_evaluations=0,
                        radial_restoration_reason=radial_restoration_reason,
                        radial_restoration_scale=radial_restoration_scale,
                        radial_restoration_value_evaluations=(
                            restoration.value_evaluations
                        ),
                    )
                )
                if restoration_succeeded:
                    status = "first_order_kkt_satisfied_after_radial_restoration"
                    success = True
                    break
                # As on the positive side, the exact boundary replay defines
                # a new local model center.  Admit precisely one full
                # covariance-metric KKT correction after a collapsed trust
                # radius, subject to the configured maximum.
                trust_radius = min(
                    maximum_trust_radius,
                    max(trust_radius, restored_stationarity),
                )
                negative_restoration_attempted_native = native.copy()
                status = "active_boundary_radial_iterate_accepted"
                continue
        converged = (
            violation <= constraint_tolerance
            and stationarity_converged
            and complementarity_converged
        )
        if converged and value < 0.0:
            # The usual inequality gate permits a small negative residual.
            # Publication requires the literal inequality, so preserve this
            # nearly converged iterate for the radial restoration below.
            status = "negative_constraint_within_tolerance_requires_restoration"
            break
        if converged:
            history.append(
                NativeScalarSQPIteration(
                    iteration=iteration,
                    objective_value=objective,
                    constraint_value=value,
                    primal_violation=violation,
                    native_gradient_l2_norm=float(np.linalg.norm(gradient)),
                    lagrange_multiplier=multiplier,
                    covariance_stationarity_norm=stationarity,
                    covariance_stationarity_relative=stationarity_relative,
                    complementarity_absolute=complementarity,
                    complementarity_relative=complementarity_relative,
                    intervention_action_norm=math.sqrt(squared_action),
                    proposed_step_action_norm=0.0,
                    accepted_step_action_norm=0.0,
                    accepted_step_scale=0.0,
                    trust_radius=trust_radius,
                    merit_penalty=merit_penalty,
                    merit_value=merit,
                    line_search_evaluations=0,
                )
            )
            status = (
                "first_order_kkt_satisfied_after_active_boundary_radial_iterate"
                if radial_restoration_applied
                else "first_order_kkt_satisfied"
            )
            success = True
            break
        if iteration == maximum_iterations:
            if radial_restoration_applied:
                status = (
                    "maximum_iterations_reached_after_active_boundary_radial_iterate"
                )
            break

        gradient_metric_squared = _nonnegative_quadratic(
            float(np.sum(gradient * covariance_gradient)),
            "covariance gradient norm",
        )
        if gradient_metric_squared <= np.finfo(np.float64).tiny:
            status = "constraint_gradient_vanished_in_covariance_range"
            break
        linear_boundary = float(np.sum(gradient * native)) - value
        if linear_boundary <= 0.0:
            linear_multiplier = 0.0
            target_dual = np.zeros_like(dual)
            target_native = np.zeros_like(native)
        else:
            linear_multiplier = linear_boundary / gradient_metric_squared
            target_dual = linear_multiplier * gradient
            target_native = linear_multiplier * covariance_gradient
        dual_step = target_dual - dual
        native_step = target_native - native
        proposed_step_squared = _nonnegative_quadratic(
            float(np.sum(dual_step * native_step)),
            "proposed step covariance action",
        )
        proposed_step_norm = math.sqrt(proposed_step_squared)
        if proposed_step_norm > trust_radius:
            ratio = trust_radius / proposed_step_norm
            dual_step *= ratio
            native_step *= ratio
        model_step_squared = _nonnegative_quadratic(
            float(np.sum(dual_step * native_step)),
            "trust-region SQP step covariance action",
        )
        model_step_norm = math.sqrt(model_step_squared)

        merit_penalty = max(
            merit_penalty,
            1.5 * linear_multiplier + 1.0e-12,
        )
        objective_direction = float(np.sum(dual * native_step))
        constraint_direction = float(np.sum(gradient * native_step))
        residual_direction = math.nan
        if active_boundary_merit:
            if value == 0.0:
                residual_direction = abs(constraint_direction)
            else:
                residual_direction = math.copysign(1.0, value) * (constraint_direction)
            directional_merit = objective_direction + merit_penalty * residual_direction
        elif value < 0.0:
            directional_merit = (
                objective_direction - merit_penalty * constraint_direction
            )
        else:
            directional_merit = objective_direction
        if directional_merit >= 0.0 and (
            (active_boundary_merit and residual_direction < 0.0)
            or (
                not active_boundary_merit and value < 0.0 and constraint_direction > 0.0
            )
        ):
            residual_decrease = (
                -residual_direction if active_boundary_merit else constraint_direction
            )
            needed = objective_direction / residual_decrease
            merit_penalty = max(merit_penalty, 1.5 * needed + 1.0e-12)
            if active_boundary_merit:
                directional_merit = (
                    objective_direction + merit_penalty * residual_direction
                )
            else:
                directional_merit = (
                    objective_direction - merit_penalty * constraint_direction
                )
        if directional_merit >= 0.0:
            status = "no_exact_merit_descent_direction"
            break

        scale = 1.0
        accepted_dual: np.ndarray | None = None
        accepted_native: np.ndarray | None = None
        accepted_value = math.nan
        accepted_second_order_correction = False
        accepted_second_order_correction_norm = 0.0
        accepted_second_order_correction_value_evaluations = 0
        line_evaluations = 0
        allow_second_order_correction = (
            active_boundary_merit
            and abs(value) <= constraint_tolerance
            and stationarity_near_threshold
            and not stationarity_converged
            and complementarity_converged
            and model_step_norm > np.finfo(np.float64).tiny
        )
        for _ in range(maximum_line_search_steps):
            trial_dual = dual + scale * dual_step
            trial_native = native + scale * native_step
            trial_value = float(constraint_value(trial_native.copy()))
            value_evaluations += 1
            line_evaluations += 1
            if not math.isfinite(trial_value):
                scale *= backtrack_factor
                continue
            trial_squared_action = _nonnegative_quadratic(
                float(np.sum(trial_dual * trial_native)),
                "trial covariance action",
            )
            trial_residual = (
                abs(trial_value) if active_boundary_merit else max(0.0, -trial_value)
            )
            trial_merit = 0.5 * trial_squared_action + merit_penalty * trial_residual
            armijo_bound = merit + armijo_fraction * scale * directional_merit
            objective_armijo_bound = (
                objective + armijo_fraction * scale * objective_direction
            )
            temporary_constraint_funnel = max(
                10.0 * constraint_tolerance,
                2.0 * (scale * model_step_norm) ** 2,
            )
            tangent_filter_accepts = (
                allow_second_order_correction
                and trial_residual <= temporary_constraint_funnel
                and 0.5 * trial_squared_action <= objective_armijo_bound
            )
            if trial_merit <= armijo_bound:
                accepted_dual = trial_dual
                accepted_native = trial_native
                accepted_value = trial_value
                break
            if allow_second_order_correction:
                correction_coefficient = -trial_value / gradient_metric_squared
                primary_step_norm = scale * model_step_norm
                prior_corrected_residual = abs(trial_value)
                for correction_evaluations in range(1, 5):
                    correction_norm = abs(correction_coefficient) * math.sqrt(
                        gradient_metric_squared
                    )
                    correction_is_safeguarded = (
                        math.isfinite(correction_coefficient)
                        and math.isfinite(correction_norm)
                        and correction_norm > 0.0
                        and correction_norm <= primary_step_norm
                    )
                    if not correction_is_safeguarded:
                        break
                    corrected_dual = trial_dual + correction_coefficient * gradient
                    corrected_native = (
                        trial_native + correction_coefficient * covariance_gradient
                    )
                    corrected_step_squared = _nonnegative_quadratic(
                        float(
                            np.sum(
                                (corrected_dual - dual) * (corrected_native - native)
                            )
                        ),
                        "second-order corrected step covariance action",
                    )
                    corrected_step_norm = math.sqrt(corrected_step_squared)
                    trust_slack = (
                        16.0 * np.finfo(np.float64).eps * max(1.0, trust_radius)
                    )
                    if corrected_step_norm <= trust_radius + trust_slack:
                        corrected_value = float(
                            constraint_value(corrected_native.copy())
                        )
                        value_evaluations += 1
                        line_evaluations += 1
                        corrected_residual = abs(corrected_value)
                        if (
                            math.isfinite(corrected_value)
                            and corrected_residual < prior_corrected_residual
                        ):
                            corrected_squared_action = _nonnegative_quadratic(
                                float(np.sum(corrected_dual * corrected_native)),
                                "second-order corrected covariance action",
                            )
                            corrected_merit = (
                                0.5 * corrected_squared_action
                                + merit_penalty * abs(corrected_value)
                            )
                            trust_funnel_accepts = (
                                corrected_residual <= constraint_tolerance
                                and 0.5 * corrected_squared_action
                                <= objective_armijo_bound
                            )
                            if (
                                corrected_merit <= armijo_bound
                                or trust_funnel_accepts
                            ):
                                accepted_dual = corrected_dual
                                accepted_native = corrected_native
                                accepted_value = corrected_value
                                accepted_second_order_correction = True
                                accepted_second_order_correction_norm = correction_norm
                                accepted_second_order_correction_value_evaluations = (
                                    correction_evaluations
                                )
                                break
                            prior_corrected_residual = corrected_residual
                            correction_coefficient += (
                                -corrected_value / gradient_metric_squared
                            )
                            continue
                    break
                if accepted_dual is not None:
                    break
            if tangent_filter_accepts:
                accepted_dual = trial_dual
                accepted_native = trial_native
                accepted_value = trial_value
                break
            scale *= backtrack_factor

        if accepted_dual is None or accepted_native is None:
            trust_radius *= 0.25
            if trust_radius <= 10.0 * np.finfo(np.float64).eps:
                status = "line_search_failed_at_minimum_trust_radius"
                break
            history.append(
                NativeScalarSQPIteration(
                    iteration=iteration,
                    objective_value=objective,
                    constraint_value=value,
                    primal_violation=violation,
                    native_gradient_l2_norm=float(np.linalg.norm(gradient)),
                    lagrange_multiplier=multiplier,
                    covariance_stationarity_norm=stationarity,
                    covariance_stationarity_relative=stationarity_relative,
                    complementarity_absolute=complementarity,
                    complementarity_relative=complementarity_relative,
                    intervention_action_norm=math.sqrt(squared_action),
                    proposed_step_action_norm=proposed_step_norm,
                    accepted_step_action_norm=0.0,
                    accepted_step_scale=0.0,
                    trust_radius=trust_radius,
                    merit_penalty=merit_penalty,
                    merit_value=merit,
                    line_search_evaluations=line_evaluations,
                )
            )
            continue

        accepted_step_squared = _nonnegative_quadratic(
            float(np.sum((accepted_dual - dual) * (accepted_native - native))),
            "accepted step covariance action",
        )
        accepted_step_norm = math.sqrt(accepted_step_squared)
        history.append(
            NativeScalarSQPIteration(
                iteration=iteration,
                objective_value=objective,
                constraint_value=value,
                primal_violation=violation,
                native_gradient_l2_norm=float(np.linalg.norm(gradient)),
                lagrange_multiplier=multiplier,
                covariance_stationarity_norm=stationarity,
                covariance_stationarity_relative=stationarity_relative,
                complementarity_absolute=complementarity,
                complementarity_relative=complementarity_relative,
                intervention_action_norm=math.sqrt(squared_action),
                proposed_step_action_norm=proposed_step_norm,
                accepted_step_action_norm=accepted_step_norm,
                accepted_step_scale=scale,
                trust_radius=trust_radius,
                merit_penalty=merit_penalty,
                merit_value=merit,
                line_search_evaluations=line_evaluations,
                second_order_correction_applied=(accepted_second_order_correction),
                second_order_correction_action_norm=(
                    accepted_second_order_correction_norm
                ),
                second_order_correction_value_evaluations=(
                    accepted_second_order_correction_value_evaluations
                ),
            )
        )
        dual = accepted_dual
        native = accepted_native
        final_value = accepted_value
        if scale == 1.0 and accepted_step_norm >= 0.8 * trust_radius:
            trust_radius = min(maximum_trust_radius, 2.0 * trust_radius)
        elif scale < 0.5:
            trust_radius = max(10.0 * np.finfo(np.float64).eps, 0.5 * trust_radius)

    (
        multiplier,
        stationarity,
        stationarity_relative,
        violation,
        complementarity,
        complementarity_relative,
        squared_action,
        stationarity_reference,
        gradient_covariance_norm,
    ) = _native_kkt_diagnostics(
        dual,
        native,
        final_value,
        final_gradient,
        final_covariance_gradient,
    )
    stationarity_converged = (
        stationarity <= stationarity_tolerance
        if relative_stationarity_tolerance is None
        else stationarity_relative <= relative_stationarity_tolerance
    )
    if (
        not success
        and final_value < 0.0
        and stationarity_converged
        and squared_action > np.finfo(np.float64).tiny
    ):
        radial_restoration_reason = (
            "negative_constraint_at_otherwise_stationary_sqp_termination"
        )
        if multiplier > np.finfo(np.float64).tiny:
            if relative_complementarity_tolerance is None:
                complementarity_value_limit = complementarity_tolerance / multiplier
            else:
                complementarity_value_limit = (
                    relative_complementarity_tolerance * squared_action / multiplier
                )
            target_positive_value = min(
                constraint_tolerance,
                0.5 * complementarity_value_limit,
            )
        else:
            target_positive_value = constraint_tolerance
        restoration = _restore_nonnegative_constraint_along_native_ray(
            constraint_value,
            constraint_value_gradient,
            common,
            phase_specific,
            dual=dual,
            native=native,
            value=final_value,
            gradient=final_gradient,
            covariance_block_rows=covariance_block_rows,
            target_positive_value=max(target_positive_value, np.finfo(np.float64).tiny),
        )
        if restoration is None:
            radial_restoration_reason += ":no_nonnegative_radial_bracket"
        else:
            radial_restoration_scale = restoration.scale
            radial_restoration_value_evaluations += restoration.value_evaluations
            radial_restoration_gradient_evaluations += restoration.gradient_evaluations
            radial_restoration_covariance_passes += restoration.covariance_passes
            value_evaluations += restoration.value_evaluations
            gradient_evaluations += restoration.gradient_evaluations
            covariance_passes += restoration.covariance_passes
            (
                restored_multiplier,
                restored_stationarity,
                restored_stationarity_relative,
                restored_violation,
                restored_complementarity,
                restored_complementarity_relative,
                restored_squared_action,
                restored_stationarity_reference,
                restored_gradient_covariance_norm,
            ) = _native_kkt_diagnostics(
                restoration.dual,
                restoration.native,
                restoration.value,
                restoration.gradient,
                restoration.covariance_gradient,
            )
            restored_stationarity_converged = (
                restored_stationarity <= stationarity_tolerance
                if relative_stationarity_tolerance is None
                else restored_stationarity_relative <= relative_stationarity_tolerance
            )
            restored_complementarity_converged = (
                restored_complementarity <= complementarity_tolerance
                if relative_complementarity_tolerance is None
                else restored_complementarity_relative
                <= relative_complementarity_tolerance
            )
            restoration_succeeded = (
                restoration.value >= 0.0
                and restored_violation <= constraint_tolerance
                and restored_stationarity_converged
                and restored_complementarity_converged
            )
            if restoration_succeeded:
                previous_dual = dual
                previous_native = native
                step_squared = _nonnegative_quadratic(
                    float(
                        np.sum(
                            (restoration.dual - previous_dual)
                            * (restoration.native - previous_native)
                        )
                    ),
                    "radial restoration covariance action",
                )
                step_norm = math.sqrt(step_squared)
                dual = restoration.dual
                native = restoration.native
                final_value = restoration.value
                final_gradient = restoration.gradient
                final_covariance_gradient = restoration.covariance_gradient
                multiplier = restored_multiplier
                stationarity = restored_stationarity
                stationarity_relative = restored_stationarity_relative
                violation = restored_violation
                complementarity = restored_complementarity
                complementarity_relative = restored_complementarity_relative
                squared_action = restored_squared_action
                stationarity_reference = restored_stationarity_reference
                gradient_covariance_norm = restored_gradient_covariance_norm
                history.append(
                    NativeScalarSQPIteration(
                        iteration=max((item.iteration for item in history), default=-1)
                        + 1,
                        objective_value=0.5 * squared_action,
                        constraint_value=final_value,
                        primal_violation=violation,
                        native_gradient_l2_norm=float(np.linalg.norm(final_gradient)),
                        lagrange_multiplier=multiplier,
                        covariance_stationarity_norm=stationarity,
                        covariance_stationarity_relative=stationarity_relative,
                        complementarity_absolute=complementarity,
                        complementarity_relative=complementarity_relative,
                        intervention_action_norm=math.sqrt(squared_action),
                        proposed_step_action_norm=step_norm,
                        accepted_step_action_norm=step_norm,
                        accepted_step_scale=1.0,
                        trust_radius=trust_radius,
                        merit_penalty=merit_penalty,
                        merit_value=(
                            0.5 * squared_action + merit_penalty * abs(final_value)
                        ),
                        line_search_evaluations=0,
                        radial_restoration_reason=radial_restoration_reason,
                        radial_restoration_scale=radial_restoration_scale,
                        radial_restoration_value_evaluations=(
                            radial_restoration_value_evaluations
                        ),
                    )
                )
                radial_restoration_applied = True
                status = "first_order_kkt_satisfied_after_radial_restoration"
                success = True
            else:
                radial_restoration_reason += ":candidate_failed_original_kkt_gates"
    return NativeScalarSQPResult(
        native_interventions=native.copy(),
        dual_variables=dual.copy(),
        success=success,
        status=status,
        objective_value=0.5 * squared_action,
        squared_action=squared_action,
        constraint_value=final_value,
        primal_violation=violation,
        native_constraint_gradient=final_gradient.copy(),
        lagrange_multiplier=multiplier,
        covariance_stationarity_norm=stationarity,
        covariance_stationarity_relative=stationarity_relative,
        complementarity_absolute=complementarity,
        complementarity_relative=complementarity_relative,
        stationarity_reference_norm=stationarity_reference,
        constraint_gradient_covariance_norm=gradient_covariance_norm,
        iterations=tuple(history),
        value_evaluations=value_evaluations,
        gradient_evaluations=gradient_evaluations,
        covariance_passes=covariance_passes,
        radial_restoration_applied=radial_restoration_applied,
        radial_restoration_reason=radial_restoration_reason,
        radial_restoration_scale=radial_restoration_scale,
        radial_restoration_value_evaluations=(radial_restoration_value_evaluations),
        radial_restoration_gradient_evaluations=(
            radial_restoration_gradient_evaluations
        ),
        radial_restoration_covariance_passes=radial_restoration_covariance_passes,
    )
