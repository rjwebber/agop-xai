"""Linear algebra for a model-constrained AGOP sufficiency experiment.

The routines in this module deliberately keep the *construction* of a release
state separate from the later Zebiak--Cane forecast.  A response matrix maps a
small set of dimensionless, covariance-scaled native controls into the
standardized four-ocean-field observation at release.  The constrained solve
uses only that release response.  The future Nino-3 value is therefore an
independent diagnostic rather than part of the optimization objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class AGOPAlignmentSolution:
    """Solution of the linearized minimum-action AGOP alignment problem."""

    control: np.ndarray
    release_displacement: np.ndarray
    agop_direction: np.ndarray
    requested_projection: float
    realized_projection: float
    perpendicular_norm: float
    control_norm: float
    objective_value: float
    lagrange_multiplier: float
    kkt_stationarity_norm: float
    response_rank: int
    response_condition_number: float


def unit_fixed_phase_direction(
    direction: np.ndarray,
    *,
    phase_features: int = 2,
) -> np.ndarray:
    """Return a unit spatial direction after holding phase coordinates fixed.

    AGOP explanations for the fresh data contain 2,160 spatial coefficients
    followed by sine and cosine of annual phase.  Time is passive in the
    authentic ZC adjoint, so the scientific intervention must not alter those
    final two coordinates.
    """

    result = np.asarray(direction, dtype=np.float64).copy()
    if result.ndim != 1 or result.size <= phase_features:
        raise ValueError("direction must be a vector containing spatial features")
    if phase_features < 0:
        raise ValueError("phase_features must be nonnegative")
    if not np.isfinite(result).all():
        raise ValueError("direction must be finite")
    if phase_features:
        result[-phase_features:] = 0.0
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("fixed-phase direction has zero norm")
    result /= norm
    return result


def half_cosine_iau_weights(step_count: int) -> np.ndarray:
    """Return smooth positive incremental-analysis weights summing to one."""

    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise TypeError("step_count must be an integer")
    if step_count <= 0:
        raise ValueError("step_count must be positive")
    phase = np.arange(step_count + 1, dtype=np.float64) / step_count
    cumulative = 0.5 * (1.0 - np.cos(np.pi * phase))
    weights = np.diff(cumulative)
    weights /= np.sum(weights, dtype=np.float64)
    return weights


def solve_linearized_agop_alignment(
    response: np.ndarray,
    agop_direction: np.ndarray,
    *,
    projection: float,
    perpendicular_penalty: float,
    rank_tolerance: float | None = None,
) -> AGOPAlignmentSolution:
    r"""Solve the minimum-action release-alignment problem in closed form.

    With standardized release response ``r = A a`` and unit AGOP direction
    ``e``, this solves

    .. math::

       \min_a \tfrac12\|a\|_2^2
          + \tfrac\beta2\|(I-ee^T)Aa\|_2^2
       \quad\text{s.t.}\quad e^T A a = \alpha.

    The identity term makes the quadratic positive definite even when ``A``
    is rank deficient.  In covariance-whitened native coordinates,
    ``||a||^2`` is the empirical Mahalanobis action.
    """

    matrix = np.asarray(response, dtype=np.float64)
    direction = np.asarray(agop_direction, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("response must be a nonempty two-dimensional matrix")
    if direction.shape != (matrix.shape[0],):
        raise ValueError("agop_direction does not match the response rows")
    if not np.isfinite(matrix).all() or not np.isfinite(direction).all():
        raise ValueError("response and agop_direction must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if not np.isclose(direction_norm, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("agop_direction must have unit Euclidean norm")
    alpha = float(projection)
    beta = float(perpendicular_penalty)
    if not math.isfinite(alpha):
        raise ValueError("projection must be finite")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("perpendicular_penalty must be finite and nonnegative")

    along_response = direction @ matrix
    perpendicular_response = matrix - np.outer(direction, along_response)
    quadratic = np.eye(matrix.shape[1], dtype=np.float64)
    if beta:
        quadratic += beta * (perpendicular_response.T @ perpendicular_response)
    factor = scipy.linalg.cho_factor(quadratic, lower=True, check_finite=True)
    inverse_constraint = scipy.linalg.cho_solve(
        factor, along_response, check_finite=True
    )
    denominator = float(along_response @ inverse_constraint)
    scale = max(
        1.0,
        float(np.linalg.norm(along_response))
        * float(np.linalg.norm(inverse_constraint)),
    )
    if denominator <= np.finfo(np.float64).eps * scale:
        raise ValueError(
            "balanced controls cannot produce a nonzero displacement along "
            "the requested AGOP direction"
        )

    multiplier = -alpha / denominator
    control = -multiplier * inverse_constraint
    release = matrix @ control
    realized = float(direction @ release)
    perpendicular = release - direction * realized
    objective = 0.5 * float(control @ control) + 0.5 * beta * float(
        perpendicular @ perpendicular
    )
    stationarity = quadratic @ control + multiplier * along_response

    singular_values = scipy.linalg.svdvals(matrix, check_finite=True)
    if rank_tolerance is None:
        tolerance = (
            max(matrix.shape)
            * np.finfo(np.float64).eps
            * float(singular_values[0])
        )
    else:
        tolerance = float(rank_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("rank_tolerance must be finite and nonnegative")
    positive = singular_values[singular_values > tolerance]
    condition = (
        float(positive[0] / positive[-1]) if positive.size else math.inf
    )

    return AGOPAlignmentSolution(
        control=control,
        release_displacement=release,
        agop_direction=direction.copy(),
        requested_projection=alpha,
        realized_projection=realized,
        perpendicular_norm=float(np.linalg.norm(perpendicular)),
        control_norm=float(np.linalg.norm(control)),
        objective_value=objective,
        lagrange_multiplier=multiplier,
        kkt_stationarity_norm=float(np.linalg.norm(stationarity)),
        response_rank=int(np.count_nonzero(singular_values > tolerance)),
        response_condition_number=condition,
    )


def control_space_forecast_gradient(
    native_control_map: np.ndarray,
    native_state_gradient: np.ndarray,
) -> np.ndarray:
    """Apply ``B.T`` to a packed native-state forecast gradient."""

    basis = np.asarray(native_control_map, dtype=np.float64)
    gradient = np.asarray(native_state_gradient, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("native_control_map must be a matrix")
    if gradient.shape != (basis.shape[0],):
        raise ValueError("native_state_gradient does not match the map rows")
    if not np.isfinite(basis).all() or not np.isfinite(gradient).all():
        raise ValueError("native control map and gradient must be finite")
    return basis.T @ gradient
