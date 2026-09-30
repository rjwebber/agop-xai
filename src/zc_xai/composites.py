"""Event-equal composite helpers for the Zebiak--Cane experiments.

The plotting script deliberately keeps event selection here, separate from the
figure code, so the scientific definition can be unit tested and reused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearResidualFit:
    """Training-population fit for a one-predictor standardized residual."""

    response_mean: float
    predictor_mean: float
    slope: float
    residual_scale: float

    def transform(
        self,
        response: np.ndarray,
        predictor: np.ndarray,
    ) -> np.ndarray:
        """Return residuals in units of the training residual standard deviation."""

        response_values = np.asarray(response, dtype=np.float64)
        predictor_values = np.asarray(predictor, dtype=np.float64)
        if response_values.shape != predictor_values.shape:
            raise ValueError("response and predictor must have the same shape")
        if not (
            np.isfinite(response_values).all()
            and np.isfinite(predictor_values).all()
        ):
            raise ValueError("response and predictor must be finite")
        residual = (
            response_values
            - self.response_mean
            - self.slope * (predictor_values - self.predictor_mean)
        )
        return residual / self.residual_scale


def threshold_event_peaks(
    target_values: np.ndarray,
    target_steps: np.ndarray,
    *,
    threshold_c: float,
) -> np.ndarray:
    """Return one peak step per complete contiguous threshold exceedance.

    ``target_steps`` must be a contiguous held-out target-date interval.  An
    exceedance touching either end is omitted because its event peak may lie
    outside the observed interval.  Ties are resolved at the earliest step.
    """

    values = np.asarray(target_values, dtype=np.float64)
    steps = np.asarray(target_steps, dtype=np.int64)
    if values.ndim != 1 or steps.ndim != 1 or values.size != steps.size:
        raise ValueError("target_values and target_steps must be equal-length rows")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("target_values must be nonempty and finite")
    if not math.isfinite(threshold_c):
        raise ValueError("threshold_c must be finite")
    if not np.array_equal(steps, np.arange(steps[0], steps[-1] + 1)):
        raise ValueError("target_steps must be sorted, unique, and contiguous")

    above = values >= threshold_c
    padded = np.concatenate(([False], above, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    peaks: list[int] = []
    for start, stop in zip(starts, stops, strict=True):
        if start == 0 or stop == values.size:
            continue
        local_position = int(np.argmax(values[start:stop]))
        peaks.append(int(steps[start + local_position]))
    return np.asarray(peaks, dtype=np.int64)


def strongest_fraction(
    peak_steps: np.ndarray,
    full_target: np.ndarray,
    *,
    percent: float,
) -> np.ndarray:
    """Return the strongest ``ceil(percent * n / 100)`` peaks, in rank order."""

    peaks = np.asarray(peak_steps, dtype=np.int64)
    target = np.asarray(full_target)
    if peaks.ndim != 1 or peaks.size == 0:
        raise ValueError("peak_steps must be a nonempty one-dimensional array")
    if not 0.0 < percent <= 100.0 or not math.isfinite(percent):
        raise ValueError("percent must lie in (0, 100]")
    if np.any(peaks < 0) or np.any(peaks >= target.size):
        raise IndexError("peak_steps contains an out-of-range step")
    count = max(1, int(math.ceil(peaks.size * percent / 100.0)))
    order = np.argsort(-np.asarray(target[peaks], dtype=np.float64), kind="stable")
    return np.asarray(peaks[order[:count]], dtype=np.int64)


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Event-resampled percentile interval for a mean along axis zero."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim < 1 or samples.shape[0] < 2 or not np.isfinite(samples).all():
        raise ValueError("values must contain at least two finite event rows")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        samples.shape[0],
        np.full(samples.shape[0], 1.0 / samples.shape[0]),
        size=resamples,
    )
    bootstrapped = np.tensordot(
        counts / samples.shape[0],
        samples,
        axes=(1, 0),
    )
    tail = 0.5 * (1.0 - confidence)
    return (
        np.quantile(bootstrapped, tail, axis=0),
        np.quantile(bootstrapped, 1.0 - tail, axis=0),
    )


def rounded_symmetric_limit(values: np.ndarray, *, increment: float = 0.5) -> float:
    """Round a finite nonzero maximum magnitude upward to a simple limit."""

    array = np.asarray(values, dtype=np.float64)
    if increment <= 0.0 or not math.isfinite(increment):
        raise ValueError("increment must be finite and positive")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be nonempty and finite")
    maximum = float(np.max(np.abs(array)))
    if maximum <= np.finfo(np.float64).eps:
        return increment
    return increment * math.ceil(maximum / increment)


def geographic_mask(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    latitude_bounds: tuple[float, float],
    longitude_bounds: tuple[float, float],
) -> np.ndarray:
    """Return an inclusive rectangular mask on a monotone 0--360-degree grid."""

    latitude_values = np.asarray(latitudes, dtype=np.float64)
    longitude_values = np.asarray(longitudes, dtype=np.float64)
    if (
        latitude_values.ndim != 1
        or longitude_values.ndim != 1
        or latitude_values.size == 0
        or longitude_values.size == 0
        or not np.isfinite(latitude_values).all()
        or not np.isfinite(longitude_values).all()
        or np.any(np.diff(latitude_values) <= 0.0)
        or np.any(np.diff(longitude_values) <= 0.0)
    ):
        raise ValueError(
            "latitude and longitude coordinates must be finite and monotone"
        )
    lat_min, lat_max = map(float, latitude_bounds)
    lon_min, lon_max = map(float, longitude_bounds)
    if not all(map(math.isfinite, (lat_min, lat_max, lon_min, lon_max))):
        raise ValueError("geographic bounds must be finite")
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("geographic bounds must be ordered")
    mask = (
        (latitude_values[:, None] >= lat_min)
        & (latitude_values[:, None] <= lat_max)
        & (longitude_values[None, :] >= lon_min)
        & (longitude_values[None, :] <= lon_max)
    )
    if not mask.any():
        raise ValueError("geographic bounds select no grid centers")
    return mask


def area_weighted_mean(
    values: np.ndarray,
    mask: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    """Average the final latitude/longitude axes using cosine-area weights."""

    samples = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    latitude_values = np.asarray(latitudes, dtype=np.float64)
    if samples.ndim < 2 or samples.shape[-2:] != selected.shape:
        raise ValueError("values must end with the mask's latitude/longitude shape")
    if latitude_values.shape != (selected.shape[0],):
        raise ValueError("latitudes do not match the mask")
    if not np.isfinite(samples).all() or not np.isfinite(latitude_values).all():
        raise ValueError("values and latitudes must be finite")
    weights = selected * np.cos(np.deg2rad(latitude_values))[:, None]
    total_weight = float(weights.sum())
    if total_weight <= np.finfo(np.float64).eps:
        raise ValueError("the selected region has no positive area weight")
    normalized = weights / total_weight
    return np.tensordot(samples, normalized, axes=((-2, -1), (0, 1)))


def fit_population_standardization(values: np.ndarray) -> tuple[float, float]:
    """Fit a finite population mean and standard deviation to scalar values."""

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size < 2 or not np.isfinite(samples).all():
        raise ValueError("at least two finite scalar values are required")
    mean = float(samples.mean())
    scale = float(samples.std(ddof=0))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("scalar population has zero variance")
    return mean, scale


def apply_population_standardization(
    values: np.ndarray,
    *,
    mean: float,
    scale: float,
) -> np.ndarray:
    """Apply a recorded scalar population standardization."""

    samples = np.asarray(values, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError("scalar values must be finite")
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("mean and scale must define a finite positive standardizer")
    return (samples - mean) / scale


def fit_linear_residual(
    response: np.ndarray,
    predictor: np.ndarray,
) -> LinearResidualFit:
    """Fit ``response`` after removing its linear dependence on ``predictor``."""

    response_values = np.asarray(response, dtype=np.float64).reshape(-1)
    predictor_values = np.asarray(predictor, dtype=np.float64).reshape(-1)
    if response_values.shape != predictor_values.shape or response_values.size < 2:
        raise ValueError("response and predictor must contain equal nontrivial rows")
    if not (
        np.isfinite(response_values).all()
        and np.isfinite(predictor_values).all()
    ):
        raise ValueError("response and predictor must be finite")
    response_mean = float(response_values.mean())
    predictor_mean = float(predictor_values.mean())
    centered_predictor = predictor_values - predictor_mean
    predictor_sum_squares = float(np.dot(centered_predictor, centered_predictor))
    if predictor_sum_squares <= np.finfo(np.float64).eps:
        raise ValueError("predictor has zero variance")
    slope = float(
        np.dot(response_values - response_mean, centered_predictor)
        / predictor_sum_squares
    )
    residual = (
        response_values
        - response_mean
        - slope * centered_predictor
    )
    residual_scale = float(residual.std(ddof=0))
    if residual_scale <= np.finfo(np.float64).eps:
        raise ValueError("linear residual has zero variance")
    return LinearResidualFit(
        response_mean=response_mean,
        predictor_mean=predictor_mean,
        slope=slope,
        residual_scale=residual_scale,
    )
