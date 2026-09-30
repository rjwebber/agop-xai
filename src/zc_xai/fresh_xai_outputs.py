"""Shared validation helpers for fresh-ZC XAI tables and map figures."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .data import ZCData
from .xai import coherence_score

FRESH_TABLE_SCHEMA_VERSION = 2
FRESH_XAI_BUNDLE_SCHEMA_VERSION = 2
FRESH_TABLE_METHODS = ("AGOP", "GradientSHAP", "IG", "GRAD")
FRESH_TABLE_ARCHITECTURES = ("mlp", "cnn", "vit")
FRESH_XAI_GRADIENT_BATCH_SIZE = 1024
FRESH_IG_GRADIENT_COUNT = 1024
FRESH_EXPECTED_GRADIENT_COUNT = 1024
FRESH_ROBUSTNESS_NEIGHBOR_PERCENT = 1.0
FRESH_XAI_TRAINING_POPULATION = (
    "all lead-valid predictors in the fixed 10,000-year training block"
)


def split_spatial_phase(
    values: np.ndarray,
    data: ZCData,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate a fresh input vector into its spatial tensor and phase scalars."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.n_phase_features != 2:
        raise ValueError("Fresh XAI outputs require exactly two annual-phase scalars.")
    spatial_size = int(np.prod(data.spatial_input_shape))
    if vector.size != spatial_size + data.n_phase_features:
        raise ValueError(
            f"Input has {vector.size} coefficients; expected "
            f"{spatial_size + data.n_phase_features}."
        )
    spatial = vector[:spatial_size].reshape(data.spatial_input_shape)
    phase = vector[spatial_size:].copy()
    if not np.isfinite(spatial).all() or not np.isfinite(phase).all():
        raise ValueError("Fresh XAI vectors must be finite.")
    return spatial, phase


def spatial_coherence(values: np.ndarray, data: ZCData) -> float:
    """Compute coherence only on spatial fields, explicitly excluding phase."""

    spatial, _ = split_spatial_phase(values, data)
    return coherence_score(spatial.reshape(-1), data.spatial_input_shape)


def phase_coefficient_summary(values: np.ndarray, data: ZCData) -> dict[str, float]:
    """Report the two unplotted phase coefficients and their squared mass."""

    spatial, phase = split_spatial_phase(values, data)
    total_mass = float(np.square(np.asarray(values, dtype=np.float64)).sum())
    if total_mass <= np.finfo(np.float64).eps:
        raise ValueError("Phase mass is undefined for a zero explanation.")
    phase_mass = float(np.square(phase).sum())
    return {
        "annual_phase_sin_coefficient": float(phase[0]),
        "annual_phase_cos_coefficient": float(phase[1]),
        "annual_phase_squared_mass_fraction": phase_mass / total_mass,
        "spatial_squared_mass_fraction": (
            float(np.square(spatial).sum()) / total_mass
        ),
    }


def validate_unit_explanation(values: np.ndarray, feature_count: int) -> np.ndarray:
    """Return one finite unit explanation after strict schema validation."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (feature_count,) or not np.isfinite(vector).all():
        raise ValueError("Explanation has an invalid shape or non-finite values.")
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=2.0e-6, abs_tol=2.0e-6):
        raise ValueError(f"Explanation is not unit norm (norm={norm:.9g}).")
    return vector


def exact_robustness_summary(overlaps: np.ndarray) -> dict[str, Any]:
    """Summarize an exhaustively evaluated finite neighborhood population."""

    values = np.asarray(overlaps, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Robustness overlaps must be a finite nonempty vector.")
    if np.any((values < -1.0000001) | (values > 1.0000001)):
        raise ValueError("Robustness overlaps lie outside the cosine range.")
    values = np.clip(values, -1.0, 1.0)
    sample_sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "robustness": float(values.mean()),
        "explanation_change": float(1.0 - values.mean()),
        "population_standard_deviation": sample_sd,
        "finite_population_standard_error": 0.0,
        "evaluated_count": int(values.size),
        "exhaustive": True,
    }
