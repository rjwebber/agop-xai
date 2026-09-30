"""Reproducible XAI directions for native ZC sufficiency experiments.

The intervention calculation acts only on the 2,160 standardized spatial
coordinates.  Annual sine/cosine phase features are retained by the forecast
model but are passive in the ZC dynamics, so they are removed and the spatial
direction is renormalized before defining a release constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agop_control import unit_fixed_phase_direction
from .data import ZCData
from .fresh_case_studies import expected_gradient_references
from .io import sha256_array
from .training import LoadedExperiment
from .xai import (
    AgopExplainer,
    AgopFactor,
    GradientExplainer,
    GradientShapExplainer,
    IntegratedGradientsExplainer,
)

XAI_METHODS = ("AGOP", "GRAD", "IG", "GradientSHAP")
XAI_METHOD_KEYS = {
    "AGOP": "agop",
    "GRAD": "grad",
    "IG": "ig",
    "GradientSHAP": "gradientshap",
}
GRADIENT_BATCH_SIZE = 1024
INTEGRATED_GRADIENTS_STEPS = 1024
GRADIENT_SHAP_SAMPLES = 1024
GRADIENT_SHAP_BACKGROUND_SEED = 42
GRADIENT_SHAP_ALPHA_SEED = 43


@dataclass(frozen=True)
class DirectXAITarget:
    """One event explanation and its fixed-phase, event-oriented direction."""

    method: str
    explanation: np.ndarray
    fixed_phase_explanation: np.ndarray
    raw_spatial_direction: np.ndarray
    oriented_spatial_direction: np.ndarray
    raw_event_projection: float
    oriented_event_projection: float
    orientation_multiplier: float
    provenance: dict[str, Any]
    reference_arrays: dict[str, np.ndarray]


def _method_explainer(
    method: str,
    *,
    data: ZCData,
    experiment: LoadedExperiment,
    agop_factor: AgopFactor | None,
    device: str,
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    if method == "AGOP":
        if agop_factor is None:
            raise ValueError("AGOP requires its validated exact factor")
        return (
            AgopExplainer(agop_factor),
            {
                "definition": "exact empirical AGOP square root times input",
                "reference_population": "all experiment.fit_inputs",
                "reference_count": int(agop_factor.reference_indices.size),
                "reference_indices_sha256": sha256_array(
                    agop_factor.reference_indices
                ),
                "rank": int(agop_factor.approximation_rank),
            },
            {},
        )
    if agop_factor is not None:
        raise ValueError(f"{method} must not receive an AGOP factor")
    if method == "GRAD":
        return (
            GradientExplainer(
                experiment.model,
                batch_size=GRADIENT_BATCH_SIZE,
                device=device,
            ),
            {
                "definition": "forecast gradient at the standardized event input",
                "gradient_batch_size": GRADIENT_BATCH_SIZE,
            },
            {},
        )
    if method == "IG":
        return (
            IntegratedGradientsExplainer(
                experiment.model,
                n_steps=INTEGRATED_GRADIENTS_STEPS,
                gradient_batch_size=GRADIENT_BATCH_SIZE,
                device=device,
            ),
            {
                "definition": "Integrated Gradients from the all-zero baseline",
                "coordinate_system": "training-standardized model input",
                "baseline": "all-zero vector",
                "quadrature": "1024 equal subintervals, right endpoints",
                "steps": INTEGRATED_GRADIENTS_STEPS,
                "gradient_batch_size": GRADIENT_BATCH_SIZE,
            },
            {},
        )
    if method == "GradientSHAP":
        candidates = np.asarray(experiment.fit_inputs, dtype=np.int64)
        if candidates.ndim != 1 or np.unique(candidates).size != candidates.size:
            raise ValueError("experiment.fit_inputs must be distinct 1-D indices")
        background_indices, alphas = expected_gradient_references(
            candidates,
            count=GRADIENT_SHAP_SAMPLES,
            seed=GRADIENT_SHAP_BACKGROUND_SEED,
        )
        backgrounds = data.load_inputs(
            background_indices,
            standardizer=experiment.standardizer,
        )
        return (
            GradientShapExplainer(
                experiment.model,
                backgrounds=backgrounds,
                alphas=alphas,
                gradient_batch_size=GRADIENT_BATCH_SIZE,
                device=device,
            ),
            {
                "definition": "expected gradients with empirical training baselines",
                "reference_population": "experiment.fit_inputs only",
                "population_count": int(candidates.size),
                "population_indices_sha256": sha256_array(candidates),
                "sample_count": GRADIENT_SHAP_SAMPLES,
                "sampling": "uniform without replacement",
                "background_seed": GRADIENT_SHAP_BACKGROUND_SEED,
                "alpha_distribution": "independent Uniform(0,1)",
                "alpha_seed": GRADIENT_SHAP_ALPHA_SEED,
                "background_indices_sha256": sha256_array(background_indices),
                "alphas_sha256": sha256_array(alphas),
                "gradient_batch_size": GRADIENT_BATCH_SIZE,
            },
            {
                "gradientshap_background_indices": background_indices,
                "gradientshap_alphas": alphas,
            },
        )
    raise ValueError(f"unknown XAI method: {method!r}")


def build_direct_xai_target(
    method: str,
    *,
    data: ZCData,
    experiment: LoadedExperiment,
    event_standardized: np.ndarray,
    agop_factor: AgopFactor | None = None,
    device: str = "cpu",
) -> DirectXAITarget:
    """Build a unit spatial direction and orient its inequality toward the event.

    XAI signs are not comparable across the four methods.  We therefore retain
    the raw direction and multiply it by the sign of its projection on the
    standardized event.  The reported inequality then always points from the
    standardized origin toward the selected warm or cold event.  This policy is
    especially important for the cold GRAD direction, whose un-oriented local
    gradient naturally points toward increasing forecast values.
    """

    if method not in XAI_METHODS:
        raise ValueError(f"unknown XAI method: {method!r}")
    event = np.asarray(event_standardized, dtype=np.float64)
    if event.shape != data.input_shape or not np.isfinite(event).all():
        raise ValueError("event_standardized has the wrong shape or is nonfinite")
    if data.n_phase_features != 2:
        raise ValueError(
            "direct ZC controls require exactly two passive phase features"
        )
    explainer, parameters, references = _method_explainer(
        method,
        data=data,
        experiment=experiment,
        agop_factor=agop_factor,
        device=device,
    )
    explanation = np.asarray(explainer.explain(event[None])[0], dtype=np.float64)
    if explanation.shape != event.shape:
        raise ValueError("XAI explanation has an unexpected shape")
    fixed_phase = unit_fixed_phase_direction(explanation, phase_features=2)
    raw_spatial = np.asarray(fixed_phase[:-2], dtype=np.float64)
    raw_projection = float(raw_spatial @ event[:-2])
    if not np.isfinite(raw_projection) or abs(raw_projection) <= np.finfo(float).eps:
        raise ValueError("event has zero projection on its fixed-phase XAI direction")
    orientation = 1.0 if raw_projection > 0.0 else -1.0
    oriented = orientation * raw_spatial
    oriented_projection = orientation * raw_projection
    provenance = {
        "key": XAI_METHOD_KEYS[method],
        "label": method,
        "parameters": parameters,
        "explanation_sha256": sha256_array(explanation),
        "fixed_phase_explanation_sha256": sha256_array(fixed_phase),
        "raw_spatial_direction_sha256": sha256_array(raw_spatial),
        "oriented_spatial_direction_sha256": sha256_array(oriented),
        "event_standardized_sha256": sha256_array(event),
        "phase_coordinate_policy": (
            "drop the final sine/cosine coordinates, then renormalize"
        ),
        "orientation_policy": (
            "multiply the raw fixed-phase direction by sign(e_raw^T x_event), "
            "so the >= constraint points from the standardized origin toward "
            "the selected event"
        ),
        "orientation_multiplier": orientation,
        "raw_event_projection": raw_projection,
        "oriented_event_projection": oriented_projection,
    }
    return DirectXAITarget(
        method=method,
        explanation=explanation,
        fixed_phase_explanation=fixed_phase,
        raw_spatial_direction=raw_spatial,
        oriented_spatial_direction=oriented,
        raw_event_projection=raw_projection,
        oriented_event_projection=oriented_projection,
        orientation_multiplier=orientation,
        provenance=provenance,
        reference_arrays=references,
    )
