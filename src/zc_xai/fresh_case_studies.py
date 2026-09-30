"""Production XAI case evaluation for the fresh four-ocean-field experiments.

This module deliberately does not support the historical processed-data schema.
It binds every case to the canonical 10,000/1,000/1,000-year blocks, evaluates
robustness over the complete nearest-one-percent training population, and treats
the two annual-phase coordinates as non-spatial. Phase remains part of the
forecast and of vector-based XAI scores, but is excluded from map rendering and
coherence.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .data import FRESH_SCHEMA_VERSION, ZCData
from .fresh_xai_outputs import FRESH_XAI_TRAINING_POPULATION
from .io import sha256_array
from .training import LoadedExperiment
from .xai import (
    AgopExplainer,
    AgopFactor,
    GradientExplainer,
    GradientShapExplainer,
    IntegratedGradientsExplainer,
    NeighborSample,
    attribution_score,
    coherence_score,
    input_gradients,
    robustness_score,
    sample_empirical_neighbors,
    sensitivity_score,
)

METHOD_ORDER = ("AGOP", "GradientSHAP", "IG", "GRAD")
METHOD_ARCHIVE_KEYS = {
    "AGOP": "agop",
    "GradientSHAP": "gradientshap",
    "IG": "ig",
    "GRAD": "grad",
}


@dataclass(frozen=True)
class FreshXAISettings:
    """Numerical choices fixed for the revised fresh-data manuscript."""

    neighbor_percent: float = 1.0
    distance_batch_size: int = 8192
    integrated_gradients_steps: int = 1024
    expected_gradients_samples: int = 1024
    expected_gradients_seed: int = 42
    gradient_batch_size: int = 1024

    def validate(self) -> None:
        if not 0.0 < self.neighbor_percent <= 100.0:
            raise ValueError("neighbor_percent must lie in (0, 100].")
        for name in (
            "distance_batch_size",
            "integrated_gradients_steps",
            "expected_gradients_samples",
            "gradient_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.expected_gradients_seed < 0:
            raise ValueError("expected_gradients_seed must be nonnegative.")


@dataclass(frozen=True)
class FreshEvent:
    case_id: str
    lead_months: int
    input_step: int
    target_step: int
    target_nino3_c: float
    selection_rule: str


@dataclass
class FreshCaseResult:
    rows: list[dict[str, Any]]
    arrays: dict[str, np.ndarray]
    neighbors: NeighborSample
    runtime: dict[str, Any]


def spatial_feature_count(data: ZCData) -> int:
    """Return the spatial prefix length, rejecting ambiguous input layouts."""

    expected = int(np.prod(data.spatial_input_shape))
    if expected + int(data.n_phase_features) != data.n_features:
        raise ValueError("Fresh input is not a spatial prefix plus phase scalars.")
    return expected


def spatial_part(data: ZCData, values: np.ndarray) -> np.ndarray:
    """Extract and reshape spatial coordinates, intentionally omitting phase."""

    flat = np.asarray(values).reshape(-1)
    count = spatial_feature_count(data)
    if flat.size != data.n_features:
        raise ValueError(
            f"Expected {data.n_features} input coordinates; found {flat.size}."
        )
    return flat[:count].reshape(data.spatial_input_shape)


def validate_primary_experiment(
    data: ZCData,
    experiment: LoadedExperiment,
    *,
    lead_months: int,
) -> None:
    """Bind a loaded checkpoint to the canonical fresh core4 fixed blocks."""

    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("Fresh case studies require the zc-v3 data schema.")
    if data.input_profile != "core4":
        raise ValueError("Fresh manuscript case studies require input_profile='core4'.")
    spec = experiment.spec
    if (
        spec.architecture != "cnn"
        or spec.input_profile != "core4"
        or spec.lead_months != lead_months
        or spec.train_years != 10_000.0
        or spec.common_period_max_lead_months is not None
    ):
        raise ValueError("Checkpoint is not the canonical fixed-block core4 CNN.")
    fixed = data.fixed_supervised_split(lead_months)
    expected_development = np.arange(
        fixed.train_block[0],
        fixed.validation_block[1] - fixed.lead_steps,
        dtype=np.int64,
    )
    expected_standardization = np.arange(
        fixed.train_block[0], fixed.train_block[1], dtype=np.int64
    )
    comparisons = (
        ("optimization", experiment.fit_inputs, fixed.train_inputs),
        ("validation", experiment.validation_inputs, fixed.validation_inputs),
        ("test", experiment.test_inputs, fixed.test_inputs),
        ("development", experiment.development_inputs, expected_development),
        (
            "standardization",
            experiment.standardization_inputs,
            expected_standardization,
        ),
    )
    for label, actual, expected in comparisons:
        if not np.array_equal(actual, expected):
            raise ValueError(f"Checkpoint does not use the fixed {label} population.")


def fixed_test_extreme(
    data: ZCData,
    *,
    lead_months: int,
    kind: str,
) -> FreshEvent:
    """Select the true extreme target from the lead-specific fixed test block."""

    if kind not in {"maximum", "minimum"}:
        raise ValueError("kind must be 'maximum' or 'minimum'.")
    fixed = data.fixed_supervised_split(lead_months)
    targets = data.load_targets(fixed.test_inputs, lead_steps=fixed.lead_steps)
    position = int(np.argmax(targets) if kind == "maximum" else np.argmin(targets))
    input_step = int(fixed.test_inputs[position])
    target_step = input_step + fixed.lead_steps
    phase = "el_nino" if kind == "maximum" else "la_nina"
    return FreshEvent(
        case_id=f"{phase}_{lead_months}m",
        lead_months=lead_months,
        input_step=input_step,
        target_step=target_step,
        target_nino3_c=float(targets[position]),
        selection_rule=(
            f"{kind} true Nino-3 target among predictors wholly contained in "
            "the fixed 1,000-year test block"
        ),
    )


def target_locked_event(
    data: ZCData,
    *,
    case_id: str,
    lead_months: int,
    target_step: int,
    selection_rule: str,
) -> FreshEvent:
    """Construct one lead-specific input for an explicitly shared test target."""

    fixed = data.fixed_supervised_split(lead_months)
    input_step = int(target_step) - fixed.lead_steps
    position = int(np.searchsorted(fixed.test_inputs, input_step))
    if (
        position >= fixed.test_inputs.size
        or int(fixed.test_inputs[position]) != input_step
    ):
        raise ValueError(
            f"Target step {target_step} is not eligible at lead {lead_months} months."
        )
    return FreshEvent(
        case_id=case_id,
        lead_months=lead_months,
        input_step=input_step,
        target_step=int(target_step),
        target_nino3_c=float(data.target[target_step]),
        selection_rule=selection_rule,
    )


def common_fixed_test_targets(data: ZCData, leads: list[int]) -> np.ndarray:
    """Return target dates eligible in the fixed test block at every lead."""

    if not leads or any(lead <= 0 for lead in leads):
        raise ValueError("At least one positive lead is required.")
    starts = []
    stops = []
    for lead in leads:
        fixed = data.fixed_supervised_split(lead)
        starts.append(fixed.test_block[0] + fixed.lead_steps)
        stops.append(fixed.test_block[1])
    start = max(starts)
    stop = min(stops)
    if stop <= start:
        raise ValueError("Requested leads have no common fixed-test target dates.")
    return np.arange(start, stop, dtype=np.int64)


def exhaustive_nearest_neighbors(
    data: ZCData,
    experiment: LoadedExperiment,
    *,
    query_index: int,
    settings: FreshXAISettings,
) -> NeighborSample:
    """Return every member of the nearest-X% empirical neighborhood."""

    settings.validate()
    training_inputs = np.asarray(experiment.fit_inputs, dtype=np.int64)
    sample = sample_empirical_neighbors(
        data,
        experiment.standardizer,
        query_index=query_index,
        candidate_indices=training_inputs,
        neighbor_percent=settings.neighbor_percent,
        # The helper clips to the neighborhood count. Passing the complete
        # candidate count therefore requests the population rather than a sample.
        n_samples=training_inputs.size,
        seed=0,
        distance_batch_size=settings.distance_batch_size,
    )
    # Store the deterministic distance/tie-broken neighborhood ordering instead
    # of the irrelevant random permutation produced when the entire set is drawn.
    return NeighborSample(
        query_index=sample.query_index,
        neighbor_percent=sample.neighbor_percent,
        candidate_count=sample.candidate_count,
        neighborhood_count=sample.neighborhood_count,
        sampled_indices=sample.neighborhood_indices.copy(),
        sampled_rms_distances=sample.neighborhood_rms_distances.copy(),
        neighborhood_indices=sample.neighborhood_indices.copy(),
        neighborhood_rms_distances=sample.neighborhood_rms_distances.copy(),
        seed=0,
    )


def expected_gradient_references(
    candidates: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose distinct empirical baselines and one independent alpha per row."""

    values = np.asarray(candidates, dtype=np.int64)
    if values.ndim != 1 or np.unique(values).size != values.size:
        raise ValueError("Expected-gradient candidates must be unique 1-D indices.")
    if count <= 0 or count > values.size:
        raise ValueError("Expected-gradient sample count is outside the population.")
    indices = np.asarray(
        np.random.default_rng(seed).choice(values, size=count, replace=False),
        dtype=np.int64,
    )
    alphas = np.random.default_rng(seed + 1).uniform(0.0, 1.0, size=count).astype(
        np.float32
    )
    return indices, alphas


def build_explainers(
    data: ZCData,
    experiment: LoadedExperiment,
    factor: AgopFactor,
    settings: FreshXAISettings,
    *,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the fixed production explainers and their reference archive."""

    settings.validate()
    if not np.array_equal(factor.reference_indices, experiment.fit_inputs):
        raise ValueError("AGOP must use every fixed-block training predictor.")
    if factor.approximation_rank != data.n_features:
        raise ValueError("AGOP must retain the complete exact eigenbasis.")
    indices, alphas = expected_gradient_references(
        experiment.fit_inputs,
        count=settings.expected_gradients_samples,
        seed=settings.expected_gradients_seed,
    )
    backgrounds = data.load_inputs(indices, standardizer=experiment.standardizer)
    explainers = {
        "AGOP": AgopExplainer(factor),
        "GradientSHAP": GradientShapExplainer(
            experiment.model,
            backgrounds=backgrounds,
            alphas=alphas,
            gradient_batch_size=settings.gradient_batch_size,
            device=device,
        ),
        "IG": IntegratedGradientsExplainer(
            experiment.model,
            n_steps=settings.integrated_gradients_steps,
            gradient_batch_size=settings.gradient_batch_size,
            device=device,
        ),
        "GRAD": GradientExplainer(
            experiment.model,
            batch_size=settings.gradient_batch_size,
            device=device,
        ),
    }
    references = {
        "expected_gradient_background_indices": indices,
        "expected_gradient_alphas": alphas,
        "agop_reference_indices": experiment.fit_inputs.astype(np.int64),
        "normalization_mean": experiment.standardizer.mean.astype(np.float32),
        "normalization_scale": experiment.standardizer.scale.astype(np.float32),
    }
    return explainers, references


def evaluate_fresh_case(
    data: ZCData,
    experiment: LoadedExperiment,
    explainers: dict[str, Any],
    event: FreshEvent,
    settings: FreshXAISettings,
    *,
    device: str,
    validate_primary: bool = True,
) -> FreshCaseResult:
    """Evaluate all methods and scores for one fixed-test event."""

    if validate_primary:
        validate_primary_experiment(data, experiment, lead_months=event.lead_months)
    elif (
        data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION
        or data.input_profile != "core4"
        or experiment.spec.architecture != "cnn"
        or experiment.spec.input_profile != "core4"
        or experiment.spec.lead_months != event.lead_months
        or experiment.spec.train_years != 10_000.0
    ):
        raise ValueError("Case does not use a fresh full-size core4 CNN.")
    if event.input_step not in experiment.test_inputs:
        raise ValueError("Event input is outside this checkpoint's fixed test pairs.")
    expected_target = event.input_step + experiment.lead_steps
    if expected_target != event.target_step:
        raise ValueError("Event input, lead, and target are inconsistent.")
    if not math.isclose(
        event.target_nino3_c,
        float(data.target[event.target_step]),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("Event target value disagrees with processed data.")

    timer = time.perf_counter()
    neighbors = exhaustive_nearest_neighbors(
        data,
        experiment,
        query_index=event.input_step,
        settings=settings,
    )
    neighbor_selection_seconds = time.perf_counter() - timer
    query = data.load_inputs(
        np.asarray([event.input_step], dtype=np.int64),
        standardizer=experiment.standardizer,
    )
    neighbor_inputs = data.load_inputs(
        neighbors.neighborhood_indices,
        standardizer=experiment.standardizer,
    )
    timer = time.perf_counter()
    gradient = input_gradients(
        experiment.model,
        query,
        batch_size=settings.gradient_batch_size,
        device=device,
    )[0]
    gradient_seconds = time.perf_counter() - timer

    arrays: dict[str, np.ndarray] = {
        "event_input_standardized": query[0].astype(np.float32),
        "event_input_physical": data.load_inputs(
            np.asarray([event.input_step], dtype=np.int64)
        )[0].astype(np.float32),
        "event_input_step": np.asarray(event.input_step, dtype=np.int64),
        "event_target_step": np.asarray(event.target_step, dtype=np.int64),
        "event_target_nino3_c": np.asarray(event.target_nino3_c, dtype=np.float32),
    }
    rows: list[dict[str, Any]] = []
    method_runtime: dict[str, Any] = {}
    spatial_count = spatial_feature_count(data)
    for method in METHOD_ORDER:
        explainer = explainers[method]
        timer = time.perf_counter()
        explanation = np.asarray(explainer.explain(query)[0], dtype=np.float64)
        target_seconds = time.perf_counter() - timer
        timer = time.perf_counter()
        neighbor_explanations = np.asarray(
            explainer.explain(neighbor_inputs), dtype=np.float64
        )
        neighbor_seconds = time.perf_counter() - timer
        robustness = robustness_score(
            explanation,
            neighbor_explanations,
            neighborhood_count=neighbors.neighborhood_count,
        )
        attribution, attribution_ratio, attribution_alpha = attribution_score(
            experiment.model,
            query[0],
            explanation,
            device=device,
        )
        spatial = spatial_part(data, explanation)
        rows.append(
            {
                "case_id": event.case_id,
                "method": method,
                "architecture": experiment.spec.architecture,
                "lead_months": event.lead_months,
                "sensitivity": sensitivity_score(gradient, explanation),
                "attribution": attribution,
                "attribution_ratio": attribution_ratio,
                "attribution_alpha": attribution_alpha,
                "robustness": robustness.score,
                "explanation_change": 1.0 - robustness.score,
                "robustness_population_sd": robustness.sample_standard_deviation,
                "robustness_finite_population_se": (
                    robustness.finite_population_standard_error
                ),
                "coherence": coherence_score(spatial, data.spatial_input_shape),
                "neighbor_percent": settings.neighbor_percent,
                "neighborhood_count": neighbors.neighborhood_count,
                "robustness_evaluated_count": neighbors.neighborhood_count,
                "robustness_exhaustive": True,
                "spatial_squared_mass_fraction": float(
                    np.square(explanation[:spatial_count]).sum()
                ),
                "phase_squared_mass_fraction": float(
                    np.square(explanation[spatial_count:]).sum()
                ),
                "event_input_step": event.input_step,
                "event_target_step": event.target_step,
                "event_target_nino3_c": event.target_nino3_c,
                "checkpoint_sha256": experiment.checkpoint_sha256,
            }
        )
        key = METHOD_ARCHIVE_KEYS[method]
        arrays[key] = explanation.astype(np.float32)
        arrays[f"{key}_robustness_overlaps"] = robustness.overlaps.astype(np.float32)
        method_runtime[method] = {
            "target_explanation_seconds": target_seconds,
            "neighbor_explanations_seconds": neighbor_seconds,
            "total_explanation_seconds": target_seconds + neighbor_seconds,
        }
    return FreshCaseResult(
        rows=rows,
        arrays=arrays,
        neighbors=neighbors,
        runtime={
            "neighbor_selection_seconds": neighbor_selection_seconds,
            "query_gradient_seconds": gradient_seconds,
            "methods": method_runtime,
        },
    )


def settings_metadata(settings: FreshXAISettings) -> dict[str, Any]:
    return {
        **asdict(settings),
        "robustness_definition": (
            "mean signed cosine similarity over every empirical state in the "
            "nearest one percent of lead-valid fixed training predictors"
        ),
        "robustness_candidate_population": FRESH_XAI_TRAINING_POPULATION,
        "neighbor_sampling": "none; exhaustive finite population",
        "expected_gradients_definition": (
            "1024 distinct empirical training-predictor baselines without "
            "replacement, with one independent Uniform(0,1) interpolation "
            "coefficient each"
        ),
        "expected_gradients_candidate_population": FRESH_XAI_TRAINING_POPULATION,
        "agop_definition": (
            "exact dense empirical gradient outer-product matrix over every fixed "
            "training predictor, with a complete float64 eigendecomposition"
        ),
        "coordinate_policy": {
            "vector_scores": "all standardized model-input coordinates",
            "spatial_plotting": "four spatial fields; phase scalars excluded",
            "coherence": "four spatial fields; phase scalars excluded",
        },
    }


def experiment_metadata(
    data: ZCData,
    experiment: LoadedExperiment,
    *,
    artifact_root: Any,
    agop_provenance: dict[str, Any],
) -> dict[str, Any]:
    root = artifact_root.expanduser().resolve()
    return {
        "spec": asdict(experiment.spec),
        "training_config": asdict(experiment.training_config),
        "artifact_id": experiment.artifact_dir.relative_to(root).as_posix(),
        "checkpoint_sha256": experiment.checkpoint_sha256,
        "generation_id": experiment.metrics.get("generation_id"),
        "test_r2": experiment.metrics["test_r2"],
        "fixed_blocks": experiment.metrics.get("fixed_blocks"),
        "fit_indices_sha256": sha256_array(experiment.fit_inputs),
        "validation_indices_sha256": sha256_array(experiment.validation_inputs),
        "test_indices_sha256": sha256_array(experiment.test_inputs),
        "standardization_indices_sha256": sha256_array(
            experiment.standardization_inputs
        ),
        "normalization_mean_sha256": sha256_array(experiment.standardizer.mean),
        "normalization_scale_sha256": sha256_array(experiment.standardizer.scale),
        "normalization_count": experiment.standardizer.count,
        "spatial_input_shape": list(data.spatial_input_shape),
        "phase_features": data.n_phase_features,
        "agop": agop_provenance,
    }
