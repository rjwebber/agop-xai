"""Validated reuse of common-target Figure 5 CNNs for manuscript Figure 7."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import Standardizer, ZCData
from .figure5 import (
    FIGURE5_TRAINING_SCHEMA_VERSION,
    _load_cached_result,
    figure5_experiment_directory,
    select_figure5_inputs,
)
from .io import (
    load_json,
    sha256_array,
    sha256_file,
    sha256_json,
    write_json,
    write_npz,
)
from .models import build_model
from .training import ExperimentSpec, LoadedExperiment, TrainingConfig, resolve_device
from .xai import AgopFactor, build_exact_dense_agop_factor

FIGURE7_AGOP_CACHE_SCHEMA_VERSION = 1


def load_common_target_cnn(
    data: ZCData,
    artifact_root: Path,
    *,
    lead_months: int,
    seed: int,
    maximum_lead_months: int,
    device: str = "cpu",
) -> LoadedExperiment:
    """Load and fully validate a full-size Figure 5 common-target CNN cache."""

    spec = ExperimentSpec(
        architecture="cnn",
        lead_months=lead_months,
        train_years=10_000.0,
        seed=seed,
        common_period_max_lead_months=maximum_lead_months,
        input_profile="core4",
    )
    artifact_dir = figure5_experiment_directory(artifact_root, spec)
    completion_path = artifact_dir / "completed.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            "Missing full-size common-target Figure 5 CNN cache: "
            f"{completion_path}"
        )
    completion = load_json(completion_path)
    try:
        config = TrainingConfig(**completion["training_config"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Figure 5 completion record has an invalid configuration."
        ) from error
    cached = _load_cached_result(data, artifact_dir, spec, config)
    if cached is None:
        raise FileNotFoundError(f"Incomplete Figure 5 experiment: {artifact_dir}")

    checkpoint_path = artifact_dir / "checkpoint.pt"
    resolved = resolve_device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved,
        weights_only=True,
    )
    if (
        checkpoint.get("schema_version") != FIGURE5_TRAINING_SCHEMA_VERSION
        or checkpoint.get("spec") != asdict(spec)
        or checkpoint.get("training_config") != asdict(config)
        or checkpoint.get("data_metadata_sha256") != data.metadata_sha256
        or checkpoint.get("generation_id") != cached.metrics.get("generation_id")
        or checkpoint.get("lead_steps") != lead_months * data.steps_per_month
        or checkpoint.get("spatial_input_shape") != list(data.spatial_input_shape)
        or checkpoint.get("phase_features") != data.n_phase_features
    ):
        raise ValueError("Figure 5 checkpoint identity or model shape is inconsistent.")
    model = build_model(
        "cnn",
        data.spatial_input_shape,
        phase_features=data.n_phase_features,
    ).to(resolved)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with np.load(artifact_dir / "normalization.npz", allow_pickle=False) as archive:
        standardizer = Standardizer(
            mean=np.asarray(archive["mean"], dtype=np.float32),
            scale=np.asarray(archive["scale"], dtype=np.float32),
            scale_floor=float(archive["scale_floor"]),
            count=int(archive["count"]),
        )
    with np.load(artifact_dir / "indices.npz", allow_pickle=False) as archive:
        fit = np.asarray(archive["fit_inputs"], dtype=np.int64)
        standardization = np.asarray(
            archive["standardization_inputs"], dtype=np.int64
        )
        validation = np.asarray(archive["validation_inputs"], dtype=np.int64)
        test = np.asarray(archive["test_inputs"], dtype=np.int64)
    selected = select_figure5_inputs(data, spec)
    for label, actual, expected in (
        ("fit", fit, selected.fit_inputs),
        ("standardization", standardization, selected.standardization_inputs),
        ("validation", validation, selected.validation_inputs),
        ("test", test, selected.test_inputs),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"Figure 5 cache has inconsistent {label} indices.")
    if (
        standardizer.mean.shape != data.input_shape
        or standardizer.scale.shape != data.input_shape
        or standardizer.count != standardization.size
        or not np.isfinite(standardizer.mean).all()
        or not np.isfinite(standardizer.scale).all()
        or np.any(standardizer.scale <= 0.0)
    ):
        raise ValueError("Figure 5 normalization cache is invalid.")
    # These are the predictor populations associated with the common target-date
    # windows.  No validation or test state contributes to fit or normalization.
    development = np.concatenate((fit, validation))
    return LoadedExperiment(
        artifact_dir=artifact_dir,
        spec=spec,
        training_config=config,
        model=model,
        standardizer=standardizer,
        standardization_inputs=standardization,
        development_inputs=development,
        fit_inputs=fit,
        embargo_inputs=np.empty(0, dtype=np.int64),
        validation_inputs=validation,
        test_inputs=test,
        lead_steps=selected.lead_steps,
        metrics=cached.metrics,
        checkpoint_sha256=cached.checkpoint_sha256,
    )


def _factor_identity(
    data: ZCData,
    experiment: LoadedExperiment,
    *,
    gradient_batch_size: int,
    device: str,
) -> dict[str, Any]:
    return {
        "purpose": "exact full-fit AGOP factor for a common-target CNN",
        "data_metadata_sha256": data.metadata_sha256,
        "input_profile": data.input_profile,
        "checkpoint_sha256": experiment.checkpoint_sha256,
        "spec": asdict(experiment.spec),
        "fit_indices_sha256": sha256_array(experiment.fit_inputs),
        "fit_count": experiment.fit_inputs.size,
        "normalization_mean_sha256": sha256_array(experiment.standardizer.mean),
        "normalization_scale_sha256": sha256_array(experiment.standardizer.scale),
        "gradient_batch_size": gradient_batch_size,
        "gradient_device": str(device),
        "feature_count": data.n_features,
        "method": "exact dense float64 matrix and complete float64 eigendecomposition",
    }


def load_or_build_exact_factor(
    data: ZCData,
    experiment: LoadedExperiment,
    cache_dir: Path,
    *,
    gradient_batch_size: int,
    device: str,
    overwrite: bool = False,
    retain_matrix_cache: bool = True,
) -> tuple[AgopFactor, dict[str, Any]]:
    """Load or atomically construct a full exact AGOP for a trained CNN."""

    directory = cache_dir.expanduser().resolve()
    factor_path = directory / "full_eigensystem.npz"
    manifest_path = directory / "full_eigensystem.npz.json"
    matrix_path = (
        directory / "dense_agop_matrix.npy" if retain_matrix_cache else None
    )
    identity = _factor_identity(
        data,
        experiment,
        gradient_batch_size=gradient_batch_size,
        device=device,
    )
    identity_sha256 = sha256_json(identity)
    if factor_path.is_file() and not overwrite:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"AGOP cache manifest is missing: {manifest_path}")
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema_version") != FIGURE7_AGOP_CACHE_SCHEMA_VERSION
            or manifest.get("identity") != identity
            or manifest.get("identity_sha256") != identity_sha256
            or manifest.get("file_sha256") != sha256_file(factor_path)
        ):
            raise ValueError("Exact AGOP cache identity or hash mismatch.")
        with np.load(factor_path, allow_pickle=False) as archive:
            basis = np.asarray(archive["basis"], dtype=np.float64)
            roots = np.asarray(archive["root_eigenvalues"], dtype=np.float64)
            references = np.asarray(archive["reference_indices"], dtype=np.int64)
        if (
            basis.shape != (data.n_features, data.n_features)
            or roots.shape != (data.n_features,)
            or not np.array_equal(references, experiment.fit_inputs)
            or not np.isfinite(basis).all()
            or not np.isfinite(roots).all()
            or np.any(roots < 0.0)
        ):
            raise ValueError("Exact AGOP factor payload is invalid.")
        factor = AgopFactor(
            basis=basis,
            root_eigenvalues=roots,
            center=np.zeros(data.n_features, dtype=np.float64),
            reference_indices=references,
            approximation_rank=data.n_features,
            solver_metadata=manifest.get("solver_metadata"),
        )
        return factor, {
            "cache_hit": True,
            "factor_file": factor_path.name,
            "factor_sha256": manifest["file_sha256"],
            "identity_sha256": identity_sha256,
            "reference_count": references.size,
            "rank": data.n_features,
        }

    directory.mkdir(parents=True, exist_ok=True)
    factor = build_exact_dense_agop_factor(
        experiment.model,
        data,
        experiment.standardizer,
        experiment.fit_inputs,
        gradient_batch_size=gradient_batch_size,
        device=device,
        matrix_cache_path=matrix_path,
        overwrite_matrix_cache=overwrite,
        cache_identity=identity,
    )
    write_npz(
        factor_path,
        overwrite=True,
        compressed=False,
        basis=np.asarray(factor.basis, dtype=np.float64),
        root_eigenvalues=np.asarray(factor.root_eigenvalues, dtype=np.float64),
        reference_indices=np.asarray(factor.reference_indices, dtype=np.int64),
    )
    digest = sha256_file(factor_path)
    manifest = {
        "schema_version": FIGURE7_AGOP_CACHE_SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "file": factor_path.name,
        "file_sha256": digest,
        "solver_metadata": factor.solver_metadata,
    }
    if matrix_path is not None:
        manifest.update(
            {
                "matrix_file": matrix_path.name,
                "matrix_sha256": sha256_file(matrix_path),
            }
        )
    write_json(manifest_path, manifest, overwrite=True)
    return factor, {
        "cache_hit": False,
        "factor_file": factor_path.name,
        "factor_sha256": digest,
        "identity_sha256": identity_sha256,
        "reference_count": factor.reference_indices.size,
        "rank": factor.approximation_rank,
    }
