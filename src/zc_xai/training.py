"""Deterministic, cached PyTorch training for Zebiak-Cane forecasts."""

from __future__ import annotations

import copy
import logging
import math
import os
import platform
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# CUDA needs this setting before creating a cuBLAS handle for deterministic
# matrix multiplication. Respect either valid value if the caller set one.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - dependency guidance
    raise ModuleNotFoundError(
        "PyTorch is required for forecasting experiments. Install the project "
        "with `python -m pip install -e .`."
    ) from error

from .data import (
    FRESH_SCHEMA_VERSION,
    DevelopmentSelection,
    Standardizer,
    SupervisedSplit,
    ZCData,
)
from .io import atomic_output_path, load_json, sha256_file, write_json, write_npz
from .models import ARCHITECTURES, ArchitectureName, build_model, parameter_count

LOGGER = logging.getLogger(__name__)
TRAINING_SCHEMA_VERSION = 6
TRAINING_SOURCE_FILES = ("data.py", "models.py", "training.py")
STANDARDIZATION_POPULATION = "optimization fit selection only"
FIXED_STANDARDIZATION_POPULATION = "all states in fixed 10000-year training block"
VALIDATION_STRATEGY = "chronological tail with lead-time embargo"
FIXED_VALIDATION_STRATEGY = "fixed 10000/1000/1000-year chronological blocks"


@dataclass(frozen=True)
class ExperimentSpec:
    architecture: ArchitectureName
    lead_months: int
    train_years: float
    seed: int = 42
    train_fraction: float = 0.9
    validation_fraction: float = 0.2
    common_period_max_lead_months: int | None = None
    input_profile: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    maximum_epochs: int = 100
    patience: int | None = None
    minimum_improvement: float = 1.0e-4
    learning_rate: float | None = 1.0e-3
    weight_decay: float = 1.0e-4
    statistics_batch_size: int = 1024
    deterministic: bool = True


@dataclass
class LoadedExperiment:
    artifact_dir: Path
    spec: ExperimentSpec
    training_config: TrainingConfig
    model: nn.Module
    standardizer: Standardizer
    standardization_inputs: np.ndarray
    development_inputs: np.ndarray
    fit_inputs: np.ndarray
    embargo_inputs: np.ndarray
    validation_inputs: np.ndarray
    test_inputs: np.ndarray
    lead_steps: int
    metrics: dict[str, Any]
    checkpoint_sha256: str


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto`, `cpu`, `mps`, or `cuda` with clear availability errors."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _year_label(years: float) -> str:
    if float(years).is_integer():
        return str(int(years))
    return f"{years:.6g}".replace(".", "p")


def experiment_directory(root: Path, spec: ExperimentSpec) -> Path:
    directory = root.expanduser().resolve() / "models"
    if spec.input_profile is not None:
        directory /= spec.input_profile
    directory = (
        directory
        / spec.architecture
        / f"lead-{spec.lead_months:02d}m"
        / f"years-{_year_label(spec.train_years)}"
    )
    if spec.common_period_max_lead_months is not None:
        directory /= f"common-max-lead-{spec.common_period_max_lead_months:02d}m"
    return directory / f"seed-{spec.seed:06d}"


def _learning_rate(spec: ExperimentSpec, config: TrainingConfig) -> float:
    if config.learning_rate is not None:
        return config.learning_rate
    # A shared default is part of the compact-model channel-sensitivity test.
    # The validation curves will tell us if the normalized ViT still needs a
    # smaller rate; do not special-case it before running that experiment.
    return 1.0e-3


def resolved_patience(spec: ExperimentSpec, config: TrainingConfig) -> int:
    """Use one simple default unless the caller explicitly overrides it."""

    if config.patience is not None:
        return config.patience
    return 10


def _training_source_sha256() -> dict[str, str]:
    source_directory = Path(__file__).resolve().parent
    return {
        name: sha256_file(source_directory / name) for name in TRAINING_SOURCE_FILES
    }


def _validate_configuration(spec: ExperimentSpec, config: TrainingConfig) -> None:
    if spec.architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {spec.architecture!r}")
    if spec.lead_months <= 0:
        raise ValueError("lead_months must be positive")
    if spec.seed < 0:
        raise ValueError("seed must be nonnegative")
    if (
        spec.input_profile is not None
        and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", spec.input_profile) is None
    ):
        raise ValueError(
            "input_profile must contain only lowercase letters, digits, '_' or '-'."
        )
    if spec.train_years <= 0.0 or not math.isfinite(spec.train_years):
        raise ValueError("train_years must be finite and positive")
    if config.batch_size <= 0 or config.maximum_epochs <= 0:
        raise ValueError("batch_size and maximum_epochs must be positive")
    if config.patience is not None and config.patience <= 0:
        raise ValueError("patience must be positive when specified")
    if config.statistics_batch_size <= 0:
        raise ValueError("statistics_batch_size must be positive")
    if not 0.0 < spec.train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    if not 0.0 < spec.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if (
        spec.common_period_max_lead_months is not None
        and spec.common_period_max_lead_months < spec.lead_months
    ):
        raise ValueError(
            "common_period_max_lead_months cannot be shorter than lead_months"
        )
    if (
        not math.isfinite(config.minimum_improvement)
        or config.minimum_improvement < 0.0
    ):
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config.learning_rate is not None and (
        not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0
    ):
        raise ValueError("learning_rate must be finite and positive")


def _validate_fresh_experiment_spec(data: ZCData, spec: ExperimentSpec) -> None:
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        return
    if spec.train_years != 10_000.0:
        raise ValueError(
            "Fresh ZC experiments use the fixed 10,000-year training block; "
            "train_years must be 10000."
        )
    if spec.train_fraction != 0.9:
        raise ValueError(
            "Fresh ZC experiments use fixed blocks; train_fraction must remain "
            "at its unused compatibility default 0.9."
        )
    if spec.validation_fraction != 0.2:
        raise ValueError(
            "Fresh ZC experiments use the fixed 1,000-year validation block; "
            "validation_fraction must remain at its unused compatibility default 0.2."
        )
    if spec.common_period_max_lead_months is not None:
        raise ValueError(
            "Fresh fixed-block experiments do not support "
            "common_period_max_lead_months."
        )


def _ordered_training_batches(
    indices: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Shuffle example membership, sorting only within each I/O batch."""

    shuffled = rng.permutation(np.asarray(indices, dtype=np.int64))
    return [
        np.sort(shuffled[start : start + batch_size])
        for start in range(0, shuffled.size, batch_size)
    ]


def _tensor_batch(
    data: ZCData,
    indices: np.ndarray,
    standardizer: Standardizer,
    *,
    lead_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = data.load_inputs(indices, standardizer=standardizer)
    targets = data.load_targets(indices, lead_steps=lead_steps)
    return (
        torch.from_numpy(features).to(device=device),
        torch.from_numpy(targets).to(device=device),
    )


def mean_squared_error(
    model: nn.Module,
    data: ZCData,
    indices: np.ndarray,
    standardizer: Standardizer,
    *,
    lead_steps: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_squared_error = 0.0
    total_count = 0
    with torch.inference_mode():
        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            features, targets = _tensor_batch(
                data,
                batch_indices,
                standardizer,
                lead_steps=lead_steps,
                device=device,
            )
            residual = model(features) - targets
            total_squared_error += float(torch.square(residual).sum().cpu())
            total_count += targets.numel()
    if total_count == 0:
        raise ValueError("Cannot evaluate an empty index set.")
    return total_squared_error / total_count


def predict(
    experiment: LoadedExperiment,
    data: ZCData,
    input_indices: np.ndarray,
    *,
    batch_size: int,
    device: str = "auto",
) -> np.ndarray:
    resolved_device = resolve_device(device)
    model = experiment.model.to(resolved_device)
    model.eval()
    indices = np.asarray(input_indices, dtype=np.int64)
    output = np.empty(indices.size, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, indices.size, batch_size):
            stop = min(start + batch_size, indices.size)
            values = data.load_inputs(
                indices[start:stop],
                standardizer=experiment.standardizer,
            )
            predictions = model(torch.from_numpy(values).to(resolved_device))
            output[start:stop] = predictions.detach().cpu().numpy()
    experiment.model = model.cpu()
    return output


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.size < 2:
        raise ValueError("R2 inputs must have the same shape and at least two values.")
    residual_sum = float(np.square(targets - predictions).sum())
    total_sum = float(np.square(targets - targets.mean()).sum())
    if total_sum == 0.0:
        raise ValueError("R2 is undefined for a constant target.")
    return 1.0 - residual_sum / total_sum


def _save_torch_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    with atomic_output_path(path, overwrite=True) as temporary_path:
        torch.save(payload, temporary_path)


def _comparison_inputs(data: ZCData, spec: ExperimentSpec):
    """Return the lead split plus fair development/test input periods."""

    split = data.supervised_split(spec.lead_months, train_fraction=spec.train_fraction)
    maximum_lead = spec.common_period_max_lead_months
    if maximum_lead is None:
        return split, split.train_inputs, split.test_inputs

    common = data.supervised_split(maximum_lead, train_fraction=spec.train_fraction)
    common_development_targets = np.arange(
        common.lead_steps,
        common.split_step,
        dtype=np.int64,
    )
    common_test_targets = np.arange(
        common.split_step + common.lead_steps,
        data.n_time_steps,
        dtype=np.int64,
    )
    development_inputs = common_development_targets - split.lead_steps
    test_inputs = common_test_targets - split.lead_steps
    return split, development_inputs, test_inputs


def train_experiment(
    data: ZCData,
    artifact_root: Path,
    spec: ExperimentSpec,
    config: TrainingConfig,
    *,
    device: str = "auto",
    overwrite: bool = False,
) -> LoadedExperiment:
    """Train one cached experiment or load an exactly matching completed run."""

    _validate_configuration(spec, config)
    _validate_fresh_experiment_spec(data, spec)
    data_profile = getattr(data, "input_profile", None)
    profile_matches_legacy = (
        int(getattr(data, "n_phase_features", 0)) == 0 and spec.input_profile is None
    )
    if spec.input_profile != data_profile and not profile_matches_legacy:
        raise ValueError(
            "ExperimentSpec.input_profile must match the processed-data view: "
            f"{spec.input_profile!r} versus {data_profile!r}."
        )
    artifact_dir = experiment_directory(artifact_root, spec)
    checkpoint_path = artifact_dir / "checkpoint.pt"
    if checkpoint_path.is_file() and not overwrite:
        try:
            existing = load_experiment(data, artifact_root, spec, device="cpu")
        except FileNotFoundError:
            LOGGER.warning("Retraining incomplete cached experiment: %s", artifact_dir)
        else:
            if existing.training_config != config:
                raise ValueError(
                    f"Cached experiment {artifact_dir} used a different training "
                    "configuration. Use --overwrite to retrain it explicitly."
                )
            return existing

    resolved_device = resolve_device(device)
    seed_everything(spec.seed, deterministic=config.deterministic)
    if data.metadata.get("schema_version") == FRESH_SCHEMA_VERSION:
        fixed = data.fixed_supervised_split(spec.lead_months)
        split = SupervisedSplit(
            train_inputs=fixed.train_inputs,
            test_inputs=fixed.test_inputs,
            split_step=fixed.test_block[0],
            lead_steps=fixed.lead_steps,
        )
        embargo_inputs = np.arange(
            fixed.train_inputs[-1] + 1,
            fixed.validation_inputs[0],
            dtype=np.int64,
        )
        selection = DevelopmentSelection(
            development_inputs=np.concatenate(
                (fixed.train_inputs, embargo_inputs, fixed.validation_inputs)
            ),
            fit_inputs=fixed.train_inputs,
            embargo_inputs=embargo_inputs,
            validation_inputs=fixed.validation_inputs,
            validation_embargo_steps=fixed.lead_steps,
            requested_years=10_000.0,
            actual_years=fixed.train_inputs.size / data.steps_per_year,
            window_start_step=fixed.train_block[0],
            window_stop_step_exclusive=fixed.validation_block[1],
        )
        test_inputs = fixed.test_inputs
        validation_strategy = FIXED_VALIDATION_STRATEGY
        fixed_blocks = {
            "train": list(fixed.train_block),
            "validation": list(fixed.validation_block),
            "test": list(fixed.test_block),
        }
        standardization_inputs = np.arange(
            fixed.train_block[0], fixed.train_block[1], dtype=np.int64
        )
        standardization_population = FIXED_STANDARDIZATION_POPULATION
    else:
        split, available_development_inputs, test_inputs = _comparison_inputs(
            data, spec
        )
        validation_embargo_steps = (
            spec.common_period_max_lead_months or spec.lead_months
        ) * data.steps_per_month
        selection = data.development_selection(
            available_development_inputs,
            train_years=spec.train_years,
            validation_fraction=spec.validation_fraction,
            validation_embargo_steps=validation_embargo_steps,
            seed=spec.seed,
        )
        validation_strategy = VALIDATION_STRATEGY
        fixed_blocks = None
        standardization_inputs = selection.fit_inputs
        standardization_population = STANDARDIZATION_POPULATION
    LOGGER.info(
        "Training %s: lead=%s months, requested years=%s, actual years=%.3f, "
        "fit=%s, embargo=%s, validation=%s, device=%s",
        spec.architecture.upper(),
        spec.lead_months,
        spec.train_years,
        selection.actual_years,
        f"{selection.fit_inputs.size:,}",
        f"{selection.embargo_inputs.size:,}",
        f"{selection.validation_inputs.size:,}",
        resolved_device,
    )
    standardizer = data.compute_standardizer(
        standardization_inputs,
        batch_size=config.statistics_batch_size,
    )
    phase_features = int(getattr(data, "n_phase_features", 0))
    spatial_input_shape = tuple(getattr(data, "spatial_input_shape", data.input_shape))
    model = build_model(
        spec.architecture,
        spatial_input_shape,
        phase_features=phase_features,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_learning_rate(spec, config),
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    rng = np.random.default_rng(spec.seed + 1)
    best_validation_loss = float("inf")
    patience_reference_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    patience = resolved_patience(spec, config)
    history: list[dict[str, float | int]] = []
    training_started = time.monotonic()

    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        training_squared_error = 0.0
        training_count = 0
        for batch_indices in _ordered_training_batches(
            selection.fit_inputs,
            batch_size=config.batch_size,
            rng=rng,
        ):
            features, targets = _tensor_batch(
                data,
                batch_indices,
                standardizer,
                lead_steps=split.lead_steps,
                device=resolved_device,
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            training_squared_error += float(loss.detach().cpu()) * targets.numel()
            training_count += targets.numel()

        training_loss = training_squared_error / training_count
        validation_loss = mean_squared_error(
            model,
            data,
            selection.validation_inputs,
            standardizer,
            lead_steps=split.lead_steps,
            batch_size=config.batch_size,
            device=resolved_device,
        )
        history.append(
            {
                "epoch": epoch,
                "training_mse": training_loss,
                "validation_mse": validation_loss,
            }
        )
        LOGGER.info(
            "Epoch %s: training MSE %.6g; validation MSE %.6g",
            epoch,
            training_loss,
            validation_loss,
        )

        # Always retain the true lowest-validation checkpoint.  The minimum
        # improvement threshold controls only whether the patience clock is
        # reset; using it as the checkpoint threshold can save an older,
        # slightly worse model.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_state = {name: tensor.cpu() for name, tensor in best_state.items()}

        if patience_reference_loss - validation_loss >= config.minimum_improvement:
            patience_reference_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                LOGGER.info("Early stopping after epoch %s", epoch)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(resolved_device)
    test_predictions = _predict_model(
        model,
        data,
        test_inputs,
        standardizer,
        batch_size=config.batch_size,
        device=resolved_device,
    )
    test_targets = data.load_targets(test_inputs, lead_steps=split.lead_steps)
    test_r2 = r2_score(test_targets, test_predictions)
    elapsed_seconds = time.monotonic() - training_started
    LOGGER.info("Best epoch %s; held-out test R2 %.6f", best_epoch, test_r2)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_id = str(uuid.uuid4())
    source_sha256 = _training_source_sha256()
    normalization_path = artifact_dir / "normalization.npz"
    indices_path = artifact_dir / "indices.npz"
    metrics_path = artifact_dir / "metrics.json"
    completion_path = artifact_dir / "completed.json"
    write_npz(
        normalization_path,
        overwrite=True,
        mean=standardizer.mean,
        scale=standardizer.scale,
        scale_floor=np.asarray(standardizer.scale_floor),
        count=np.asarray(standardizer.count),
    )
    write_npz(
        indices_path,
        overwrite=True,
        development_inputs=selection.development_inputs,
        standardization_inputs=standardization_inputs,
        fit_inputs=selection.fit_inputs,
        embargo_inputs=selection.embargo_inputs,
        validation_inputs=selection.validation_inputs,
        test_inputs=test_inputs,
    )
    checkpoint_payload = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "state_dict": best_state,
        "architecture": spec.architecture,
        "input_shape": list(data.input_shape),
        "spatial_input_shape": list(spatial_input_shape),
        "phase_features": phase_features,
        "spec": asdict(spec),
        "training_config": asdict(config),
        "resolved_patience": patience,
        "standardization_population": standardization_population,
        "validation_strategy": validation_strategy,
        "lead_steps": split.lead_steps,
        "resolved_learning_rate": _learning_rate(spec, config),
        "best_epoch": best_epoch,
        "data_metadata_sha256": data.metadata_sha256,
        "generation_id": generation_id,
        "training_source_sha256": source_sha256,
    }
    _save_torch_checkpoint(checkpoint_path, checkpoint_payload)
    checkpoint_digest = sha256_file(checkpoint_path)
    metrics = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "spec": asdict(spec),
        "training_config": asdict(config),
        "resolved_patience": patience,
        "standardization_population": standardization_population,
        "validation_strategy": validation_strategy,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": _learning_rate(spec, config),
            "weight_decay": config.weight_decay,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "data": data.provenance(),
        "model": {
            "parameter_count": parameter_count(model),
            "checkpoint_sha256": checkpoint_digest,
        },
        "selection": _selection_metadata(
            selection,
            steps_per_year=data.steps_per_year,
            validation_strategy=validation_strategy,
            fixed_blocks=fixed_blocks,
        ),
        "lead_steps": split.lead_steps,
        "split_step": split.split_step,
        "evaluation": {
            "common_period_max_lead_months": spec.common_period_max_lead_months,
            "test_input_start_step": int(test_inputs[0]),
            "test_input_stop_step_exclusive": int(test_inputs[-1]) + 1,
            "test_target_start_step": int(test_inputs[0] + split.lead_steps),
            "test_target_stop_step_exclusive": int(
                test_inputs[-1] + split.lead_steps + 1
            ),
            "test_count": int(test_inputs.size),
        },
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_loss,
        "test_r2": test_r2,
        "elapsed_seconds": elapsed_seconds,
        "device_used": str(resolved_device),
        "determinism": {
            "enabled": config.deterministic,
            "torch_deterministic_algorithms_enforced": config.deterministic,
            "unsupported_operation_policy": (
                "raise an error" if config.deterministic else "not applicable"
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_allowed": False,
        },
        "generation_id": generation_id,
        "training_source_sha256": source_sha256,
        "history": history,
    }
    write_json(metrics_path, metrics, overwrite=True)
    completion = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "generation_id": generation_id,
        "spec": asdict(spec),
        "training_config": asdict(config),
        "resolved_patience": patience,
        "data_metadata_sha256": data.metadata_sha256,
        "training_source_sha256": source_sha256,
        "files": {
            "checkpoint.pt": sha256_file(checkpoint_path),
            "normalization.npz": sha256_file(normalization_path),
            "indices.npz": sha256_file(indices_path),
            "metrics.json": sha256_file(metrics_path),
        },
    }
    # This manifest is the commit marker for one internally consistent generation.
    write_json(completion_path, completion, overwrite=True)
    model.cpu()
    return load_experiment(data, artifact_root, spec, device="cpu")


def _selection_metadata(
    selection: DevelopmentSelection,
    *,
    steps_per_year: int,
    validation_strategy: str,
    fixed_blocks: dict[str, list[int]] | None,
) -> dict[str, Any]:
    metadata = {
        "requested_years": selection.requested_years,
        "actual_years": selection.actual_years,
        "fit_years": selection.fit_inputs.size / steps_per_year,
        "embargo_years": selection.embargo_inputs.size / steps_per_year,
        "validation_years": selection.validation_inputs.size / steps_per_year,
        "window_start_step": selection.window_start_step,
        "window_stop_step_exclusive": selection.window_stop_step_exclusive,
        "development_count": int(selection.development_inputs.size),
        "fit_count": int(selection.fit_inputs.size),
        "embargo_count": int(selection.embargo_inputs.size),
        "validation_count": int(selection.validation_inputs.size),
        "validation_embargo_steps": selection.validation_embargo_steps,
        "validation_strategy": validation_strategy,
    }
    if fixed_blocks is not None:
        metadata["fixed_blocks"] = fixed_blocks
    return metadata


def _predict_model(
    model: nn.Module,
    data: ZCData,
    indices: np.ndarray,
    standardizer: Standardizer,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = np.empty(indices.size, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, indices.size, batch_size):
            stop = min(start + batch_size, indices.size)
            features = data.load_inputs(
                indices[start:stop],
                standardizer=standardizer,
            )
            prediction = model(torch.from_numpy(features).to(device))
            output[start:stop] = prediction.detach().cpu().numpy()
    return output


def load_experiment(
    data: ZCData,
    artifact_root: Path,
    spec: ExperimentSpec,
    *,
    device: str = "cpu",
) -> LoadedExperiment:
    _validate_fresh_experiment_spec(data, spec)
    artifact_dir = experiment_directory(artifact_root, spec)
    checkpoint_path = artifact_dir / "checkpoint.pt"
    normalization_path = artifact_dir / "normalization.npz"
    indices_path = artifact_dir / "indices.npz"
    metrics_path = artifact_dir / "metrics.json"
    completion_path = artifact_dir / "completed.json"
    required = (
        checkpoint_path,
        normalization_path,
        indices_path,
        metrics_path,
        completion_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The requested trained experiment is incomplete. Missing:\n  "
            + "\n  ".join(missing)
        )

    completion = load_json(completion_path)
    if completion.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError(
            f"Completed artifact {artifact_dir} uses an older training schema. "
            "Retrain it explicitly with --overwrite."
        )
    if completion.get("spec") != asdict(spec):
        raise ValueError(f"Completion manifest does not match {spec}.")
    if completion.get("data_metadata_sha256") != data.metadata_sha256:
        raise ValueError("Cached experiment uses different processed-data metadata.")
    current_source_sha256 = _training_source_sha256()
    if completion.get("training_source_sha256") != current_source_sha256:
        raise ValueError(
            "Cached experiment was produced by different training/model source code. "
            "Retrain it explicitly with --overwrite."
        )
    expected_files = completion.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError(f"Invalid completion manifest: {completion_path}")
    artifact_paths = {
        "checkpoint.pt": checkpoint_path,
        "normalization.npz": normalization_path,
        "indices.npz": indices_path,
        "metrics.json": metrics_path,
    }
    for name, path in artifact_paths.items():
        if expected_files.get(name) != sha256_file(path):
            raise ValueError(
                f"Cached experiment has an incomplete or mismatched {name}: "
                f"{artifact_dir}. Retrain with --overwrite."
            )

    resolved_device = resolve_device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=True,
    )
    if checkpoint.get("training_schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses an older training schema. "
            "Retrain it explicitly with --overwrite."
        )
    if checkpoint["spec"] != asdict(spec):
        raise ValueError(f"Checkpoint specification does not match {spec}.")
    if checkpoint["data_metadata_sha256"] != data.metadata_sha256:
        raise ValueError(
            "Checkpoint was trained from different processed-data metadata."
        )
    if checkpoint.get("generation_id") != completion.get("generation_id"):
        raise ValueError("Checkpoint and completion manifest generations differ.")
    if checkpoint.get("training_source_sha256") != current_source_sha256:
        raise ValueError("Checkpoint source fingerprint does not match this code.")
    is_fresh = data.metadata.get("schema_version") == FRESH_SCHEMA_VERSION
    expected_standardization_population = (
        FIXED_STANDARDIZATION_POPULATION if is_fresh else STANDARDIZATION_POPULATION
    )
    if (
        checkpoint.get("standardization_population")
        != expected_standardization_population
    ):
        raise ValueError("Checkpoint uses an incompatible normalization population.")
    expected_validation_strategy = (
        FIXED_VALIDATION_STRATEGY if is_fresh else VALIDATION_STRATEGY
    )
    if checkpoint.get("validation_strategy") != expected_validation_strategy:
        raise ValueError("Checkpoint uses an incompatible validation strategy.")
    phase_features = int(checkpoint.get("phase_features", 0))
    if phase_features != int(getattr(data, "n_phase_features", 0)):
        raise ValueError("Checkpoint and processed data use different phase features.")
    spatial_input_shape = tuple(getattr(data, "spatial_input_shape", data.input_shape))
    if list(spatial_input_shape) != checkpoint.get("spatial_input_shape"):
        raise ValueError("Checkpoint and processed data use different spatial shapes.")
    model = build_model(
        spec.architecture,
        spatial_input_shape,
        phase_features=phase_features,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with np.load(normalization_path, allow_pickle=False) as archive:
        standardizer = Standardizer(
            mean=np.asarray(archive["mean"], dtype=np.float32),
            scale=np.asarray(archive["scale"], dtype=np.float32),
            scale_floor=float(archive["scale_floor"]),
            count=int(archive["count"]),
        )
    if standardizer.mean.shape != data.input_shape:
        raise ValueError("Cached input mean has the wrong shape.")
    if standardizer.scale.shape != data.input_shape:
        raise ValueError("Cached input scale has the wrong shape.")
    if not np.isfinite(standardizer.mean).all():
        raise ValueError("Cached input mean contains non-finite values.")
    if not np.isfinite(standardizer.scale).all() or np.any(standardizer.scale <= 0):
        raise ValueError("Cached input scale must be finite and strictly positive.")
    if not math.isfinite(standardizer.scale_floor) or standardizer.scale_floor <= 0:
        raise ValueError("Cached standardization floor is invalid.")
    with np.load(indices_path, allow_pickle=False) as archive:
        development_inputs = np.asarray(archive["development_inputs"], dtype=np.int64)
        standardization_inputs = np.asarray(
            archive["standardization_inputs"], dtype=np.int64
        )
        fit_inputs = np.asarray(archive["fit_inputs"], dtype=np.int64)
        embargo_inputs = np.asarray(archive["embargo_inputs"], dtype=np.int64)
        validation_inputs = np.asarray(archive["validation_inputs"], dtype=np.int64)
        test_inputs = np.asarray(archive["test_inputs"], dtype=np.int64)
    metrics = load_json(metrics_path)
    saved_config = TrainingConfig(**checkpoint["training_config"])
    if completion.get("training_config") != asdict(saved_config):
        raise ValueError("Training configuration differs across cached artifacts.")
    if metrics.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("Metrics use an incompatible schema.")
    if metrics.get("spec") != asdict(spec):
        raise ValueError("Metrics specification differs from the checkpoint.")
    if metrics.get("training_config") != asdict(saved_config):
        raise ValueError("Metrics training configuration differs from the checkpoint.")
    if metrics.get("standardization_population") != expected_standardization_population:
        raise ValueError("Metrics use an incompatible normalization population.")
    if metrics.get("validation_strategy") != expected_validation_strategy:
        raise ValueError("Metrics use an incompatible validation strategy.")
    if metrics.get("generation_id") != completion.get("generation_id"):
        raise ValueError("Metrics and completion manifest generations differ.")
    checkpoint_digest = sha256_file(checkpoint_path)
    if metrics.get("model", {}).get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("Metrics identify a different checkpoint.")

    if is_fresh:
        fixed = data.fixed_supervised_split(spec.lead_months)
        split = SupervisedSplit(
            train_inputs=fixed.train_inputs,
            test_inputs=fixed.test_inputs,
            split_step=fixed.test_block[0],
            lead_steps=fixed.lead_steps,
        )
        available_development_inputs = np.arange(
            fixed.train_block[0],
            fixed.validation_block[1] - fixed.lead_steps,
            dtype=np.int64,
        )
        expected_test_inputs = fixed.test_inputs
        expected_standardization_inputs = np.arange(
            fixed.train_block[0], fixed.train_block[1], dtype=np.int64
        )
    else:
        split, available_development_inputs, expected_test_inputs = _comparison_inputs(
            data, spec
        )
        expected_standardization_inputs = fit_inputs
    if int(checkpoint["lead_steps"]) != split.lead_steps:
        raise ValueError("Checkpoint lead does not match the processed-data cadence.")
    for label, indices in (
        ("development", development_inputs),
        ("fit", fit_inputs),
        ("embargo", embargo_inputs),
        ("validation", validation_inputs),
        ("test", test_inputs),
    ):
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"Cached {label} indices must be nonempty and 1-D.")
        if np.any(indices < 0) or np.any(
            indices + split.lead_steps >= data.n_time_steps
        ):
            raise ValueError(f"Cached {label} indices are outside the usable range.")
        if not np.array_equal(indices, np.unique(indices)):
            raise ValueError(f"Cached {label} indices must be sorted and unique.")
    if not np.array_equal(test_inputs, expected_test_inputs):
        raise ValueError("Cached held-out indices do not match the experiment spec.")
    if not np.array_equal(standardization_inputs, expected_standardization_inputs):
        raise ValueError(
            "Cached normalization indices do not match the experiment spec."
        )
    if not np.all(np.isin(development_inputs, available_development_inputs)):
        raise ValueError("Cached development indices are outside the allowed period.")
    if not np.array_equal(
        development_inputs,
        np.arange(development_inputs[0], development_inputs[-1] + 1),
    ):
        raise ValueError("Cached development window must be contiguous.")
    if (
        np.intersect1d(fit_inputs, embargo_inputs).size
        or np.intersect1d(fit_inputs, validation_inputs).size
        or np.intersect1d(embargo_inputs, validation_inputs).size
    ):
        raise ValueError("Optimization, embargo, and validation indices overlap.")
    if not np.array_equal(
        np.concatenate((fit_inputs, embargo_inputs, validation_inputs)),
        development_inputs,
    ):
        raise ValueError(
            "Optimization, embargo, and validation blocks do not partition "
            "development chronologically."
        )
    expected_embargo_steps = (
        spec.common_period_max_lead_months or spec.lead_months
    ) * data.steps_per_month
    if embargo_inputs.size != expected_embargo_steps:
        raise ValueError("Cached validation embargo does not match the forecast lead.")
    if standardizer.count != standardization_inputs.size:
        raise ValueError("Normalization count does not match its recorded population.")
    return LoadedExperiment(
        artifact_dir=artifact_dir,
        spec=spec,
        training_config=saved_config,
        model=model,
        standardizer=standardizer,
        standardization_inputs=standardization_inputs,
        development_inputs=development_inputs,
        fit_inputs=fit_inputs,
        embargo_inputs=embargo_inputs,
        validation_inputs=validation_inputs,
        test_inputs=test_inputs,
        lead_steps=int(checkpoint["lead_steps"]),
        metrics=metrics,
        checkpoint_sha256=checkpoint_digest,
    )
