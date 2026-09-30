"""Fresh-data training and cache contract for the Figure 5 model grid.

The main forecasting experiments use fixed 10,000/1,000/1,000-year
train/validation/test blocks.  Figure 5 additionally varies the amount of data
drawn from the fixed training block.  This module implements that one special
case without changing the primary-model training contract or invalidating its
cached checkpoints.
"""

from __future__ import annotations

import copy
import logging
import math
import platform
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import FRESH_SCHEMA_VERSION, ZCData
from .io import atomic_output_path, load_json, sha256_file, write_json, write_npz
from .models import build_model, parameter_count
from .training import (
    ExperimentSpec,
    TrainingConfig,
    _ordered_training_batches,
    _predict_model,
    _tensor_batch,
    mean_squared_error,
    r2_score,
    resolve_device,
    resolved_patience,
    seed_everything,
)

FIGURE5_TRAINING_SCHEMA_VERSION = 1
FIGURE5_SELECTION_STRATEGY = (
    "seeded contiguous predictor window inside the fixed 10000-year training block; "
    "fixed 1000-year validation and 1000-year test blocks remain untouched"
)
FIGURE5_SUBSET_NORMALIZATION_POPULATION = (
    "selected Figure 5 training predictors only; validation and test states excluded"
)
FIGURE5_FULL_NORMALIZATION_POPULATION = (
    "all raw states in the fixed 10000-year training block; validation and test "
    "states excluded"
)
FIGURE5_SOURCE_FILES = (
    "data.py",
    "figure5.py",
    "models.py",
    "training.py",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Figure5Selection:
    fit_inputs: np.ndarray
    standardization_inputs: np.ndarray
    validation_inputs: np.ndarray
    test_inputs: np.ndarray
    lead_steps: int
    common_maximum_lead_steps: int
    requested_years: float
    actual_years: float
    available_training_pair_years: float
    window_offset: int
    train_block: tuple[int, int]
    validation_block: tuple[int, int]
    test_block: tuple[int, int]


@dataclass(frozen=True)
class Figure5Result:
    artifact_dir: Path
    metrics: dict[str, Any]
    checkpoint_sha256: str
    cache_hit: bool


def _year_label(years: float) -> str:
    if float(years).is_integer():
        return str(int(years))
    return f"{years:.6g}".replace(".", "p")


def figure5_experiment_directory(root: Path, spec: ExperimentSpec) -> Path:
    if spec.input_profile is None:
        raise ValueError("Fresh Figure 5 experiments require an input profile.")
    directory = (
        root.expanduser().resolve()
        / "figure5"
        / spec.input_profile
        / spec.architecture
        / f"lead-{spec.lead_months:02d}m"
        / f"years-{_year_label(spec.train_years)}"
    )
    if spec.common_period_max_lead_months is not None:
        directory /= (
            f"common-max-lead-{spec.common_period_max_lead_months:02d}m"
        )
    return directory / f"seed-{spec.seed:06d}"


def _source_sha256() -> dict[str, str]:
    source_directory = Path(__file__).resolve().parent
    return {
        name: sha256_file(source_directory / name) for name in FIGURE5_SOURCE_FILES
    }


def _validate_spec(data: ZCData, spec: ExperimentSpec, config: TrainingConfig) -> None:
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("Figure 5 fresh training requires the zc-v3 data set.")
    if spec.input_profile != data.input_profile:
        raise ValueError(
            "Experiment input_profile must match the data view: "
            f"{spec.input_profile!r} versus {data.input_profile!r}."
        )
    if spec.lead_months <= 0:
        raise ValueError("lead_months must be positive.")
    if spec.train_years <= 0.0 or spec.train_years > 10_000.0:
        raise ValueError("Fresh Figure 5 train_years must lie in (0, 10000].")
    if spec.seed < 0:
        raise ValueError("seed must be nonnegative.")
    if spec.train_fraction != 0.9 or spec.validation_fraction != 0.2:
        raise ValueError(
            "Fresh Figure 5 uses fixed blocks; compatibility fractions must retain "
            "their defaults."
        )
    if (
        spec.common_period_max_lead_months is not None
        and spec.common_period_max_lead_months < spec.lead_months
    ):
        raise ValueError("The common maximum lead cannot be shorter than the lead.")
    if config.batch_size <= 0 or config.maximum_epochs <= 0:
        raise ValueError("batch_size and maximum_epochs must be positive.")
    if config.patience is not None and config.patience <= 0:
        raise ValueError("patience must be positive when supplied.")
    if config.statistics_batch_size <= 0:
        raise ValueError("statistics_batch_size must be positive.")
    if config.learning_rate is None or config.learning_rate <= 0.0:
        raise ValueError("Figure 5 requires an explicit positive learning rate.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative.")


def _block_inputs(
    block: tuple[int, int],
    *,
    lead_steps: int,
    common_maximum_lead_steps: int,
) -> np.ndarray:
    start, stop = block
    targets = np.arange(
        start + common_maximum_lead_steps,
        stop,
        dtype=np.int64,
    )
    if targets.size == 0:
        raise ValueError("The common lead leaves no usable pairs in a fixed block.")
    return targets - lead_steps


def select_figure5_inputs(data: ZCData, spec: ExperimentSpec) -> Figure5Selection:
    """Select a reproducible training window and fixed validation/test pairs."""

    fixed = data.fixed_supervised_split(spec.lead_months)
    lead_steps = spec.lead_months * data.steps_per_month
    maximum_lead_months = spec.common_period_max_lead_months or spec.lead_months
    common_steps = maximum_lead_months * data.steps_per_month
    available = _block_inputs(
        fixed.train_block,
        lead_steps=lead_steps,
        common_maximum_lead_steps=common_steps,
    )
    validation = _block_inputs(
        fixed.validation_block,
        lead_steps=lead_steps,
        common_maximum_lead_steps=common_steps,
    )
    test = _block_inputs(
        fixed.test_block,
        lead_steps=lead_steps,
        common_maximum_lead_steps=common_steps,
    )

    requested_count = max(1, int(round(spec.train_years * data.steps_per_year)))
    count = min(requested_count, available.size)
    maximum_offset = available.size - count
    if maximum_offset == 0:
        offset = 0
    else:
        offset = int(np.random.default_rng(spec.seed).integers(maximum_offset + 1))
    fit = np.asarray(available[offset : offset + count], dtype=np.int64)
    if math.isclose(spec.train_years, 10_000.0):
        standardization_inputs = np.arange(
            fixed.train_block[0],
            fixed.train_block[1],
            dtype=np.int64,
        )
    else:
        standardization_inputs = fit
    return Figure5Selection(
        fit_inputs=fit,
        standardization_inputs=standardization_inputs,
        validation_inputs=validation,
        test_inputs=test,
        lead_steps=lead_steps,
        common_maximum_lead_steps=common_steps,
        requested_years=float(spec.train_years),
        actual_years=fit.size / data.steps_per_year,
        available_training_pair_years=available.size / data.steps_per_year,
        window_offset=offset,
        train_block=fixed.train_block,
        validation_block=fixed.validation_block,
        test_block=fixed.test_block,
    )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    with atomic_output_path(path, overwrite=True) as temporary_path:
        torch.save(payload, temporary_path)


def _load_cached_result(
    data: ZCData,
    artifact_dir: Path,
    spec: ExperimentSpec,
    config: TrainingConfig,
) -> Figure5Result | None:
    completion_path = artifact_dir / "completed.json"
    if not completion_path.is_file():
        return None
    completion = load_json(completion_path)
    expected = {
        "schema_version": FIGURE5_TRAINING_SCHEMA_VERSION,
        "spec": asdict(spec),
        "training_config": asdict(config),
        "data_metadata_sha256": data.metadata_sha256,
        "source_sha256": _source_sha256(),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(
                f"Cached Figure 5 experiment {artifact_dir} has incompatible {key}. "
                "Use --overwrite-models to replace it intentionally."
            )
    files = completion.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Invalid Figure 5 completion manifest: {completion_path}")
    for filename, expected_digest in files.items():
        path = artifact_dir / filename
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise ValueError(f"Corrupt Figure 5 cache file: {path}")
    metrics = load_json(artifact_dir / "metrics.json")
    return Figure5Result(
        artifact_dir=artifact_dir,
        metrics=metrics,
        checkpoint_sha256=files["checkpoint.pt"],
        cache_hit=True,
    )


def train_figure5_experiment(
    data: ZCData,
    artifact_root: Path,
    spec: ExperimentSpec,
    config: TrainingConfig,
    *,
    device: str = "auto",
    overwrite: bool = False,
) -> Figure5Result:
    """Train or validate one fresh Figure 5 forecasting experiment."""

    _validate_spec(data, spec, config)
    artifact_dir = figure5_experiment_directory(artifact_root, spec)
    if not overwrite:
        cached = _load_cached_result(data, artifact_dir, spec, config)
        if cached is not None:
            LOGGER.info("Using completed Figure 5 cache: %s", artifact_dir)
            return cached

    selection = select_figure5_inputs(data, spec)
    resolved_device = resolve_device(device)
    LOGGER.info(
        "Training Figure 5 %s: lead=%s months, years=%g, seed=%s, "
        "fit=%s, validation=%s, test=%s, device=%s",
        spec.architecture.upper(),
        spec.lead_months,
        spec.train_years,
        spec.seed,
        f"{selection.fit_inputs.size:,}",
        f"{selection.validation_inputs.size:,}",
        f"{selection.test_inputs.size:,}",
        resolved_device,
    )
    seed_everything(spec.seed, deterministic=config.deterministic)
    standardizer = data.compute_standardizer(
        selection.standardization_inputs,
        batch_size=config.statistics_batch_size,
    )
    model = build_model(
        spec.architecture,
        data.spatial_input_shape,
        phase_features=data.n_phase_features,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    rng = np.random.default_rng(spec.seed + 1)
    patience = resolved_patience(spec, config)
    best_validation_loss = math.inf
    patience_reference_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.monotonic()

    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        squared_error = 0.0
        count = 0
        for indices in _ordered_training_batches(
            selection.fit_inputs,
            batch_size=config.batch_size,
            rng=rng,
        ):
            features, targets = _tensor_batch(
                data,
                indices,
                standardizer,
                lead_steps=selection.lead_steps,
                device=resolved_device,
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach().cpu()) * targets.numel()
            count += targets.numel()
        training_loss = squared_error / count
        validation_loss = mean_squared_error(
            model,
            data,
            selection.validation_inputs,
            standardizer,
            lead_steps=selection.lead_steps,
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
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.cpu()
                for name, tensor in copy.deepcopy(model.state_dict()).items()
            }
        if patience_reference_loss - validation_loss >= config.minimum_improvement:
            patience_reference_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                LOGGER.info("Early stopping after epoch %s", epoch)
                break

    if best_state is None:
        raise RuntimeError("Figure 5 training did not produce a finite checkpoint.")
    model.load_state_dict(best_state)
    model.to(resolved_device)
    predictions = _predict_model(
        model,
        data,
        selection.test_inputs,
        standardizer,
        batch_size=config.batch_size,
        device=resolved_device,
    )
    targets = data.load_targets(
        selection.test_inputs,
        lead_steps=selection.lead_steps,
    )
    test_r2 = r2_score(targets, predictions)
    elapsed_seconds = time.monotonic() - started
    LOGGER.info(
        "Best epoch %s; held-out test R2 %.6f; elapsed %.1f s",
        best_epoch,
        test_r2,
        elapsed_seconds,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_id = str(uuid.uuid4())
    source_sha256 = _source_sha256()
    checkpoint_path = artifact_dir / "checkpoint.pt"
    normalization_path = artifact_dir / "normalization.npz"
    indices_path = artifact_dir / "indices.npz"
    metrics_path = artifact_dir / "metrics.json"
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
        fit_inputs=selection.fit_inputs,
        standardization_inputs=selection.standardization_inputs,
        validation_inputs=selection.validation_inputs,
        test_inputs=selection.test_inputs,
    )
    _save_checkpoint(
        checkpoint_path,
        {
            "schema_version": FIGURE5_TRAINING_SCHEMA_VERSION,
            "state_dict": best_state,
            "architecture": spec.architecture,
            "spatial_input_shape": list(data.spatial_input_shape),
            "phase_features": data.n_phase_features,
            "spec": asdict(spec),
            "training_config": asdict(config),
            "best_epoch": best_epoch,
            "lead_steps": selection.lead_steps,
            "generation_id": generation_id,
            "data_metadata_sha256": data.metadata_sha256,
            "source_sha256": source_sha256,
        },
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    metrics = {
        "schema_version": FIGURE5_TRAINING_SCHEMA_VERSION,
        "spec": asdict(spec),
        "training_config": asdict(config),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(config.learning_rate),
            "weight_decay": config.weight_decay,
        },
        "data": data.provenance(),
        "model": {
            "parameter_count": parameter_count(model),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "selection": {
            "strategy": FIGURE5_SELECTION_STRATEGY,
            "normalization_population": (
                FIGURE5_FULL_NORMALIZATION_POPULATION
                if math.isclose(spec.train_years, 10_000.0)
                else FIGURE5_SUBSET_NORMALIZATION_POPULATION
            ),
            "normalization_count": int(selection.standardization_inputs.size),
            "requested_years": selection.requested_years,
            "actual_years": selection.actual_years,
            "available_training_pair_years": (
                selection.available_training_pair_years
            ),
            "fit_years": selection.actual_years,
            "validation_years": (
                selection.validation_inputs.size / data.steps_per_year
            ),
            "test_years": selection.test_inputs.size / data.steps_per_year,
            "window_offset": selection.window_offset,
            "window_start_step": int(selection.fit_inputs[0]),
            "window_stop_step_exclusive": int(selection.fit_inputs[-1]) + 1,
            "fixed_blocks": {
                "train": list(selection.train_block),
                "validation": list(selection.validation_block),
                "test": list(selection.test_block),
            },
            "common_maximum_lead_steps": selection.common_maximum_lead_steps,
        },
        "evaluation": {
            "test_input_start_step": int(selection.test_inputs[0]),
            "test_input_stop_step_exclusive": int(selection.test_inputs[-1]) + 1,
            "test_target_start_step": int(
                selection.test_inputs[0] + selection.lead_steps
            ),
            "test_target_stop_step_exclusive": int(
                selection.test_inputs[-1] + selection.lead_steps + 1
            ),
            "test_count": int(selection.test_inputs.size),
        },
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_loss,
        "test_r2": test_r2,
        "elapsed_seconds": elapsed_seconds,
        "device_used": str(resolved_device),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "history": history,
        "generation_id": generation_id,
        "source_sha256": source_sha256,
    }
    write_json(metrics_path, metrics, overwrite=True)
    files = {
        "checkpoint.pt": checkpoint_sha256,
        "normalization.npz": sha256_file(normalization_path),
        "indices.npz": sha256_file(indices_path),
        "metrics.json": sha256_file(metrics_path),
    }
    write_json(
        artifact_dir / "completed.json",
        {
            "schema_version": FIGURE5_TRAINING_SCHEMA_VERSION,
            "generation_id": generation_id,
            "spec": asdict(spec),
            "training_config": asdict(config),
            "data_metadata_sha256": data.metadata_sha256,
            "source_sha256": source_sha256,
            "files": files,
        },
        overwrite=True,
    )
    model.cpu()
    return Figure5Result(
        artifact_dir=artifact_dir,
        metrics=metrics,
        checkpoint_sha256=checkpoint_sha256,
        cache_hit=False,
    )
