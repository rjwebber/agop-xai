#!/usr/bin/env python3
"""Benchmark the three exact-dense AGOP stages on a trained fresh-ZC model.

The production solver streams input gradients directly into the empirical
matrix.  This benchmark deliberately inserts a validated float32 gradient
cache so that it can time, independently,

1. standardized input loading and neural-network gradient evaluation,
2. float64 dense ``M = G.T @ G / n`` accumulation, and
3. the complete float64 symmetric eigendecomposition.

With the same gradient batch size, stages 1--3 use the same gradient values,
DSYRK update order, and EVD driver as ``build_exact_dense_agop_factor``.  The
gradient cache is a diagnostic timing artifact, not part of the intended
production workflow.
"""

from __future__ import annotations

import argparse
import logging
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy.linalg import blas, eigh
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import (  # noqa: E402
    FRESH_INPUT_PROFILES,
    FRESH_SCHEMA_VERSION,
    Standardizer,
    ZCData,
)
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    load_json,
    sha256_array,
    sha256_file,
    sha256_json,
    write_json,
    write_npz,
)
from zc_xai.models import ARCHITECTURES  # noqa: E402
from zc_xai.training import (  # noqa: E402
    ExperimentSpec,
    load_experiment,
    resolve_device,
)
from zc_xai.xai import (  # noqa: E402
    _clip_numerical_negative_eigenvalues,
    _exact_dense_agop_identity,
    _validated_dense_agop_inputs,
    input_gradients,
)

LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
DEFAULT_DATA_DIR = Path("data/processed/zc-v3")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/zc-v3")
DEFAULT_OUTPUT_ROOT = Path("outputs/fresh_agop_benchmark")


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Separately time fresh-ZC input gradients, exact float64 dense "
            "AGOP accumulation, and the complete float64 eigendecomposition."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Stage caches and report directory. If omitted, a portable name is "
            "constructed beneath outputs/fresh_agop_benchmark."
        ),
    )
    parser.add_argument(
        "--input-profile",
        choices=tuple(FRESH_INPUT_PROFILES),
        default="core4",
    )
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="cnn")
    parser.add_argument("--lead-months", type=positive_int, default=10)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--reference-count",
        type=nonnegative_int,
        default=0,
        help=(
            "Use this chronological prefix of fixed-block training predictors; "
            "zero means every usable training predictor."
        ),
    )
    parser.add_argument(
        "--gradient-batch-size",
        type=positive_int,
        default=256,
        help=(
            "Batch size for both gradient evaluation and DSYRK updates, matching "
            "the streamed production solver's floating-point update order."
        ),
    )
    parser.add_argument(
        "--skip-data-checksums",
        action="store_true",
        help=(
            "Skip the expensive full processed-file checksum pass. This is useful "
            "for development only and is recorded prominently in the report."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically rebuild all stage caches for this exact run identity.",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_output_dir(args: argparse.Namespace, reference_count: int) -> Path:
    reference_label = "all" if args.reference_count == 0 else str(reference_count)
    name = (
        f"{args.input_profile}-{args.architecture}-lead-{args.lead_months:02d}m-"
        f"seed-{args.seed:06d}-refs-{reference_label}"
    )
    return DEFAULT_OUTPUT_ROOT / name


def _artifact_id(artifact_dir: Path, artifact_root: Path) -> str:
    root = artifact_root.expanduser().resolve()
    artifact = artifact_dir.expanduser().resolve()
    try:
        return artifact.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            "The trained model artifact is outside --artifacts-dir; cannot write "
            "portable provenance."
        ) from error


def _software_identity(device: str) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "device": device,
    }


def _machine_description() -> dict[str, str]:
    # No host name, user name, or absolute path: the report remains portable.
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _cache_manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def _validated_array_cache(
    path: Path,
    *,
    stage: str,
    identity_sha256: str,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Open a complete immutable array cache only after validating its manifest."""

    started = time.perf_counter()
    manifest_path = _cache_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Incomplete {stage} cache: expected both {path.name} and "
            f"{manifest_path.name}. Use --overwrite to rebuild it."
        )
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("stage") != stage
        or manifest.get("identity_sha256") != identity_sha256
        or not isinstance(manifest.get("identity"), dict)
        or sha256_json(manifest["identity"]) != identity_sha256
    ):
        raise ValueError(
            f"The existing {stage} cache has another identity. Choose another "
            "--output-dir or pass --overwrite."
        )
    digest = sha256_file(path)
    if manifest.get("file_sha256") != digest:
        raise ValueError(f"The {stage} cache is incomplete or corrupted: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape or values.dtype != expected_dtype:
        raise ValueError(f"The {stage} cache has an invalid array schema: {path}")
    validation_seconds = time.perf_counter() - started
    return values, {
        "cache_hit": True,
        "cache_validation_seconds_this_invocation": validation_seconds,
        "file": path.name,
        "file_sha256": digest,
        "build_timing": manifest.get("build_timing", {}),
    }


def generate_gradient_cache(
    model: nn.Module,
    data: ZCData,
    standardizer: Standardizer,
    references: np.ndarray,
    *,
    batch_size: int,
    device: str,
    path: Path,
    identity: dict[str, Any],
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate or validate a float32 cache of standardized-input gradients."""

    references = _validated_dense_agop_inputs(
        data,
        standardizer,
        references,
        gradient_batch_size=batch_size,
    )
    identity_sha256 = sha256_json(identity)
    expected_shape = (references.size, data.n_features)
    expected_dtype = np.dtype(np.float32)
    if path.exists() and not overwrite:
        return _validated_array_cache(
            path,
            stage="input_gradients",
            identity_sha256=identity_sha256,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
        )

    stage_started = time.perf_counter()
    input_loading_seconds = 0.0
    gradient_evaluation_seconds = 0.0
    cache_assignment_seconds = 0.0
    with atomic_output_path(path, overwrite=True) as temporary_path:
        destination = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=expected_dtype,
            shape=expected_shape,
        )
        for start in range(0, references.size, batch_size):
            stop = min(start + batch_size, references.size)
            timer = time.perf_counter()
            inputs = data.load_inputs(
                references[start:stop],
                standardizer=standardizer,
            )
            input_loading_seconds += time.perf_counter() - timer

            timer = time.perf_counter()
            gradients = input_gradients(
                model,
                inputs,
                batch_size=batch_size,
                device=device,
            )
            gradient_evaluation_seconds += time.perf_counter() - timer
            flattened = gradients.reshape(stop - start, data.n_features)
            if not np.isfinite(flattened).all():
                raise ValueError("Fresh AGOP gradients contain non-finite values.")

            timer = time.perf_counter()
            destination[start:stop] = flattened
            cache_assignment_seconds += time.perf_counter() - timer
            LOGGER.info(
                "Fresh AGOP gradients: %s / %s",
                f"{stop:,}",
                f"{references.size:,}",
            )
        destination.flush()
        del destination
    stage_seconds = time.perf_counter() - stage_started

    integrity_started = time.perf_counter()
    digest = sha256_file(path)
    integrity_seconds = time.perf_counter() - integrity_started
    accounted = (
        input_loading_seconds
        + gradient_evaluation_seconds
        + cache_assignment_seconds
    )
    build_timing = {
        "stage_wall_seconds_excluding_integrity_hash": stage_seconds,
        "standardized_input_loading_seconds": input_loading_seconds,
        "gradient_evaluation_seconds": gradient_evaluation_seconds,
        "gradient_cache_assignment_seconds": cache_assignment_seconds,
        "other_and_publish_seconds": max(0.0, stage_seconds - accounted),
        "integrity_hash_seconds": integrity_seconds,
        "rows_per_second_for_gradient_evaluation": (
            references.size / gradient_evaluation_seconds
        ),
    }
    write_json(
        _cache_manifest_path(path),
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage": "input_gradients",
            "created_at_utc": _utc_now(),
            "identity": identity,
            "identity_sha256": identity_sha256,
            "file": path.name,
            "file_sha256": digest,
            "shape": list(expected_shape),
            "dtype": expected_dtype.str,
            "build_timing": build_timing,
        },
        overwrite=True,
    )
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return values, {
        "cache_hit": False,
        "cache_validation_seconds_this_invocation": 0.0,
        "file": path.name,
        "file_sha256": digest,
        "build_timing": build_timing,
    }


def accumulate_dense_from_gradient_cache(
    gradients: np.ndarray,
    *,
    block_rows: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Form the exact float64 empirical Gram matrix in streamed DSYRK blocks."""

    values = np.asarray(gradients)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Gradient cache must be a nonempty two-dimensional array.")
    if values.dtype != np.float32:
        raise ValueError("Gradient cache must use float32, matching input_gradients.")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")

    started = time.perf_counter()
    read_and_cast_seconds = 0.0
    dsyrk_seconds = 0.0
    n_rows, n_features = values.shape
    matrix = np.zeros((n_features, n_features), dtype=np.float64, order="F")
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        timer = time.perf_counter()
        block = np.asfortranarray(
            np.asarray(values[start:stop], dtype=np.float64)
        )
        read_and_cast_seconds += time.perf_counter() - timer
        if not np.isfinite(block).all():
            raise ValueError("Gradient cache contains non-finite values.")

        timer = time.perf_counter()
        matrix = blas.dsyrk(
            alpha=1.0,
            a=block,
            beta=1.0,
            c=matrix,
            trans=1,
            lower=0,
            overwrite_c=1,
        )
        dsyrk_seconds += time.perf_counter() - timer
    finalize_started = time.perf_counter()
    matrix /= n_rows
    upper = np.triu(matrix)
    matrix = upper + np.triu(upper, k=1).T
    finalize_seconds = time.perf_counter() - finalize_started
    total_seconds = time.perf_counter() - started
    return matrix, {
        "dense_accumulation_seconds": total_seconds,
        "gradient_cache_read_and_float64_cast_seconds": read_and_cast_seconds,
        "dsyrk_seconds": dsyrk_seconds,
        "normalization_and_symmetrization_seconds": finalize_seconds,
        "other_seconds": max(
            0.0,
            total_seconds
            - read_and_cast_seconds
            - dsyrk_seconds
            - finalize_seconds,
        ),
    }


def build_matrix_cache(
    gradients: np.ndarray,
    *,
    block_rows: int,
    path: Path,
    identity: dict[str, Any],
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    identity_sha256 = sha256_json(identity)
    dimension = int(gradients.shape[1])
    expected_shape = (dimension, dimension)
    if path.exists() and not overwrite:
        return _validated_array_cache(
            path,
            stage="dense_agop_matrix",
            identity_sha256=identity_sha256,
            expected_shape=expected_shape,
            expected_dtype=np.dtype(np.float64),
        )

    matrix, build_timing = accumulate_dense_from_gradient_cache(
        gradients,
        block_rows=block_rows,
    )
    timer = time.perf_counter()
    with (
        atomic_output_path(path, overwrite=True) as temporary_path,
        temporary_path.open("wb") as stream,
    ):
        np.save(stream, matrix, allow_pickle=False)
    build_timing["matrix_cache_write_and_publish_seconds"] = (
        time.perf_counter() - timer
    )
    timer = time.perf_counter()
    digest = sha256_file(path)
    build_timing["integrity_hash_seconds"] = time.perf_counter() - timer
    write_json(
        _cache_manifest_path(path),
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage": "dense_agop_matrix",
            "created_at_utc": _utc_now(),
            "identity": identity,
            "identity_sha256": identity_sha256,
            "file": path.name,
            "file_sha256": digest,
            "shape": list(expected_shape),
            "dtype": np.dtype(np.float64).str,
            "build_timing": build_timing,
        },
        overwrite=True,
    )
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return values, {
        "cache_hit": False,
        "cache_validation_seconds_this_invocation": 0.0,
        "file": path.name,
        "file_sha256": digest,
        "build_timing": build_timing,
    }


def eigendecompose_dense_matrix(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the production float64 EVD call and numerical clipping policy."""

    values = np.asarray(matrix)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[0] != values.shape[1]
        or values.dtype != np.float64
    ):
        raise ValueError("Dense AGOP matrix must be square, nonempty, and float64.")
    validation_started = time.perf_counter()
    if not np.isfinite(values).all() or not np.array_equal(values, values.T):
        raise ValueError("Dense AGOP matrix must be finite and exactly symmetric.")
    validation_seconds = time.perf_counter() - validation_started

    copy_started = time.perf_counter()
    workspace = np.array(values, dtype=np.float64, order="F", copy=True)
    copy_seconds = time.perf_counter() - copy_started
    solver_started = time.perf_counter()
    eigenvalues, basis = eigh(
        workspace,
        lower=False,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    solver_seconds = time.perf_counter() - solver_started
    post_started = time.perf_counter()
    eigenvalues, clipping = _clip_numerical_negative_eigenvalues(eigenvalues)
    order = np.arange(eigenvalues.size - 1, -1, -1)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    basis = np.asarray(basis[:, order], dtype=np.float64)
    root_eigenvalues = np.sqrt(eigenvalues)
    post_seconds = time.perf_counter() - post_started
    return basis, root_eigenvalues, {
        "matrix_validation_seconds": validation_seconds,
        "fortran_workspace_copy_seconds": copy_seconds,
        "full_eigendecomposition_seconds": solver_seconds,
        "ordering_clipping_and_square_root_seconds": post_seconds,
        "eigenvalue_clipping": clipping,
        "largest_eigenvalue": float(eigenvalues[0]),
        "trace": float(eigenvalues.sum()),
    }


def build_eigensystem_cache(
    matrix: np.ndarray,
    *,
    path: Path,
    identity: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    identity_sha256 = sha256_json(identity)
    manifest_path = _cache_manifest_path(path)
    if path.exists() and not overwrite:
        started = time.perf_counter()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Incomplete eigensystem cache: {manifest_path.name} is missing."
            )
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("stage") != "full_eigensystem"
            or manifest.get("identity_sha256") != identity_sha256
            or not isinstance(manifest.get("identity"), dict)
            or sha256_json(manifest["identity"]) != identity_sha256
        ):
            raise ValueError(
                "The existing eigensystem cache has another identity. Choose "
                "another --output-dir or pass --overwrite."
            )
        digest = sha256_file(path)
        if manifest.get("file_sha256") != digest:
            raise ValueError(f"The eigensystem cache is corrupted: {path}")
        with np.load(path, allow_pickle=False) as archive:
            basis = archive["basis"]
            root = archive["root_eigenvalues"]
            dimension = int(matrix.shape[0])
            if (
                basis.shape != (dimension, dimension)
                or root.shape != (dimension,)
                or basis.dtype != np.float64
                or root.dtype != np.float64
            ):
                raise ValueError("The eigensystem cache has an invalid schema.")
        return {
            "cache_hit": True,
            "cache_validation_seconds_this_invocation": (
                time.perf_counter() - started
            ),
            "file": path.name,
            "file_sha256": digest,
            "build_timing": manifest.get("build_timing", {}),
            "spectrum": manifest.get("spectrum", {}),
        }

    basis, root, decomposition = eigendecompose_dense_matrix(matrix)
    cache_started = time.perf_counter()
    write_npz(
        path,
        overwrite=True,
        compressed=False,
        basis=basis,
        root_eigenvalues=root,
    )
    decomposition["eigensystem_cache_write_and_publish_seconds"] = (
        time.perf_counter() - cache_started
    )
    integrity_started = time.perf_counter()
    digest = sha256_file(path)
    decomposition["integrity_hash_seconds"] = (
        time.perf_counter() - integrity_started
    )
    spectrum = {
        "largest_eigenvalue": decomposition["largest_eigenvalue"],
        "trace": decomposition["trace"],
        "numerical_rank_positive_eigenvalues": int(np.count_nonzero(root > 0.0)),
    }
    write_json(
        manifest_path,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage": "full_eigensystem",
            "created_at_utc": _utc_now(),
            "identity": identity,
            "identity_sha256": identity_sha256,
            "file": path.name,
            "file_sha256": digest,
            "basis_shape": list(basis.shape),
            "root_eigenvalues_shape": list(root.shape),
            "dtype": np.dtype(np.float64).str,
            "build_timing": decomposition,
            "spectrum": spectrum,
        },
        overwrite=True,
    )
    return {
        "cache_hit": False,
        "cache_validation_seconds_this_invocation": 0.0,
        "file": path.name,
        "file_sha256": digest,
        "build_timing": decomposition,
        "spectrum": spectrum,
    }


def _report_payload(
    *,
    run_identity: dict[str, Any],
    run_identity_sha256: str,
    stages: dict[str, dict[str, Any]],
    output_files: dict[str, str],
) -> dict[str, Any]:
    complete = set(stages) == {"input_gradients", "dense_matrix", "eigensystem"}
    component_seconds: dict[str, float] = {}
    gradient_timing = stages.get("input_gradients", {}).get("build_timing", {})
    matrix_timing = stages.get("dense_matrix", {}).get("build_timing", {})
    eigensystem_timing = stages.get("eigensystem", {}).get("build_timing", {})
    if all(
        isinstance(timing, dict)
        for timing in (gradient_timing, matrix_timing, eigensystem_timing)
    ):
        keys = {
            "standardized_input_loading": (
                gradient_timing,
                "standardized_input_loading_seconds",
            ),
            "input_gradient_evaluation": (
                gradient_timing,
                "gradient_evaluation_seconds",
            ),
            "dense_matrix_accumulation": (
                matrix_timing,
                "dense_accumulation_seconds",
            ),
            "full_dense_eigendecomposition": (
                eigensystem_timing,
                "full_eigendecomposition_seconds",
            ),
        }
        for label, (timing, key) in keys.items():
            value = timing.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                component_seconds[label] = float(value)
        if len(component_seconds) == len(keys):
            component_seconds["four_component_sum_not_wall_time"] = sum(
                component_seconds.values()
            )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "complete" if complete else "in_progress",
        "updated_at_utc": _utc_now(),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "machine": _machine_description(),
        "stages": stages,
        "measured_component_seconds": component_seconds,
        "output_files": output_files,
        "interpretation": {
            "gradient_cache_role": (
                "diagnostic timing boundary only; production streams gradients "
                "directly into M"
            ),
            "matrix_definition": "sum_i grad_i grad_i^T / reference_count",
            "equivalence_to_production": (
                "same standardized inputs, float32 gradients, batch order, "
                "float64 DSYRK updates, clipping policy, and SciPy EVD driver"
            ),
            "timing_boundary": (
                "integrity hashing and artifact writes are reported separately "
                "from gradient evaluation, DSYRK, and EVD"
            ),
        },
    }


def _write_progress(
    report_path: Path,
    *,
    run_identity: dict[str, Any],
    run_identity_sha256: str,
    stages: dict[str, dict[str, Any]],
    output_files: dict[str, str],
) -> None:
    write_json(
        report_path,
        _report_payload(
            run_identity=run_identity,
            run_identity_sha256=run_identity_sha256,
            stages=stages,
            output_files=output_files,
        ),
        overwrite=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    data_dir = args.data_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    resolved_device = str(resolve_device(args.device))
    LOGGER.info("Loading fresh-ZC profile=%s", args.input_profile)
    data = ZCData(
        data_dir,
        input_profile=args.input_profile,
        verify_checksums=not args.skip_data_checksums,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("This benchmark accepts only the fresh zc-v3 data set.")
    spec = ExperimentSpec(
        architecture=args.architecture,
        lead_months=args.lead_months,
        train_years=10_000.0,
        seed=args.seed,
        input_profile=args.input_profile,
    )
    experiment = load_experiment(data, artifacts_dir, spec, device="cpu")
    fixed = data.fixed_supervised_split(args.lead_months)
    if not np.array_equal(experiment.fit_inputs, fixed.train_inputs):
        raise ValueError(
            "The trained artifact does not use every fixed-block training predictor."
        )
    reference_count = (
        fixed.train_inputs.size
        if args.reference_count == 0
        else args.reference_count
    )
    if reference_count > fixed.train_inputs.size:
        raise ValueError(
            f"--reference-count exceeds the {fixed.train_inputs.size:,} usable "
            "fixed-block training predictors."
        )
    references = np.asarray(
        fixed.train_inputs[:reference_count],
        dtype=np.int64,
    )
    references = _validated_dense_agop_inputs(
        data,
        experiment.standardizer,
        references,
        gradient_batch_size=args.gradient_batch_size,
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _default_output_dir(args, reference_count)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = output_dir / "input_gradients.npy"
    matrix_path = output_dir / "dense_agop_matrix.npy"
    eigensystem_path = output_dir / "full_eigensystem.npz"
    report_path = output_dir / "report.json"
    output_files = {
        "gradient_cache": gradient_path.name,
        "dense_matrix_cache": matrix_path.name,
        "full_eigensystem_cache": eigensystem_path.name,
        "report": report_path.name,
    }

    caller_identity = {
        "benchmark_script_version": SCRIPT_VERSION,
        "checkpoint_sha256": experiment.checkpoint_sha256,
        "checkpoint_generation_id": experiment.metrics.get("generation_id"),
        "artifact_id": _artifact_id(experiment.artifact_dir, artifacts_dir),
        "gradient_batch_size": args.gradient_batch_size,
        "resolved_device": resolved_device,
        "software": _software_identity(resolved_device),
    }
    exact_identity = _exact_dense_agop_identity(
        experiment.model,
        data,
        experiment.standardizer,
        references,
        cache_identity=caller_identity,
    )
    run_identity = {
        "purpose": "fresh exact-dense AGOP component timing",
        "data": data.provenance(),
        "processed_checksums_verified": not args.skip_data_checksums,
        "model": {
            "architecture": args.architecture,
            "input_profile": args.input_profile,
            "lead_months": args.lead_months,
            "seed": args.seed,
            "artifact_id": caller_identity["artifact_id"],
            "checkpoint_sha256": experiment.checkpoint_sha256,
            "generation_id": caller_identity["checkpoint_generation_id"],
            "training_source_sha256": experiment.metrics.get(
                "training_source_sha256"
            ),
        },
        "references": {
            "selection": "chronological prefix of fixed training predictors",
            "requested_count_zero_means_all": args.reference_count,
            "count": int(references.size),
            "available_fixed_training_predictors": int(fixed.train_inputs.size),
            "indices_sha256": sha256_array(references),
        },
        "standardizer": {
            "mean_sha256": sha256_array(experiment.standardizer.mean),
            "scale_sha256": sha256_array(experiment.standardizer.scale),
            "count": experiment.standardizer.count,
        },
        "numeric_method": {
            "gradient_dtype": np.dtype(np.float32).str,
            "gradient_batch_size": args.gradient_batch_size,
            "matrix_dtype": np.dtype(np.float64).str,
            "matrix_kernel": "scipy.linalg.blas.dsyrk upper triangle",
            "eigendecomposition_dtype": np.dtype(np.float64).str,
            "eigendecomposition_driver": "scipy.linalg.eigh driver=evd",
            "full_basis": True,
        },
        "software": _software_identity(resolved_device),
        "script_sha256": sha256_file(Path(__file__)),
        "exact_dense_identity_sha256": sha256_json(exact_identity),
    }
    run_identity_sha256 = sha256_json(run_identity)
    stages: dict[str, dict[str, Any]] = {}
    if report_path.exists() and not args.overwrite:
        prior = load_json(report_path)
        if (
            prior.get("run_identity_sha256") != run_identity_sha256
            or prior.get("run_identity") != run_identity
        ):
            raise ValueError(
                "The existing report has another identity. Choose another "
                "--output-dir or pass --overwrite."
            )
        prior_stages = prior.get("stages")
        if isinstance(prior_stages, dict):
            stages = dict(prior_stages)

    gradient_identity = {
        "stage": "input_gradients",
        "exact_dense_identity": exact_identity,
        "gradient_dtype": np.dtype(np.float32).str,
        "gradient_batch_size": args.gradient_batch_size,
        "resolved_device": resolved_device,
        "software": _software_identity(resolved_device),
    }
    gradients, gradient_record = generate_gradient_cache(
        experiment.model,
        data,
        experiment.standardizer,
        references,
        batch_size=args.gradient_batch_size,
        device=resolved_device,
        path=gradient_path,
        identity=gradient_identity,
        overwrite=args.overwrite,
    )
    stages["input_gradients"] = gradient_record
    _write_progress(
        report_path,
        run_identity=run_identity,
        run_identity_sha256=run_identity_sha256,
        stages=stages,
        output_files=output_files,
    )

    matrix_identity = {
        "stage": "dense_agop_matrix",
        "gradient_file_sha256": gradient_record["file_sha256"],
        "reference_count": int(references.size),
        "feature_count": data.n_features,
        "block_rows": args.gradient_batch_size,
        "dtype": np.dtype(np.float64).str,
        "kernel": "scipy.linalg.blas.dsyrk upper triangle",
    }
    matrix, matrix_record = build_matrix_cache(
        gradients,
        block_rows=args.gradient_batch_size,
        path=matrix_path,
        identity=matrix_identity,
        overwrite=args.overwrite,
    )
    stages["dense_matrix"] = matrix_record
    _write_progress(
        report_path,
        run_identity=run_identity,
        run_identity_sha256=run_identity_sha256,
        stages=stages,
        output_files=output_files,
    )

    eigensystem_identity = {
        "stage": "full_eigensystem",
        "matrix_file_sha256": matrix_record["file_sha256"],
        "feature_count": data.n_features,
        "dtype": np.dtype(np.float64).str,
        "driver": "scipy.linalg.eigh driver=evd",
        "negative_eigenvalue_policy": (
            "clip only roundoff-scale negatives; reject material negatives"
        ),
    }
    eigensystem_record = build_eigensystem_cache(
        matrix,
        path=eigensystem_path,
        identity=eigensystem_identity,
        overwrite=args.overwrite,
    )
    stages["eigensystem"] = eigensystem_record
    _write_progress(
        report_path,
        run_identity=run_identity,
        run_identity_sha256=run_identity_sha256,
        stages=stages,
        output_files=output_files,
    )
    LOGGER.info("Fresh AGOP timing report complete: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
