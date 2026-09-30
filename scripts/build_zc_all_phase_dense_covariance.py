#!/usr/bin/env python3
"""Build the common all-phase native ZC covariance as a dense matrix.

The source artifact contains one centered sample factor for every phase of the
36-step model year.  This script forms

    C = (1 / 36) sum_p (X_p - mu_p).T (X_p - mu_p) / (n - 1)

in float64 with BLAS ``dsyrk``.  The upper triangle is accumulated in a
Fortran-order ``.npy`` memory map and mirrored only after all phases complete.

Completed phases are transactionally checkpointed as immutable generations.
An interrupted run can therefore resume at a phase boundary without either
losing or double-counting a committed phase.  ``covariance.npy`` and its final
manifest are not published until the complete matrix passes source-bound
diagonal, symmetry, finiteness, and direct-application audits.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg.blas import dsyrk

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.io import (  # noqa: E402
    load_json,
    sha256_array,
    sha256_file,
    write_json,
)
from zc_xai.native_covariance import (  # noqa: E402
    DENSE_NATIVE_COVARIANCE_SCHEMA_VERSION,
    DENSE_NATIVE_COVARIANCE_STATUS,
    NATIVE_COVARIANCE_SCHEMA_VERSION,
    NATIVE_COVARIANCE_STATUS,
    DenseNativeCovarianceOperator,
)

SCRIPT_VERSION = "1.0.0"
DENSE_COVARIANCE_SCHEMA_VERSION = DENSE_NATIVE_COVARIANCE_SCHEMA_VERSION
DENSE_COVARIANCE_STATUS = DENSE_NATIVE_COVARIANCE_STATUS
PROGRESS_STATUS = "in_progress_dense_native_covariance"
READY_STATUS = "ready_to_publish_dense_native_covariance"
FLOAT32_DTYPE = np.dtype("<f4")
FLOAT64_DTYPE = np.dtype("<f8")
DEFAULT_FACTOR_MANIFEST = Path(
    "outputs/zc_native_covariance/training-years10000-phases00-35/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/zc_native_covariance/training-years10000-phases00-35"
)
WORK_DIRECTORY_NAME = ".dense_covariance_work"
PROGRESS_FILE_NAME = "progress.json"
FINALIZING_FILE_NAME = "covariance.finalizing.npy"
FINAL_COVARIANCE_FILE_NAME = "covariance.npy"
FINAL_MANIFEST_FILE_NAME = "dense_manifest.json"


@dataclass(frozen=True)
class DenseCovarianceSource:
    """Validated phase factors and moments used by the dense build."""

    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    phase_offsets: np.ndarray
    samples: tuple[np.ndarray, ...]
    means: np.ndarray
    coordinate_variances: np.ndarray
    source_records: tuple[dict[str, Any], ...]
    moments_path: Path
    moments_sha256: str

    @property
    def phase_count(self) -> int:
        return int(self.phase_offsets.size)

    @property
    def sample_count(self) -> int:
        return int(self.samples[0].shape[0])

    @property
    def state_size(self) -> int:
        return int(self.samples[0].shape[1])


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument(
        "--factor-manifest", type=Path, default=DEFAULT_FACTOR_MANIFEST
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument("--expected-state-size", type=positive_int, default=28_591)
    result.add_argument("--expected-samples", type=positive_int, default=10_000)
    result.add_argument("--expected-phase-count", type=positive_int, default=36)
    result.add_argument(
        "--block-rows",
        type=positive_int,
        default=10_000,
        help=(
            "Rows centered in memory per dsyrk call. The default processes one "
            "phase per call (about 2.13 GiB for the production dimensions); "
            "smaller values reduce memory but repeatedly stream the dense matrix."
        ),
    )
    result.add_argument(
        "--audit-vectors",
        type=positive_int,
        default=3,
        help="Random vectors used for an independent factor-vs-dense audit.",
    )
    result.add_argument(
        "--audit-block-columns",
        type=positive_int,
        default=128,
        help="Dense columns inspected at once during final audits/mirroring.",
    )
    result.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="Trust recorded source-array hashes instead of recomputing them.",
    )
    mode = result.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after validating the source-bound checkpoint.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard this builder's prior final/work artifacts and start again.",
    )
    return result


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), start=parent.resolve())


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_dense_array(path: Path, *, state_size: int, mode: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    covariance = np.load(path, mmap_mode=mode, allow_pickle=False)
    if covariance.shape != (state_size, state_size):
        raise ValueError(
            f"dense covariance has shape {covariance.shape}, expected "
            f"({state_size}, {state_size})"
        )
    if covariance.dtype != FLOAT64_DTYPE:
        raise TypeError("dense covariance must be little-endian float64")
    if not covariance.flags.f_contiguous:
        raise ValueError("dense covariance must have Fortran storage order")
    return covariance


def _load_source(
    manifest_path: Path,
    *,
    expected_state_size: int,
    expected_samples: int,
    expected_phase_count: int,
    verify_source_hashes: bool,
) -> DenseCovarianceSource:
    """Load and fail closed on the complete all-phase factor artifact."""

    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != NATIVE_COVARIANCE_SCHEMA_VERSION:
        raise ValueError("factor manifest has an unsupported schema")
    if manifest.get("status") != NATIVE_COVARIANCE_STATUS:
        raise ValueError("factor manifest is not complete")

    dimensions = manifest.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("factor manifest lacks dimensions")
    phase_offsets = np.asarray(dimensions.get("phase_offsets"), dtype=np.int64)
    expected_offsets = np.arange(expected_phase_count, dtype=np.int64)
    if not np.array_equal(phase_offsets, expected_offsets):
        raise ValueError(
            "dense common covariance requires every phase offset exactly once: "
            f"expected {expected_offsets.tolist()}, got {phase_offsets.tolist()}"
        )
    required_dimensions = {
        "state_size": expected_state_size,
        "sample_count_per_phase": expected_samples,
        "phase_count": expected_phase_count,
    }
    for key, expected in required_dimensions.items():
        if dimensions.get(key) != expected:
            raise ValueError(f"factor manifest disagrees with required {key}")

    raw_records = manifest.get("phase_factors")
    if not isinstance(raw_records, list) or len(raw_records) != expected_phase_count:
        raise ValueError("factor manifest has the wrong number of phase factors")
    by_phase: dict[int, dict[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict) or "phase_offset" not in raw or "file" not in raw:
            raise ValueError("factor manifest contains a malformed phase record")
        phase = int(raw["phase_offset"])
        if phase in by_phase:
            raise ValueError(f"duplicate factor record for phase {phase}")
        by_phase[phase] = dict(raw)
    if set(by_phase) != set(int(value) for value in expected_offsets):
        raise ValueError("factor records do not cover all declared phases")

    moments_record = manifest.get("means_and_variances")
    if not isinstance(moments_record, dict) or "file" not in moments_record:
        raise ValueError("factor manifest lacks phase moments")
    moments_path = (manifest_path.parent / str(moments_record["file"])).resolve()
    if not moments_path.is_file():
        raise FileNotFoundError(moments_path)
    moments_sha256 = sha256_file(moments_path)
    if moments_sha256 != moments_record.get("sha256"):
        raise ValueError("phase moments fail their recorded SHA-256")
    with np.load(moments_path, allow_pickle=False) as moments:
        stored_offsets = np.asarray(moments["phase_offsets"], dtype=np.int64)
        means = np.asarray(moments["means"], dtype=np.float64)
        variances = np.asarray(
            moments["coordinate_variances"], dtype=np.float64
        )
        pooled_variance = np.asarray(
            moments["pooled_coordinate_variance"], dtype=np.float64
        )
    if not np.array_equal(stored_offsets, phase_offsets):
        raise ValueError("phase moments and factor manifest offsets differ")
    expected_moment_shape = (expected_phase_count, expected_state_size)
    if means.shape != expected_moment_shape or variances.shape != expected_moment_shape:
        raise ValueError("phase moments have incompatible dimensions")
    if pooled_variance.shape != (expected_state_size,):
        raise ValueError("pooled coordinate variance has incompatible dimensions")
    if not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise ValueError("phase moments contain nonfinite values")
    if np.any(variances < 0.0):
        raise ValueError("phase coordinate variances contain negative values")
    if sha256_array(means) != moments_record.get("means_sha256"):
        raise ValueError("phase means fail their recorded array SHA-256")
    if sha256_array(variances) != moments_record.get(
        "coordinate_variances_sha256"
    ):
        raise ValueError("phase variances fail their recorded array SHA-256")
    if sha256_array(pooled_variance) != moments_record.get(
        "pooled_coordinate_variance_sha256"
    ):
        raise ValueError("pooled variance fails its recorded array SHA-256")
    computed_pooled = np.mean(variances, axis=0, dtype=np.float64)
    if not np.array_equal(computed_pooled, pooled_variance):
        raise ValueError("pooled variance is not the equal-phase mean")

    samples: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    for phase in expected_offsets:
        record = by_phase[int(phase)]
        path = (manifest_path.parent / str(record["file"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        size_bytes = path.stat().st_size
        if record.get("size_bytes") != size_bytes:
            raise ValueError(f"phase-{phase} source size changed")
        recorded_hash = record.get("sha256")
        if not _valid_sha256(recorded_hash):
            raise ValueError(f"phase-{phase} source lacks a valid SHA-256")
        actual_hash = sha256_file(path) if verify_source_hashes else recorded_hash
        if actual_hash != recorded_hash:
            raise ValueError(f"phase-{phase} source fails its recorded SHA-256")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (expected_samples, expected_state_size):
            raise ValueError(
                f"phase-{phase} source has incompatible shape {values.shape}"
            )
        if values.dtype != FLOAT32_DTYPE:
            raise TypeError(f"phase-{phase} source must be little-endian float32")
        if record.get("shape") != list(values.shape):
            raise ValueError(f"phase-{phase} source shape disagrees with manifest")
        if record.get("dtype") != values.dtype.str:
            raise ValueError(f"phase-{phase} source dtype disagrees with manifest")
        if record.get("mean_sha256") != sha256_array(means[int(phase)]):
            raise ValueError(f"phase-{phase} mean disagrees with manifest")
        if record.get("coordinate_variance_sha256") != sha256_array(
            variances[int(phase)]
        ):
            raise ValueError(f"phase-{phase} variance disagrees with manifest")
        samples.append(values)
        source_records.append(
            {
                "phase_offset": int(phase),
                "file": str(path),
                "sha256": str(actual_hash),
                "size_bytes": size_bytes,
                "shape": list(values.shape),
                "dtype": values.dtype.str,
                "mean_sha256": str(record["mean_sha256"]),
                "coordinate_variance_sha256": str(
                    record["coordinate_variance_sha256"]
                ),
            }
        )

    return DenseCovarianceSource(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        phase_offsets=phase_offsets,
        samples=tuple(samples),
        means=means,
        coordinate_variances=variances,
        source_records=tuple(source_records),
        moments_path=moments_path,
        moments_sha256=moments_sha256,
    )


def _build_binding(
    source: DenseCovarianceSource,
    *,
    block_rows: int,
    audit_vectors: int,
    audit_block_columns: int,
    source_hashes_recomputed: bool,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "schema_version": DENSE_COVARIANCE_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(script_path),
        "factor_manifest": str(source.manifest_path),
        "factor_manifest_sha256": source.manifest_sha256,
        "moments_file": str(source.moments_path),
        "moments_sha256": source.moments_sha256,
        "state_size": source.state_size,
        "sample_count_per_phase": source.sample_count,
        "phase_offsets": source.phase_offsets.tolist(),
        "phase_source_sha256": [
            record["sha256"] for record in source.source_records
        ],
        "block_rows": block_rows,
        "audit_vectors": audit_vectors,
        "audit_block_columns": audit_block_columns,
        "source_hashes_recomputed": source_hashes_recomputed,
        "dtype": FLOAT64_DTYPE.str,
        "order": "F",
        "triangle_accumulated": "upper",
        "phase_weight": 1.0 / source.phase_count,
        "sample_covariance_denominator": source.sample_count - 1,
    }


def _initial_progress(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DENSE_COVARIANCE_SCHEMA_VERSION,
        "status": PROGRESS_STATUS,
        "binding": binding,
        "created_utc": datetime.now(UTC).isoformat(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "completed_phase_count": 0,
        "completed_phase_offsets": [],
        "active_generation": None,
        "phase_checkpoints": [],
        "accumulation_runtime_seconds": 0.0,
    }


def _validate_progress(
    progress: dict[str, Any],
    *,
    binding: dict[str, Any],
    work_dir: Path,
    source: DenseCovarianceSource,
) -> dict[str, Any]:
    if progress.get("schema_version") != DENSE_COVARIANCE_SCHEMA_VERSION:
        raise ValueError("dense progress has an unsupported schema")
    if progress.get("status") not in (PROGRESS_STATUS, READY_STATUS):
        raise ValueError("dense progress has an unsupported status")
    if progress.get("binding") != binding:
        raise ValueError("dense resume binding differs from the requested build")
    completed = progress.get("completed_phase_count")
    if not isinstance(completed, int) or not 0 <= completed <= source.phase_count:
        raise ValueError("dense progress has an invalid completed phase count")
    if progress.get("completed_phase_offsets") != source.phase_offsets[
        :completed
    ].tolist():
        raise ValueError("dense progress phases are not a canonical prefix")
    checkpoints = progress.get("phase_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != completed:
        raise ValueError("dense progress checkpoint count is inconsistent")
    active = progress.get("active_generation")
    if completed == 0:
        if active is not None:
            raise ValueError("zero-phase progress unexpectedly names a generation")
    else:
        if not isinstance(active, dict) or "file" not in active:
            raise ValueError("dense progress lacks its active generation")
        active_path = (work_dir / str(active["file"])).resolve()
        if active_path.parent != work_dir.resolve():
            raise ValueError("dense progress generation leaves its work directory")
        _load_dense_array(active_path, state_size=source.state_size, mode="r")
        if active.get("sha256") != sha256_file(active_path):
            raise ValueError("dense checkpoint fails its recorded SHA-256")
        if active.get("completed_phase_count") != completed:
            raise ValueError("active generation phase count is inconsistent")
    return progress


def _write_progress(path: Path, progress: dict[str, Any]) -> None:
    document = dict(progress)
    document["updated_utc"] = datetime.now(UTC).isoformat()
    write_json(path, document, overwrite=True)
    progress.clear()
    progress.update(document)


def _create_zero_generation(path: Path, *, state_size: int) -> np.ndarray:
    covariance = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=FLOAT64_DTYPE,
        shape=(state_size, state_size),
        fortran_order=True,
    )
    covariance.fill(0.0)
    covariance.flush()
    return covariance


def _dsyrk_update(
    covariance: np.ndarray, centered_samples: np.ndarray, *, alpha: float
) -> None:
    """Update the upper triangle and require BLAS to use the memory map in place."""

    updated = dsyrk(
        alpha=alpha,
        a=centered_samples,
        beta=1.0,
        c=covariance,
        trans=1,
        lower=0,
        overwrite_c=1,
    )
    if not np.shares_memory(updated, covariance):
        raise RuntimeError("dsyrk copied the dense covariance instead of updating it")


def _accumulate_phase(
    covariance: np.ndarray,
    *,
    samples: np.ndarray,
    mean: np.ndarray,
    block_rows: int,
    alpha: float,
) -> None:
    for start in range(0, samples.shape[0], block_rows):
        stop = min(start + block_rows, samples.shape[0])
        centered = np.array(
            samples[start:stop], dtype=np.float64, order="F", copy=True
        )
        if not np.isfinite(centered).all():
            raise ValueError(
                f"native samples contain nonfinite values near row {start}"
            )
        centered -= mean
        _dsyrk_update(covariance, centered, alpha=alpha)


def _diagonal_audit(
    covariance: np.ndarray,
    *,
    expected_diagonal: np.ndarray,
    completed_phase_count: int,
) -> dict[str, Any]:
    actual = np.asarray(np.diag(covariance), dtype=np.float64)
    defect = np.abs(actual - expected_diagonal)
    maximum_absolute = float(np.max(defect, initial=0.0))
    scale = max(float(np.max(np.abs(expected_diagonal), initial=0.0)), 1.0)
    maximum_relative_to_global_scale = maximum_absolute / scale
    if maximum_absolute > 1.0e-11 + 5.0e-10 * scale:
        raise RuntimeError(
            "dense checkpoint diagonal disagrees with factor variances: "
            f"max_abs={maximum_absolute:.6e}"
        )
    return {
        "completed_phase_count": completed_phase_count,
        "expected_diagonal_sha256": sha256_array(expected_diagonal),
        "actual_diagonal_sha256": sha256_array(actual),
        "maximum_absolute_defect": maximum_absolute,
        "maximum_relative_to_global_scale": maximum_relative_to_global_scale,
        "trace": float(np.sum(actual, dtype=np.float64)),
    }


def _generation_path(work_dir: Path, completed_phase_count: int) -> Path:
    return work_dir / f"covariance.through-{completed_phase_count:02d}-phases.npy"


def _discard_unreferenced_work_files(
    work_dir: Path, *, active_file: str | None, keep_finalizing: bool
) -> None:
    keep = {active_file, PROGRESS_FILE_NAME}
    if keep_finalizing:
        keep.add(FINALIZING_FILE_NAME)
    for path in work_dir.iterdir():
        if path.name in keep:
            continue
        if path.is_file() and (
            path.name.startswith("covariance.through-")
            or path.name == FINALIZING_FILE_NAME
        ):
            path.unlink()


def _checkpoint_next_phase(
    *,
    source: DenseCovarianceSource,
    work_dir: Path,
    progress_path: Path,
    progress: dict[str, Any],
    block_rows: int,
) -> None:
    completed = int(progress["completed_phase_count"])
    if completed >= source.phase_count:
        return
    previous = progress.get("active_generation")
    previous_file = None if previous is None else str(previous["file"])
    next_count = completed + 1
    stage_path = _generation_path(work_dir, next_count)
    stage_path.unlink(missing_ok=True)
    if previous_file is None:
        covariance = _create_zero_generation(
            stage_path, state_size=source.state_size
        )
    else:
        previous_path = work_dir / previous_file
        shutil.copyfile(previous_path, stage_path)
        covariance = _load_dense_array(
            stage_path, state_size=source.state_size, mode="r+"
        )

    phase_started = time.perf_counter()
    alpha = 1.0 / (
        source.phase_count * (source.sample_count - 1)
    )
    _accumulate_phase(
        covariance,
        samples=source.samples[completed],
        mean=source.means[completed],
        block_rows=block_rows,
        alpha=alpha,
    )
    expected_diagonal = np.sum(
        source.coordinate_variances[:next_count], axis=0, dtype=np.float64
    ) / source.phase_count
    audit = _diagonal_audit(
        covariance,
        expected_diagonal=expected_diagonal,
        completed_phase_count=next_count,
    )
    covariance.flush()
    del covariance
    _fsync_file(stage_path)
    stage_hash = sha256_file(stage_path)
    elapsed = time.perf_counter() - phase_started
    audit.update(
        {
            "phase_offset": int(source.phase_offsets[completed]),
            "generation_sha256": stage_hash,
            "runtime_seconds": elapsed,
        }
    )
    checkpoints = list(progress["phase_checkpoints"])
    checkpoints.append(audit)
    progress.update(
        {
            "status": PROGRESS_STATUS,
            "completed_phase_count": next_count,
            "completed_phase_offsets": source.phase_offsets[:next_count].tolist(),
            "active_generation": {
                "file": stage_path.name,
                "sha256": stage_hash,
                "size_bytes": stage_path.stat().st_size,
                "shape": [source.state_size, source.state_size],
                "dtype": FLOAT64_DTYPE.str,
                "order": "F",
                "triangle_accumulated": "upper",
                "completed_phase_count": next_count,
            },
            "phase_checkpoints": checkpoints,
            "accumulation_runtime_seconds": float(
                progress["accumulation_runtime_seconds"]
            )
            + elapsed,
        }
    )
    # The new immutable generation is durable before the atomic progress update.
    _write_progress(progress_path, progress)
    if previous_file is not None and previous_file != stage_path.name:
        (work_dir / previous_file).unlink(missing_ok=True)
    _fsync_directory(work_dir)
    print(
        f"completed phase {int(source.phase_offsets[completed])} "
        f"({next_count}/{source.phase_count}) in {elapsed:.1f} s",
        flush=True,
    )


def _mirror_upper_triangle(
    covariance: np.ndarray, *, block_columns: int
) -> None:
    state_size = covariance.shape[0]
    for start in range(0, state_size, block_columns):
        stop = min(start + block_columns, state_size)
        if start:
            covariance[start:stop, :start] = covariance[
                :start, start:stop
            ].T
        diagonal = covariance[start:stop, start:stop]
        lower = np.tril_indices(stop - start, k=-1)
        diagonal[lower] = diagonal.T[lower]


def _dense_structure_audit(
    covariance: np.ndarray,
    *,
    expected_diagonal: np.ndarray,
    completed_phase_count: int,
    block_columns: int,
) -> dict[str, Any]:
    maximum_symmetry_defect = 0.0
    finite = True
    for start in range(0, covariance.shape[1], block_columns):
        stop = min(start + block_columns, covariance.shape[1])
        columns = np.asarray(covariance[:, start:stop])
        finite = finite and bool(np.isfinite(columns).all())
        transposed_rows = np.asarray(covariance[start:stop, :]).T
        maximum_symmetry_defect = max(
            maximum_symmetry_defect,
            float(np.max(np.abs(columns - transposed_rows), initial=0.0)),
        )
    if not finite:
        raise RuntimeError("dense covariance contains nonfinite values")
    if maximum_symmetry_defect != 0.0:
        raise RuntimeError("dense covariance is not exactly symmetric")
    diagonal_audit = _diagonal_audit(
        covariance,
        expected_diagonal=expected_diagonal,
        completed_phase_count=completed_phase_count,
    )
    diagonal = np.asarray(np.diag(covariance), dtype=np.float64)
    return {
        "all_values_finite": finite,
        "maximum_absolute_symmetry_defect": maximum_symmetry_defect,
        "minimum_diagonal": float(np.min(diagonal)),
        "maximum_diagonal": float(np.max(diagonal)),
        "trace": float(np.sum(diagonal, dtype=np.float64)),
        "diagonal": diagonal_audit,
    }


def _direct_application_audit(
    covariance: np.ndarray,
    *,
    source: DenseCovarianceSource,
    block_rows: int,
    vector_count: int,
    seed: int = 20260928,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    vectors = generator.standard_normal((source.state_size, vector_count))
    vectors /= np.linalg.norm(vectors, axis=0, keepdims=True)
    dense_products = np.asarray(covariance @ vectors, dtype=np.float64)
    factor_products = np.zeros_like(dense_products)
    alpha = 1.0 / (
        source.phase_count * (source.sample_count - 1)
    )
    for samples, mean in zip(source.samples, source.means, strict=True):
        for start in range(0, source.sample_count, block_rows):
            stop = min(start + block_rows, source.sample_count)
            centered = np.asarray(samples[start:stop], dtype=np.float64)
            centered -= mean
            projections = centered @ vectors
            factor_products += alpha * (centered.T @ projections)
    defect = dense_products - factor_products
    per_vector_absolute = np.linalg.norm(defect, axis=0)
    reference_norm = np.linalg.norm(factor_products, axis=0)
    per_vector_relative = per_vector_absolute / np.maximum(
        reference_norm, np.finfo(np.float64).tiny
    )
    maximum_relative = float(np.max(per_vector_relative))
    if maximum_relative > 2.0e-10:
        raise RuntimeError(
            "dense covariance fails its direct factor application audit: "
            f"max_relative={maximum_relative:.6e}"
        )
    quadratic_forms = np.einsum(
        "ij,ij->j", vectors, dense_products, optimize=True
    )
    quadratic_scale = max(float(np.max(np.abs(quadratic_forms))), 1.0)
    if float(np.min(quadratic_forms)) < -1.0e-11 * quadratic_scale:
        raise RuntimeError("dense covariance has a negative audited quadratic form")
    return {
        "random_seed": seed,
        "vector_count": vector_count,
        "vectors_sha256": sha256_array(vectors),
        "dense_products_sha256": sha256_array(dense_products),
        "factor_products_sha256": sha256_array(factor_products),
        "per_vector_absolute_l2_defect": per_vector_absolute.tolist(),
        "per_vector_relative_l2_defect": per_vector_relative.tolist(),
        "maximum_relative_l2_defect": maximum_relative,
        "quadratic_forms": quadratic_forms.tolist(),
        "minimum_quadratic_form": float(np.min(quadratic_forms)),
    }


def _ready_progress(
    *,
    source: DenseCovarianceSource,
    work_dir: Path,
    progress_path: Path,
    progress: dict[str, Any],
    block_rows: int,
    audit_vectors: int,
    audit_block_columns: int,
) -> None:
    active = progress.get("active_generation")
    if not isinstance(active, dict):
        raise RuntimeError("complete phase progress has no active generation")
    active_path = work_dir / str(active["file"])
    finalizing_path = work_dir / FINALIZING_FILE_NAME
    finalizing_path.unlink(missing_ok=True)
    shutil.copyfile(active_path, finalizing_path)
    covariance = _load_dense_array(
        finalizing_path, state_size=source.state_size, mode="r+"
    )
    finalization_started = time.perf_counter()
    _mirror_upper_triangle(covariance, block_columns=audit_block_columns)
    expected_diagonal = np.mean(
        source.coordinate_variances, axis=0, dtype=np.float64
    )
    structure_audit = _dense_structure_audit(
        covariance,
        expected_diagonal=expected_diagonal,
        completed_phase_count=source.phase_count,
        block_columns=audit_block_columns,
    )
    application_audit = _direct_application_audit(
        covariance,
        source=source,
        block_rows=block_rows,
        vector_count=audit_vectors,
    )
    covariance.flush()
    del covariance
    _fsync_file(finalizing_path)
    final_hash = sha256_file(finalizing_path)
    finalization_runtime = time.perf_counter() - finalization_started
    progress.update(
        {
            "status": READY_STATUS,
            "final_covariance": {
                "staging_file": finalizing_path.name,
                "file": FINAL_COVARIANCE_FILE_NAME,
                "sha256": final_hash,
                "size_bytes": finalizing_path.stat().st_size,
                "shape": [source.state_size, source.state_size],
                "dtype": FLOAT64_DTYPE.str,
                "order": "F",
            },
            "final_validation": {
                "structure": structure_audit,
                "direct_application": application_audit,
            },
            "finalization_runtime_seconds": finalization_runtime,
        }
    )
    _write_progress(progress_path, progress)


def _final_manifest(
    *,
    source: DenseCovarianceSource,
    output_dir: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    covariance = dict(progress["final_covariance"])
    covariance.pop("staging_file", None)
    phase_sources = []
    for record in source.source_records:
        published = dict(record)
        published["file"] = _relative(Path(str(record["file"])), output_dir)
        phase_sources.append(published)
    return {
        "schema_version": DENSE_COVARIANCE_SCHEMA_VERSION,
        "status": DENSE_COVARIANCE_STATUS,
        "script": {
            "version": SCRIPT_VERSION,
            "file": _relative(Path(__file__).resolve(), output_dir),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "completed_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(progress["accumulation_runtime_seconds"])
        + float(progress.get("finalization_runtime_seconds", 0.0)),
        "dimensions": {
            "state_size": source.state_size,
            "sample_count_per_phase": source.sample_count,
            "phase_count": source.phase_count,
            "phase_offsets": source.phase_offsets.tolist(),
        },
        "covariance": covariance,
        "common_covariance_policy": source.manifest.get(
            "common_covariance_policy", "equal-phase pooled covariance"
        ),
        "source_capture": source.manifest.get("source_capture"),
        "mathematical_definition": {
            "phase_mean": "mu_p=(1/n) sum_y X_{p,y}",
            "phase_covariance": (
                "C_p=(X_p-1 mu_p^T)^T (X_p-1 mu_p^T)/(n-1)"
            ),
            "pooled_covariance": "C=(1/q) sum_p C_p",
            "q": source.phase_count,
            "n": source.sample_count,
            "centering": (
                "a separate empirical mean is removed from each annual phase "
                "before equal-phase pooling"
            ),
            "implementation": (
                "float64 upper-triangle scipy BLAS dsyrk accumulation, followed "
                "by exact blockwise mirroring into a Fortran-order dense matrix"
            ),
        },
        "source_factor_manifest": {
            "file": _relative(source.manifest_path, output_dir),
            "sha256": source.manifest_sha256,
            "schema_version": source.manifest.get("schema_version"),
            "status": source.manifest.get("status"),
            "moments_file": _relative(source.moments_path, output_dir),
            "moments_sha256": source.moments_sha256,
            "source_capture": source.manifest.get("source_capture"),
        },
        "phase_sources": phase_sources,
        "build": {
            "block_rows": progress["binding"]["block_rows"],
            "phase_weight": progress["binding"]["phase_weight"],
            "sample_covariance_denominator": progress["binding"][
                "sample_covariance_denominator"
            ],
            "source_hashes_recomputed": progress["binding"][
                "source_hashes_recomputed"
            ],
            "transactional_checkpoint_unit": "one completed annual phase",
            "phase_checkpoints": progress["phase_checkpoints"],
        },
        "validation": progress["final_validation"],
    }


def _publish_ready(
    *,
    source: DenseCovarianceSource,
    output_dir: Path,
    work_dir: Path,
    progress: dict[str, Any],
) -> Path:
    covariance_record = progress.get("final_covariance")
    if not isinstance(covariance_record, dict):
        raise RuntimeError("ready progress lacks the final covariance record")
    expected_hash = covariance_record.get("sha256")
    final_path = output_dir / FINAL_COVARIANCE_FILE_NAME
    staging_path = work_dir / str(covariance_record["staging_file"])
    if final_path.exists():
        _load_dense_array(final_path, state_size=source.state_size, mode="r")
        if sha256_file(final_path) != expected_hash:
            raise ValueError("orphaned final covariance fails its ready SHA-256")
    else:
        if not staging_path.is_file():
            raise FileNotFoundError(staging_path)
        if sha256_file(staging_path) != expected_hash:
            raise ValueError("finalizing covariance fails its ready SHA-256")
        os.replace(staging_path, final_path)
        _fsync_directory(output_dir)

    manifest_path = output_dir / FINAL_MANIFEST_FILE_NAME
    manifest = _final_manifest(
        source=source, output_dir=output_dir, progress=progress
    )
    if manifest_path.exists():
        if load_json(manifest_path) != manifest:
            raise FileExistsError(
                f"different final manifest already exists: {manifest_path}"
            )
    else:
        write_json(manifest_path, manifest, overwrite=False)
    return manifest_path


def _remove_prior_artifacts(output_dir: Path, work_dir: Path) -> None:
    for name in (FINAL_MANIFEST_FILE_NAME, FINAL_COVARIANCE_FILE_NAME):
        (output_dir / name).unlink(missing_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    factor_manifest = _resolve(args.factor_manifest)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / FINAL_MANIFEST_FILE_NAME
    covariance_path = output_dir / FINAL_COVARIANCE_FILE_NAME
    work_dir = output_dir / WORK_DIRECTORY_NAME
    progress_path = work_dir / PROGRESS_FILE_NAME
    if factor_manifest == manifest_path:
        raise ValueError("factor and dense manifests may not be the same file")

    if args.overwrite:
        _remove_prior_artifacts(output_dir, work_dir)
    elif manifest_path.exists():
        if args.resume:
            manifest = load_json(manifest_path)
            record = manifest.get("covariance")
            if (
                manifest.get("status") == DENSE_COVARIANCE_STATUS
                and isinstance(record, dict)
                and covariance_path.is_file()
                and sha256_file(covariance_path) == record.get("sha256")
            ):
                print(manifest_path)
                return 0
        raise FileExistsError(
            f"dense covariance manifest already exists: {manifest_path}"
        )
    elif covariance_path.exists() and not args.resume:
        raise FileExistsError(
            "unpublished covariance exists; use --resume to validate/publish it "
            "or --overwrite to discard it"
        )

    source = _load_source(
        factor_manifest,
        expected_state_size=args.expected_state_size,
        expected_samples=args.expected_samples,
        expected_phase_count=args.expected_phase_count,
        verify_source_hashes=not args.skip_source_hashes,
    )
    binding = _build_binding(
        source,
        block_rows=args.block_rows,
        audit_vectors=args.audit_vectors,
        audit_block_columns=args.audit_block_columns,
        source_hashes_recomputed=not args.skip_source_hashes,
    )

    if args.resume:
        if not progress_path.is_file():
            raise FileNotFoundError("--resume requested but no dense progress exists")
        progress = _validate_progress(
            load_json(progress_path),
            binding=binding,
            work_dir=work_dir,
            source=source,
        )
    else:
        if work_dir.exists():
            raise FileExistsError(
                f"dense work directory exists; use --resume or --overwrite: {work_dir}"
            )
        work_dir.mkdir(parents=False)
        progress = _initial_progress(binding)
        _write_progress(progress_path, progress)

    active = progress.get("active_generation")
    active_file = None if active is None else str(active["file"])
    _discard_unreferenced_work_files(
        work_dir,
        active_file=active_file,
        keep_finalizing=progress["status"] == READY_STATUS,
    )

    if progress["status"] == PROGRESS_STATUS:
        while int(progress["completed_phase_count"]) < source.phase_count:
            _checkpoint_next_phase(
                source=source,
                work_dir=work_dir,
                progress_path=progress_path,
                progress=progress,
                block_rows=args.block_rows,
            )
        _ready_progress(
            source=source,
            work_dir=work_dir,
            progress_path=progress_path,
            progress=progress,
            block_rows=args.block_rows,
            audit_vectors=args.audit_vectors,
            audit_block_columns=args.audit_block_columns,
        )

    manifest_path = _publish_ready(
        source=source,
        output_dir=output_dir,
        work_dir=work_dir,
        progress=progress,
    )
    loaded = DenseNativeCovarianceOperator.load(manifest_path, verify_hashes=True)
    if loaded.state_size != source.state_size or loaded.order != "F":
        raise RuntimeError("published dense covariance failed its loader audit")
    shutil.rmtree(work_dir)
    _fsync_directory(output_dir)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
