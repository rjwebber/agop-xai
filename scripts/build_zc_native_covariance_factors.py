#!/usr/bin/env python3
"""Publish exact matrix-free empirical covariances of native ZC states.

For every requested seasonal phase this script centers the cached native-state
samples and exposes their rectangular sample factor.  The sample arrays are
referenced in place; no 28,591-by-28,591 covariance matrix is materialized or
copied.  The common covariance is the equal-phase average after removing a
separate mean at every phase.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.io import (  # noqa: E402
    load_json,
    sha256_array,
    sha256_file,
    write_json,
    write_npz,
)
from zc_xai.native_covariance import (  # noqa: E402
    NATIVE_COVARIANCE_SCHEMA_VERSION,
    NATIVE_COVARIANCE_STATUS,
    CenteredSampleCovarianceFactor,
    NativeCovarianceBundle,
    PooledCenteredSampleCovarianceFactor,
)

SCRIPT_VERSION = "1.2.0"
DEFAULT_CACHE_DIR = Path(
    "outputs/zc_native_phase_cache/training-years10000-phases00-35"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/zc_native_covariance/training-years10000-phases00-35"
)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--phase-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    result.add_argument(
        "--capture-manifest",
        type=Path,
        help="Defaults to <phase-cache-dir>/capture_complete.json.",
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument(
        "--expected-samples",
        type=positive_int,
        default=10_000,
        help="Fail closed unless the cache contains this many trajectories.",
    )
    result.add_argument(
        "--block-rows",
        type=positive_int,
        default=128,
        help="Rows converted to float64 at once while accumulating moments.",
    )
    result.add_argument(
        "--skip-cache-hashes",
        action="store_true",
        help="Skip expensive source-array hashes (not recommended for publication).",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), start=parent.resolve())


def _phase_records(
    capture: dict[str, Any], cache_dir: Path
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    raw_offsets = capture.get("phase_offsets")
    raw_records = capture.get("cache_files")
    if not isinstance(raw_offsets, list) or not isinstance(raw_records, list):
        raise ValueError("capture manifest lacks phase_offsets/cache_files")
    offsets = np.asarray(raw_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size == 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("capture phase offsets must be nonempty and increasing")
    if len(raw_records) != offsets.size:
        raise ValueError("capture cache-file count does not match phase offsets")

    by_offset: dict[int, dict[str, Any]] = {}
    for position, raw in enumerate(raw_records):
        if not isinstance(raw, dict) or "file" not in raw:
            raise ValueError("malformed cache-file record")
        inferred = int(raw.get("phase_offset", offsets[position]))
        if inferred in by_offset:
            raise ValueError(f"duplicate cache record for phase {inferred}")
        record = dict(raw)
        record["phase_offset"] = inferred
        record["source_path"] = (cache_dir / str(record["file"])).resolve()
        by_offset[inferred] = record
    if set(by_offset) != set(int(value) for value in offsets):
        raise ValueError("capture cache files do not cover the declared phases")
    return offsets, [by_offset[int(offset)] for offset in offsets]


def _validate_annual_capture(
    capture: dict[str, Any], offsets: np.ndarray, *, expected_samples: int
) -> None:
    """Fail closed on a supported 10,000-year scientific sampling schedule.

    The legacy event-window products contain nine phases.  The common
    covariance product contains every one of the 36 within-year phases.  Both
    remain loadable so old artifacts can still be audited, while the command
    defaults now select the all-phase product.
    """

    if expected_samples != 10_000:
        return
    is_legacy_window = offsets.size == 9
    is_all_phase = np.array_equal(offsets, np.arange(36, dtype=np.int64))
    if (
        offsets.ndim != 1
        or not (is_legacy_window or is_all_phase)
        or np.any(np.diff(offsets) <= 0)
        or int(offsets[0]) < 0
        or int(offsets[-1]) >= 36
    ):
        raise ValueError(
            "annual covariance requires either exactly nine increasing phase "
            "offsets or all offsets 0 through 35"
        )
    expected_bases = np.arange(0, 360_000, 36, dtype=np.int64)
    expected_phase_indices = expected_bases[None, :] + offsets[:, None]
    required = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": 10_000,
        "state_size": 28_591,
        "dtype": "<f4",
        "training_interval": [0, 360_000],
        "base_input_indices_sha256": sha256_array(expected_bases),
        "phase_input_indices_sha256": sha256_array(expected_phase_indices),
    }
    for key, expected in required.items():
        if capture.get(key) != expected:
            raise ValueError(f"annual capture disagrees with required {key}")
    schedule = capture.get("schedule")
    expected_schedule = {
        "source_checkpoint_count": 1_000,
        "source_checkpoint_stride_steps": 360,
        "annual_base_stride_steps": 36,
        "years_per_source_checkpoint": 10,
        "maximum_replay_offset_steps": 9 * 36 + int(offsets[-1]),
    }
    if not isinstance(schedule, dict):
        raise ValueError("annual capture has no sampling schedule")
    for key, expected in expected_schedule.items():
        if schedule.get(key) != expected:
            raise ValueError(f"annual capture schedule disagrees for {key}")
    overlap = capture.get("overlap_audit")
    if not isinstance(overlap, dict):
        raise ValueError("annual capture lacks the exhaustive decadal overlap audit")
    overlap_status = overlap.get("status")
    phase_count = int(offsets.size)
    expected_overlap_rows = phase_count * 1_000
    expected_overlap_values = expected_overlap_rows * 28_591
    common_overlap_valid = (
        overlap.get("rows_per_phase") == 1_000
        and overlap.get("total_values_compared") == expected_overlap_values
    )
    if overlap_status == "passed_full_bitwise_overlap":
        overlap_valid = common_overlap_valid
    elif overlap_status == "passed_full_bitwise_independent_overlap_replay":
        replay_hashes = overlap.get("replayed_block_sha256")
        provenance = overlap.get("provenance")
        storage = overlap.get("storage")
        expected_block_keys = {str(index) for index in range(1_000)}
        overlap_valid = (
            common_overlap_valid
            and overlap.get("schema_version") == 1
            and overlap.get("validation_mode") == "independent_overlap_replay"
            and overlap.get("reference_cache_used") is False
            and overlap.get("phase_offsets") == offsets.tolist()
            and overlap.get("total_rows_compared") == expected_overlap_rows
            and overlap.get("source_checkpoint_count") == 1_000
            and overlap.get("replay_relative_offsets_sha256")
            == sha256_array(offsets[None, :])
            and isinstance(replay_hashes, dict)
            and set(replay_hashes) == expected_block_keys
            and all(
                isinstance(value, str) and len(value) == 64
                for value in replay_hashes.values()
            )
            and isinstance(overlap.get("ordered_block_hashes_sha256"), str)
            and len(overlap["ordered_block_hashes_sha256"]) == 64
            and isinstance(provenance, dict)
            and provenance.get("history_sha256") == capture.get("history_sha256")
            and provenance.get("phase_replay_executable_sha256")
            == capture.get("phase_replay_executable_sha256")
            and isinstance(storage, dict)
            and storage.get("persistent_reference_cache_created") is False
            and storage.get("persistent_reference_cache_bytes") == 0
        )
    else:
        overlap_valid = False
    if not overlap_valid:
        raise ValueError("annual capture lacks the exhaustive decadal overlap audit")
    for key in (
        "history_sha256",
        "production_report_sha256",
        "phase_replay_executable_sha256",
        "state_manifest_sha256",
        "data_metadata_sha256",
    ):
        value = capture.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"annual capture lacks a valid {key}")


def _moments(
    samples: np.ndarray, *, block_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    n, d = samples.shape
    total = np.zeros(d, dtype=np.float64)
    for start in range(0, n, block_rows):
        block = np.asarray(samples[start : start + block_rows], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError(f"native cache contains nonfinite values near row {start}")
        total += np.sum(block, axis=0, dtype=np.float64)
    mean = total / n
    centered_squares = np.zeros(d, dtype=np.float64)
    for start in range(0, n, block_rows):
        block = np.asarray(samples[start : start + block_rows], dtype=np.float64)
        block -= mean
        centered_squares += np.einsum("ij,ij->j", block, block, optimize=True)
    variance = centered_squares / (n - 1)
    if np.any(variance < 0.0) or not np.isfinite(variance).all():
        raise RuntimeError("native coordinate variances are not finite/nonnegative")
    return mean, variance


def _transpose_audits(
    factors: tuple[CenteredSampleCovarianceFactor, ...], *, seed: int = 20260907
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """Test every F/F.T pair and reuse the products for the pooled audit."""

    generator = np.random.default_rng(seed)
    covector = generator.standard_normal(factors[0].state_size)
    covector /= np.linalg.norm(covector)
    phase_rows: list[dict[str, float | int]] = []
    phase_lhs: list[float] = []
    phase_rhs: list[float] = []
    projected_blocks: list[np.ndarray] = []
    for factor in factors:
        coefficients = generator.standard_normal(factor.coefficient_size)
        coefficients = factor.project_coefficients(coefficients)
        applied = factor.apply(coefficients)
        transposed = factor.transpose(covector)
        lhs = float(covector @ applied)
        rhs = float(coefficients @ transposed)
        scale = max(abs(lhs), abs(rhs), np.finfo(np.float64).tiny)
        phase_lhs.append(lhs)
        phase_rhs.append(rhs)
        projected_blocks.append(coefficients)
        phase_rows.append(
            {
                "phase_offset": factor.phase_offset,
                "absolute_defect": abs(lhs - rhs),
                "relative_defect": abs(lhs - rhs) / scale,
            }
        )

    pooled = PooledCenteredSampleCovarianceFactor(factors)
    pooled_coefficients = np.concatenate(projected_blocks)
    # These are algebraically the pooled apply/transpose inner products and
    # avoid rereading all nine potentially 1.1 GB arrays a second time.
    phase_scale = 1.0 / math.sqrt(len(factors))
    lhs = phase_scale * math.fsum(phase_lhs)
    rhs = phase_scale * math.fsum(phase_rhs)
    scale = max(abs(lhs), abs(rhs), np.finfo(np.float64).tiny)
    projected = pooled.project_coefficients(pooled_coefficients)
    projection_defect = float(np.max(np.abs(projected - pooled_coefficients)))
    pooled_row: dict[str, float | int] = {
        "phase_count": len(factors),
        "coefficient_size": pooled.coefficient_size,
        "absolute_defect": abs(lhs - rhs),
        "relative_defect": abs(lhs - rhs) / scale,
        "coefficient_projection_max_abs_defect": projection_defect,
    }
    return phase_rows, pooled_row


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.perf_counter()
    cache_dir = _resolve(args.phase_cache_dir)
    capture_path = _resolve(
        args.capture_manifest or cache_dir / "capture_complete.json"
    )
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    moments_path = output_dir / "phase_moments.npz"
    if not args.overwrite:
        for path in (manifest_path, moments_path):
            if path.exists():
                raise FileExistsError(f"output exists; use --overwrite: {path}")

    capture = load_json(capture_path)
    capture_schema = capture.get("schema_version")
    capture_status = capture.get("status")
    if capture_schema is not None and capture_schema != 1:
        raise ValueError("capture manifest has an unsupported schema")
    if capture_status is not None and capture_status != "complete":
        raise ValueError("native phase capture is not complete")
    if args.expected_samples == 10_000 and (
        capture_schema != 1 or capture_status != "complete"
    ):
        raise ValueError(
            "the annual scientific cache needs a complete schema-1 manifest"
        )
    offsets, records = _phase_records(capture, cache_dir)
    _validate_annual_capture(
        capture, offsets, expected_samples=args.expected_samples
    )
    declared_count = capture.get("sample_count", capture.get("checkpoint_count"))
    if declared_count is not None and int(declared_count) != args.expected_samples:
        raise ValueError(
            f"capture declares {declared_count} samples, expected "
            f"{args.expected_samples}"
        )

    samples_by_phase: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    state_size: int | None = None
    for offset, record in zip(offsets, records, strict=True):
        path = Path(record["source_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if "size_bytes" in record and int(record["size_bytes"]) != actual_size:
            raise ValueError(f"native cache size changed: {path}")
        actual_hash = None
        if not args.skip_cache_hashes:
            actual_hash = sha256_file(path)
            if record.get("sha256") is not None and actual_hash != record["sha256"]:
                raise ValueError(f"native cache hash changed: {path}")
        elif record.get("sha256") is not None:
            actual_hash = str(record["sha256"])

        samples = np.load(path, mmap_mode="r", allow_pickle=False)
        if samples.ndim != 2 or samples.shape[0] != args.expected_samples:
            raise ValueError(
                f"native phase {offset} has shape {samples.shape}, expected "
                f"({args.expected_samples}, d)"
            )
        if samples.dtype != np.dtype("<f4"):
            raise TypeError(f"native phase {offset} must be little-endian float32")
        if state_size is None:
            state_size = int(samples.shape[1])
        if samples.shape[1] != state_size:
            raise ValueError("native cache phases have different state dimensions")
        if "shape" in record and list(samples.shape) != list(record["shape"]):
            raise ValueError(f"native cache shape disagrees with manifest: {path}")
        if "dtype" in record and samples.dtype.str != record["dtype"]:
            raise ValueError(f"native cache dtype disagrees with manifest: {path}")
        mean, variance = _moments(samples, block_rows=args.block_rows)
        samples_by_phase.append(samples)
        means.append(mean)
        variances.append(variance)
        source_records.append(
            {
                "phase_offset": int(offset),
                "file": _relative(path, output_dir),
                "sha256": actual_hash,
                "size_bytes": actual_size,
                "shape": list(samples.shape),
                "dtype": samples.dtype.str,
                "mean_sha256": sha256_array(mean),
                "coordinate_variance_sha256": sha256_array(variance),
                "covariance_trace": float(np.sum(variance, dtype=np.float64)),
                "zero_coordinate_variance_count": int(
                    np.count_nonzero(variance == 0.0)
                ),
                "factor_rank_upper_bound": min(
                    args.expected_samples - 1, int(samples.shape[1])
                ),
            }
        )

    assert state_size is not None
    means_array = np.stack(means)
    variances_array = np.stack(variances)
    pooled_variance = np.mean(variances_array, axis=0, dtype=np.float64)
    write_npz(
        moments_path,
        overwrite=args.overwrite,
        compressed=True,
        phase_offsets=offsets,
        means=means_array,
        coordinate_variances=variances_array,
        pooled_coordinate_variance=pooled_variance,
    )

    factors = tuple(
        CenteredSampleCovarianceFactor(
            samples=samples,
            mean=mean,
            phase_offset=int(offset),
            source_path=Path(record["source_path"]),
        )
        for samples, mean, offset, record in zip(
            samples_by_phase, means, offsets, records, strict=True
        )
    )
    phase_audits, pooled_audit = _transpose_audits(factors)
    script_path = Path(__file__).resolve()
    capture_hash = sha256_file(capture_path)
    moments_hash = sha256_file(moments_path)
    phase_count = int(offsets.size)
    phase_nullity_lower_bound = 1
    pooled_coefficient_size = phase_count * args.expected_samples
    pooled_rank_upper_bound = min(
        phase_count * (args.expected_samples - 1), state_size
    )
    manifest: dict[str, Any] = {
        "schema_version": NATIVE_COVARIANCE_SCHEMA_VERSION,
        "status": NATIVE_COVARIANCE_STATUS,
        "script": {
            "version": SCRIPT_VERSION,
            "file": _relative(script_path, output_dir),
            "sha256": sha256_file(script_path),
        },
        "completed_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "dimensions": {
            "state_size": state_size,
            "sample_count_per_phase": args.expected_samples,
            "phase_count": phase_count,
            "phase_offsets": offsets.tolist(),
            "phase_coefficient_size": args.expected_samples,
            "phase_rank_upper_bound": min(args.expected_samples - 1, state_size),
            "phase_known_centering_nullity": phase_nullity_lower_bound,
            "pooled_coefficient_size": pooled_coefficient_size,
            "pooled_rank_upper_bound": pooled_rank_upper_bound,
        },
        "mathematical_definition": {
            "phase_factor": "F_k=(X_k-1 mu_k^T)^T/sqrt(n-1)",
            "phase_covariance": "C_k=F_k F_k^T",
            "pooled_factor": "F_pool=[F_1 ... F_q]/sqrt(q)",
            "pooled_covariance": "C_pool=F_pool F_pool^T=(1/q) sum_k C_k",
            "centering": (
                "a distinct empirical native-state mean mu_k is removed at "
                "every phase before pooling"
            ),
            "action": "min_{F u=a} ||u||_2^2 = a^T C^+ a for a in range(C)",
            "implementation": (
                "memory-mapped centered sample factors; no dense covariance is formed"
            ),
        },
        "common_covariance_policy": "equal-phase pooled covariance",
        "phase_factors": source_records,
        "means_and_variances": {
            "file": moments_path.name,
            "sha256": moments_hash,
            "size_bytes": moments_path.stat().st_size,
            "means_sha256": sha256_array(means_array),
            "coordinate_variances_sha256": sha256_array(variances_array),
            "pooled_coordinate_variance_sha256": sha256_array(pooled_variance),
            "pooled_covariance_trace": float(
                np.sum(pooled_variance, dtype=np.float64)
            ),
        },
        "source_capture": {
            "manifest_file": _relative(capture_path, output_dir),
            "manifest_sha256": capture_hash,
            "manifest_schema_version": capture.get("schema_version"),
            "manifest_status": capture.get("status"),
            "history_sha256": capture.get("history_sha256"),
            "production_report_sha256": capture.get("production_report_sha256"),
            "phase_replay_executable_sha256": capture.get(
                "phase_replay_executable_sha256"
            ),
            "state_manifest_sha256": capture.get("state_manifest_sha256"),
            "data_metadata_sha256": capture.get("data_metadata_sha256"),
            "normalization_sha256": capture.get("normalization_sha256"),
            "base_input_indices_sha256": capture.get(
                "base_input_indices_sha256"
            ),
            "phase_input_indices_sha256": capture.get(
                "phase_input_indices_sha256"
            ),
            "training_half_open_interval": capture.get(
                "training_interval", capture.get("training_half_open_interval")
            ),
            "sampling_schedule": capture.get(
                "schedule", capture.get("sampling_schedule")
            ),
            "source_manifest_sha256": capture.get("source_manifest_sha256"),
            "reference_cache_manifest_sha256": capture.get(
                "reference_cache_manifest_sha256"
            ),
            "overlap_audit": capture.get("overlap_audit"),
        },
        "validation": {
            "random_seed": 20260907,
            "phase_transpose_inner_product_audits": phase_audits,
            "pooled_transpose_inner_product_audit": pooled_audit,
            "source_hashes_recomputed": not args.skip_cache_hashes,
        },
        "limitations": [
            (
                "The empirical covariance is singular whenever sample count does "
                "not exceed state dimension; action is the Moore-Penrose "
                "pseudoinverse action on its sampled range."
            ),
            (
                "The sample-coordinate factor contains one known all-ones null "
                "per centered phase; minimum-norm optimization removes these "
                "null components automatically."
            ),
            (
                "Other linear dependencies may exist; callers must use "
                "minimum-norm solvers rather than invert a sample Gram matrix "
                "blindly."
            ),
        ],
    }
    write_json(manifest_path, manifest, overwrite=args.overwrite)
    # Exercise the public loading API against the just-published artifact.
    # Source hashes were already recomputed above.  Avoid rereading the entire
    # 9.59 GiB annual cache solely to repeat the same hashes during this reload.
    loaded = NativeCovarianceBundle.load(manifest_path, verify_hashes=False)
    if loaded.phase_offsets.tolist() != offsets.tolist():
        raise RuntimeError("published native covariance bundle failed reload audit")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
