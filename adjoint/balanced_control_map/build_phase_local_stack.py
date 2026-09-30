#!/usr/bin/env python3
"""Build coherent training-only ZC control maps for phases 26 through 34.

Each ten-year production checkpoint is advanced once to the last requested
phase while complete restart states are retained at every intermediate step.
One pooled sample-space EOF basis is then fitted across all requested phases.
Phase-specific native and core4 regressions use that same basis, avoiding the
arbitrary rotations that would result from stacking separate PCA fits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import generate_fresh_zc_dataset as fresh_generator  # noqa: E402

from adjoint.balanced_control_map import (  # noqa: E402
    PhaseLocalControlStack,
    fit_phase_local_control_stack,
    load_independent_control_layout,
    read_restart_payload,
)
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    load_json,
    sha256_array,
    sha256_file,
    write_json,
)

SCRIPT_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1
DEFAULT_PHASE_OFFSETS = tuple(range(26, 35))
CORE4_FIELDS = (
    "sst_anomaly",
    "thermocline_depth",
    "zonal_ocean_current",
    "meridional_ocean_current",
)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def validation_fraction(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 < value < 0.5:
        raise argparse.ArgumentTypeError("value must lie strictly between 0 and 0.5")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument(
        "--history",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3/runs/production/outhst"),
    )
    result.add_argument(
        "--production-report",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3/production_report.json"),
    )
    result.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    result.add_argument(
        "--normalization",
        type=Path,
        default=Path(
            "artifacts/zc-v3/models/core4/cnn/lead-10m/years-10000/"
            "seed-000042/normalization.npz"
        ),
    )
    result.add_argument(
        "--state-manifest",
        type=Path,
        default=Path("adjoint/fortran_kernel/state_manifest.json"),
    )
    result.add_argument(
        "--source-dir",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3/source"),
    )
    result.add_argument(
        "--executable",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3/source/zeqfc1"),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/zc_balanced_control_map/"
            "training-phases26-34-common-rank36"
        ),
    )
    result.add_argument(
        "--phase-offsets",
        nargs="+",
        type=int,
        default=DEFAULT_PHASE_OFFSETS,
        help="Strictly increasing offsets from each ten-year checkpoint.",
    )
    result.add_argument("--rank", type=positive_int, default=36)
    result.add_argument(
        "--jobs", type=positive_int, default=min(4, os.cpu_count() or 1)
    )
    result.add_argument(
        "--validation-fraction", type=validation_fraction, default=0.2
    )
    result.add_argument("--ridge-fraction", type=float, default=1.0e-3)
    result.add_argument(
        "--keep-native-cache",
        action="store_true",
        help="Keep the roughly 1 GB intermediate phase-state cache after success.",
    )
    result.add_argument(
        "--reuse-native-cache",
        action="store_true",
        help=(
            "Reuse a source-bound completed cache.  An older unmanifested cache "
            "is accepted only after schema, finiteness, and exact replay audits."
        ),
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _selected_checkpoints(
    history: Path,
    *,
    production_report: dict[str, Any],
    training_interval: tuple[int, int],
    phase_offsets: np.ndarray,
    layout: Any,
) -> tuple[np.ndarray, np.ndarray]:
    chunk_bytes = int(production_report["checkpoint_chunk_bytes"])
    checkpoint_count = int(production_report["checkpoint_count"])
    spinup_steps = int(production_report["configuration"]["spinup_steps"])
    expected_size = chunk_bytes * checkpoint_count
    if history.stat().st_size != expected_size:
        raise ValueError("production history byte count disagrees with its report")
    if sha256_file(history) != production_report["history_sha256"]:
        raise ValueError("production history fails its recorded SHA-256")

    selected_chunks: list[int] = []
    base_indices: list[int] = []
    lower, upper = training_interval
    with history.open("rb") as stream:
        for chunk_index in range(checkpoint_count):
            payload = stream.read(chunk_bytes)
            if len(payload) != chunk_bytes:
                raise RuntimeError("production history is truncated")
            sample = read_restart_payload(payload, layout)
            base = sample.nt - spinup_steps
            indices = base + phase_offsets
            if np.all((indices >= lower) & (indices < upper)):
                selected_chunks.append(chunk_index)
                base_indices.append(base)
    chunks = np.asarray(selected_chunks, dtype=np.int64)
    bases = np.asarray(base_indices, dtype=np.int64)
    if chunks.size < 2 or np.any(np.diff(bases) <= 0):
        raise RuntimeError("too few ordered common-phase training checkpoints")
    stride = int(production_report["configuration"]["checkpoint_steps"])
    if np.any(np.diff(bases) != stride):
        raise RuntimeError("selected checkpoints do not retain the production stride")
    return chunks, bases


def _capture_native_phases(
    *,
    history: Path,
    chunk_bytes: int,
    chunk_indices: np.ndarray,
    base_indices: np.ndarray,
    phase_offsets: np.ndarray,
    layout: Any,
    source_dir: Path,
    executable: Path,
    cache_dir: Path,
    jobs: int,
) -> tuple[list[np.memmap], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [
        cache_dir / f"native_phase_{int(offset):02d}.npy"
        for offset in phase_offsets
    ]
    for path in cache_paths:
        if path.exists():
            raise FileExistsError(
                f"incomplete native cache already exists; use --overwrite: {path}"
            )
    arrays = [
        np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(chunk_indices.size, layout.compact_size),
        )
        for path in cache_paths
    ]
    maximum_offset = int(phase_offsets[-1])

    def replay_one(item: tuple[int, int, int]) -> tuple[int, np.ndarray, float]:
        row, chunk_index, base_index = item
        with history.open("rb") as stream:
            stream.seek(chunk_index * chunk_bytes)
            payload = stream.read(chunk_bytes)
        if len(payload) != chunk_bytes:
            raise RuntimeError(f"could not read production checkpoint {chunk_index}")
        sparse = read_restart_payload(payload, layout)
        with tempfile.TemporaryDirectory(prefix="zc-common-phase-") as temporary:
            root = Path(temporary)
            restart = root / "sparse.hst"
            restart.write_bytes(payload)
            run_dir = root / "run"
            fresh_generator.prepare_run(
                source_dir,
                executable,
                run_dir,
                nstart=3,
                tfind=fresh_generator.model_time(sparse.nt),
                tzero=fresh_generator.model_time(sparse.nt),
                tend=fresh_generator.model_time(sparse.nt + maximum_offset),
                ntape=1,
                nrewnd=10,
                nic=0,
                write_start=(
                    fresh_generator.model_time(sparse.nt + maximum_offset) + 1.0
                ),
                write_end=(
                    fresh_generator.model_time(sparse.nt + maximum_offset) + 1.0
                ),
                restart=restart,
            )
            runtime = fresh_generator.run_model(run_dir)
            output = run_dir / "outhst"
            expected_bytes = (maximum_offset + 1) * chunk_bytes
            if output.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"multi-phase replay produced {output.stat().st_size} bytes; "
                    f"expected {expected_bytes}"
                )
            samples = np.empty(
                (phase_offsets.size, layout.compact_size), dtype=np.float32
            )
            with output.open("rb") as stream:
                for position, offset in enumerate(phase_offsets):
                    stream.seek(int(offset) * chunk_bytes)
                    phase_payload = stream.read(chunk_bytes)
                    phase_sample = read_restart_payload(phase_payload, layout)
                    if phase_sample.nt != sparse.nt + int(offset):
                        raise RuntimeError("phase replay produced an incorrect clock")
                    samples[position] = phase_sample.native_controls.astype(np.float32)
        if base_index + maximum_offset < 0:
            raise RuntimeError("invalid retained input index")
        return row, samples, float(runtime["elapsed_seconds"])

    work = [
        (row, int(chunk), int(base))
        for row, (chunk, base) in enumerate(
            zip(chunk_indices, base_indices, strict=True)
        )
    ]
    model_seconds: list[float] = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        for row, samples, elapsed in executor.map(replay_one, work):
            for position, array in enumerate(arrays):
                array[row] = samples[position]
            model_seconds.append(elapsed)
    wall_seconds = time.monotonic() - started
    for array in arrays:
        array.flush()
    del arrays
    reopened = [
        np.load(path, mmap_mode="r", allow_pickle=False) for path in cache_paths
    ]
    report = {
        "method": (
            "one authentic replay per sparse checkpoint with NTAPE=1, retaining "
            "the requested intermediate restart records"
        ),
        "checkpoint_count": int(chunk_indices.size),
        "phase_offsets": phase_offsets.tolist(),
        "chunk_indices_sha256": sha256_array(chunk_indices),
        "base_input_indices_sha256": sha256_array(base_indices),
        "history_sha256": sha256_file(history),
        "phase_replay_executable_sha256": sha256_file(executable),
        "maximum_forward_offset_steps": maximum_offset,
        "jobs": jobs,
        "wall_seconds": wall_seconds,
        "sum_model_runtime_seconds": float(sum(model_seconds)),
        "maximum_model_runtime_seconds": float(max(model_seconds)),
        "cache_files": [
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in cache_paths
        ],
    }
    write_json(cache_dir / "capture_complete.json", report, overwrite=False)
    return reopened, report


def _audit_unmanifested_native_cache(
    cache_dir: Path,
    *,
    history: Path,
    chunk_bytes: int,
    chunk_indices: np.ndarray,
    base_indices: np.ndarray,
    phase_offsets: np.ndarray,
    layout: Any,
    source_dir: Path,
    executable: Path,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Seal a legacy cache only after exact authentic replay spot checks.

    This recovery path exists for captures made before completion manifests
    were added.  It validates every value for finiteness and repeats the first,
    middle, and last trajectories through the authentic executable.  Every
    audited compact native state must match bit for bit.
    """

    cache_paths = [
        cache_dir / f"native_phase_{int(offset):02d}.npy"
        for offset in phase_offsets
    ]
    arrays: list[np.ndarray] = []
    for path in cache_paths:
        if not path.is_file():
            raise ValueError(f"unmanifested native cache is incomplete: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (chunk_indices.size, layout.compact_size)
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(f"unmanifested native cache has bad schema: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"unmanifested native cache is nonfinite: {path}")
        arrays.append(array)

    audit_rows = np.unique(
        np.asarray((0, chunk_indices.size // 2, chunk_indices.size - 1), dtype=int)
    )
    maximum_offset = int(phase_offsets[-1])
    audit_records: list[dict[str, Any]] = []
    for row in audit_rows:
        chunk_index = int(chunk_indices[row])
        with history.open("rb") as stream:
            stream.seek(chunk_index * chunk_bytes)
            payload = stream.read(chunk_bytes)
        if len(payload) != chunk_bytes:
            raise RuntimeError(f"could not audit production checkpoint {chunk_index}")
        sparse = read_restart_payload(payload, layout)
        with tempfile.TemporaryDirectory(prefix="zc-cache-audit-") as temporary:
            root = Path(temporary)
            restart = root / "sparse.hst"
            restart.write_bytes(payload)
            run_dir = root / "run"
            fresh_generator.prepare_run(
                source_dir,
                executable,
                run_dir,
                nstart=3,
                tfind=fresh_generator.model_time(sparse.nt),
                tzero=fresh_generator.model_time(sparse.nt),
                tend=fresh_generator.model_time(sparse.nt + maximum_offset),
                ntape=1,
                nrewnd=10,
                nic=0,
                write_start=(
                    fresh_generator.model_time(sparse.nt + maximum_offset) + 1.0
                ),
                write_end=(
                    fresh_generator.model_time(sparse.nt + maximum_offset) + 1.0
                ),
                restart=restart,
            )
            runtime = fresh_generator.run_model(run_dir)
            output = run_dir / "outhst"
            if output.stat().st_size != (maximum_offset + 1) * chunk_bytes:
                raise RuntimeError("cache audit replay produced the wrong byte count")
            with output.open("rb") as stream:
                for position, offset in enumerate(phase_offsets):
                    stream.seek(int(offset) * chunk_bytes)
                    phase_payload = stream.read(chunk_bytes)
                    phase_sample = read_restart_payload(phase_payload, layout)
                    expected_nt = sparse.nt + int(offset)
                    if phase_sample.nt != expected_nt:
                        raise RuntimeError("cache audit replay produced a wrong clock")
                    replayed = phase_sample.native_controls.astype(np.float32)
                    if not np.array_equal(replayed, arrays[position][row]):
                        raise RuntimeError(
                            "unmanifested native cache failed exact replay audit at "
                            f"row {int(row)}, phase {int(offset)}"
                        )
        audit_records.append(
            {
                "row": int(row),
                "chunk_index": chunk_index,
                "base_input_index": int(base_indices[row]),
                "model_runtime_seconds": float(runtime["elapsed_seconds"]),
                "comparison": "bitwise equality at every requested phase",
            }
        )

    report = {
        "method": (
            "adopted an older unmanifested phase cache after complete schema and "
            "finiteness checks plus exact authentic replay audits"
        ),
        "checkpoint_count": int(chunk_indices.size),
        "phase_offsets": phase_offsets.tolist(),
        "chunk_indices_sha256": sha256_array(chunk_indices),
        "base_input_indices_sha256": sha256_array(base_indices),
        "history_sha256": sha256_file(history),
        "phase_replay_executable_sha256": sha256_file(executable),
        "maximum_forward_offset_steps": maximum_offset,
        "jobs": None,
        "wall_seconds": None,
        "sum_model_runtime_seconds": float(
            sum(record["model_runtime_seconds"] for record in audit_records)
        ),
        "maximum_model_runtime_seconds": float(
            max(record["model_runtime_seconds"] for record in audit_records)
        ),
        "cache_recovery_audits": audit_records,
        "cache_files": [
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in cache_paths
        ],
    }
    write_json(cache_dir / "capture_complete.json", report, overwrite=False)
    return arrays, report


def _load_native_cache(
    cache_dir: Path,
    *,
    phase_offsets: np.ndarray,
    chunk_indices: np.ndarray,
    base_indices: np.ndarray,
    compact_size: int,
    history_sha256: str,
    executable_sha256: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Load only a complete cache bound to the current source population."""

    report_path = cache_dir / "capture_complete.json"
    if not report_path.is_file():
        raise ValueError(f"native cache has no completion manifest: {cache_dir}")
    report = load_json(report_path)
    expected_scalars = {
        "phase_offsets": phase_offsets.tolist(),
        "checkpoint_count": int(chunk_indices.size),
        "chunk_indices_sha256": sha256_array(chunk_indices),
        "base_input_indices_sha256": sha256_array(base_indices),
        "history_sha256": history_sha256,
        "phase_replay_executable_sha256": executable_sha256,
    }
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            raise ValueError(f"native cache binding differs for {key}")
    records = report.get("cache_files")
    if not isinstance(records, list) or len(records) != phase_offsets.size:
        raise ValueError("native cache manifest has the wrong phase-file count")
    arrays: list[np.ndarray] = []
    for offset, raw in zip(phase_offsets, records, strict=True):
        if not isinstance(raw, dict):
            raise ValueError("native cache manifest has an invalid file record")
        expected_name = f"native_phase_{int(offset):02d}.npy"
        if raw.get("file") != expected_name:
            raise ValueError("native cache phase filename mismatch")
        path = cache_dir / expected_name
        if not path.is_file() or path.stat().st_size != int(raw.get("size_bytes", -1)):
            raise ValueError(f"native cache file is missing or has changed: {path}")
        if sha256_file(path) != raw.get("sha256"):
            raise ValueError(f"native cache SHA-256 mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.shape != (chunk_indices.size, compact_size)
            or array.dtype != np.float32
        ):
            raise ValueError(f"native cache array has the wrong schema: {path}")
        arrays.append(array)
    return arrays, report


def _load_standardized_phase_observations(
    data_dir: Path,
    indices: np.ndarray,
    normalization: Path,
    *,
    nominal_phase_offset: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one nominal modulo-36 phase bin in frozen standardized units.

    The fresh file's phase sine/cosine is evaluated from a large float32
    absolute clock and drifts by up to roughly 0.003 across 12,000 years.  The
    native model step and the saved fields nevertheless share the exact same
    integer modulo-36 phase bin.  The control map excludes the two phase
    features, so the integer bin is the appropriate binding and the small
    trigonometric spread is recorded rather than rejected.
    """

    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size < 2:
        raise ValueError("phase observation indices must be a nonempty vector")
    if np.any(selected % 36 != nominal_phase_offset % 36):
        raise ValueError("observation indices leave their nominal modulo-36 phase")
    if np.any(np.diff(selected) != 360):
        raise ValueError("phase observation indices do not retain ten-year spacing")
    metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path)
    records = metadata.get("field_files")
    if not isinstance(records, dict):
        raise ValueError("fresh metadata has no field-file mapping")
    rows = selected.size
    raw = np.empty((rows, 4 * 20 * 27), dtype=np.float64)
    field_records: dict[str, Any] = {}
    for field_position, field in enumerate(CORE4_FIELDS):
        record = records.get(field)
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise ValueError(f"fresh metadata has no file for {field}")
        field_path = data_dir / record["file"]
        values = np.load(field_path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 3 or values.shape[1:] != (20, 27):
            raise ValueError(f"fresh field has unexpected shape: {field_path}")
        raw[:, field_position * 540 : (field_position + 1) * 540] = np.asarray(
            values[selected], dtype=np.float64
        ).reshape(rows, 540)
        field_records[field] = {
            "file": record["file"],
            "sha256": record.get("sha256"),
        }
    with np.load(normalization, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
        count = int(archive["count"])
    if mean.shape != (2162,) or scale.shape != (2162,):
        raise ValueError("normalization is not the core4-plus-phase contract")
    if np.any(scale[:2160] <= 0.0) or not np.isfinite(scale[:2160]).all():
        raise ValueError("core4 normalization scale is invalid")
    standardized = (raw - mean[:2160]) / scale[:2160]
    phase_record = metadata.get("phase")
    if not isinstance(phase_record, dict) or not isinstance(
        phase_record.get("file"), str
    ):
        raise ValueError("fresh metadata has no annual-phase file")
    phase = np.asarray(
        np.load(data_dir / phase_record["file"], mmap_mode="r", allow_pickle=False)[
            selected
        ],
        dtype=np.float64,
    )
    phase_distance = np.linalg.norm(phase - phase[0], axis=1)
    return standardized, {
        "metadata_sha256": sha256_file(metadata_path),
        "normalization_sha256": sha256_file(normalization),
        "normalization_training_count": count,
        "nominal_phase_offset_modulo_36": int(nominal_phase_offset % 36),
        "phase_policy": "exact input-index modulo 36; float32 phase drift recorded",
        "first_phase_sin_cos": phase[0].tolist(),
        "last_phase_sin_cos": phase[-1].tolist(),
        "maximum_phase_sin_cos_distance_from_first": float(
            np.max(phase_distance, initial=0.0)
        ),
        "field_records": field_records,
    }


def _phase_transpose_diagnostics(
    model: PhaseLocalControlStack, *, seed: int = 9301
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    native_defects = []
    observation_defects = []
    for offset in model.phase_offsets:
        control = rng.normal(size=model.rank)
        native_seed = rng.normal(size=model.layout.packed_size)
        observation_seed = rng.normal(size=model.observation_size)
        native_left = float(
            model.apply_B(control, phase_offset=int(offset)) @ native_seed
        )
        native_right = float(
            control @ model.apply_BT(native_seed, phase_offset=int(offset))
        )
        observation_left = float(
            model.apply_G(control, phase_offset=int(offset)) @ observation_seed
        )
        observation_right = float(
            control
            @ model.apply_GT(observation_seed, phase_offset=int(offset))
        )
        native_defects.append(
            abs(native_left - native_right)
            / max(abs(native_left), abs(native_right), 1.0)
        )
        observation_defects.append(
            abs(observation_left - observation_right)
            / max(abs(observation_left), abs(observation_right), 1.0)
        )
    return {
        "seed": seed,
        "maximum_B_BT_relative_dot_defect": float(max(native_defects)),
        "maximum_G_GT_relative_dot_defect": float(max(observation_defects)),
    }


def _holdout_diagnostics(
    model: PhaseLocalControlStack,
    native_holdout: list[np.ndarray],
    observation_holdout: list[np.ndarray],
    *,
    ridge_fraction: float,
) -> dict[str, Any]:
    alpha = ridge_fraction * float(model.pooled_eigenvalues[0])
    by_phase: dict[str, Any] = {}
    for position, offset in enumerate(model.phase_offsets):
        observation_target = (
            np.asarray(observation_holdout[position], dtype=np.float64)
            - model.observation_mean[position]
        )
        gram = model.G[position].T @ model.G[position] + alpha * np.eye(model.rank)
        controls = scipy.linalg.solve(
            gram,
            (observation_target @ model.G[position]).T,
            assume_a="pos",
            check_finite=True,
        ).T
        observation_prediction = controls @ model.G[position].T
        observation_residual = observation_prediction - observation_target
        target_norm = np.linalg.norm(observation_target, axis=1)
        prediction_norm = np.linalg.norm(observation_prediction, axis=1)
        cosine = np.divide(
            np.sum(observation_target * observation_prediction, axis=1),
            target_norm * prediction_norm,
            out=np.zeros_like(target_norm),
            where=(target_norm > 0.0) & (prediction_norm > 0.0),
        )

        native_target = (
            np.asarray(native_holdout[position], dtype=np.float64)
            - model.native_mean[position]
        )
        native_prediction = controls @ model.B[position].T
        positive_scale = model.native_scale[position] > 0.0
        normalized_native_residual = (
            native_prediction[:, positive_scale] - native_target[:, positive_scale]
        ) / model.native_scale[position, positive_scale]
        by_phase[str(int(offset))] = {
            "observation_median_cosine": float(np.median(cosine)),
            "observation_median_standardized_rms_residual": float(
                np.median(np.sqrt(np.mean(observation_residual**2, axis=1)))
            ),
            "native_median_coordinate_standardized_rms_residual": float(
                np.median(
                    np.sqrt(np.mean(normalized_native_residual**2, axis=1))
                )
            ),
        }
    return {
        "ridge_fraction_of_leading_pooled_eigenvalue": ridge_fraction,
        "ridge_alpha": alpha,
        "by_phase": by_phase,
    }


def main() -> int:
    args = parser().parse_args()
    if not math.isfinite(args.ridge_fraction) or args.ridge_fraction < 0.0:
        raise ValueError("ridge-fraction must be finite and nonnegative")
    phases = np.asarray(args.phase_offsets, dtype=np.int64)
    if (
        phases.ndim != 1
        or phases.size == 0
        or phases[0] < 0
        or np.any(np.diff(phases) <= 0)
    ):
        raise ValueError("phase offsets must be nonnegative and strictly increasing")

    history = _resolve(args.history)
    production_report_path = _resolve(args.production_report)
    data_dir = _resolve(args.data_dir)
    normalization = _resolve(args.normalization)
    state_manifest = _resolve(args.state_manifest)
    source_dir = _resolve(args.source_dir)
    executable = _resolve(args.executable)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "phase_local_control_stack.npz"
    report_path = output_dir / "report.json"
    if not args.overwrite and (artifact_path.exists() or report_path.exists()):
        raise FileExistsError("phase-local artifact already exists; use --overwrite")
    for required in (
        history,
        production_report_path,
        data_dir / "metadata.json",
        normalization,
        state_manifest,
        source_dir,
        executable,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    started_utc = datetime.now(UTC).isoformat()
    production_report = load_json(production_report_path)
    metadata = load_json(data_dir / "metadata.json")
    executable_hash = sha256_file(executable)
    expected_executable_hashes = {
        str(entry["event_replay_executable_sha256"])
        for entry in metadata.get("event_restart_checkpoints", [])
        if isinstance(entry, dict) and entry.get("event_replay_executable_sha256")
    }
    if expected_executable_hashes and executable_hash not in expected_executable_hashes:
        raise ValueError("phase-replay executable is not release-bound by fresh data")
    training_raw = metadata.get("chronological_split", {}).get("train")
    if not isinstance(training_raw, list) or len(training_raw) != 2:
        raise ValueError("fresh metadata has no training interval")
    training_interval = (int(training_raw[0]), int(training_raw[1]))
    layout = load_independent_control_layout(state_manifest)
    chunks, base_indices = _selected_checkpoints(
        history,
        production_report=production_report,
        training_interval=training_interval,
        phase_offsets=phases,
        layout=layout,
    )
    cache_dir = output_dir / "native_cache"
    chunk_bytes = int(production_report["checkpoint_chunk_bytes"])
    if cache_dir.exists() and args.reuse_native_cache:
        if (cache_dir / "capture_complete.json").is_file():
            native, capture_report = _load_native_cache(
                cache_dir,
                phase_offsets=phases,
                chunk_indices=chunks,
                base_indices=base_indices,
                compact_size=layout.compact_size,
                history_sha256=str(production_report["history_sha256"]),
                executable_sha256=executable_hash,
            )
        else:
            native, capture_report = _audit_unmanifested_native_cache(
                cache_dir,
                history=history,
                chunk_bytes=chunk_bytes,
                chunk_indices=chunks,
                base_indices=base_indices,
                phase_offsets=phases,
                layout=layout,
                source_dir=source_dir,
                executable=executable,
            )
    else:
        if cache_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"native cache exists; use --reuse-native-cache or --overwrite: "
                    f"{cache_dir}"
                )
            shutil.rmtree(cache_dir)
        native, capture_report = _capture_native_phases(
            history=history,
            chunk_bytes=chunk_bytes,
            chunk_indices=chunks,
            base_indices=base_indices,
            phase_offsets=phases,
            layout=layout,
            source_dir=source_dir,
            executable=executable,
            cache_dir=cache_dir,
            jobs=args.jobs,
        )

    observations: list[np.ndarray] = []
    observation_sources: list[dict[str, Any]] = []
    input_indices: list[np.ndarray] = []
    for offset in phases:
        indices = base_indices + int(offset)
        values, source = _load_standardized_phase_observations(
            data_dir,
            indices,
            normalization,
            nominal_phase_offset=int(offset),
        )
        observations.append(values)
        observation_sources.append(source)
        input_indices.append(indices)
    expected_indices = np.stack(
        [base_indices + int(offset) for offset in phases], axis=0
    )
    if not all(
        np.array_equal(actual, expected)
        for actual, expected in zip(input_indices, expected_indices, strict=True)
    ):
        raise RuntimeError("phase input-index construction changed unexpectedly")

    validation_count = max(2, int(round(args.validation_fraction * chunks.size)))
    fit_count = int(chunks.size - validation_count)
    if args.rank >= fit_count:
        raise ValueError("rank must be smaller than the diagnostic fit sample count")
    diagnostic = fit_phase_local_control_stack(
        [values[:fit_count] for values in native],
        [values[:fit_count] for values in observations],
        phase_offsets=phases,
        layout=layout,
        rank=args.rank,
    )
    holdout = _holdout_diagnostics(
        diagnostic,
        [values[fit_count:] for values in native],
        [values[fit_count:] for values in observations],
        ridge_fraction=args.ridge_fraction,
    )
    final = fit_phase_local_control_stack(
        native,
        observations,
        phase_offsets=phases,
        layout=layout,
        rank=args.rank,
    )
    transpose = _phase_transpose_diagnostics(final)
    if max(
        transpose["maximum_B_BT_relative_dot_defect"],
        transpose["maximum_G_GT_relative_dot_defect"],
    ) > 5.0e-12:
        raise RuntimeError("phase-local transpose diagnostics failed")

    source_metadata = {
        "script_version": SCRIPT_VERSION,
        "production_report_sha256": sha256_file(production_report_path),
        "history_sha256": sha256_file(history),
        "data_metadata_sha256": sha256_file(data_dir / "metadata.json"),
        "normalization_sha256": sha256_file(normalization),
        "state_manifest_sha256": sha256_file(state_manifest),
        "phase_replay_executable_sha256": executable_hash,
        "phase_offsets": phases.tolist(),
        "base_input_indices_sha256": sha256_array(base_indices),
        "construction": (
            "one pooled sample Gram K=mean_k(Zc_k Zc_k.T)/(n-1); "
            "B_k=Xc_k.T U/sqrt(n-1); G_k=Zc_k.T U/sqrt(n-1)"
        ),
    }
    with atomic_output_path(artifact_path, overwrite=args.overwrite) as temporary:
        final.save(temporary, metadata=source_metadata)
    loaded, loaded_metadata = PhaseLocalControlStack.load(artifact_path)
    if loaded_metadata != source_metadata:
        raise RuntimeError("phase-local artifact metadata failed round trip")

    pooled_gram = np.einsum("pdr,pds->rs", final.G, final.G) / final.phase_count
    capture_report["cache_retained_after_success"] = bool(args.keep_native_cache)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "validated_coherent_phase_local_training_secant_stack",
        "started_utc": started_utc,
        "completed_utc": datetime.now(UTC).isoformat(),
        "artifact": {
            "file": artifact_path.name,
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
        "dimensions": {
            "phase_count": final.phase_count,
            "phase_offsets": phases.tolist(),
            "training_samples": final.sample_count,
            "packed_native_state": final.layout.packed_size,
            "compact_independent_native_controls": final.layout.compact_size,
            "standardized_core4_observation": final.observation_size,
            "rank": final.rank,
        },
        "construction": {
            "definition": source_metadata["construction"],
            "reason": (
                "The shared sample vectors give every B_k column the same "
                "training-trajectory contrast. Separate phasewise PCA fits would "
                "have arbitrary signs and orthogonal rotations."
            ),
            "reduced_control_covariance": (
                "U.T U = I, so sqrt(n-1) U has empirical covariance I"
            ),
            "pooled_observation_eigenvalues": final.pooled_eigenvalues.tolist(),
            "pooled_observation_variance_trace": (
                final.pooled_observation_variance_trace
            ),
            "retained_pooled_variance_fraction": float(
                np.sum(final.pooled_eigenvalues)
                / final.pooled_observation_variance_trace
            ),
            "sample_basis_orthogonality_max_abs": float(
                np.max(
                    np.abs(
                        final.sample_vectors.T @ final.sample_vectors
                        - np.eye(final.rank)
                    ),
                    initial=0.0,
                )
            ),
            "pooled_G_gram_max_abs_defect": float(
                np.max(
                    np.abs(
                        pooled_gram - np.diag(final.pooled_eigenvalues)
                    ),
                    initial=0.0,
                )
            ),
        },
        "capture": capture_report,
        "source": {
            **source_metadata,
            "training_half_open_interval": list(training_interval),
            "first_base_input_index": int(base_indices[0]),
            "last_base_input_index": int(base_indices[-1]),
            "base_input_indices_sha256": sha256_array(base_indices),
            "phase_input_indices_sha256": [
                sha256_array(indices) for indices in input_indices
            ],
            "observation_sources": observation_sources,
        },
        "chronological_training_only_holdout": {
            "fit_sample_count": fit_count,
            "holdout_sample_count": validation_count,
            **holdout,
        },
        "transpose_diagnostics": transpose,
        "limitations": [
            "The phase-local maps are empirical secant regressions through sparse "
            "authentic training trajectories, not an exact nonlinear balance manifold.",
            "Only manifest-declared independent real controls are changed; active "
            "diagnostic and boundary memories must readjust through model dynamics.",
            "The stack is tied to offsets 26--34 and must not be used at other "
            "annual phases without another phase-matched construction.",
            "Finite controls and all changed branch paths require authentic nonlinear "
            "replay rather than tangent-only interpretation.",
        ],
    }
    write_json(report_path, report, overwrite=args.overwrite)
    if not args.keep_native_cache:
        shutil.rmtree(cache_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
