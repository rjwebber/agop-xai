#!/usr/bin/env python3
"""Capture annual, phase-local native ZC controls from certified history.

The canonical fresh production run retains a complete native restart every ten
years.  This utility replays each training-only restart once and extracts ten
annual samples at each requested within-year phase.  It deliberately suppresses
the 13-field output stream and does not alter the model equations or executable.

The default schedule captures all 36 within-year phases for all 10,000 years in
the fresh training block.  Output is 36 ``float32`` ``.npy`` arrays with shape
``(10000, 28591)`` plus source-bound progress and completion manifests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import generate_fresh_zc_dataset as fresh_generator  # noqa: E402

from adjoint.balanced_control_map import (  # noqa: E402
    IndependentControlLayout,
    load_independent_control_layout,
    read_restart_payload,
)
from zc_xai.io import (  # noqa: E402
    load_json,
    sha256_array,
    sha256_file,
    sha256_json,
    write_json,
)

SCRIPT_VERSION = "1.2.0"
REPORT_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1
DEFAULT_PHASE_OFFSETS = tuple(range(fresh_generator.STEPS_PER_YEAR))
FLOAT32_DTYPE = np.dtype("<f4")


@dataclasses.dataclass(frozen=True)
class CaptureSchedule:
    """Exact sparse-checkpoint and annual-row indexing for one capture."""

    source_chunk_indices: np.ndarray
    source_base_input_indices: np.ndarray
    annual_base_input_indices: np.ndarray
    phase_input_indices: np.ndarray
    relative_offsets: np.ndarray
    source_checkpoint_stride_steps: int
    annual_base_stride_steps: int
    years_per_source_checkpoint: int

    @property
    def source_checkpoint_count(self) -> int:
        return int(self.source_chunk_indices.size)

    @property
    def sample_count(self) -> int:
        return int(self.annual_base_input_indices.size)

    @property
    def maximum_replay_offset_steps(self) -> int:
        return int(np.max(self.relative_offsets))


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
    overlap_mode = result.add_mutually_exclusive_group()
    overlap_mode.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=Path(
            "outputs/zc_balanced_control_map/"
            "training-phases26-34-common-rank36/native_cache"
        ),
        help="Existing ten-year cache used for the exhaustive y=0 overlap audit.",
    )
    overlap_mode.add_argument(
        "--independent-overlap-replay",
        action="store_true",
        help=(
            "Validate y=0 rows by independently replaying every source checkpoint "
            "only through the requested phases instead of reading a legacy cache."
        ),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/zc_native_phase_cache/"
            "training-years10000-phases00-35"
        ),
    )
    result.add_argument(
        "--phase-offsets", nargs="+", type=int, default=DEFAULT_PHASE_OFFSETS
    )
    result.add_argument(
        "--jobs", type=positive_int, default=min(4, os.cpu_count() or 1)
    )
    result.add_argument(
        "--progress-interval",
        type=positive_int,
        default=8,
        help="Flush arrays and atomically publish progress after this many blocks.",
    )
    mode = result.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after validating every previously committed block.",
    )
    mode.add_argument("--overwrite", action="store_true")
    return result


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _phase_offsets(values: Sequence[int]) -> np.ndarray:
    phases = np.asarray(values, dtype=np.int64)
    if (
        phases.ndim != 1
        or phases.size == 0
        or phases[0] < 0
        or phases[-1] >= fresh_generator.STEPS_PER_YEAR
        or np.any(np.diff(phases) <= 0)
    ):
        raise ValueError(
            "phase offsets must be strictly increasing integers in [0, 36)"
        )
    return phases


def _history_samples(
    history: Path,
    *,
    production_report: dict[str, Any],
    layout: IndependentControlLayout,
) -> tuple[np.ndarray, np.ndarray]:
    chunk_bytes = int(production_report["checkpoint_chunk_bytes"])
    checkpoint_count = int(production_report["checkpoint_count"])
    expected_size = chunk_bytes * checkpoint_count
    if history.stat().st_size != expected_size:
        raise ValueError(
            f"history has {history.stat().st_size} bytes, expected {expected_size}"
        )
    expected_hash = production_report.get("history_sha256")
    if not isinstance(expected_hash, str) or sha256_file(history) != expected_hash:
        raise ValueError("production history fails its recorded SHA-256")

    chunk_indices = np.arange(checkpoint_count, dtype=np.int64)
    nt_values = np.empty(checkpoint_count, dtype=np.int64)
    with history.open("rb") as stream:
        for chunk_index in range(checkpoint_count):
            payload = stream.read(chunk_bytes)
            if len(payload) != chunk_bytes:
                raise RuntimeError(f"truncated history checkpoint {chunk_index}")
            nt_values[chunk_index] = read_restart_payload(payload, layout).nt
    if np.any(np.diff(nt_values) <= 0):
        raise RuntimeError("production checkpoint clocks are not strictly increasing")
    return chunk_indices, nt_values


def build_capture_schedule(
    chunk_indices: np.ndarray,
    nt_values: np.ndarray,
    *,
    spinup_steps: int,
    source_checkpoint_stride_steps: int,
    training_interval: tuple[int, int],
    phase_offsets: np.ndarray,
    steps_per_year: int = fresh_generator.STEPS_PER_YEAR,
) -> CaptureSchedule:
    """Tile each sparse source checkpoint into complete annual phase rows."""

    chunks = np.asarray(chunk_indices, dtype=np.int64)
    clocks = np.asarray(nt_values, dtype=np.int64)
    phases = np.asarray(phase_offsets, dtype=np.int64)
    if chunks.ndim != 1 or clocks.shape != chunks.shape or chunks.size < 2:
        raise ValueError("chunk indices and clocks must be equal nonempty vectors")
    if source_checkpoint_stride_steps <= 0 or steps_per_year <= 0:
        raise ValueError("checkpoint and annual strides must be positive")
    years, remainder = divmod(source_checkpoint_stride_steps, steps_per_year)
    if remainder or years <= 0:
        raise ValueError("source checkpoint stride is not whole model years")
    if np.any(np.diff(clocks) != source_checkpoint_stride_steps):
        raise ValueError("history clocks disagree with the reported checkpoint stride")

    lower, upper = training_interval
    if lower < 0 or upper <= lower:
        raise ValueError("training interval must be a nonempty half-open interval")
    relative = (
        np.arange(years, dtype=np.int64)[:, None] * steps_per_year
        + phases[None, :]
    )
    selected_chunks: list[int] = []
    selected_sparse_bases: list[int] = []
    annual_blocks: list[np.ndarray] = []
    for chunk, nt in zip(chunks, clocks, strict=True):
        sparse_base = int(nt) - spinup_steps
        annual = sparse_base + np.arange(years, dtype=np.int64) * steps_per_year
        indices = annual[:, None] + phases[None, :]
        eligible_year = np.all((indices >= lower) & (indices < upper), axis=1)
        partial_year = np.any((indices >= lower) & (indices < upper), axis=1)
        if np.any(partial_year != eligible_year):
            raise RuntimeError("a training boundary cuts through a requested phase set")
        eligible_count = int(np.count_nonzero(eligible_year))
        if eligible_count not in (0, years):
            raise RuntimeError("a training boundary cuts through a source checkpoint")
        if eligible_count:
            selected_chunks.append(int(chunk))
            selected_sparse_bases.append(sparse_base)
            annual_blocks.append(annual)

    if not annual_blocks:
        raise RuntimeError(
            "no complete annual phase sets fall in the training interval"
        )
    sparse_bases = np.asarray(selected_sparse_bases, dtype=np.int64)
    if np.any(np.diff(sparse_bases) != source_checkpoint_stride_steps):
        raise RuntimeError("selected source checkpoints do not retain their stride")
    annual_bases = np.concatenate(annual_blocks)
    if np.any(np.diff(annual_bases) != steps_per_year):
        raise RuntimeError("annual base indices are not contiguous model years")
    expected_count, extra = divmod(upper - lower, steps_per_year)
    if extra or annual_bases.size != expected_count:
        raise RuntimeError(
            "training length and selected annual sample count are inconsistent"
        )
    phase_indices = annual_bases[None, :] + phases[:, None]
    if np.any((phase_indices < lower) | (phase_indices >= upper)):
        raise RuntimeError("constructed phase indices leave the training interval")
    if any(
        np.any(phase_indices[position] % steps_per_year != phase % steps_per_year)
        for position, phase in enumerate(phases)
    ):
        raise RuntimeError("constructed rows leave their integer annual phase bins")

    return CaptureSchedule(
        source_chunk_indices=np.asarray(selected_chunks, dtype=np.int64),
        source_base_input_indices=sparse_bases,
        annual_base_input_indices=annual_bases,
        phase_input_indices=phase_indices,
        relative_offsets=relative,
        source_checkpoint_stride_steps=source_checkpoint_stride_steps,
        annual_base_stride_steps=steps_per_year,
        years_per_source_checkpoint=years,
    )


def _static_binding(
    *,
    history: Path,
    production_report_path: Path,
    data_metadata_path: Path,
    state_manifest: Path,
    source_dir: Path,
    executable: Path,
    reference_cache_dir: Path | None,
    production_report: dict[str, Any],
    training_interval: tuple[int, int],
    phase_offsets: np.ndarray,
    schedule: CaptureSchedule,
    state_size: int,
) -> dict[str, Any]:
    source_files = fresh_generator.source_manifest(source_dir)
    source = {
        "history": str(history),
        "production_report": str(production_report_path),
        "data_metadata": str(data_metadata_path),
        "state_manifest": str(state_manifest),
        "source_dir": str(source_dir),
        "executable": str(executable),
    }
    binding = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "capture_script_sha256": sha256_file(Path(__file__).resolve()),
        "status": "in_progress",
        "phase_offsets": phase_offsets.tolist(),
        "sample_count": schedule.sample_count,
        "state_size": state_size,
        "dtype": FLOAT32_DTYPE.str,
        "training_interval": list(training_interval),
        "base_input_indices_sha256": sha256_array(
            schedule.annual_base_input_indices
        ),
        "phase_input_indices_sha256": sha256_array(schedule.phase_input_indices),
        "history_sha256": sha256_file(history),
        "production_report_sha256": sha256_file(production_report_path),
        "phase_replay_executable_sha256": sha256_file(executable),
        "state_manifest_sha256": sha256_file(state_manifest),
        "data_metadata_sha256": sha256_file(data_metadata_path),
        "source_manifest_sha256": fresh_generator.manifest_sha256(source_files),
        "source": source,
        "schedule": {
            "source_checkpoint_count": schedule.source_checkpoint_count,
            "source_history_checkpoint_count": int(
                production_report["checkpoint_count"]
            ),
            "source_checkpoint_stride_steps": (
                schedule.source_checkpoint_stride_steps
            ),
            "annual_base_stride_steps": schedule.annual_base_stride_steps,
            "years_per_source_checkpoint": schedule.years_per_source_checkpoint,
            "maximum_replay_offset_steps": (
                schedule.maximum_replay_offset_steps
            ),
            "source_chunk_indices_sha256": sha256_array(
                schedule.source_chunk_indices
            ),
            "source_base_input_indices_sha256": sha256_array(
                schedule.source_base_input_indices
            ),
            "relative_offsets_sha256": sha256_array(schedule.relative_offsets),
        },
    }
    if reference_cache_dir is None:
        binding["overlap_validation_mode"] = "independent_overlap_replay"
    else:
        binding["reference_cache_manifest_sha256"] = sha256_file(
            reference_cache_dir / "capture_complete.json"
        )
        source["reference_cache_dir"] = str(reference_cache_dir)
    return binding


def _cache_paths(output_dir: Path, phases: np.ndarray) -> list[Path]:
    return [output_dir / f"native_phase_{int(phase):02d}.npy" for phase in phases]


def _create_arrays(
    paths: Sequence[Path], *, sample_count: int, state_size: int
) -> list[np.memmap]:
    return [
        np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=FLOAT32_DTYPE,
            shape=(sample_count, state_size),
        )
        for path in paths
    ]


def _open_arrays(
    paths: Sequence[Path], *, sample_count: int, state_size: int, mode: str
) -> list[np.memmap]:
    arrays: list[np.memmap] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"native cache file is missing: {path}")
        array = np.load(path, mmap_mode=mode, allow_pickle=False)
        if array.shape != (sample_count, state_size) or array.dtype != FLOAT32_DTYPE:
            raise ValueError(f"native cache file has an incompatible schema: {path}")
        arrays.append(array)
    return arrays


def _canonical_block(
    arrays: Sequence[np.ndarray], block_index: int, years_per_block: int
) -> np.ndarray:
    start = block_index * years_per_block
    stop = start + years_per_block
    return np.stack([array[start:stop] for array in arrays], axis=0)


def _block_hash(values: np.ndarray) -> str:
    return sha256_array(np.asarray(values, dtype=FLOAT32_DTYPE))


def _progress_document(
    binding: dict[str, Any],
    *,
    completed: dict[int, str],
    model_runtime: dict[int, float],
) -> dict[str, Any]:
    document = dict(binding)
    document.update(
        {
            "updated_utc": datetime.now(UTC).isoformat(),
            "completed_source_blocks": sorted(completed),
            "completed_block_sha256": {
                str(index): completed[index] for index in sorted(completed)
            },
            "model_runtime_seconds_by_block": {
                str(index): model_runtime[index] for index in sorted(model_runtime)
            },
        }
    )
    return document


def _load_progress(
    path: Path,
    *,
    expected_binding: dict[str, Any],
    arrays: Sequence[np.ndarray],
    years_per_block: int,
) -> tuple[dict[int, str], dict[int, float]]:
    document = load_json(path)
    for key, expected in expected_binding.items():
        if document.get(key) != expected:
            raise ValueError(f"resume manifest binding differs for {key}")
    raw_hashes = document.get("completed_block_sha256")
    raw_runtime = document.get("model_runtime_seconds_by_block")
    if not isinstance(raw_hashes, dict) or not isinstance(raw_runtime, dict):
        raise ValueError("resume manifest has no completed block records")
    completed = {int(key): str(value) for key, value in raw_hashes.items()}
    runtime = {int(key): float(value) for key, value in raw_runtime.items()}
    expected_indices = sorted(completed)
    if document.get("completed_source_blocks") != expected_indices:
        raise ValueError("resume manifest completed-block list is inconsistent")
    if set(runtime) != set(completed):
        raise ValueError("resume manifest runtime and hash blocks differ")
    for block_index, expected_hash in completed.items():
        if not 0 <= block_index < expected_binding["schedule"][
            "source_checkpoint_count"
        ]:
            raise ValueError("resume manifest contains an invalid block index")
        actual_hash = _block_hash(
            _canonical_block(arrays, block_index, years_per_block)
        )
        if actual_hash != expected_hash:
            raise ValueError(
                f"committed native cache block {block_index} fails SHA-256"
            )
    return completed, runtime


def _replay_block(
    *,
    block_index: int,
    chunk_index: int,
    sparse_base_input_index: int,
    history: Path,
    chunk_bytes: int,
    spinup_steps: int,
    phase_offsets: np.ndarray,
    relative_offsets: np.ndarray,
    layout: IndependentControlLayout,
    source_dir: Path,
    executable: Path,
) -> tuple[int, np.ndarray, float]:
    with history.open("rb") as stream:
        stream.seek(chunk_index * chunk_bytes)
        payload = stream.read(chunk_bytes)
    if len(payload) != chunk_bytes:
        raise RuntimeError(f"could not read source history chunk {chunk_index}")
    sparse = read_restart_payload(payload, layout)
    if sparse.nt - spinup_steps != sparse_base_input_index:
        raise RuntimeError("source history checkpoint changed its bound clock")

    maximum_offset = int(np.max(relative_offsets))
    with tempfile.TemporaryDirectory(prefix="zc-annual-native-") as temporary:
        root = Path(temporary)
        restart = root / "sparse.hst"
        restart.write_bytes(payload)
        run_dir = root / "run"
        final_time = fresh_generator.model_time(sparse.nt + maximum_offset)
        fresh_generator.prepare_run(
            source_dir,
            executable,
            run_dir,
            nstart=3,
            tfind=fresh_generator.model_time(sparse.nt),
            tzero=fresh_generator.model_time(sparse.nt),
            tend=final_time,
            ntape=1,
            nrewnd=10,
            nic=0,
            write_start=final_time + 1.0,
            write_end=final_time + 1.0,
            restart=restart,
        )
        runtime = fresh_generator.run_model(run_dir)
        if int(runtime["fresh_fields_size_bytes"]) != 0:
            raise RuntimeError("annual native replay unexpectedly wrote field output")
        output = run_dir / "outhst"
        expected_bytes = (maximum_offset + 1) * chunk_bytes
        if output.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"replay block {block_index} produced {output.stat().st_size} "
                f"history bytes; expected {expected_bytes}"
            )

        samples = np.empty(
            (phase_offsets.size, relative_offsets.shape[0], layout.compact_size),
            dtype=FLOAT32_DTYPE,
        )
        with output.open("rb") as stream:
            for year_position in range(relative_offsets.shape[0]):
                for phase_position, phase in enumerate(phase_offsets):
                    offset = int(relative_offsets[year_position, phase_position])
                    stream.seek(offset * chunk_bytes)
                    phase_payload = stream.read(chunk_bytes)
                    if len(phase_payload) != chunk_bytes:
                        raise RuntimeError("phase extraction reached an unexpected EOF")
                    sample = read_restart_payload(phase_payload, layout)
                    expected_nt = sparse.nt + offset
                    if sample.nt != expected_nt:
                        raise RuntimeError(
                            f"replay block {block_index}, phase {int(phase)} "
                            f"has NT={sample.nt}, expected {expected_nt}"
                        )
                    expected_time = np.float32(
                        fresh_generator.model_time(expected_nt)
                    )
                    if (
                        np.float32(sample.time_months).tobytes()
                        != expected_time.tobytes()
                    ):
                        raise RuntimeError(
                            "phase replay clock value is not bitwise exact"
                        )
                    samples[phase_position, year_position] = (
                        sample.native_controls.astype(FLOAT32_DTYPE)
                    )
    if not np.isfinite(samples).all():
        raise RuntimeError(f"replay block {block_index} contains nonfinite controls")
    return block_index, samples, float(runtime["elapsed_seconds"])


def _flush_progress(
    arrays: Sequence[np.memmap],
    progress_path: Path,
    binding: dict[str, Any],
    completed: dict[int, str],
    model_runtime: dict[int, float],
) -> None:
    for array in arrays:
        array.flush()
    write_json(
        progress_path,
        _progress_document(
            binding, completed=completed, model_runtime=model_runtime
        ),
        overwrite=True,
    )


def _audit_reference_overlap(
    arrays: Sequence[np.ndarray],
    *,
    phases: np.ndarray,
    years_per_source_checkpoint: int,
    source_base_input_indices: np.ndarray,
    reference_cache_dir: Path,
    expected_history_sha256: str,
    expected_executable_sha256: str,
    state_size: int,
) -> dict[str, Any]:
    manifest_path = reference_cache_dir / "capture_complete.json"
    manifest = load_json(manifest_path)
    expected_count = int(source_base_input_indices.size)
    required = {
        "phase_offsets": phases.tolist(),
        "checkpoint_count": expected_count,
        "history_sha256": expected_history_sha256,
        "phase_replay_executable_sha256": expected_executable_sha256,
        "base_input_indices_sha256": sha256_array(source_base_input_indices),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"reference overlap cache differs for {key}")
    records = manifest.get("cache_files")
    if not isinstance(records, list) or len(records) != phases.size:
        raise ValueError("reference overlap cache has the wrong file count")

    reference_file_records: list[dict[str, Any]] = []
    annual_rows = (
        np.arange(expected_count, dtype=np.int64)
        * years_per_source_checkpoint
    )
    for position, (phase, record) in enumerate(zip(phases, records, strict=True)):
        if not isinstance(record, dict):
            raise ValueError("reference overlap cache has an invalid file record")
        expected_name = f"native_phase_{int(phase):02d}.npy"
        if record.get("file") != expected_name:
            raise ValueError("reference overlap cache phase filename differs")
        reference_path = reference_cache_dir / expected_name
        if (
            not reference_path.is_file()
            or reference_path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(reference_path) != record.get("sha256")
        ):
            raise ValueError(f"reference cache file fails provenance: {reference_path}")
        reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
        if reference.shape != (expected_count, state_size):
            raise ValueError("reference overlap cache has an unexpected shape")
        if reference.dtype != FLOAT32_DTYPE:
            raise ValueError("reference overlap cache has an unexpected dtype")
        candidate = np.asarray(arrays[position][annual_rows])
        if not np.array_equal(candidate, reference):
            unequal = np.argwhere(candidate != reference)
            first = unequal[0].tolist() if unequal.size else None
            raise RuntimeError(
                f"annual cache fails full phase-{int(phase)} overlap audit; "
                f"first unequal coordinate {first}"
            )
        reference_file_records.append(
            {
                "phase_offset": int(phase),
                "file": expected_name,
                "sha256": str(record["sha256"]),
                "rows_compared": expected_count,
                "values_compared": expected_count * state_size,
                "bitwise_equal": True,
            }
        )
    return {
        "status": "passed_full_bitwise_overlap",
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": sha256_file(manifest_path),
        "mapping": (
            "reference row j equals annual row "
            "j * years_per_source_checkpoint (within-decade y=0)"
        ),
        "annual_row_indices_sha256": sha256_array(annual_rows),
        "rows_per_phase": expected_count,
        "total_values_compared": expected_count * state_size * phases.size,
        "files": reference_file_records,
    }


def _audit_independent_overlap_replay(
    arrays: Sequence[np.ndarray],
    *,
    phases: np.ndarray,
    schedule: CaptureSchedule,
    history: Path,
    chunk_bytes: int,
    spinup_steps: int,
    layout: IndependentControlLayout,
    source_dir: Path,
    executable: Path,
    jobs: int,
    expected_history_sha256: str,
    expected_executable_sha256: str,
) -> dict[str, Any]:
    """Independently replay and audit every source-checkpoint overlap row.

    Each sparse production checkpoint is replayed only through the last requested
    phase in its first year.  The transient samples are compared by float32 bit
    pattern with rows ``0, years_per_source_checkpoint, ...`` of the completed
    annual cache and discarded immediately after hashing.
    """

    phase_offsets = np.asarray(phases, dtype=np.int64)
    if phase_offsets.ndim != 1 or phase_offsets.size == 0:
        raise ValueError("independent overlap phases must be a nonempty vector")
    if jobs <= 0:
        raise ValueError("independent overlap jobs must be positive")
    if len(arrays) != phase_offsets.size:
        raise ValueError("independent overlap array and phase counts differ")
    expected_shape = (schedule.sample_count, layout.compact_size)
    for array in arrays:
        if array.shape != expected_shape or array.dtype != FLOAT32_DTYPE:
            raise ValueError("independent overlap annual cache schema differs")

    annual_rows = (
        np.arange(schedule.source_checkpoint_count, dtype=np.int64)
        * schedule.years_per_source_checkpoint
    )
    if annual_rows.size == 0 or int(annual_rows[-1]) >= schedule.sample_count:
        raise RuntimeError("independent overlap row mapping leaves the annual cache")
    relative_offsets = phase_offsets[None, :]
    replay_hashes: dict[int, str] = {}
    model_runtime: dict[int, float] = {}
    started = time.monotonic()

    def run_one(item: tuple[int, int, int]) -> tuple[int, np.ndarray, float]:
        block, chunk, base = item
        return _replay_block(
            block_index=block,
            chunk_index=chunk,
            sparse_base_input_index=base,
            history=history,
            chunk_bytes=chunk_bytes,
            spinup_steps=spinup_steps,
            phase_offsets=phase_offsets,
            relative_offsets=relative_offsets,
            layout=layout,
            source_dir=source_dir,
            executable=executable,
        )

    work = [
        (block, int(chunk), int(base))
        for block, (chunk, base) in enumerate(
            zip(
                schedule.source_chunk_indices,
                schedule.source_base_input_indices,
                strict=True,
            )
        )
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        for batch_start in range(0, len(work), jobs):
            futures = [
                executor.submit(run_one, item)
                for item in work[batch_start : batch_start + jobs]
            ]
            for future in concurrent.futures.as_completed(futures):
                block, samples, elapsed = future.result()
                if block in replay_hashes:
                    raise RuntimeError(
                        f"independent overlap replay repeated block {block}"
                    )
                expected_sample_shape = (
                    phase_offsets.size,
                    1,
                    layout.compact_size,
                )
                if samples.shape != expected_sample_shape:
                    raise RuntimeError(
                        f"independent overlap block {block} has shape "
                        f"{samples.shape}, expected {expected_sample_shape}"
                    )
                replayed = np.ascontiguousarray(samples[:, 0], dtype=FLOAT32_DTYPE)
                annual = np.ascontiguousarray(
                    np.stack(
                        [array[int(annual_rows[block])] for array in arrays],
                        axis=0,
                    ),
                    dtype=FLOAT32_DTYPE,
                )
                unequal_bits = replayed.view("<u4") != annual.view("<u4")
                if np.any(unequal_bits):
                    first = np.argwhere(unequal_bits)[0].tolist()
                    raise RuntimeError(
                        f"annual cache fails independent phase-"
                        f"{int(phase_offsets[first[0]])} overlap replay at block "
                        f"{block}, annual row {int(annual_rows[block])}; first "
                        f"unequal phase/state coordinate {first}"
                    )
                replay_hash = _block_hash(replayed[:, None, :])
                annual_hash = _block_hash(annual[:, None, :])
                if replay_hash != annual_hash:
                    raise RuntimeError(
                        f"independent overlap block {block} hash differs after "
                        "bitwise equality"
                    )
                replay_hashes[block] = replay_hash
                model_runtime[block] = elapsed

    if len(replay_hashes) != schedule.source_checkpoint_count:
        raise RuntimeError("independent overlap audit omitted a source checkpoint")
    ordered_hashes = [replay_hashes[index] for index in range(len(work))]
    return {
        "schema_version": 1,
        "status": "passed_full_bitwise_independent_overlap_replay",
        "validation_mode": "independent_overlap_replay",
        "reference_cache_used": False,
        "mapping": (
            "source block j y=0 replay equals annual row "
            "j * years_per_source_checkpoint"
        ),
        "phase_offsets": phase_offsets.tolist(),
        "annual_row_indices_sha256": sha256_array(annual_rows),
        "rows_per_phase": schedule.source_checkpoint_count,
        "total_rows_compared": (
            schedule.source_checkpoint_count * phase_offsets.size
        ),
        "total_values_compared": (
            schedule.source_checkpoint_count
            * phase_offsets.size
            * layout.compact_size
        ),
        "source_checkpoint_count": schedule.source_checkpoint_count,
        "replay_relative_offsets_sha256": sha256_array(relative_offsets),
        "replayed_block_sha256": {
            str(index): replay_hashes[index] for index in range(len(work))
        },
        "ordered_block_hashes_sha256": sha256_json(
            {"replayed_block_sha256": ordered_hashes}
        ),
        "provenance": {
            "history_sha256": expected_history_sha256,
            "phase_replay_executable_sha256": expected_executable_sha256,
            "source_chunk_indices_sha256": sha256_array(
                schedule.source_chunk_indices
            ),
            "source_base_input_indices_sha256": sha256_array(
                schedule.source_base_input_indices
            ),
        },
        "storage": {
            "persistent_reference_cache_created": False,
            "persistent_reference_cache_bytes": 0,
            "policy": "each independently replayed block is discarded after audit",
        },
        "runtime": {
            "jobs": jobs,
            "wall_seconds": time.monotonic() - started,
            "sum_model_runtime_seconds": float(sum(model_runtime.values())),
            "maximum_model_runtime_seconds": float(max(model_runtime.values())),
            "model_runtime_seconds_by_block": {
                str(index): model_runtime[index] for index in range(len(work))
            },
            "field_output_bytes": 0,
            "maximum_replay_offset_steps": int(np.max(relative_offsets)),
            "transient_history_bytes_per_worker": (
                (int(np.max(relative_offsets)) + 1) * chunk_bytes
            ),
        },
    }


def _cache_file_records(
    paths: Sequence[Path], *, sample_count: int, state_size: int
) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (sample_count, state_size) or array.dtype != FLOAT32_DTYPE:
            raise RuntimeError(f"completed cache has a bad schema: {path}")
        records.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "shape": [sample_count, state_size],
                "dtype": FLOAT32_DTYPE.str,
            }
        )
    return records


def _validate_completed_cache(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "capture_complete.json"
    report = load_json(report_path)
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("completed annual cache has an unknown schema")
    if report.get("status") != "complete":
        raise ValueError("annual cache completion report is not complete")
    records = report.get("cache_files")
    if not isinstance(records, list) or not records:
        raise ValueError("annual cache completion report has no files")
    for raw in records:
        if not isinstance(raw, dict) or not isinstance(raw.get("file"), str):
            raise ValueError("annual cache completion file record is invalid")
        path = output_dir / raw["file"]
        if (
            not path.is_file()
            or path.stat().st_size != int(raw.get("size_bytes", -1))
            or sha256_file(path) != raw.get("sha256")
        ):
            raise ValueError(f"completed annual cache file changed: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != raw.get("shape") or array.dtype.str != raw.get("dtype"):
            raise ValueError(f"completed annual cache schema changed: {path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    phases = _phase_offsets(args.phase_offsets)
    history = _resolve(args.history)
    production_report_path = _resolve(args.production_report)
    data_dir = _resolve(args.data_dir)
    data_metadata_path = data_dir / "metadata.json"
    state_manifest = _resolve(args.state_manifest)
    source_dir = _resolve(args.source_dir)
    executable = _resolve(args.executable)
    reference_cache_dir = (
        None
        if args.independent_overlap_replay
        else _resolve(args.reference_cache_dir)
    )
    output_dir = _resolve(args.output_dir)
    progress_path = output_dir / "capture_progress.json"
    completion_path = output_dir / "capture_complete.json"

    required = [
        history,
        production_report_path,
        data_metadata_path,
        state_manifest,
        source_dir,
        executable,
    ]
    if reference_cache_dir is not None:
        required.append(reference_cache_dir / "capture_complete.json")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    resume_completed_cache = completion_path.is_file() and args.resume
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif not args.resume:
            raise FileExistsError(
                f"output directory exists; use --resume or --overwrite: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    production_report = load_json(production_report_path)
    data_metadata = load_json(data_metadata_path)
    configuration = production_report.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("production report has no configuration")
    split = data_metadata.get("chronological_split", {}).get("train")
    if not isinstance(split, list) or len(split) != 2:
        raise ValueError("processed metadata has no training interval")
    training_interval = (int(split[0]), int(split[1]))
    integration = data_metadata.get("integration", {})
    if int(integration.get("steps_per_year", -1)) != fresh_generator.STEPS_PER_YEAR:
        raise ValueError("processed data and generator have different annual cadence")
    if int(configuration.get("spinup_steps", -1)) <= 0:
        raise ValueError("production report has no positive spinup duration")
    source_stride = int(configuration.get("checkpoint_steps", -1))
    if source_stride <= 0:
        raise ValueError("production report has no positive checkpoint stride")

    layout = load_independent_control_layout(state_manifest)
    chunk_indices, nt_values = _history_samples(
        history, production_report=production_report, layout=layout
    )
    schedule = build_capture_schedule(
        chunk_indices,
        nt_values,
        spinup_steps=int(configuration["spinup_steps"]),
        source_checkpoint_stride_steps=source_stride,
        training_interval=training_interval,
        phase_offsets=phases,
    )
    binding = _static_binding(
        history=history,
        production_report_path=production_report_path,
        data_metadata_path=data_metadata_path,
        state_manifest=state_manifest,
        source_dir=source_dir,
        executable=executable,
        reference_cache_dir=reference_cache_dir,
        production_report=production_report,
        training_interval=training_interval,
        phase_offsets=phases,
        schedule=schedule,
        state_size=layout.compact_size,
    )
    if resume_completed_cache:
        report = _validate_completed_cache(output_dir)
        for key, expected in binding.items():
            if key == "status":
                continue
            if report.get(key) != expected:
                raise ValueError(f"completed cache binding differs for {key}")
        print(
            f"Validated complete annual native cache with "
            f"{report['sample_count']} samples: {output_dir}"
        )
        return 0
    paths = _cache_paths(output_dir, phases)
    if args.resume:
        arrays = _open_arrays(
            paths,
            sample_count=schedule.sample_count,
            state_size=layout.compact_size,
            mode="r+",
        )
        if not progress_path.is_file():
            raise ValueError("resume requested but capture_progress.json is missing")
        completed, model_runtime = _load_progress(
            progress_path,
            expected_binding=binding,
            arrays=arrays,
            years_per_block=schedule.years_per_source_checkpoint,
        )
    else:
        arrays = _create_arrays(
            paths,
            sample_count=schedule.sample_count,
            state_size=layout.compact_size,
        )
        completed = {}
        model_runtime = {}
        _flush_progress(arrays, progress_path, binding, completed, model_runtime)

    work = [
        (block, int(chunk), int(base))
        for block, (chunk, base) in enumerate(
            zip(
                schedule.source_chunk_indices,
                schedule.source_base_input_indices,
                strict=True,
            )
        )
        if block not in completed
    ]
    started = time.monotonic()
    dirty = 0

    def run_one(item: tuple[int, int, int]) -> tuple[int, np.ndarray, float]:
        block, chunk, base = item
        return _replay_block(
            block_index=block,
            chunk_index=chunk,
            sparse_base_input_index=base,
            history=history,
            chunk_bytes=int(production_report["checkpoint_chunk_bytes"]),
            spinup_steps=int(configuration["spinup_steps"]),
            phase_offsets=phases,
            relative_offsets=schedule.relative_offsets,
            layout=layout,
            source_dir=source_dir,
            executable=executable,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for batch_start in range(0, len(work), args.jobs):
                batch = work[batch_start : batch_start + args.jobs]
                futures = [executor.submit(run_one, item) for item in batch]
                for future in concurrent.futures.as_completed(futures):
                    block, samples, elapsed = future.result()
                    start = block * schedule.years_per_source_checkpoint
                    stop = start + schedule.years_per_source_checkpoint
                    for position, array in enumerate(arrays):
                        array[start:stop] = samples[position]
                    completed[block] = _block_hash(samples)
                    model_runtime[block] = elapsed
                    dirty += 1
                    if dirty >= args.progress_interval:
                        _flush_progress(
                            arrays,
                            progress_path,
                            binding,
                            completed,
                            model_runtime,
                        )
                        dirty = 0
                        print(
                            f"Captured {len(completed)}/"
                            f"{schedule.source_checkpoint_count} source blocks"
                        )
    finally:
        if dirty:
            _flush_progress(
                arrays, progress_path, binding, completed, model_runtime
            )

    if len(completed) != schedule.source_checkpoint_count:
        raise RuntimeError("capture ended before every source block completed")
    for array in arrays:
        array.flush()
    readonly = _open_arrays(
        paths,
        sample_count=schedule.sample_count,
        state_size=layout.compact_size,
        mode="r",
    )
    if reference_cache_dir is None:
        overlap = _audit_independent_overlap_replay(
            readonly,
            phases=phases,
            schedule=schedule,
            history=history,
            chunk_bytes=int(production_report["checkpoint_chunk_bytes"]),
            spinup_steps=int(configuration["spinup_steps"]),
            layout=layout,
            source_dir=source_dir,
            executable=executable,
            jobs=args.jobs,
            expected_history_sha256=binding["history_sha256"],
            expected_executable_sha256=binding[
                "phase_replay_executable_sha256"
            ],
        )
    else:
        overlap = _audit_reference_overlap(
            readonly,
            phases=phases,
            years_per_source_checkpoint=schedule.years_per_source_checkpoint,
            source_base_input_indices=schedule.source_base_input_indices,
            reference_cache_dir=reference_cache_dir,
            expected_history_sha256=binding["history_sha256"],
            expected_executable_sha256=binding[
                "phase_replay_executable_sha256"
            ],
            state_size=layout.compact_size,
        )
    file_records = _cache_file_records(
        paths, sample_count=schedule.sample_count, state_size=layout.compact_size
    )
    report = dict(binding)
    report.update(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "cache_files": file_records,
            "overlap_audit": overlap,
            "runtime": {
                "jobs": args.jobs,
                "source_blocks_replayed_this_invocation": len(work),
                "wall_seconds_this_invocation": time.monotonic() - started,
                "sum_model_runtime_seconds": float(sum(model_runtime.values())),
                "maximum_model_runtime_seconds": float(max(model_runtime.values())),
                "field_output_bytes": 0,
                "transient_history_bytes_per_worker": (
                    (schedule.maximum_replay_offset_steps + 1)
                    * int(production_report["checkpoint_chunk_bytes"])
                ),
            },
        }
    )
    write_json(completion_path, report, overwrite=False)
    progress_path.unlink()
    print(
        f"Captured and validated {schedule.sample_count} annual native states "
        f"at {phases.size} phases: {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
