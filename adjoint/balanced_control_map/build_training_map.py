#!/usr/bin/env python3
"""Build a training-only empirical balanced-control map for fresh core4.

The production history contains a complete native checkpoint every ten model
years.  This builder pairs the checkpoints whose *next* compact output lies in
the 10,000-year training block with that fresh core4 state.  All retained
samples share one annual phase.  No validation/test state, AGOP vector, CNN
forecast, or future target is used to fit the map.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import generate_fresh_zc_dataset as fresh_generator  # noqa: E402

from adjoint.balanced_control_map import (  # noqa: E402
    BalancedControlMap,
    IndependentControlLayout,
    fit_balanced_control_map,
    load_independent_control_layout,
    paired_reconstruction_diagnostics,
    read_restart_payload,
    read_restart_sample,
)

SCRIPT_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1
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


def fraction(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 < value < 0.5:
        raise argparse.ArgumentTypeError("value must lie strictly between 0 and 0.5")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
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
        help="Release-certified source/runtime tree used only for phase replay.",
    )
    result.add_argument(
        "--executable",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3/source/zeqfc1"),
        help="Release-certified forward executable used only for phase replay.",
    )
    result.add_argument(
        "--reference-input-index",
        type=int,
        default=402_758,
        help=(
            "Only this index's annual phase is used. The default is locked neutral "
            "member 04 at release_index minus the nine-step adjustment window."
        ),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/zc_balanced_control_map/training-phase26-rank36"),
    )
    result.add_argument("--rank", type=positive_int, default=36)
    result.add_argument(
        "--jobs", type=positive_int, default=min(4, os.cpu_count() or 1)
    )
    result.add_argument(
        "--ridge-fraction",
        type=nonnegative_float,
        default=1.0e-3,
        help="Tikhonov alpha divided by the leading retained eigenvalue.",
    )
    result.add_argument(
        "--validation-fraction",
        type=fraction,
        default=0.2,
        help="Chronological training-only tail reserved for out-of-fit diagnostics.",
    )
    result.add_argument("--dot-test-seed", type=int, default=7301)
    result.add_argument("--dot-test-count", type=positive_int, default=8)
    result.add_argument("--overwrite", action="store_true")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def collect_training_native_samples(
    history: Path,
    *,
    production_report: dict[str, Any],
    training_interval: tuple[int, int],
    layout: IndependentControlLayout,
    reference_input_index: int,
    source_dir: Path,
    executable: Path,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay sparse checkpoints to one requested annual phase.

    For a phase-aligned native checkpoint immediately before fresh input row
    ``i``, the paired observation is exactly row ``i``.  The model replay is
    necessary: pairing an unevolved sparse checkpoint with a later-phase row
    would silently break the native/observation alignment.
    """

    chunk_bytes = int(production_report["checkpoint_chunk_bytes"])
    checkpoint_count = int(production_report["checkpoint_count"])
    configuration = production_report["configuration"]
    spinup_steps = int(configuration["spinup_steps"])
    checkpoint_steps = int(configuration["checkpoint_steps"])
    steps_per_year = 36
    if checkpoint_steps % steps_per_year:
        raise ValueError("checkpoint stride is not an integer number of model years")
    phase_offset = reference_input_index % steps_per_year
    expected_size = chunk_bytes * checkpoint_count
    if history.stat().st_size != expected_size:
        raise ValueError(
            f"history size is {history.stat().st_size}, expected {expected_size}"
        )
    expected_hash = production_report.get("history_sha256")
    actual_hash = sha256_file(history)
    if expected_hash != actual_hash:
        raise ValueError("production history fails its recorded SHA-256")

    train_start, train_stop = training_interval
    rows: list[np.ndarray] = []
    input_indices: list[int] = []
    clocks: list[float] = []
    nt_values: list[int] = []
    chunk_indices: list[int] = []
    with history.open("rb") as stream:
        for chunk_index in range(checkpoint_count):
            payload = stream.read(chunk_bytes)
            if len(payload) != chunk_bytes:
                raise ValueError(f"truncated history checkpoint {chunk_index}")
            sample = read_restart_payload(payload, layout)
            base_input_index = sample.nt - spinup_steps
            input_index = base_input_index + phase_offset
            if train_start <= input_index < train_stop:
                if phase_offset == 0:
                    rows.append(sample.native_controls)
                input_indices.append(input_index)
                clocks.append(
                    float(fresh_generator.model_time(sample.nt + phase_offset))
                )
                nt_values.append(sample.nt + phase_offset)
                chunk_indices.append(chunk_index)
    if not input_indices:
        raise RuntimeError("no production checkpoints map into the training interval")
    indices = np.asarray(input_indices, dtype=np.int64)
    if np.any(np.diff(indices) != checkpoint_steps):
        raise RuntimeError(
            "selected training checkpoints do not have the expected stride"
        )
    expected_count = math.ceil(
        (train_stop - max(train_start, indices[0])) / checkpoint_steps
    )
    if indices.size != expected_count:
        raise RuntimeError(
            f"selected {indices.size} checkpoints, expected {expected_count}"
        )

    replay_elapsed: list[float] = []
    wall_started = time.monotonic()
    if phase_offset:
        def replay_one(item: tuple[int, int, int]) -> tuple[np.ndarray, float]:
            chunk_index, input_index, target_nt = item
            with history.open("rb") as source:
                source.seek(chunk_index * chunk_bytes)
                payload = source.read(chunk_bytes)
            if len(payload) != chunk_bytes:
                raise RuntimeError(f"could not reread history chunk {chunk_index}")
            sparse = read_restart_payload(payload, layout)
            expected_target_nt = sparse.nt + phase_offset
            if target_nt != expected_target_nt:
                raise RuntimeError("phase-replay target clock changed")
            with tempfile.TemporaryDirectory(prefix="zc-balanced-phase-") as temporary:
                temporary_root = Path(temporary)
                restart = temporary_root / "sparse.hst"
                restart.write_bytes(payload)
                run_dir = temporary_root / "run"
                sparse_time = fresh_generator.model_time(sparse.nt)
                exact_time = fresh_generator.model_time(target_nt)
                fresh_generator.prepare_run(
                    source_dir,
                    executable,
                    run_dir,
                    nstart=3,
                    tfind=sparse_time,
                    tzero=sparse_time,
                    tend=exact_time,
                    ntape=0,
                    nrewnd=11,
                    nic=0,
                    write_start=exact_time + 1.0,
                    write_end=exact_time + 1.0,
                    restart=restart,
                )
                runtime = fresh_generator.run_model(run_dir)
                aligned = read_restart_sample(run_dir / "outhst", layout)
                if aligned.nt != target_nt:
                    raise RuntimeError(
                        f"phase replay for input {input_index} ended at NT "
                        f"{aligned.nt}, expected {target_nt}"
                    )
                return aligned.native_controls, float(runtime["elapsed_seconds"])

        work = list(zip(chunk_indices, input_indices, nt_values, strict=True))
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(replay_one, work))
        rows = [result[0] for result in results]
        replay_elapsed = [result[1] for result in results]
    if len(rows) != indices.size:
        raise RuntimeError("native phase-aligned sample count mismatch")
    replay_wall_seconds = time.monotonic() - wall_started
    return np.stack(rows), indices, {
        "checkpoint_count": int(indices.size),
        "checkpoint_stride_steps": checkpoint_steps,
        "reference_input_index_used_for_phase_only": reference_input_index,
        "reference_phase_offset_steps": phase_offset,
        "first_input_index": int(indices[0]),
        "last_input_index": int(indices[-1]),
        "first_checkpoint_nt": int(nt_values[0]),
        "last_checkpoint_nt": int(nt_values[-1]),
        "first_checkpoint_time_months": float(clocks[0]),
        "last_checkpoint_time_months": float(clocks[-1]),
        "history_sha256": actual_hash,
        "history_chunk_bytes": chunk_bytes,
        "phase_alignment": {
            "method": (
                "direct sparse checkpoint"
                if phase_offset == 0
                else "authentic forward replay from every sparse checkpoint"
            ),
            "forward_steps_per_checkpoint": phase_offset,
            "parallel_jobs": jobs,
            "wall_seconds": replay_wall_seconds,
            "sum_model_runtime_seconds": float(sum(replay_elapsed)),
            "maximum_model_runtime_seconds": (
                None if not replay_elapsed else float(max(replay_elapsed))
            ),
        },
    }


def load_standardized_core4_samples(
    data_dir: Path,
    indices: np.ndarray,
    normalization: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load fresh core4 in the saved training-standardized coordinate system."""

    metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path)
    field_records = metadata.get("field_files")
    if not isinstance(field_records, dict):
        raise ValueError("fresh metadata has no field_files mapping")
    shape = metadata.get("integration", {}).get("field_shape")
    if shape is None:
        shape = metadata.get("clean_data", {}).get("field_shape")
    rows = indices.size
    raw = np.empty((rows, 4 * 20 * 27), dtype=np.float64)
    field_hashes: dict[str, str] = {}
    for field_index, field in enumerate(CORE4_FIELDS):
        record = field_records.get(field)
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise ValueError(f"fresh metadata has no file record for {field}")
        field_path = data_dir / record["file"]
        array = np.load(field_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[1:] != (20, 27):
            raise ValueError(f"unexpected fresh field shape for {field}: {array.shape}")
        raw[:, field_index * 540 : (field_index + 1) * 540] = np.asarray(
            array[indices], dtype=np.float64
        ).reshape(rows, 540)
        if record.get("sha256"):
            field_hashes[field] = str(record["sha256"])
    with np.load(normalization, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
        count = int(archive["count"])
    if mean.shape != (2162,) or scale.shape != (2162,):
        raise ValueError("normalization is not the fresh core4-plus-phase contract")
    if np.any(scale[:2160] <= 0.0) or not np.isfinite(scale[:2160]).all():
        raise ValueError("core4 normalization scale is invalid")
    standardized = (raw - mean[:2160]) / scale[:2160]
    phase_path = data_dir / metadata["phase"]["file"]
    phases = np.asarray(
        np.load(phase_path, mmap_mode="r", allow_pickle=False)[indices],
        dtype=np.float64,
    )
    phase_spread = np.max(np.abs(phases - phases[0]), axis=0)
    if np.any(phase_spread > 2.0e-6):
        raise RuntimeError("production checkpoints are not all at one annual phase")
    return standardized, {
        "metadata_sha256": sha256_file(metadata_path),
        "normalization_sha256": sha256_file(normalization),
        "normalization_training_count": count,
        "field_sha256_from_metadata": field_hashes,
        "annual_phase_sin_cos": phases[0].tolist(),
        "maximum_absolute_phase_spread": float(np.max(phase_spread)),
        "fresh_state_semantics": (
            "Each native checkpoint is immediately before the paired compact "
            "output update. H1/U1/V1 carried by the checkpoint equal the paired "
            "fresh fields; SST is advanced once by SSTA before output."
        ),
    }


def transpose_tests(
    model: BalancedControlMap,
    *,
    seed: int,
    count: int,
) -> dict[str, Any]:
    """Exercise B/B.T and G/G.T in their public packed interfaces."""

    rng = np.random.default_rng(seed)
    native_defects: list[float] = []
    observation_defects: list[float] = []
    for _ in range(count):
        control = rng.standard_normal(model.rank)
        packed_seed = rng.standard_normal(model.layout.packed_size)
        left = float(model.apply_B(control, packed=True) @ packed_seed)
        right = float(control @ model.apply_BT(packed_seed, packed=True))
        scale = max(abs(left), abs(right), 1.0)
        native_defects.append(abs(left - right) / scale)

        observation_seed = rng.standard_normal(model.observation_size)
        left = float(model.apply_G(control) @ observation_seed)
        right = float(control @ model.apply_GT(observation_seed))
        scale = max(abs(left), abs(right), 1.0)
        observation_defects.append(abs(left - right) / scale)
    return {
        "seed": seed,
        "direction_count": count,
        "maximum_B_BT_relative_dot_defect": float(max(native_defects)),
        "maximum_G_GT_relative_dot_defect": float(max(observation_defects)),
        "status": (
            "passed"
            if max((*native_defects, *observation_defects)) <= 5.0e-13
            else "failed"
        ),
    }


def _prepare_output(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "balanced_control_map.npz"
    report = output_dir / "report.json"
    existing = [path for path in (artifact, report) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output exists; pass --overwrite: " + ", ".join(map(str, existing))
        )
    return artifact, report


def main() -> int:
    args = parser().parse_args()
    history = _resolve(args.history)
    production_report_path = _resolve(args.production_report)
    data_dir = _resolve(args.data_dir)
    normalization = _resolve(args.normalization)
    state_manifest = _resolve(args.state_manifest)
    source_dir = _resolve(args.source_dir)
    executable = _resolve(args.executable)
    output_dir = _resolve(args.output_dir)
    artifact_path, report_path = _prepare_output(output_dir, args.overwrite)
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

    metadata = load_json(data_dir / "metadata.json")
    split = metadata.get("chronological_split", {}).get("train")
    if not isinstance(split, list) or len(split) != 2:
        raise ValueError("fresh metadata has no two-value training split")
    training_interval = (int(split[0]), int(split[1]))
    production_report = load_json(production_report_path)
    if not 0 <= args.reference_input_index < int(
        metadata["integration"]["retained_steps"]
    ):
        raise ValueError("reference input index is outside the fresh data")
    expected_executables = {
        str(entry["event_replay_executable_sha256"])
        for entry in metadata.get("event_restart_checkpoints", [])
        if isinstance(entry, dict) and entry.get("event_replay_executable_sha256")
    }
    executable_hash = sha256_file(executable)
    if expected_executables and executable_hash not in expected_executables:
        raise ValueError(
            "phase-replay executable is not release-bound by fresh metadata"
        )
    layout = load_independent_control_layout(state_manifest)
    native, indices, history_report = collect_training_native_samples(
        history,
        production_report=production_report,
        training_interval=training_interval,
        layout=layout,
        reference_input_index=args.reference_input_index,
        source_dir=source_dir,
        executable=executable,
        jobs=args.jobs,
    )
    observations, observation_report = load_standardized_core4_samples(
        data_dir, indices, normalization
    )
    if args.rank >= native.shape[0]:
        raise ValueError("rank must be smaller than the selected sample count")

    validation_count = max(2, int(round(args.validation_fraction * native.shape[0])))
    fit_count = native.shape[0] - validation_count
    if args.rank >= fit_count:
        raise ValueError("rank must be smaller than the cross-validation fit count")
    diagnostic_map = fit_balanced_control_map(
        native[:fit_count],
        observations[:fit_count],
        layout=layout,
        rank=args.rank,
    )
    heldout = paired_reconstruction_diagnostics(
        diagnostic_map,
        native[fit_count:],
        observations[fit_count:],
        ridge_fraction=args.ridge_fraction,
    )
    final_map = fit_balanced_control_map(
        native,
        observations,
        layout=layout,
        rank=args.rank,
    )
    dot_report = transpose_tests(
        final_map, seed=args.dot_test_seed, count=args.dot_test_count
    )
    if dot_report["status"] != "passed":
        raise RuntimeError("balanced-map transpose tests failed")

    source_metadata = {
        "script_version": SCRIPT_VERSION,
        "state_manifest_sha256": sha256_file(state_manifest),
        "production_report_sha256": sha256_file(production_report_path),
        "phase_replay_executable_sha256": executable_hash,
        "training_input_indices": indices.tolist(),
        "construction": "B=Xc.T@U_r/sqrt(n-1); G=Zc.T@U_r/sqrt(n-1)",
    }
    temporary_artifact = artifact_path.with_name(artifact_path.name + ".tmp.npz")
    final_map.save(temporary_artifact, metadata=source_metadata)
    loaded_map, loaded_metadata = BalancedControlMap.load(temporary_artifact)
    if loaded_metadata != source_metadata:
        raise RuntimeError("serialized map metadata failed its round trip")
    roundtrip_test = transpose_tests(
        loaded_map, seed=args.dot_test_seed, count=args.dot_test_count
    )
    if roundtrip_test["status"] != "passed":
        raise RuntimeError("serialized balanced map failed transpose tests")
    os.replace(temporary_artifact, artifact_path)

    retained_fraction = float(
        np.sum(final_map.eigenvalues) / final_map.observation_variance_trace
    )
    gram = final_map.G.T @ final_map.G
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "validated_empirical_low_rank_secant_map",
        "map_artifact": {
            "file": artifact_path.name,
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
        "dimensions": {
            "packed_native_state": layout.packed_size,
            "compact_independent_native_controls": layout.compact_size,
            "standardized_core4_observations": final_map.observation_size,
            "reduced_controls": final_map.rank,
            "authentic_training_samples": final_map.sample_count,
        },
        "independent_control_segments": [
            {
                "name": segment.name,
                "packed_half_open": [segment.packed_start, segment.packed_stop],
                "compact_half_open": [segment.compact_start, segment.compact_stop],
                "shape": list(segment.shape),
                "order": segment.order,
                "role": segment.role,
                "stored_units": (
                    "degrees C" if segment.name == "TO" else "native ZC storage units"
                ),
            }
            for segment in layout.segments
        ],
        "construction": {
            "definition": source_metadata["construction"],
            "control_units": (
                "dimensionless covariance-whitened sample coordinates; ||a||^2 "
                "is the declared empirical Mahalanobis action"
            ),
            "B_units": (
                "Each B row has the native storage unit of its manifest coordinate "
                "per dimensionless reduced control. Mixed native rows must not be "
                "combined into an unscaled Euclidean importance norm."
            ),
            "G_units": "training-standardized core4 units per reduced control",
            "observation_eigenvalues": final_map.eigenvalues.tolist(),
            "observation_variance_trace": final_map.observation_variance_trace,
            "retained_observation_variance_fraction": retained_fraction,
            "retained_condition_number": float(
                final_map.eigenvalues[0] / final_map.eigenvalues[-1]
            ),
            "maximum_G_gram_absolute_defect": float(
                np.max(
                    np.abs(gram - np.diag(final_map.eigenvalues)), initial=0.0
                )
            ),
        },
        "regularized_lift": {
            "objective": "0.5*||G a-d||^2 + 0.5*alpha*||a||^2",
            "default_alpha": float(args.ridge_fraction * final_map.eigenvalues[0]),
            "default_alpha_fraction_of_leading_eigenvalue": args.ridge_fraction,
            "exact_inverse_claimed": False,
            "reason": (
                "G has rank at most the retained sample rank, far below 2160. "
                "The lift returns a regularized projection into range(G)."
            ),
        },
        "transpose_tests": dot_report,
        "serialized_roundtrip_transpose_tests": roundtrip_test,
        "chronological_training_only_holdout": {
            "fit_sample_count": fit_count,
            "holdout_sample_count": validation_count,
            "fit_first_input_index": int(indices[0]),
            "fit_last_input_index": int(indices[fit_count - 1]),
            "holdout_first_input_index": int(indices[fit_count]),
            "holdout_last_input_index": int(indices[-1]),
            **heldout.summary,
        },
        "source": {
            **history_report,
            **observation_report,
            "training_half_open_interval": list(training_interval),
            "production_report_sha256": sha256_file(production_report_path),
            "state_manifest_sha256": sha256_file(state_manifest),
            "phase_replay_executable_sha256": executable_hash,
        },
        "limitations": [
            "B spans empirical secants through authentic checkpoints; a finite "
            "linear combination is near, not exactly on, the nonlinear ZC manifold.",
            "The available production checkpoints sample one annual phase every "
            "ten model years. This artifact is phase-local and should not be used "
            "unchanged at distant phases without a new phase-matched ensemble.",
            "Only state-manifest independent controls AKB/UB/HB/V/TO are changed. "
            "Active boundary and atmospheric memory remain fixed and must rebalance.",
            "The paired SST is one SSTA update after the checkpoint; G is an empirical "
            "one-step secant image, not the exact differentiated observation map.",
            "Run23 gradients are provisional_major_tape_conditioned real32 products. "
            "B.T projection preserves that qualification and adds no certification.",
        ],
        "recommended_use": (
            "Apply modest controls around an authentic same/near-phase checkpoint, "
            "report ||a|| and the unresolved core4 residual, then allow at least one "
            "complete unforced ZC step. Validate new phases and large actions with "
            "actual forward replay."
        ),
    }
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
