#!/usr/bin/env python3
"""Run a release-constrained AGOP nudge through the authentic ZC model.

This is the end-to-end causal sufficiency experiment.  It constructs a
linearized release response from the validated one-step ZC tangent, chooses a
minimum-action control using *only* that release response, and then applies the
finite control to the nonlinear ZC model.  Ten-month ZC outcomes and CNN
forecasts are evaluated only after the controller and all doses have been
frozen.

The default is the locked neutral member 04 pilot.  ``--cohort full`` runs the
same protocol on all 20 future-blind neutral members selected by the earlier
ensemble experiment.  The primary controller uses nine coherently aligned,
phase-local balanced maps; ``--control-map constant`` retains the older fixed
phase-26 map as a declared sensitivity analysis.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The release-response workers invoke many small BLAS operations around
# external Fortran processes.  One BLAS thread per independent worker avoids
# oversubscription and the macOS OpenBLAS fork deadlock seen in the first cache
# attempt.  Explicit caller settings still take precedence.
for _thread_environment_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_environment_variable, "1")

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
import scipy.stats  # noqa: E402
import torch  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import benchmark_fresh_xai_methods as fresh_xai_benchmark  # noqa: E402
import generate_fresh_zc_dataset as fresh_generator  # noqa: E402

from adjoint.balanced_control_map import (  # noqa: E402
    BalancedControlMap,
    PhaseLocalControlStack,
)
from zc_xai.adjoint_chain import (  # noqa: E402
    DistributedControlOperator,
    three_mode_half_cosine_iau_basis,
)
from zc_xai.agop_control import (  # noqa: E402
    AGOPAlignmentSolution,
    solve_linearized_agop_alignment,
    unit_fixed_phase_direction,
)
from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    load_json,
    sha256_array,
    sha256_file,
    sha256_json,
    write_json,
    write_npz,
)
from zc_xai.native_observation import (  # noqa: E402
    FrozenCore4ObservationChain,
    PackedZCState,
    nino3_from_real_state,
)
from zc_xai.training import ExperimentSpec, load_experiment  # noqa: E402
from zc_xai.xai import AgopExplainer  # noqa: E402
from zc_xai.zc_controlled_bridge import (  # noqa: E402
    ControlledReplay,
    ZeroControlPath,
    build_zero_control_path,
    form_authentic_release_response,
    replay_control,
    verify_zero_control_replay,
)

SCRIPT_VERSION = "1.0.1"
REPORT_SCHEMA_VERSION = 1
CONTROL_STEPS = 9
RELEASE_BOUNDARIES = (9, 10)
TOTAL_STEPS = 40
PILOT_RELEASE_INDEX = 402_767
LOCKED_NEUTRAL_INDICES = np.asarray(
    (
        397_763,
        398_879,
        399_059,
        402_767,
        402_911,
        406_223,
        406_763,
        407_447,
        409_139,
        409_427,
        411_119,
        414_647,
        414_791,
        415_655,
        416_699,
        418_931,
        419_579,
        422_027,
        429_155,
        429_623,
    ),
    dtype=np.int64,
)
DEFAULT_FRACTIONS = (-0.5, 0.25, 0.5, 1.0)
DEFAULT_GAMMAS = (0.0, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
DEFAULT_PHASE_LOCAL_MAP_DIR = Path(
    "outputs/zc_balanced_control_map/training-phases26-34-common-rank36"
)
DEFAULT_CONSTANT_MAP_DIR = Path(
    "outputs/zc_balanced_control_map/training-phase26-rank36"
)
TEMPORAL_ACTION_SCALE = float(
    np.sum(three_mode_half_cosine_iau_basis(CONTROL_STEPS)[:, 0] ** 2)
)


@dataclass(frozen=True)
class BetaChoice:
    """A release-only choice from the fixed dimensionless penalty grid."""

    gamma: float
    beta: float
    target_cosine: float
    ideal_cosine: float
    predicted_cosine: float
    control_norm: float
    control_norm_inflation: float
    squared_action_inflation: float
    criterion_met: bool
    selection_status: str
    solution: AGOPAlignmentSolution


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/zc-v3"))
    parser.add_argument(
        "--agop-benchmark-dir",
        type=Path,
        default=Path(
            "outputs/fresh_agop_benchmark/"
            "core4-cnn-lead-10m-seed-000042-refs-all-batch-1024"
        ),
    )
    parser.add_argument(
        "--control-map",
        choices=("phase-local", "constant"),
        default="phase-local",
        help=(
            "Use the coherent phase-26...34 map stack (primary), or hold the "
            "older phase-26 map fixed for a sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--balanced-map-dir",
        type=Path,
        help="Override the validated artifact directory for --control-map.",
    )
    parser.add_argument(
        "--neutral-ensemble-dir",
        type=Path,
        default=Path("outputs/zc_agop_neutral_ensemble/core4-cnn-lead-10m-seed-000042"),
    )
    parser.add_argument(
        "--adjoint-validation-dir",
        type=Path,
        default=Path("outputs/zc_adjoint/kernel_replay_validation_run23_final3"),
    )
    parser.add_argument(
        "--input-cache-dir",
        type=Path,
        default=Path("outputs/zc_agop_adjoint_nudging_inputs"),
        help=(
            "Validated packed initial states and optional precomputed response "
            "matrices. Missing entries are generated locally."
        ),
    )
    parser.add_argument(
        "--primal-executable",
        type=Path,
        default=Path("adjoint/controlled_window/build/zc_one_step"),
    )
    parser.add_argument(
        "--tangent-executable",
        type=Path,
        default=Path("adjoint/controlled_window/build/zc_one_step_tangent"),
    )
    parser.add_argument(
        "--cohort",
        choices=("pilot", "full"),
        default="pilot",
        help="Locked member 04 only, or all 20 locked future-blind members.",
    )
    parser.add_argument(
        "--member",
        action="append",
        type=int,
        dest="members",
        help=(
            "Run one locked release input index; repeat for several. This "
            "overrides --cohort and enables safe independent cache workers."
        ),
    )
    parser.add_argument(
        "--response-only",
        action="store_true",
        help=(
            "Validate the zero path, form/cache A, write a compact report, and "
            "stop before solving or replaying finite controls."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lead-months", type=int, default=10)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Nonzero fractions of each neutral-to-extreme AGOP-coordinate gap.",
    )
    parser.add_argument(
        "--penalty-gammas",
        nargs="+",
        type=nonnegative_float,
        default=DEFAULT_GAMMAS,
        help="Dimensionless grid gamma=beta*||A_perp||_2^2.",
    )
    parser.add_argument("--minimum-range-cosine", type=nonnegative_float, default=0.5)
    parser.add_argument(
        "--target-range-cosine-fraction", type=positive_float, default=0.80
    )
    parser.add_argument(
        "--maximum-control-norm-inflation", type=positive_float, default=3.0
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Destination; by default uses a distinct pilot/full directory under "
            "outputs/zc_agop_adjoint_nudging."
        ),
    )
    parser.add_argument("--skip-data-checksums", action="store_true")
    parser.add_argument(
        "--recompute-responses",
        action="store_true",
        help="Ignore any response .npy files in the validated input cache.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (REPOSITORY_ROOT / expanded).resolve()


def _prepare_output(path: Path, overwrite: bool) -> Path:
    destination = _resolve(path)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {destination}\n"
                "Choose another --output-dir or pass --overwrite explicitly."
            )
        if destination == REPOSITORY_ROOT or REPOSITORY_ROOT not in destination.parents:
            raise ValueError("Refusing to replace a broad directory")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    return destination


def _fraction_label(value: float) -> str:
    prefix = "p" if value >= 0.0 else "m"
    return prefix + f"{abs(value):g}".replace(".", "p")


def _member_label(release_index: int) -> str:
    matches = np.flatnonzero(release_index == LOCKED_NEUTRAL_INDICES)
    if matches.size != 1:
        raise ValueError(f"{release_index} is not in the locked neutral cohort")
    return f"member_{int(matches[0]) + 1:02d}_i{release_index}"


def _cohort_indices(name: str) -> np.ndarray:
    if name == "pilot":
        return np.asarray([PILOT_RELEASE_INDEX], dtype=np.int64)
    if name == "full":
        return LOCKED_NEUTRAL_INDICES.copy()
    raise ValueError(f"unknown cohort {name!r}")


def _requested_indices(name: str, members: list[int] | None) -> np.ndarray:
    if not members:
        return _cohort_indices(name)
    requested = np.asarray(members, dtype=np.int64)
    if np.unique(requested).size != requested.size:
        raise ValueError("--member release indices must not be repeated")
    unknown = np.setdiff1d(requested, LOCKED_NEUTRAL_INDICES)
    if unknown.size:
        raise ValueError(f"--member contains unlocked release indices: {unknown}")
    return requested


def _validate_fraction_grid(values: list[float] | tuple[float, ...]) -> np.ndarray:
    fractions = np.asarray(values, dtype=np.float64)
    if (
        fractions.ndim != 1
        or fractions.size == 0
        or not np.isfinite(fractions).all()
        or np.any(fractions == 0.0)
    ):
        raise ValueError("fractions must be a finite nonempty list excluding zero")
    required = np.asarray(DEFAULT_FRACTIONS)
    if not all(np.any(np.isclose(fractions, value)) for value in required):
        raise ValueError("fractions must include -0.5, 0.25, 0.5, and 1.0")
    return np.asarray(sorted(set(float(value) for value in fractions)))


def _validate_gamma_grid(values: list[float] | tuple[float, ...]) -> np.ndarray:
    gammas = np.asarray(values, dtype=np.float64)
    if (
        gammas.ndim != 1
        or gammas.size == 0
        or not np.isfinite(gammas).all()
        or np.any(gammas < 0.0)
        or not np.any(gammas == 0.0)
    ):
        raise ValueError("penalty gammas must be finite/nonnegative and include zero")
    return np.asarray(sorted(set(float(value) for value in gammas)))


def _load_balanced_map(
    directory: Path,
    *,
    control_map_kind: str,
    normalization_path: Path,
    state_manifest_sha256: str,
) -> tuple[BalancedControlMap | PhaseLocalControlStack, np.ndarray, dict[str, Any]]:
    report_path = directory / "report.json"
    report = load_json(report_path)
    if control_map_kind == "phase-local":
        artifact_path = directory / "phase_local_control_stack.npz"
        expected_status = "validated_coherent_phase_local_training_secant_stack"
        recorded = report.get("artifact")
    elif control_map_kind == "constant":
        artifact_path = directory / "balanced_control_map.npz"
        expected_status = "validated_empirical_low_rank_secant_map"
        recorded = report.get("map_artifact")
    else:  # defensive even though argparse constrains this value
        raise ValueError(f"unknown control-map kind {control_map_kind!r}")
    if report.get("schema_version") != 1 or report.get("status") != expected_status:
        raise ValueError(
            f"balanced-control report is not the validated {control_map_kind} map"
        )
    if not isinstance(recorded, dict) or recorded.get("sha256") != sha256_file(
        artifact_path
    ):
        raise ValueError("balanced-control artifact hash does not match its report")
    source = report.get("source")
    if not isinstance(source, dict):
        raise ValueError("balanced-control report lacks source provenance")
    if source.get("normalization_sha256") != sha256_file(normalization_path):
        raise ValueError("balanced-control map uses another normalization artifact")
    if source.get("state_manifest_sha256") != state_manifest_sha256:
        raise ValueError("balanced-control map uses another packed-state manifest")
    if control_map_kind == "phase-local":
        model, descriptor_metadata = PhaseLocalControlStack.load(artifact_path)
        expected_offsets = np.arange(26, 35, dtype=np.int64)
        if not np.array_equal(model.phase_offsets, expected_offsets):
            raise ValueError("phase-local map does not cover phases 26 through 34")
        packed_B = model.packed_control_stack()
        retained_fraction = report["construction"][
            "retained_pooled_variance_fraction"
        ]
        phase_offsets = model.phase_offsets.tolist()
        constant_across_steps = False
    else:
        model, descriptor_metadata = BalancedControlMap.load(artifact_path)
        packed_B = np.repeat(
            model.packed_control_matrix()[None, :, :], CONTROL_STEPS, axis=0
        )
        retained_fraction = report["construction"][
            "retained_observation_variance_fraction"
        ]
        phase_offsets = [26] * CONTROL_STEPS
        constant_across_steps = True
    if model.rank != 36 or model.observation_size != 2160:
        raise ValueError("this experiment requires the validated rank-36 core4 map")
    if model.layout.packed_size != 59_148:
        raise ValueError("balanced-control map has another packed-state size")
    if packed_B.shape != (CONTROL_STEPS, 59_148, 36):
        raise ValueError(f"balanced-control stack has wrong shape {packed_B.shape}")
    return model, np.asarray(packed_B, dtype=np.float64), {
        "control_map_kind": control_map_kind,
        "directory": str(directory),
        "report_file": report_path.name,
        "report_sha256": sha256_file(report_path),
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "descriptor_metadata": descriptor_metadata,
        "retained_observation_variance_fraction": retained_fraction,
        "phase_offsets": phase_offsets,
        "constant_across_control_steps": constant_across_steps,
        "map_status": report["status"],
    }


def _validate_locked_checkpoints(
    neutral_dir: Path, release_indices: np.ndarray
) -> tuple[dict[int, Path], dict[str, Any]]:
    report_path = neutral_dir / "report.json"
    report = load_json(report_path)
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("neutral-ensemble source report is not complete schema 1")
    raw_members = report.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 20:
        raise ValueError("neutral-ensemble report does not contain 20 locked members")
    recorded_indices = np.asarray(
        [entry["release_input_index"] for entry in raw_members], dtype=np.int64
    )
    if not np.array_equal(recorded_indices, LOCKED_NEUTRAL_INDICES):
        raise ValueError("neutral-ensemble report does not match the locked cohort")
    definition = report.get("scientific_definition")
    selection = (
        None if not isinstance(definition, dict) else definition.get("selection")
    )
    if not isinstance(selection, dict):
        raise ValueError("neutral-ensemble report lacks its locked selection contract")
    expected_selection_flags = {
        "future_blind": True,
        "future_targets_used_for_selection": False,
        "cnn_forecasts_used_for_selection": False,
        "intervention_outcomes_used_for_selection": False,
    }
    for key, expected in expected_selection_flags.items():
        if selection.get(key) is not expected:
            raise ValueError(f"neutral selection flag {key} is not {expected}")
    if selection.get("selected_indices") != LOCKED_NEUTRAL_INDICES.tolist():
        raise ValueError("neutral selection indices differ from the locked cohort")
    if selection.get("selected_indices_sha256") != sha256_array(
        LOCKED_NEUTRAL_INDICES
    ):
        raise ValueError("neutral selection index hash differs from the locked cohort")

    result: dict[int, Path] = {}
    checkpoint_records: list[dict[str, Any]] = []
    by_index = {int(entry["release_input_index"]): entry for entry in raw_members}
    for release_index_value in release_indices:
        release_index = int(release_index_value)
        entry = by_index[release_index]
        checkpoint_record = entry.get("checkpoint")
        if not isinstance(checkpoint_record, dict):
            raise ValueError(f"missing checkpoint provenance for {release_index}")
        filename = checkpoint_record.get("checkpoint_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("unsafe checkpoint filename in neutral report")
        checkpoint = neutral_dir / "checkpoints" / filename
        digest = sha256_file(checkpoint)
        if digest != checkpoint_record.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint hash mismatch for {release_index}")
        if checkpoint_record.get("input_index") != release_index - CONTROL_STEPS:
            raise ValueError(f"checkpoint input index mismatch for {release_index}")
        result[release_index] = checkpoint
        checkpoint_records.append(
            {
                "release_input_index": release_index,
                "file": filename,
                "sha256": digest,
                "size_bytes": checkpoint.stat().st_size,
                "pre_input_nt": checkpoint_record.get("pre_input_nt"),
            }
        )
    return result, {
        "source_report": str(report_path),
        "source_report_sha256": sha256_file(report_path),
        "locked_release_indices_sha256": sha256_array(LOCKED_NEUTRAL_INDICES),
        "selection_was_future_blind": True,
        "checkpoints": checkpoint_records,
    }


def _run_kernel_capture(run_dir: Path) -> float:
    started = time.monotonic()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(run_dir / "zeqfc1")],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode:
        log_path = run_dir / "run.log"
        raise RuntimeError(f"packed-state capture failed; inspect {log_path}")
    return elapsed


def _cached_member_record(
    cache_manifest: dict[str, Any] | None, release_index: int
) -> dict[str, Any] | None:
    if cache_manifest is None:
        return None
    raw_members = cache_manifest.get("members")
    if not isinstance(raw_members, list):
        raise ValueError("input-cache manifest members must be a list")
    matches = [
        member
        for member in raw_members
        if isinstance(member, dict)
        and member.get("release_input_index") == release_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"input-cache manifest has {len(matches)} records for {release_index}"
        )
    return matches[0]


def _validate_input_cache(
    directory: Path,
    *,
    state_manifest_sha256: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None, {"used": False, "reason": "manifest_not_found"}
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("input-cache manifest has an unsupported schema")
    cohort = manifest.get("locked_cohort")
    contract = manifest.get("state_contract")
    if not isinstance(cohort, dict) or not isinstance(contract, dict):
        raise ValueError("input-cache manifest lacks cohort/state contracts")
    if cohort.get("release_input_indices_sha256") != sha256_array(
        LOCKED_NEUTRAL_INDICES
    ):
        raise ValueError("input-cache manifest belongs to another neutral cohort")
    if contract.get("manifest_sha256") != state_manifest_sha256:
        raise ValueError("input-cache manifest uses another packed-state layout")
    return manifest, {
        "used": True,
        "directory": str(directory),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _runtime_context_provenance(runtime_template: Path) -> dict[str, Any]:
    """Hash every static/runtime input used to initialize each one-step process."""

    required = (
        runtime_template / "fc.data",
        runtime_template / "zeq9fsu.hst",
        runtime_template / "modified_means.namelist",
        runtime_template / "scales_EOF.namelist",
    )
    paths = [*required, *sorted((runtime_template / "Data").glob("**/*"))]
    records = []
    for path in paths:
        if not path.is_file():
            if path in required:
                raise FileNotFoundError(path)
            continue
        records.append(
            {
                "path": path.relative_to(runtime_template).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    identity = {"files": records}
    return {
        "directory": str(runtime_template),
        "files": records,
        "sha256": sha256_json(identity),
    }


def _validate_dynamic_bridge_provenance(
    validation_dir: Path,
    *,
    kernel_executable: Path,
    primal_executable: Path,
    tangent_executable: Path,
    runtime_template: Path,
) -> dict[str, Any]:
    """Validate the prior replay audit and the exact controlled-driver build."""

    replay_path = validation_dir / "replay_report.json"
    replay = load_json(replay_path)
    kernel_digest = sha256_file(kernel_executable)
    if replay.get("schema_version") != 2 or replay.get("status") != "passed":
        raise ValueError("kernel replay audit is not a passed schema-2 report")
    if replay.get("kernel_executable_sha256") != kernel_digest:
        raise ValueError("kernel replay report does not match its executable")
    executed = replay.get("executed_staged_executable_sha256")
    if not isinstance(executed, dict) or executed.get("zc_kernel_replay") != (
        kernel_digest
    ):
        raise ValueError("kernel replay report lacks the executed kernel binding")
    for family in ("cases", "segmented_replay_cases", "workspace_poison_cases"):
        entries = replay.get(family)
        if not isinstance(entries, list) or not entries or not all(
            isinstance(entry, dict) and entry.get("passed") is True
            for entry in entries
        ):
            raise ValueError(f"kernel replay report has a failed/missing {family}")

    build_path = primal_executable.parent / "build_provenance.txt"
    build_values: dict[str, str] = {}
    for line in build_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            build_values[key] = value
    required_build_keys = {
        "driver_sha256",
        "tangent_driver_sha256",
        "canonical_build_manifest_sha256",
        "executable_sha256",
        "tangent_executable_sha256",
    }
    if not required_build_keys.issubset(build_values):
        raise ValueError("controlled one-step build provenance is incomplete")
    controlled_source = primal_executable.parent.parent
    expected_build = {
        "driver_sha256": sha256_file(controlled_source / "zc_one_step_driver.F"),
        "tangent_driver_sha256": sha256_file(
            controlled_source / "zc_one_step_tangent_driver.F"
        ),
        "executable_sha256": sha256_file(primal_executable),
        "tangent_executable_sha256": sha256_file(tangent_executable),
    }
    for key, value in expected_build.items():
        if build_values[key] != value:
            raise ValueError(f"controlled one-step build binding differs for {key}")
    canonical_manifest = (
        REPOSITORY_ROOT
        / "adjoint/tapenade_toolchain/build/coupled_run23_final3/"
        "coupled_compiled/build_input_manifest.sha256"
    )
    if build_values["canonical_build_manifest_sha256"] != sha256_file(
        canonical_manifest
    ):
        raise ValueError("controlled build uses another canonical AD manifest")

    runtime = _runtime_context_provenance(runtime_template)
    return {
        "replay_report": str(replay_path),
        "replay_report_sha256": sha256_file(replay_path),
        "controlled_build_provenance": str(build_path),
        "controlled_build_provenance_sha256": sha256_file(build_path),
        "canonical_build_manifest": str(canonical_manifest),
        "canonical_build_manifest_sha256": sha256_file(canonical_manifest),
        "runtime_context": runtime,
    }


def _materialize_initial_state(
    *,
    release_index: int,
    checkpoint: Path,
    destination: Path,
    runtime_source: Path,
    kernel_executable: Path,
    reusable_pilot_state: Path,
    cache_dir: Path,
    cache_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    start_index = release_index - CONTROL_STEPS
    pre_nt = 3_600 + start_index
    source_kind = "fresh_kernel_capture_from_locked_restart"
    capture_seconds = 0.0
    cached = _cached_member_record(cache_manifest, release_index)
    if cached is not None:
        cached_checkpoint = cached.get("checkpoint")
        if not isinstance(cached_checkpoint, dict) or cached_checkpoint.get(
            "sha256"
        ) != sha256_file(checkpoint):
            raise ValueError(
                "cached packed state is bound to another checkpoint for "
                f"{release_index}"
            )
        packed = cached.get("packed_state")
        if not isinstance(packed, dict):
            raise ValueError("input-cache member lacks packed_state provenance")
        relative = packed.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("input-cache packed-state path is unsafe")
        source = cache_dir / relative
        digest = sha256_file(source)
        if digest != packed.get("sha256"):
            raise ValueError(f"cached packed-state hash mismatch for {release_index}")
        if cached.get("pre_input_nt") != pre_nt:
            raise ValueError(f"cached packed-state clock mismatch for {release_index}")
        shutil.copyfile(source, destination)
        source_kind = "copied_from_validated_locked_input_cache"
    elif release_index == PILOT_RELEASE_INDEX and reusable_pilot_state.is_file():
        shutil.copyfile(reusable_pilot_state, destination)
        source_kind = "copied_from_validated_member04_40step_kernel_replay"
    else:
        with tempfile.TemporaryDirectory(prefix="zc-state-capture-") as raw:
            run_dir = Path(raw) / "capture"
            fresh_generator.prepare_run(
                runtime_source,
                kernel_executable,
                run_dir,
                nstart=3,
                tfind=fresh_generator.model_time(pre_nt),
                tzero=fresh_generator.model_time(pre_nt),
                tend=fresh_generator.model_time(pre_nt + 1),
                ntape=0,
                nrewnd=11,
                nic=0,
                write_start=fresh_generator.model_time(pre_nt + 3),
                write_end=fresh_generator.model_time(pre_nt + 3),
                restart=checkpoint,
            )
            capture_seconds = _run_kernel_capture(run_dir)
            captured = run_dir / "kernel_initial_state.bin"
            if not captured.is_file():
                raise RuntimeError("kernel replay did not publish its initial state")
            shutil.copyfile(captured, destination)
    return {
        "file": destination.name,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "source_kind": source_kind,
        "checkpoint_sha256": sha256_file(checkpoint),
        "pre_input_nt": pre_nt,
        "capture_wall_seconds": capture_seconds,
        "input_cache_manifest_sha256": (
            None if cache_manifest is None else sha256_file(cache_dir / "manifest.json")
        ),
    }


def _cached_response(
    cache_dir: Path,
    *,
    control_map_kind: str,
    member_label: str,
    release_index: int,
    packed_state_sha256: str,
    packed_B_sha256: str,
    temporal_basis_sha256: str,
    primal_executable_sha256: str,
    tangent_executable_sha256: str,
    normalization_sha256: str,
    state_manifest_sha256: str,
    runtime_context_sha256: str,
    replay_report_sha256: str,
    controlled_build_provenance_sha256: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load only a completely bound response cache; never adopt loose arrays."""

    response_dir = (
        cache_dir
        / "responses"
        / f"{control_map_kind}-{packed_B_sha256[:16]}"
    )
    path = response_dir / f"{member_label}_A.npy"
    if not path.is_file():
        return None, {"used": False, "reason": "response_not_found"}
    sidecar_path = path.with_suffix(".manifest.json")
    if not sidecar_path.is_file():
        return None, {
            "used": False,
            "reason": "unbound_response_requires_independent_recomputation",
            "path": str(path),
        }
    record = load_json(sidecar_path)
    response = np.load(path, allow_pickle=False)
    if response.dtype != np.float64 or response.shape != (2160, 108):
        raise ValueError(
            f"cached response has dtype/shape {response.dtype}/{response.shape}"
        )
    if not np.isfinite(response).all():
        raise ValueError("cached response contains nonfinite values")
    digest = sha256_file(path)
    array_digest = sha256_array(response)
    expected = {
        "schema_version": 2,
        "member": member_label,
        "release_input_index": release_index,
        "file": path.name,
        "sha256": digest,
        "array_sha256": array_digest,
        "dtype": "float64",
        "shape": [2160, 108],
        "control_map_kind": control_map_kind,
        "packed_state_sha256": packed_state_sha256,
        "packed_B_sha256": packed_B_sha256,
        "temporal_basis_sha256": temporal_basis_sha256,
        "primal_executable_sha256": primal_executable_sha256,
        "tangent_executable_sha256": tangent_executable_sha256,
        "normalization_sha256": normalization_sha256,
        "state_manifest_sha256": state_manifest_sha256,
        "runtime_context_sha256": runtime_context_sha256,
        "replay_report_sha256": replay_report_sha256,
        "controlled_build_provenance_sha256": controlled_build_provenance_sha256,
        "control_steps": CONTROL_STEPS,
        "release_boundaries": list(RELEASE_BOUNDARIES),
        "phase_coordinates_dropped": True,
        "forward_tangent_invocations": 108,
    }
    missing = sorted(set(expected) - set(record))
    if missing:
        return None, {
            "used": False,
            "reason": "incomplete_binding_requires_independent_recomputation",
            "path": str(path),
            "missing_binding_fields": missing,
        }
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError(
                f"cached response binding {key} differs for {release_index}"
            )
    return np.asarray(response), {
        "used": True,
        "path": str(path),
        "file_sha256": digest,
        "array_sha256": array_digest,
        "manifest_bindings_present": True,
        "complete_binding_fields_present": True,
        "all_required_bindings_validated": True,
        "sidecar_manifest": sidecar_path.name,
        "sidecar_manifest_sha256": sha256_file(sidecar_path),
    }


def _publish_response_cache(
    cache_dir: Path,
    response: np.ndarray,
    *,
    control_map_kind: str,
    member_label: str,
    release_index: int,
    packed_state_sha256: str,
    packed_B_sha256: str,
    temporal_basis_sha256: str,
    primal_executable_sha256: str,
    tangent_executable_sha256: str,
    normalization_sha256: str,
    state_manifest_sha256: str,
    runtime_context_sha256: str,
    replay_report_sha256: str,
    controlled_build_provenance_sha256: str,
) -> dict[str, Any]:
    """Publish a freshly formed response and bind every scientific input."""

    matrix = np.asarray(response)
    if matrix.dtype != np.float64 or matrix.shape != (2160, 108):
        raise ValueError("only the canonical float64 (2160,108) A can be cached")
    response_dir = (
        cache_dir
        / "responses"
        / f"{control_map_kind}-{packed_B_sha256[:16]}"
    )
    response_dir.mkdir(parents=True, exist_ok=True)
    path = response_dir / f"{member_label}_A.npy"
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if not np.array_equal(existing, matrix):
            raise FileExistsError(
                f"a different immutable response cache already exists: {path}"
            )
    else:
        with (
            atomic_output_path(path, overwrite=False) as temporary,
            temporary.open("wb") as stream,
        ):
            np.save(stream, matrix, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    manifest = {
        "schema_version": 2,
        "member": member_label,
        "release_input_index": release_index,
        "file": path.name,
        "sha256": sha256_file(path),
        "array_sha256": sha256_array(matrix),
        "dtype": "float64",
        "shape": [2160, 108],
        "control_map_kind": control_map_kind,
        "packed_state_sha256": packed_state_sha256,
        "packed_B_sha256": packed_B_sha256,
        "temporal_basis_sha256": temporal_basis_sha256,
        "primal_executable_sha256": primal_executable_sha256,
        "tangent_executable_sha256": tangent_executable_sha256,
        "normalization_sha256": normalization_sha256,
        "state_manifest_sha256": state_manifest_sha256,
        "runtime_context_sha256": runtime_context_sha256,
        "replay_report_sha256": replay_report_sha256,
        "controlled_build_provenance_sha256": controlled_build_provenance_sha256,
        "control_steps": CONTROL_STEPS,
        "release_boundaries": list(RELEASE_BOUNDARIES),
        "phase_coordinates_dropped": True,
        "forward_tangent_invocations": 108,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    sidecar = path.with_suffix(".manifest.json")
    if sidecar.exists():
        existing_manifest = load_json(sidecar)
        for key, value in manifest.items():
            if key != "created_utc" and existing_manifest.get(key) != value:
                raise FileExistsError(
                    f"response cache sidecar has another binding: {sidecar}"
                )
    else:
        write_json(sidecar, manifest, overwrite=False)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "array_sha256": sha256_array(matrix),
        "sidecar_manifest": str(sidecar),
        "sidecar_manifest_sha256": sha256_file(sidecar),
    }


def _range_cosine(response: np.ndarray, direction: np.ndarray) -> tuple[float, int]:
    left, singular_values, _ = scipy.linalg.svd(
        response, full_matrices=False, check_finite=True
    )
    tolerance = (
        max(response.shape) * np.finfo(np.float64).eps * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        return 0.0, 0
    return float(np.linalg.norm(left[:, :rank].T @ direction)), rank


def select_release_penalty(
    response: np.ndarray,
    direction: np.ndarray,
    gammas: np.ndarray,
    *,
    minimum_range_cosine: float,
    target_range_cosine_fraction: float,
    maximum_control_norm_inflation: float,
) -> tuple[BetaChoice, list[dict[str, Any]], dict[str, Any]]:
    """Choose beta without consulting a forecast or future ZC outcome."""

    along = direction @ response
    perpendicular = response - np.outer(direction, along)
    perpendicular_scale = float(scipy.linalg.svdvals(perpendicular)[0])
    maximum_cosine, response_rank = _range_cosine(response, direction)
    target_cosine = max(
        minimum_range_cosine,
        target_range_cosine_fraction * maximum_cosine,
    )
    ideal_cosine = 0.95 * maximum_cosine
    choices: list[BetaChoice] = []
    baseline_norm: float | None = None
    for gamma in gammas:
        beta = (
            0.0 if perpendicular_scale == 0.0 else float(gamma / perpendicular_scale**2)
        )
        solution = solve_linearized_agop_alignment(
            response,
            direction,
            projection=1.0,
            perpendicular_penalty=beta,
        )
        if baseline_norm is None:
            baseline_norm = solution.control_norm
        release_norm = float(np.linalg.norm(solution.release_displacement))
        cosine = solution.realized_projection / release_norm
        norm_inflation = solution.control_norm / baseline_norm
        squared_inflation = norm_inflation**2
        criterion_met = (
            cosine >= target_cosine and norm_inflation <= maximum_control_norm_inflation
        )
        choices.append(
            BetaChoice(
                gamma=float(gamma),
                beta=beta,
                target_cosine=target_cosine,
                ideal_cosine=ideal_cosine,
                predicted_cosine=cosine,
                control_norm=solution.control_norm,
                control_norm_inflation=norm_inflation,
                squared_action_inflation=squared_inflation,
                criterion_met=criterion_met,
                selection_status="candidate",
                solution=solution,
            )
        )

    eligible = [choice for choice in choices if choice.criterion_met]
    if eligible:
        selected_raw = eligible[0]
        status = "smallest_gamma_meeting_release_alignment_and_action_limits"
    else:
        action_safe = [
            choice
            for choice in choices
            if choice.control_norm_inflation <= maximum_control_norm_inflation
        ]
        selected_raw = max(
            action_safe,
            key=lambda choice: (choice.predicted_cosine, -choice.gamma),
        )
        status = "fallback_best_release_cosine_within_control_norm_limit"
    selected = replace(selected_raw, selection_status=status)
    grid_report = [
        {
            "gamma": choice.gamma,
            "beta": choice.beta,
            "predicted_release_cosine": choice.predicted_cosine,
            "target_release_cosine": choice.target_cosine,
            "ideal_95_percent_range_cosine": choice.ideal_cosine,
            "unit_projection_control_l2": choice.control_norm,
            "control_norm_inflation_vs_gamma0": choice.control_norm_inflation,
            "squared_action_inflation_vs_gamma0": (choice.squared_action_inflation),
            "criterion_met": choice.criterion_met,
            "kkt_stationarity_norm": choice.solution.kkt_stationarity_norm,
        }
        for choice in choices
    ]
    geometry = {
        "response_rank": response_rank,
        "control_dimension": response.shape[1],
        "observation_dimension": response.shape[0],
        "maximum_attainable_release_cosine": maximum_cosine,
        "minimum_acceptable_range_cosine": target_cosine,
        "ideal_95_percent_range_cosine": ideal_cosine,
        "perpendicular_response_spectral_norm": perpendicular_scale,
    }
    return selected, grid_report, geometry


def _response_orthogonal_control(
    response: np.ndarray, direction: np.ndarray, *, norm: float
) -> np.ndarray:
    constraint = response.T @ direction
    constraint_squared = float(constraint @ constraint)
    if constraint_squared <= np.finfo(np.float64).tiny:
        raise ValueError("release response has no AGOP-direction component")
    projector = (
        np.eye(response.shape[1])
        - np.outer(constraint, constraint) / constraint_squared
    )
    perpendicular = response - np.outer(direction, direction @ response)
    _, _, right = scipy.linalg.svd(
        perpendicular @ projector, full_matrices=False, check_finite=True
    )
    control = projector @ right[0]
    control_norm = float(np.linalg.norm(control))
    if control_norm <= 100.0 * np.finfo(np.float64).eps:
        raise ValueError("response-orthogonal control subspace is empty")
    return control * (norm / control_norm)


def _shuffled_control(
    primary_control: np.ndarray,
    response: np.ndarray,
    direction: np.ndarray,
    *,
    temporal_rank: int,
    spatial_rank: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = primary_control.reshape(temporal_rank, spatial_rank).copy()
    for block in blocks:
        block[:] = block[rng.permutation(spatial_rank)]
    shuffled = blocks.reshape(-1)
    constraint = response.T @ direction
    constraint_squared = float(constraint @ constraint)
    shuffled -= constraint * float(constraint @ shuffled) / constraint_squared
    shuffled_norm = float(np.linalg.norm(shuffled))
    target_norm = float(np.linalg.norm(primary_control))
    if shuffled_norm <= 100.0 * np.finfo(np.float64).eps:
        raise ValueError("shuffled control vanished after response-orthogonalization")
    return shuffled * (target_norm / shuffled_norm)


def _model_forecast(model: torch.nn.Module, standardized: np.ndarray) -> float:
    values = np.asarray(standardized, dtype=np.float32)
    with torch.inference_mode():
        result = model(torch.from_numpy(values[None]))
    return float(result.detach().cpu().reshape(-1)[0])


def _clock_equal(replay: ControlledReplay, baseline: ZeroControlPath) -> bool:
    return all(
        np.array_equal(actual.integers, expected.integers)
        and np.array_equal(actual.passive_time, expected.passive_time)
        for actual, expected in zip(replay.states, baseline.states, strict=True)
    )


def _branch_change_count(replay: ControlledReplay, baseline: ZeroControlPath) -> int:
    return int(
        sum(
            np.count_nonzero(actual != expected)
            for actual, expected in zip(replay.tapes, baseline.tapes, strict=True)
        )
    )


def _sst_cap_count(state: PackedZCState, chain: FrozenCore4ObservationChain) -> int:
    segment = chain.layout.real_segments["TT"]
    values = state.real32[segment.start : segment.stop].reshape(
        segment.shape, order="F"
    )
    interior = values[5:25, 5:32]
    return int(np.count_nonzero(interior >= np.float32(30.0 - 1.0e-5)))


def _maximum_sst_cap_count(
    states: tuple[PackedZCState, ...], chain: FrozenCore4ObservationChain
) -> int:
    return max(_sst_cap_count(state, chain) for state in states)


def _solution_report(solution: AGOPAlignmentSolution) -> dict[str, Any]:
    return {
        "requested_projection": solution.requested_projection,
        "linear_realized_projection": solution.realized_projection,
        "linear_perpendicular_norm": solution.perpendicular_norm,
        "normalized_temporal_action_sqrt": solution.control_norm,
        "half_normalized_temporal_action": 0.5 * solution.control_norm**2,
        "sum_step_reduced_mahalanobis_action": (
            TEMPORAL_ACTION_SCALE * solution.control_norm**2
        ),
        "objective_value": solution.objective_value,
        "lagrange_multiplier": solution.lagrange_multiplier,
        "kkt_stationarity_norm": solution.kkt_stationarity_norm,
        "response_rank": solution.response_rank,
        "response_condition_number": solution.response_condition_number,
    }


def _evaluate_replay(
    *,
    label: str,
    family: str,
    fraction: float | None,
    control: np.ndarray,
    requested_projection: float,
    linear_release: np.ndarray,
    replay: ControlledReplay,
    baseline: ZeroControlPath,
    baseline_release: np.ndarray,
    direction: np.ndarray,
    chain: FrozenCore4ObservationChain,
    model: torch.nn.Module,
    baseline_release_cap_count: int,
    expected_aligned_cosine: float | None,
    direction_control_projection_tolerance: float,
) -> tuple[dict[str, Any], np.ndarray]:
    release = chain.forward(
        replay.states[RELEASE_BOUNDARIES[0]],
        replay.states[RELEASE_BOUNDARIES[1]],
    )
    delta = release[:-2].astype(np.float64) - baseline_release
    norm = float(np.linalg.norm(delta))
    projection = float(direction @ delta)
    raw_cosine = 0.0 if norm == 0.0 else projection / norm
    aligned_cosine = (
        None
        if requested_projection == 0.0 or norm == 0.0
        else math.copysign(1.0, requested_projection) * raw_cosine
    )
    ratio = None if requested_projection == 0.0 else projection / requested_projection
    linear_norm = float(np.linalg.norm(linear_release))
    linearization_error = float(np.linalg.norm(delta - linear_release))
    release_cap_count = _maximum_sst_cap_count(
        replay.states[: RELEASE_BOUNDARIES[1] + 1], chain
    )
    full_path_cap_count = _maximum_sst_cap_count(replay.states, chain)
    clocks_equal = _clock_equal(replay, baseline)
    finite = bool(np.isfinite(delta).all())
    primary = family == "agop"
    safety_checks = {
        "all_values_finite": finite,
        "integer_and_passive_clocks_match_unforced_path": clocks_equal,
        "no_new_30c_sst_cap_points_through_release": (
            release_cap_count <= baseline_release_cap_count
        ),
        "release_standardized_rms_at_most_one": norm / math.sqrt(2160.0) <= 1.0,
    }
    if primary:
        assert ratio is not None and aligned_cosine is not None
        assert expected_aligned_cosine is not None
        safety_checks.update(
            {
                "realized_projection_has_requested_sign_and_50_150_percent_size": (
                    0.5 <= ratio <= 1.5
                ),
                "realized_alignment_meets_release_only_gate": (
                    aligned_cosine >= max(0.5, 0.8 * expected_aligned_cosine)
                ),
            }
        )
    elif family not in {"exact_zero"}:
        safety_checks["nonlinear_agop_projection_remains_near_zero"] = abs(
            projection
        ) <= direction_control_projection_tolerance
    final_nino3 = float(nino3_from_real_state(replay.states[TOTAL_STEPS].real32))
    free_nino3 = np.asarray(
        [
            nino3_from_real_state(state.real32)
            for state in replay.states[RELEASE_BOUNDARIES[1] :]
        ],
        dtype=np.float64,
    )
    return {
        "label": label,
        "family": family,
        "fraction_of_neutral_to_extreme_gap": fraction,
        "requested_release_agop_projection": requested_projection,
        "realized_release_agop_projection": projection,
        "realized_fraction_of_requested_projection": ratio,
        "realized_release_cosine_with_agop": raw_cosine,
        "realized_release_aligned_cosine": aligned_cosine,
        "direction_control_absolute_projection_tolerance": (
            direction_control_projection_tolerance
        ),
        "realized_release_standardized_l2": norm,
        "realized_release_standardized_rms": norm / math.sqrt(2160.0),
        "linear_release_standardized_l2": linear_norm,
        "nonlinear_minus_linear_release_l2": linearization_error,
        "nonlinear_minus_linear_relative_l2": (
            None if linear_norm == 0.0 else linearization_error / linear_norm
        ),
        "normalized_temporal_action_sqrt": float(np.linalg.norm(control)),
        "half_normalized_temporal_action": float(0.5 * control @ control),
        "sum_step_reduced_mahalanobis_action": float(
            TEMPORAL_ACTION_SCALE * (control @ control)
        ),
        "cnn_forecast_from_realized_release_c": _model_forecast(model, release),
        "zc_release_nino3_c": float(
            nino3_from_real_state(replay.states[RELEASE_BOUNDARIES[1]].real32)
        ),
        "zc_ten_month_nino3_c": final_nino3,
        "zc_peak_during_free_forecast_c": float(np.max(free_nino3)),
        "zc_minimum_during_free_forecast_c": float(np.min(free_nino3)),
        "branch_tape_entries_changed_vs_unforced": _branch_change_count(
            replay, baseline
        ),
        "maximum_30c_sst_cap_point_count_through_release": release_cap_count,
        "maximum_30c_sst_cap_point_count_full_path": full_path_cap_count,
        "safety_checks": safety_checks,
        "release_safety_passed": bool(all(safety_checks.values())),
        "runtime_wall_seconds": replay.wall_seconds,
    }, delta


def _case_controls(
    *,
    response: np.ndarray,
    direction: np.ndarray,
    chosen_beta: float,
    projection_gap: float,
    fractions: np.ndarray,
    temporal_rank: int,
    spatial_rank: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    solution_reports: list[dict[str, Any]] = []
    for fraction_value in fractions:
        fraction = float(fraction_value)
        requested = fraction * projection_gap
        solution = solve_linearized_agop_alignment(
            response,
            direction,
            projection=requested,
            perpendicular_penalty=chosen_beta,
        )
        cases.append(
            {
                "label": f"agop_{_fraction_label(fraction)}",
                "family": "agop",
                "fraction": fraction,
                "requested": requested,
                "control": solution.control,
                "linear_release": solution.release_displacement,
                "expected_aligned_cosine": abs(solution.realized_projection)
                / np.linalg.norm(solution.release_displacement),
            }
        )
        solution_reports.append(
            {
                "label": cases[-1]["label"],
                **_solution_report(solution),
            }
        )

    half_index = next(
        index for index, case in enumerate(cases) if math.isclose(case["fraction"], 0.5)
    )
    half = cases[half_index]
    target_norm = float(np.linalg.norm(half["control"]))
    orthogonal = _response_orthogonal_control(response, direction, norm=target_norm)
    shuffled = _shuffled_control(
        half["control"],
        response,
        direction,
        temporal_rank=temporal_rank,
        spatial_rank=spatial_rank,
        seed=seed,
    )
    for family, base in (
        ("response_orthogonal_equal_action", orthogonal),
        ("shuffled_response_orthogonal_equal_action", shuffled),
    ):
        for sign in (-1.0, 1.0):
            control = sign * base
            cases.append(
                {
                    "label": f"{family}_{'plus' if sign > 0 else 'minus'}",
                    "family": family,
                    "fraction": None,
                    "requested": 0.0,
                    "control": control,
                    "linear_release": response @ control,
                    "expected_aligned_cosine": None,
                }
            )
    return cases, solution_reports


def _bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, resamples: int = 10_000
) -> list[float] | None:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError("bootstrap sample must be one-dimensional")
    if sample.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.mean(
        sample[rng.integers(0, sample.size, size=(resamples, sample.size))],
        axis=1,
    )
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _sign_test(changes: np.ndarray) -> dict[str, Any]:
    values = np.asarray(changes, dtype=np.float64)
    nonzero = values[values != 0.0]
    positives = int(np.count_nonzero(nonzero > 0.0))
    count = int(nonzero.size)
    if count == 0:
        two_sided = None
        greater = None
        less = None
    else:
        two_sided = float(
            scipy.stats.binomtest(
                positives, count, p=0.5, alternative="two-sided"
            ).pvalue
        )
        greater = float(
            scipy.stats.binomtest(
                positives, count, p=0.5, alternative="greater"
            ).pvalue
        )
        less = float(
            scipy.stats.binomtest(
                positives, count, p=0.5, alternative="less"
            ).pvalue
        )
    return {
        "nonzero_pair_count": count,
        "positive_pair_count": positives,
        "exact_two_sided_p_value": two_sided,
        "exact_one_sided_greater_p_value": greater,
        "exact_one_sided_less_p_value": less,
    }


def _sample_sd_se(values: np.ndarray) -> tuple[float | None, float | None]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.size < 2:
        return None, None
    standard_deviation = float(np.std(sample, ddof=1))
    return standard_deviation, standard_deviation / math.sqrt(sample.size)


def _aggregate(
    labels: list[str],
    final_nino3: np.ndarray,
    safety_passed: np.ndarray,
    *,
    training_q95: float,
    seed: int,
) -> dict[str, Any]:
    baseline = final_nino3[:, 0]
    result: dict[str, Any] = {}
    for position, label in enumerate(labels):
        values = final_nino3[:, position]
        changes = values - baseline
        safe_mask = np.asarray(safety_passed[:, position], dtype=bool)
        safe_values = values[safe_mask]
        safe_changes = changes[safe_mask]
        paired_sd, paired_se = _sample_sd_se(changes)
        safe_paired_sd, safe_paired_se = _sample_sd_se(safe_changes)
        result[label] = {
            "member_count": int(values.size),
            "count_safety_passed": int(np.count_nonzero(safe_mask)),
            "count_safety_rejected": int(np.count_nonzero(~safe_mask)),
            "mean_ten_month_nino3_c": float(np.mean(values)),
            "median_ten_month_nino3_c": float(np.median(values)),
            "mean_paired_change_c": float(np.mean(changes)),
            "median_paired_change_c": float(np.median(changes)),
            "paired_change_standard_deviation_c": paired_sd,
            "paired_change_standard_error_c": paired_se,
            "minimum_paired_change_c": float(np.min(changes)),
            "maximum_paired_change_c": float(np.max(changes)),
            "count_positive_paired_change": int(np.count_nonzero(changes > 0.0)),
            "count_above_training_q95": int(np.count_nonzero(values > training_q95)),
            "safety_filtered_mean_ten_month_nino3_c": (
                None if safe_values.size == 0 else float(np.mean(safe_values))
            ),
            "safety_filtered_mean_paired_change_c": (
                None if safe_changes.size == 0 else float(np.mean(safe_changes))
            ),
            "safety_filtered_median_paired_change_c": (
                None if safe_changes.size == 0 else float(np.median(safe_changes))
            ),
            "safety_filtered_paired_change_standard_deviation_c": safe_paired_sd,
            "safety_filtered_paired_change_standard_error_c": safe_paired_se,
            "safety_filtered_count_above_training_q95": int(
                np.count_nonzero(safe_values > training_q95)
            ),
            "safety_filtered_count_positive_paired_change": int(
                np.count_nonzero(safe_changes > 0.0)
            ),
            "safety_filtered_paired_mean_bootstrap_95_interval_c": (
                _bootstrap_mean_interval(
                    safe_changes,
                    seed=seed + position,
                )
            ),
            "safety_filtered_exact_sign_test": _sign_test(safe_changes),
        }
    return result


def _dose_monotonicity(
    labels: list[str], final_nino3: np.ndarray, safety_passed: np.ndarray
) -> dict[str, Any]:
    dose_labels = ("agop_p0p25", "agop_p0p5", "agop_p1")
    positions = [labels.index(label) for label in dose_labels]
    eligible = np.all(safety_passed[:, positions], axis=1)
    values = final_nino3[:, positions]
    monotone = np.logical_and(
        values[:, 0] <= values[:, 1], values[:, 1] <= values[:, 2]
    )
    count = int(np.count_nonzero(eligible))
    monotone_count = int(np.count_nonzero(monotone & eligible))
    return {
        "positive_dose_labels": list(dose_labels),
        "eligible_members_all_three_release_safe": count,
        "monotone_non_decreasing_member_count": monotone_count,
        "monotone_fraction_of_eligible": (
            None if count == 0 else monotone_count / count
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.seed != 42:
        raise ValueError("the current locked model, AGOP, and controls require seed 42")
    if args.lead_months != 10:
        raise ValueError("the current authenticated experiment uses a 10-month lead")
    if not 0.0 < args.target_range_cosine_fraction <= 1.0:
        raise ValueError("target range-cosine fraction must lie in (0, 1]")
    if not 0.0 <= args.minimum_range_cosine <= 1.0:
        raise ValueError("minimum range cosine must lie in [0, 1]")
    fractions = _validate_fraction_grid(args.fractions)
    gammas = _validate_gamma_grid(args.penalty_gammas)
    release_indices = _requested_indices(args.cohort, args.members)
    selection_label = args.cohort
    if args.members:
        selection_label = "members-" + "-".join(str(value) for value in release_indices)
    if args.response_only:
        selection_label += "-response-only"
    default_output = Path(
        "outputs/zc_agop_adjoint_nudging/"
        f"core4-cnn-lead-10m-seed-000042-{selection_label}-"
        f"{args.control_map}-rank36"
    )
    output_dir = _prepare_output(args.output_dir or default_output, args.overwrite)
    started_utc = datetime.now(UTC).isoformat()
    started = time.monotonic()

    data_dir = _resolve(args.data_dir)
    artifacts_dir = _resolve(args.artifacts_dir)
    agop_dir = _resolve(args.agop_benchmark_dir)
    default_balanced_dir = (
        DEFAULT_PHASE_LOCAL_MAP_DIR
        if args.control_map == "phase-local"
        else DEFAULT_CONSTANT_MAP_DIR
    )
    balanced_dir = _resolve(args.balanced_map_dir or default_balanced_dir)
    neutral_dir = _resolve(args.neutral_ensemble_dir)
    validation_dir = _resolve(args.adjoint_validation_dir)
    input_cache_dir = _resolve(args.input_cache_dir)
    primal_executable = _resolve(args.primal_executable)
    tangent_executable = _resolve(args.tangent_executable)
    runtime_source = validation_dir / "verified_inputs" / "source"
    kernel_executable = validation_dir / "verified_inputs" / "zc_kernel_replay"
    runtime_template = validation_dir / "neutral_member_04_040step" / "kernel"
    reusable_pilot_state = runtime_template / "kernel_initial_state.bin"
    for required in (
        primal_executable,
        tangent_executable,
        kernel_executable,
        reusable_pilot_state,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    for required in (runtime_source, runtime_template):
        if not required.is_dir():
            raise FileNotFoundError(required)
    dynamic_bridge_provenance = _validate_dynamic_bridge_provenance(
        validation_dir,
        kernel_executable=kernel_executable,
        primal_executable=primal_executable,
        tangent_executable=tangent_executable,
        runtime_template=runtime_template,
    )

    data = ZCData(
        data_dir,
        input_profile="core4",
        verify_checksums=not args.skip_data_checksums,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("this experiment requires the fresh zc-v3 data set")
    if data.steps_per_month != 3:
        raise ValueError("the controlled-window indexing requires three steps/month")
    spec = ExperimentSpec(
        architecture="cnn",
        lead_months=10,
        train_years=10_000.0,
        seed=42,
        input_profile="core4",
    )
    experiment = load_experiment(data, artifacts_dir, spec, device="cpu")
    fixed = data.fixed_supervised_split(10)
    canonical_standardization = np.arange(
        fixed.train_block[0], fixed.train_block[1], dtype=np.int64
    )
    if not np.array_equal(experiment.standardization_inputs, canonical_standardization):
        raise ValueError("model does not use the canonical 10,000-year normalizer")
    observation_chain = FrozenCore4ObservationChain(experiment.standardizer)
    normalization_path = experiment.artifact_dir / "normalization.npz"
    input_cache_manifest, input_cache_provenance = _validate_input_cache(
        input_cache_dir,
        state_manifest_sha256=observation_chain.layout.manifest_sha256,
    )

    factor, factor_provenance = fresh_xai_benchmark.load_validated_full_agop_factor(
        agop_dir,
        data=data,
        experiment=experiment,
        fixed_training_inputs=fixed.train_inputs,
    )
    event = next(
        entry
        for entry in data.metadata["event_restart_checkpoints"]
        if entry["label"] == "extreme_el_nino"
    )
    event_index = int(event["input_index"])
    event_standardized = data.load_inputs(
        np.asarray([event_index]), standardizer=experiment.standardizer
    )[0].astype(np.float64)
    explanation = AgopExplainer(factor).explain(event_standardized[None])[0]
    full_direction = unit_fixed_phase_direction(explanation, phase_features=2)
    direction = np.asarray(full_direction[:-2], dtype=np.float64)
    event_coordinate = float(event_standardized[:-2] @ direction)

    balanced_map, packed_B, balanced_provenance = _load_balanced_map(
        balanced_dir,
        control_map_kind=args.control_map,
        normalization_path=normalization_path,
        state_manifest_sha256=observation_chain.layout.manifest_sha256,
    )
    temporal_basis = three_mode_half_cosine_iau_basis(CONTROL_STEPS)
    operator = DistributedControlOperator(packed_B, temporal_basis)
    packed_B_sha256 = sha256_array(packed_B)
    temporal_basis_sha256 = sha256_array(temporal_basis)
    primal_executable_sha256 = sha256_file(primal_executable)
    tangent_executable_sha256 = sha256_file(tangent_executable)
    normalization_sha256 = sha256_file(normalization_path)
    runtime_context_sha256 = dynamic_bridge_provenance["runtime_context"]["sha256"]
    replay_report_sha256 = dynamic_bridge_provenance["replay_report_sha256"]
    controlled_build_provenance_sha256 = dynamic_bridge_provenance[
        "controlled_build_provenance_sha256"
    ]
    if not np.all(release_indices % data.steps_per_year == 35):
        raise ValueError("locked releases no longer share the certified annual phase")
    if not np.all((release_indices - CONTROL_STEPS) % data.steps_per_year == 26):
        raise ValueError("locked control starts no longer match map phase 26")

    checkpoints, cohort_provenance = _validate_locked_checkpoints(
        neutral_dir, release_indices
    )
    initial_dir = output_dir / "initial_states"
    initial_dir.mkdir()
    initial_reports: list[dict[str, Any]] = []
    member_reports: list[dict[str, Any]] = []
    responses: list[np.ndarray] = []
    all_controls: list[np.ndarray] = []
    all_release_deltas: list[np.ndarray] = []
    all_final_nino3: list[np.ndarray] = []
    all_forecasts: list[np.ndarray] = []
    all_release_projections: list[np.ndarray] = []
    all_release_cosines: list[np.ndarray] = []
    all_release_rms: list[np.ndarray] = []
    all_safety: list[np.ndarray] = []
    case_labels: list[str] | None = None

    training_targets = np.asarray(
        data.target[fixed.train_block[0] : fixed.train_block[1]], dtype=np.float64
    )
    training_q95 = float(np.quantile(training_targets, 0.95))

    for member_position, release_index_value in enumerate(release_indices):
        release_index = int(release_index_value)
        label = _member_label(release_index)
        print(
            f"[{member_position + 1}/{release_indices.size}] {label}: "
            "materializing and replaying the authentic zero path",
            flush=True,
        )
        initial_path = initial_dir / f"{label}.bin"
        initial_report = _materialize_initial_state(
            release_index=release_index,
            checkpoint=checkpoints[release_index],
            destination=initial_path,
            runtime_source=runtime_source,
            kernel_executable=kernel_executable,
            reusable_pilot_state=reusable_pilot_state,
            cache_dir=input_cache_dir,
            cache_manifest=input_cache_manifest,
        )
        initial_reports.append(initial_report)
        initial_state = observation_chain.read_state(initial_path)
        expected_nt = 3_600 + release_index - CONTROL_STEPS
        nt_segment = observation_chain.layout.integer_segments["NT"]
        initial_nt = int(initial_state.integers[nt_segment.start])
        if initial_nt != expected_nt:
            raise RuntimeError(
                f"{label} initial NT is {initial_nt}, expected {expected_nt}"
            )
        baseline = build_zero_control_path(
            initial_state,
            runtime_template,
            primal_executable,
            TOTAL_STEPS,
        )
        baseline_release_full = observation_chain.forward(
            baseline.states[RELEASE_BOUNDARIES[0]],
            baseline.states[RELEASE_BOUNDARIES[1]],
        )
        expected_release = data.load_inputs(
            np.asarray([release_index]), standardizer=experiment.standardizer
        )[0]
        if not np.array_equal(baseline_release_full, expected_release):
            error = float(
                np.max(
                    np.abs(
                        baseline_release_full.astype(np.float64)
                        - expected_release.astype(np.float64)
                    )
                )
            )
            raise RuntimeError(
                f"{label} release observation differs from zc-v3; max={error:g}"
            )
        baseline_release = baseline_release_full[:-2].astype(np.float64)
        baseline_final = float(
            nino3_from_real_state(baseline.states[TOTAL_STEPS].real32)
        )
        expected_final = float(data.target[release_index + 30])
        if np.float32(baseline_final).view(np.uint32) != np.float32(
            expected_final
        ).view(np.uint32):
            raise RuntimeError(
                f"{label} 10-month target differs from zc-v3: "
                f"{baseline_final} versus {expected_final}"
            )
        baseline_release_cap_count = _maximum_sst_cap_count(
            baseline.states[: RELEASE_BOUNDARIES[1] + 1], observation_chain
        )
        baseline_full_path_cap_count = _maximum_sst_cap_count(
            baseline.states, observation_chain
        )

        response: np.ndarray | None = None
        response_cache_report: dict[str, Any] = {
            "used": False,
            "reason": "explicit_recompute_requested",
        }
        response_started = time.monotonic()
        if not args.recompute_responses:
            response, response_cache_report = _cached_response(
                input_cache_dir,
                control_map_kind=args.control_map,
                member_label=label,
                release_index=release_index,
                packed_state_sha256=initial_report["sha256"],
                packed_B_sha256=packed_B_sha256,
                temporal_basis_sha256=temporal_basis_sha256,
                primal_executable_sha256=primal_executable_sha256,
                tangent_executable_sha256=tangent_executable_sha256,
                normalization_sha256=normalization_sha256,
                state_manifest_sha256=observation_chain.layout.manifest_sha256,
                runtime_context_sha256=runtime_context_sha256,
                replay_report_sha256=replay_report_sha256,
                controlled_build_provenance_sha256=(
                    controlled_build_provenance_sha256
                ),
            )
        response_was_formed = response is None
        if response is None:
            print(
                f"[{member_position + 1}/{release_indices.size}] {label}: "
                f"forming the 2160 x {operator.control_size} tangent response",
                flush=True,
            )
            response = form_authentic_release_response(
                baseline,
                packed_B,
                temporal_basis,
                tangent_executable,
                runtime_template,
                observation_chain=observation_chain,
                control_steps=CONTROL_STEPS,
                release=RELEASE_BOUNDARIES,
                drop_phase=True,
            )
        else:
            print(
                f"[{member_position + 1}/{release_indices.size}] {label}: "
                "using the validated cached tangent response",
                flush=True,
            )
        response_seconds = time.monotonic() - response_started
        responses.append(response)
        if response_was_formed:
            response_publication = _publish_response_cache(
                input_cache_dir,
                response,
                control_map_kind=args.control_map,
                member_label=label,
                release_index=release_index,
                packed_state_sha256=initial_report["sha256"],
                packed_B_sha256=packed_B_sha256,
                temporal_basis_sha256=temporal_basis_sha256,
                primal_executable_sha256=primal_executable_sha256,
                tangent_executable_sha256=tangent_executable_sha256,
                normalization_sha256=normalization_sha256,
                state_manifest_sha256=observation_chain.layout.manifest_sha256,
                runtime_context_sha256=runtime_context_sha256,
                replay_report_sha256=replay_report_sha256,
                controlled_build_provenance_sha256=(
                    controlled_build_provenance_sha256
                ),
            )
        else:
            response_publication = {
                "published": False,
                "reason": "validated_immutable_cache_hit",
            }
        if args.response_only:
            member_reports.append(
                {
                    "member": label,
                    "release_input_index": release_index,
                    "preconditioning_start_input_index": (
                        release_index - CONTROL_STEPS
                    ),
                    "target_input_index": release_index + 30,
                    "baseline_ten_month_nino3_c": baseline_final,
                    "baseline_path_wall_seconds": baseline.wall_seconds,
                    "release_observation_bitwise_equal_to_zc_v3": True,
                    "ten_month_target_bitwise_equal_to_zc_v3": True,
                    "response_formation_wall_seconds": response_seconds,
                    "response_sha256": sha256_array(response),
                    "response_cache_input": response_cache_report,
                    "response_cache_publication": response_publication,
                }
            )
            continue
        selected, penalty_grid, geometry = select_release_penalty(
            response,
            direction,
            gammas,
            minimum_range_cosine=args.minimum_range_cosine,
            target_range_cosine_fraction=args.target_range_cosine_fraction,
            maximum_control_norm_inflation=args.maximum_control_norm_inflation,
        )
        if geometry["maximum_attainable_release_cosine"] < (args.minimum_range_cosine):
            selection_warning = (
                "AGOP direction is poorly represented by this member's dynamic "
                "release response"
            )
        else:
            selection_warning = None
        neutral_coordinate = float(direction @ baseline_release)
        projection_gap = event_coordinate - neutral_coordinate
        if projection_gap <= 0.0:
            raise RuntimeError(f"{label} has a nonpositive neutral-to-event gap")
        direction_control_projection_tolerance = max(
            0.5, 0.1 * 0.5 * projection_gap
        )
        cases, solution_reports = _case_controls(
            response=response,
            direction=direction,
            chosen_beta=selected.beta,
            projection_gap=projection_gap,
            fractions=fractions,
            temporal_rank=operator.temporal_rank,
            spatial_rank=operator.spatial_rank,
            seed=args.seed,
        )
        labels = ["zero", *(case["label"] for case in cases)]
        if case_labels is None:
            case_labels = labels
        elif labels != case_labels:
            raise RuntimeError("case ordering changed between cohort members")

        zero_control = np.zeros(operator.control_size, dtype=np.float64)
        zero_replay = replay_control(
            initial_state,
            operator,
            zero_control,
            runtime_template,
            primal_executable,
            total_steps=TOTAL_STEPS,
        )
        verify_zero_control_replay(baseline, zero_replay)
        zero_report, zero_delta = _evaluate_replay(
            label="zero",
            family="exact_zero",
            fraction=0.0,
            control=zero_control,
            requested_projection=0.0,
            linear_release=np.zeros(2160, dtype=np.float64),
            replay=zero_replay,
            baseline=baseline,
            baseline_release=baseline_release,
            direction=direction,
            chain=observation_chain,
            model=experiment.model,
            baseline_release_cap_count=baseline_release_cap_count,
            expected_aligned_cosine=None,
            direction_control_projection_tolerance=(
                direction_control_projection_tolerance
            ),
        )
        if np.any(zero_delta) or zero_report["zc_ten_month_nino3_c"] != baseline_final:
            raise RuntimeError(f"{label} explicit zero replay is not exact")

        controls = [zero_control]
        deltas = [zero_delta]
        reports = [zero_report]
        print(
            f"[{member_position + 1}/{release_indices.size}] {label}: "
            f"replaying {len(cases)} frozen nonlinear controls",
            flush=True,
        )
        for case in cases:
            replay = replay_control(
                initial_state,
                operator,
                case["control"],
                runtime_template,
                primal_executable,
                total_steps=TOTAL_STEPS,
            )
            case_report, delta = _evaluate_replay(
                label=case["label"],
                family=case["family"],
                fraction=case["fraction"],
                control=case["control"],
                requested_projection=case["requested"],
                linear_release=case["linear_release"],
                replay=replay,
                baseline=baseline,
                baseline_release=baseline_release,
                direction=direction,
                chain=observation_chain,
                model=experiment.model,
                baseline_release_cap_count=baseline_release_cap_count,
                expected_aligned_cosine=case["expected_aligned_cosine"],
                direction_control_projection_tolerance=(
                    direction_control_projection_tolerance
                ),
            )
            controls.append(case["control"])
            deltas.append(delta)
            reports.append(case_report)

        member_reports.append(
            {
                "member": label,
                "release_input_index": release_index,
                "preconditioning_start_input_index": (release_index - CONTROL_STEPS),
                "target_input_index": release_index + 30,
                "baseline_release_nino3_c": float(
                    nino3_from_real_state(baseline.states[RELEASE_BOUNDARIES[1]].real32)
                ),
                "baseline_ten_month_nino3_c": baseline_final,
                "baseline_cnn_forecast_c": _model_forecast(
                    experiment.model, baseline_release_full
                ),
                "baseline_path_wall_seconds": baseline.wall_seconds,
                "baseline_maximum_30c_sst_cap_point_count_through_release": (
                    baseline_release_cap_count
                ),
                "baseline_maximum_30c_sst_cap_point_count_full_path": (
                    baseline_full_path_cap_count
                ),
                "event_agop_coordinate": event_coordinate,
                "neutral_agop_coordinate": neutral_coordinate,
                "neutral_to_extreme_agop_projection_gap": projection_gap,
                "response_formation_wall_seconds": response_seconds,
                "response_sha256": sha256_array(response),
                "response_cache": response_cache_report,
                "response_cache_publication": response_publication,
                "response_geometry": geometry,
                "penalty_grid": penalty_grid,
                "selected_penalty": {
                    "gamma": selected.gamma,
                    "beta": selected.beta,
                    "selection_status": selected.selection_status,
                    "criterion_met": selected.criterion_met,
                    "predicted_release_cosine": selected.predicted_cosine,
                    "target_release_cosine": selected.target_cosine,
                    "ideal_95_percent_range_cosine": selected.ideal_cosine,
                    "control_norm_inflation_vs_gamma0": (
                        selected.control_norm_inflation
                    ),
                    "squared_action_inflation_vs_gamma0": (
                        selected.squared_action_inflation
                    ),
                },
                "selection_warning": selection_warning,
                "linear_solutions": solution_reports,
                "zero_control_bitwise_equal_at_all_boundaries_and_tapes": True,
                "cases": reports,
            }
        )
        controls_array = np.stack(controls)
        deltas_array = np.stack(deltas)
        all_controls.append(controls_array)
        all_release_deltas.append(deltas_array)
        all_final_nino3.append(
            np.asarray(
                [report["zc_ten_month_nino3_c"] for report in reports],
                dtype=np.float64,
            )
        )
        all_forecasts.append(
            np.asarray(
                [report["cnn_forecast_from_realized_release_c"] for report in reports],
                dtype=np.float64,
            )
        )
        all_release_projections.append(
            np.asarray(
                [report["realized_release_agop_projection"] for report in reports],
                dtype=np.float64,
            )
        )
        all_release_cosines.append(
            np.asarray(
                [report["realized_release_cosine_with_agop"] for report in reports],
                dtype=np.float64,
            )
        )
        all_release_rms.append(
            np.asarray(
                [report["realized_release_standardized_rms"] for report in reports],
                dtype=np.float64,
            )
        )
        all_safety.append(
            np.asarray(
                [report["release_safety_passed"] for report in reports], dtype=bool
            )
        )

    if args.response_only:
        responses_array = np.stack(responses)
        arrays_path = output_dir / "release_responses.npz"
        write_npz(
            arrays_path,
            overwrite=False,
            release_indices=release_indices,
            release_response=responses_array,
            temporal_basis=temporal_basis,
        )
        response_identity = {
            "script_version": SCRIPT_VERSION,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "mode": "response_only",
            "release_indices_sha256": sha256_array(release_indices),
            "control_map_kind": args.control_map,
            "packed_B_sha256": packed_B_sha256,
            "temporal_basis_sha256": temporal_basis_sha256,
            "tangent_executable_sha256": tangent_executable_sha256,
            "normalization_sha256": normalization_sha256,
            "state_manifest_sha256": observation_chain.layout.manifest_sha256,
            "runtime_context_sha256": runtime_context_sha256,
            "replay_report_sha256": replay_report_sha256,
            "controlled_build_provenance_sha256": (
                controlled_build_provenance_sha256
            ),
        }
        response_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "status": "complete_response_only",
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.monotonic() - started,
            "run_identity": response_identity,
            "run_identity_sha256": sha256_json(response_identity),
            "members": member_reports,
            "initial_states": initial_reports,
            "cohort": cohort_provenance,
            "input_cache": input_cache_provenance,
            "balanced_control_map": balanced_provenance,
            "dynamic_bridge": dynamic_bridge_provenance,
            "agop": factor_provenance,
            "output_files": {
                "arrays": arrays_path.name,
                "arrays_sha256": sha256_file(arrays_path),
                "arrays_size_bytes": arrays_path.stat().st_size,
            },
            "neutral_or_control_future_outcomes_used_for_response": False,
            "agop_anchor_selection": (
                "fixed upstream maximum held-out 10-month El Nino target"
            ),
        }
        response_report_path = output_dir / "report.json"
        write_json(response_report_path, response_report, overwrite=False)
        print(f"Complete response cache: {response_report_path}", flush=True)
        return 0

    assert case_labels is not None
    responses_array = np.stack(responses)
    controls_array = np.stack(all_controls)
    release_deltas_array = np.stack(all_release_deltas)
    final_nino3_array = np.stack(all_final_nino3)
    forecasts_array = np.stack(all_forecasts)
    release_projections_array = np.stack(all_release_projections)
    release_cosines_array = np.stack(all_release_cosines)
    release_rms_array = np.stack(all_release_rms)
    safety_array = np.stack(all_safety)
    aggregate = _aggregate(
        case_labels,
        final_nino3_array,
        safety_array,
        training_q95=training_q95,
        seed=args.seed + 40_000,
    )
    dose_monotonicity = _dose_monotonicity(
        case_labels, final_nino3_array, safety_array
    )
    positive_one_label = "agop_p1"
    negative_half_label = "agop_m0p5"
    positive_summary = aggregate[positive_one_label]
    negative_summary = aggregate[negative_half_label]
    safe_full_gap_count = positive_summary["count_safety_passed"]
    safe_full_gap_crossings = positive_summary[
        "safety_filtered_count_above_training_q95"
    ]
    safe_full_gap_mean = positive_summary[
        "safety_filtered_mean_paired_change_c"
    ]
    if release_indices.size == 1:
        interpretation = "descriptive_pilot_no_population_inference"
    elif safe_full_gap_crossings > 0:
        interpretation = "supports_dynamically_reachable_agop_aligned_intervention"
    elif safe_full_gap_count > 0 and safe_full_gap_mean > 0.0:
        interpretation = "supports_a_warming_direction_but_not_extreme_sufficiency"
    elif safe_full_gap_count == 0:
        interpretation = "inconclusive_no_full_gap_case_passed_release_safety"
    else:
        interpretation = "does_not_support_the_tested_sufficiency_hypothesis"

    arrays_path = output_dir / "results.npz"
    write_npz(
        arrays_path,
        overwrite=False,
        release_indices=release_indices,
        case_labels=np.asarray(case_labels),
        fractions=fractions,
        penalty_gammas=gammas,
        agop_direction=direction,
        event_standardized_input=event_standardized,
        temporal_basis=temporal_basis,
        release_response=responses_array,
        control_coefficients=controls_array,
        nonlinear_release_displacements=release_deltas_array,
        final_nino3_c=final_nino3_array,
        cnn_release_forecasts_c=forecasts_array,
        release_agop_projections=release_projections_array,
        release_agop_cosines=release_cosines_array,
        release_standardized_rms=release_rms_array,
        release_safety_passed=safety_array,
    )

    run_identity = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "cohort": args.cohort,
        "release_indices_sha256": sha256_array(release_indices),
        "fractions": fractions.tolist(),
        "penalty_gammas": gammas.tolist(),
        "minimum_range_cosine": args.minimum_range_cosine,
        "target_range_cosine_fraction": args.target_range_cosine_fraction,
        "maximum_control_norm_inflation": args.maximum_control_norm_inflation,
        "direction_control_seed": args.seed,
        "control_steps": CONTROL_STEPS,
        "release_boundaries": list(RELEASE_BOUNDARIES),
        "total_steps": TOTAL_STEPS,
        "model_checkpoint_sha256": experiment.checkpoint_sha256,
        "agop_direction_sha256": sha256_array(direction),
        "balanced_map_sha256": balanced_provenance["artifact_sha256"],
        "control_map_kind": args.control_map,
        "packed_B_sha256": packed_B_sha256,
        "primal_executable_sha256": sha256_file(primal_executable),
        "tangent_executable_sha256": sha256_file(tangent_executable),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "run_identity": run_identity,
        "run_identity_sha256": sha256_json(run_identity),
        "scientific_question": (
            "Is the fixed extreme-event AGOP direction, when reached through a "
            "minimum-action dynamically propagated control from authentic neutral "
            "ZC states, sufficient to cause a strong El Nino ten months later?"
        ),
        "future_blind_controller": {
            "neutral_state_selection_is_future_blind": True,
            "controller_selection_uses_only_release_response": True,
            "neutral_or_control_future_zc_outcomes_used_for_selection": False,
            "cnn_forecasts_used_for_controller_selection": False,
            "agop_anchor_was_selected_upstream_using_future_target": True,
            "agop_anchor_qualification": (
                "The fixed explanation anchor is the maximum held-out 10-month "
                "El Nino target by design; this outcome-informed anchor is not "
                "reselected using any neutral/control replay."
            ),
            "dose_grid_was_fixed_before_nonlinear_replay": True,
            "penalty_rule": (
                "smallest gamma on the fixed grid attaining max(0.5, 80% of "
                "the maximum release-range cosine) with at most threefold "
                "control-norm inflation relative to gamma zero; 95% of the "
                "range cosine is retained as an ideal diagnostic"
            ),
            "direction_control_nonlinear_leakage_gate": (
                "absolute realized AGOP projection no larger than max(0.5, "
                "10% of the +0.5-dose requested projection)"
            ),
        },
        "interpretation": {
            "classification": interpretation,
            "positive_full_gap": positive_summary,
            "negative_half_gap": negative_summary,
            "positive_dose_monotonicity": dose_monotonicity,
            "training_nino3_q95_c": training_q95,
            "classification_uses_only_release_safe_primary_cases": True,
            "population_inference_performed": False,
            "resampling_qualification": (
                "Bootstrap intervals and exact sign tests are descriptive across "
                "the deterministic 20-of-21 eligible neutral-state near-census; "
                "they require an additional exchangeability or superpopulation "
                "assumption for population-level inference."
            ),
            "future_cap_policy": (
                "Post-release 30 C cap contacts are reported as outcome "
                "saturation diagnostics but are not used for post-treatment "
                "case selection."
            ),
            "claim_boundary": (
                "A positive outcome supports a dynamically reachable intervention "
                "aligned with the AGOP direction for the tested phase-local rank-36 "
                "control family and neutral states. Because the realized release "
                "change is not parallel to the AGOP vector, it does not by itself "
                "establish sufficiency of the pure AGOP vector, necessity, "
                "uniqueness, or observational realism."
            ),
        },
        "data": {
            "directory": str(data_dir),
            "metadata_sha256": data.metadata_sha256,
            "schema_version": data.metadata["schema_version"],
            "input_profile": data.input_profile,
            "input_shape": list(data.input_shape),
            "checksums_verified": not args.skip_data_checksums,
        },
        "model": {
            "artifact_directory": str(experiment.artifact_dir),
            "checkpoint_sha256": experiment.checkpoint_sha256,
            "normalization_sha256": sha256_file(normalization_path),
            "normalization_mean_sha256": sha256_array(experiment.standardizer.mean),
            "normalization_scale_sha256": sha256_array(experiment.standardizer.scale),
            "normalization_count": experiment.standardizer.count,
        },
        "agop": {
            **factor_provenance,
            "event_input_index": event_index,
            "event_target_input_index": event_index + 30,
            "event_target_nino3_c": float(data.target[event_index + 30]),
            "spatial_direction_sha256": sha256_array(direction),
            "spatial_direction_l2": float(np.linalg.norm(direction)),
            "event_agop_coordinate": event_coordinate,
            "phase_coordinates_held_fixed": True,
        },
        "balanced_control_map": {
            **balanced_provenance,
            "packed_B_sha256": packed_B_sha256,
            "G_sha256": sha256_array(balanced_map.G),
            "spatial_rank": balanced_map.rank,
            "constant_across_control_steps": balanced_provenance[
                "constant_across_control_steps"
            ],
            "control_units": (
                "dimensionless covariance-whitened coordinates; coefficient "
                "norm squared is normalized temporal action"
            ),
            "temporal_basis_column_squared_norm": TEMPORAL_ACTION_SCALE,
            "actual_reduced_action_relation": (
                "sum_k ||u_k||^2 = temporal_basis_column_squared_norm * "
                "||coefficient_vector||^2"
            ),
        },
        "dynamic_bridge": {
            **dynamic_bridge_provenance,
            "primal_executable": str(primal_executable),
            "primal_executable_sha256": sha256_file(primal_executable),
            "tangent_executable": str(tangent_executable),
            "tangent_executable_sha256": sha256_file(tangent_executable),
            "kernel_capture_executable": str(kernel_executable),
            "kernel_capture_executable_sha256": sha256_file(kernel_executable),
            "runtime_template": str(runtime_template),
            "state_manifest": str(observation_chain.layout.manifest_path),
            "state_manifest_sha256": observation_chain.layout.manifest_sha256,
            "release_observation": (
                "H1/U1/V1 from boundary s9, SST anomaly from s10, passive annual "
                "phase from s10 and held fixed"
            ),
            "release_response_construction": (
                "108 independent forward tangent propagations, one per control "
                "coordinate"
            ),
            "adjoint_usage": (
                "The validated reverse executable was not invoked because 108 "
                "control coordinates are fewer than 2160 observation coordinates; "
                "the controller uses the formed response and its algebraic A.T."
            ),
        },
        "cohort": cohort_provenance,
        "input_cache": input_cache_provenance,
        "initial_states": initial_reports,
        "members": member_reports,
        "aggregate_by_case": aggregate,
        "output_files": {
            "arrays": arrays_path.name,
            "arrays_sha256": sha256_file(arrays_path),
            "arrays_size_bytes": arrays_path.stat().st_size,
        },
        "limitations": [
            (
                "The balanced controls are empirical rank-36 secant regressions, "
                "not an exact inverse of the nonlinear ZC balance manifold."
            ),
            *(
                [
                    (
                        "The constant-map sensitivity holds the phase-26 spatial "
                        "map fixed through nine steps; the primary phase-local "
                        "analysis instead uses coherently aligned phases 26--34."
                    )
                ]
                if args.control_map == "constant"
                else []
            ),
            (
                "The one-step tangent and inherited adjoint products retain the "
                "provisional_major_tape_conditioned real32 qualification documented "
                "by the run23-final3 validation."
            ),
            (
                "Finite controls can cross branches and depart from the tangent "
                "prediction; every nonlinear release error and branch-tape change "
                "is therefore reported rather than suppressed."
            ),
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report, overwrite=False)
    print(f"Complete: {report_path}")
    safe_mean_text = (
        "n/a" if safe_full_gap_mean is None else f"{safe_full_gap_mean:+.3f}"
    )
    print(
        f"Interpretation: {interpretation}; release-safe full-gap mean paired "
        f"change {safe_mean_text} C; {safe_full_gap_crossings}/"
        f"{safe_full_gap_count} release-safe cases above training q95 "
        f"({release_indices.size} total members).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
