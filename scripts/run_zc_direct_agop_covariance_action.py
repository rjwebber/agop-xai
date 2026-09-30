#!/usr/bin/env python3
"""Run direct native-covariance AGOP action experiments in the ZC model.

Ten held-out release times are sampled uniformly, with a fixed seed, from all
test inputs at the selected extreme event's annual phase that leave room for
the nine controlled transitions and the ten-month forecast.  No state-value,
neural-network forecast, or future-outcome filter enters this selection.

For each member, nine physical increments in the 28,591 independent native
coordinates minimize empirical covariance action subject to the nonlinear
release constraint ``e.T x_release >= e.T x_extreme``.  The primary arm uses
one dense covariance pooled over every training-block time point after
removing all 36 phase-specific means.  A scientifically labelled legacy mode
retains the earlier event-window covariance and phase-specific sensitivity
arm.  SQP values use only the ten-step release path.  The remaining thirty
steps and their Nino-3 outcome are evaluated only after a control is frozen.

The canonical O3 executable is retained as an exact audit against processed
zc-v3.  Optimization uses the same ZC source compiled at O0, matching the
Tapenade tangent/reverse floating-point contract.  This avoids changing a
discrete atmospheric-convergence branch solely because an O3 primal and O0
reverse-forward replay round a near-threshold iterate differently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import benchmark_fresh_xai_methods as fresh_xai_benchmark  # noqa: E402
import run_zc_agop_adjoint_nudging as established  # noqa: E402
import run_zc_agop_interventions as impulse  # noqa: E402

from adjoint.balanced_control_map import (  # noqa: E402
    IndependentControlLayout,
    load_independent_control_layout,
)
from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.direct_xai_target import (  # noqa: E402
    XAI_METHODS,
    build_direct_xai_target,
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
from zc_xai.native_covariance import NativeCovarianceBundle  # noqa: E402
from zc_xai.native_observation import (  # noqa: E402
    FrozenCore4ObservationChain,
    PackedZCState,
    nino3_from_real_state,
)
from zc_xai.nonlinear_covariance_control import (  # noqa: E402
    NativeScalarSQPIteration,
    NativeScalarSQPResult,
    solve_minimum_native_covariance_action_sqp,
)
from zc_xai.training import (  # noqa: E402
    ExperimentSpec,
    LoadedExperiment,
    load_experiment,
)
from zc_xai.zc_controlled_adjoint_bridge import (  # noqa: E402
    FortranOneStepAdjointRunner,
    reverse_controlled_release_scalar_with_runner,
)
from zc_xai.zc_controlled_bridge import (  # noqa: E402
    ControlledReplay,
    FortranOneStepRunner,
    OneStepRunner,
    ZeroControlPath,
    _build_zero_control_path_with_runner,
    _controlled_input_state,
    verify_zero_control_replay,
)

SCRIPT_VERSION = "1.24.0"
REPORT_SCHEMA_VERSION = 2
CONTINUATION_JOURNAL_SCHEMA_VERSION = 3
# These source files participate directly in the scientific calculation.  Their
# content hashes are part of the run identity, so a cached result cannot silently
# cross a change to the solver, covariance, observation, or Fortran bridge.
SCIENTIFIC_SOURCE_PATHS = (
    Path("scripts/benchmark_fresh_xai_methods.py"),
    Path("scripts/run_zc_agop_adjoint_nudging.py"),
    Path("scripts/run_zc_agop_interventions.py"),
    Path("src/zc_xai/agop_control.py"),
    Path("src/zc_xai/data.py"),
    Path("src/zc_xai/direct_xai_target.py"),
    Path("src/zc_xai/io.py"),
    Path("src/zc_xai/models.py"),
    Path("src/zc_xai/nonlinear_covariance_control.py"),
    Path("src/zc_xai/native_covariance.py"),
    Path("src/zc_xai/native_observation.py"),
    Path("src/zc_xai/training.py"),
    Path("src/zc_xai/xai.py"),
    Path("src/zc_xai/zc_controlled_adjoint_bridge.py"),
    Path("src/zc_xai/zc_controlled_bridge.py"),
    Path("adjoint/balanced_control_map/__init__.py"),
    Path("adjoint/balanced_control_map/balanced_map.py"),
    Path("adjoint/balanced_control_map/phase_local_stack.py"),
)
SELECTION_SEED = 42
MEMBER_COUNT = 10
CONTROL_STEPS = 9
RELEASE_BOUNDARIES = (9, 10)
TOTAL_STEPS = 40
EXPECTED_WARM_RELEASE_INDICES = np.asarray(
    (399095, 399203, 399419, 403271, 411515, 411695, 419399, 421055, 423611, 426779),
    dtype=np.int64,
)
EXPECTED_COLD_RELEASE_INDICES = np.asarray(
    (399086, 399194, 399410, 403262, 411506, 411686, 419390, 421046, 423602, 426770),
    dtype=np.int64,
)
EXPECTED_RELEASE_INDICES_BY_EVENT = {
    "extreme_el_nino": EXPECTED_WARM_RELEASE_INDICES,
    "extreme_la_nina": EXPECTED_COLD_RELEASE_INDICES,
}
# Backward-compatible name retained for tests and downstream readers of the
# original warm experiment.
EXPECTED_RELEASE_INDICES = EXPECTED_WARM_RELEASE_INDICES
CASE_NAMES = ("pooled", "phase-specific")
ANNUAL_SHARED_COVARIANCE_POLICY = "annual-all-phase-shared"
LEGACY_EVENT_WINDOW_COVARIANCE_POLICY = "legacy-event-window-nine-phase"
COVARIANCE_POLICIES = (
    ANNUAL_SHARED_COVARIANCE_POLICY,
    LEGACY_EVENT_WINDOW_COVARIANCE_POLICY,
)
ANNUAL_PHASES = tuple(range(36))
ANNUAL_COVARIANCE_SAMPLE_COUNT_PER_PHASE = 10_000
ANNUAL_COVARIANCE_STATE_SIZE = 28_591
ANNUAL_COVARIANCE_TRAINING_INTERVAL = (0, 360_000)
DEFAULT_ANNUAL_DENSE_COVARIANCE_MANIFEST = Path(
    "outputs/zc_native_covariance/"
    "training-years10000-phases00-35/dense_manifest.json"
)
DENSE_COVARIANCE_SCHEMA_VERSION = 1
DENSE_COVARIANCE_STATUS = "complete_dense_native_covariance"
DEFAULT_CONTINUATION_FRACTIONS = (1.0,)
DEFAULT_ADAPTIVE_MINIMUM_FRACTION_STEP = 0.00625
DEFAULT_MAXIMUM_ADAPTIVE_SUBDIVISIONS = 32
INTERMEDIATE_CONTINUATION_STATIONARITY_MULTIPLIER = 2.0
INTERMEDIATE_CONTINUATION_CONSTRAINT_MULTIPLIER = 5.0


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


def finite_nonzero_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value == 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonzero")
    return value


def continuation_fractions(text: str) -> tuple[float, ...]:
    """Parse a strictly increasing comma-separated continuation schedule."""

    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "continuation fractions must be comma-separated numbers"
        ) from error
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("continuation fractions must be finite")
    if any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("continuation fractions must lie in (0, 1]")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise argparse.ArgumentTypeError(
            "continuation fractions must be strictly increasing"
        )
    if values[-1] != 1.0:
        raise argparse.ArgumentTypeError(
            "the final continuation fraction must be exactly 1.0"
        )
    return values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    result.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    result.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/zc-v3"))
    result.add_argument(
        "--agop-benchmark-dir",
        type=Path,
        default=Path(
            "outputs/fresh_agop_benchmark/"
            "core4-cnn-lead-10m-seed-000042-refs-all-batch-1024"
        ),
        help="Validated exact-AGOP cache; read only when --xai-method=AGOP.",
    )
    result.add_argument(
        "--xai-method",
        choices=XAI_METHODS,
        default="AGOP",
        help="Explanation whose fixed-phase spatial direction defines the target.",
    )
    result.add_argument(
        "--covariance-policy",
        choices=COVARIANCE_POLICIES,
        default=ANNUAL_SHARED_COVARIANCE_POLICY,
        help=(
            "Scientific population used to price native interventions. The "
            "default is one common covariance over all 36 annual phases after "
            "phase-specific centering. The explicitly labelled legacy policy "
            "reproduces the former nine-phase event-window construction."
        ),
    )
    result.add_argument(
        "--covariance-manifest",
        type=Path,
        default=None,
        help=(
            "Native covariance manifest. The annual policy defaults to the "
            "10,000-year phases00-35 dense artifact for both warm and cold "
            "runs. The legacy policy defaults to the former event-window "
            "rectangular-factor artifact. An override must satisfy the selected "
            "policy's full schema and provenance checks."
        ),
    )
    result.add_argument(
        "--generation-workspace",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3"),
    )
    result.add_argument(
        "--adjoint-validation-dir",
        type=Path,
        default=Path("outputs/zc_adjoint/kernel_replay_validation_run23_final3"),
    )
    result.add_argument(
        "--primal-executable",
        type=Path,
        default=Path("adjoint/controlled_window/build/zc_one_step"),
    )
    result.add_argument(
        "--tangent-executable",
        type=Path,
        default=Path("adjoint/controlled_window/build/zc_one_step_tangent"),
        help="Used only to authenticate the canonical O3 controlled build.",
    )
    result.add_argument(
        "--differentiable-primal-executable",
        type=Path,
        default=Path("adjoint/controlled_reverse/build-v5/zc_one_step_primal"),
        help=(
            "O0 primal compiled with the same floating-point contract as the "
            "split reverse sweep. The canonical O3 primal remains the data audit."
        ),
    )
    result.add_argument(
        "--differentiable-tangent-executable",
        type=Path,
        default=Path("adjoint/controlled_reverse/build-v5/zc_one_step_tangent"),
        help="Matching O0 tangent executable, authenticated but not run by SQP.",
    )
    result.add_argument(
        "--adjoint-executable",
        type=Path,
        default=Path("adjoint/controlled_reverse/build-v5/zc_one_step_adjoint"),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/zc_direct_agop_covariance_action/core4-cnn-lead10-seed42"
        ),
    )
    result.add_argument(
        "--target-event",
        choices=("extreme_el_nino", "extreme_la_nina"),
        default="extreme_el_nino",
        help=(
            "Extreme event whose spatial AGOP direction and projection define "
            "the release constraint and annual phase of the uniformly sampled "
            "ten-member release cohort."
        ),
    )
    result.add_argument(
        "--trajectory",
        choices=("uniform-cohort", "authentic-extreme"),
        default="uniform-cohort",
        help=(
            "Optimize either the fixed uniformly sampled Figure 9 cohort or "
            "the authentic trajectory that produces --target-event."
        ),
    )
    result.add_argument(
        "--projection-change-multiple",
        type=finite_nonzero_float,
        default=None,
        help=(
            "Authentic-extreme mode only. Impose the signed release change "
            "e^T(x_release-x_extreme)=c e^T x_extreme. Negative changes are "
            "represented by reversing the inequality direction so the same "
            "active-boundary SQP enforces the target equality."
        ),
    )
    result.add_argument(
        "--initial-dual-npz",
        type=Path,
        default=None,
        help=(
            "Optional failure-checkpoint or result NPZ containing "
            "dual_variables for an exact-target SQP warm start."
        ),
    )
    result.add_argument(
        "--initial-dual-scale",
        type=positive_float,
        default=1.0,
        help=(
            "Positive multiplier applied to dual_variables loaded from "
            "--initial-dual-npz before the exact-target SQP solve. The source "
            "file hash, raw dual hash, scale, and scaled dual hash are all "
            "bound into the run and continuation-journal identities."
        ),
    )
    result.add_argument(
        "--selection-seed",
        type=nonnegative_int,
        default=SELECTION_SEED,
        help=(
            "Random seed for the uniform, without-replacement selection of "
            "the ten same-phase test trajectories."
        ),
    )
    result.add_argument(
        "--member",
        action="append",
        type=int,
        dest="members",
        help="Run one release index from the fixed sample; repeat as needed.",
    )
    result.add_argument(
        "--case",
        choices=("both", *CASE_NAMES),
        default="pooled",
        help=(
            "Covariance arm. The annual shared policy supports only pooled. "
            "The phase-specific and both choices require the explicitly "
            "labelled legacy event-window covariance policy."
        ),
    )
    result.add_argument("--maximum-iterations", type=positive_int, default=30)
    result.add_argument(
        "--continuation-fractions",
        type=continuation_fractions,
        default=DEFAULT_CONTINUATION_FRACTIONS,
        help=(
            "Strictly increasing comma-separated fractions of the required "
            "AGOP-projection displacement. Each stage must pass the unchanged "
            "solver gates, and only the final exact-target stage is published."
        ),
    )
    result.add_argument(
        "--scale-continuation-warm-start",
        action="store_true",
        help=(
            "At each continuation stage after the first, multiply the previous "
            "dual solution by the ratio of the new and previous target fractions. "
            "This is exact extrapolation for a linear homogeneous response."
        ),
    )
    result.add_argument(
        "--adaptive-continuation",
        action="store_true",
        help=(
            "When a requested continuation stage is rejected, bisect the gap "
            "from the last accepted stage and retry from that accepted dual. "
            "The rejected iterate is never used, and the exact-target final "
            "stage must still pass the unchanged publication gates."
        ),
    )
    result.add_argument(
        "--zero-dual-restart-after-warm-rejection",
        action="store_true",
        help=(
            "After a continuation target rejects a previous-stage warm start, "
            "ensure that exact target has been attempted once from zero dual "
            "before considering adaptive bisection. An authenticated prior "
            "zero-dual rejection at the same target is not recomputed, and no "
            "rejected dual is ever reused."
        ),
    )
    result.add_argument(
        "--direct-exact-target-first",
        action="store_true",
        help=(
            "Before continuation, attempt the exact fraction-1 target once from "
            "zero dual under the unchanged final publication gates. A finite "
            "strict-gate rejection atomically activates the configured "
            "continuation schedule; infrastructure errors and nonfinite solver "
            "results fail closed."
        ),
    )
    result.add_argument(
        "--adaptive-minimum-fraction-step",
        type=positive_float,
        default=DEFAULT_ADAPTIVE_MINIMUM_FRACTION_STEP,
        help=(
            "Smallest inserted continuation increment as a fraction of the "
            "full projection displacement."
        ),
    )
    result.add_argument(
        "--maximum-adaptive-subdivisions",
        type=positive_int,
        default=DEFAULT_MAXIMUM_ADAPTIVE_SUBDIVISIONS,
        help="Maximum number of midpoint stages inserted per member and case.",
    )
    result.add_argument(
        "--resume-continuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume each member/case from its newest atomically committed, "
            "accepted continuation stage. Every checkpoint is rehashed and "
            "must exactly match the current model, data, covariance, XAI, "
            "source, solver, run, and case identities. --no-resume-continuation "
            "refuses an existing journal but still journals newly accepted "
            "stages for a later safe restart."
        ),
    )
    result.add_argument("--constraint-tolerance", type=positive_float, default=1e-5)
    result.add_argument("--stationarity-tolerance", type=positive_float, default=1e-5)
    result.add_argument(
        "--complementarity-tolerance", type=positive_float, default=1e-5
    )
    result.add_argument(
        "--relative-stationarity-tolerance",
        type=positive_float,
        default=2e-2,
        help=(
            "Dimensionless covariance-metric KKT tolerance; stationarity is "
            "scaled by max(action norm, multiplier times gradient norm). The "
            "2%% default is above the measured float32 branch-noise floor."
        ),
    )
    result.add_argument(
        "--relative-complementarity-tolerance",
        type=positive_float,
        default=1e-4,
        help=(
            "Dimensionless complementarity tolerance scaled by squared action; "
            "the default is 0.01%%."
        ),
    )
    result.add_argument("--initial-trust-radius", type=positive_float, default=10.0)
    result.add_argument("--maximum-trust-radius", type=positive_float, default=100.0)
    result.add_argument("--covariance-block-rows", type=positive_int, default=1024)
    result.add_argument(
        "--covariance-centered-cache-gib",
        type=nonnegative_float,
        default=0.0,
        help=(
            "Opt-in RAM budget for one exact contiguous float64 centered cache "
            "of all native covariance phases. Zero streams the verified source "
            "factors. A positive but insufficient budget fails before allocation."
        ),
    )
    result.add_argument("--timeout-seconds", type=positive_float, default=120.0)
    result.add_argument("--skip-data-checksums", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _configure_centered_covariance_cache(
    covariance: NativeCovarianceBundle,
    *,
    requested_gib: float,
) -> tuple[NativeCovarianceBundle, dict[str, Any], dict[str, Any]]:
    """Apply the explicit exact-cache policy before any scientific solve."""

    cache_budget_bytes = int(requested_gib * 1024**3)
    cache_required_bytes = covariance.estimated_centered_cache_bytes
    cache_setup_seconds = 0.0
    if requested_gib > 0.0:
        if cache_budget_bytes < cache_required_bytes:
            raise MemoryError(
                "Exact centered native covariance cache requires "
                f"{cache_required_bytes} bytes, but "
                f"--covariance-centered-cache-gib provides {cache_budget_bytes}"
            )
        print(
            "Building one process-wide exact centered native covariance cache "
            f"({cache_required_bytes / 1024**3:.2f} GiB)",
            flush=True,
        )
        cache_started = time.monotonic()
        covariance = covariance.with_contiguous_float64_cache(
            maximum_bytes=cache_budget_bytes
        )
        cache_setup_seconds = time.monotonic() - cache_started
        cache_status = "enabled_exact_centered_float64"
        cache_enabled = True
    else:
        cache_status = "disabled_streaming_verified_sources"
        cache_enabled = False
    cache_realized_bytes = covariance.centered_cache_bytes
    if cache_enabled and cache_realized_bytes != cache_required_bytes:
        raise RuntimeError("Exact centered native covariance cache is incomplete")
    identity = {
        "enabled": cache_enabled,
        "status": cache_status,
        "requested_gib": requested_gib,
        "budget_bytes": cache_budget_bytes,
        "required_exact_bytes": cache_required_bytes,
        "realized_exact_bytes": cache_realized_bytes,
        "storage": "C-contiguous centered float64 phase samples",
        "operator_approximation": False,
    }
    runtime = {
        **identity,
        "setup_wall_seconds_this_process": cache_setup_seconds,
    }
    return covariance, identity, runtime


def _validate_annual_dense_covariance_manifest(
    document: dict[str, Any],
    *,
    data_metadata_sha256: str,
    state_manifest_sha256: str,
    state_size: int,
) -> None:
    """Fail closed unless a dense artifact is the canonical annual covariance."""

    if (
        document.get("schema_version") != DENSE_COVARIANCE_SCHEMA_VERSION
        or document.get("status") != DENSE_COVARIANCE_STATUS
    ):
        raise ValueError("annual native covariance has an unsupported dense schema")
    dimensions = document.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("annual native covariance lacks dimensions")
    expected_dimensions = {
        "state_size": state_size,
        "sample_count_per_phase": ANNUAL_COVARIANCE_SAMPLE_COUNT_PER_PHASE,
        "phase_count": len(ANNUAL_PHASES),
        "phase_offsets": list(ANNUAL_PHASES),
    }
    for key, expected in expected_dimensions.items():
        if dimensions.get(key) != expected:
            raise ValueError(
                f"annual native covariance dimensions disagree for {key}: "
                f"expected {expected!r}"
            )
    if state_size != ANNUAL_COVARIANCE_STATE_SIZE:
        raise ValueError(
            "annual native covariance requires the 28,591-coordinate "
            "independent-state layout"
        )
    if document.get("common_covariance_policy") != "equal-phase pooled covariance":
        raise ValueError("annual native covariance is not equal-phase pooled")

    covariance = document.get("covariance")
    if not isinstance(covariance, dict):
        raise ValueError("annual native covariance manifest lacks its dense matrix")
    expected_matrix = {
        "shape": [state_size, state_size],
        "dtype": np.dtype("<f8").str,
        "order": "F",
    }
    for key, expected in expected_matrix.items():
        if covariance.get(key) != expected:
            raise ValueError(
                f"annual native covariance matrix disagrees for {key}: "
                f"expected {expected!r}"
            )
    for key in ("file", "sha256"):
        value = covariance.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"annual native covariance matrix lacks {key}")
    if len(str(covariance["sha256"])) != 64:
        raise ValueError("annual native covariance matrix has an invalid SHA-256")

    source = document.get("source_capture")
    if not isinstance(source, dict):
        raise ValueError("annual native covariance lacks source-capture provenance")
    if source.get("training_half_open_interval") != list(
        ANNUAL_COVARIANCE_TRAINING_INTERVAL
    ):
        raise ValueError(
            "annual native covariance does not use the full 10,000-year "
            "training block"
        )
    if source.get("data_metadata_sha256") != data_metadata_sha256:
        raise ValueError("annual native covariance belongs to another zc-v3 data set")
    if source.get("state_manifest_sha256") != state_manifest_sha256:
        raise ValueError(
            "annual native covariance uses another independent-state layout"
        )


def _load_annual_dense_covariance(
    manifest_path: Path,
    *,
    data_metadata_sha256: str,
    state_manifest_sha256: str,
    state_size: int,
) -> tuple[Any, dict[str, Any]]:
    """Load the annual dense operator through one isolated integration seam."""

    document = load_json(manifest_path)
    _validate_annual_dense_covariance_manifest(
        document,
        data_metadata_sha256=data_metadata_sha256,
        state_manifest_sha256=state_manifest_sha256,
        state_size=state_size,
    )
    # DenseNativeCovarianceOperator is intentionally imported here.  This keeps the
    # policy driver importable while the independently versioned dense builder
    # and loader are installed, and isolates the only adapter needed if that
    # loader's module boundary changes.
    try:
        from zc_xai.native_covariance import DenseNativeCovarianceOperator
    except ImportError as error:  # pragma: no cover - isolated integration seam
        raise RuntimeError(
            "DenseNativeCovarianceOperator is required for the annual all-phase "
            "covariance policy"
        ) from error
    operator = DenseNativeCovarianceOperator.load(
        manifest_path, verify_hashes=True
    )
    if int(operator.state_size) != state_size:
        raise ValueError("loaded annual native covariance has the wrong state size")
    return operator, document


def _dense_covariance_runtime(
    document: dict[str, Any], *, requested_centered_cache_gib: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Describe why the legacy centered-sample cache is inapplicable to dense C."""

    covariance = document["covariance"]
    identity = {
        "enabled": False,
        "status": "not_applicable_precompiled_dense_covariance",
        "requested_gib": requested_centered_cache_gib,
        "budget_bytes": int(requested_centered_cache_gib * 1024**3),
        "required_exact_bytes": 0,
        "realized_exact_bytes": 0,
        "storage": "precompiled Fortran-order dense float64 covariance",
        "dense_covariance_size_bytes": covariance.get("size_bytes"),
        "centered_sample_cache_request_applied": False,
        "operator_approximation": False,
    }
    return identity, {**identity, "setup_wall_seconds_this_process": 0.0}


def _scientific_source_provenance() -> dict[str, str]:
    """Return hashes for every local Python source that defines the calculation."""

    provenance: dict[str, str] = {}
    for relative_path in SCIENTIFIC_SOURCE_PATHS:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"scientific source file is missing: {path}")
        provenance[relative_path.as_posix()] = sha256_file(path)
    return provenance


def _ensure_xai_target_artifact(
    output_dir: Path,
    *,
    event_standardized: np.ndarray,
    target: Any,
) -> dict[str, Any]:
    """Persist and validate every numerical array defining the XAI direction."""

    arrays = {
        "event_input_standardized": np.asarray(event_standardized, dtype=np.float64),
        "explanation_unit": np.asarray(target.explanation, dtype=np.float64),
        "fixed_phase_explanation_unit": np.asarray(
            target.fixed_phase_explanation, dtype=np.float64
        ),
        "raw_spatial_direction_unit": np.asarray(
            target.raw_spatial_direction, dtype=np.float64
        ),
        "oriented_spatial_direction_unit": np.asarray(
            target.oriented_spatial_direction, dtype=np.float64
        ),
        **{key: np.asarray(value) for key, value in target.reference_arrays.items()},
    }
    path = output_dir / "xai_target.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError("cached XAI target artifact has different arrays")
            for key, expected in arrays.items():
                actual = archive[key]
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or sha256_array(actual) != sha256_array(expected)
                ):
                    raise ValueError(f"cached XAI target array changed: {key}")
    else:
        write_npz(path, overwrite=False, **arrays)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": sha256_array(value),
            }
            for key, value in arrays.items()
        },
    }


def control_phases_for_release_phase(
    release_phase: int, *, steps_per_year: int
) -> tuple[int, ...]:
    """Return the nine temporally ordered phases preceding a release."""

    if isinstance(release_phase, bool) or not isinstance(release_phase, int):
        raise TypeError("release_phase must be an integer")
    if isinstance(steps_per_year, bool) or not isinstance(steps_per_year, int):
        raise TypeError("steps_per_year must be an integer")
    if steps_per_year <= CONTROL_STEPS:
        raise ValueError("steps_per_year must exceed the controlled window")
    if not 0 <= release_phase < steps_per_year:
        raise ValueError("release_phase must lie inside the annual cycle")
    return tuple(
        (release_phase - CONTROL_STEPS + offset) % steps_per_year
        for offset in range(CONTROL_STEPS)
    )


def default_covariance_manifest(
    control_phases: Sequence[int] | None = None,
    *,
    covariance_policy: str = ANNUAL_SHARED_COVARIANCE_POLICY,
) -> Path:
    """Return the policy-bound covariance artifact.

    The annual artifact is deliberately independent of the event release phase.
    Event-specific path selection survives only behind the explicitly labelled
    legacy policy.
    """

    if covariance_policy == ANNUAL_SHARED_COVARIANCE_POLICY:
        return DEFAULT_ANNUAL_DENSE_COVARIANCE_MANIFEST
    if covariance_policy != LEGACY_EVENT_WINDOW_COVARIANCE_POLICY:
        raise ValueError(f"unknown covariance policy: {covariance_policy}")
    if control_phases is None:
        raise ValueError("the legacy covariance policy requires control phases")
    phases = tuple(int(value) for value in control_phases)
    if len(phases) != CONTROL_STEPS:
        raise ValueError("the legacy covariance manifest requires nine control phases")
    phase_label = "-".join(str(value) for value in phases)
    if all(right == left + 1 for left, right in zip(phases, phases[1:], strict=False)):
        phase_label = f"{phases[0]}-{phases[-1]}"
    return Path(
        "outputs/zc_native_covariance/"
        f"training-years10000-phases{phase_label}/manifest.json"
    )


def covariance_cases(case: str, *, covariance_policy: str) -> tuple[str, ...]:
    """Resolve cases while preventing implicit use of the legacy geometry."""

    if case == "both":
        cases = CASE_NAMES
    elif case in CASE_NAMES:
        cases = (case,)
    else:
        raise ValueError(f"unknown covariance case: {case}")
    if (
        covariance_policy == ANNUAL_SHARED_COVARIANCE_POLICY
        and cases != ("pooled",)
    ):
        raise ValueError(
            "the annual all-phase shared covariance policy supports only the "
            "pooled case; select --covariance-policy "
            f"{LEGACY_EVENT_WINDOW_COVARIANCE_POLICY} explicitly to reproduce "
            "the former phase-specific sensitivity arm"
        )
    if covariance_policy not in COVARIANCE_POLICIES:
        raise ValueError(f"unknown covariance policy: {covariance_policy}")
    return cases


def select_uniform_release_indices(
    test_inputs: np.ndarray,
    *,
    test_interval: tuple[int, int],
    event_index: int,
    steps_per_year: int,
    seed: int = SELECTION_SEED,
    count: int = MEMBER_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Select uniformly without using any member state or outcome values."""

    candidates = np.asarray(test_inputs, dtype=np.int64)
    if candidates.ndim != 1 or np.any(np.diff(candidates) <= 0):
        raise ValueError("test inputs must be a strictly increasing vector")
    lower, upper = (int(test_interval[0]), int(test_interval[1]))
    eligible = candidates[
        (candidates - CONTROL_STEPS >= lower)
        & (candidates + (TOTAL_STEPS - RELEASE_BOUNDARIES[1]) < upper)
        & (candidates % steps_per_year == event_index % steps_per_year)
    ]
    if eligible.size < count:
        raise RuntimeError("too few same-phase test releases for uniform sampling")
    generator = np.random.default_rng(seed)
    selected = np.sort(generator.choice(eligible, size=count, replace=False))
    return np.asarray(selected, dtype=np.int64), np.asarray(eligible, dtype=np.int64)


def _validate_seed_42_release_sample(
    selected: np.ndarray, *, target_event: str, seed: int
) -> None:
    """Guard the published seed-42 cohorts without constraining other seeds."""

    if seed != SELECTION_SEED:
        return
    expected_release_indices = EXPECTED_RELEASE_INDICES_BY_EVENT[target_event]
    if not np.array_equal(selected, expected_release_indices):
        raise RuntimeError(f"fixed seed-42 {target_event} release sample changed")


def _copy_state(state: PackedZCState) -> PackedZCState:
    return PackedZCState(
        real32=np.array(state.real32, copy=True),
        complex64=np.array(state.complex64, copy=True),
        real64=np.array(state.real64, copy=True),
        integers=np.array(state.integers, copy=True),
        passive_time=np.array(state.passive_time, copy=True),
    )


def replay_native_interventions_with_runner(
    initial_state: PackedZCState,
    compact_interventions: np.ndarray,
    layout: IndependentControlLayout,
    runner: OneStepRunner,
    *,
    total_steps: int,
) -> ControlledReplay:
    """Insert nine compact physical increments immediately before steps 0--8."""

    compact = np.asarray(compact_interventions, dtype=np.float64)
    expected = (CONTROL_STEPS, layout.compact_size)
    if compact.shape != expected or not np.isfinite(compact).all():
        raise ValueError(f"compact interventions must be finite with shape {expected}")
    if total_steps < CONTROL_STEPS:
        raise ValueError("total_steps cannot end inside the controlled window")
    packed = layout.compact_to_packed(compact)
    zero = np.zeros(layout.packed_size, dtype=np.float64)
    states = [_copy_state(initial_state)]
    inputs: list[PackedZCState] = []
    tapes: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    elapsed = 0.0
    state = states[0]
    for step in range(total_steps):
        increment = packed[step] if step < CONTROL_STEPS else zero
        step_input = _controlled_input_state(state, increment)
        result = runner.advance(step_input)
        inputs.append(step_input)
        increments.append(np.asarray(increment, dtype=np.float64).copy())
        tapes.append(np.asarray(result.tape, dtype=np.int32).copy())
        state = _copy_state(result.state)
        states.append(state)
        elapsed += float(result.wall_seconds)
    return ControlledReplay(
        states=tuple(states),
        step_inputs=tuple(inputs),
        tapes=tuple(tapes),
        applied_real32_increments=tuple(increments),
        wall_seconds=elapsed,
    )


def _model_forecast(experiment: LoadedExperiment, values: np.ndarray) -> float:
    experiment.model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)[None])
        return float(experiment.model(tensor).detach().cpu().item())


class ReleaseConstraintOracle:
    """Release-only nonlinear value and matched-O0 reverse-gradient oracle."""

    def __init__(
        self,
        *,
        initial_state: PackedZCState,
        zero_release_replay: ControlledReplay,
        layout: IndependentControlLayout,
        observation_chain: FrozenCore4ObservationChain,
        direction: np.ndarray,
        target_projection: float,
        primal_runner: OneStepRunner,
        adjoint_runner: FortranOneStepAdjointRunner,
    ) -> None:
        self.initial_state = initial_state
        self.layout = layout
        self.chain = observation_chain
        self.direction = np.asarray(direction, dtype=np.float64)
        self.target = float(target_projection)
        self.primal_runner = primal_runner
        self.adjoint_runner = adjoint_runner
        self.forward_replays = 0
        self.reverse_sweeps = 0
        self.reverse_wall_seconds = 0.0
        zero = np.zeros((CONTROL_STEPS, layout.compact_size), dtype=np.float64)
        self._zero_key = sha256_array(zero)
        self._last: tuple[str, ControlledReplay, np.ndarray, float] | None = None
        self._zero = self._record(self._zero_key, zero_release_replay)
        self._zero_gradient: np.ndarray | None = None

    def _record(
        self, key: str, replay: ControlledReplay
    ) -> tuple[str, ControlledReplay, np.ndarray, float]:
        release = self.chain.forward(
            replay.states[RELEASE_BOUNDARIES[0]],
            replay.states[RELEASE_BOUNDARIES[1]],
        )
        projection = float(self.direction @ release[:-2].astype(np.float64))
        return key, replay, release, projection - self.target

    def evaluate(
        self, compact_interventions: np.ndarray
    ) -> tuple[ControlledReplay, np.ndarray, float]:
        values = np.asarray(compact_interventions, dtype=np.float64)
        expected = (CONTROL_STEPS, self.layout.compact_size)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"oracle controls must be finite with shape {expected}")
        key = sha256_array(values)
        if key == self._zero_key:
            record = self._zero
        elif self._last is not None and key == self._last[0]:
            record = self._last
        else:
            replay = replay_native_interventions_with_runner(
                self.initial_state,
                values,
                self.layout,
                self.primal_runner,
                total_steps=RELEASE_BOUNDARIES[1],
            )
            self.forward_replays += 1
            record = self._record(key, replay)
            self._last = record
        return record[1], record[2], record[3]

    def value(self, compact_interventions: np.ndarray) -> float:
        return self.evaluate(compact_interventions)[2]

    def value_gradient(
        self, compact_interventions: np.ndarray
    ) -> tuple[float, np.ndarray]:
        replay, _, value = self.evaluate(compact_interventions)
        key = sha256_array(np.asarray(compact_interventions, dtype=np.float64))
        if key == self._zero_key and self._zero_gradient is not None:
            return value, self._zero_gradient.copy()
        reverse = reverse_controlled_release_scalar_with_runner(
            replay,
            self.chain,
            self.direction,
            self.adjoint_runner,
            control_steps=CONTROL_STEPS,
            release=RELEASE_BOUNDARIES,
        )
        self.reverse_sweeps += 1
        self.reverse_wall_seconds += reverse.wall_seconds
        compact_gradient = self.layout.packed_to_compact(
            reverse.native_control_covectors
        )
        compact_gradient = np.asarray(compact_gradient, dtype=np.float64)
        if key == self._zero_key:
            self._zero_gradient = compact_gradient.copy()
        return value, compact_gradient


class ProjectionContinuationFailure(RuntimeError):
    """Carry a non-publishable SQP iterate out of a failed continuation stage."""

    def __init__(
        self,
        message: str,
        *,
        result: NativeScalarSQPResult,
        stage_fraction: float,
        stage_records: Sequence[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.result = result
        self.stage_fraction = float(stage_fraction)
        self.stage_records = tuple(stage_records)


_CONTINUATION_RESULT_ARRAY_NAMES = (
    "dual_variables",
    "native_interventions",
    "native_constraint_gradient",
)


def _continuation_result_metadata(result: NativeScalarSQPResult) -> dict[str, Any]:
    """Serialize the non-array portion of an accepted SQP result."""

    return {
        key: value
        for key, value in {
            **result.__dict__,
            "iterations": [asdict(item) for item in result.iterations],
        }.items()
        if key not in _CONTINUATION_RESULT_ARRAY_NAMES
    }


def _continuation_result_from_checkpoint(
    document: dict[str, Any], arrays: dict[str, np.ndarray]
) -> NativeScalarSQPResult:
    """Reconstruct a final accepted result after validating its array hashes."""

    metadata = document.get("accepted_result")
    if not isinstance(metadata, dict):
        raise ValueError("final continuation checkpoint has no accepted result")
    raw_iterations = metadata.get("iterations")
    if not isinstance(raw_iterations, list):
        raise ValueError("continuation checkpoint has invalid SQP iterations")
    scalar = dict(metadata)
    scalar["iterations"] = tuple(
        NativeScalarSQPIteration(**item) for item in raw_iterations
    )
    try:
        return NativeScalarSQPResult(
            native_interventions=arrays["native_interventions"],
            dual_variables=arrays["dual_variables"],
            native_constraint_gradient=arrays["native_constraint_gradient"],
            **scalar,
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "continuation checkpoint has invalid SQP result fields"
        ) from error


def _zero_dual_solve_identity_sha256(
    *,
    continuation_context_sha256: str,
    fraction: float,
    target_projection: float,
) -> str:
    """Identify one exact zero-initialized continuation solve.

    Float hex strings avoid treating merely nearby adaptive fractions as the
    same target.  The continuation context is the durable journal identity in
    production, which already binds the scientific inputs, covariance, solver,
    gates, baseline, full target, and requested fallback schedule.
    """

    return sha256_json(
        {
            "schema_version": 1,
            "continuation_context_sha256": continuation_context_sha256,
            "fraction_hex": float(fraction).hex(),
            "target_projection_hex": float(target_projection).hex(),
            "initial_dual_variables": None,
        }
    )


def _rejected_zero_dual_ledger(
    stage_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the ordered audit ledger of zero-dual solver rejections."""

    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in stage_records:
        if (
            record.get("initialization_mode") != "zero_dual"
            or record.get("accepted_for_continuation") is not False
        ):
            continue
        identity = record.get("zero_dual_solve_identity_sha256")
        if not isinstance(identity, str) or identity in seen:
            raise ValueError(
                "continuation history repeats or omits a rejected zero-dual "
                "solve identity"
            )
        seen.add(identity)
        ledger.append(
            {
                "zero_dual_solve_identity_sha256": identity,
                "stage": record.get("stage"),
                "fraction": record.get("fraction"),
                "fraction_hex": float(record["fraction"]).hex(),
                "target_projection_hex": float(record["target_projection"]).hex(),
                "result_dual_sha256": record.get("result_dual_sha256"),
                "stage_record_sha256": sha256_json(record),
            }
        )
    return ledger


class DurableContinuationJournal:
    """Atomic, identity-bound checkpoints for accepted continuation stages.

    A checkpoint is explicitly non-publishable.  Intermediate checkpoints keep
    the exact accepted dual required by the next SQP stage; the final checkpoint
    additionally stores the full accepted SQP result so an interruption between
    the solve and the frozen replay does not force the exact-target stage to be
    recomputed.  Rejected iterates are recorded in the accumulated stage log but
    their arrays are never written and can therefore never become warm starts.
    """

    def __init__(
        self,
        case_dir: Path,
        *,
        run_identity: dict[str, Any],
        case_identity: dict[str, Any],
        covariance_case: str,
        baseline_projection: float,
        target_projection: float,
        requested_fractions: Sequence[float],
        resume_enabled: bool,
    ) -> None:
        self.directory = case_dir / "continuation_journal"
        self.pending_path = self.directory / "pending_retry.json"
        self.resume_enabled = bool(resume_enabled)
        self.identity = {
            "schema_version": CONTINUATION_JOURNAL_SCHEMA_VERSION,
            "run_identity": run_identity,
            "run_identity_sha256": sha256_json(run_identity),
            "case_identity": case_identity,
            "case_identity_sha256": sha256_json(case_identity),
            "covariance_case": covariance_case,
            "baseline_projection": float(baseline_projection),
            "target_projection": float(target_projection),
            "requested_fractions": [float(value) for value in requested_fractions],
            "direct_exact_target_first": bool(
                case_identity["solver_settings"].get("direct_exact_target_first", False)
            ),
            "control_steps": CONTROL_STEPS,
            "acceptance_policy": _projection_continuation_acceptance_policy(),
        }
        self.identity_sha256 = sha256_json(self.identity)
        self._checkpoint_count = 0
        self._latest_record: dict[str, Any] | None = None
        self._pending_record: dict[str, Any] | None = None
        self.resumed_this_invocation = False
        self.resumed_attempt_count = 0
        self._pending_active = False

    @staticmethod
    def _document_bytes(document: dict[str, Any]) -> np.ndarray:
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return np.frombuffer(encoded, dtype=np.uint8).copy()

    @staticmethod
    def _array_record(array: np.ndarray) -> dict[str, Any]:
        return {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": sha256_array(array),
        }

    @staticmethod
    def _decode_document(path: Path, archive: Any) -> dict[str, Any]:
        if "document_utf8" not in archive.files:
            raise ValueError(f"continuation checkpoint has no document: {path}")
        payload = np.asarray(archive["document_utf8"])
        if payload.ndim != 1 or payload.dtype != np.dtype(np.uint8):
            raise ValueError(f"continuation checkpoint document is malformed: {path}")
        try:
            document = json.loads(payload.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"continuation checkpoint document is corrupted: {path}"
            ) from error
        if not isinstance(document, dict):
            raise ValueError(
                f"continuation checkpoint document is not an object: {path}"
            )
        return document

    def _load_one(self, path: Path, *, expected_sequence: int) -> dict[str, Any]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                document = self._decode_document(path, archive)
                if (
                    document.get("schema_version")
                    != CONTINUATION_JOURNAL_SCHEMA_VERSION
                ):
                    raise ValueError(f"continuation checkpoint schema changed: {path}")
                if document.get("status") != "accepted_stage_nonpublishable":
                    raise ValueError(
                        f"continuation checkpoint status is invalid: {path}"
                    )
                if document.get("scientific_result_published") is not False:
                    raise ValueError(
                        "continuation checkpoint incorrectly claims publication: "
                        f"{path}"
                    )
                if document.get("sequence") != expected_sequence:
                    raise ValueError(
                        f"continuation checkpoint sequence is invalid: {path}"
                    )
                if (
                    document.get("journal_identity_sha256") != self.identity_sha256
                    or document.get("journal_identity") != self.identity
                ):
                    raise ValueError(
                        "continuation checkpoint belongs to another scientific, "
                        f"solver, run, or case identity: {path}"
                    )
                records = document.get("stage_records")
                scheduler = document.get("scheduler")
                array_records = document.get("arrays")
                accepted_fraction = document.get("accepted_fraction")
                if (
                    not isinstance(records, list)
                    or not records
                    or any(
                        not isinstance(item, dict)
                        or isinstance(item.get("fraction"), bool)
                        or not isinstance(item.get("fraction"), (int, float))
                        or not math.isfinite(float(item["fraction"]))
                        or not isinstance(item.get("accepted_for_continuation"), bool)
                        for item in records
                    )
                    or not isinstance(scheduler, dict)
                    or not isinstance(array_records, dict)
                    or isinstance(accepted_fraction, bool)
                    or not isinstance(accepted_fraction, (int, float))
                    or not math.isfinite(float(accepted_fraction))
                ):
                    raise ValueError(
                        f"continuation checkpoint structure is invalid: {path}"
                    )
                last_record = records[-1]
                if (
                    not isinstance(last_record, dict)
                    or last_record.get("accepted_for_continuation") is not True
                    or last_record.get("fraction") != accepted_fraction
                    or last_record.get("acceptance_basis") == "rejected"
                ):
                    raise ValueError(
                        "continuation checkpoint does not end at an accepted "
                        f"stage: {path}"
                    )
                planned = scheduler.get("planned_fractions")
                next_position = scheduler.get("next_position")
                subdivision_count = scheduler.get("adaptive_subdivision_count")
                pending_zero_dual_retry_fraction = scheduler.get(
                    "pending_zero_dual_retry_fraction", object()
                )
                direct_status = scheduler.get("direct_exact_target_status")
                direct_enabled = self.identity["direct_exact_target_first"]
                if (
                    not isinstance(planned, list)
                    or not planned
                    or any(
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not math.isfinite(float(item))
                        or not 0.0 < float(item) <= 1.0
                        for item in planned
                    )
                    or any(
                        float(right) <= float(left)
                        for left, right in zip(planned, planned[1:], strict=False)
                    )
                    or float(planned[-1]) != 1.0
                    or (
                        direct_status != "accepted"
                        and any(
                            requested not in [float(item) for item in planned]
                            for requested in self.identity["requested_fractions"]
                        )
                    )
                    or direct_status
                    not in (
                        {"accepted", "rejected"}
                        if direct_enabled
                        else {"not_requested"}
                    )
                    or (direct_status == "accepted" and planned != [1.0])
                    or isinstance(next_position, bool)
                    or not isinstance(next_position, int)
                    or not 1 <= next_position <= len(planned)
                    or next_position != expected_sequence
                    or float(planned[next_position - 1]) != float(accepted_fraction)
                    or isinstance(subdivision_count, bool)
                    or not isinstance(subdivision_count, int)
                    or subdivision_count < 0
                    or pending_zero_dual_retry_fraction is not None
                ):
                    raise ValueError(
                        f"continuation checkpoint scheduler state is invalid: {path}"
                    )
                accepted_records = [
                    item
                    for item in records
                    if item.get("accepted_for_continuation") is True
                ]
                if (
                    len(accepted_records) != next_position
                    or [float(item["fraction"]) for item in accepted_records]
                    != [float(item) for item in planned[:next_position]]
                    or any(
                        item.get("rejected_iterate_used_as_warm_start") is not False
                        for item in records
                        if item.get("accepted_for_continuation") is False
                    )
                    or scheduler.get("rejected_zero_dual_attempts")
                    != _rejected_zero_dual_ledger(records)
                ):
                    raise ValueError(
                        f"continuation checkpoint acceptance history is invalid: {path}"
                    )
                arrays: dict[str, np.ndarray] = {}
                expected_array_names = {"dual_variables"}
                if next_position == len(planned):
                    expected_array_names.update(
                        ("native_interventions", "native_constraint_gradient")
                    )
                if set(array_records) != expected_array_names:
                    raise ValueError(
                        f"continuation checkpoint array inventory is invalid: {path}"
                    )
                if set(archive.files) != {"document_utf8", *expected_array_names}:
                    raise ValueError(
                        f"continuation checkpoint archive inventory is invalid: {path}"
                    )
                for name in expected_array_names:
                    value = np.asarray(archive[name])
                    record = array_records[name]
                    if (
                        not isinstance(record, dict)
                        or list(value.shape) != record.get("shape")
                        or value.dtype.str != record.get("dtype")
                        or sha256_array(value) != record.get("sha256")
                        or value.dtype != np.dtype(np.float64)
                        or value.ndim != 2
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(
                            f"continuation checkpoint array is corrupted: {path}:{name}"
                        )
                    arrays[name] = value.copy()
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith(
                "continuation checkpoint"
            ):
                raise
            raise ValueError(
                f"cannot validate continuation checkpoint: {path}"
            ) from error
        if arrays["dual_variables"].shape[0] != CONTROL_STEPS:
            raise ValueError(f"continuation checkpoint dual shape is invalid: {path}")
        dual_sha256 = sha256_array(arrays["dual_variables"])
        if (
            document.get("accepted_dual_sha256") != dual_sha256
            or document["stage_records"][-1].get("result_dual_sha256") != dual_sha256
        ):
            raise ValueError(
                f"continuation checkpoint dual provenance is invalid: {path}"
            )
        if len(arrays) == 3 and any(
            value.shape != arrays["dual_variables"].shape for value in arrays.values()
        ):
            raise ValueError(f"continuation checkpoint result shapes disagree: {path}")
        final_result = (
            _continuation_result_from_checkpoint(document, arrays)
            if next_position == len(planned)
            else None
        )
        if final_result is not None:
            checkpoint_gates = _solver_publication_gates(
                final_result,
                solver_settings=self.identity["case_identity"]["solver_settings"],
            )
            if not checkpoint_gates["all_passed"] or checkpoint_gates != document[
                "stage_records"
            ][-1].get("publication_gates"):
                raise ValueError(
                    f"final continuation checkpoint gates are invalid: {path}"
                )
        return {
            "document": document,
            "dual_variables": arrays["dual_variables"],
            "final_result": final_result,
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    def _load_pending_transition(
        self,
        previous: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate an unfinished scheduler transition without solver arrays."""

        path = self.pending_path
        try:
            document = load_json(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot validate pending continuation transition: {path}"
            ) from error
        if (
            document.get("schema_version") != CONTINUATION_JOURNAL_SCHEMA_VERSION
            or document.get("status") != "pending_scheduler_transition_nonpublishable"
            or document.get("scientific_result_published") is not False
            or document.get("canonical_result_or_report_created") is not False
        ):
            raise ValueError(f"pending continuation transition is invalid: {path}")
        if (
            document.get("journal_identity_sha256") != self.identity_sha256
            or document.get("journal_identity") != self.identity
        ):
            raise ValueError(
                "pending continuation transition belongs to another scientific, "
                f"solver, run, or case identity: {path}"
            )
        records = document.get("stage_records")
        scheduler = document.get("scheduler")
        if (
            not isinstance(records, list)
            or not records
            or any(not isinstance(item, dict) for item in records)
            or records[-1].get("accepted_for_continuation") is not False
            or not isinstance(scheduler, dict)
            or document.get("stage_records_sha256")
            != sha256_json({"stage_records": records})
            or document.get("scheduler_sha256") != sha256_json({"scheduler": scheduler})
        ):
            raise ValueError(
                f"pending continuation transition structure is invalid: {path}"
            )
        base_sequence = document.get("base_accepted_checkpoint_sequence")
        base_sha256 = document.get("base_accepted_checkpoint_sha256")
        expected_sequence = (
            0 if previous is None else int(previous["document"]["sequence"])
        )
        expected_sha256 = None if previous is None else previous["sha256"]
        if base_sequence != expected_sequence or base_sha256 != expected_sha256:
            raise ValueError(
                f"pending continuation transition has the wrong accepted base: {path}"
            )

        planned = scheduler.get("planned_fractions")
        next_position = scheduler.get("next_position")
        subdivision_count = scheduler.get("adaptive_subdivision_count")
        pending_zero_dual = scheduler.get("pending_zero_dual_retry_fraction", object())
        direct_status = scheduler.get("direct_exact_target_status")
        direct_enabled = self.identity["direct_exact_target_first"]
        if (
            not isinstance(planned, list)
            or not planned
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not 0.0 < float(item) <= 1.0
                for item in planned
            )
            or any(
                float(right) <= float(left)
                for left, right in zip(planned, planned[1:], strict=False)
            )
            or float(planned[-1]) != 1.0
            or any(
                requested not in [float(item) for item in planned]
                for requested in self.identity["requested_fractions"]
            )
            or isinstance(next_position, bool)
            or not isinstance(next_position, int)
            or not 0 <= next_position < len(planned)
            or isinstance(subdivision_count, bool)
            or not isinstance(subdivision_count, int)
            or subdivision_count < 0
            or direct_status
            not in ({"rejected"} if direct_enabled else {"not_requested"})
            or (
                pending_zero_dual is not None
                and (
                    isinstance(pending_zero_dual, bool)
                    or not isinstance(pending_zero_dual, (int, float))
                    or not math.isfinite(float(pending_zero_dual))
                    or float(pending_zero_dual) != float(planned[next_position])
                )
            )
        ):
            raise ValueError(f"pending continuation scheduler state is invalid: {path}")
        accepted_records = [
            item for item in records if item.get("accepted_for_continuation") is True
        ]
        if (
            len(accepted_records) != next_position
            or [float(item["fraction"]) for item in accepted_records]
            != [float(item) for item in planned[:next_position]]
            or sum(
                item.get("adaptive_subdivision_triggered") is True for item in records
            )
            != subdivision_count
            or scheduler.get("rejected_zero_dual_attempts")
            != _rejected_zero_dual_ledger(records)
        ):
            raise ValueError(
                f"pending continuation acceptance history is invalid: {path}"
            )
        last = records[-1]
        if pending_zero_dual is not None:
            valid_transition = bool(
                last.get("zero_dual_restart_triggered") is True
                and last.get("zero_dual_retry_fraction") == pending_zero_dual
                and last.get("adaptive_subdivision_triggered") is False
                and last.get("adaptive_retry_fraction") is None
            )
        else:
            direct_fallback = bool(
                last.get("direct_exact_target_trial") is True
                and last.get("direct_fallback_triggered") is True
                and next_position == 0
                and direct_status == "rejected"
                and planned == self.identity["requested_fractions"]
            )
            adaptive_retry = bool(
                last.get("zero_dual_restart_triggered") is False
                and last.get("zero_dual_retry_fraction") is None
                and last.get("adaptive_subdivision_triggered") is True
                and last.get("adaptive_retry_fraction") == planned[next_position]
            )
            valid_transition = direct_fallback or adaptive_retry
        if not valid_transition:
            raise ValueError(
                f"pending continuation retry transition is invalid: {path}"
            )
        current = {
            "document": document,
            "dual_variables": None if previous is None else previous["dual_variables"],
            "final_result": None,
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        self._validate_attempt_linkage(
            current,
            previous,
            expected_new_acceptances=0,
        )
        return current

    def _validate_attempt_linkage(
        self,
        current: dict[str, Any],
        previous: dict[str, Any] | None,
        *,
        expected_new_acceptances: int = 1,
    ) -> None:
        """Prove that every attempted stage starts from the last accepted dual."""

        document = current["document"]
        records = document["stage_records"]
        prefix_count = (
            0 if previous is None else len(previous["document"]["stage_records"])
        )
        new_records = records[prefix_count:]
        accepted_count = sum(
            item["accepted_for_continuation"] is True for item in new_records
        )
        if (
            not new_records
            or accepted_count != expected_new_acceptances
            or (
                expected_new_acceptances == 1
                and new_records[-1]["accepted_for_continuation"] is not True
            )
            or (
                expected_new_acceptances == 0
                and new_records[-1]["accepted_for_continuation"] is not False
            )
        ):
            raise ValueError(
                "continuation journal adds the wrong accepted-stage transition: "
                f"{current['file']}"
            )
        prior_dual = None if previous is None else previous["dual_variables"]
        prior_fraction = (
            None
            if previous is None
            else float(previous["document"]["accepted_fraction"])
        )
        prior_sha256 = None if prior_dual is None else sha256_array(prior_dual)
        solver_settings = self.identity["case_identity"]["solver_settings"]
        scale_enabled = solver_settings.get("scale_continuation_warm_start", False)
        direct_enabled = solver_settings.get("direct_exact_target_first", False)
        zero_dual_fallback_enabled = solver_settings.get(
            "zero_dual_restart_after_warm_rejection", False
        )
        run_identity = self.identity["run_identity"]
        external_sha256 = run_identity.get("initial_dual_variables_sha256")
        external_raw_sha256 = run_identity.get(
            "initial_dual_raw_variables_sha256", external_sha256
        )
        external_scale = run_identity.get("initial_dual_scale")
        if external_scale is None:
            # Journals created before explicit source scaling used the raw
            # source dual unchanged.
            external_scale = 1.0
        external_scale = float(external_scale)
        requested = self.identity["requested_fractions"]
        pending_zero_dual_fraction: float | None = None
        zero_dual_retry_fractions: list[float] = []
        rejected_zero_dual_by_identity: dict[str, dict[str, Any]] = {}
        for prefix_record in records[:prefix_count]:
            if prefix_record.get("zero_dual_restart_triggered") is True:
                retry_fraction = float(prefix_record["zero_dual_retry_fraction"])
                if retry_fraction in zero_dual_retry_fractions:
                    raise ValueError(
                        "continuation journal schedules more than one zero-dual "
                        f"retry for fraction {retry_fraction:g}: {current['file']}"
                    )
                zero_dual_retry_fractions.append(retry_fraction)
            if (
                prefix_record.get("initialization_mode") == "zero_dual"
                and prefix_record.get("accepted_for_continuation") is False
            ):
                identity = prefix_record.get("zero_dual_solve_identity_sha256")
                if not isinstance(identity, str) or identity in (
                    rejected_zero_dual_by_identity
                ):
                    raise ValueError(
                        "continuation journal repeats a rejected zero-dual solve: "
                        f"{current['file']}"
                    )
                rejected_zero_dual_by_identity[identity] = prefix_record
        for offset, record in enumerate(new_records):
            absolute_position = prefix_count + offset + 1
            fraction = float(record["fraction"])
            if (
                pending_zero_dual_fraction is not None
                and fraction != pending_zero_dual_fraction
            ):
                raise ValueError(
                    "continuation journal does not perform its pending zero-dual "
                    f"retry immediately: {current['file']}"
                )
            external = absolute_position == 1 and external_sha256 is not None
            zero_dual_attempt = pending_zero_dual_fraction == fraction
            if zero_dual_attempt:
                expected_previous_sha256 = prior_sha256
                expected_initial_sha256 = None
                expected_scale = None
            elif external:
                expected_previous_sha256 = external_raw_sha256
                expected_initial_sha256 = external_sha256
                expected_scale: float | None = external_scale
            elif prior_dual is not None:
                if prior_fraction is None:  # pragma: no cover - invariant
                    raise RuntimeError("accepted continuation dual lacks its fraction")
                expected_previous_sha256 = prior_sha256
                expected_scale = fraction / prior_fraction if scale_enabled else 1.0
                expected_initial_sha256 = sha256_array(expected_scale * prior_dual)
            else:
                expected_previous_sha256 = None
                expected_initial_sha256 = None
                expected_scale = None
            expected_initialization_mode = (
                "zero_dual"
                if expected_initial_sha256 is None
                else "external_warm_start"
                if external
                else "accepted_stage_warm_start"
            )
            result_sha256 = record.get("result_dual_sha256")
            warm_previous_stage_attempt = bool(
                prior_dual is not None and not zero_dual_attempt and not external
            )
            direct_trial = bool(direct_enabled and absolute_position == 1)
            stage_target = (
                float(self.identity["target_projection"])
                if fraction == 1.0
                else float(self.identity["baseline_projection"])
                + fraction
                * (
                    float(self.identity["target_projection"])
                    - float(self.identity["baseline_projection"])
                )
            )
            zero_identity = _zero_dual_solve_identity_sha256(
                continuation_context_sha256=self.identity_sha256,
                fraction=fraction,
                target_projection=stage_target,
            )
            prior_zero_rejection = rejected_zero_dual_by_identity.get(zero_identity)
            zero_dual_triggered = bool(
                record["accepted_for_continuation"] is False
                and zero_dual_fallback_enabled
                and warm_previous_stage_attempt
                and not direct_trial
                and prior_zero_rejection is None
                and fraction not in zero_dual_retry_fractions
            )
            zero_dual_skipped = bool(
                record["accepted_for_continuation"] is False
                and zero_dual_fallback_enabled
                and warm_previous_stage_attempt
                and not direct_trial
                and prior_zero_rejection is not None
            )
            expected_prior_rejection = (
                None
                if not zero_dual_skipped
                else {
                    "stage": prior_zero_rejection["stage"],
                    "zero_dual_solve_identity_sha256": zero_identity,
                    "result_dual_sha256": prior_zero_rejection["result_dual_sha256"],
                    "stage_record_sha256": sha256_json(prior_zero_rejection),
                }
            )
            if (
                record.get("stage") != absolute_position
                or record.get("fraction_in_original_requested_schedule")
                is not (fraction in requested)
                or record.get("scaled_warm_start_enabled") is not scale_enabled
                or record.get("warm_started_from_external_exact_target_checkpoint")
                is not external
                or record.get("warm_started_from_previous_stage")
                is not (expected_initial_sha256 is not None)
                or record.get("last_accepted_fraction_before_attempt")
                != (0.0 if prior_fraction is None else prior_fraction)
                or record.get("warm_start_scale") != expected_scale
                or record.get("previous_result_dual_sha256") != expected_previous_sha256
                or record.get("initial_dual_sha256") != expected_initial_sha256
                or record.get("initialization_mode") != expected_initialization_mode
                or record.get("candidate_zero_dual_solve_identity_sha256")
                != zero_identity
                or record.get("zero_dual_solve_identity_sha256")
                != (zero_identity if expected_initial_sha256 is None else None)
                or record.get("direct_exact_target_first_enabled") is not direct_enabled
                or record.get("direct_exact_target_trial") is not direct_trial
                or record.get("attempt_role")
                != (
                    "direct_exact_target_zero_trial"
                    if direct_trial
                    else "continuation_stage"
                )
                or record.get("target_projection") != stage_target
                or not isinstance(result_sha256, str)
                or len(result_sha256) != 64
                or any(
                    character not in "0123456789abcdef" for character in result_sha256
                )
                or record.get("rejected_iterate_used_as_warm_start") is not False
                or record.get("zero_dual_restart_fallback_enabled")
                is not zero_dual_fallback_enabled
                or record.get("zero_dual_restart_attempt") is not zero_dual_attempt
                or record.get("zero_dual_restart_triggered") is not zero_dual_triggered
                or record.get("zero_dual_retry_fraction")
                != (fraction if zero_dual_triggered else None)
                or record.get("zero_dual_restart_skipped_prior_rejection")
                is not zero_dual_skipped
                or record.get("prior_rejected_zero_dual_attempt")
                != expected_prior_rejection
            ):
                raise ValueError(
                    "continuation checkpoint warm-start linkage is invalid at "
                    f"attempt {absolute_position}: {current['file']}"
                )
            if record["accepted_for_continuation"] is True:
                current_sha256 = sha256_array(current["dual_variables"])
                if result_sha256 != current_sha256:
                    raise ValueError(
                        "continuation checkpoint accepted result hash is invalid: "
                        f"{current['file']}"
                    )
                prior_dual = current["dual_variables"]
                prior_fraction = fraction
                prior_sha256 = current_sha256
                pending_zero_dual_fraction = None
            elif zero_dual_triggered:
                pending_zero_dual_fraction = fraction
                zero_dual_retry_fractions.append(fraction)
            else:
                pending_zero_dual_fraction = None
            if (
                record.get("initialization_mode") == "zero_dual"
                and record["accepted_for_continuation"] is False
            ):
                if zero_identity in rejected_zero_dual_by_identity:
                    raise ValueError(
                        "continuation journal repeats a rejected zero-dual solve: "
                        f"{current['file']}"
                    )
                rejected_zero_dual_by_identity[zero_identity] = record
        if document["scheduler"].get("pending_zero_dual_retry_fraction") != (
            pending_zero_dual_fraction
        ):
            raise ValueError(
                "continuation journal pending zero-dual-retry state is invalid: "
                f"{current['file']}"
            )
        if document["scheduler"].get("zero_dual_retry_fractions") != (
            zero_dual_retry_fractions
        ):
            raise ValueError(
                "continuation journal zero-dual retry history is invalid: "
                f"{current['file']}"
            )
        if document["scheduler"].get("rejected_zero_dual_attempts") != (
            _rejected_zero_dual_ledger(records)
        ):
            raise ValueError(
                "continuation journal rejected-zero-dual ledger is invalid: "
                f"{current['file']}"
            )
        self._validate_scheduler_replay(document, file_name=current["file"])

    def _validate_scheduler_replay(
        self,
        document: dict[str, Any],
        *,
        file_name: str,
    ) -> None:
        """Rebuild every retry and midpoint from the requested schedule."""

        requested = [float(value) for value in self.identity["requested_fractions"]]
        solver_settings = self.identity["case_identity"]["solver_settings"]
        adaptive_enabled = solver_settings.get("adaptive_continuation", False)
        direct_enabled = solver_settings.get("direct_exact_target_first", False)
        minimum_step = float(
            solver_settings.get(
                "adaptive_minimum_fraction_step",
                DEFAULT_ADAPTIVE_MINIMUM_FRACTION_STEP,
            )
        )
        maximum_subdivisions = solver_settings.get(
            "maximum_adaptive_subdivisions",
            DEFAULT_MAXIMUM_ADAPTIVE_SUBDIVISIONS,
        )
        if (
            not isinstance(adaptive_enabled, bool)
            or not isinstance(direct_enabled, bool)
            or not math.isfinite(minimum_step)
            or minimum_step <= 0.0
            or minimum_step > 1.0
            or isinstance(maximum_subdivisions, bool)
            or not isinstance(maximum_subdivisions, int)
            or maximum_subdivisions <= 0
        ):
            raise ValueError(f"continuation scheduler policy is invalid: {file_name}")
        records = document["stage_records"]
        planned = [1.0] if direct_enabled else list(requested)
        next_position = 0
        subdivision_count = 0
        pending_zero_dual: float | None = None
        retry_fractions: list[float] = []
        direct_status = "pending" if direct_enabled else "not_requested"
        start = 0
        if direct_enabled:
            if not records:
                raise ValueError(
                    f"direct-first continuation journal has no trial: {file_name}"
                )
            direct = records[0]
            expected_eligibility = {
                "all_result_metrics_finite": direct["intermediate_continuation_gate"][
                    "checks"
                ]["all_result_metrics_finite"],
                "all_result_arrays_finite": direct["intermediate_continuation_gate"][
                    "checks"
                ]["all_result_arrays_finite"],
            }
            expected_eligibility["eligible"] = bool(
                expected_eligibility["all_result_metrics_finite"]
                and expected_eligibility["all_result_arrays_finite"]
            )
            if (
                direct.get("attempt_role") != "direct_exact_target_zero_trial"
                or direct.get("direct_exact_target_trial") is not True
                or float(direct["fraction"]) != 1.0
                or direct.get("initialization_mode") != "zero_dual"
                or direct.get("zero_dual_restart_attempt") is not False
                or direct.get("zero_dual_restart_triggered") is not False
                or direct.get("zero_dual_restart_skipped_prior_rejection") is not False
                or direct.get("adaptive_subdivision_triggered") is not False
                or direct.get("adaptive_retry_fraction") is not None
                or direct.get("direct_fallback_eligibility") != expected_eligibility
            ):
                raise ValueError(
                    f"direct-first continuation trial is invalid: {file_name}"
                )
            if direct["accepted_for_continuation"] is True:
                if (
                    direct.get("direct_fallback_triggered") is not False
                    or direct["publication_gates"].get("all_passed") is not True
                    or len(records) != 1
                ):
                    raise ValueError(
                        f"accepted direct-first trial is invalid: {file_name}"
                    )
                direct_status = "accepted"
                next_position = 1
                start = 1
            else:
                if (
                    direct.get("direct_fallback_triggered") is not True
                    or direct["publication_gates"].get("all_passed") is not False
                    or expected_eligibility["eligible"] is not True
                ):
                    raise ValueError(
                        f"direct-first fallback transition is invalid: {file_name}"
                    )
                direct_status = "rejected"
                planned = list(requested)
                next_position = 0
                start = 1
        for record in records[start:]:
            if (
                record.get("attempt_role") != "continuation_stage"
                or record.get("direct_exact_target_trial") is not False
                or record.get("direct_fallback_triggered") is not False
                or record.get("direct_fallback_eligibility") is not None
            ):
                raise ValueError(
                    "continuation scheduler mixes direct and fallback roles: "
                    f"{file_name}"
                )
            if next_position >= len(planned):
                raise ValueError(
                    f"continuation scheduler has attempts after completion: {file_name}"
                )
            fraction = float(record["fraction"])
            expected_fraction = (
                planned[next_position]
                if pending_zero_dual is None
                else pending_zero_dual
            )
            if fraction != expected_fraction:
                raise ValueError(
                    f"continuation scheduler attempt order is invalid: {file_name}"
                )
            if (
                record.get("adaptive_continuation_enabled") is not adaptive_enabled
                or record.get("adaptive_subdivision_count_before_attempt")
                != subdivision_count
            ):
                raise ValueError(
                    "continuation scheduler attempt policy record is invalid: "
                    f"{file_name}"
                )
            if record["accepted_for_continuation"] is True:
                if (
                    record.get("adaptive_subdivision_triggered") is not False
                    or record.get("adaptive_retry_fraction") is not None
                ):
                    raise ValueError(
                        "accepted continuation stage mutates the retry schedule: "
                        f"{file_name}"
                    )
                next_position += 1
                pending_zero_dual = None
                continue
            if record.get("zero_dual_restart_triggered") is True:
                if (
                    pending_zero_dual is not None
                    or fraction in retry_fractions
                    or record.get("adaptive_subdivision_triggered") is not False
                    or record.get("adaptive_retry_fraction") is not None
                ):
                    raise ValueError(
                        "continuation scheduler has an invalid zero-dual retry: "
                        f"{file_name}"
                    )
                pending_zero_dual = fraction
                retry_fractions.append(fraction)
                continue
            pending_zero_dual = None
            if record.get("adaptive_subdivision_triggered") is not True:
                raise ValueError(
                    "continuation journal persists a terminal rejected attempt: "
                    f"{file_name}"
                )
            lower_fraction = 0.0 if next_position == 0 else planned[next_position - 1]
            half_step = 0.5 * (fraction - lower_fraction)
            midpoint = lower_fraction + half_step
            if (
                not adaptive_enabled
                or subdivision_count >= maximum_subdivisions
                or half_step + 8.0 * np.finfo(np.float64).eps < minimum_step
                or not lower_fraction < midpoint < fraction
                or record.get("adaptive_retry_fraction") != midpoint
            ):
                raise ValueError(
                    f"continuation scheduler midpoint is invalid: {file_name}"
                )
            planned.insert(next_position, midpoint)
            subdivision_count += 1

        scheduler = document["scheduler"]
        if (
            scheduler.get("planned_fractions") != planned
            or scheduler.get("next_position") != next_position
            or scheduler.get("adaptive_subdivision_count") != subdivision_count
            or scheduler.get("pending_zero_dual_retry_fraction") != pending_zero_dual
            or scheduler.get("zero_dual_retry_fractions") != retry_fractions
            or scheduler.get("direct_exact_target_status") != direct_status
            or scheduler.get("rejected_zero_dual_attempts")
            != _rejected_zero_dual_ledger(records)
        ):
            raise ValueError(
                f"continuation scheduler replay does not match journal: {file_name}"
            )

    def load(self) -> dict[str, Any] | None:
        """Validate the full journal and return its deterministic resume state."""

        paths = sorted(self.directory.glob("accepted-*.npz"))
        pending_exists = self.pending_path.exists()
        if not paths and not pending_exists:
            self._checkpoint_count = 0
            self._latest_record = None
            self._pending_record = None
            self.resumed_attempt_count = 0
            return None
        if not self.resume_enabled:
            raise ValueError(
                "a continuation journal already exists but resume is "
                "disabled; use --resume-continuation or --overwrite"
            )
        previous: dict[str, Any] | None = None
        accepted_checkpoints: list[dict[str, Any]] = []
        for sequence, path in enumerate(paths, start=1):
            if path.name != f"accepted-{sequence:04d}.npz":
                raise ValueError(
                    f"continuation checkpoint filenames are not contiguous: {path}"
                )
            current = self._load_one(path, expected_sequence=sequence)
            if previous is not None:
                before = previous["document"]
                after = current["document"]
                before_records = before["stage_records"]
                after_records = after["stage_records"]
                if (
                    after_records[: len(before_records)] != before_records
                    or len(after_records) <= len(before_records)
                    or after["scheduler"]["next_position"]
                    <= before["scheduler"]["next_position"]
                    or float(after["accepted_fraction"])
                    <= float(before["accepted_fraction"])
                    or after.get("previous_checkpoint_sha256") != previous["sha256"]
                ):
                    raise ValueError(
                        f"continuation checkpoint history is not append-only: {path}"
                    )
            elif current["document"].get("previous_checkpoint_sha256") is not None:
                raise ValueError(
                    f"first continuation checkpoint has a predecessor: {path}"
                )
            self._validate_attempt_linkage(current, previous)
            previous = current
            accepted_checkpoints.append(current)
        self._checkpoint_count = len(paths)
        self._latest_record = previous
        self._pending_record = None
        self.resumed_this_invocation = True
        resume_record = previous
        if pending_exists:
            try:
                pending_header = load_json(self.pending_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "cannot validate pending continuation transition: "
                    f"{self.pending_path}"
                ) from error
            base_sequence = pending_header.get("base_accepted_checkpoint_sequence")
            if (
                isinstance(base_sequence, bool)
                or not isinstance(base_sequence, int)
                or not 0 <= base_sequence <= len(accepted_checkpoints)
            ):
                raise ValueError(
                    "pending continuation transition has an impossible accepted "
                    f"base: {self.pending_path}"
                )
            base = (
                None if base_sequence == 0 else accepted_checkpoints[base_sequence - 1]
            )
            pending = self._load_pending_transition(base)
            if base_sequence == len(accepted_checkpoints):
                self._pending_record = pending
                resume_record = pending
            else:
                # A process may be killed after the accepted NPZ is atomically
                # linked but before the obsolete pending JSON is unlinked.  It
                # is safe to ignore only when the accepted chain demonstrably
                # incorporates the entire pending history.
                latest_records = previous["document"]["stage_records"]
                pending_records = pending["document"]["stage_records"]
                if latest_records[: len(pending_records)] != pending_records:
                    raise ValueError(
                        "stale pending continuation transition is not incorporated "
                        f"by the accepted checkpoint chain: {self.pending_path}"
                    )
        if resume_record is None:  # pragma: no cover - guarded above
            raise RuntimeError("continuation journal has no resume record")
        document = resume_record["document"]
        scheduler = document["scheduler"]
        self.resumed_attempt_count = len(document["stage_records"])
        return {
            **resume_record,
            "stage_records": list(document["stage_records"]),
            "planned_fractions": [
                float(value) for value in scheduler["planned_fractions"]
            ],
            "next_position": int(scheduler["next_position"]),
            "adaptive_subdivision_count": int(scheduler["adaptive_subdivision_count"]),
            "pending_zero_dual_retry_fraction": scheduler[
                "pending_zero_dual_retry_fraction"
            ],
            "direct_exact_target_status": scheduler["direct_exact_target_status"],
            "previous_fraction": (
                None
                if self._latest_record is None
                else float(self._latest_record["document"]["accepted_fraction"])
            ),
        }

    def commit_pending_transition(
        self,
        *,
        stage_records: Sequence[dict[str, Any]],
        planned_fractions: Sequence[float],
        next_position: int,
        adaptive_subdivision_count: int,
        pending_zero_dual_retry_fraction: float | None,
        direct_exact_target_status: str = "not_requested",
    ) -> dict[str, Any]:
        """Atomically persist retry scheduling, never a rejected solver array."""

        if (
            not stage_records
            or stage_records[-1].get("accepted_for_continuation") is not False
        ):
            raise ValueError("pending transition must end at a rejected stage")
        scheduler = {
            "planned_fractions": [float(value) for value in planned_fractions],
            "next_position": int(next_position),
            "adaptive_subdivision_count": int(adaptive_subdivision_count),
            "pending_zero_dual_retry_fraction": (
                None
                if pending_zero_dual_retry_fraction is None
                else float(pending_zero_dual_retry_fraction)
            ),
            "zero_dual_retry_fractions": [
                float(record["zero_dual_retry_fraction"])
                for record in stage_records
                if record.get("zero_dual_restart_triggered") is True
            ],
            "rejected_zero_dual_attempts": _rejected_zero_dual_ledger(stage_records),
            "direct_exact_target_status": direct_exact_target_status,
        }
        records = list(stage_records)
        document = {
            "schema_version": CONTINUATION_JOURNAL_SCHEMA_VERSION,
            "status": "pending_scheduler_transition_nonpublishable",
            "scientific_result_published": False,
            "canonical_result_or_report_created": False,
            "journal_identity": self.identity,
            "journal_identity_sha256": self.identity_sha256,
            "base_accepted_checkpoint_sequence": self._checkpoint_count,
            "base_accepted_checkpoint_sha256": (
                None if self._latest_record is None else self._latest_record["sha256"]
            ),
            "stage_records": records,
            "stage_records_sha256": sha256_json({"stage_records": records}),
            "scheduler": scheduler,
            "scheduler_sha256": sha256_json({"scheduler": scheduler}),
            "committed_utc": datetime.now(UTC).isoformat(),
        }
        current = {
            "document": document,
            "dual_variables": (
                None
                if self._latest_record is None
                else self._latest_record["dual_variables"]
            ),
            "final_result": None,
            "file": self.pending_path.name,
        }
        self._validate_attempt_linkage(
            current,
            self._latest_record,
            expected_new_acceptances=0,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.pending_path, document, overwrite=True)
        current.update(
            sha256=sha256_file(self.pending_path),
            size_bytes=self.pending_path.stat().st_size,
        )
        self._pending_record = current
        return current

    def commit(
        self,
        *,
        result: NativeScalarSQPResult,
        stage_records: Sequence[dict[str, Any]],
        planned_fractions: Sequence[float],
        next_position: int,
        adaptive_subdivision_count: int,
        direct_exact_target_status: str = "not_requested",
    ) -> dict[str, Any]:
        """Atomically commit one accepted state; never accept a rejected result."""

        if (
            not stage_records
            or stage_records[-1].get("accepted_for_continuation") is not True
        ):
            raise ValueError("only an accepted continuation stage may be checkpointed")
        fraction = float(stage_records[-1]["fraction"])
        final = next_position == len(planned_fractions)
        last_record = stage_records[-1]
        solver_settings = self.identity["case_identity"]["solver_settings"]
        publication_gates = _solver_publication_gates(
            result, solver_settings=solver_settings
        )
        if last_record.get("publication_gates") != publication_gates:
            raise ValueError("accepted continuation gate record does not match result")
        intermediate_gates = last_record.get("intermediate_continuation_gate")
        if final and not publication_gates["all_passed"]:
            raise ValueError(
                "final continuation checkpoint must pass publication gates"
            )
        if (
            not final
            and not publication_gates["all_passed"]
            and (
                not isinstance(intermediate_gates, dict)
                or intermediate_gates.get("all_passed") is not True
            )
        ):
            raise ValueError("intermediate continuation checkpoint failed its gates")
        dual = np.asarray(result.dual_variables, dtype=np.float64)
        arrays: dict[str, np.ndarray] = {"dual_variables": dual}
        if final:
            arrays.update(
                native_interventions=np.asarray(
                    result.native_interventions, dtype=np.float64
                ),
                native_constraint_gradient=np.asarray(
                    result.native_constraint_gradient, dtype=np.float64
                ),
            )
        if any(
            value.ndim != 2 or not np.isfinite(value).all() for value in arrays.values()
        ):
            raise ValueError(
                "accepted continuation result arrays must be finite matrices"
            )
        if dual.shape[0] != CONTROL_STEPS:
            raise ValueError("accepted continuation dual has the wrong control count")
        if stage_records[-1].get("result_dual_sha256") != sha256_array(dual):
            raise ValueError("accepted continuation stage does not identify its dual")
        sequence = self._checkpoint_count + 1
        if next_position != sequence:
            raise ValueError(
                "accepted continuation checkpoint sequence does not match scheduler"
            )
        document = {
            "schema_version": CONTINUATION_JOURNAL_SCHEMA_VERSION,
            "status": "accepted_stage_nonpublishable",
            "scientific_result_published": False,
            "canonical_result_or_report_created": False,
            "journal_identity": self.identity,
            "journal_identity_sha256": self.identity_sha256,
            "sequence": sequence,
            "previous_checkpoint_sha256": (
                None if self._latest_record is None else self._latest_record["sha256"]
            ),
            "accepted_fraction": fraction,
            "accepted_dual_sha256": sha256_array(dual),
            "arrays": {
                name: self._array_record(value) for name, value in arrays.items()
            },
            "accepted_result": _continuation_result_metadata(result) if final else None,
            "stage_records": list(stage_records),
            "scheduler": {
                "planned_fractions": [float(value) for value in planned_fractions],
                "next_position": int(next_position),
                "adaptive_subdivision_count": int(adaptive_subdivision_count),
                "pending_zero_dual_retry_fraction": None,
                "zero_dual_retry_fractions": [
                    float(record["zero_dual_retry_fraction"])
                    for record in stage_records
                    if record.get("zero_dual_restart_triggered") is True
                ],
                "rejected_zero_dual_attempts": _rejected_zero_dual_ledger(
                    stage_records
                ),
                "direct_exact_target_status": direct_exact_target_status,
            },
            "committed_utc": datetime.now(UTC).isoformat(),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"accepted-{sequence:04d}.npz"
        self._validate_attempt_linkage(
            {
                "document": document,
                "dual_variables": dual,
                "file": path.name,
            },
            self._latest_record,
        )
        write_npz(
            path,
            overwrite=False,
            document_utf8=self._document_bytes(document),
            **arrays,
        )
        record = {
            "document": document,
            "dual_variables": dual.copy(),
            "final_result": result if final else None,
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        self._checkpoint_count = sequence
        self._latest_record = record
        self._pending_record = None
        if self.pending_path.exists():
            self.pending_path.unlink()
        return record

    def audit_record(self) -> dict[str, Any] | None:
        """Return provenance for the newest non-publishable accepted checkpoint."""

        if self._latest_record is None and self._pending_record is None:
            return None
        active = self._pending_record or self._latest_record
        document = active["document"]
        return {
            "status": document["status"],
            "scientific_result_published": False,
            "journal_identity_sha256": self.identity_sha256,
            "accepted_checkpoint_count": self._checkpoint_count,
            "latest_accepted_fraction": (
                None
                if self._latest_record is None
                else self._latest_record["document"]["accepted_fraction"]
            ),
            "pending_retry_active": self._pending_record is not None,
            "direct_exact_target_status": document["scheduler"][
                "direct_exact_target_status"
            ],
            "rejected_zero_dual_attempt_count": len(
                document["scheduler"]["rejected_zero_dual_attempts"]
            ),
            "latest_file": active["file"],
            "latest_sha256": active["sha256"],
            "latest_size_bytes": active["size_bytes"],
            "resumed_this_invocation": self.resumed_this_invocation,
            "resumed_attempt_count": self.resumed_attempt_count,
        }


def _projection_continuation_acceptance_policy() -> dict[str, Any]:
    """Describe the fail-closed distinction between warm starts and publication."""

    return {
        "intermediate_stage_scope": "fractions_strictly_below_1_only",
        "intermediate_stage_purpose": "warm_start_only_not_scientific_publication",
        "permitted_original_gate_failures": [
            "solver_declared_success",
            "solver_constraint_is_literally_nonnegative",
            "absolute_primal_violation_within_configured_tolerance",
            "stationarity_within_configured_tolerance",
        ],
        "requires_finite_result_metrics_and_arrays": True,
        "constraint_residual_is_two_sided_at_intermediate_stages": True,
        "constraint_tolerance_multiplier": (
            INTERMEDIATE_CONTINUATION_CONSTRAINT_MULTIPLIER
        ),
        "requires_absolute_constraint_residual_and_primal_violation_within_"
        "relaxed_tolerance": True,
        "requires_complementarity_within_configured_tolerance": True,
        "stationarity_tolerance_multiplier": (
            INTERMEDIATE_CONTINUATION_STATIONARITY_MULTIPLIER
        ),
        "stationarity_metric": "relative_when_configured_absolute_otherwise",
        "final_stage_policy": "original_publication_gates_unchanged",
        "direct_exact_target_first_policy": (
            "when enabled, first attempt the exact target from zero dual and "
            "publish immediately only if every unchanged final gate passes; "
            "a finite strict-gate rejection activates the configured adaptive "
            "continuation schedule, while exceptions or nonfinite results fail closed"
        ),
        "adaptive_subdivision_policy": (
            "after the one permitted zero-dual retry, bisect only the interval "
            "after the last accepted fraction; discard every rejected iterate "
            "and retry from the last accepted dual"
        ),
        "zero_dual_restart_policy": (
            "when enabled, a target rejected from a previous-stage warm start "
            "is attempted at most once at that fraction with no initial dual "
            "before any adaptive midpoint is inserted; an authenticated prior "
            "zero-dual rejection at the identical target is not recomputed"
        ),
    }


def _solve_projection_continuation(
    oracle: ReleaseConstraintOracle,
    covariance: Any,
    *,
    covariance_case: str,
    baseline_projection: float,
    target_projection: float,
    fractions: Sequence[float],
    solver_settings: dict[str, Any],
    initial_dual_variables: np.ndarray | None = None,
    initial_dual_scale: float = 1.0,
    initial_dual_raw_sha256: str | None = None,
    continuation_journal: DurableContinuationJournal | None = None,
) -> tuple[NativeScalarSQPResult, list[dict[str, Any]]]:
    """Solve nested projection targets and return only the exact-target iterate.

    The oracle stores the final constraint ``p(a) - p_target``.  An intermediate
    target is therefore implemented by adding a scalar offset to its value;
    the exact adjoint gradient and the oracle's replay caches are unchanged.
    The exact-target final stage must satisfy the unchanged publication gates.
    A nonfinal stage that narrowly misses solver-declared success, strict
    feasibility, or strict stationarity may still provide a warm start when it
    passes the separately recorded, fail-closed intermediate gate.  No
    intermediate scientific result artifact is published.  If adaptive
    continuation is enabled, a rejected stage causes the interval after the
    last accepted fraction to be bisected.  Earlier accepted stages are not
    recomputed, and the rejected dual is never reused.  The exact-target final
    stage still has to pass the unchanged publication gates.
    """

    schedule = tuple(float(value) for value in fractions)
    if (
        not schedule
        or any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in schedule
        )
        or any(
            right <= left for left, right in zip(schedule, schedule[1:], strict=False)
        )
        or schedule[-1] != 1.0
    ):
        raise ValueError(
            "continuation fractions must be strictly increasing in (0, 1] "
            "and end at exactly 1.0"
        )
    baseline = float(baseline_projection)
    target = float(target_projection)
    if not math.isfinite(baseline) or not math.isfinite(target):
        raise ValueError("continuation projections must be finite")

    full_gap = target - baseline
    # If the unnudged release already satisfies the event-oriented inequality,
    # the origin is the exact global minimum: its covariance action is zero.
    # Let the unchanged SQP/KKT machinery certify and publish that solution
    # instead of treating a nonpositive requested gap as an input error.
    baseline_already_feasible = full_gap <= 0.0
    scale_warm_start = solver_settings.get("scale_continuation_warm_start", False)
    if not isinstance(scale_warm_start, bool):
        raise ValueError("scale_continuation_warm_start must be a bool")
    adaptive_continuation = solver_settings.get("adaptive_continuation", False)
    if not isinstance(adaptive_continuation, bool):
        raise ValueError("adaptive_continuation must be a bool")
    zero_dual_restart = solver_settings.get(
        "zero_dual_restart_after_warm_rejection", False
    )
    if not isinstance(zero_dual_restart, bool):
        raise ValueError("zero_dual_restart_after_warm_rejection must be a bool")
    direct_exact_target_first = solver_settings.get("direct_exact_target_first", False)
    if not isinstance(direct_exact_target_first, bool):
        raise ValueError("direct_exact_target_first must be a bool")
    if direct_exact_target_first and schedule[0] == 1.0:
        raise ValueError(
            "direct_exact_target_first requires a continuation fallback schedule "
            "with at least one fraction below 1"
        )
    adaptive_minimum_step = float(
        solver_settings.get(
            "adaptive_minimum_fraction_step",
            DEFAULT_ADAPTIVE_MINIMUM_FRACTION_STEP,
        )
    )
    if (
        not math.isfinite(adaptive_minimum_step)
        or adaptive_minimum_step <= 0.0
        or adaptive_minimum_step > 1.0
    ):
        raise ValueError("adaptive_minimum_fraction_step must be finite and in (0, 1]")
    maximum_adaptive_subdivisions = solver_settings.get(
        "maximum_adaptive_subdivisions",
        DEFAULT_MAXIMUM_ADAPTIVE_SUBDIVISIONS,
    )
    if (
        isinstance(maximum_adaptive_subdivisions, bool)
        or not isinstance(maximum_adaptive_subdivisions, int)
        or maximum_adaptive_subdivisions <= 0
    ):
        raise ValueError("maximum_adaptive_subdivisions must be a positive integer")
    external_initial_dual = (
        None
        if initial_dual_variables is None
        else np.asarray(initial_dual_variables, dtype=np.float64).copy()
    )
    external_initial_dual_scale = float(initial_dual_scale)
    if (
        not math.isfinite(external_initial_dual_scale)
        or external_initial_dual_scale <= 0.0
    ):
        raise ValueError("initial_dual_scale must be finite and positive")
    if external_initial_dual is None:
        if initial_dual_raw_sha256 is not None or external_initial_dual_scale != 1.0:
            raise ValueError(
                "external initial-dual provenance requires initial_dual_variables"
            )
    else:
        if initial_dual_raw_sha256 is None:
            if external_initial_dual_scale != 1.0:
                raise ValueError(
                    "a scaled external initial dual requires its raw SHA-256"
                )
            initial_dual_raw_sha256 = sha256_array(external_initial_dual)
        if (
            len(initial_dual_raw_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in initial_dual_raw_sha256
            )
        ):
            raise ValueError("initial_dual_raw_sha256 must be a lowercase SHA-256")
    if direct_exact_target_first and external_initial_dual is not None:
        raise ValueError(
            "direct_exact_target_first requires the canonical zero-dual start; "
            "an external initial dual is incompatible"
        )
    if external_initial_dual is not None and len(schedule) != 1:
        raise ValueError(
            "an external exact-target dual warm start requires a one-stage "
            "continuation schedule"
        )
    previous_dual: np.ndarray | None = None
    previous_fraction: float | None = None
    stage_records: list[dict[str, Any]] = []
    final_result: NativeScalarSQPResult | None = None
    planned_fractions = [1.0] if direct_exact_target_first else list(schedule)
    next_position = 0
    adaptive_subdivision_count = 0
    pending_zero_dual_retry_fraction: float | None = None
    zero_dual_retry_fractions: set[float] = set()
    direct_exact_target_status = (
        "pending" if direct_exact_target_first else "not_requested"
    )
    rejected_zero_dual_by_identity: dict[str, dict[str, Any]] = {}
    resumed_checkpoint: dict[str, Any] | None = None
    if continuation_journal is not None:
        resumed_checkpoint = continuation_journal.load()
    if resumed_checkpoint is not None:
        checkpoint_dual = resumed_checkpoint["dual_variables"]
        previous_dual = None if checkpoint_dual is None else checkpoint_dual.copy()
        previous_fraction = resumed_checkpoint["previous_fraction"]
        stage_records = list(resumed_checkpoint["stage_records"])
        planned_fractions = list(resumed_checkpoint["planned_fractions"])
        next_position = int(resumed_checkpoint["next_position"])
        adaptive_subdivision_count = int(
            resumed_checkpoint["adaptive_subdivision_count"]
        )
        pending_zero_dual_retry_fraction = resumed_checkpoint[
            "pending_zero_dual_retry_fraction"
        ]
        direct_exact_target_status = resumed_checkpoint["direct_exact_target_status"]
        zero_dual_retry_fractions = {
            float(record["zero_dual_retry_fraction"])
            for record in stage_records
            if record.get("zero_dual_restart_triggered") is True
        }
        rejected_zero_dual_by_identity = {
            str(record["zero_dual_solve_identity_sha256"]): record
            for record in stage_records
            if record.get("initialization_mode") == "zero_dual"
            and record.get("accepted_for_continuation") is False
        }
        if next_position == len(planned_fractions):
            final_result = resumed_checkpoint["final_result"]
            if final_result is None:  # pragma: no cover - validated by journal
                raise RuntimeError("final continuation checkpoint has no result")
            return final_result, stage_records
    continuation_context_sha256 = (
        continuation_journal.identity_sha256
        if continuation_journal is not None
        else sha256_json(
            {
                "schema_version": 1,
                "covariance_case": covariance_case,
                "baseline_projection_hex": baseline.hex(),
                "target_projection_hex": target.hex(),
                "requested_fraction_hex": [value.hex() for value in schedule],
                "solver_settings": solver_settings,
            }
        )
    )
    while next_position < len(planned_fractions):
        fraction = planned_fractions[next_position]
        position = len(stage_records) + 1
        direct_exact_target_trial = direct_exact_target_status == "pending"
        if direct_exact_target_trial and fraction != 1.0:
            raise RuntimeError("direct exact-target trial is not fraction 1")
        stage_target = target if fraction == 1.0 else baseline + fraction * full_gap
        value_offset = target - stage_target
        if value_offset == 0.0:
            stage_value = oracle.value
            stage_value_gradient = oracle.value_gradient
        else:

            def stage_value(
                native: np.ndarray, *, _offset: float = value_offset
            ) -> float:
                return oracle.value(native) + _offset

            def stage_value_gradient(
                native: np.ndarray, *, _offset: float = value_offset
            ) -> tuple[float, np.ndarray]:
                value, gradient = oracle.value_gradient(native)
                return value + _offset, gradient

        initial_dual: np.ndarray | None = None
        warm_start_scale: float | None = None
        previous_dual_sha256: str | None = None
        externally_warm_started = position == 1 and external_initial_dual is not None
        zero_dual_restart_attempt = pending_zero_dual_retry_fraction == fraction
        if zero_dual_restart_attempt:
            previous_dual_sha256 = (
                None if previous_dual is None else sha256_array(previous_dual)
            )
        elif externally_warm_started:
            initial_dual = external_initial_dual
            warm_start_scale = external_initial_dual_scale
            previous_dual_sha256 = initial_dual_raw_sha256
        elif previous_dual is not None:
            if previous_fraction is None:  # pragma: no cover - internal invariant
                raise RuntimeError("continuation warm start lacks a prior fraction")
            warm_start_scale = fraction / previous_fraction if scale_warm_start else 1.0
            previous_dual_sha256 = sha256_array(previous_dual)
            initial_dual = warm_start_scale * previous_dual
        initialization_mode = (
            "zero_dual"
            if initial_dual is None
            else "external_warm_start"
            if externally_warm_started
            else "accepted_stage_warm_start"
        )
        candidate_zero_dual_identity = _zero_dual_solve_identity_sha256(
            continuation_context_sha256=continuation_context_sha256,
            fraction=fraction,
            target_projection=stage_target,
        )
        if (
            initialization_mode == "zero_dual"
            and candidate_zero_dual_identity in rejected_zero_dual_by_identity
        ):
            raise RuntimeError(
                "scheduler attempted to repeat an authenticated rejected "
                "zero-dual solve"
            )

        counters_before = (
            oracle.forward_replays,
            oracle.reverse_sweeps,
            oracle.reverse_wall_seconds,
        )
        stage_started = time.monotonic()
        result = solve_minimum_native_covariance_action_sqp(
            stage_value,
            stage_value_gradient,
            covariance,
            control_steps=CONTROL_STEPS,
            initial_dual_variables=initial_dual,
            known_active_boundary=solver_settings["known_active_boundary"],
            maximum_iterations=int(solver_settings["maximum_iterations"]),
            constraint_tolerance=float(solver_settings["constraint_tolerance"]),
            stationarity_tolerance=float(solver_settings["stationarity_tolerance"]),
            complementarity_tolerance=float(
                solver_settings["complementarity_tolerance"]
            ),
            relative_stationarity_tolerance=solver_settings[
                "relative_stationarity_tolerance"
            ],
            relative_complementarity_tolerance=solver_settings[
                "relative_complementarity_tolerance"
            ],
            initial_trust_radius=float(solver_settings["initial_trust_radius"]),
            maximum_trust_radius=float(solver_settings["maximum_trust_radius"]),
            covariance_block_rows=int(solver_settings["covariance_block_rows"]),
        )
        gates = _solver_publication_gates(result, solver_settings=solver_settings)
        intermediate_gate = _intermediate_continuation_gate(
            result,
            solver_settings=solver_settings,
            publication_gates=gates,
        )
        direct_fallback_eligibility: dict[str, bool] | None = None
        if direct_exact_target_trial:
            direct_fallback_eligibility = {
                "all_result_metrics_finite": bool(
                    intermediate_gate["checks"]["all_result_metrics_finite"]
                ),
                "all_result_arrays_finite": bool(
                    intermediate_gate["checks"]["all_result_arrays_finite"]
                ),
            }
            direct_fallback_eligibility["eligible"] = bool(
                direct_fallback_eligibility["all_result_metrics_finite"]
                and direct_fallback_eligibility["all_result_arrays_finite"]
            )
        is_final_stage = fraction == 1.0
        intermediate_gate_used = bool(
            not is_final_stage
            and not gates["all_passed"]
            and intermediate_gate["all_passed"]
        )
        accepted_for_continuation = bool(
            gates["all_passed"]
            or (not is_final_stage and intermediate_gate["all_passed"])
        )
        if (
            direct_exact_target_trial
            and direct_fallback_eligibility is not None
            and not direct_fallback_eligibility["eligible"]
        ):
            # The unchanged final gates intentionally inspect only publication
            # quantities.  The direct-first protocol additionally fails closed
            # when *any* returned solver metric or array is nonfinite: such a
            # result is neither publishable nor eligible to activate fallback.
            accepted_for_continuation = False
        if accepted_for_continuation and gates["all_passed"]:
            acceptance_basis = "original_publication_gates"
        elif intermediate_gate_used:
            acceptance_basis = "fail_closed_intermediate_warm_start_gate"
        else:
            acceptance_basis = "rejected"
        stage_gap = stage_target - baseline
        stage_record = {
            "stage": position,
            "attempt_role": (
                "direct_exact_target_zero_trial"
                if direct_exact_target_trial
                else "continuation_stage"
            ),
            "fraction": fraction,
            "fraction_in_original_requested_schedule": fraction in schedule,
            "target_projection": stage_target,
            "projection_displacement_from_baseline": stage_gap,
            "baseline_already_feasible_zero_action_case": (baseline_already_feasible),
            "warm_started_from_previous_stage": initial_dual is not None,
            "warm_started_from_external_exact_target_checkpoint": (
                externally_warm_started
            ),
            "scaled_warm_start_enabled": scale_warm_start,
            "warm_start_scale": warm_start_scale,
            "previous_result_dual_sha256": previous_dual_sha256,
            "initial_dual_sha256": (
                None if initial_dual is None else sha256_array(initial_dual)
            ),
            "initialization_mode": initialization_mode,
            "candidate_zero_dual_solve_identity_sha256": (candidate_zero_dual_identity),
            "zero_dual_solve_identity_sha256": (
                candidate_zero_dual_identity
                if initialization_mode == "zero_dual"
                else None
            ),
            "result_dual_sha256": sha256_array(result.dual_variables),
            "publication_gates": gates,
            "intermediate_continuation_gate": intermediate_gate,
            "intermediate_gate_applicable": not is_final_stage,
            "intermediate_gate_used_to_continue": intermediate_gate_used,
            "accepted_for_continuation": accepted_for_continuation,
            "acceptance_basis": acceptance_basis,
            "adaptive_continuation_enabled": adaptive_continuation,
            "last_accepted_fraction_before_attempt": (
                0.0 if previous_fraction is None else previous_fraction
            ),
            "adaptive_subdivision_count_before_attempt": (adaptive_subdivision_count),
            "adaptive_subdivision_triggered": False,
            "adaptive_retry_fraction": None,
            "rejected_iterate_used_as_warm_start": False,
            "zero_dual_restart_fallback_enabled": zero_dual_restart,
            "zero_dual_restart_attempt": zero_dual_restart_attempt,
            "zero_dual_restart_triggered": False,
            "zero_dual_retry_fraction": None,
            "zero_dual_restart_skipped_prior_rejection": False,
            "prior_rejected_zero_dual_attempt": None,
            "direct_exact_target_first_enabled": direct_exact_target_first,
            "direct_exact_target_trial": direct_exact_target_trial,
            "direct_fallback_eligibility": direct_fallback_eligibility,
            "direct_fallback_triggered": False,
            "process_resumed_from_accepted_checkpoint": (
                resumed_checkpoint is not None
            ),
            "resume_checkpoint_sha256": (
                None if resumed_checkpoint is None else resumed_checkpoint["sha256"]
            ),
            "solver": _solver_report(
                result,
                covariance_case=covariance_case,
                initial_projection_gap=stage_gap,
                stationarity_reference_norm=result.stationarity_reference_norm,
                constraint_gradient_covariance_norm=(
                    result.constraint_gradient_covariance_norm
                ),
                reporting_covariance_operator_calls=0,
                covariance_backend=str(
                    solver_settings.get(
                        "covariance_backend", "legacy_centered_sample_factors"
                    )
                ),
                covariance_source_phase_count=int(
                    solver_settings.get(
                        "covariance_source_phase_count", CONTROL_STEPS
                    )
                ),
            ),
            "runtime": {
                "solver_wall_seconds": time.monotonic() - stage_started,
                "release_only_forward_replays": (
                    oracle.forward_replays - counters_before[0]
                ),
                "release_only_reverse_sweeps": (
                    oracle.reverse_sweeps - counters_before[1]
                ),
                "reverse_fortran_wall_seconds": (
                    oracle.reverse_wall_seconds - counters_before[2]
                ),
            },
        }
        stage_records.append(stage_record)
        if not accepted_for_continuation:
            if direct_exact_target_trial:
                if (
                    not direct_fallback_eligibility
                    or not (direct_fallback_eligibility["eligible"])
                ):
                    raise ProjectionContinuationFailure(
                        "refusing direct exact-target fallback because the solver "
                        "returned nonfinite metrics or arrays",
                        result=result,
                        stage_fraction=fraction,
                        stage_records=stage_records,
                    )
                direct_exact_target_status = "rejected"
                planned_fractions = list(schedule)
                next_position = 0
                stage_record["direct_fallback_triggered"] = True
                rejected_zero_dual_by_identity[candidate_zero_dual_identity] = (
                    stage_record
                )
                if continuation_journal is not None:
                    continuation_journal.commit_pending_transition(
                        stage_records=stage_records,
                        planned_fractions=planned_fractions,
                        next_position=next_position,
                        adaptive_subdivision_count=adaptive_subdivision_count,
                        pending_zero_dual_retry_fraction=None,
                        direct_exact_target_status=direct_exact_target_status,
                    )
                continue
            warm_previous_stage_attempt = bool(
                initial_dual is not None
                and previous_dual is not None
                and not externally_warm_started
                and not zero_dual_restart_attempt
            )
            can_retry_from_zero_dual = bool(
                zero_dual_restart
                and warm_previous_stage_attempt
                and candidate_zero_dual_identity not in rejected_zero_dual_by_identity
                and fraction not in zero_dual_retry_fractions
            )
            if can_retry_from_zero_dual:
                pending_zero_dual_retry_fraction = fraction
                zero_dual_retry_fractions.add(fraction)
                stage_record["zero_dual_restart_triggered"] = True
                stage_record["zero_dual_retry_fraction"] = fraction
                if continuation_journal is not None:
                    continuation_journal.commit_pending_transition(
                        stage_records=stage_records,
                        planned_fractions=planned_fractions,
                        next_position=next_position,
                        adaptive_subdivision_count=adaptive_subdivision_count,
                        pending_zero_dual_retry_fraction=(
                            pending_zero_dual_retry_fraction
                        ),
                        direct_exact_target_status=direct_exact_target_status,
                    )
                # Retain the last accepted dual only as provenance.  The exact
                # same target is retried with ``initial_dual_variables=None``.
                continue
            prior_zero_rejection = rejected_zero_dual_by_identity.get(
                candidate_zero_dual_identity
            )
            if (
                zero_dual_restart
                and warm_previous_stage_attempt
                and prior_zero_rejection is not None
            ):
                stage_record["zero_dual_restart_skipped_prior_rejection"] = True
                stage_record["prior_rejected_zero_dual_attempt"] = {
                    "stage": prior_zero_rejection["stage"],
                    "zero_dual_solve_identity_sha256": (candidate_zero_dual_identity),
                    "result_dual_sha256": prior_zero_rejection["result_dual_sha256"],
                    "stage_record_sha256": sha256_json(prior_zero_rejection),
                }
            pending_zero_dual_retry_fraction = None
            lower_fraction = 0.0 if previous_fraction is None else previous_fraction
            half_step = 0.5 * (fraction - lower_fraction)
            can_subdivide = bool(
                adaptive_continuation
                and adaptive_subdivision_count < maximum_adaptive_subdivisions
                and half_step + 8.0 * np.finfo(np.float64).eps >= adaptive_minimum_step
            )
            if can_subdivide:
                midpoint = lower_fraction + half_step
                if not lower_fraction < midpoint < fraction:  # pragma: no cover
                    raise RuntimeError("adaptive continuation midpoint collapsed")
                planned_fractions.insert(next_position, midpoint)
                adaptive_subdivision_count += 1
                stage_record["adaptive_subdivision_triggered"] = True
                stage_record["adaptive_retry_fraction"] = midpoint
                if initialization_mode == "zero_dual":
                    rejected_zero_dual_by_identity[candidate_zero_dual_identity] = (
                        stage_record
                    )
                if continuation_journal is not None:
                    continuation_journal.commit_pending_transition(
                        stage_records=stage_records,
                        planned_fractions=planned_fractions,
                        next_position=next_position,
                        adaptive_subdivision_count=adaptive_subdivision_count,
                        pending_zero_dual_retry_fraction=None,
                        direct_exact_target_status=direct_exact_target_status,
                    )
                # Deliberately retain ``previous_dual`` and
                # ``previous_fraction`` from the last accepted stage.  The
                # rejected result is diagnostic only and cannot contaminate a
                # subsequent warm start.
                continue
            publication_failed = [
                name for name, passed in gates["checks"].items() if not passed
            ]
            intermediate_failed = [
                name
                for name, passed in intermediate_gate["checks"].items()
                if not passed
            ]
            if is_final_stage:
                reason = (
                    "final exact-target stage requires the unchanged original "
                    "publication gates: " + ", ".join(publication_failed)
                )
            else:
                reason = (
                    "failed original publication gates ("
                    + (", ".join(publication_failed) or "none")
                    + ") and fail-closed intermediate warm-start gate ("
                    + ", ".join(intermediate_failed)
                    + ")"
                )
            raise ProjectionContinuationFailure(
                "refusing continuation stage "
                f"{position}/{len(schedule)} (fraction={fraction:g}): {reason}",
                result=result,
                stage_fraction=fraction,
                stage_records=stage_records,
            )
        if direct_exact_target_trial:
            direct_exact_target_status = "accepted"
        previous_dual = result.dual_variables.copy()
        previous_fraction = fraction
        pending_zero_dual_retry_fraction = None
        final_result = result
        next_position += 1
        if continuation_journal is not None:
            continuation_journal.commit(
                result=result,
                stage_records=stage_records,
                planned_fractions=planned_fractions,
                next_position=next_position,
                adaptive_subdivision_count=adaptive_subdivision_count,
                direct_exact_target_status=direct_exact_target_status,
            )

    if final_result is None:  # pragma: no cover - guarded by schedule validation
        raise RuntimeError("continuation produced no final result")
    return final_result, stage_records


def _validate_reverse_build(
    adjoint_executable: Path,
    differentiable_primal_executable: Path,
    differentiable_tangent_executable: Path,
) -> dict[str, Any]:
    build_dir = adjoint_executable.parent
    if (
        build_dir.name != "build-v5"
        or differentiable_primal_executable.parent != build_dir
        or differentiable_tangent_executable.parent != build_dir
    ):
        raise ValueError(
            "the direct experiment requires all matched executables from build-v5"
        )
    provenance_path = adjoint_executable.parent / "build_provenance.txt"
    values: dict[str, str] = {}
    for line in provenance_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    required = {
        "schema",
        "driver_sha256",
        "runtime_sha256",
        "canonical_build_manifest_sha256",
        "canonical_object_manifest_sha256",
        "scientific_primal_flags",
        "zc_one_step_primal_sha256",
        "zc_one_step_tangent_sha256",
        "adjoint_executable_sha256",
    }
    if not required.issubset(values):
        raise ValueError("build-v5 matched provenance is incomplete")
    driver = adjoint_executable.parent.parent / "zc_one_step_adjoint_driver.F"
    if values["schema"] != "zc-controlled-one-step-consistent-build-v2":
        raise ValueError("build-v5 reverse provenance has another schema")
    if values["driver_sha256"] != sha256_file(driver):
        raise ValueError("build-v5 reverse driver hash changed")
    expected_flags = (
        "-std=legacy -O0 -g -fcheck=all -fbacktrace "
        "-fallow-argument-mismatch -ffixed-line-length-none"
    )
    if values["scientific_primal_flags"] != expected_flags:
        raise ValueError("build-v5 scientific primal is not the matched O0 build")
    if values["zc_one_step_primal_sha256"] != sha256_file(
        differentiable_primal_executable
    ):
        raise ValueError("build-v5 differentiable primal hash changed")
    if values["zc_one_step_tangent_sha256"] != sha256_file(
        differentiable_tangent_executable
    ):
        raise ValueError("build-v5 differentiable tangent hash changed")
    if values["adjoint_executable_sha256"] != sha256_file(adjoint_executable):
        raise ValueError("build-v5 reverse executable hash changed")
    return {
        "scientific_floating_point_contract": "GNU Fortran O0",
        "primal_executable": str(differentiable_primal_executable),
        "primal_executable_sha256": sha256_file(differentiable_primal_executable),
        "tangent_executable": str(differentiable_tangent_executable),
        "tangent_executable_sha256": sha256_file(differentiable_tangent_executable),
        "executable": str(adjoint_executable),
        "executable_sha256": sha256_file(adjoint_executable),
        "build_provenance": str(provenance_path),
        "build_provenance_sha256": sha256_file(provenance_path),
        "values": values,
    }


def _ensure_initial_state(
    member_dir: Path,
    *,
    release_index: int,
    generation_workspace: Path,
    runtime_source: Path,
    kernel_executable: Path,
) -> tuple[Path, dict[str, Any]]:
    report_path = member_dir / "initial_state_report.json"
    checkpoint_path = member_dir / "preconditioning_start.hst"
    packed_path = member_dir / "initial_state.bin"
    if report_path.is_file():
        report = load_json(report_path)
        expected = {
            "schema_version": 1,
            "status": "complete",
            "release_input_index": release_index,
            "preconditioning_start_input_index": release_index - CONTROL_STEPS,
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(f"cached initial-state binding changed in {member_dir}")
        files = (("checkpoint", checkpoint_path), ("packed_state", packed_path))
        for key, path in files:
            record = report.get(key)
            if (
                not isinstance(record, dict)
                or record.get("sha256") != sha256_file(path)
                or record.get("size_bytes") != path.stat().st_size
            ):
                raise ValueError(f"cached {key} fails provenance in {member_dir}")
        return packed_path, report

    member_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="initial-", dir=member_dir) as raw:
        temporary = Path(raw)
        staging = temporary / "staging"
        staging.mkdir()
        temporary_checkpoint = staging / "preconditioning_start.hst"
        checkpoint_report = impulse._extract_authentic_checkpoint(
            label=f"i{release_index}",
            input_index=release_index - CONTROL_STEPS,
            generation_workspace=generation_workspace,
            destination=temporary_checkpoint,
        )
        temporary_packed = staging / "initial_state.bin"
        packed_report = established._materialize_initial_state(
            release_index=release_index,
            checkpoint=temporary_checkpoint,
            destination=temporary_packed,
            runtime_source=runtime_source,
            kernel_executable=kernel_executable,
            reusable_pilot_state=staging / "not-reused.bin",
            cache_dir=staging,
            cache_manifest=None,
        )
        with atomic_output_path(checkpoint_path, overwrite=True) as destination:
            shutil.copyfile(temporary_checkpoint, destination)
        with atomic_output_path(packed_path, overwrite=True) as destination:
            shutil.copyfile(temporary_packed, destination)
    report = {
        "schema_version": 1,
        "status": "complete",
        "release_input_index": release_index,
        "preconditioning_start_input_index": release_index - CONTROL_STEPS,
        "checkpoint": {
            **checkpoint_report,
            "file": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "packed_state": {
            **packed_report,
            "file": packed_path.name,
            "sha256": sha256_file(packed_path),
            "size_bytes": packed_path.stat().st_size,
        },
    }
    write_json(report_path, report, overwrite=False)
    return packed_path, report


def _branch_changes(replay: ControlledReplay, baseline: ZeroControlPath) -> int:
    return int(
        sum(
            np.count_nonzero(actual != expected)
            for actual, expected in zip(replay.tapes, baseline.tapes, strict=True)
        )
    )


def _clock_equal(replay: ControlledReplay, baseline: ZeroControlPath) -> bool:
    return all(
        np.array_equal(actual.integers, expected.integers)
        and np.array_equal(actual.passive_time, expected.passive_time)
        for actual, expected in zip(replay.states, baseline.states, strict=True)
    )


def _compiler_transfer_audit(
    *,
    canonical_replay: ControlledReplay,
    differentiable_replay: ControlledReplay,
    canonical_baseline: ZeroControlPath,
    differentiable_baseline: ZeroControlPath,
    observation_chain: FrozenCore4ObservationChain,
    direction: np.ndarray,
    target_projection: float,
    canonical_baseline_final_nino3: float,
    differentiable_baseline_final_nino3: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Audit a frozen O0-optimized control by replaying it under canonical O3."""

    canonical_release = observation_chain.forward(
        canonical_replay.states[RELEASE_BOUNDARIES[0]],
        canonical_replay.states[RELEASE_BOUNDARIES[1]],
    )
    differentiable_release = observation_chain.forward(
        differentiable_replay.states[RELEASE_BOUNDARIES[0]],
        differentiable_replay.states[RELEASE_BOUNDARIES[1]],
    )
    canonical_nino3 = np.asarray(
        [nino3_from_real_state(state.real32) for state in canonical_replay.states],
        dtype=np.float64,
    )
    differentiable_nino3 = np.asarray(
        [nino3_from_real_state(state.real32) for state in differentiable_replay.states],
        dtype=np.float64,
    )
    canonical_projection = float(direction @ canonical_release[:-2].astype(np.float64))
    differentiable_projection = float(
        direction @ differentiable_release[:-2].astype(np.float64)
    )
    canonical_constraint = canonical_projection - target_projection
    differentiable_constraint = differentiable_projection - target_projection
    canonical_clocks_match = _clock_equal(canonical_replay, canonical_baseline)
    differentiable_clocks_match = _clock_equal(
        differentiable_replay, differentiable_baseline
    )
    finite = bool(
        np.isfinite(canonical_release).all()
        and np.isfinite(differentiable_release).all()
        and np.isfinite(canonical_nino3).all()
        and np.isfinite(differentiable_nino3).all()
    )
    if not finite or not canonical_clocks_match or not differentiable_clocks_match:
        raise RuntimeError(
            "frozen-control compiler transfer replay failed structural integrity"
        )
    cross_compiler_branch_changes = int(
        sum(
            np.count_nonzero(canonical != differentiable)
            for canonical, differentiable in zip(
                canonical_replay.tapes,
                differentiable_replay.tapes,
                strict=True,
            )
        )
    )
    canonical_terminal = float(canonical_nino3[-1])
    differentiable_terminal = float(differentiable_nino3[-1])
    report = {
        "purpose": (
            "Forward-only transfer of the frozen matched-O0 optimized controls "
            "to the canonical O3 executable"
        ),
        "used_by_optimizer": False,
        "numerical_differences_used_as_solver_gate": False,
        "gross_structural_integrity_required": True,
        "canonical_o3": {
            "release_xai_projection": canonical_projection,
            "release_agop_projection": canonical_projection,
            "release_constraint_value": canonical_constraint,
            "release_constraint_literally_satisfied": canonical_constraint >= 0.0,
            "terminal_nino3_c": canonical_terminal,
            "paired_terminal_nino3_change_c": (
                canonical_terminal - canonical_baseline_final_nino3
            ),
            "branch_tape_entries_changed_vs_unforced_o3": _branch_changes(
                canonical_replay, canonical_baseline
            ),
            "integer_and_passive_clocks_match_unforced_o3": canonical_clocks_match,
        },
        "matched_o0": {
            "release_xai_projection": differentiable_projection,
            "release_agop_projection": differentiable_projection,
            "release_constraint_value": differentiable_constraint,
            "terminal_nino3_c": differentiable_terminal,
            "paired_terminal_nino3_change_c": (
                differentiable_terminal - differentiable_baseline_final_nino3
            ),
            "branch_tape_entries_changed_vs_unforced_o0": _branch_changes(
                differentiable_replay, differentiable_baseline
            ),
            "integer_and_passive_clocks_match_unforced_o0": (
                differentiable_clocks_match
            ),
        },
        "matched_o0_minus_canonical_o3": {
            "release_xai_projection": (
                differentiable_projection - canonical_projection
            ),
            "release_agop_projection": (
                differentiable_projection - canonical_projection
            ),
            "release_constraint_value": (
                differentiable_constraint - canonical_constraint
            ),
            "release_standardized_max_abs": float(
                np.max(
                    np.abs(
                        differentiable_release.astype(np.float64)
                        - canonical_release.astype(np.float64)
                    ),
                    initial=0.0,
                )
            ),
            "terminal_nino3_c": differentiable_terminal - canonical_terminal,
            "branch_tape_entries_different_between_compilers": (
                cross_compiler_branch_changes
            ),
        },
        "all_values_finite": finite,
    }
    return report, canonical_release, canonical_nino3


def _cross_build_zero_audit(
    differentiable: ZeroControlPath,
    canonical: ZeroControlPath,
) -> dict[str, float | bool]:
    """Certify that O0 and canonical O3 zero paths represent the same solution.

    The optimized path must use O0 because Tapenade's split reverse sweep was
    compiled at O0.  The independently generated data use the canonical O3
    build.  We therefore retain the exact O3/data comparison and separately
    enforce small accumulated O0--O3 discrepancies over all forty boundaries.
    """

    if differentiable.total_steps != canonical.total_steps:
        raise ValueError("cross-build paths have different lengths")
    if any(
        not np.array_equal(left, right)
        for left, right in zip(differentiable.tapes, canonical.tapes, strict=True)
    ):
        raise RuntimeError("O0 and canonical O3 zero paths select different branches")
    real_max = 0.0
    real_relative = 0.0
    complex_max = 0.0
    complex_relative = 0.0
    passive_exact = True
    for left, right in zip(differentiable.states, canonical.states, strict=True):
        passive_exact = passive_exact and (
            np.array_equal(left.real64, right.real64)
            and np.array_equal(left.integers, right.integers)
            and np.array_equal(left.passive_time, right.passive_time)
        )
        real_difference = left.real32.astype(np.float64) - right.real32.astype(
            np.float64
        )
        complex_difference = left.complex64.astype(
            np.complex128
        ) - right.complex64.astype(np.complex128)
        real_max = max(real_max, float(np.max(np.abs(real_difference), initial=0.0)))
        complex_max = max(
            complex_max,
            float(np.max(np.abs(complex_difference), initial=0.0)),
        )
        real_relative = max(
            real_relative,
            float(np.linalg.norm(real_difference))
            / max(float(np.linalg.norm(right.real32.astype(np.float64))), 1e-300),
        )
        complex_relative = max(
            complex_relative,
            float(np.linalg.norm(complex_difference))
            / max(
                float(np.linalg.norm(right.complex64.astype(np.complex128))),
                1e-300,
            ),
        )
    if not passive_exact:
        raise RuntimeError("O0 and canonical O3 zero paths differ in passive state")
    limits = {
        # Packed REAL coordinates have heterogeneous physical scales, so this
        # absolute guard catches only gross corruption; the relative and
        # standardized-release audits are the scientifically scaled checks.
        "real_max": 1e-2,
        "real_relative_l2": 1e-6,
        "complex_max": 1e-4,
        "complex_relative_l2": 1e-5,
    }
    measured = {
        "real_max": real_max,
        "real_relative_l2": real_relative,
        "complex_max": complex_max,
        "complex_relative_l2": complex_relative,
    }
    if any(measured[key] > limit for key, limit in limits.items()):
        raise RuntimeError(
            "O0 and canonical O3 zero paths exceed the forty-step cross-build "
            f"limits: measured={measured}, limits={limits}"
        )
    return {
        "all_branch_tapes_equal": True,
        "passive_state_bitwise_equal": True,
        **{f"maximum_{key}": value for key, value in measured.items()},
        **{f"hard_limit_{key}": value for key, value in limits.items()},
    }


def _normalized_kkt(
    result: NativeScalarSQPResult,
    *,
    initial_projection_gap: float,
    stationarity_reference_norm: float,
) -> dict[str, float]:
    return {
        "primal_violation_over_initial_projection_gap": (
            result.primal_violation / max(abs(initial_projection_gap), 1e-300)
        ),
        "stationarity_over_kkt_term_norm": (
            result.covariance_stationarity_norm
            / max(stationarity_reference_norm, np.finfo(np.float64).tiny)
        ),
        "complementarity_over_squared_action": (
            result.complementarity_absolute
            / max(result.squared_action, np.finfo(np.float64).tiny)
        ),
    }


def _solver_publication_gates(
    result: NativeScalarSQPResult,
    *,
    solver_settings: dict[str, Any],
    replay_constraint: float | None = None,
) -> dict[str, Any]:
    """Evaluate the original configured KKT gates without relaxing them."""

    relative_stationarity_tolerance = solver_settings.get(
        "relative_stationarity_tolerance"
    )
    relative_complementarity_tolerance = solver_settings.get(
        "relative_complementarity_tolerance"
    )
    if relative_stationarity_tolerance is None:
        stationarity_value = float(result.covariance_stationarity_norm)
        stationarity_tolerance = float(solver_settings["stationarity_tolerance"])
        stationarity_metric = "absolute_covariance_metric_norm"
    else:
        stationarity_value = float(result.covariance_stationarity_relative)
        stationarity_tolerance = float(relative_stationarity_tolerance)
        stationarity_metric = "relative_covariance_metric_norm"
    if relative_complementarity_tolerance is None:
        complementarity_value = float(result.complementarity_absolute)
        complementarity_tolerance = float(solver_settings["complementarity_tolerance"])
        complementarity_metric = "absolute"
    else:
        complementarity_value = float(result.complementarity_relative)
        complementarity_tolerance = float(relative_complementarity_tolerance)
        complementarity_metric = "relative_to_squared_action"

    replay_value = None if replay_constraint is None else float(replay_constraint)
    finite_values = (
        float(result.constraint_value),
        float(result.primal_violation),
        stationarity_value,
        complementarity_value,
    )
    if replay_value is not None:
        finite_values += (replay_value,)
    checks = {
        "solver_declared_success": result.success is True,
        "solver_constraint_is_literally_nonnegative": (
            math.isfinite(float(result.constraint_value))
            and float(result.constraint_value) >= 0.0
        ),
        "absolute_primal_violation_within_configured_tolerance": (
            float(result.primal_violation)
            <= float(solver_settings["constraint_tolerance"])
        ),
        "stationarity_within_configured_tolerance": (
            stationarity_value <= stationarity_tolerance
        ),
        "complementarity_within_configured_tolerance": (
            complementarity_value <= complementarity_tolerance
        ),
        "all_gate_values_finite": all(math.isfinite(value) for value in finite_values),
    }
    if replay_value is not None:
        checks.update(
            {
                "frozen_replay_constraint_is_literally_nonnegative": (
                    math.isfinite(replay_value) and replay_value >= 0.0
                ),
                "frozen_replay_matches_solver_constraint": math.isclose(
                    replay_value,
                    float(result.constraint_value),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
            }
        )
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "constraint_tolerance": float(solver_settings["constraint_tolerance"]),
        "stationarity": {
            "metric": stationarity_metric,
            "value": stationarity_value,
            "tolerance": stationarity_tolerance,
        },
        "complementarity": {
            "metric": complementarity_metric,
            "value": complementarity_value,
            "tolerance": complementarity_tolerance,
        },
    }


def _intermediate_continuation_gate(
    result: NativeScalarSQPResult,
    *,
    solver_settings: dict[str, Any],
    publication_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Accept only a finite, feasible near-KKT iterate as a nonfinal warm start.

    This gate never authorizes scientific publication.  It permits a nonfinal
    continuation stage to miss solver-declared success, literal nonnegativity,
    and/or the strict feasibility or stationarity thresholds.  Its two-sided
    constraint residual and primal violation are bounded by five times the
    configured constraint tolerance, while stationarity is bounded by twice
    its configured threshold.  Complementarity remains unrelaxed.
    """

    publication = (
        _solver_publication_gates(result, solver_settings=solver_settings)
        if publication_gates is None
        else publication_gates
    )
    relative_stationarity_tolerance = solver_settings.get(
        "relative_stationarity_tolerance"
    )
    if relative_stationarity_tolerance is None:
        stationarity_value = float(result.covariance_stationarity_norm)
        strict_stationarity_tolerance = float(solver_settings["stationarity_tolerance"])
        stationarity_metric = "absolute_covariance_metric_norm"
    else:
        stationarity_value = float(result.covariance_stationarity_relative)
        strict_stationarity_tolerance = float(relative_stationarity_tolerance)
        stationarity_metric = "relative_covariance_metric_norm"

    relative_complementarity_tolerance = solver_settings.get(
        "relative_complementarity_tolerance"
    )
    if relative_complementarity_tolerance is None:
        complementarity_value = float(result.complementarity_absolute)
        complementarity_tolerance = float(solver_settings["complementarity_tolerance"])
        complementarity_metric = "absolute"
    else:
        complementarity_value = float(result.complementarity_relative)
        complementarity_tolerance = float(relative_complementarity_tolerance)
        complementarity_metric = "relative_to_squared_action"

    scalar_metrics = {
        "objective_value": float(result.objective_value),
        "squared_action": float(result.squared_action),
        "constraint_value": float(result.constraint_value),
        "primal_violation": float(result.primal_violation),
        "lagrange_multiplier": float(result.lagrange_multiplier),
        "covariance_stationarity_norm": float(result.covariance_stationarity_norm),
        "covariance_stationarity_relative": float(
            result.covariance_stationarity_relative
        ),
        "complementarity_absolute": float(result.complementarity_absolute),
        "complementarity_relative": float(result.complementarity_relative),
        "stationarity_reference_norm": float(result.stationarity_reference_norm),
        "constraint_gradient_covariance_norm": float(
            result.constraint_gradient_covariance_norm
        ),
    }
    arrays = {
        "native_interventions": np.asarray(result.native_interventions),
        "dual_variables": np.asarray(result.dual_variables),
        "native_constraint_gradient": np.asarray(result.native_constraint_gradient),
    }
    permitted_publication_failures = {
        "solver_declared_success",
        "solver_constraint_is_literally_nonnegative",
        "absolute_primal_violation_within_configured_tolerance",
        "stationarity_within_configured_tolerance",
    }
    publication_failures = {
        name for name, passed in publication["checks"].items() if not passed
    }
    relaxed_stationarity_tolerance = (
        INTERMEDIATE_CONTINUATION_STATIONARITY_MULTIPLIER
        * strict_stationarity_tolerance
    )
    strict_constraint_tolerance = float(solver_settings["constraint_tolerance"])
    relaxed_constraint_tolerance = (
        INTERMEDIATE_CONTINUATION_CONSTRAINT_MULTIPLIER * strict_constraint_tolerance
    )
    checks = {
        "original_publication_failures_limited_to_documented_intermediate_"
        "exceptions": publication_failures <= permitted_publication_failures,
        "all_result_metrics_finite": all(
            math.isfinite(value) for value in scalar_metrics.values()
        ),
        "all_result_arrays_finite": all(
            np.isfinite(values).all() for values in arrays.values()
        ),
        "absolute_constraint_residual_within_five_times_configured_tolerance": (
            math.isfinite(float(result.constraint_value))
            and abs(float(result.constraint_value)) <= relaxed_constraint_tolerance
        ),
        "absolute_primal_violation_is_nonnegative_and_within_five_times_"
        "configured_tolerance": (
            math.isfinite(float(result.primal_violation))
            and 0.0 <= float(result.primal_violation) <= relaxed_constraint_tolerance
        ),
        "stationarity_is_nonnegative_and_within_twice_configured_tolerance": (
            math.isfinite(stationarity_value)
            and 0.0 <= stationarity_value <= relaxed_stationarity_tolerance
        ),
        "complementarity_is_nonnegative_and_within_configured_tolerance": (
            math.isfinite(complementarity_value)
            and 0.0 <= complementarity_value <= complementarity_tolerance
        ),
    }
    return {
        "all_passed": all(checks.values()),
        "warm_start_only": True,
        "scientific_publication_authorized": False,
        "checks": checks,
        "original_publication_gate_failures": sorted(publication_failures),
        "constraint": {
            "value": float(result.constraint_value),
            "primal_violation": float(result.primal_violation),
            "strict_publication_tolerance": strict_constraint_tolerance,
            "intermediate_warm_start_multiplier": (
                INTERMEDIATE_CONTINUATION_CONSTRAINT_MULTIPLIER
            ),
            "intermediate_warm_start_tolerance": relaxed_constraint_tolerance,
            "two_sided_residual": True,
            "literal_nonnegative_required": False,
        },
        "stationarity": {
            "metric": stationarity_metric,
            "value": stationarity_value,
            "strict_publication_tolerance": strict_stationarity_tolerance,
            "intermediate_warm_start_multiplier": (
                INTERMEDIATE_CONTINUATION_STATIONARITY_MULTIPLIER
            ),
            "intermediate_warm_start_tolerance": relaxed_stationarity_tolerance,
        },
        "complementarity": {
            "metric": complementarity_metric,
            "value": complementarity_value,
            "tolerance": complementarity_tolerance,
        },
        "finite_metric_names": sorted(scalar_metrics),
        "finite_array_names": sorted(arrays),
    }


def _require_publishable_solver_result(
    result: NativeScalarSQPResult,
    *,
    solver_settings: dict[str, Any],
    replay_constraint: float | None = None,
) -> dict[str, Any]:
    gates = _solver_publication_gates(
        result,
        solver_settings=solver_settings,
        replay_constraint=replay_constraint,
    )
    if not gates["all_passed"]:
        failed = [name for name, passed in gates["checks"].items() if not passed]
        raise RuntimeError(
            "refusing to publish a case that failed the original solver gates: "
            + ", ".join(failed)
        )
    return gates


def _completed_case_gate_failures(
    report: dict[str, Any], identity: dict[str, Any]
) -> list[str]:
    """Independently reject a cached report that is not scientifically valid."""

    failures: list[str] = []
    solver = report.get("solver")
    release = report.get("release")
    settings = identity.get("solver_settings")
    publication = report.get("publication_gates")
    if not isinstance(solver, dict):
        return ["missing solver record"]
    if not isinstance(release, dict):
        return ["missing release record"]
    if not isinstance(settings, dict):
        return ["missing solver settings"]
    if solver.get("success") is not True:
        failures.append("solver did not declare success")
    try:
        solver_constraint = float(solver["constraint_value"])
        release_constraint = float(release["constraint_value"])
        primal_violation = float(solver["primal_violation"])
        normalized = solver["normalized_kkt"]
        stationarity = float(normalized["stationarity_over_kkt_term_norm"])
        complementarity = float(normalized["complementarity_over_squared_action"])
        numeric_values = (
            solver_constraint,
            release_constraint,
            primal_violation,
            stationarity,
            complementarity,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            failures.append("nonfinite solver gate value")
        if solver_constraint < 0.0:
            failures.append("solver constraint is negative")
        if release_constraint < 0.0:
            failures.append("frozen replay constraint is negative")
        if not math.isclose(
            solver_constraint, release_constraint, rel_tol=0.0, abs_tol=1e-9
        ):
            failures.append("solver and frozen replay constraints disagree")
        if primal_violation > float(settings["constraint_tolerance"]):
            failures.append("primal violation exceeds configured tolerance")
        relative_stationarity = settings.get("relative_stationarity_tolerance")
        if relative_stationarity is not None:
            if stationarity > float(relative_stationarity):
                failures.append("relative stationarity exceeds configured tolerance")
        elif float(solver["covariance_stationarity_norm"]) > float(
            settings["stationarity_tolerance"]
        ):
            failures.append("absolute stationarity exceeds configured tolerance")
        relative_complementarity = settings.get("relative_complementarity_tolerance")
        if relative_complementarity is not None:
            if complementarity > float(relative_complementarity):
                failures.append("relative complementarity exceeds configured tolerance")
        elif float(solver["complementarity_absolute"]) > float(
            settings["complementarity_tolerance"]
        ):
            failures.append("absolute complementarity exceeds configured tolerance")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"malformed solver gate record: {error}")
    if not isinstance(publication, dict) or publication.get("all_passed") is not True:
        failures.append("publication gate record is absent or failed")
    else:
        checks = publication.get("checks")
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            failures.append("publication gate record contains a failed check")
    transfer = report.get("canonical_o3_frozen_control_transfer_audit")
    if not isinstance(transfer, dict):
        failures.append("canonical-O3 frozen-control transfer audit is absent")
    else:
        canonical = transfer.get("canonical_o3")
        if (
            transfer.get("all_values_finite") is not True
            or transfer.get("used_by_optimizer") is not False
            or not isinstance(canonical, dict)
            or canonical.get("integer_and_passive_clocks_match_unforced_o3") is not True
        ):
            failures.append("canonical-O3 frozen-control transfer audit is invalid")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict):
        failures.append("complete report has no runtime accounting")
    else:
        try:
            solver_wall = float(runtime["solver_total_wall_seconds"])
            post_wall = float(
                runtime["post_solver_replay_and_report_preparation_wall_seconds"]
            )
            total_wall = float(runtime["total_wall_seconds"])
            if not all(
                math.isfinite(value) and value >= 0.0
                for value in (solver_wall, post_wall, total_wall)
            ) or not math.isclose(
                total_wall,
                solver_wall + post_wall,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                failures.append("complete report runtime accounting is inconsistent")
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"malformed complete-report runtime: {error}")
    return failures


def _stationarity_reference_norm(
    result: NativeScalarSQPResult,
    covariance: Any,
    *,
    block_rows: int,
) -> tuple[float, float, int]:
    """Return cached KKT scales and zero extra covariance-operator calls."""

    del covariance, block_rows
    return (
        result.stationarity_reference_norm,
        result.constraint_gradient_covariance_norm,
        0,
    )


def _solver_report(
    result: NativeScalarSQPResult,
    *,
    covariance_case: str,
    initial_projection_gap: float,
    stationarity_reference_norm: float,
    constraint_gradient_covariance_norm: float,
    reporting_covariance_operator_calls: int,
    covariance_backend: str = "legacy_centered_sample_factors",
    covariance_source_phase_count: int = CONTROL_STEPS,
) -> dict[str, Any]:
    accepted = sum(item.accepted_step_action_norm > 0.0 for item in result.iterations)
    rejected = sum(
        item.proposed_step_action_norm > 0.0 and item.accepted_step_action_norm == 0.0
        for item in result.iterations
    )
    if covariance_case not in CASE_NAMES:
        raise ValueError(f"unknown covariance case: {covariance_case}")
    if covariance_source_phase_count <= 0:
        raise ValueError("covariance_source_phase_count must be positive")
    if covariance_backend == "precompiled_dense_covariance":
        phase_multiplier = 0
        dense_multiplier = 1
        counting_definition = (
            "Each covariance-operator call is one application of the single "
            "precompiled dense covariance pooled over all 36 annual phases. "
            "The centered 10,000-state phase factors are not read at solve time."
        )
    elif covariance_backend == "legacy_centered_sample_factors":
        phase_multiplier = (
            covariance_source_phase_count if covariance_case == "pooled" else 1
        )
        dense_multiplier = 0
        counting_definition = (
            "A covariance-operator call applies all source phases together for "
            "pooled C, but applies one phase for phase-specific C_i. A "
            "phase-factor pass is one complete read of one centered "
            "10,000-state annual-phase sample factor."
        )
    else:
        raise ValueError(f"unknown covariance backend: {covariance_backend}")
    solver_operator_calls = int(result.covariance_passes)
    reporting_operator_calls = int(reporting_covariance_operator_calls)
    radial_operator_calls = int(result.radial_restoration_covariance_passes)
    return {
        "success": result.success,
        "status": result.status,
        "objective_value": result.objective_value,
        "squared_action": result.squared_action,
        "action_norm": math.sqrt(max(0.0, result.squared_action)),
        "constraint_value": result.constraint_value,
        "primal_violation": result.primal_violation,
        "lagrange_multiplier": result.lagrange_multiplier,
        "covariance_stationarity_norm": result.covariance_stationarity_norm,
        "complementarity_absolute": result.complementarity_absolute,
        "stationarity_reference_norm": stationarity_reference_norm,
        "constraint_gradient_covariance_norm": (constraint_gradient_covariance_norm),
        "normalized_kkt": _normalized_kkt(
            result,
            initial_projection_gap=initial_projection_gap,
            stationarity_reference_norm=stationarity_reference_norm,
        ),
        "major_iteration_records": len(result.iterations),
        "accepted_update_count": int(accepted),
        "rejected_update_count": int(rejected),
        "value_evaluations": result.value_evaluations,
        "gradient_evaluations": result.gradient_evaluations,
        "covariance_work": {
            "solver_covariance_operator_apply_calls": solver_operator_calls,
            "solver_phase_factor_full_sample_passes": (
                phase_multiplier * solver_operator_calls
            ),
            "solver_dense_covariance_matrix_apply_calls": (
                dense_multiplier * solver_operator_calls
            ),
            "reporting_covariance_operator_apply_calls": reporting_operator_calls,
            "reporting_phase_factor_full_sample_passes": (
                phase_multiplier * reporting_operator_calls
            ),
            "reporting_dense_covariance_matrix_apply_calls": (
                dense_multiplier * reporting_operator_calls
            ),
            "covariance_backend": covariance_backend,
            "covariance_source_phase_count": covariance_source_phase_count,
            "counting_definition": counting_definition,
        },
        "radial_restoration": {
            "applied": result.radial_restoration_applied,
            "reason": result.radial_restoration_reason,
            "scale": result.radial_restoration_scale,
            "value_evaluations": result.radial_restoration_value_evaluations,
            "gradient_evaluations": result.radial_restoration_gradient_evaluations,
            "covariance_operator_apply_calls": radial_operator_calls,
            "phase_factor_full_sample_passes": (
                phase_multiplier * radial_operator_calls
            ),
            "dense_covariance_matrix_apply_calls": (
                dense_multiplier * radial_operator_calls
            ),
        },
        "iterations": [asdict(item) for item in result.iterations],
    }


def _case_identity(
    *,
    run_identity_sha256: str,
    release_index: int,
    initial_state_sha256: str,
    covariance_case: str,
    solver_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_identity_sha256": run_identity_sha256,
        "release_input_index": release_index,
        "initial_state_sha256": initial_state_sha256,
        "covariance_case": covariance_case,
        "solver_settings": solver_settings,
    }


def _load_completed_case(
    case_dir: Path, identity: dict[str, Any]
) -> dict[str, Any] | None:
    report_path = case_dir / "report.json"
    if not report_path.is_file():
        return None
    report = load_json(report_path)
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != "complete"
        or report.get("case_identity_sha256") != sha256_json(identity)
        or report.get("case_identity") != identity
    ):
        raise ValueError(f"completed case has another identity: {case_dir}")
    artifact = report.get("artifact")
    artifact_path = case_dir / "result.npz"
    if (
        not isinstance(artifact, dict)
        or not artifact_path.is_file()
        or artifact.get("sha256") != sha256_file(artifact_path)
        or artifact.get("size_bytes") != artifact_path.stat().st_size
    ):
        raise ValueError(f"completed case artifact changed: {case_dir}")
    gate_failures = _completed_case_gate_failures(report, identity)
    if gate_failures:
        raise ValueError(
            f"completed case failed scientific publication gates: {case_dir}: "
            + "; ".join(gate_failures)
        )
    return report


def _next_failure_attempt(failures_dir: Path) -> int:
    attempts: list[int] = []
    for path in failures_dir.glob("attempt-*.json"):
        try:
            attempts.append(int(path.stem.removeprefix("attempt-")))
        except ValueError:
            continue
    return max(attempts, default=0) + 1


def _publish_case_failure(
    case_dir: Path,
    *,
    identity: dict[str, Any],
    stage: str,
    error: Exception,
    runtime: dict[str, Any],
    solver_result: NativeScalarSQPResult | None = None,
    solver_settings: dict[str, Any] | None = None,
    covariance_case: str | None = None,
    initial_projection_gap: float | None = None,
    continuation_stages: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Publish an explicit, retryable case failure without a result artifact."""

    case_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = case_dir / "failures"
    failures_dir.mkdir(exist_ok=True)
    attempt = _next_failure_attempt(failures_dir)
    nonpublishable_diagnostic: dict[str, Any] | None = None
    if solver_result is not None:
        if (
            solver_settings is None
            or covariance_case is None
            or initial_projection_gap is None
        ):
            raise ValueError(
                "solver failure diagnostics require settings, covariance case, "
                "and the initial projection gap"
            )
        checkpoint_path = failures_dir / f"attempt-{attempt:04d}-solver-checkpoint.npz"
        write_npz(
            checkpoint_path,
            overwrite=False,
            dual_variables=solver_result.dual_variables,
            native_interventions=solver_result.native_interventions,
            native_constraint_gradient=solver_result.native_constraint_gradient,
        )
        nonpublishable_diagnostic = {
            "status": "nonpublishable_solver_checkpoint",
            "scientific_result_published": False,
            "canonical_result_or_report_created": False,
            "checkpoint": {
                "file": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "dual_variables_sha256": sha256_array(solver_result.dual_variables),
                "native_interventions_sha256": sha256_array(
                    solver_result.native_interventions
                ),
                "native_constraint_gradient_sha256": sha256_array(
                    solver_result.native_constraint_gradient
                ),
            },
            "warm_start_note": (
                "dual_variables can initialize a new, fully gated SQP solve; "
                "this is a warm start rather than an exact resume because the "
                "trust radius and merit state are not checkpointed"
            ),
            "publication_gates": _solver_publication_gates(
                solver_result, solver_settings=solver_settings
            ),
            "solver": _solver_report(
                solver_result,
                covariance_case=covariance_case,
                initial_projection_gap=float(initial_projection_gap),
                stationarity_reference_norm=(solver_result.stationarity_reference_norm),
                constraint_gradient_covariance_norm=(
                    solver_result.constraint_gradient_covariance_norm
                ),
                reporting_covariance_operator_calls=0,
                covariance_backend=str(
                    solver_settings.get(
                        "covariance_backend", "legacy_centered_sample_factors"
                    )
                ),
                covariance_source_phase_count=int(
                    solver_settings.get(
                        "covariance_source_phase_count", CONTROL_STEPS
                    )
                ),
            ),
            "continuation_stages": list(continuation_stages),
        }
    failure = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "failed",
        "retry_policy": "retry_on_next_invocation",
        "case_identity": identity,
        "case_identity_sha256": sha256_json(identity),
        "attempt": attempt,
        "stage": stage,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        },
        "runtime_before_failure": runtime,
        "scientific_result_published": False,
        "nonpublishable_solver_diagnostic": nonpublishable_diagnostic,
        "failed_utc": datetime.now(UTC).isoformat(),
    }
    write_json(
        failures_dir / f"attempt-{attempt:04d}.json",
        failure,
        overwrite=False,
    )
    write_json(case_dir / "failure.json", failure, overwrite=True)
    return failure


def _clear_active_case_failure(case_dir: Path) -> None:
    """Clear the active marker after success while retaining attempt history."""

    (case_dir / "failure.json").unlink(missing_ok=True)


def _continuation_runtime_totals(
    stage_records: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    """Sum checkpointed stage work once, including rejected attempts."""

    totals: dict[str, float | int] = {
        "solver_wall_seconds": 0.0,
        "release_only_forward_replays": 0,
        "release_only_reverse_sweeps": 0,
        "reverse_fortran_wall_seconds": 0.0,
    }
    integer_keys = {
        "release_only_forward_replays",
        "release_only_reverse_sweeps",
    }
    for position, stage in enumerate(stage_records, start=1):
        runtime = stage.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError(f"continuation stage {position} has no runtime record")
        for key in totals:
            value = runtime.get(key)
            if key in integer_keys:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"continuation stage {position} has invalid {key}")
                totals[key] = int(totals[key]) + value
            else:
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(f"continuation stage {position} has invalid {key}")
                totals[key] = float(totals[key]) + numeric
    return totals


def _publish_case(
    case_dir: Path,
    *,
    identity: dict[str, Any],
    result: NativeScalarSQPResult,
    full_replay: ControlledReplay,
    canonical_full_replay: ControlledReplay,
    baseline: ZeroControlPath,
    canonical_baseline: ZeroControlPath,
    baseline_release: np.ndarray,
    baseline_final_nino3: float,
    canonical_baseline_final_nino3: float,
    observation_chain: FrozenCore4ObservationChain,
    experiment: LoadedExperiment,
    direction: np.ndarray,
    target_projection: float,
    initial_projection_gap: float,
    stationarity_reference_norm: float,
    constraint_gradient_covariance_norm: float,
    reporting_covariance_operator_calls: int,
    solver_settings: dict[str, Any],
    covariance_case: str,
    oracle_runtime: dict[str, Any],
    continuation_stages: Sequence[dict[str, Any]],
    durable_continuation_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    release = observation_chain.forward(
        full_replay.states[RELEASE_BOUNDARIES[0]],
        full_replay.states[RELEASE_BOUNDARIES[1]],
    )
    release_spatial = release[:-2].astype(np.float64)
    baseline_spatial = baseline_release[:-2].astype(np.float64)
    projection = float(direction @ release_spatial)
    constraint = projection - target_projection
    publication_gates = _require_publishable_solver_result(
        result,
        solver_settings=solver_settings,
        replay_constraint=constraint,
    )
    nino3 = np.asarray(
        [nino3_from_real_state(state.real32) for state in full_replay.states],
        dtype=np.float64,
    )
    final_nino3 = float(nino3[-1])
    transfer_audit, canonical_release, canonical_nino3 = _compiler_transfer_audit(
        canonical_replay=canonical_full_replay,
        differentiable_replay=full_replay,
        canonical_baseline=canonical_baseline,
        differentiable_baseline=baseline,
        observation_chain=observation_chain,
        direction=direction,
        target_projection=target_projection,
        canonical_baseline_final_nino3=canonical_baseline_final_nino3,
        differentiable_baseline_final_nino3=baseline_final_nino3,
    )
    delta = release_spatial - baseline_spatial
    delta_projection = float(direction @ delta)
    perpendicular = delta - delta_projection * direction
    artifact_path = case_dir / "result.npz"
    write_npz(
        artifact_path,
        overwrite=True,
        native_interventions=result.native_interventions,
        dual_variables=result.dual_variables,
        native_constraint_gradient=result.native_constraint_gradient,
        release_standardized=release,
        release_delta_standardized_spatial=delta,
        nino3_by_boundary_c=nino3,
        canonical_o3_release_standardized=canonical_release,
        canonical_o3_nino3_by_boundary_c=canonical_nino3,
    )
    solver = _solver_report(
        result,
        covariance_case=covariance_case,
        initial_projection_gap=initial_projection_gap,
        stationarity_reference_norm=stationarity_reference_norm,
        constraint_gradient_covariance_norm=constraint_gradient_covariance_norm,
        reporting_covariance_operator_calls=reporting_covariance_operator_calls,
        covariance_backend=str(
            solver_settings.get(
                "covariance_backend", "legacy_centered_sample_factors"
            )
        ),
        covariance_source_phase_count=int(
            solver_settings.get("covariance_source_phase_count", CONTROL_STEPS)
        ),
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "case_identity": identity,
        "case_identity_sha256": sha256_json(identity),
        "publication_gates": publication_gates,
        "solver": solver,
        "projection_continuation": {
            "acceptance_policy": _projection_continuation_acceptance_policy(),
            "direct_exact_target_first_enabled": bool(
                continuation_stages
                and continuation_stages[0].get(
                    "direct_exact_target_first_enabled", False
                )
            ),
            "direct_exact_target_status": (
                "accepted"
                if continuation_stages
                and continuation_stages[0].get("direct_exact_target_trial") is True
                and continuation_stages[0].get("accepted_for_continuation") is True
                else "rejected_fallback_activated"
                if continuation_stages
                and continuation_stages[0].get("direct_exact_target_trial") is True
                else "not_requested"
            ),
            "adaptive_fallback_activated": bool(
                continuation_stages
                and continuation_stages[0].get("direct_fallback_triggered", False)
            ),
            "skipped_redundant_zero_dual_solve_count": sum(
                item.get("zero_dual_restart_skipped_prior_rejection") is True
                for item in continuation_stages
            ),
            "fractions": [item["fraction"] for item in continuation_stages],
            "stage_count": len(continuation_stages),
            "used_intermediate_targets": len(continuation_stages) > 1,
            "used_fail_closed_intermediate_warm_start_gate": any(
                bool(item.get("intermediate_gate_used_to_continue", False))
                for item in continuation_stages
            ),
            "intermediate_scientific_results_published": False,
            "final_stage_uses_exact_target": bool(
                continuation_stages
                and continuation_stages[-1]["fraction"] == 1.0
                and continuation_stages[-1]["target_projection"] == target_projection
            ),
            "stages": list(continuation_stages),
            "durable_checkpoint": durable_continuation_checkpoint,
        },
        "release": {
            "target_extreme_xai_projection": target_projection,
            "realized_xai_projection": projection,
            "target_extreme_agop_projection": target_projection,
            "realized_agop_projection": projection,
            "constraint_value": constraint,
            "standardized_spatial_xai_projection_change": delta_projection,
            "standardized_spatial_agop_projection_change": delta_projection,
            "standardized_spatial_l2_change": float(np.linalg.norm(delta)),
            "standardized_spatial_rms_change": float(
                np.linalg.norm(delta) / math.sqrt(delta.size)
            ),
            "cosine_with_xai_direction": float(
                delta_projection / max(np.linalg.norm(delta), 1e-300)
            ),
            "cosine_with_agop": float(
                delta_projection / max(np.linalg.norm(delta), 1e-300)
            ),
            "standardized_spatial_perpendicular_l2_change": float(
                np.linalg.norm(perpendicular)
            ),
            "cnn_forecast_c": _model_forecast(experiment, release),
            "nino3_c": float(nino3_from_real_state(full_replay.states[10].real32)),
        },
        "outcome_after_control_frozen": {
            "target_input_index_offset_from_release": 30,
            "terminal_nino3_c": final_nino3,
            "paired_terminal_nino3_change_c": final_nino3 - baseline_final_nino3,
            "peak_nino3_from_release_through_terminal_c": float(np.max(nino3[10:])),
            "minimum_nino3_from_release_through_terminal_c": float(np.min(nino3[10:])),
        },
        "canonical_o3_frozen_control_transfer_audit": transfer_audit,
        "diagnostics_not_gates": {
            "all_values_finite": bool(
                np.isfinite(result.native_interventions).all()
                and np.isfinite(release).all()
                and np.isfinite(nino3).all()
            ),
            "integer_and_passive_clocks_match_unforced": _clock_equal(
                full_replay, baseline
            ),
            "branch_tape_entries_changed_vs_unforced": _branch_changes(
                full_replay, baseline
            ),
            "maximum_step_compact_intervention_l2": float(
                np.max(np.linalg.norm(result.native_interventions, axis=1))
            ),
            "maximum_step_compact_intervention_rms": float(
                np.max(np.sqrt(np.mean(result.native_interventions**2, axis=1)))
            ),
            "used_as_exclusion_or_safety_gate": False,
        },
        "runtime": {
            **oracle_runtime,
            "frozen_full_replay_fortran_wall_seconds": full_replay.wall_seconds,
            "canonical_o3_frozen_full_replay_fortran_wall_seconds": (
                canonical_full_replay.wall_seconds
            ),
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "member",
        "release_input_index",
        "case",
        "baseline_release_projection",
        "release_projection",
        "target_projection",
        "baseline_terminal_nino3_c",
        "terminal_nino3_c",
        "paired_terminal_nino3_change_c",
        "canonical_o3_release_projection",
        "canonical_o3_release_constraint",
        "canonical_o3_terminal_nino3_c",
        "matched_o0_minus_canonical_o3_terminal_nino3_c",
        "cnn_forecast_c",
        "solver_success",
        "accepted_update_count",
        "rejected_update_count",
        "normalized_primal_violation",
        "normalized_stationarity",
        "normalized_complementarity",
    ]
    with (
        atomic_output_path(path, overwrite=True) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    target_projection: float,
    extreme_target_nino3_c: float,
    xai_label: str = "AGOP",
) -> None:
    if not rows:
        return
    members = list(dict.fromkeys(row["member"] for row in rows))
    by_member = {
        member: {row["case"]: row for row in rows if row["member"] == member}
        for member in members
    }
    positions = np.arange(1, len(members) + 1, dtype=float)
    baseline_rows = [next(iter(by_member[member].values())) for member in members]
    colors = {
        "baseline": "#777777",
        "pooled": "#648FFF",
        "phase-specific": "#DC267F",
    }
    offsets = {"pooled": -0.10, "phase-specific": 0.10}
    markers = {"pooled": "s", "phase-specific": "D"}
    labels = {"pooled": r"Pooled $C$", "phase-specific": r"Phase-specific $C_i$"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    baseline_projection = [row["baseline_release_projection"] for row in baseline_rows]
    baseline_nino3 = [row["baseline_terminal_nino3_c"] for row in baseline_rows]
    axes[0].scatter(
        positions,
        baseline_projection,
        color=colors["baseline"],
        marker="o",
        facecolor="white",
        linewidth=1.4,
        s=42,
        zorder=3,
        label="Unnudged",
    )
    axes[1].scatter(
        positions,
        baseline_nino3,
        color=colors["baseline"],
        marker="o",
        facecolor="white",
        linewidth=1.4,
        s=42,
        zorder=3,
        label="Unnudged",
    )
    for case in CASE_NAMES:
        available = [
            (position, by_member[member][case], baseline_rows[index])
            for index, (position, member) in enumerate(
                zip(positions, members, strict=True)
            )
            if case in by_member[member]
        ]
        if not available:
            continue
        x = np.asarray([item[0] + offsets[case] for item in available])
        release_projection = [item[1]["release_projection"] for item in available]
        terminal_nino3 = [item[1]["terminal_nino3_c"] for item in available]
        axes[0].scatter(
            x,
            release_projection,
            color=colors[case],
            marker=markers[case],
            s=40,
            zorder=4,
            label=labels[case],
        )
        axes[1].scatter(
            x,
            terminal_nino3,
            color=colors[case],
            marker=markers[case],
            s=40,
            zorder=4,
            label=labels[case],
        )
        for position, row, baseline in available:
            axes[0].plot(
                [position, position + offsets[case]],
                [baseline["baseline_release_projection"], row["release_projection"]],
                color=colors[case],
                lw=0.8,
                alpha=0.30,
                zorder=1,
            )
            axes[1].plot(
                [position, position + offsets[case]],
                [baseline["baseline_terminal_nino3_c"], row["terminal_nino3_c"]],
                color=colors[case],
                lw=0.8,
                alpha=0.30,
                zorder=1,
            )
    axes[0].axhline(
        target_projection,
        color="black",
        lw=1.2,
        ls="--",
        label=f"Extreme {xai_label} coordinate",
        zorder=2,
    )
    axes[1].axhline(
        extreme_target_nino3_c,
        color="black",
        lw=1.2,
        ls="--",
        label=rf"Extreme event ({extreme_target_nino3_c:.2f} $^\circ$C)",
        zorder=2,
    )
    for panel, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.set_xticks(positions)
        axis.set_xlabel("Random test trajectory")
        axis.grid(axis="y", alpha=0.25)
        axis.text(0.015, 0.97, panel, transform=axis.transAxes, va="top")
    axes[0].set_ylabel(f"{xai_label} projection at release")
    axes[1].set_ylabel(r"Niño-3 after 10 months ($^\circ$C)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    with atomic_output_path(path, overwrite=True) as temporary:
        fig.savefig(temporary, bbox_inches="tight")
    plt.close(fig)


def _active_cases(run_manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return the cases intentionally included in this run.

    Manifests written before version 1.5 did not record this field and always
    expected both covariance arms, so the legacy fallback is exact.
    """

    raw = run_manifest.get("active_cases", list(CASE_NAMES))
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or value not in CASE_NAMES for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError("run manifest has invalid active_cases")
    return tuple(raw)


def _xai_target_record(run_manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the generic target record, with exact legacy-AGOP fallback."""

    target = run_manifest.get("xai_target", run_manifest.get("agop_target"))
    if not isinstance(target, dict):
        raise ValueError("run manifest has no XAI target record")
    return target


def _finalize(output_dir: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    target_record = _xai_target_record(run_manifest)
    target = float(target_record["target_projection"])
    run_identity = run_manifest.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("run manifest has no run identity")
    solver_settings = run_identity.get("solver_settings")
    if not isinstance(solver_settings, dict):
        raise ValueError("run identity has no solver settings")
    active_cases = _active_cases(run_manifest)
    for position, raw_index in enumerate(run_manifest["selection"]["selected_indices"]):
        release_index = int(raw_index)
        member = f"member_{position + 1:02d}_i{release_index}"
        member_dir = output_dir / "members" / member
        baseline_path = member_dir / "baseline_report.json"
        initial_path = member_dir / "initial_state_report.json"
        baseline = load_json(baseline_path) if baseline_path.is_file() else None
        initial = load_json(initial_path) if initial_path.is_file() else None
        for case in active_cases:
            case_dir = member_dir / case
            report_path = case_dir / "report.json"
            failure_path = case_dir / "failure.json"
            if not report_path.is_file():
                if failure_path.is_file():
                    failure = load_json(failure_path)
                    identity = failure.get("case_identity")
                    expected_identity: dict[str, Any] | None = None
                    if isinstance(initial, dict):
                        packed_state = initial.get("packed_state")
                        if isinstance(packed_state, dict) and isinstance(
                            packed_state.get("sha256"), str
                        ):
                            expected_identity = _case_identity(
                                run_identity_sha256=run_manifest["run_identity_sha256"],
                                release_index=release_index,
                                initial_state_sha256=packed_state["sha256"],
                                covariance_case=case,
                                solver_settings=solver_settings,
                            )
                    if (
                        failure.get("schema_version") != REPORT_SCHEMA_VERSION
                        or failure.get("status") != "failed"
                        or not isinstance(identity, dict)
                        or failure.get("case_identity_sha256") != sha256_json(identity)
                        or identity.get("run_identity_sha256")
                        != run_manifest["run_identity_sha256"]
                        or identity.get("release_input_index") != release_index
                        or identity.get("covariance_case") != case
                        or (
                            expected_identity is not None
                            and identity != expected_identity
                        )
                    ):
                        raise ValueError(
                            f"active failure has invalid identity: {failure_path}"
                        )
                    error = failure.get("error")
                    if not isinstance(error, dict):
                        raise ValueError(
                            f"active failure has no error record: {failure_path}"
                        )
                    failures.append(
                        {
                            "member": member,
                            "release_input_index": release_index,
                            "case": case,
                            "attempt": failure.get("attempt"),
                            "stage": failure.get("stage"),
                            "error_type": error.get("type"),
                            "error_message": error.get("message"),
                            "file": str(failure_path.relative_to(output_dir)),
                        }
                    )
                continue
            if not isinstance(baseline, dict) or not isinstance(initial, dict):
                failures.append(
                    {
                        "member": member,
                        "release_input_index": release_index,
                        "case": case,
                        "attempt": None,
                        "stage": "case_validation",
                        "error_type": "InvalidCompletedCaseReport",
                        "error_message": (
                            "completed case lacks its baseline or initial-state report"
                        ),
                        "file": str(report_path.relative_to(output_dir)),
                    }
                )
                continue
            packed_state = initial.get("packed_state")
            if not isinstance(packed_state, dict) or not isinstance(
                packed_state.get("sha256"), str
            ):
                raise ValueError(f"invalid initial-state report: {initial_path}")
            identity = _case_identity(
                run_identity_sha256=run_manifest["run_identity_sha256"],
                release_index=release_index,
                initial_state_sha256=packed_state["sha256"],
                covariance_case=case,
                solver_settings=solver_settings,
            )
            try:
                report = _load_completed_case(case_dir, identity)
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "member": member,
                        "release_input_index": release_index,
                        "case": case,
                        "attempt": None,
                        "stage": "case_validation",
                        "error_type": "InvalidCompletedCaseReport",
                        "error_message": str(error),
                        "file": str(report_path.relative_to(output_dir)),
                    }
                )
                continue
            if report is None:
                raise RuntimeError("case report disappeared during finalization")
            normalized = report["solver"]["normalized_kkt"]
            transfer = report["canonical_o3_frozen_control_transfer_audit"]
            rows.append(
                {
                    "member": member,
                    "release_input_index": release_index,
                    "case": case,
                    "baseline_release_projection": baseline["release_agop_projection"],
                    "release_projection": report["release"]["realized_agop_projection"],
                    "target_projection": target,
                    "baseline_terminal_nino3_c": baseline["terminal_nino3_c"],
                    "terminal_nino3_c": report["outcome_after_control_frozen"][
                        "terminal_nino3_c"
                    ],
                    "paired_terminal_nino3_change_c": report[
                        "outcome_after_control_frozen"
                    ]["paired_terminal_nino3_change_c"],
                    "canonical_o3_release_projection": transfer["canonical_o3"][
                        "release_agop_projection"
                    ],
                    "canonical_o3_release_constraint": transfer["canonical_o3"][
                        "release_constraint_value"
                    ],
                    "canonical_o3_terminal_nino3_c": transfer["canonical_o3"][
                        "terminal_nino3_c"
                    ],
                    "matched_o0_minus_canonical_o3_terminal_nino3_c": transfer[
                        "matched_o0_minus_canonical_o3"
                    ]["terminal_nino3_c"],
                    "cnn_forecast_c": report["release"]["cnn_forecast_c"],
                    "solver_success": report["solver"]["success"],
                    "accepted_update_count": report["solver"]["accepted_update_count"],
                    "rejected_update_count": report["solver"]["rejected_update_count"],
                    "normalized_primal_violation": normalized[
                        "primal_violation_over_initial_projection_gap"
                    ],
                    "normalized_stationarity": normalized[
                        "stationarity_over_kkt_term_norm"
                    ],
                    "normalized_complementarity": normalized[
                        "complementarity_over_squared_action"
                    ],
                }
            )
    csv_path = output_dir / "summary.csv"
    figure_path = output_dir / "summary.pdf"
    _write_csv(csv_path, rows)
    extreme_target_nino3_c = float(target_record["extreme_target_nino3_c"])
    _write_figure(
        figure_path,
        rows,
        target_projection=target,
        extreme_target_nino3_c=extreme_target_nino3_c,
        xai_label=str(target_record.get("method", {}).get("label", "AGOP")),
    )
    aggregates: dict[str, Any] = {}
    for case in active_cases:
        subset = [row for row in rows if row["case"] == case]
        if not subset:
            continue
        changes = np.asarray(
            [row["paired_terminal_nino3_change_c"] for row in subset],
            dtype=np.float64,
        )
        terminal = np.asarray(
            [row["terminal_nino3_c"] for row in subset], dtype=np.float64
        )
        aggregates[case] = {
            "count": len(subset),
            "paired_terminal_nino3_change_c": {
                "mean": float(np.mean(changes)),
                "median": float(np.median(changes)),
                "minimum": float(np.min(changes)),
                "maximum": float(np.max(changes)),
                "positive_count": int(np.count_nonzero(changes > 0.0)),
                "negative_count": int(np.count_nonzero(changes < 0.0)),
            },
            "terminal_nino3_c": {
                "mean": float(np.mean(terminal)),
                "minimum": float(np.min(terminal)),
                "maximum": float(np.max(terminal)),
            },
            "accepted_update_count": {
                "minimum": int(min(row["accepted_update_count"] for row in subset)),
                "maximum": int(max(row["accepted_update_count"] for row in subset)),
            },
            "maximum_normalized_stationarity": float(
                max(row["normalized_stationarity"] for row in subset)
            ),
            "maximum_normalized_complementarity": float(
                max(row["normalized_complementarity"] for row in subset)
            ),
        }
    expected_rows = len(run_manifest["selection"]["selected_indices"]) * len(
        active_cases
    )
    pending_rows = expected_rows - len(rows) - len(failures)
    if pending_rows < 0:
        raise RuntimeError("case accounting exceeds the run manifest")
    summary_status = (
        "complete"
        if len(rows) == expected_rows and not failures and pending_rows == 0
        else "partial_with_failures"
        if failures
        else "partial"
    )
    updated_utc = datetime.now(UTC).isoformat()
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": summary_status,
        "run_identity_sha256": run_manifest["run_identity_sha256"],
        "active_cases": list(active_cases),
        "completed_case_count": len(rows),
        "failed_case_count": len(failures),
        "pending_case_count": pending_rows,
        "expected_case_count": expected_rows,
        "extreme_target_nino3_c": extreme_target_nino3_c,
        "aggregates": aggregates,
        "rows": rows,
        "failures": failures,
        "csv": {"file": csv_path.name, "sha256": sha256_file(csv_path)},
        "pdf": (
            None
            if not figure_path.is_file()
            else {"file": figure_path.name, "sha256": sha256_file(figure_path)}
        ),
        "updated_utc": updated_utc,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary, overwrite=True)
    finalized_manifest = dict(run_manifest)
    finalized_manifest["status"] = summary_status
    finalized_manifest["finalization"] = {
        "completed_case_count": len(rows),
        "failed_case_count": len(failures),
        "pending_case_count": pending_rows,
        "expected_case_count": expected_rows,
        "summary_file": summary_path.name,
        "summary_sha256": sha256_file(summary_path),
        "updated_utc": updated_utc,
    }
    write_json(output_dir / "run_manifest.json", finalized_manifest, overwrite=True)
    return summary


def _load_scaled_initial_dual(
    path: Path,
    *,
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load, scale, and fully identify an external SQP dual warm start."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("initial dual scale must be finite and positive")
    with np.load(path, allow_pickle=False) as archive:
        if "dual_variables" not in archive.files:
            raise ValueError("--initial-dual-npz has no dual_variables array")
        raw = np.asarray(archive["dual_variables"], dtype=np.float64).copy()
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError(
            "--initial-dual-npz dual_variables must be a finite rank-two array"
        )
    scaled = float(scale) * raw
    if not np.isfinite(scaled).all():
        raise ValueError("--initial-dual-scale produced nonfinite dual variables")
    provenance = {
        "initial_dual_npz_sha256": sha256_file(path),
        "initial_dual_raw_variables_sha256": sha256_array(raw),
        "initial_dual_scale": float(scale),
        "initial_dual_scaled_variables_sha256": sha256_array(scaled),
    }
    return scaled, provenance


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.monotonic()
    data_dir = _resolve(args.data_dir)
    artifacts_dir = _resolve(args.artifacts_dir)
    agop_dir = _resolve(args.agop_benchmark_dir)
    generation_workspace = _resolve(args.generation_workspace)
    validation_dir = _resolve(args.adjoint_validation_dir)
    primal_executable = _resolve(args.primal_executable)
    tangent_executable = _resolve(args.tangent_executable)
    differentiable_primal_executable = _resolve(args.differentiable_primal_executable)
    differentiable_tangent_executable = _resolve(args.differentiable_tangent_executable)
    adjoint_executable = _resolve(args.adjoint_executable)
    initial_dual_path = (
        None if args.initial_dual_npz is None else _resolve(args.initial_dual_npz)
    )
    initial_dual_variables: np.ndarray | None = None
    initial_dual_provenance: dict[str, Any] = {
        "initial_dual_npz_sha256": None,
        "initial_dual_raw_variables_sha256": None,
        "initial_dual_scale": None,
        "initial_dual_scaled_variables_sha256": None,
    }
    if initial_dual_path is not None:
        initial_dual_variables, initial_dual_provenance = _load_scaled_initial_dual(
            initial_dual_path,
            scale=args.initial_dual_scale,
        )
    elif args.initial_dual_scale != 1.0:
        raise ValueError("--initial-dual-scale requires --initial-dual-npz")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and args.overwrite:
        if output_dir == REPOSITORY_ROOT or REPOSITORY_ROOT not in output_dir.parents:
            raise ValueError("refusing to replace a broad output directory")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = ZCData(
        data_dir,
        input_profile="core4",
        verify_checksums=not args.skip_data_checksums,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("the direct experiment requires fresh zc-v3")
    if data.steps_per_month != 3 or data.steps_per_year != 36:
        raise ValueError("the direct experiment requires three steps per month")
    specification = ExperimentSpec(
        architecture="cnn",
        lead_months=10,
        train_years=10_000.0,
        seed=42,
        input_profile="core4",
    )
    experiment = load_experiment(data, artifacts_dir, specification, device="cpu")
    fixed = data.fixed_supervised_split(10)
    canonical_training = np.arange(fixed.train_block[0], fixed.train_block[1])
    if not np.array_equal(experiment.standardization_inputs, canonical_training):
        raise ValueError("CNN normalization no longer uses all 10,000 training years")
    if not np.array_equal(experiment.fit_inputs, fixed.train_inputs):
        raise ValueError("CNN fit inputs no longer equal the fixed training predictors")
    target_event = next(
        item
        for item in data.metadata["event_restart_checkpoints"]
        if item["label"] == args.target_event
    )
    target_event_index = int(target_event["input_index"])
    release_phase = target_event_index % data.steps_per_year
    control_phases = control_phases_for_release_phase(
        release_phase,
        steps_per_year=data.steps_per_year,
    )
    covariance_manifest = _resolve(
        default_covariance_manifest(
            control_phases,
            covariance_policy=args.covariance_policy,
        )
        if args.covariance_manifest is None
        else args.covariance_manifest
    )
    test_interval = tuple(
        int(value) for value in data.metadata["chronological_split"]["test"]
    )
    if args.trajectory == "authentic-extreme":
        if args.projection_change_multiple is None:
            raise ValueError(
                "--trajectory authentic-extreme requires --projection-change-multiple"
            )
        if args.members:
            raise ValueError("--member cannot be used with --authentic-extreme")
        selected = np.asarray([target_event_index], dtype=np.int64)
        eligible = selected.copy()
        requested = selected
    else:
        if args.projection_change_multiple is not None:
            raise ValueError(
                "--projection-change-multiple requires --trajectory authentic-extreme"
            )
        selected, eligible = select_uniform_release_indices(
            fixed.test_inputs,
            test_interval=test_interval,
            event_index=target_event_index,
            steps_per_year=data.steps_per_year,
            seed=args.selection_seed,
        )
        _validate_seed_42_release_sample(
            selected,
            target_event=args.target_event,
            seed=args.selection_seed,
        )
        requested = selected
        if args.members:
            requested = np.asarray(args.members, dtype=np.int64)
            if np.unique(requested).size != requested.size:
                raise ValueError("--member values must not repeat")
            unknown = np.setdiff1d(requested, selected)
            if unknown.size:
                raise ValueError(f"--member is outside the fixed sample: {unknown}")
    cases = covariance_cases(
        args.case,
        covariance_policy=args.covariance_policy,
    )

    factor = None
    factor_provenance: dict[str, Any] = {}
    agop_identity: dict[str, Any] | None = None
    if args.xai_method == "AGOP":
        factor, factor_provenance = fresh_xai_benchmark.load_validated_full_agop_factor(
            agop_dir,
            data=data,
            experiment=experiment,
            fixed_training_inputs=fixed.train_inputs,
        )
        agop_identity = {
            key: value
            for key, value in factor_provenance.items()
            if key != "load_and_validation_seconds_this_invocation"
        }
    event_standardized = data.load_inputs(
        np.asarray([target_event_index]), standardizer=experiment.standardizer
    )[0].astype(np.float64)
    extreme_target_nino3_c = float(
        data.load_targets(np.asarray([target_event_index]), lead_steps=30)[0]
    )
    xai_target = build_direct_xai_target(
        args.xai_method,
        data=data,
        experiment=experiment,
        event_standardized=event_standardized,
        agop_factor=factor,
        device="cpu",
    )
    xai_target_artifact = _ensure_xai_target_artifact(
        output_dir,
        event_standardized=event_standardized,
        target=xai_target,
    )
    base_direction = xai_target.raw_spatial_direction
    natural_event_projection = xai_target.raw_event_projection
    projection_change_multiple = args.projection_change_multiple
    if projection_change_multiple is None:
        direction = xai_target.oriented_spatial_direction
        target_projection = xai_target.oriented_event_projection
        target_projection_in_base_direction = natural_event_projection
    else:
        dose_orientation = 1.0 if projection_change_multiple > 0.0 else -1.0
        direction = dose_orientation * xai_target.oriented_spatial_direction
        target_projection_in_base_direction = (
            1.0 + projection_change_multiple
        ) * natural_event_projection
        target_projection = dose_orientation * (
            (1.0 + projection_change_multiple) * xai_target.oriented_event_projection
        )

    observation_chain = FrozenCore4ObservationChain(experiment.standardizer)
    layout = load_independent_control_layout(observation_chain.layout.manifest_path)
    if args.covariance_policy == ANNUAL_SHARED_COVARIANCE_POLICY:
        pooled_covariance, covariance_document = _load_annual_dense_covariance(
            covariance_manifest,
            data_metadata_sha256=data.metadata_sha256,
            state_manifest_sha256=observation_chain.layout.manifest_sha256,
            state_size=layout.compact_size,
        )
        covariance_cache_identity, covariance_cache_runtime = (
            _dense_covariance_runtime(
                covariance_document,
                requested_centered_cache_gib=(
                    args.covariance_centered_cache_gib
                ),
            )
        )
        phase_specific_covariance: tuple[Any, ...] = ()
        covariance_training_phases = ANNUAL_PHASES
        covariance_backend = "precompiled_dense_covariance"
        covariance_source_phase_count = len(ANNUAL_PHASES)
    elif args.covariance_policy == LEGACY_EVENT_WINDOW_COVARIANCE_POLICY:
        covariance = NativeCovarianceBundle.load(
            covariance_manifest, verify_hashes=True
        )
        if (
            covariance.phase_offsets.tolist() != list(control_phases)
            or covariance.phase_factors[0].sample_count
            != ANNUAL_COVARIANCE_SAMPLE_COUNT_PER_PHASE
            or covariance.phase_factors[0].state_size != layout.compact_size
        ):
            raise ValueError(
                "legacy native covariance does not match the nine phases "
                "immediately before target-event release phase "
                f"{release_phase}: expected {list(control_phases)}"
            )
        source_covariance = covariance.manifest.get("source_capture", {})
        if source_covariance.get("data_metadata_sha256") != data.metadata_sha256:
            raise ValueError("native covariance belongs to another zc-v3 data set")
        if (
            source_covariance.get("state_manifest_sha256")
            != observation_chain.layout.manifest_sha256
        ):
            raise ValueError(
                "native covariance uses another independent-state layout"
            )
        covariance, covariance_cache_identity, covariance_cache_runtime = (
            _configure_centered_covariance_cache(
                covariance,
                requested_gib=args.covariance_centered_cache_gib,
            )
        )
        pooled_covariance = covariance.pooled_factor
        phase_specific_covariance = tuple(
            covariance.phase_factor(phase) for phase in control_phases
        )
        covariance_training_phases = control_phases
        covariance_backend = "legacy_centered_sample_factors"
        covariance_source_phase_count = len(control_phases)
    else:  # pragma: no cover - argparse and covariance_cases reject this first
        raise ValueError(f"unknown covariance policy: {args.covariance_policy}")

    runtime_source = validation_dir / "verified_inputs" / "source"
    kernel_executable = validation_dir / "verified_inputs" / "zc_kernel_replay"
    runtime_template = validation_dir / "neutral_member_04_040step" / "kernel"
    dynamic_provenance = established._validate_dynamic_bridge_provenance(
        validation_dir,
        kernel_executable=kernel_executable,
        primal_executable=primal_executable,
        tangent_executable=tangent_executable,
        runtime_template=runtime_template,
    )
    reverse_provenance = _validate_reverse_build(
        adjoint_executable,
        differentiable_primal_executable,
        differentiable_tangent_executable,
    )
    solver_settings = {
        "maximum_iterations": args.maximum_iterations,
        "continuation_fractions": list(args.continuation_fractions),
        "scale_continuation_warm_start": args.scale_continuation_warm_start,
        "adaptive_continuation": args.adaptive_continuation,
        "direct_exact_target_first": args.direct_exact_target_first,
        "zero_dual_restart_after_warm_rejection": (
            args.zero_dual_restart_after_warm_rejection
        ),
        "adaptive_minimum_fraction_step": (args.adaptive_minimum_fraction_step),
        "maximum_adaptive_subdivisions": args.maximum_adaptive_subdivisions,
        "known_active_boundary": True,
        "constraint_tolerance": args.constraint_tolerance,
        "stationarity_tolerance": args.stationarity_tolerance,
        "complementarity_tolerance": args.complementarity_tolerance,
        "relative_stationarity_tolerance": (args.relative_stationarity_tolerance),
        "relative_complementarity_tolerance": (args.relative_complementarity_tolerance),
        "initial_trust_radius": args.initial_trust_radius,
        "maximum_trust_radius": args.maximum_trust_radius,
        "covariance_block_rows": args.covariance_block_rows,
        "covariance_policy": args.covariance_policy,
        "covariance_backend": covariance_backend,
        "covariance_source_phase_count": covariance_source_phase_count,
        "covariance_centered_cache": covariance_cache_identity,
    }
    scientific_source_sha256 = _scientific_source_provenance()
    software_versions = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
    }
    run_identity = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "data_metadata_sha256": data.metadata_sha256,
        "model_checkpoint_sha256": experiment.checkpoint_sha256,
        "normalization_sha256": sha256_file(
            experiment.artifact_dir / "normalization.npz"
        ),
        "xai_method": xai_target.provenance,
        "xai_target_artifact_sha256": xai_target_artifact["sha256"],
        "agop": agop_identity,
        "base_xai_direction_sha256": sha256_array(base_direction),
        "xai_direction_sha256": sha256_array(direction),
        "target_event_label": args.target_event,
        "extreme_event_input_index": target_event_index,
        "trajectory": args.trajectory,
        "projection_change_multiple": projection_change_multiple,
        **initial_dual_provenance,
        # Retain the original identity key as an explicit alias for the exact
        # array supplied to SQP.  Existing journal linkage and downstream
        # validators use this name; the raw/scaled fields above remove the
        # ambiguity introduced by an external multiplier.
        "initial_dual_variables_sha256": initial_dual_provenance[
            "initial_dual_scaled_variables_sha256"
        ],
        "natural_event_projection_in_base_direction": natural_event_projection,
        "target_projection_in_base_direction": target_projection_in_base_direction,
        "target_projection": target_projection,
        "extreme_target_nino3_c": extreme_target_nino3_c,
        "covariance_manifest": str(covariance_manifest),
        "covariance_manifest_sha256": sha256_file(covariance_manifest),
        "covariance_policy": args.covariance_policy,
        "covariance_backend": covariance_backend,
        "covariance_training_phases": list(covariance_training_phases),
        "primal_executable_sha256": sha256_file(primal_executable),
        "differentiable_primal_executable_sha256": sha256_file(
            differentiable_primal_executable
        ),
        "differentiable_tangent_executable_sha256": sha256_file(
            differentiable_tangent_executable
        ),
        "adjoint_executable_sha256": sha256_file(adjoint_executable),
        "state_manifest_sha256": observation_chain.layout.manifest_sha256,
        "scientific_python_source_sha256": scientific_source_sha256,
        "software_versions": software_versions,
        "selection_seed": args.selection_seed,
        "selection_reference_event_label": args.target_event,
        "selection_reference_event_input_index": target_event_index,
        "active_cases": list(cases),
        "control_steps": CONTROL_STEPS,
        "control_phases": list(control_phases),
        "release_boundaries": list(RELEASE_BOUNDARIES),
        "total_steps": TOTAL_STEPS,
        "solver_settings": solver_settings,
        "projection_continuation_acceptance_policy": (
            _projection_continuation_acceptance_policy()
        ),
    }
    identity_sha = sha256_json(run_identity)
    run_manifest_path = output_dir / "run_manifest.json"
    selection = {
        "rule": (
            "authentic target-event trajectory"
            if args.trajectory == "authentic-extreme"
            else (
                "uniform without replacement from boundary-eligible "
                "same-integer-phase fixed test inputs"
            )
        ),
        "seed": args.selection_seed,
        "uniform_without_replacement": args.trajectory == "uniform-cohort",
        "sorted_for_execution": True,
        "no_neutral_filter": True,
        "no_state_value_filter": args.trajectory == "uniform-cohort",
        "no_future_outcome_filter": args.trajectory == "uniform-cohort",
        "future_targets_used_for_selection": args.trajectory == "authentic-extreme",
        "cnn_forecasts_used_for_selection": False,
        "candidate_count": int(eligible.size),
        "candidate_indices_sha256": sha256_array(eligible),
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": sha256_array(selected),
        "test_half_open_interval": list(test_interval),
        "reference_event_label": args.target_event,
        "reference_event_input_index": target_event_index,
        "event_phase_modulo_36": release_phase,
        "target_event_label": args.target_event,
        "target_event_phase_modulo_36": release_phase,
    }
    run_manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "prepared",
        "run_identity": run_identity,
        "run_identity_sha256": identity_sha,
        "active_cases": list(cases),
        "selection": selection,
        "xai_target": {
            "method": {
                "key": xai_target.provenance["key"],
                "label": xai_target.provenance["label"],
            },
            "target_event_label": args.target_event,
            "extreme_event_input_index": target_event_index,
            "event_phase_modulo_36": target_event_index % data.steps_per_year,
            "direction_sha256": sha256_array(direction),
            "base_direction_sha256": sha256_array(base_direction),
            "orientation_multiplier": xai_target.orientation_multiplier,
            "orientation_policy": xai_target.provenance["orientation_policy"],
            "direction_orientation": (
                xai_target.orientation_multiplier
                if projection_change_multiple is None
                else float(np.sign(projection_change_multiple))
                * xai_target.orientation_multiplier
            ),
            "trajectory": args.trajectory,
            "projection_change_multiple": projection_change_multiple,
            "natural_event_projection_in_base_direction": natural_event_projection,
            "raw_event_projection": xai_target.raw_event_projection,
            "oriented_event_projection": xai_target.oriented_event_projection,
            "target_projection_in_base_direction": target_projection_in_base_direction,
            "target_projection": target_projection,
            "extreme_target_nino3_c": extreme_target_nino3_c,
            "definition": (
                "The authentic-extreme signed-dose target is "
                "e^T(x_release-x_extreme)=c e^T x_extreme in standardized "
                "core4 spatial coordinates; the inequality direction is "
                "reversed for c<0 and the annual phase coordinates are excluded."
                if projection_change_multiple is not None
                else (
                    "e^T x_extreme in standardized core4 spatial coordinates; "
                    "annual phase coordinates excluded; e is event-oriented "
                    "using the recorded sign policy"
                )
            ),
        },
        "provenance": {
            "dynamic_primal": dynamic_provenance,
            "matched_o0_primal_tangent_reverse": reverse_provenance,
            "covariance_manifest_sha256": sha256_file(covariance_manifest),
            "agop_load": factor_provenance if args.xai_method == "AGOP" else None,
            "xai_reference_arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256": sha256_array(value),
                }
                for key, value in xai_target.reference_arrays.items()
            },
            "xai_target_artifact": xai_target_artifact,
        },
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "software_versions": software_versions,
        },
        "covariance_runtime": covariance_cache_runtime,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    if args.xai_method == "AGOP":
        # Preserve the established public key consumed by the nonlinear AGOP
        # dose-response pipeline while all new readers use ``xai_target``.
        run_manifest["agop_target"] = dict(run_manifest["xai_target"])
    if run_manifest_path.is_file():
        existing = load_json(run_manifest_path)
        if (
            existing.get("run_identity_sha256") != identity_sha
            or existing.get("run_identity") != run_identity
        ):
            raise ValueError("existing output directory has another run identity")
        run_manifest = existing
    else:
        write_json(run_manifest_path, run_manifest, overwrite=False)

    members_root = output_dir / "members"
    members_root.mkdir(exist_ok=True)
    for release_index_value in requested:
        release_index = int(release_index_value)
        position = int(np.flatnonzero(selected == release_index)[0])
        member_label = f"member_{position + 1:02d}_i{release_index}"
        member_dir = members_root / member_label
        packed_path, initial_report = _ensure_initial_state(
            member_dir,
            release_index=release_index,
            generation_workspace=generation_workspace,
            runtime_source=runtime_source,
            kernel_executable=kernel_executable,
        )
        initial_state = observation_chain.read_state(packed_path)
        expected_nt = 3_600 + release_index - CONTROL_STEPS
        nt_segment = observation_chain.layout.integer_segments["NT"]
        if int(initial_state.integers[nt_segment.start]) != expected_nt:
            raise RuntimeError("materialized initial state has the wrong NT")
        print(f"{member_label}: authentic baseline", flush=True)
        with (
            FortranOneStepRunner(
                runtime_template,
                primal_executable=primal_executable,
                manifest_path=observation_chain.layout.manifest_path,
                timeout_seconds=args.timeout_seconds,
            ) as canonical_runner,
            FortranOneStepRunner(
                runtime_template,
                primal_executable=differentiable_primal_executable,
                manifest_path=observation_chain.layout.manifest_path,
                timeout_seconds=args.timeout_seconds,
            ) as primal_runner,
            FortranOneStepAdjointRunner(
                runtime_template,
                adjoint_executable,
                manifest_path=observation_chain.layout.manifest_path,
                timeout_seconds=args.timeout_seconds,
            ) as adjoint_runner,
        ):
            canonical_baseline = _build_zero_control_path_with_runner(
                initial_state, canonical_runner, total_steps=TOTAL_STEPS
            )
            zeros = np.zeros((CONTROL_STEPS, layout.compact_size), dtype=np.float64)
            canonical_explicit_zero = replay_native_interventions_with_runner(
                initial_state,
                zeros,
                layout,
                canonical_runner,
                total_steps=TOTAL_STEPS,
            )
            verify_zero_control_replay(canonical_baseline, canonical_explicit_zero)
            baseline = _build_zero_control_path_with_runner(
                initial_state, primal_runner, total_steps=TOTAL_STEPS
            )
            explicit_zero = replay_native_interventions_with_runner(
                initial_state,
                zeros,
                layout,
                primal_runner,
                total_steps=TOTAL_STEPS,
            )
            verify_zero_control_replay(baseline, explicit_zero)
            cross_build_audit = _cross_build_zero_audit(baseline, canonical_baseline)
            canonical_release = observation_chain.forward(
                canonical_baseline.states[9], canonical_baseline.states[10]
            )
            baseline_release = observation_chain.forward(
                baseline.states[9], baseline.states[10]
            )
            expected_release = data.load_inputs(
                np.asarray([release_index]), standardizer=experiment.standardizer
            )[0]
            if not np.array_equal(canonical_release, expected_release):
                raise RuntimeError(
                    "canonical O3 release does not match processed zc-v3"
                )
            baseline_final = float(nino3_from_real_state(baseline.states[40].real32))
            canonical_final = float(
                nino3_from_real_state(canonical_baseline.states[40].real32)
            )
            expected_final = float(data.target[release_index + 30])
            actual_bits = np.float32(canonical_final).view(np.uint32)
            expected_bits = np.float32(expected_final).view(np.uint32)
            if actual_bits != expected_bits:
                raise RuntimeError("canonical O3 ten-month Nino-3 does not match zc-v3")
            release_difference = baseline_release[:-2].astype(
                np.float64
            ) - canonical_release[:-2].astype(np.float64)
            release_nino3 = float(nino3_from_real_state(baseline.states[10].real32))
            canonical_release_nino3 = float(
                nino3_from_real_state(canonical_baseline.states[10].real32)
            )
            scientific_cross_build = {
                "standardized_release_max_abs": float(
                    np.max(np.abs(release_difference), initial=0.0)
                ),
                "standardized_release_l2": float(np.linalg.norm(release_difference)),
                "xai_projection_signed_difference": float(
                    direction @ release_difference
                ),
                "agop_projection_signed_difference": float(
                    direction @ release_difference
                ),
                "release_nino3_signed_difference_c": (
                    release_nino3 - canonical_release_nino3
                ),
                "terminal_nino3_signed_difference_c": (
                    baseline_final - canonical_final
                ),
            }
            scientific_limits = {
                "standardized_release_max_abs": 1e-3,
                "standardized_release_l2": 1e-2,
                "xai_projection_signed_difference": 1e-4,
                "agop_projection_signed_difference": 1e-4,
                "release_nino3_signed_difference_c": 1e-4,
                "terminal_nino3_signed_difference_c": 1e-4,
            }
            if any(
                abs(scientific_cross_build[key]) > limit
                for key, limit in scientific_limits.items()
            ):
                raise RuntimeError(
                    "O0 and canonical O3 zero paths disagree in scientific "
                    f"coordinates: measured={scientific_cross_build}, "
                    f"limits={scientific_limits}"
                )
            cross_build_audit.update(scientific_cross_build)
            cross_build_audit.update(
                {f"hard_limit_{key}": value for key, value in scientific_limits.items()}
            )
            baseline_projection = float(direction @ baseline_release[:-2])
            baseline_projection_in_base_direction = float(
                base_direction @ baseline_release[:-2]
            )
            initial_gap = target_projection - baseline_projection
            baseline_report = {
                "schema_version": 1,
                "status": "complete",
                "release_input_index": release_index,
                "preconditioning_start_input_index": release_index - CONTROL_STEPS,
                "target_input_index": release_index + 30,
                "initial_state_report_sha256": sha256_file(
                    member_dir / "initial_state_report.json"
                ),
                "release_bitwise_equal_to_zc_v3": True,
                "terminal_nino3_bitwise_equal_to_zc_v3": True,
                "canonical_explicit_zero_bitwise_equal_at_all_states_and_tapes": (True),
                "differentiable_explicit_zero_bitwise_equal_at_all_states_and_tapes": (
                    True
                ),
                "differentiable_o0_vs_canonical_o3_zero_path": cross_build_audit,
                "release_xai_projection": baseline_projection,
                "release_agop_projection": baseline_projection,
                "release_projection_in_raw_xai_direction": (
                    baseline_projection_in_base_direction
                ),
                "release_projection_in_base_agop_direction": (
                    baseline_projection_in_base_direction
                ),
                "target_projection_in_raw_xai_direction": (
                    target_projection_in_base_direction
                ),
                "target_projection_in_base_agop_direction": (
                    target_projection_in_base_direction
                ),
                "projection_change_multiple": projection_change_multiple,
                "initial_projection_gap": initial_gap,
                "release_nino3_c": float(release_nino3),
                "cnn_forecast_c": _model_forecast(experiment, baseline_release),
                "terminal_nino3_c": baseline_final,
                "canonical_o3_release_agop_projection": float(
                    direction @ canonical_release[:-2]
                ),
                "canonical_o3_release_nino3_c": float(canonical_release_nino3),
                "canonical_o3_terminal_nino3_c": canonical_final,
                "fortran_wall_seconds": (
                    canonical_baseline.wall_seconds
                    + canonical_explicit_zero.wall_seconds
                    + baseline.wall_seconds
                    + explicit_zero.wall_seconds
                ),
            }
            write_json(
                member_dir / "baseline_report.json",
                baseline_report,
                overwrite=True,
            )
            zero_release = ControlledReplay(
                states=explicit_zero.states[:11],
                step_inputs=explicit_zero.step_inputs[:10],
                tapes=explicit_zero.tapes[:10],
                applied_real32_increments=explicit_zero.applied_real32_increments[:10],
                wall_seconds=0.0,
            )
            oracle = ReleaseConstraintOracle(
                initial_state=initial_state,
                zero_release_replay=zero_release,
                layout=layout,
                observation_chain=observation_chain,
                direction=direction,
                target_projection=target_projection,
                primal_runner=primal_runner,
                adjoint_runner=adjoint_runner,
            )
            for case in cases:
                identity = _case_identity(
                    run_identity_sha256=identity_sha,
                    release_index=release_index,
                    initial_state_sha256=initial_report["packed_state"]["sha256"],
                    covariance_case=case,
                    solver_settings=solver_settings,
                )
                case_dir = member_dir / case
                cached = _load_completed_case(case_dir, identity)
                if cached is not None:
                    _clear_active_case_failure(case_dir)
                    print(f"{member_label}/{case}: validated cached result", flush=True)
                    continue
                print(
                    f"{member_label}/{case}: nonlinear covariance-action SQP",
                    flush=True,
                )
                covariance_argument: Any
                if case == "pooled":
                    covariance_argument = pooled_covariance
                else:
                    covariance_argument = phase_specific_covariance
                counters_before = (
                    oracle.forward_replays,
                    oracle.reverse_sweeps,
                    oracle.reverse_wall_seconds,
                )
                solve_started = time.monotonic()
                stage = "projection_continuation_sqp"
                result_for_failure: NativeScalarSQPResult | None = None
                continuation_records: list[dict[str, Any]] = []
                continuation_journal = DurableContinuationJournal(
                    case_dir,
                    run_identity=run_identity,
                    case_identity=identity,
                    covariance_case=case,
                    baseline_projection=baseline_projection,
                    target_projection=target_projection,
                    requested_fractions=args.continuation_fractions,
                    resume_enabled=args.resume_continuation,
                )
                try:
                    result, continuation_records = _solve_projection_continuation(
                        oracle,
                        covariance_argument,
                        covariance_case=case,
                        baseline_projection=baseline_projection,
                        target_projection=target_projection,
                        fractions=args.continuation_fractions,
                        solver_settings=solver_settings,
                        initial_dual_variables=initial_dual_variables,
                        initial_dual_scale=(
                            1.0
                            if initial_dual_path is None
                            else float(args.initial_dual_scale)
                        ),
                        initial_dual_raw_sha256=initial_dual_provenance[
                            "initial_dual_raw_variables_sha256"
                        ],
                        continuation_journal=continuation_journal,
                    )
                    solve_returned = time.monotonic()
                    cumulative_stage_runtime = _continuation_runtime_totals(
                        continuation_records
                    )
                    resumed_stage_runtime = _continuation_runtime_totals(
                        continuation_records[
                            : continuation_journal.resumed_attempt_count
                        ]
                    )
                    result_for_failure = result
                    stage = "kkt_reporting"
                    (
                        stationarity_reference,
                        gradient_covariance_norm,
                        reporting_covariance_passes,
                    ) = _stationarity_reference_norm(
                        result,
                        covariance_argument,
                        block_rows=args.covariance_block_rows,
                    )
                    stage = "frozen_full_replay"
                    frozen = replay_native_interventions_with_runner(
                        initial_state,
                        result.native_interventions,
                        layout,
                        primal_runner,
                        total_steps=TOTAL_STEPS,
                    )
                    stage = "canonical_o3_frozen_full_replay"
                    canonical_frozen = replay_native_interventions_with_runner(
                        initial_state,
                        result.native_interventions,
                        layout,
                        canonical_runner,
                        total_steps=TOTAL_STEPS,
                    )
                    stage = "case_publication"
                    report = _publish_case(
                        case_dir,
                        identity=identity,
                        result=result,
                        full_replay=frozen,
                        canonical_full_replay=canonical_frozen,
                        baseline=baseline,
                        canonical_baseline=canonical_baseline,
                        baseline_release=baseline_release,
                        baseline_final_nino3=baseline_final,
                        canonical_baseline_final_nino3=canonical_final,
                        observation_chain=observation_chain,
                        experiment=experiment,
                        direction=direction,
                        target_projection=target_projection,
                        initial_projection_gap=initial_gap,
                        stationarity_reference_norm=stationarity_reference,
                        constraint_gradient_covariance_norm=(gradient_covariance_norm),
                        reporting_covariance_operator_calls=(
                            reporting_covariance_passes
                        ),
                        solver_settings=solver_settings,
                        covariance_case=case,
                        oracle_runtime={
                            "covariance_cache": covariance_cache_runtime,
                            "release_only_forward_replays": cumulative_stage_runtime[
                                "release_only_forward_replays"
                            ],
                            "release_only_reverse_sweeps": cumulative_stage_runtime[
                                "release_only_reverse_sweeps"
                            ],
                            "reverse_fortran_wall_seconds": cumulative_stage_runtime[
                                "reverse_fortran_wall_seconds"
                            ],
                            "current_invocation_release_only_forward_replays": (
                                oracle.forward_replays - counters_before[0]
                            ),
                            "current_invocation_release_only_reverse_sweeps": (
                                oracle.reverse_sweeps - counters_before[1]
                            ),
                            "current_invocation_reverse_fortran_wall_seconds": (
                                oracle.reverse_wall_seconds - counters_before[2]
                            ),
                        },
                        continuation_stages=continuation_records,
                        durable_continuation_checkpoint=(
                            continuation_journal.audit_record()
                        ),
                    )
                    report_prepared = time.monotonic()
                    post_solver_wall_seconds = report_prepared - solve_returned
                    solver_total_wall_seconds = float(
                        cumulative_stage_runtime["solver_wall_seconds"]
                    )
                    report["runtime"].update(
                        {
                            "solver_total_wall_seconds": solver_total_wall_seconds,
                            "resumed_checkpointed_solver_wall_seconds": float(
                                resumed_stage_runtime["solver_wall_seconds"]
                            ),
                            "post_solver_replay_and_report_preparation_wall_seconds": (
                                post_solver_wall_seconds
                            ),
                            "total_wall_seconds": (
                                solver_total_wall_seconds + post_solver_wall_seconds
                            ),
                            "current_invocation_wall_seconds_before_atomic_report_"
                            "publication": report_prepared - solve_started,
                            "total_wall_seconds_definition": (
                                "sum of every checkpointed continuation-attempt "
                                "solver_wall_seconds exactly once, including "
                                "rejected attempts, plus the successful "
                                "invocation's post-solver KKT reporting, frozen "
                                "replays, artifact write, and report preparation; "
                                "atomic report fsync is excluded"
                            ),
                        }
                    )
                    write_json(case_dir / "report.json", report, overwrite=False)
                    _clear_active_case_failure(case_dir)
                except Exception as error:
                    diagnostic_gap = initial_gap
                    if isinstance(error, ProjectionContinuationFailure):
                        result_for_failure = error.result
                        continuation_records = list(error.stage_records)
                        diagnostic_gap = error.stage_fraction * initial_gap
                        stage = (
                            "projection_continuation_stage_"
                            f"{len(continuation_records)}_solver_publication_gates"
                        )
                    runtime = {
                        "covariance_cache": covariance_cache_runtime,
                        "solver_wall_seconds_before_failure": (
                            time.monotonic() - solve_started
                        ),
                        "release_only_forward_replays": (
                            oracle.forward_replays - counters_before[0]
                        ),
                        "release_only_reverse_sweeps": (
                            oracle.reverse_sweeps - counters_before[1]
                        ),
                        "reverse_fortran_wall_seconds": (
                            oracle.reverse_wall_seconds - counters_before[2]
                        ),
                    }
                    failure = _publish_case_failure(
                        case_dir,
                        identity=identity,
                        stage=stage,
                        error=error,
                        runtime=runtime,
                        solver_result=result_for_failure,
                        solver_settings=solver_settings,
                        covariance_case=case,
                        initial_projection_gap=diagnostic_gap,
                        continuation_stages=continuation_records,
                    )
                    print(
                        f"{member_label}/{case}: FAILED at {stage}: "
                        f"{failure['error']['type']}: "
                        f"{failure['error']['message']}; continuing",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

    summary = _finalize(output_dir, run_manifest)
    print(
        f"{summary['status']}: {summary['completed_case_count']}/"
        f"{summary['expected_case_count']} cases; "
        f"wall={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return 1 if summary["failed_case_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
