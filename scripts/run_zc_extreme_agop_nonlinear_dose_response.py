#!/usr/bin/env python3
"""Run and assemble nonlinear AGOP dose responses for authentic ZC extremes.

This is a thin orchestrator around ``run_zc_direct_agop_covariance_action.py``.
Each event/coefficient pair is therefore a complete Figure-9-style nonlinear,
adjoint-based minimum pooled-covariance-action optimization, not a rescaling of
a previously optimized nudge.  Nine independent native interventions act over
three months and the requested AGOP projection is attained at the ten-month
forecast release.  The subsequent ten months are an unforced ZC forecast.
The publication bundle uses the same fixed dose family for both events.
Missing doses are solved outward from zero so a completed adjacent dose can
initialize the dual variables of a new, independently publication-gated
exact-target solve.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
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

SCRIPT_VERSION = "2.2.0"
SCHEMA_VERSION = 2
EXPERIMENT_TYPE = "extreme-event-nonlinear-agop-dose-response"
METHOD = "nonlinear minimum pooled-covariance action SQP"
ANNUAL_COVARIANCE_POLICY = "annual-all-phase-shared"
DEFAULT_ANNUAL_COVARIANCE_MANIFEST = Path(
    "outputs/zc_native_covariance/training-years10000-phases00-35/dense_manifest.json"
)
EVENT_ORDER = ("extreme_el_nino", "extreme_la_nina")
COMMON_COEFFICIENTS = (
    -1.0,
    -0.9,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.1,
    0.2,
    0.3,
)
COEFFICIENTS_BY_EVENT = {event: COMMON_COEFFICIENTS for event in EVENT_ORDER}
SUPPORTED_COEFFICIENTS = COMMON_COEFFICIENTS
CONTROL_STEPS = 9
RELEASE_BOUNDARIES = (9, 10)
FREE_FORECAST_STEPS = 30
BOUNDARY_COUNT = 41
CANONICAL_PROJECTION_RELATIVE_TOLERANCE = 1.0e-4


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (REPOSITORY_ROOT / expanded).resolve()
    )


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def _coefficient(text: str) -> float:
    value = float(text)
    for allowed in SUPPORTED_COEFFICIENTS:
        if math.isclose(value, allowed, rel_tol=0.0, abs_tol=1.0e-12):
            return allowed
    raise argparse.ArgumentTypeError(
        "coefficient must be one of -1.0,-0.9,...,-0.1,0.1,0.2,0.3"
    )


def _case_label(coefficient: float) -> str:
    sign = "plus" if coefficient > 0.0 else "minus"
    magnitude = f"{abs(coefficient):g}".replace(".", "p")
    return f"{sign}_{magnitude}"


def _continuation_fractions(coefficient: float) -> tuple[float, ...]:
    """Limit each continuation stage to 0.1 natural AGOP dose."""

    count = int(round(abs(coefficient) / 0.1))
    return tuple(index / count for index in range(1, count + 1))


def _selected_coefficients(
    event: str,
    requested: tuple[float, ...] | None,
) -> tuple[float, ...]:
    """Return requested doses in the event's canonical plotting order."""

    supported = COEFFICIENTS_BY_EVENT[event]
    if requested is None:
        return supported
    requested_set = set(requested)
    return tuple(value for value in supported if value in requested_set)


def _scientific_definition() -> dict[str, Any]:
    return {
        "method": METHOD,
        "covariance_case": "pooled",
        "covariance_policy": ANNUAL_COVARIANCE_POLICY,
        "objective": "0.5 * sum_k a_k^T C_pool^+ a_k",
        "control_steps": CONTROL_STEPS,
        "control_months": 3,
        "release_boundaries": list(RELEASE_BOUNDARIES),
        "free_forecast_steps": FREE_FORECAST_STEPS,
        "free_forecast_months": 10,
        "phase_coordinates_controlled": False,
        "release_constraint": (
            "e^T(x_release-x_extreme)=c e^T x_extreme in standardized "
            "core4 spatial coordinates"
        ),
        "signed_constraint_implementation": (
            "the inequality orientation is reversed for c<0; the "
            "minimum-action solution lies on the active equality boundary"
        ),
        "solver_initialization": (
            "missing doses are attempted outward from zero; when the completed "
            "same-sign dose 0.1 closer to zero is available, its dual variables "
            "initialize a new one-stage exact-target SQP solve"
        ),
        "derivatives": "exact split reverse-mode derivatives of nonlinear ZC",
        "publication_trajectory": "canonical O3 frozen-control replay",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--covariance-manifest",
        type=Path,
        default=DEFAULT_ANNUAL_COVARIANCE_MANIFEST,
        help=(
            "Completed dense covariance pooled over all 36 annual phases. "
            "The annual all-phase policy is fixed by this orchestrator."
        ),
    )
    parser.add_argument(
        "--generation-workspace",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3"),
    )
    parser.add_argument(
        "--adjoint-validation-dir",
        type=Path,
        default=Path("outputs/zc_adjoint/kernel_replay_validation_run23_final3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/zc_extreme_agop_nonlinear_dose_response/core4-cnn-lead10-seed42"
        ),
    )
    parser.add_argument(
        "--event",
        choices=("both", *EVENT_ORDER),
        default="both",
        help="Run both events, or one event independently/concurrently.",
    )
    parser.add_argument(
        "--coefficient",
        action="append",
        type=_coefficient,
        dest="coefficients",
        help=(
            "Run one signed dose; repeat as needed. With --event both, doses "
            "unsupported for one event are skipped for that event."
        ),
    )
    parser.add_argument(
        "--maximum-iterations",
        type=_positive_int,
        default=40,
        help=(
            "Per-continuation-stage SQP iteration limit. Figure 10 uses 40 so "
            "a boundary-restored iterate still has room for a final KKT update."
        ),
    )
    parser.add_argument("--constraint-tolerance", type=_positive_float, default=1.0e-5)
    parser.add_argument(
        "--stationarity-tolerance", type=_positive_float, default=1.0e-5
    )
    parser.add_argument(
        "--complementarity-tolerance", type=_positive_float, default=1.0e-5
    )
    parser.add_argument(
        "--relative-stationarity-tolerance", type=_positive_float, default=2.0e-2
    )
    parser.add_argument(
        "--relative-complementarity-tolerance",
        type=_positive_float,
        default=1.0e-4,
    )
    parser.add_argument("--initial-trust-radius", type=_positive_float, default=10.0)
    parser.add_argument("--maximum-trust-radius", type=_positive_float, default=100.0)
    parser.add_argument("--covariance-block-rows", type=_positive_int, default=1024)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--skip-data-checksums", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _direct_command(
    args: argparse.Namespace,
    *,
    event: str,
    coefficient: float,
    output_dir: Path,
    initial_dual_npz: Path | None = None,
    initial_dual_scale: float = 1.0,
) -> list[str]:
    if not math.isfinite(initial_dual_scale) or initial_dual_scale <= 0.0:
        raise ValueError("initial_dual_scale must be finite and positive")
    if initial_dual_npz is None and initial_dual_scale != 1.0:
        raise ValueError("initial_dual_scale requires initial_dual_npz")
    continuation = (
        (1.0,) if initial_dual_npz is not None else _continuation_fractions(coefficient)
    )
    fractions = ",".join(f"{value:g}" for value in continuation)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/run_zc_direct_agop_covariance_action.py"),
        "--data-dir",
        str(_resolve(args.data_dir)),
        "--artifacts-dir",
        str(_resolve(args.artifacts_dir)),
        "--agop-benchmark-dir",
        str(_resolve(args.agop_benchmark_dir)),
        "--covariance-policy",
        ANNUAL_COVARIANCE_POLICY,
        "--covariance-manifest",
        str(
            _resolve(
                getattr(
                    args,
                    "covariance_manifest",
                    DEFAULT_ANNUAL_COVARIANCE_MANIFEST,
                )
            )
        ),
        "--generation-workspace",
        str(_resolve(args.generation_workspace)),
        "--adjoint-validation-dir",
        str(_resolve(args.adjoint_validation_dir)),
        "--target-event",
        event,
        "--trajectory",
        "authentic-extreme",
        "--projection-change-multiple",
        f"{coefficient:g}",
        "--case",
        "pooled",
        "--continuation-fractions",
        fractions,
        "--maximum-iterations",
        str(args.maximum_iterations),
        "--constraint-tolerance",
        f"{args.constraint_tolerance:.17g}",
        "--stationarity-tolerance",
        f"{args.stationarity_tolerance:.17g}",
        "--complementarity-tolerance",
        f"{args.complementarity_tolerance:.17g}",
        "--relative-stationarity-tolerance",
        f"{args.relative_stationarity_tolerance:.17g}",
        "--relative-complementarity-tolerance",
        f"{args.relative_complementarity_tolerance:.17g}",
        "--initial-trust-radius",
        f"{args.initial_trust_radius:.17g}",
        "--maximum-trust-radius",
        f"{args.maximum_trust_radius:.17g}",
        "--covariance-block-rows",
        str(args.covariance_block_rows),
        "--timeout-seconds",
        f"{args.timeout_seconds:.17g}",
        "--output-dir",
        str(output_dir),
    ]
    if initial_dual_npz is not None:
        command.extend(("--initial-dual-npz", str(initial_dual_npz)))
        if initial_dual_scale != 1.0:
            command.extend(("--initial-dual-scale", f"{initial_dual_scale:.17g}"))
    elif len(continuation) > 1:
        command.append("--scale-continuation-warm-start")
    if args.skip_data_checksums:
        command.append("--skip-data-checksums")
    return command


def _result_paths(run_dir: Path, event_index: int) -> tuple[Path, Path]:
    member_dir = run_dir / "members" / f"member_01_i{event_index}"
    return member_dir / "pooled" / "report.json", member_dir / "pooled" / "result.npz"


def _adjacent_coefficient(coefficient: float) -> float | None:
    """Return the canonical 0.1 dose immediately closer to zero.

    Rescue runs may use noncanonical bridge coefficients such as ``-0.85``.
    Using :func:`round` here is unsafe because Python's ties-to-even rule maps
    ``8.5`` to ``8`` and would incorrectly identify ``-0.7`` as that bridge's
    parent.  Taking the ceiling before stepping inward maps both a canonical
    dose and every bridge in the following tenth to the same canonical parent.
    """

    magnitude_tenths = int(math.ceil(abs(coefficient) * 10.0 - 1.0e-12))
    if magnitude_tenths <= 1:
        return None
    return math.copysign((magnitude_tenths - 1) / 10.0, coefficient)


def _solve_order(coefficients: tuple[float, ...]) -> tuple[float, ...]:
    """Order requested doses outward from zero for adjacent dual warm starts."""

    negatives = sorted((value for value in coefficients if value < 0.0), reverse=True)
    positives = sorted(value for value in coefficients if value > 0.0)
    return tuple((*negatives, *positives))


def _ancestry_contract(
    identity: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Return scientific identity that a numerical warm start must share."""

    return {
        key: identity.get(key)
        for key in (
            "data_metadata_sha256",
            "model_checkpoint_sha256",
            "normalization_sha256",
            "base_agop_direction_sha256",
            "agop_direction_sha256",
            "covariance_manifest_sha256",
            "primal_executable_sha256",
            "differentiable_primal_executable_sha256",
            "differentiable_tangent_executable_sha256",
            "adjoint_executable_sha256",
            "state_manifest_sha256",
            "target_event_label",
            "extreme_event_input_index",
            "trajectory",
            "control_steps",
            "release_boundaries",
            "total_steps",
        )
    } | {
        "target_base_direction_sha256": target.get("base_direction_sha256"),
        "target_direction_sha256": target.get("direction_sha256"),
        "target_event_phase_modulo_36": target.get("event_phase_modulo_36"),
    }


def _warm_start_relation(
    previous_canonical: float,
    current: float,
    source: float,
    *,
    initial_dual_scale: float = 1.0,
) -> str | None:
    """Classify an authenticated same-event external-dual relationship."""

    if not math.isfinite(initial_dual_scale) or initial_dual_scale <= 0.0:
        return None
    scaled_outer_source = bool(
        source != 0.0
        and math.copysign(1.0, source) == math.copysign(1.0, current)
        and abs(source) > abs(current)
        and math.isclose(
            initial_dual_scale,
            current / source,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    if scaled_outer_source:
        return "verified_scaled_outer_dose"
    if initial_dual_scale != 1.0:
        return None

    if math.isclose(source, previous_canonical, rel_tol=0.0, abs_tol=1.0e-12):
        return "verified_adjacent_dose"
    lower, upper = sorted((previous_canonical, current))
    if lower < source < upper and math.copysign(1.0, source) == math.copysign(
        1.0, current
    ):
        return "verified_fine_homotopy_bridge"
    return None


def _ancestry_search_root(run_dir: Path) -> Path:
    """Find the common dose-response root containing canonical and retry runs."""

    for candidate in run_dir.parents:
        if (candidate / "retries").is_dir():
            return candidate
    return run_dir.parent


def _display_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _find_verified_warm_start_ancestor(
    run_dir: Path,
    coefficient: float,
    *,
    initial_dual_npz_sha256: str,
    initial_dual_raw_variables_sha256: str,
    initial_dual_scale: float,
    initial_dual_scaled_variables_sha256: str,
    current_identity: dict[str, Any],
    current_target: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve and authenticate an allowed initial-dual ancestor."""

    previous = _adjacent_coefficient(coefficient)
    if previous is None:
        return None
    search_root = _ancestry_search_root(run_dir)
    expected_contract = _ancestry_contract(current_identity, current_target)
    matches: list[dict[str, Any]] = []
    for manifest_path in sorted(search_root.rglob("run_manifest.json")):
        candidate_dir = manifest_path.parent
        if candidate_dir == run_dir:
            continue
        candidate_manifest = load_json(manifest_path)
        candidate_identity = candidate_manifest.get("run_identity")
        candidate_target = candidate_manifest.get("agop_target")
        if (
            candidate_manifest.get("status") != "complete"
            or not isinstance(candidate_identity, dict)
            or not isinstance(candidate_target, dict)
        ):
            continue
        try:
            source_coefficient = float(candidate_identity["projection_change_multiple"])
            source_event_index = int(candidate_identity["extreme_event_input_index"])
        except (KeyError, TypeError, ValueError):
            continue
        relation = _warm_start_relation(
            previous,
            coefficient,
            source_coefficient,
            initial_dual_scale=initial_dual_scale,
        )
        if (
            relation is None
            or _ancestry_contract(candidate_identity, candidate_target)
            != expected_contract
        ):
            continue
        source_report_path, source_result_path = _result_paths(
            candidate_dir, source_event_index
        )
        if (
            not source_result_path.is_file()
            or sha256_file(source_result_path) != initial_dual_npz_sha256
        ):
            continue
        with np.load(source_result_path, allow_pickle=False) as archive:
            if "dual_variables" not in archive.files:
                raise ValueError(
                    f"warm-start ancestor has no dual_variables: {source_result_path}"
                )
            raw_dual = np.asarray(archive["dual_variables"], dtype=np.float64)
        if (
            raw_dual.ndim != 2
            or not np.isfinite(raw_dual).all()
            or sha256_array(raw_dual) != initial_dual_raw_variables_sha256
            or sha256_array(initial_dual_scale * raw_dual)
            != initial_dual_scaled_variables_sha256
        ):
            raise ValueError(
                "warm-start source dual or its scaled image disagrees with the "
                f"current run identity: {source_result_path}"
            )
        validated = _validated_case(
            candidate_dir,
            source_coefficient,
            _resolve_warm_start_ancestry=False,
        )
        if validated is None:
            raise ValueError(
                "warm-start ancestor has no completed publication-gated result: "
                f"{candidate_dir}"
            )
        matches.append(
            {
                "source_kind": relation,
                "source_coefficient": source_coefficient,
                "source_case_label": _case_label(source_coefficient),
                "source_run_directory": _display_source_path(candidate_dir),
                "source_manifest_file": _display_source_path(manifest_path),
                "source_manifest_sha256": sha256_file(manifest_path),
                "source_report_file": _display_source_path(source_report_path),
                "source_report_sha256": sha256_file(source_report_path),
                "source_result_file": _display_source_path(source_result_path),
                "source_result_sha256": sha256_file(source_result_path),
            }
        )
    if len(matches) > 1:
        raise ValueError(
            "initial-dual hash resolves to multiple valid warm-start ancestors: "
            f"{run_dir}"
        )
    return None if not matches else matches[0]


def _validated_case(
    run_dir: Path,
    coefficient: float,
    *,
    _resolve_warm_start_ancestry: bool = True,
) -> dict[str, Any] | None:
    """Validate one direct-driver result and the canonical release target."""

    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    target = manifest.get("agop_target")
    identity = manifest.get("run_identity")
    if not isinstance(target, dict) or not isinstance(identity, dict):
        raise ValueError(f"missing AGOP target: {run_dir}")
    if (
        manifest.get("active_cases") != ["pooled"]
        or target.get("trajectory") != "authentic-extreme"
        or not math.isclose(
            float(target.get("projection_change_multiple")),
            coefficient,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(f"direct nonlinear case has another identity: {run_dir}")
    if manifest.get("status") != "complete" or summary.get("status") != "complete":
        return None
    if (
        summary.get("completed_case_count") != 1
        or summary.get("failed_case_count") != 0
    ):
        raise ValueError(f"completed direct case has invalid accounting: {run_dir}")
    event_index = int(target["extreme_event_input_index"])
    report_path, result_path = _result_paths(run_dir, event_index)
    report = load_json(report_path)
    artifact = report.get("artifact")
    gates = report.get("publication_gates")
    transfer = report.get("canonical_o3_frozen_control_transfer_audit")
    if (
        report.get("status") != "complete"
        or not isinstance(gates, dict)
        or gates.get("all_passed") is not True
        or not isinstance(artifact, dict)
        or artifact.get("sha256") != sha256_file(result_path)
        or artifact.get("size_bytes") != result_path.stat().st_size
        or not isinstance(transfer, dict)
        or transfer.get("all_values_finite") is not True
    ):
        raise ValueError(f"direct result failed publication integrity: {run_dir}")
    canonical = transfer.get("canonical_o3")
    if not isinstance(canonical, dict):
        raise ValueError(f"missing canonical transfer replay: {run_dir}")
    orientation = float(target["direction_orientation"])
    natural_q = float(target["natural_event_projection_in_base_direction"])
    requested = float(target["target_projection_in_base_direction"])
    realized = orientation * float(canonical["release_agop_projection"])
    residual = realized - requested
    tolerance = max(
        1.0e-5,
        CANONICAL_PROJECTION_RELATIVE_TOLERANCE * abs(natural_q),
    )
    if abs(residual) > tolerance:
        raise RuntimeError(
            "canonical O3 trajectory missed its requested release projection: "
            f"residual={residual:.6g}, tolerance={tolerance:.6g}, run={run_dir}"
        )
    with np.load(result_path, allow_pickle=False) as archive:
        trajectory = np.asarray(
            archive["canonical_o3_nino3_by_boundary_c"], dtype=np.float64
        )
    if trajectory.shape != (BOUNDARY_COUNT,) or not np.isfinite(trajectory).all():
        raise ValueError(f"canonical trajectory has the wrong shape: {run_dir}")
    initial_dual_npz_sha256 = identity.get("initial_dual_npz_sha256")
    initial_dual_variables_sha256 = identity.get("initial_dual_variables_sha256")
    explicit_raw_sha256 = identity.get("initial_dual_raw_variables_sha256")
    explicit_scale = identity.get("initial_dual_scale")
    explicit_scaled_sha256 = identity.get("initial_dual_scaled_variables_sha256")
    externally_warm_started = initial_dual_npz_sha256 is not None
    solver_settings = identity.get("solver_settings")
    ancestry: dict[str, Any] | None = None
    if externally_warm_started:
        explicit_scaling_fields = (
            explicit_raw_sha256,
            explicit_scale,
            explicit_scaled_sha256,
        )
        if all(value is None for value in explicit_scaling_fields):
            # Backward-compatible validation for exact-target warm starts
            # produced before explicit source scaling was supported.
            initial_dual_raw_variables_sha256 = initial_dual_variables_sha256
            initial_dual_scale = 1.0
            initial_dual_scaled_variables_sha256 = initial_dual_variables_sha256
        elif any(value is None for value in explicit_scaling_fields):
            raise ValueError(f"incomplete external warm-start identity: {run_dir}")
        else:
            if isinstance(explicit_scale, bool) or not isinstance(
                explicit_scale, (int, float)
            ):
                raise ValueError(f"invalid external warm-start scale: {run_dir}")
            initial_dual_raw_variables_sha256 = explicit_raw_sha256
            initial_dual_scale = float(explicit_scale)
            initial_dual_scaled_variables_sha256 = explicit_scaled_sha256
        if (
            not isinstance(initial_dual_npz_sha256, str)
            or not isinstance(initial_dual_variables_sha256, str)
            or not isinstance(initial_dual_raw_variables_sha256, str)
            or not math.isfinite(initial_dual_scale)
            or initial_dual_scale <= 0.0
            or not isinstance(initial_dual_scaled_variables_sha256, str)
            or initial_dual_variables_sha256
            != initial_dual_scaled_variables_sha256
            or not isinstance(solver_settings, dict)
            or solver_settings.get("continuation_fractions") != [1.0]
            or solver_settings.get("scale_continuation_warm_start") is not False
        ):
            raise ValueError(f"invalid external warm-start identity: {run_dir}")
        if _resolve_warm_start_ancestry:
            ancestry = _find_verified_warm_start_ancestor(
                run_dir,
                coefficient,
                initial_dual_npz_sha256=initial_dual_npz_sha256,
                initial_dual_raw_variables_sha256=(
                    initial_dual_raw_variables_sha256
                ),
                initial_dual_scale=initial_dual_scale,
                initial_dual_scaled_variables_sha256=(
                    initial_dual_scaled_variables_sha256
                ),
                current_identity=identity,
                current_target=target,
            )
    elif any(
        value is not None
        for value in (
            initial_dual_variables_sha256,
            explicit_raw_sha256,
            explicit_scale,
            explicit_scaled_sha256,
        )
    ):
        raise ValueError(f"orphan initial-dual identity: {run_dir}")
    else:
        initial_dual_raw_variables_sha256 = None
        initial_dual_scale = None
        initial_dual_scaled_variables_sha256 = None
    source_kind = (
        ancestry["source_kind"]
        if ancestry is not None
        else ("external_exact_target_checkpoint" if externally_warm_started else None)
    )
    row = summary["rows"][0]
    return {
        "event_label": str(target["target_event_label"]),
        "event_input_index": event_index,
        "coefficient": coefficient,
        "natural_projection_q": natural_q,
        "baseline_original_e_projection": float(row["baseline_release_projection"])
        * orientation,
        "requested_original_e_projection": requested,
        "realized_original_e_projection": realized,
        "canonical_projection_residual": residual,
        "canonical_projection_tolerance": tolerance,
        "canonical_projection_gate_passed": True,
        "canonical_nino3": trajectory,
        "terminal_nino3_c": float(trajectory[-1]),
        "report": report,
        "report_path": report_path,
        "result_path": result_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "warm_start": {
            "used": externally_warm_started,
            "source_kind": source_kind,
            "adjacent_source_verified": source_kind == "verified_adjacent_dose",
            "fine_homotopy_bridge_verified": (
                source_kind == "verified_fine_homotopy_bridge"
            ),
            "scaled_outer_source_verified": (
                source_kind == "verified_scaled_outer_dose"
            ),
            "source_coefficient": (
                None if ancestry is None else ancestry["source_coefficient"]
            ),
            "source_case_label": (
                None if ancestry is None else ancestry["source_case_label"]
            ),
            "source_run_directory": (
                None if ancestry is None else ancestry["source_run_directory"]
            ),
            "source_manifest_file": (
                None if ancestry is None else ancestry["source_manifest_file"]
            ),
            "source_manifest_sha256": (
                None if ancestry is None else ancestry["source_manifest_sha256"]
            ),
            "source_report_file": (
                None if ancestry is None else ancestry["source_report_file"]
            ),
            "source_report_sha256": (
                None if ancestry is None else ancestry["source_report_sha256"]
            ),
            "source_result_file": (
                None if ancestry is None else ancestry["source_result_file"]
            ),
            "source_result_sha256": (
                None if ancestry is None else ancestry["source_result_sha256"]
            ),
            "initial_dual_npz_sha256": initial_dual_npz_sha256,
            "initial_dual_variables_sha256": initial_dual_variables_sha256,
            "initial_dual_raw_variables_sha256": (
                initial_dual_raw_variables_sha256
            ),
            "initial_dual_scale": initial_dual_scale,
            "initial_dual_scaled_variables_sha256": (
                initial_dual_scaled_variables_sha256
            ),
            "continuation_fractions": (
                None
                if not isinstance(solver_settings, dict)
                else solver_settings.get("continuation_fractions")
            ),
        },
    }


def _reuse_pilot_if_available(output_root: Path, destination: Path) -> bool:
    """Reuse the already validated warm +0.1 pilot when identities match."""

    if destination.exists():
        return False
    pilot = output_root / "pilots" / "warm_plus_0p1"
    if not pilot.is_dir():
        return False
    try:
        candidate = _validated_case(pilot, 0.1)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False
    if candidate is None or candidate["event_label"] != "extreme_el_nino":
        return False
    shutil.copytree(pilot, destination)
    return True


def _event_report(
    output_root: Path,
    event: str,
    completed: dict[float, dict[str, Any]],
    nino3_all: np.ndarray,
) -> dict[str, Any]:
    first = completed[next(iter(completed))]
    event_index = int(first["event_input_index"])
    baseline = np.asarray(
        nino3_all[event_index - 10 : event_index + FREE_FORECAST_STEPS + 1],
        dtype=np.float64,
    )
    if baseline.shape != (BOUNDARY_COUNT,) or not np.isfinite(baseline).all():
        raise ValueError(f"archived baseline is unavailable for {event}")
    expected_coefficients = COEFFICIENTS_BY_EVENT[event]
    coefficients = [value for value in expected_coefficients if value in completed]
    arrays: dict[str, Any] = {
        "coefficients": np.asarray(coefficients, dtype=np.float64),
        "event_input_index": np.asarray(event_index, dtype=np.int64),
        "event_target_index": np.asarray(
            event_index + FREE_FORECAST_STEPS, dtype=np.int64
        ),
        "natural_projection_q": np.asarray(
            first["natural_projection_q"], dtype=np.float64
        ),
        "baseline_release_projection": np.asarray(
            first["baseline_original_e_projection"], dtype=np.float64
        ),
        "nino3_baseline": baseline,
    }
    cases: dict[str, Any] = {}
    event_dir = output_root / event
    for coefficient in coefficients:
        item = completed[coefficient]
        label = _case_label(coefficient)
        release = item["report"]["release"]
        arrays[f"nino3__{label}"] = item["canonical_nino3"]
        arrays[f"requested_release_projection__{label}"] = np.asarray(
            item["requested_original_e_projection"], dtype=np.float64
        )
        arrays[f"realized_release_projection__{label}"] = np.asarray(
            item["realized_original_e_projection"], dtype=np.float64
        )
        cases[label] = {
            "coefficient": coefficient,
            "run_directory": label,
            "run_manifest_file": str(item["manifest_path"].relative_to(event_dir)),
            "run_manifest_sha256": sha256_file(item["manifest_path"]),
            "summary_file": str(item["summary_path"].relative_to(event_dir)),
            "summary_sha256": sha256_file(item["summary_path"]),
            "report_file": str(item["report_path"].relative_to(event_dir)),
            "report_sha256": sha256_file(item["report_path"]),
            "result_file": str(item["result_path"].relative_to(event_dir)),
            "result_sha256": sha256_file(item["result_path"]),
            "publication_gates": item["report"]["publication_gates"],
            "solver": {
                key: item["report"]["solver"][key]
                for key in (
                    "success",
                    "status",
                    "objective_value",
                    "action_norm",
                    "constraint_value",
                    "accepted_update_count",
                    "gradient_evaluations",
                )
            },
            "requested_original_e_projection": item["requested_original_e_projection"],
            "realized_original_e_projection": item["realized_original_e_projection"],
            "canonical_projection_residual": item["canonical_projection_residual"],
            "canonical_projection_tolerance": item["canonical_projection_tolerance"],
            "canonical_projection_gate_passed": True,
            "adjacent_dual_warm_start": item["warm_start"],
            "terminal_nino3_c": item["terminal_nino3_c"],
            "terminal_change_from_original_c": item["terminal_nino3_c"]
            - float(baseline[-1]),
            "release_nino3_c": float(release["nino3_c"]),
            "release_cnn_forecast_c": float(release["cnn_forecast_c"]),
            "release_cosine_with_agop": float(release["cosine_with_agop"]),
            "release_standardized_spatial_rms_change": float(
                release["standardized_spatial_rms_change"]
            ),
        }
    trajectory_path = event_dir / "trajectories.npz"
    write_npz(trajectory_path, overwrite=True, **arrays)
    status = "complete" if coefficients == list(expected_coefficients) else "partial"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "experiment_type": EXPERIMENT_TYPE,
        "scientific_definition": _scientific_definition(),
        "event_label": event,
        "event_input_index": event_index,
        "event_target_index": event_index + FREE_FORECAST_STEPS,
        "event_target_nino3_c": float(baseline[-1]),
        "natural_projection_q": first["natural_projection_q"],
        "baseline_original_e_projection": first["baseline_original_e_projection"],
        "coefficient_order": list(expected_coefficients),
        "completed_coefficients": coefficients,
        "cases": cases,
        "outputs": {
            "trajectories_file": trajectory_path.name,
            "trajectories_sha256": sha256_file(trajectory_path),
        },
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    write_json(event_dir / "report.json", report, overwrite=True)
    return report


def _combine_if_complete(output_root: Path) -> dict[str, Any] | None:
    reports: dict[str, dict[str, Any]] = {}
    archives: dict[str, dict[str, np.ndarray]] = {}
    for event in EVENT_ORDER:
        report_path = output_root / event / "report.json"
        archive_path = output_root / event / "trajectories.npz"
        if not report_path.is_file() or not archive_path.is_file():
            return None
        report = load_json(report_path)
        expected_coefficients = COEFFICIENTS_BY_EVENT[event]
        if (
            report.get("schema_version") != SCHEMA_VERSION
            or report.get("status") != "complete"
            or report.get("experiment_type") != EXPERIMENT_TYPE
            or report.get("event_label") != event
            or report.get("coefficient_order") != list(expected_coefficients)
            or report.get("completed_coefficients") != list(expected_coefficients)
        ):
            return None
        outputs = report.get("outputs")
        if (
            not isinstance(outputs, dict)
            or outputs.get("trajectories_file") != archive_path.name
            or outputs.get("trajectories_sha256") != sha256_file(archive_path)
        ):
            raise ValueError(f"event archive failed its report checksum: {event}")
        reports[event] = report
        with np.load(archive_path, allow_pickle=False) as source:
            archives[event] = {key: np.asarray(source[key]) for key in source.files}
        expected = np.asarray(expected_coefficients, dtype=np.float64)
        if not np.array_equal(archives[event].get("coefficients"), expected):
            raise ValueError(f"event archive has another coefficient order: {event}")

    combined_arrays: dict[str, Any] = {}
    for event in EVENT_ORDER:
        expected_coefficients = COEFFICIENTS_BY_EVENT[event]
        archive = archives[event]
        requested = np.asarray(
            [
                archive[
                    f"requested_release_projection__{_case_label(coefficient)}"
                ].item()
                for coefficient in expected_coefficients
            ],
            dtype=np.float64,
        )
        realized = np.asarray(
            [
                archive[
                    f"realized_release_projection__{_case_label(coefficient)}"
                ].item()
                for coefficient in expected_coefficients
            ],
            dtype=np.float64,
        )
        perturbed = np.stack(
            [
                archive[f"nino3__{_case_label(coefficient)}"]
                for coefficient in expected_coefficients
            ]
        )
        baseline = np.asarray(archive["nino3_baseline"], dtype=np.float64)
        if (
            requested.shape != (len(expected_coefficients),)
            or realized.shape != requested.shape
            or perturbed.shape != (len(expected_coefficients), BOUNDARY_COUNT)
            or baseline.shape != (BOUNDARY_COUNT,)
            or not all(
                np.isfinite(values).all()
                for values in (requested, realized, perturbed, baseline)
            )
        ):
            raise ValueError(f"event archive has invalid trajectory arrays: {event}")
        suffix = f"__{event}"
        combined_arrays.update(
            {
                f"coefficients{suffix}": np.asarray(
                    expected_coefficients, dtype=np.float64
                ),
                f"event_input_index{suffix}": np.asarray(
                    reports[event]["event_input_index"], dtype=np.int64
                ),
                f"event_target_index{suffix}": np.asarray(
                    reports[event]["event_target_index"], dtype=np.int64
                ),
                f"natural_projection_q{suffix}": np.asarray(
                    reports[event]["natural_projection_q"], dtype=np.float64
                ),
                f"baseline_release_projection{suffix}": np.asarray(
                    reports[event]["baseline_original_e_projection"],
                    dtype=np.float64,
                ),
                f"requested_release_projections{suffix}": requested,
                f"realized_release_projections{suffix}": realized,
                f"nino3_baseline{suffix}": baseline,
                f"nino3_perturbed{suffix}": perturbed,
            }
        )
    combined_path = output_root / "trajectories.npz"
    write_npz(combined_path, overwrite=True, **combined_arrays)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment_type": EXPERIMENT_TYPE,
        "scientific_definition": _scientific_definition(),
        "event_order": list(EVENT_ORDER),
        "coefficients_by_event": {
            event: list(COEFFICIENTS_BY_EVENT[event]) for event in EVENT_ORDER
        },
        "events": {
            event: {
                "coefficients": list(COEFFICIENTS_BY_EVENT[event]),
                "report_file": f"{event}/report.json",
                "report_sha256": sha256_file(output_root / event / "report.json"),
                "trajectories_file": f"{event}/trajectories.npz",
                "trajectories_sha256": sha256_file(
                    output_root / event / "trajectories.npz"
                ),
                "cases": reports[event]["cases"],
            }
            for event in EVENT_ORDER
        },
        "outputs": {
            "trajectories_file": combined_path.name,
            "trajectories_sha256": sha256_file(combined_path),
        },
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output_root / "report.json", report, overwrite=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    events = EVENT_ORDER if args.event == "both" else (args.event,)
    requested_coefficients = (
        None
        if args.coefficients is None
        else tuple(dict.fromkeys(args.coefficients).keys())
    )
    if args.event != "both" and not _selected_coefficients(
        args.event, requested_coefficients
    ):
        parser.error(
            f"none of the requested coefficients is supported for {args.event}"
        )
    output_root = _resolve(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    nino3_path = _resolve(args.data_dir) / "nino3_index.npy"
    nino3_all = np.load(nino3_path, mmap_mode="r", allow_pickle=False)
    if nino3_all.ndim != 1 or nino3_all.dtype != np.dtype("<f4"):
        raise ValueError("fresh zc-v3 nino3_index.npy has an unexpected layout")

    for event in events:
        selected_coefficients = _selected_coefficients(event, requested_coefficients)
        if not selected_coefficients:
            print(
                f"{event}: no supported requested coefficient; skipping",
                flush=True,
            )
            continue
        event_dir = output_root / event
        event_dir.mkdir(parents=True, exist_ok=True)
        for coefficient in _solve_order(selected_coefficients):
            case_dir = event_dir / _case_label(coefficient)
            if args.overwrite and case_dir.is_dir():
                if REPOSITORY_ROOT not in case_dir.parents:
                    raise ValueError("refusing to remove output outside the repository")
                shutil.rmtree(case_dir)
            cached = _validated_case(case_dir, coefficient)
            if cached is not None:
                if cached["event_label"] != event:
                    raise ValueError(
                        f"cached case belongs to another event: {case_dir}"
                    )
                print(
                    f"{event}/{_case_label(coefficient)}: validated cache",
                    flush=True,
                )
                continue
            if (
                event == "extreme_el_nino"
                and coefficient == 0.1
                and _reuse_pilot_if_available(output_root, case_dir)
            ):
                print("extreme_el_nino/plus_0p1: reused validated pilot", flush=True)
                continue
            initial_dual_npz: Path | None = None
            adjacent = _adjacent_coefficient(coefficient)
            if adjacent is not None and adjacent in COEFFICIENTS_BY_EVENT[event]:
                source = _validated_case(event_dir / _case_label(adjacent), adjacent)
                if source is not None:
                    if source["event_label"] != event:
                        raise ValueError(
                            "adjacent warm-start cache belongs to another event"
                        )
                    initial_dual_npz = source["result_path"]
            command = _direct_command(
                args,
                event=event,
                coefficient=coefficient,
                output_dir=case_dir,
                initial_dual_npz=initial_dual_npz,
            )
            warm_start_note = (
                "" if initial_dual_npz is None else f" from {_case_label(adjacent)}"
            )
            print(
                f"{event}/{_case_label(coefficient)}: launching nonlinear SQP"
                f"{warm_start_note}",
                flush=True,
            )
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"nonlinear SQP failed for {event}, c={coefficient:+g}"
                )
            result = _validated_case(case_dir, coefficient)
            if result is None or result["event_label"] != event:
                raise RuntimeError("direct driver returned no validated result")

        completed_cases: dict[float, dict[str, Any]] = {}
        for coefficient in COEFFICIENTS_BY_EVENT[event]:
            candidate = _validated_case(
                event_dir / _case_label(coefficient), coefficient
            )
            if candidate is not None:
                if candidate["event_label"] != event:
                    raise ValueError("event directory contains a mismatched case")
                completed_cases[coefficient] = candidate
        if completed_cases:
            _event_report(output_root, event, completed_cases, nino3_all)

    combined = _combine_if_complete(output_root)
    if combined is None:
        for stale_name in ("report.json", "trajectories.npz"):
            stale_path = output_root / stale_name
            if stale_path.is_file():
                stale_path.unlink(missing_ok=True)
        print(
            "Requested cases finished; combined artifact awaits both events "
            "and all doses.",
            flush=True,
        )
    else:
        print(f"Combined nonlinear dose response saved in {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
