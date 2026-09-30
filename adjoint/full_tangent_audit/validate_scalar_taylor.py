#!/usr/bin/env python3
"""Validate the scalar ZC adjoint with canonical-primal Taylor sweeps."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import validate_adjoint_dot as dot_audit
import validate_tangent as tangent_audit
from output_safety import transactional_output

NREAL = tangent_audit.NREAL
NCOMPLEX = tangent_audit.NCOMPLEX
NDOUBLE = tangent_audit.NDOUBLE
NINTEGER = tangent_audit.NINTEGER
NTIME = tangent_audit.NTIME
NSTATES = 32
NSTEPS = 31
TAPE_WIDTH = 3
HEADER_BYTES = 7 * 4


def read_path_final_real_and_tape(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header = np.fromfile(path, dtype=np.int32, count=7)
    expected = np.array(
        [NSTEPS, NREAL, NCOMPLEX, NDOUBLE, NINTEGER, NTIME, TAPE_WIDTH],
        dtype=np.int32,
    )
    if not np.array_equal(header, expected):
        raise RuntimeError(f"unexpected path header: {header.tolist()}")
    real_values = np.fromfile(
        path,
        dtype=np.float32,
        count=NREAL * NSTATES,
        offset=HEADER_BYTES,
    ).reshape((NREAL, NSTATES), order="F")
    tape_offset = HEADER_BYTES + (
        NREAL * NSTATES * 4
        + NCOMPLEX * NSTATES * 8
        + NDOUBLE * NSTATES * 8
        + NINTEGER * NSTATES * 4
        + NTIME * NSTATES * 4
    )
    tape = np.fromfile(
        path, dtype=np.int32, count=TAPE_WIDTH * NSTEPS, offset=tape_offset
    )
    if tape.size != TAPE_WIDTH * NSTEPS:
        raise RuntimeError("truncated path tape")
    return real_values[:, -1].copy(), tape.reshape((NSTEPS, TAPE_WIDTH))


def read_initial_nt(path: Path) -> int:
    offset = NREAL * 4 + NCOMPLEX * 8 + NDOUBLE * 8
    values = np.fromfile(path, dtype=np.int32, count=NINTEGER, offset=offset)
    if values.size != NINTEGER:
        raise RuntimeError("truncated initial packed state")
    return int(values[0])


def canonical_double(values: np.ndarray) -> float:
    selected: list[float] = []
    for j in tangent_audit.NINO3_J:
        for i in tangent_audit.NINO3_I:
            k = tangent_audit.NINO3_OFFSET + i + 30 * (j - 1) - 1
            selected.append(float(values[k]))
    return math.fsum(selected) / 66.0


def tape_delta(candidate: np.ndarray, baseline: np.ndarray) -> dict:
    locations = np.argwhere(candidate != baseline)
    return {
        "matches_baseline": bool(locations.size == 0),
        "mismatch_count": int(locations.shape[0]),
        "mismatches": [
            {
                "step_one_based": int(step) + 1,
                "slot_one_based": int(slot) + 1,
                "baseline": int(baseline[step, slot]),
                "candidate": int(candidate[step, slot]),
            }
            for step, slot in locations
        ],
    }


def resolve_case_files(case_dir: Path, gradient_name: str) -> dict[str, Path | str]:
    """Resolve either the packaged runner layout or the legacy flat layout."""
    packaged = {
        "initial_state": case_dir / "runtime/kernel_initial_state.bin",
        "path": case_dir / "artifacts/zc_31step_path.bin",
        "gradient": case_dir / "artifacts" / gradient_name,
        "restart": case_dir / "runtime/zeq9fsu.hst",
        "run_report": case_dir / "run_report.json",
    }
    flat = {
        "initial_state": case_dir / "kernel_initial_state.bin",
        "path": case_dir / "zc_31step_path.bin",
        "gradient": case_dir / gradient_name,
        "restart": case_dir / "zeq9fsu.hst",
        "run_report": case_dir / "run_report.json",
    }
    for layout, candidate in (("packaged_runner_v1", packaged), ("flat_v1", flat)):
        if all(path.is_file() for path in candidate.values()):
            return {"layout": layout, **candidate}
    expected = [str(path) for path in (*packaged.values(), *flat.values())]
    raise FileNotFoundError(
        f"case {case_dir} does not match a supported layout; expected one complete "
        f"set among: {expected}"
    )


def verify_taylor_build(
    *,
    primal: Path,
    source: Path,
    kernel_build_report: Path,
    kernel_source_manifest: Path,
    coupled_build_manifest: Path,
    coupled_build_input_manifest: Path,
    coupled_build_snapshot: Path,
    generator_path: Path,
) -> dict[str, object]:
    """Bind the finite-difference oracle to the exact coupled snapshot."""

    if source != coupled_build_snapshot / "kernel_source":
        raise RuntimeError("kernel source is not from the coupled-build snapshot")
    if (
        kernel_build_report
        != coupled_build_snapshot / "kernel_parent/build_report.json"
    ):
        raise RuntimeError("kernel report is not from the coupled-build snapshot")
    if (
        kernel_source_manifest
        != coupled_build_snapshot / "kernel_parent/source_manifest.json"
    ):
        raise RuntimeError("kernel source manifest is not from the coupled snapshot")
    coupled = tangent_audit.parse_key_values(coupled_build_manifest)
    if coupled.get("build_input_manifest_sha256") != dot_audit.sha256(
        coupled_build_input_manifest
    ):
        raise RuntimeError("coupled build does not bind its input snapshot")
    if coupled.get("kernel_parent_build_report_sha256") != dot_audit.sha256(
        kernel_build_report
    ):
        raise RuntimeError("coupled build does not bind the kernel build report")
    tangent_audit.verify_sha256_manifest(
        coupled_build_snapshot,
        coupled_build_input_manifest,
        exact_inventory=True,
    )
    report = json.loads(kernel_build_report.read_text(encoding="utf-8"))
    if report.get("kernel_executable_sha256") != dot_audit.sha256(primal):
        raise RuntimeError("primal executable does not match kernel build report")
    if report.get("source_manifest_file_sha256") != dot_audit.sha256(
        kernel_source_manifest
    ):
        raise RuntimeError("kernel source manifest does not match build report")
    expected_source = report.get("staged_source_manifest_sha256")
    observed_source = tangent_audit.tree_file_hashes(source)
    # The dataset generator's canonical digest implementation may differ from
    # the compact fallback above, so the exact per-file source mapping remains
    # the authoritative comparison when present.
    report_source = report.get("staged_source_file_sha256")
    if report_source is not None and report_source != observed_source:
        raise RuntimeError("kernel source tree differs from build report")
    generator_hash = report.get("builder_source_sha256", {}).get(
        "generate_fresh_zc_dataset.py"
    )
    if generator_hash != dot_audit.sha256(generator_path):
        raise RuntimeError("dataset generator differs from kernel build report")
    return {
        "kernel_build_report_sha256": dot_audit.sha256(kernel_build_report),
        "kernel_source_manifest_sha256": dot_audit.sha256(kernel_source_manifest),
        "coupled_build_manifest_sha256": dot_audit.sha256(coupled_build_manifest),
        "coupled_build_input_manifest_sha256": dot_audit.sha256(
            coupled_build_input_manifest
        ),
        "primal_executable_sha256": dot_audit.sha256(primal),
        "kernel_source_tree_sha256": expected_source
        or dot_audit.tree_manifest(source)[0],
        "generator_script_sha256": dot_audit.sha256(generator_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--case-labels",
        nargs="+",
        default=["neutral_member_04", "extreme_el_nino", "extreme_la_nina"],
    )
    parser.add_argument("--direction-seed", type=int, default=9401)
    parser.add_argument(
        "--direction-mode",
        choices=("random_segment_scaled", "adjoint_aligned"),
        default="random_segment_scaled",
    )
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[
            0.2,
            0.1,
            0.05,
            0.025,
            0.0125,
            0.00625,
            0.003125,
            0.0015625,
            0.00078125,
            0.000390625,
        ],
    )
    parser.add_argument("--primal-executable", type=Path, required=True)
    parser.add_argument("--reverse-executable", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument(
        "--kernel-source",
        type=Path,
        required=True,
        help=(
            "Immutable coupled-build kernel_source snapshot used only for "
            "compile-provenance binding."
        ),
    )
    parser.add_argument(
        "--runtime-source",
        type=Path,
        help=(
            "Full runnable source tree; defaults to verified_inputs/source "
            "within --replay-validation-dir and, if supplied, must equal it."
        ),
    )
    parser.add_argument(
        "--replay-validation-dir",
        type=Path,
        required=True,
        help=(
            "Passed schema-2 forward replay validation whose authenticated "
            "verified_inputs/source supplies the finite-difference runtime."
        ),
    )
    parser.add_argument("--kernel-build-report", type=Path, required=True)
    parser.add_argument("--kernel-source-manifest", type=Path, required=True)
    parser.add_argument("--coupled-build-manifest", type=Path, required=True)
    parser.add_argument("--coupled-build-input-manifest", type=Path, required=True)
    parser.add_argument("--coupled-build-snapshot", type=Path, required=True)
    parser.add_argument(
        "--gradient-name",
        default="zc_nino3_gradient.bin",
        help="Gradient filename within each case directory.",
    )
    args = parser.parse_args()
    validator_path = Path(__file__).resolve()
    validator_sha256 = tangent_audit.sha256(validator_path)

    root = args.project_root.resolve()
    output_requested = args.output_dir
    primal = args.primal_executable.resolve()
    reverse_executable = args.reverse_executable.resolve()
    cases_root = args.cases_root.resolve()
    kernel_source = args.kernel_source.resolve()
    replay_validation_dir = args.replay_validation_dir.resolve()
    replay_report_path = replay_validation_dir / "replay_report.json"
    runtime_source = (
        args.runtime_source.resolve()
        if args.runtime_source is not None
        else replay_validation_dir / "verified_inputs/source"
    )
    kernel_build_report = args.kernel_build_report.resolve()
    kernel_source_manifest = args.kernel_source_manifest.resolve()
    coupled_build_manifest = args.coupled_build_manifest.resolve()
    coupled_build_input_manifest = args.coupled_build_input_manifest.resolve()
    coupled_build_snapshot = args.coupled_build_snapshot.resolve()
    manifest_path = root / "adjoint/fortran_kernel/state_manifest.json"
    state_manifest_sha256 = tangent_audit.sha256(manifest_path)

    sys.path.insert(0, str(root / "scripts"))
    import generate_fresh_zc_dataset as generator  # noqa: PLC0415

    generator_path = root / "scripts/generate_fresh_zc_dataset.py"

    for required in (
        primal,
        reverse_executable,
        kernel_build_report,
        kernel_source_manifest,
        coupled_build_manifest,
        coupled_build_input_manifest,
        generator_path,
        replay_report_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    for required in (
        cases_root,
        kernel_source,
        runtime_source,
        coupled_build_snapshot,
    ):
        if not required.is_dir():
            raise FileNotFoundError(required)
    protected_case_files: list[Path] = []
    resolved_cases: dict[str, dict[str, Path | str]] = {}
    producer_bindings: dict[str, dict[str, object]] = {}
    case_input_hashes: dict[str, dict[str, str]] = {}
    for case_label in args.case_labels:
        source_case = cases_root / case_label
        case_files = resolve_case_files(source_case, args.gradient_name)
        resolved_cases[case_label] = case_files
        protected_case_files.extend(
            path for path in case_files.values() if isinstance(path, Path)
        )
        producer_bindings[case_label] = dot_audit.verify_scalar_gradient_producer(
            run_report_path=case_files["run_report"],
            path_file=case_files["path"],
            gradient_file=case_files["gradient"],
            reverse_executable=reverse_executable,
            coupled_build_manifest=coupled_build_manifest,
            coupled_build_input_manifest=coupled_build_input_manifest,
        )
        case_input_hashes[case_label] = {
            key: dot_audit.sha256(path)
            for key, path in case_files.items()
            if isinstance(path, Path)
        }

    build_provenance = verify_taylor_build(
        primal=primal,
        source=kernel_source,
        kernel_build_report=kernel_build_report,
        kernel_source_manifest=kernel_source_manifest,
        coupled_build_manifest=coupled_build_manifest,
        coupled_build_input_manifest=coupled_build_input_manifest,
        coupled_build_snapshot=coupled_build_snapshot,
        generator_path=generator_path,
    )
    replay_binding = tangent_audit.verify_replay_report_binding(
        replay_report_path,
        kernel_build_report=kernel_build_report,
        primal=primal,
    )
    runtime_source_binding = tangent_audit.verify_replay_runtime_source(
        runtime_source,
        replay_report_path,
        kernel_build_report=kernel_build_report,
        generator=generator,
    )
    kernel_source_hashes = tangent_audit.tree_file_hashes(kernel_source)
    runtime_source_hashes = tangent_audit.tree_file_hashes(runtime_source)
    bound_primal = primal
    bound_kernel_source = kernel_source
    bound_runtime_source = runtime_source

    output_transaction = transactional_output(
        output_requested,
        overwrite=args.overwrite,
        protected_paths=(
            primal,
            reverse_executable,
            cases_root,
            kernel_source,
            runtime_source,
            replay_validation_dir,
            replay_report_path,
            kernel_build_report,
            kernel_source_manifest,
            coupled_build_manifest,
            coupled_build_input_manifest,
            coupled_build_snapshot,
            generator_path,
            manifest_path,
            *protected_case_files,
        ),
        project_root=root,
        owner="validate_scalar_taylor",
    )
    with output_transaction as output:
        verified_inputs = output / "verified_inputs"
        verified_inputs.mkdir()
        staged_kernel_source = verified_inputs / "kernel_source"
        shutil.copytree(kernel_source, staged_kernel_source)
        if (
            tangent_audit.tree_file_hashes(staged_kernel_source)
            != kernel_source_hashes
        ):
            raise RuntimeError("staged kernel source differs from verified source")
        staged_runtime_source = verified_inputs / "runtime_source"
        shutil.copytree(runtime_source, staged_runtime_source)
        if (
            tangent_audit.tree_file_hashes(staged_runtime_source)
            != runtime_source_hashes
        ):
            raise RuntimeError("staged runtime source differs from verified source")
        staged_manifest = verified_inputs / "state_manifest.json"
        staged_manifest.write_bytes(manifest_path.read_bytes())
        if tangent_audit.sha256(staged_manifest) != state_manifest_sha256:
            raise RuntimeError("state manifest changed while staging")
        staged_primal = verified_inputs / primal.name
        staged_primal.write_bytes(primal.read_bytes())
        if dot_audit.sha256(staged_primal) != dot_audit.sha256(primal):
            raise RuntimeError("primal executable changed while staging")
        staged_primal.chmod(staged_primal.stat().st_mode | 0o100)
        staged_cases: dict[str, dict[str, Path | str]] = {}
        for case_label, case_files in resolved_cases.items():
            staged_case: dict[str, Path | str] = {"layout": case_files["layout"]}
            for key in ("initial_state", "path", "gradient", "restart", "run_report"):
                source_file = case_files[key]
                assert isinstance(source_file, Path)
                destination = verified_inputs / "cases" / case_label / source_file.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                expected = case_input_hashes[case_label][key]
                destination.write_bytes(source_file.read_bytes())
                if dot_audit.sha256(destination) != expected:
                    raise RuntimeError(
                        f"case input changed while staging: {case_label}/{key}"
                    )
                staged_case[key] = destination
            staged_cases[case_label] = staged_case
        private_input_trees = {
            "kernel source": (staged_kernel_source, kernel_source_hashes),
            "runtime source": (staged_runtime_source, runtime_source_hashes),
        }
        private_input_files = {
            "state manifest": (staged_manifest, state_manifest_sha256),
            "primal executable": (
                staged_primal,
                build_provenance["primal_executable_sha256"],
            ),
            **{
                f"case {case_label}/{key}": (
                    path,
                    case_input_hashes[case_label][key],
                )
                for case_label, case_files in staged_cases.items()
                for key, path in case_files.items()
                if isinstance(path, Path)
            },
        }
        private_input_binding = tangent_audit.verify_private_staged_inputs(
            trees=private_input_trees,
            files=private_input_files,
        )
        manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
        _, independent, segments = tangent_audit.active_masks(manifest)
        source = staged_runtime_source
        primal = staged_primal
        resolved_cases = staged_cases

        case_reports: list[dict] = []
        for case_label in args.case_labels:
            case_files = resolved_cases[case_label]
            initial_state = case_files["initial_state"]
            path_file = case_files["path"]
            gradient_file = case_files["gradient"]
            restart = case_files["restart"]
            assert isinstance(initial_state, Path)
            assert isinstance(path_file, Path)
            assert isinstance(gradient_file, Path)
            assert isinstance(restart, Path)
            for required in (initial_state, path_file, gradient_file, restart):
                if not required.is_file():
                    raise FileNotFoundError(required)

            pre_nt = read_initial_nt(initial_state)
            x = tangent_audit.state_real(initial_state)
            gradient = np.fromfile(gradient_file, dtype=np.float32)
            if gradient.shape != (NREAL,) or not np.all(np.isfinite(gradient)):
                raise RuntimeError(f"invalid gradient for {case_label}")
            if args.direction_mode == "adjoint_aligned":
                direction = np.zeros(NREAL, dtype=np.float32)
                direction[independent] = gradient[independent]
                direction_norm = float(np.linalg.norm(direction.astype(np.float64)))
                if not direction_norm:
                    raise RuntimeError(f"zero independent gradient for {case_label}")
                direction = np.asarray(direction / direction_norm, dtype=np.float32)
            else:
                direction = tangent_audit.scaled_direction(
                    x, segments, seed=args.direction_seed, independent_only=True
                )
            derivative = dot_audit.precise_dot(direction, gradient)

            path_final, path_tape = read_path_final_real_and_tape(path_file)
            case_dir = output / case_label
            case_dir.mkdir(parents=True)
            direction_file = case_dir / "direction.bin"
            direction.tofile(direction_file)
            baseline_dir = case_dir / "baseline"
            tangent_audit.prepare(
                generator,
                source_dir=source,
                executable=primal,
                run_dir=baseline_dir,
                restart=restart,
                pre_nt=pre_nt,
                steps=NSTEPS,
            )
            shutil.copy2(initial_state, baseline_dir / "kernel_input_state.bin")
            baseline_runtime = tangent_audit.run(baseline_dir)
            baseline_final = tangent_audit.state_real(
                baseline_dir / "kernel_final_state.bin"
            )
            baseline_tape = np.fromfile(
                baseline_dir / "kernel_branch_tape.bin", dtype=np.int32
            ).reshape((NSTEPS, TAPE_WIDTH))
            baseline_double = canonical_double(baseline_final)
            baseline_float = float(np.float32(baseline_double))

            records: list[dict] = []
            for h in args.steps:
                sides: dict[str, dict] = {}
                perturbed_vectors: dict[str, np.ndarray] = {}
                for sign, name in ((1.0, "plus"), (-1.0, "minus")):
                    side_dir = case_dir / f"h_{h:.10g}_{name}"
                    tangent_audit.prepare(
                        generator,
                        source_dir=source,
                        executable=primal,
                        run_dir=side_dir,
                        restart=restart,
                        pre_nt=pre_nt,
                        steps=NSTEPS,
                    )
                    perturbed = np.asarray(
                        x.astype(np.float64) + sign * h * direction.astype(np.float64),
                        dtype=np.float32,
                    )
                    perturbed_vectors[name] = perturbed
                    tangent_audit.replace_state_real(
                        initial_state, side_dir / "kernel_input_state.bin", perturbed
                    )
                    runtime = tangent_audit.run(side_dir)
                    final = tangent_audit.state_real(
                        side_dir / "kernel_final_state.bin"
                    )
                    tape = np.fromfile(
                        side_dir / "kernel_branch_tape.bin", dtype=np.int32
                    ).reshape((NSTEPS, TAPE_WIDTH))
                    value_double = canonical_double(final)
                    sides[name] = {
                        "value_double": value_double,
                        "value_float32": float(np.float32(value_double)),
                        "runtime_seconds": runtime,
                        "tape": tape,
                    }

                plus = sides["plus"]["value_double"]
                minus = sides["minus"]["value_double"]
                plus32 = sides["plus"]["value_float32"]
                minus32 = sides["minus"]["value_float32"]
                fd = (plus - minus) / (2.0 * h)
                fd32 = (plus32 - minus32) / (2.0 * h)
                plus_remainder = abs(plus - baseline_double - h * derivative)
                minus_remainder = abs(minus - baseline_double + h * derivative)
                centered_remainder = abs(plus - minus - 2.0 * h * derivative)
                rounded_plus_remainder = abs(plus32 - baseline_float - h * derivative)
                rounded_minus_remainder = abs(minus32 - baseline_float + h * derivative)
                actual_centered_direction = (
                    perturbed_vectors["plus"].astype(np.float64)
                    - perturbed_vectors["minus"].astype(np.float64)
                ) / (2.0 * h)
                direction_rounding_error = tangent_audit.relative_l2(
                    actual_centered_direction, direction.astype(np.float64), independent
                )
                records.append(
                    {
                        "h": h,
                        "plus_tape": tape_delta(sides["plus"]["tape"], baseline_tape),
                        "minus_tape": tape_delta(sides["minus"]["tape"], baseline_tape),
                        "both_major_tapes_match_baseline": bool(
                            np.array_equal(sides["plus"]["tape"], baseline_tape)
                            and np.array_equal(sides["minus"]["tape"], baseline_tape)
                        ),
                        "plus_objective_double_reduction": plus,
                        "minus_objective_double_reduction": minus,
                        "centered_fd_double_reduction": fd,
                        "centered_fd_float32_publication": fd32,
                        "adjoint_directional_derivative": derivative,
                        "centered_fd_absolute_defect": abs(fd - derivative),
                        "centered_fd_relative_defect": abs(fd - derivative)
                        / max(abs(derivative), np.finfo(float).tiny),
                        "plus_first_order_taylor_remainder": plus_remainder,
                        "minus_first_order_taylor_remainder": minus_remainder,
                        "centered_first_order_taylor_remainder": centered_remainder,
                        "plus_float32_publication_taylor_remainder": (
                            rounded_plus_remainder
                        ),
                        "minus_float32_publication_taylor_remainder": (
                            rounded_minus_remainder
                        ),
                        "taylor_remainder_scale_h_times_derivative": abs(
                            h * derivative
                        ),
                        "centered_input_direction_rounding_relative_l2": (
                            direction_rounding_error
                        ),
                        "plus_runtime_seconds": sides["plus"]["runtime_seconds"],
                        "minus_runtime_seconds": sides["minus"]["runtime_seconds"],
                    }
                )

            case_reports.append(
                {
                    "case": case_label,
                    "case_layout": case_files["layout"],
                    "initial_nt": pre_nt,
                    "direction_seed": args.direction_seed,
                    "direction_mode": args.direction_mode,
                    "direction_scope": "independent continuous controls",
                    "direction_sha256": dot_audit.sha256(direction_file),
                    "gradient_sha256": dot_audit.sha256(gradient_file),
                    "gradient_file": str(gradient_file),
                    "gradient_l2_norm": float(
                        np.linalg.norm(gradient.astype(np.float64))
                    ),
                    "direction_l2_norm": float(
                        np.linalg.norm(direction.astype(np.float64))
                    ),
                    "adjoint_directional_derivative": derivative,
                    "baseline_runtime_seconds": baseline_runtime,
                    "baseline_objective_double_reduction": baseline_double,
                    "baseline_objective_float32_publication": baseline_float,
                    "baseline_matches_bound_path_bitwise": bool(
                        np.array_equal(baseline_final, path_final)
                    ),
                    "baseline_major_tape_matches_bound_path": bool(
                        np.array_equal(baseline_tape, path_tape)
                    ),
                    "baseline_major_tape": baseline_tape.tolist(),
                    "steps": records,
                }
            )

        if (
            verify_taylor_build(
                primal=bound_primal,
                source=bound_kernel_source,
                kernel_build_report=kernel_build_report,
                kernel_source_manifest=kernel_source_manifest,
                coupled_build_manifest=coupled_build_manifest,
                coupled_build_input_manifest=coupled_build_input_manifest,
                coupled_build_snapshot=coupled_build_snapshot,
                generator_path=generator_path,
            )
            != build_provenance
        ):
            raise RuntimeError("Taylor build provenance changed during validation")
        if (
            tangent_audit.tree_file_hashes(bound_kernel_source)
            != kernel_source_hashes
        ):
            raise RuntimeError("kernel source changed during validation")
        if (
            tangent_audit.tree_file_hashes(bound_runtime_source)
            != runtime_source_hashes
        ):
            raise RuntimeError("runtime source changed during validation")
        if (
            tangent_audit.verify_replay_report_binding(
                replay_report_path,
                kernel_build_report=kernel_build_report,
                primal=bound_primal,
            )
            != replay_binding
        ):
            raise RuntimeError("replay-validation producer binding changed")
        if (
            tangent_audit.verify_replay_runtime_source(
                bound_runtime_source,
                replay_report_path,
                kernel_build_report=kernel_build_report,
                generator=generator,
            )
            != runtime_source_binding
        ):
            raise RuntimeError("runtime-source provenance changed during validation")
        for case_label, case_files in (
            (label, resolve_case_files(cases_root / label, args.gradient_name))
            for label in args.case_labels
        ):
            current = dot_audit.verify_scalar_gradient_producer(
                run_report_path=case_files["run_report"],
                path_file=case_files["path"],
                gradient_file=case_files["gradient"],
                reverse_executable=reverse_executable,
                coupled_build_manifest=coupled_build_manifest,
                coupled_build_input_manifest=coupled_build_input_manifest,
            )
            if current != producer_bindings[case_label]:
                raise RuntimeError(
                    f"gradient producer changed during validation: {case_label}"
                )
            for key, path in case_files.items():
                if (
                    isinstance(path, Path)
                    and dot_audit.sha256(path) != case_input_hashes[case_label][key]
                ):
                    raise RuntimeError(
                        f"case input changed during validation: {case_label}/{key}"
                    )
        if tangent_audit.sha256(manifest_path) != state_manifest_sha256:
            raise RuntimeError("state manifest changed during validation")
        if tangent_audit.sha256(validator_path) != validator_sha256:
            raise RuntimeError("scalar Taylor validator changed during validation")
        if (
            tangent_audit.verify_private_staged_inputs(
                trees=private_input_trees,
                files=private_input_files,
            )
            != private_input_binding
        ):
            raise RuntimeError("private staged-input binding changed")

        report = {
            "schema_version": 1,
            "status": "completed_not_certified",
            "acceptance_thresholds": None,
            "certification_assessed": False,
            "created_utc": datetime.now(UTC).isoformat(),
            "validator_script_sha256": validator_sha256,
            "claim": (
                "Centered finite-difference and first-order Taylor tests of the "
                "31-step scalar reverse gradient against the independently validated "
                "forward oracle. The three-value major tape does not observe every "
                "internal branch. This report records convergence data but applies no "
                "automatic acceptance threshold and is not certification."
            ),
            "arithmetic_note": (
                "Both the exact float32 publication and the preceding sequential "
                "float64 Nino-3 reduction are reported. The latter removes only the "
                "last scalar cast; the evolved state remains authentic float32 ZC."
            ),
            "primal_executable": str(primal),
            "primal_executable_sha256": dot_audit.sha256(primal),
            "build_provenance": build_provenance,
            "replay_validation_binding": replay_binding,
            "runtime_source_binding": runtime_source_binding,
            "state_manifest_sha256": state_manifest_sha256,
            "private_staged_input_binding": private_input_binding,
            "gradient_producer_bindings": producer_bindings,
            "cases": case_reports,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
