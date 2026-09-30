#!/usr/bin/env python3
"""Test a generic 31-step ZC reverse map with independent terminal seeds."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import validate_adjoint_dot as scalar_dot
import validate_tangent as tangent_audit
from output_safety import transactional_output

NREAL = tangent_audit.NREAL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8301, 8302, 8303])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--path-file", type=Path, required=True)
    parser.add_argument("--tangent-executable", type=Path, required=True)
    parser.add_argument("--reverse-executable", type=Path, required=True)
    parser.add_argument("--producer-run-report", type=Path, required=True)
    parser.add_argument("--tangent-build-manifest", type=Path, required=True)
    parser.add_argument("--tangent-build-inputs", type=Path, required=True)
    parser.add_argument("--coupled-build-manifest", type=Path, required=True)
    parser.add_argument("--coupled-build-input-manifest", type=Path, required=True)
    parser.add_argument("--coupled-build-snapshot", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--tangent-source-dir", type=Path, required=True)
    parser.add_argument("--reverse-source-dir", type=Path, required=True)
    args = parser.parse_args()
    validator_path = Path(__file__).resolve()
    validator_sha256 = scalar_dot.sha256(validator_path)

    root = args.project_root.resolve()
    output_requested = args.output_dir
    run_dir = args.run_dir.resolve()
    path_file = args.path_file.resolve()
    tangent_executable = args.tangent_executable.resolve()
    reverse_executable = args.reverse_executable.resolve()
    producer_run_report = args.producer_run_report.resolve()
    tangent_build_manifest = args.tangent_build_manifest.resolve()
    tangent_build_inputs = args.tangent_build_inputs.resolve()
    coupled_build_manifest = args.coupled_build_manifest.resolve()
    coupled_build_input_manifest = args.coupled_build_input_manifest.resolve()
    coupled_build_snapshot = args.coupled_build_snapshot.resolve()
    prepared_dir = args.prepared_dir.resolve()
    tangent_source_dir = args.tangent_source_dir.resolve()
    reverse_source_dir = args.reverse_source_dir.resolve()
    manifest_path = root / "adjoint/fortran_kernel/state_manifest.json"
    for required in (
        path_file,
        tangent_executable,
        reverse_executable,
        producer_run_report,
        tangent_build_manifest,
        tangent_build_inputs,
        coupled_build_manifest,
        coupled_build_input_manifest,
        manifest_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    for required in (
        run_dir,
        prepared_dir,
        tangent_source_dir,
        reverse_source_dir,
        coupled_build_snapshot,
    ):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if {
        "prepared": prepared_dir,
        "tangent": tangent_source_dir,
        "reverse": reverse_source_dir,
    } != {
        "prepared": coupled_build_snapshot / "prepared",
        "tangent": coupled_build_snapshot / "generated/tangent",
        "reverse": coupled_build_snapshot / "generated/reverse",
    }:
        raise RuntimeError("source directories are not from the coupled snapshot")

    state_manifest_sha256 = scalar_dot.sha256(manifest_path)
    prepared_hash, prepared_files = scalar_dot.tree_manifest(prepared_dir)
    tangent_hash, tangent_files = scalar_dot.tree_manifest(tangent_source_dir)
    reverse_hash, reverse_files = scalar_dot.tree_manifest(reverse_source_dir)
    tangent_build = scalar_dot.verify_tangent_build(
        tangent_executable=tangent_executable,
        tangent_build_manifest=tangent_build_manifest,
        tangent_build_inputs=tangent_build_inputs,
        coupled_build_manifest=coupled_build_manifest,
        coupled_build_input_manifest=coupled_build_input_manifest,
        coupled_build_snapshot=coupled_build_snapshot,
    )
    producer_binding = scalar_dot.verify_path_producer(
        run_report_path=producer_run_report,
        path_file=path_file,
        reverse_executable=reverse_executable,
        coupled_build_manifest=coupled_build_manifest,
        coupled_build_input_manifest=coupled_build_input_manifest,
    )
    runtime_manifest, runtime_binding = scalar_dot.verify_producer_runtime(
        run_dir, producer_run_report
    )

    output_transaction = transactional_output(
        output_requested,
        overwrite=args.overwrite,
        protected_paths=(
            run_dir,
            path_file,
            tangent_executable,
            reverse_executable,
            prepared_dir,
            tangent_source_dir,
            reverse_source_dir,
            producer_run_report,
            tangent_build_manifest,
            tangent_build_inputs,
            coupled_build_manifest,
            coupled_build_input_manifest,
            coupled_build_snapshot,
            manifest_path,
        ),
        project_root=root,
        owner="validate_generic_adjoint_dot",
    )
    with output_transaction as output:
        verified_inputs = output / "verified_inputs"
        verified_inputs.mkdir()
        staged_path = verified_inputs / path_file.name
        staged_tangent = verified_inputs / tangent_executable.name
        staged_reverse = verified_inputs / reverse_executable.name
        staged_manifest = verified_inputs / "state_manifest.json"
        for source, destination, expected_hash in (
            (path_file, staged_path, producer_binding["path_sha256"]),
            (
                tangent_executable,
                staged_tangent,
                scalar_dot.sha256(tangent_executable),
            ),
            (
                reverse_executable,
                staged_reverse,
                producer_binding["reverse_executable_sha256"],
            ),
            (manifest_path, staged_manifest, state_manifest_sha256),
        ):
            scalar_dot.stage_verified_file(source, destination, str(expected_hash))
        staged_tangent.chmod(staged_tangent.stat().st_mode | 0o100)
        staged_reverse.chmod(staged_reverse.stat().st_mode | 0o100)
        staged_run_dir = verified_inputs / "producer_runtime"
        scalar_dot.stage_producer_runtime(run_dir, staged_run_dir, runtime_manifest)

        # Parse the path and state layout only from private verified copies.
        initial = scalar_dot.read_initial_real(staged_path)
        manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
        _, _, segments = tangent_audit.active_masks(manifest)

        records: list[dict] = []
        for seed in args.seeds:
            case_dir = output / f"seed_{seed}"
            case_dir.mkdir()
            direction = tangent_audit.scaled_direction(
                initial, segments, seed=seed, independent_only=False
            )
            direction_file = case_dir / "direction.bin"
            direction.tofile(direction_file)

            rng = np.random.default_rng(seed + 100_000)
            terminal = rng.standard_normal(NREAL)
            terminal /= np.linalg.norm(terminal)
            terminal = np.asarray(terminal, dtype=np.float32)
            terminal_file = case_dir / "terminal_seed.bin"
            terminal.tofile(terminal_file)

            tangent_prefix = case_dir / "tangent_path"
            scalar_dot.run_bound_command(
                [
                    str(staged_tangent),
                    str(staged_path),
                    str(direction_file),
                    str(tangent_prefix),
                ],
                cwd=staged_run_dir,
                log_path=case_dir / "tangent.log",
                label="tangent run",
            )

            reverse_prefix = case_dir / "reverse"
            scalar_dot.run_bound_command(
                [
                    str(staged_reverse),
                    str(staged_path),
                    str(reverse_prefix),
                    str(terminal_file),
                ],
                cwd=staged_run_dir,
                log_path=case_dir / "reverse.log",
                label="reverse run",
            )

            jv = np.fromfile(case_dir / "tangent_path_jv.bin", dtype=np.float32)
            adjoint = np.fromfile(case_dir / "reverse_gradient.bin", dtype=np.float32)
            if (
                jv.shape != (NREAL,)
                or adjoint.shape != (NREAL,)
                or not np.all(np.isfinite(jv))
                or not np.all(np.isfinite(adjoint))
            ):
                raise RuntimeError(f"invalid derivative output for seed {seed}")

            lhs = scalar_dot.precise_dot(terminal, jv)
            rhs = scalar_dot.precise_dot(direction, adjoint)
            absolute_defect = abs(lhs - rhs)
            direction_norm = float(np.linalg.norm(direction.astype(np.float64)))
            terminal_norm = float(np.linalg.norm(terminal.astype(np.float64)))
            jv_norm = float(np.linalg.norm(jv.astype(np.float64)))
            adjoint_norm = float(np.linalg.norm(adjoint.astype(np.float64)))
            norm_scale = jv_norm * terminal_norm + direction_norm * adjoint_norm
            bilinear_relative_defect = absolute_defect / max(
                abs(lhs), abs(rhs), np.finfo(float).tiny
            )
            tangent_meta = scalar_dot.parse_metadata(case_dir / "tangent_path_jv.txt")
            reverse_meta = scalar_dot.parse_metadata(case_dir / "reverse_gradient.txt")
            if (
                reverse_meta.get("objective")
                != "custom terminal real-state linear functional"
                or reverse_meta.get("terminal_seed") != "custom_float32_stream"
                or reverse_meta.get("transitions") != "31"
                or reverse_meta.get("real_state_length") != str(NREAL)
            ):
                raise RuntimeError(
                    f"reverse metadata violates generic contract for seed {seed}"
                )
            records.append(
                {
                    "seed": seed,
                    "direction_sha256": scalar_dot.sha256(direction_file),
                    "terminal_seed_sha256": scalar_dot.sha256(terminal_file),
                    "direction_l2_norm": direction_norm,
                    "terminal_seed_l2_norm": terminal_norm,
                    "jv_l2_norm": jv_norm,
                    "adjoint_l2_norm": adjoint_norm,
                    "terminal_dot_jv": lhs,
                    "direction_dot_adjoint": rhs,
                    "absolute_defect": absolute_defect,
                    "bilinear_relative_defect": bilinear_relative_defect,
                    "scalar_symmetric_relative_defect": bilinear_relative_defect,
                    "norm_scaled_transpose_defect": absolute_defect
                    / max(norm_scale, np.finfo(float).tiny),
                    "norm_scaled_transpose_denominator": norm_scale,
                    "tangent_cpu_seconds": float(tangent_meta["cpu_seconds"]),
                    "reverse_cpu_seconds": float(reverse_meta["cpu_seconds"]),
                    "tangent_fwd_worst_max_abs_vs_bound_forward_path": float(
                        tangent_meta["fwd_real_worst_max_abs"]
                    ),
                    "reverse_fwd_worst_max_abs_vs_bound_forward_path": float(
                        reverse_meta["fwd_real_worst_max_abs"]
                    ),
                    "reverse_objective_kind": reverse_meta["objective"],
                    "reverse_terminal_seed_kind": reverse_meta["terminal_seed"],
                }
            )

        if (
            scalar_dot.verify_tangent_build(
                tangent_executable=tangent_executable,
                tangent_build_manifest=tangent_build_manifest,
                tangent_build_inputs=tangent_build_inputs,
                coupled_build_manifest=coupled_build_manifest,
                coupled_build_input_manifest=coupled_build_input_manifest,
                coupled_build_snapshot=coupled_build_snapshot,
            )
            != tangent_build
        ):
            raise RuntimeError("tangent build provenance changed during validation")
        if (
            scalar_dot.verify_path_producer(
                run_report_path=producer_run_report,
                path_file=path_file,
                reverse_executable=reverse_executable,
                coupled_build_manifest=coupled_build_manifest,
                coupled_build_input_manifest=coupled_build_input_manifest,
            )
            != producer_binding
        ):
            raise RuntimeError("reverse producer provenance changed during validation")
        if (
            scalar_dot.tree_manifest(prepared_dir)[0] != prepared_hash
            or scalar_dot.tree_manifest(tangent_source_dir)[0] != tangent_hash
            or scalar_dot.tree_manifest(reverse_source_dir)[0] != reverse_hash
        ):
            raise RuntimeError("coupled source tree changed during validation")
        if scalar_dot.sha256(validator_path) != validator_sha256:
            raise RuntimeError("generic dot validator changed during validation")
        if scalar_dot.sha256(manifest_path) != state_manifest_sha256:
            raise RuntimeError("state manifest changed during validation")
        if scalar_dot.sha256(staged_manifest) != state_manifest_sha256:
            raise RuntimeError("staged state manifest changed during validation")
        if scalar_dot.sha256(staged_path) != producer_binding["path_sha256"]:
            raise RuntimeError("staged forward path changed during validation")
        if scalar_dot.sha256(staged_tangent) != scalar_dot.sha256(
            tangent_executable
        ):
            raise RuntimeError("executed tangent changed during validation")
        if scalar_dot.sha256(staged_reverse) != producer_binding[
            "reverse_executable_sha256"
        ]:
            raise RuntimeError("executed reverse changed during validation")
        if (
            scalar_dot.verify_staged_producer_runtime(
                staged_run_dir, runtime_manifest
            )
            != runtime_manifest
        ):
            raise RuntimeError("staged producer runtime changed during validation")
        if scalar_dot.verify_producer_runtime(
            run_dir, producer_run_report
        ) != (runtime_manifest, runtime_binding):
            raise RuntimeError("producer runtime changed during validation")

        report = {
            "schema_version": 1,
            "status": "completed_not_certified",
            "acceptance_thresholds": None,
            "certification_assessed": False,
            "created_utc": datetime.now(UTC).isoformat(),
            "validator_script_sha256": validator_sha256,
            "claim": (
                "Generic 31-step tangent-adjoint transpose tests with random input "
                "directions over every manifest-active real-state entry and dense "
                "random terminal real-state seeds; "
                "both sweeps use one-step maps re-anchored on one bound forward path. "
                "The report includes both bilinear-relative and norm-product-scaled "
                "defects, applies no automatic threshold, and is not certification."
            ),
            "input_direction_scope": "manifest-active real-state entries",
            "state_real_length": NREAL,
            "transitions": 31,
            "bound_forward_path": str(path_file),
            "bound_forward_path_sha256": scalar_dot.sha256(path_file),
            "executables": {
                "tangent": str(tangent_executable),
                "tangent_sha256": scalar_dot.sha256(tangent_executable),
                "reverse": str(reverse_executable),
                "reverse_sha256": scalar_dot.sha256(reverse_executable),
            },
            "source_manifests": {
                "prepared_sha256": prepared_hash,
                "tangent_sha256": tangent_hash,
                "reverse_sha256": reverse_hash,
                "prepared_files": prepared_files,
                "tangent_files": tangent_files,
                "reverse_files": reverse_files,
            },
            "tangent_build_provenance": tangent_build,
            "path_producer_binding": producer_binding,
            "producer_runtime_binding": runtime_binding,
            "state_manifest_sha256": state_manifest_sha256,
            "tests": records,
            "maximum_norm_scaled_transpose_defect": max(
                item["norm_scaled_transpose_defect"] for item in records
            ),
            "maximum_bilinear_relative_defect": max(
                item["bilinear_relative_defect"] for item in records
            ),
            "maximum_absolute_defect": max(item["absolute_defect"] for item in records),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
