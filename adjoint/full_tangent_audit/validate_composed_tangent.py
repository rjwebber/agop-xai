#!/usr/bin/env python3
"""Validate a 31-step ZC tangent by composing validated one-step maps.

The Tapenade-transformed primal trajectory is deliberately discarded after
each step.  Every tangent step is restarted from the corresponding state of
the independently validated forward trajectory.  This is the forward-mode
counterpart of the checkpointed one-step reverse sweep intended for the
eventual adjoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import validate_tangent as audit
from output_safety import transactional_output


def load_cases(args: argparse.Namespace) -> tuple[list[dict], Path | None]:
    """Load only explicitly named or replay-report-bound checkpoints."""

    replay_report_path: Path | None = None
    cases: list[dict] = []
    if args.replay_validation_dir is not None:
        replay_dir = args.replay_validation_dir.resolve()
        replay_report_path = replay_dir / "replay_report.json"
        report = json.loads(replay_report_path.read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise RuntimeError("replay-validation report is not passed")
        locked = report.get("locked_checkpoint_contract")
        if not isinstance(locked, dict) or not locked:
            raise RuntimeError("replay-validation report has no checkpoint contract")
        for index, (label, specification) in enumerate(
            sorted(locked.items())
        ):
            cases.append(
                {
                    "label": label,
                    "restart": replay_dir
                    / "verified_inputs/checkpoints"
                    / f"{label}.hst",
                    "pre_nt": int(specification["pre_nt"]),
                    "advance": 0,
                    "expected_sha256": specification["sha256"],
                    "seed": 4200 + index,
                }
            )
    else:
        for index, (label, restart, pre_nt, advance) in enumerate(
            args.checkpoint or []
        ):
            restart_path = Path(restart).resolve()
            cases.append(
                {
                    "label": label,
                    "restart": restart_path,
                    "pre_nt": int(pre_nt),
                    "advance": int(advance),
                    "expected_sha256": audit.sha256(restart_path),
                    "seed": 4200 + index,
                }
            )
    return cases, replay_report_path


def prepare_initial_state(
    generator,
    *,
    source: Path,
    primal: Path,
    case: dict,
    case_root: Path,
) -> tuple[Path, Path, int]:
    """Materialize the exact packed state and matching history restart."""
    restart = Path(case["restart"])
    pre_nt = int(case["pre_nt"])
    if case["advance"]:
        run_dir = case_root / "advance_to_validation_state"
        audit.prepare(
            generator,
            source_dir=source,
            executable=primal,
            run_dir=run_dir,
            restart=restart,
            pre_nt=pre_nt,
            steps=int(case["advance"]),
        )
        audit.run(run_dir)
        return (
            run_dir / "kernel_final_state.bin",
            run_dir / "outhst",
            pre_nt + int(case["advance"]),
        )

    run_dir = case_root / "capture_initial_state"
    audit.prepare(
        generator,
        source_dir=source,
        executable=primal,
        run_dir=run_dir,
        restart=restart,
        pre_nt=pre_nt,
        steps=1,
    )
    audit.run(run_dir)
    return run_dir / "kernel_initial_state.bin", restart, pre_nt


def run_direct_window(
    generator,
    *,
    source: Path,
    executable: Path,
    run_dir: Path,
    restart: Path,
    state_path: Path,
    pre_nt: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    audit.prepare(
        generator,
        source_dir=source,
        executable=executable,
        run_dir=run_dir,
        restart=restart,
        pre_nt=pre_nt,
        steps=steps,
    )
    shutil.copy2(state_path, run_dir / "kernel_input_state.bin")
    audit.run(run_dir)
    return (
        audit.state_real(run_dir / "kernel_final_state.bin"),
        np.fromfile(run_dir / "kernel_branch_tape.bin", dtype=np.int32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[0.2, 0.1, 0.05, 0.025, 0.02, 0.01, 0.005],
    )
    parser.add_argument(
        "--local-steps",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Also test each individual map along the validated forward-oracle "
            "trajectory at these centered-difference step sizes."
        ),
    )
    parser.add_argument("--case-labels", nargs="+")
    parser.add_argument(
        "--tangent-executable",
        type=Path,
        required=True,
        help=(
            "O3 one-step tangent executable from the generation being audited; "
            "this must match one_step_executable_sha256 in the audit build manifest."
        ),
    )
    parser.add_argument(
        "--primal-executable",
        type=Path,
        required=True,
        help="Independently validated forward-oracle executable.",
    )
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
            "Full runnable source tree. With --replay-validation-dir this "
            "defaults to verified_inputs/source from that replay validation; "
            "it is required with explicit --checkpoint inputs."
        ),
    )
    parser.add_argument("--generation-provenance", type=Path, required=True)
    parser.add_argument("--tangent-build-manifest", type=Path, required=True)
    parser.add_argument("--tangent-build-inputs", type=Path, required=True)
    parser.add_argument("--coupled-build-manifest", type=Path, required=True)
    parser.add_argument("--coupled-build-input-manifest", type=Path, required=True)
    parser.add_argument("--coupled-build-snapshot", type=Path, required=True)
    parser.add_argument("--kernel-build-report", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        nargs=4,
        action="append",
        metavar=("LABEL", "RESTART", "PRE_NT", "ADVANCE"),
    )
    checkpoint_group.add_argument("--replay-validation-dir", type=Path)
    args = parser.parse_args()
    validator_path = Path(__file__).resolve()
    validator_sha256 = audit.sha256(validator_path)

    root = args.project_root.resolve()
    output_requested = args.output_dir

    sys.path.insert(0, str(root / "scripts"))
    import generate_fresh_zc_dataset as generator  # noqa: PLC0415

    kernel_source = args.kernel_source.resolve()
    primal = args.primal_executable.resolve()
    tangent = args.tangent_executable.resolve()
    generator_path = root / "scripts/generate_fresh_zc_dataset.py"
    generation_provenance = args.generation_provenance.resolve()
    tangent_build_manifest = args.tangent_build_manifest.resolve()
    tangent_build_inputs = args.tangent_build_inputs.resolve()
    coupled_build_manifest = args.coupled_build_manifest.resolve()
    coupled_build_input_manifest = args.coupled_build_input_manifest.resolve()
    coupled_build_snapshot = args.coupled_build_snapshot.resolve()
    kernel_build_report = args.kernel_build_report.resolve()
    manifest_path = root / "adjoint/fortran_kernel/state_manifest.json"
    state_manifest_sha256 = audit.sha256(manifest_path)

    cases, replay_report_path = load_cases(args)
    if args.runtime_source is not None:
        runtime_source = args.runtime_source.resolve()
    elif replay_report_path is not None:
        runtime_source = replay_report_path.parent / "verified_inputs/source"
    else:
        parser.error("--runtime-source is required with explicit --checkpoint")
    if args.case_labels:
        requested = set(args.case_labels)
        available = {case["label"] for case in cases}
        if unknown := requested - available:
            parser.error(f"unknown case labels: {sorted(unknown)}")
        cases = [case for case in cases if case["label"] in requested]

    for required in (
        primal,
        tangent,
        generation_provenance,
        tangent_build_manifest,
        tangent_build_inputs,
        coupled_build_manifest,
        coupled_build_input_manifest,
        kernel_build_report,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    for required in (kernel_source, runtime_source, coupled_build_snapshot):
        if not required.is_dir():
            raise FileNotFoundError(required)
    for case in cases:
        restart = Path(case["restart"])
        if not restart.is_file():
            raise FileNotFoundError(restart)
        if audit.sha256(restart) != case["expected_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {case['label']}")

    build_provenance = audit.verify_audit_build(
        tangent=tangent,
        primal=primal,
        source=kernel_source,
        generation_provenance=generation_provenance,
        tangent_build_manifest=tangent_build_manifest,
        tangent_build_inputs=tangent_build_inputs,
        coupled_build_manifest=coupled_build_manifest,
        coupled_build_input_manifest=coupled_build_input_manifest,
        coupled_build_snapshot=coupled_build_snapshot,
        kernel_build_report=kernel_build_report,
        generator_path=generator_path,
    )
    replay_binding = (
        audit.verify_replay_report_binding(
            replay_report_path,
            kernel_build_report=kernel_build_report,
            primal=primal,
        )
        if replay_report_path is not None
        else None
    )
    runtime_source_binding = (
        audit.verify_replay_runtime_source(
            runtime_source,
            replay_report_path,
            kernel_build_report=kernel_build_report,
            generator=generator,
        )
        if replay_report_path is not None
        else {
            "authentication": "explicit_runtime_source_not_replay_authenticated",
            "runtime_source_complete_tree_sha256": audit.file_mapping_sha256(
                audit.tree_file_hashes(runtime_source)
            ),
            "runtime_source_complete_tree_entry_count": len(
                audit.tree_file_hashes(runtime_source)
            ),
        }
    )
    kernel_source_hashes = audit.tree_file_hashes(kernel_source)
    runtime_source_hashes = audit.tree_file_hashes(runtime_source)
    checkpoint_hashes = {
        case["label"]: audit.sha256(Path(case["restart"])) for case in cases
    }
    bound_kernel_source = kernel_source
    bound_runtime_source = runtime_source
    bound_primal = primal
    bound_tangent = tangent
    bound_cases = [dict(case) for case in cases]
    replay_report_sha256 = (
        audit.sha256(replay_report_path) if replay_report_path is not None else None
    )

    output_transaction = transactional_output(
        output_requested,
        overwrite=args.overwrite,
        protected_paths=(
            kernel_source,
            runtime_source,
            primal,
            tangent,
            manifest_path,
            generation_provenance,
            tangent_build_manifest,
            tangent_build_inputs,
            coupled_build_manifest,
            coupled_build_input_manifest,
            coupled_build_snapshot,
            kernel_build_report,
            generator_path,
            *(path for path in (replay_report_path,) if path is not None),
            *(Path(case["restart"]) for case in cases),
        ),
        project_root=root,
        owner="validate_composed_tangent",
    )
    with output_transaction as output:
        verified_inputs = output / "verified_inputs"
        verified_inputs.mkdir()
        staged_kernel_source = verified_inputs / "kernel_source"
        shutil.copytree(kernel_source, staged_kernel_source)
        if audit.tree_file_hashes(staged_kernel_source) != kernel_source_hashes:
            raise RuntimeError("staged kernel source differs from verified source")
        staged_runtime_source = verified_inputs / "runtime_source"
        shutil.copytree(runtime_source, staged_runtime_source)
        if audit.tree_file_hashes(staged_runtime_source) != runtime_source_hashes:
            raise RuntimeError("staged runtime source differs from verified source")
        staged_manifest = verified_inputs / "state_manifest.json"
        staged_manifest.write_bytes(manifest_path.read_bytes())
        if audit.sha256(staged_manifest) != state_manifest_sha256:
            raise RuntimeError("state manifest changed while staging")
        staged_primal = verified_inputs / primal.name
        staged_tangent = verified_inputs / tangent.name
        for original, staged, expected in (
            (
                primal,
                staged_primal,
                build_provenance["primal_executable_sha256"],
            ),
            (
                tangent,
                staged_tangent,
                build_provenance["one_step_executable_sha256"],
            ),
        ):
            staged.write_bytes(original.read_bytes())
            if audit.sha256(staged) != expected:
                raise RuntimeError(f"executable changed while staging: {original}")
            staged.chmod(staged.stat().st_mode | 0o100)
        staged_cases: list[dict] = []
        checkpoint_dir = verified_inputs / "checkpoints"
        checkpoint_dir.mkdir()
        for case in cases:
            staged_restart = checkpoint_dir / f"{case['label']}.hst"
            staged_restart.write_bytes(Path(case["restart"]).read_bytes())
            if audit.sha256(staged_restart) != checkpoint_hashes[case["label"]]:
                raise RuntimeError(f"checkpoint changed while staging: {case['label']}")
            staged_cases.append({**case, "restart": staged_restart})
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
            "tangent executable": (
                staged_tangent,
                build_provenance["one_step_executable_sha256"],
            ),
            **{
                f"checkpoint {case['label']}": (
                    Path(case["restart"]),
                    checkpoint_hashes[case["label"]],
                )
                for case in staged_cases
            },
        }
        private_input_binding = audit.verify_private_staged_inputs(
            trees=private_input_trees,
            files=private_input_files,
        )
        manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
        active, independent, segments = audit.active_masks(manifest)
        source = staged_runtime_source
        primal = staged_primal
        tangent = staged_tangent
        cases = staged_cases

        report_cases: list[dict] = []
        for case in cases:
            case_root = output / case["label"]
            initial_state, initial_restart, initial_nt = prepare_initial_state(
                generator,
                source=source,
                primal=primal,
                case=case,
                case_root=case_root,
            )
            x = audit.state_real(initial_state)
            direction = audit.scaled_direction(
                x,
                segments,
                seed=int(case["seed"]),
                independent_only=True,
            )

            direct_baseline, direct_tape = run_direct_window(
                generator,
                source=source,
                executable=primal,
                run_dir=case_root / "direct_baseline",
                restart=initial_restart,
                state_path=initial_state,
                pre_nt=initial_nt,
                steps=args.window,
            )

            state_path = initial_state
            restart = initial_restart
            pre_nt = initial_nt
            tangent_vector = direction.copy()
            composed_tape: list[np.ndarray] = []
            one_step_records: list[dict] = []
            for step in range(args.window):
                step_root = case_root / "composition" / f"step_{step + 1:03d}"
                certified_current = audit.state_real(state_path)
                tangent_input = tangent_vector.copy()
                certified_dir = step_root / "forward_oracle"
                audit.prepare(
                    generator,
                    source_dir=source,
                    executable=primal,
                    run_dir=certified_dir,
                    restart=restart,
                    pre_nt=pre_nt,
                    steps=1,
                )
                shutil.copy2(state_path, certified_dir / "kernel_input_state.bin")
                audit.run(certified_dir)
                certified_next = audit.state_real(
                    certified_dir / "kernel_final_state.bin"
                )
                certified_tape = np.fromfile(
                    certified_dir / "kernel_branch_tape.bin", dtype=np.int32
                )

                tangent_dir = step_root / "tangent"
                audit.prepare(
                    generator,
                    source_dir=source,
                    executable=tangent,
                    run_dir=tangent_dir,
                    restart=restart,
                    pre_nt=pre_nt,
                    steps=1,
                )
                shutil.copy2(state_path, tangent_dir / "kernel_input_state.bin")
                tangent_vector.tofile(tangent_dir / "tangent_direction.bin")
                (tangent_dir / "tangent_nsteps.txt").write_text("1\n")
                audit.run(tangent_dir)
                transformed_next = np.fromfile(
                    tangent_dir / "tangent_primal_out.bin", dtype=np.float32
                )
                tangent_vector = np.fromfile(
                    tangent_dir / "tangent_jv.bin", dtype=np.float32
                )
                tangent_tape = np.fromfile(
                    tangent_dir / "tangent_branch_tape.bin", dtype=np.int32
                )
                local_fd_records: list[dict] = []
                for h in args.local_steps:
                    local_outputs: dict[str, np.ndarray] = {}
                    local_tapes: dict[str, np.ndarray] = {}
                    for sign, name in ((1.0, "plus"), (-1.0, "minus")):
                        local_dir = step_root / f"local_h_{h:.8g}_{name}"
                        audit.prepare(
                            generator,
                            source_dir=source,
                            executable=primal,
                            run_dir=local_dir,
                            restart=restart,
                            pre_nt=pre_nt,
                            steps=1,
                        )
                        perturbed = np.asarray(
                            certified_current.astype(np.float64)
                            + sign * h * tangent_input.astype(np.float64),
                            dtype=np.float32,
                        )
                        audit.replace_state_real(
                            state_path,
                            local_dir / "kernel_input_state.bin",
                            perturbed,
                        )
                        audit.run(local_dir)
                        local_outputs[name] = audit.state_real(
                            local_dir / "kernel_final_state.bin"
                        )
                        local_tapes[name] = np.fromfile(
                            local_dir / "kernel_branch_tape.bin", dtype=np.int32
                        )
                    local_fd = (
                        local_outputs["plus"].astype(np.float64)
                        - local_outputs["minus"].astype(np.float64)
                    ) / (2.0 * h)
                    local_jv = tangent_vector.astype(np.float64)
                    local_fd_records.append(
                        {
                            "h": h,
                            "major_tape_matches_forward_oracle": bool(
                                np.array_equal(local_tapes["plus"], certified_tape)
                                and np.array_equal(local_tapes["minus"], certified_tape)
                            ),
                            "full_relative_l2_error": audit.relative_l2(
                                local_fd, local_jv
                            ),
                            "full_cosine": audit.cosine(local_fd, local_jv),
                            "full_absolute_l2_defect": float(
                                np.linalg.norm(local_fd - local_jv)
                            ),
                        }
                    )
                composed_tape.append(certified_tape)
                one_step_records.append(
                    {
                        "step": step + 1,
                        "pre_nt": pre_nt,
                        "branch_tape": certified_tape.tolist(),
                        "tangent_major_tape_matches_forward_oracle": bool(
                            np.array_equal(tangent_tape, certified_tape)
                        ),
                        "transformed_primal_relative_l2_difference": audit.relative_l2(
                            transformed_next.astype(np.float64),
                            certified_next.astype(np.float64),
                        ),
                        "transformed_primal_maximum_absolute_difference": float(
                            np.max(
                                np.abs(
                                    transformed_next.astype(np.float64)
                                    - certified_next.astype(np.float64)
                                )
                            )
                        ),
                        "tangent_norm": float(
                            np.linalg.norm(tangent_vector.astype(np.float64))
                        ),
                        "local_centered_difference_steps": local_fd_records,
                    }
                )

                state_path = certified_dir / "kernel_final_state.bin"
                restart = certified_dir / "outhst"
                pre_nt += 1

            composed_baseline = audit.state_real(state_path)
            composed_tape_array = np.concatenate(composed_tape)
            jv = tangent_vector.astype(np.float64)
            fd_records: list[dict] = []
            for h in args.steps:
                side_outputs: dict[str, np.ndarray] = {}
                side_tapes: dict[str, np.ndarray] = {}
                for sign, name in ((1.0, "plus"), (-1.0, "minus")):
                    side_dir = case_root / "finite_difference" / f"h_{h:.8g}_{name}"
                    audit.prepare(
                        generator,
                        source_dir=source,
                        executable=primal,
                        run_dir=side_dir,
                        restart=initial_restart,
                        pre_nt=initial_nt,
                        steps=args.window,
                    )
                    perturbed = np.asarray(
                        x.astype(np.float64) + sign * h * direction.astype(np.float64),
                        dtype=np.float32,
                    )
                    audit.replace_state_real(
                        initial_state,
                        side_dir / "kernel_input_state.bin",
                        perturbed,
                    )
                    audit.run(side_dir)
                    side_outputs[name] = audit.state_real(
                        side_dir / "kernel_final_state.bin"
                    )
                    side_tapes[name] = np.fromfile(
                        side_dir / "kernel_branch_tape.bin", dtype=np.int32
                    )

                fd = (
                    side_outputs["plus"].astype(np.float64)
                    - side_outputs["minus"].astype(np.float64)
                ) / (2.0 * h)
                nino3_jv = float(audit.nino3_canonical(tangent_vector))
                nino3_fd = float(
                    (
                        audit.nino3_canonical(side_outputs["plus"])
                        - audit.nino3_canonical(side_outputs["minus"])
                    )
                    / np.float32(2.0 * h)
                )
                fd_records.append(
                    {
                        "h": h,
                        "major_tape_matches_baseline": bool(
                            np.array_equal(side_tapes["plus"], direct_tape)
                            and np.array_equal(side_tapes["minus"], direct_tape)
                        ),
                        "full_relative_l2_error": audit.relative_l2(fd, jv),
                        "active_output_relative_l2_error": audit.relative_l2(
                            fd, jv, active
                        ),
                        "independent_output_relative_l2_error": audit.relative_l2(
                            fd, jv, independent
                        ),
                        "full_cosine": audit.cosine(fd, jv),
                        "tangent_norm": float(np.linalg.norm(jv)),
                        "full_absolute_l2_defect": float(np.linalg.norm(fd - jv)),
                        "nino3_tangent": nino3_jv,
                        "nino3_forward_oracle_fd": nino3_fd,
                        "nino3_absolute_error": abs(nino3_fd - nino3_jv),
                        "nino3_relative_error": abs(nino3_fd - nino3_jv)
                        / max(abs(nino3_jv), np.finfo(float).tiny),
                    }
                )

            report_cases.append(
                {
                    "checkpoint": case["label"],
                    "window_steps": args.window,
                    "input_state_sha256": audit.sha256(initial_state),
                    "direction_sha256": audit.sha256(
                        case_root
                        / "composition"
                        / "step_001"
                        / "tangent"
                        / "tangent_direction.bin"
                    ),
                    "direct_and_segmented_primal_bitwise_equal": bool(
                        np.array_equal(composed_baseline, direct_baseline)
                    ),
                    "direct_and_segmented_tapes_equal": bool(
                        np.array_equal(composed_tape_array, direct_tape)
                    ),
                    "direct_and_segmented_primal_relative_l2_difference": (
                        audit.relative_l2(
                            composed_baseline.astype(np.float64),
                            direct_baseline.astype(np.float64),
                        )
                    ),
                    "one_step_records": one_step_records,
                    "centered_difference_steps": fd_records,
                }
            )

        if (
            audit.verify_audit_build(
                tangent=bound_tangent,
                primal=bound_primal,
                source=bound_kernel_source,
                generation_provenance=generation_provenance,
                tangent_build_manifest=tangent_build_manifest,
                tangent_build_inputs=tangent_build_inputs,
                coupled_build_manifest=coupled_build_manifest,
                coupled_build_input_manifest=coupled_build_input_manifest,
                coupled_build_snapshot=coupled_build_snapshot,
                kernel_build_report=kernel_build_report,
                generator_path=generator_path,
            )
            != build_provenance
        ):
            raise RuntimeError("audit build provenance changed during validation")
        if audit.tree_file_hashes(bound_kernel_source) != kernel_source_hashes:
            raise RuntimeError("kernel source changed during validation")
        if audit.tree_file_hashes(bound_runtime_source) != runtime_source_hashes:
            raise RuntimeError("runtime source changed during validation")
        for case in bound_cases:
            if audit.sha256(Path(case["restart"])) != checkpoint_hashes[case["label"]]:
                raise RuntimeError(f"checkpoint changed: {case['label']}")
        if (
            replay_report_path is not None
            and audit.sha256(replay_report_path) != replay_report_sha256
        ):
            raise RuntimeError("replay-validation report changed during validation")
        if replay_report_path is not None and audit.verify_replay_report_binding(
            replay_report_path,
            kernel_build_report=kernel_build_report,
            primal=bound_primal,
        ) != replay_binding:
            raise RuntimeError("replay-validation producer binding changed")
        current_runtime_binding = (
            audit.verify_replay_runtime_source(
                bound_runtime_source,
                replay_report_path,
                kernel_build_report=kernel_build_report,
                generator=generator,
            )
            if replay_report_path is not None
            else {
                "authentication": "explicit_runtime_source_not_replay_authenticated",
                "runtime_source_complete_tree_sha256": audit.file_mapping_sha256(
                    audit.tree_file_hashes(bound_runtime_source)
                ),
                "runtime_source_complete_tree_entry_count": len(
                    audit.tree_file_hashes(bound_runtime_source)
                ),
            }
        )
        if current_runtime_binding != runtime_source_binding:
            raise RuntimeError("runtime-source provenance changed during validation")
        if audit.sha256(validator_path) != validator_sha256:
            raise RuntimeError("composed tangent validator changed during validation")
        if audit.sha256(manifest_path) != state_manifest_sha256:
            raise RuntimeError("state manifest changed during validation")
        if (
            audit.verify_private_staged_inputs(
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
            "scientific_claim": (
                "Major-tape-conditioned comparison of a long-window tangent formed "
                "by composing one-step Tapenade maps on an independently validated "
                "ZC trajectory. The tape observes only SST substeps, atmosphere "
                "iterations, and reset—not every internal branch. This report applies "
                "no threshold and is not adjoint certification."
            ),
            "window_steps": args.window,
            "composition": (
                "Discard the transformed primal after each step and re-anchor the next "
                "tangent step at the validated float32 forward-oracle state."
            ),
            "executables": {
                "forward_oracle_sha256": audit.sha256(primal),
                "tapenade_tangent_sha256": audit.sha256(tangent),
            },
            "cases": report_cases,
            "build_provenance": build_provenance,
            "checkpoint_inputs": [
                {
                    "label": case["label"],
                    "pre_nt": case["pre_nt"],
                    "advance": case["advance"],
                    "sha256": checkpoint_hashes[case["label"]],
                }
                for case in bound_cases
            ],
            "replay_validation_report_sha256": replay_report_sha256,
            "replay_validation_binding": replay_binding,
            "runtime_source_binding": runtime_source_binding,
            "state_manifest_sha256": state_manifest_sha256,
            "private_staged_input_binding": private_input_binding,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
