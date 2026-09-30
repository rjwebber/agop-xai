#!/usr/bin/env python3
"""Validate explicit-state pack/unpack and ZC kernel replay bit for bit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

AUDIT_HELPERS = Path(__file__).resolve().parents[1] / "full_tangent_audit"
if str(AUDIT_HELPERS) not in sys.path:
    sys.path.insert(0, str(AUDIT_HELPERS))
from output_safety import transactional_output  # noqa: E402

VALIDATOR_VERSION = "0.3.8"
EXPECTED_BUILD_REPORT_SCHEMA = 2
EXPECTED_KERNEL_VERSION = "0.1.4"
SUPPORTED_DATASET_GENERATOR_VERSIONS = {"1.0.0", "1.1.0"}
STATE_FILE_SIZE = 59148 * 4 + 5280 * 8 + 1 * 8 + 4 * 4 + 2 * 4
OUTPUT_MARKER = ".zc_kernel_replay_validation_output"
OUTPUT_MARKER_CONTENT = "managed-zc-kernel-replay-validation-output-v1\n"
LOCKED_CHECKPOINTS = {
    "extreme_el_nino": {
        "pre_nt": 424799,
        "sha256": "04647489b5ffce58411eb24a8fc248f223750106ebed124f1c3764bb82d2f825",
    },
    "extreme_la_nina": {
        "pre_nt": 417878,
        "sha256": "c451ee91b9b05cf930587025790c5fa408607595ff4f8cf7cf6df897c5dba4f7",
    },
    "neutral_member_04": {
        "pre_nt": 406358,
        "sha256": "62e392ff707836a3683422db580dbbd1268a4baa79555df2401a715acdef7c74",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_generator(project_root: Path):
    sys.path.insert(0, str(project_root / "scripts"))
    import generate_fresh_zc_dataset  # noqa: PLC0415

    return generate_fresh_zc_dataset


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return document


def _hash_mapping(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(digest, str)
        for key, digest in value.items()
    ):
        raise ValueError(f"{label} must be a string-to-string hash mapping")
    return value


def validate_build_provenance(
    build_dir: Path,
    *,
    generator: Any,
    builder_path: Path,
    generator_path: Path,
) -> dict[str, Any]:
    """Bind replay to the exact staged source, executables, and build report."""

    build_report_path = build_dir / "build_report.json"
    report = _json_object(build_report_path, label="kernel build report")
    if report.get("schema_version") != EXPECTED_BUILD_REPORT_SCHEMA:
        raise ValueError(
            "kernel build report schema_version does not match the validator"
        )
    if report.get("kernel_version") != EXPECTED_KERNEL_VERSION:
        raise ValueError(
            "kernel build report kernel_version does not match the validator"
        )

    reference = build_dir / "zc_reference"
    kernel = build_dir / "zc_kernel_replay"
    expected_executables = {
        reference: report.get("reference_executable_sha256"),
        kernel: report.get("kernel_executable_sha256"),
    }
    for executable, expected_sha256 in expected_executables.items():
        if not executable.is_file():
            raise FileNotFoundError(executable)
        observed_sha256 = sha256_file(executable)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"build executable hash mismatch: {executable}")

    source_manifest_path = build_dir / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    if sha256_file(source_manifest_path) != report.get("source_manifest_file_sha256"):
        raise RuntimeError("source_manifest.json hash does not match build report")
    original_manifest = _hash_mapping(
        _json_object(source_manifest_path, label="source manifest"),
        label="source manifest",
    )
    original_manifest_sha256 = generator.manifest_sha256(original_manifest)
    if original_manifest_sha256 != report.get("source_manifest_sha256"):
        raise RuntimeError("canonical source-manifest digest does not match report")
    source_manifest_digest_path = build_dir / "source_manifest.sha256"
    if not source_manifest_digest_path.is_file():
        raise FileNotFoundError(source_manifest_digest_path)
    if sha256_file(source_manifest_digest_path) != report.get(
        "source_manifest_digest_file_sha256"
    ):
        raise RuntimeError("source_manifest.sha256 hash does not match build report")
    if source_manifest_digest_path.read_text(encoding="utf-8") != (
        original_manifest_sha256 + "\n"
    ):
        raise RuntimeError(
            "source_manifest.sha256 content does not match canonical digest"
        )

    expected_staged_manifest = dict(original_manifest)
    for field in (
        "fresh_patch_sha256",
        "kernel_patch_sha256",
        "kernel_support_source_sha256",
    ):
        expected_staged_manifest.update(
            _hash_mapping(report.get(field), label=f"build report {field}")
        )
    source_dir = build_dir / "source"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    observed_staged_manifest = generator.source_manifest(source_dir)
    if observed_staged_manifest != expected_staged_manifest:
        raise RuntimeError(
            "staged build source does not match the source and patch manifests"
        )

    builder_hashes = _hash_mapping(
        report.get("builder_source_sha256"),
        label="build report builder_source_sha256",
    )
    expected_builder_hashes = {
        "build_kernel.py": sha256_file(builder_path),
        "generate_fresh_zc_dataset.py": sha256_file(generator_path),
    }
    if builder_hashes != expected_builder_hashes:
        raise RuntimeError("live builder/generator hashes do not match build report")

    generator_record = report.get("generator")
    if not isinstance(generator_record, dict):
        raise ValueError("kernel build report has no generator version record")
    if generator_record.get("script_version") != generator.SCRIPT_VERSION:
        raise RuntimeError("live generator version does not match build report")
    if (
        generator_record.get("generation_schema_version")
        != generator.GENERATION_SCHEMA_VERSION
    ):
        raise RuntimeError("live generator schema does not match build report")

    return {
        "build_report_sha256": sha256_file(build_report_path),
        "build_report_schema_version": report["schema_version"],
        "kernel_version": report["kernel_version"],
        "reference_executable_sha256": sha256_file(reference),
        "kernel_executable_sha256": sha256_file(kernel),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "source_manifest_digest_file_sha256": sha256_file(source_manifest_digest_path),
        "canonical_source_manifest_sha256": original_manifest_sha256,
        "staged_source_manifest_sha256": generator.manifest_sha256(
            observed_staged_manifest
        ),
        "generator_script_version": generator.SCRIPT_VERSION,
        "generator_schema_version": generator.GENERATION_SCHEMA_VERSION,
        "generator_script_sha256": sha256_file(generator_path),
        "builder_script_sha256": sha256_file(builder_path),
        "staged_source_file_sha256": observed_staged_manifest,
    }


def copy_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Copy a regular file and prove that the staged bytes are the bound bytes."""

    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    if sha256_file(source) != expected_sha256:
        raise RuntimeError(f"input changed before staging: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or sha256_file(destination) != expected_sha256:
        raise RuntimeError(f"staged input differs from verified source: {source}")
    return destination


def stage_validation_inputs(
    *,
    output_dir: Path,
    build_dir: Path,
    metadata_path: Path,
    checkpoint_cases: list[dict[str, Any]],
    build_provenance: dict[str, Any],
    generator: Any,
) -> tuple[Path, Path, Path, Path, list[dict[str, Any]]]:
    """Create the immutable private copies consumed by the long validation."""

    staged_root = output_dir / "verified_inputs"
    staged_root.mkdir()
    staged_source = staged_root / "source"
    staged_source.mkdir()
    expected_source = build_provenance["staged_source_file_sha256"]
    for relative_name, expected_sha256 in expected_source.items():
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe staged-source path: {relative_name!r}")
        copy_verified_file(
            build_dir / "source" / relative,
            staged_source / relative,
            expected_sha256=expected_sha256,
        )
    observed_source = generator.source_manifest(staged_source)
    complete_source: dict[str, str] = {}
    for path in staged_source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"staged source contains a symbolic link: {path}")
        if path.is_file():
            complete_source[path.relative_to(staged_source).as_posix()] = (
                sha256_file(path)
            )
        elif not path.is_dir():
            raise ValueError(f"staged source contains a special file: {path}")
    if observed_source != expected_source or complete_source != expected_source:
        raise RuntimeError("staged kernel source differs from verified build source")

    staged_reference = copy_verified_file(
        build_dir / "zc_reference",
        staged_root / "zc_reference",
        expected_sha256=build_provenance["reference_executable_sha256"],
    )
    staged_kernel = copy_verified_file(
        build_dir / "zc_kernel_replay",
        staged_root / "zc_kernel_replay",
        expected_sha256=build_provenance["kernel_executable_sha256"],
    )
    staged_reference.chmod(staged_reference.stat().st_mode | 0o100)
    staged_kernel.chmod(staged_kernel.stat().st_mode | 0o100)

    metadata_sha256 = sha256_file(metadata_path)
    staged_metadata = copy_verified_file(
        metadata_path,
        staged_root / "processed_metadata.json",
        expected_sha256=metadata_sha256,
    )
    staged_cases: list[dict[str, Any]] = []
    for checkpoint in checkpoint_cases:
        staged_restart = copy_verified_file(
            checkpoint["path"],
            staged_root / "checkpoints" / f"{checkpoint['label']}.hst",
            expected_sha256=checkpoint["expected_sha256"],
        )
        staged_cases.append({**checkpoint, "path": staged_restart})
    return (
        staged_source,
        staged_reference,
        staged_kernel,
        staged_metadata,
        staged_cases,
    )


def verify_staged_validation_inputs(
    *,
    source_dir: Path,
    reference_executable: Path,
    kernel_executable: Path,
    metadata_path: Path,
    checkpoint_cases: list[dict[str, Any]],
    build_provenance: dict[str, Any],
    metadata_sha256: str,
    generator: Any,
) -> dict[str, Any]:
    """Authenticate every private scientific input before and after replay."""

    expected_source = build_provenance["staged_source_file_sha256"]
    observed_source = generator.source_manifest(source_dir)
    complete_source: dict[str, str] = {}
    for path in source_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"staged replay source became a symlink: {path}")
        if path.is_file():
            complete_source[path.relative_to(source_dir).as_posix()] = sha256_file(path)
        elif not path.is_dir():
            raise RuntimeError(f"staged replay source contains a special file: {path}")
    if observed_source != expected_source or complete_source != expected_source:
        raise RuntimeError("staged replay source changed during validation")

    executable_sha256 = {
        reference_executable.name: build_provenance["reference_executable_sha256"],
        kernel_executable.name: build_provenance["kernel_executable_sha256"],
    }
    for path in (reference_executable, kernel_executable):
        expected = executable_sha256[path.name]
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"staged replay executable changed: {path.name}")
    if (
        metadata_path.is_symlink()
        or not metadata_path.is_file()
        or sha256_file(metadata_path) != metadata_sha256
    ):
        raise RuntimeError("staged replay metadata changed during validation")

    checkpoint_sha256: dict[str, str] = {}
    for checkpoint in checkpoint_cases:
        label = str(checkpoint["label"])
        path = Path(checkpoint["path"])
        expected = str(checkpoint["expected_sha256"])
        if label in checkpoint_sha256:
            raise RuntimeError(f"duplicate staged replay checkpoint label: {label}")
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"staged replay checkpoint changed: {label}")
        checkpoint_sha256[label] = expected

    return {
        "source_manifest_sha256": generator.manifest_sha256(observed_source),
        "source_file_count": len(observed_source),
        "executable_sha256": executable_sha256,
        "metadata_sha256": metadata_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }


def validate_processed_metadata(
    metadata: dict[str, Any],
    *,
    reference_executable_sha256: str,
) -> list[dict[str, Any]]:
    """Extract only the two locked event checkpoints from processed metadata."""

    if metadata.get("schema_version") != "fresh-zc-interpretability-v1":
        raise ValueError("unsupported processed-data metadata schema_version")
    script_version = metadata.get("script_version")
    if script_version not in SUPPORTED_DATASET_GENERATOR_VERSIONS:
        raise ValueError("unsupported processed-data generator script_version")
    raw_events = metadata.get("event_restart_checkpoints")
    if not isinstance(raw_events, list):
        raise ValueError("processed metadata event_restart_checkpoints must be a list")
    events_by_label: dict[str, dict[str, Any]] = {}
    for event in raw_events:
        if not isinstance(event, dict) or not isinstance(event.get("label"), str):
            raise ValueError("processed metadata contains an invalid event checkpoint")
        label = event["label"]
        if label in events_by_label:
            raise ValueError(f"duplicate event checkpoint label: {label}")
        events_by_label[label] = event
    required_labels = {"extreme_el_nino", "extreme_la_nina"}
    if set(events_by_label) != required_labels:
        raise ValueError("processed metadata does not contain the locked event labels")

    cases: list[dict[str, Any]] = []
    for label in sorted(required_labels):
        event = events_by_label[label]
        locked = LOCKED_CHECKPOINTS[label]
        observed = {
            "pre_nt": event.get("pre_input_checkpoint_nt"),
            "sha256": event.get("checkpoint_sha256"),
        }
        if observed != locked:
            raise RuntimeError(f"locked checkpoint triple mismatch for {label}")
        if event.get("event_replay_executable_sha256") != reference_executable_sha256:
            raise RuntimeError(f"processed checkpoint executable mismatch for {label}")
        checkpoint_file = event.get("checkpoint_file")
        if not isinstance(checkpoint_file, str) or not checkpoint_file:
            raise ValueError(f"missing checkpoint_file for {label}")
        cases.append(
            {
                "label": label,
                "checkpoint_file": checkpoint_file,
                "expected_sha256": locked["sha256"],
                "pre_nt": locked["pre_nt"],
            }
        )
    return cases


def validate_locked_checkpoint_cases(checkpoint_cases: list[dict[str, Any]]) -> None:
    """Reject mutable/unknown labels and mismatched NT/hash combinations."""

    labels: set[str] = set()
    for checkpoint in checkpoint_cases:
        label = checkpoint.get("label")
        if label in labels:
            raise ValueError(f"duplicate checkpoint label: {label}")
        labels.add(label)
        if label not in LOCKED_CHECKPOINTS:
            raise ValueError(f"checkpoint label is not in the locked contract: {label}")
        locked = LOCKED_CHECKPOINTS[label]
        observed = {
            "pre_nt": checkpoint.get("pre_nt"),
            "sha256": checkpoint.get("expected_sha256"),
        }
        if observed != locked:
            raise RuntimeError(f"locked checkpoint triple mismatch for {label}")


def guarded_prepare_output(
    output_dir: Path,
    *,
    overwrite: bool,
    protected_paths: tuple[Path, ...],
    project_root: Path,
) -> Path:
    """Create a managed validation directory without deleting inputs."""

    raw = Path(os.path.abspath(output_dir.expanduser()))
    project_raw = Path(os.path.abspath(project_root.expanduser()))
    if project_raw not in raw.parents:
        raise ValueError(
            f"output path must be a strict descendant of the project root: "
            f"{project_raw}"
        )
    relative = raw.relative_to(project_raw)
    candidates: list[Path] = []
    candidate = project_raw
    for part in relative.parts:
        candidate /= part
        candidates.append(candidate)
    if any(candidate.is_symlink() for candidate in candidates):
        raise ValueError(f"output path may not traverse a symbolic link: {raw}")
    resolved = raw.resolve()
    protected = tuple(item.resolve() for item in protected_paths)
    project = project_root.resolve()
    home = Path.home().resolve()
    dangerous = {Path("/").resolve(), home, project}
    if project not in resolved.parents:
        raise ValueError(f"output path must resolve inside the project root: {project}")
    if resolved in dangerous or resolved in home.parents or resolved in project.parents:
        raise ValueError(f"refusing unsafe output directory: {resolved}")
    if any(
        resolved == item
        or resolved in item.parents
        or (item.is_dir() and item in resolved.parents)
        for item in protected
    ):
        raise ValueError(
            "output directory may not contain, or be inside, a protected input"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output path exists and is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is nonempty: {resolved}; pass --overwrite "
                "to replace it"
            )
        marker = resolved / OUTPUT_MARKER
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding="utf-8") != OUTPUT_MARKER_CONTENT
        ):
            raise ValueError(
                "refusing to delete a nonempty directory: invalid replay-validator "
                f"marker; {resolved}"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / OUTPUT_MARKER).write_text(OUTPUT_MARKER_CONTENT, encoding="utf-8")
    return resolved


def run_executable(run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as log:
        completed = subprocess.run(
            [str(run_dir / "zeqfc1")],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"ZC executable failed; inspect {run_dir / 'run.log'}")
    return {"elapsed_seconds": time.monotonic() - started}


def prepare_case(
    generator,
    *,
    source_dir: Path,
    executable: Path,
    run_dir: Path,
    restart: Path,
    start: float,
    end: float,
) -> None:
    generator.prepare_run(
        source_dir,
        executable,
        run_dir,
        nstart=3,
        tfind=start,
        tzero=start,
        tend=end,
        ntape=0,
        nrewnd=11,
        nic=0,
        write_start=end + 1.0,
        write_end=end + 1.0,
        restart=restart,
    )


def run_segmented_kernel(
    generator,
    *,
    source_dir: Path,
    executable: Path,
    run_root: Path,
    restart: Path,
    pre_nt: int,
    steps: int,
) -> dict[str, Any]:
    """Replay a window in fresh processes, restarting after every step.

    This is a reentrancy gate for the explicit-state contract.  A continuous
    in-process window can accidentally retain an undeclared local ``SAVE``
    variable; forcing a complete restart at every boundary exposes that bug.
    """

    current_restart = restart
    current_state: Path | None = None
    tapes: list[np.ndarray] = []
    elapsed = 0.0
    for offset in range(steps):
        segment = run_root / f"step-{offset + 1:03d}"
        start = generator.model_time(pre_nt + offset)
        end = generator.model_time(pre_nt + offset + 1)
        prepare_case(
            generator,
            source_dir=source_dir,
            executable=executable,
            run_dir=segment,
            restart=current_restart,
            start=start,
            end=end,
        )
        if current_state is not None:
            shutil.copy2(current_state, segment / "kernel_input_state.bin")
        elapsed += run_executable(segment)["elapsed_seconds"]
        tape = np.fromfile(segment / "kernel_branch_tape.bin", dtype=np.int32)
        if tape.size != 3:
            raise RuntimeError(f"Invalid one-step branch tape: {segment}")
        tapes.append(tape)
        current_restart = segment / "outhst"
        current_state = segment / "kernel_final_state.bin"
    return {
        "final_restart": current_restart,
        "branch_tape": np.concatenate(tapes),
        "elapsed_seconds": elapsed,
        "process_count": steps,
        "full_explicit_state_restored": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 10, 30, 31, 40],
        help="Window lengths measured in the model's ten-day coupled steps.",
    )
    parser.add_argument(
        "--skip-segmented-replay",
        action="store_true",
        help=(
            "Skip the slow reentrancy gate that restarts a fresh process at "
            "every boundary of the longest requested window."
        ),
    )
    parser.add_argument(
        "--extra-checkpoint",
        nargs=3,
        action="append",
        metavar=("LABEL", "PATH", "PRE_NT"),
        help=(
            "Also validate a complete native checkpoint. PRE_NT is the global "
            "model step stored in that checkpoint; the option is repeatable."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.steps):
        parser.error("all --steps values must be positive")

    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    validator_path = Path(__file__).resolve()
    validator_sha256 = sha256_file(validator_path)
    generator = load_generator(project_root)
    build_dir = args.build_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_requested = args.output_dir

    reference_executable = build_dir / "zc_reference"
    kernel_executable = build_dir / "zc_kernel_replay"
    source_dir = build_dir / "source"
    generator_path = project_root / "scripts/generate_fresh_zc_dataset.py"
    build_provenance = validate_build_provenance(
        build_dir,
        generator=generator,
        builder_path=here / "build_kernel.py",
        generator_path=generator_path,
    )

    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = _json_object(metadata_path, label="processed-data metadata")
    metadata_sha256 = sha256_file(metadata_path)
    event_specs = validate_processed_metadata(
        metadata,
        reference_executable_sha256=build_provenance["reference_executable_sha256"],
    )
    checkpoint_cases: list[dict[str, Any]] = []
    for event in event_specs:
        checkpoint_path = (data_dir / event["checkpoint_file"]).resolve()
        if data_dir not in checkpoint_path.parents:
            raise ValueError("event checkpoint path escapes the processed-data root")
        checkpoint_cases.append(
            {
                "label": event["label"],
                "path": checkpoint_path,
                "expected_sha256": event["expected_sha256"],
                "pre_nt": event["pre_nt"],
            }
        )
    for extra in args.extra_checkpoint or []:
        label, path_text, pre_nt_text = extra
        path = Path(path_text).resolve()
        checkpoint_cases.append(
            {
                "label": label,
                "path": path,
                "expected_sha256": sha256_file(path),
                "pre_nt": int(pre_nt_text),
            }
        )
    validate_locked_checkpoint_cases(checkpoint_cases)

    # Check every checkpoint before preparing the output.  This ordering
    # prevents malformed metadata, missing files, or checksum failures from
    # deleting an earlier validation result.
    for checkpoint in checkpoint_cases:
        restart = checkpoint["path"]
        if not restart.is_file():
            raise FileNotFoundError(restart)
        if sha256_file(restart) != checkpoint["expected_sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {restart}")

    original_checkpoint_cases = [dict(checkpoint) for checkpoint in checkpoint_cases]

    output_transaction = transactional_output(
        output_requested,
        overwrite=args.overwrite,
        protected_paths=(
            build_dir,
            data_dir,
            *(checkpoint["path"] for checkpoint in checkpoint_cases),
        ),
        project_root=project_root,
        owner="validate_replay",
        marker_name=OUTPUT_MARKER,
        marker_value=OUTPUT_MARKER_CONTENT,
    )
    with output_transaction as output_dir:
        (
            source_dir,
            reference_executable,
            kernel_executable,
            staged_metadata_path,
            checkpoint_cases,
        ) = stage_validation_inputs(
            output_dir=output_dir,
            build_dir=build_dir,
            metadata_path=metadata_path,
            checkpoint_cases=original_checkpoint_cases,
            build_provenance=build_provenance,
            generator=generator,
        )
        if (
            _json_object(
                staged_metadata_path,
                label="staged processed-data metadata",
            )
            != metadata
        ):
            raise RuntimeError(
                "staged processed metadata differs from verified metadata"
            )
        private_input_binding = verify_staged_validation_inputs(
            source_dir=source_dir,
            reference_executable=reference_executable,
            kernel_executable=kernel_executable,
            metadata_path=staged_metadata_path,
            checkpoint_cases=checkpoint_cases,
            build_provenance=build_provenance,
            metadata_sha256=metadata_sha256,
            generator=generator,
        )

        cases: list[dict[str, Any]] = []
        segmented_cases: list[dict[str, Any]] = []
        workspace_poison_cases: list[dict[str, Any]] = []
        all_passed = True
        longest_window = max(args.steps)
        for checkpoint in checkpoint_cases:
            restart = checkpoint["path"]
            pre_nt = int(checkpoint["pre_nt"])
            start = generator.model_time(pre_nt)
            for steps in args.steps:
                end = generator.model_time(pre_nt + steps)
                label = f"{checkpoint['label']}_{steps:03d}step"
                reference_dir = output_dir / label / "reference"
                kernel_dir = output_dir / label / "kernel"
                prepare_case(
                    generator,
                    source_dir=source_dir,
                    executable=reference_executable,
                    run_dir=reference_dir,
                    restart=restart,
                    start=start,
                    end=end,
                )
                prepare_case(
                    generator,
                    source_dir=source_dir,
                    executable=kernel_executable,
                    run_dir=kernel_dir,
                    restart=restart,
                    start=start,
                    end=end,
                )
                reference_runtime = run_executable(reference_dir)
                kernel_runtime = run_executable(kernel_dir)

                initial = kernel_dir / "kernel_initial_state.bin"
                roundtrip = kernel_dir / "kernel_roundtrip_state.bin"
                final_state = kernel_dir / "kernel_final_state.bin"
                tape_path = kernel_dir / "kernel_branch_tape.bin"
                sizes_valid = all(
                    path.stat().st_size == STATE_FILE_SIZE
                    for path in (initial, roundtrip, final_state)
                )
                roundtrip_equal = initial.read_bytes() == roundtrip.read_bytes()
                restart_equal = (reference_dir / "outhst").read_bytes() == (
                    kernel_dir / "outhst"
                ).read_bytes()
                tape = np.fromfile(tape_path, dtype=np.int32)
                tape_shape_valid = tape.size == 3 * steps
                if tape_shape_valid:
                    tape = tape.reshape(steps, 3)
                    tape_valid = bool(
                        np.all(tape[:, 0] >= 2)
                        and np.all(tape[:, 1] >= 1)
                        and np.all(np.isin(tape[:, 2], (0, 1)))
                    )
                    tape_summary: dict[str, Any] = {
                        "sst_substeps_min": int(tape[:, 0].min()),
                        "sst_substeps_max": int(tape[:, 0].max()),
                        "atmosphere_iterations_min": int(tape[:, 1].min()),
                        "atmosphere_iterations_max": int(tape[:, 1].max()),
                        "atmosphere_reset_steps_zero_based": np.flatnonzero(
                            tape[:, 2]
                        ).tolist(),
                    }
                else:
                    tape_valid = False
                    tape_summary = {}
                passed = bool(
                    sizes_valid
                    and roundtrip_equal
                    and restart_equal
                    and tape_shape_valid
                    and tape_valid
                )
                all_passed = all_passed and passed
                cases.append(
                    {
                        "label": label,
                        "checkpoint_label": checkpoint["label"],
                        "steps": steps,
                        "start_time_months": start,
                        "end_time_months": end,
                        "source_checkpoint_sha256": sha256_file(restart),
                        "pack_unpack_bitwise_equal": roundtrip_equal,
                        "kernel_restart_matches_reference_bitwise": restart_equal,
                        "reference_restart_sha256": sha256_file(
                            reference_dir / "outhst"
                        ),
                        "kernel_restart_sha256": sha256_file(kernel_dir / "outhst"),
                        "state_file_size_bytes": initial.stat().st_size,
                        "state_file_size_valid": sizes_valid,
                        "branch_tape_size_valid": tape_shape_valid,
                        "branch_tape_values_valid": tape_valid,
                        "branch_tape_summary": tape_summary,
                        "reference_runtime": reference_runtime,
                        "kernel_runtime": kernel_runtime,
                        "passed": passed,
                    }
                )

            # Prove that arrays excluded as overwritten timestep workspace really
            # cannot affect the one-step map.  Each run is a new process, which
            # also exercises the required CFORCE ISTART=0 static initialization.
            closure_root = output_dir / f"{checkpoint['label']}_workspace_closure"
            clean_dir = closure_root / "clean"
            closure_end = generator.model_time(pre_nt + 1)
            prepare_case(
                generator,
                source_dir=source_dir,
                executable=kernel_executable,
                run_dir=clean_dir,
                restart=restart,
                start=start,
                end=closure_end,
            )
            run_executable(clean_dir)
            for poison_mode in (1, 2):
                poison_dir = closure_root / f"poison-{poison_mode}"
                prepare_case(
                    generator,
                    source_dir=source_dir,
                    executable=kernel_executable,
                    run_dir=poison_dir,
                    restart=restart,
                    start=start,
                    end=closure_end,
                )
                (poison_dir / "kernel_workspace_poison.txt").write_text(
                    f"{poison_mode}\n"
                )
                run_executable(poison_dir)
                initial_equal = (
                    poison_dir / "kernel_initial_state.bin"
                ).read_bytes() == (clean_dir / "kernel_initial_state.bin").read_bytes()
                roundtrip_equal = (
                    poison_dir / "kernel_roundtrip_state.bin"
                ).read_bytes() == (
                    clean_dir / "kernel_roundtrip_state.bin"
                ).read_bytes()
                state_equal = (poison_dir / "kernel_final_state.bin").read_bytes() == (
                    clean_dir / "kernel_final_state.bin"
                ).read_bytes()
                restart_equal = (poison_dir / "outhst").read_bytes() == (
                    clean_dir / "outhst"
                ).read_bytes()
                tape_equal = (poison_dir / "kernel_branch_tape.bin").read_bytes() == (
                    clean_dir / "kernel_branch_tape.bin"
                ).read_bytes()
                poison_logged = (
                    f"ZC omitted-workspace poison mode            {poison_mode}"
                    in (poison_dir / "run.log").read_text()
                )
                passed = bool(
                    initial_equal
                    and roundtrip_equal
                    and state_equal
                    and restart_equal
                    and tape_equal
                    and poison_logged
                )
                all_passed = all_passed and passed
                workspace_poison_cases.append(
                    {
                        "checkpoint_label": checkpoint["label"],
                        "steps": 1,
                        "poison_mode": poison_mode,
                        "poison_applied_and_logged": poison_logged,
                        "fresh_process_static_context_initialized": True,
                        "initial_explicit_state_matches_clean_bitwise": initial_equal,
                        "pack_roundtrip_matches_clean_bitwise": roundtrip_equal,
                        "final_explicit_state_matches_clean_bitwise": state_equal,
                        "final_restart_matches_clean_bitwise": restart_equal,
                        "branch_tape_matches_clean_bitwise": tape_equal,
                        "clean_final_state_sha256": sha256_file(
                            clean_dir / "kernel_final_state.bin"
                        ),
                        "poison_final_state_sha256": sha256_file(
                            poison_dir / "kernel_final_state.bin"
                        ),
                        "passed": passed,
                    }
                )

            if not args.skip_segmented_replay:
                label = f"{checkpoint['label']}_{longest_window:03d}step"
                continuous_dir = output_dir / label / "kernel"
                segmented = run_segmented_kernel(
                    generator,
                    source_dir=source_dir,
                    executable=kernel_executable,
                    run_root=output_dir / label / "segmented",
                    restart=restart,
                    pre_nt=pre_nt,
                    steps=longest_window,
                )
                continuous_tape = np.fromfile(
                    continuous_dir / "kernel_branch_tape.bin", dtype=np.int32
                )
                restart_equal = (
                    segmented["final_restart"].read_bytes()
                    == (continuous_dir / "outhst").read_bytes()
                )
                tape_equal = np.array_equal(segmented["branch_tape"], continuous_tape)
                passed = bool(restart_equal and tape_equal)
                all_passed = all_passed and passed
                segmented_cases.append(
                    {
                        "checkpoint_label": checkpoint["label"],
                        "steps": longest_window,
                        "fresh_process_per_step": True,
                        "full_explicit_state_restored": segmented[
                            "full_explicit_state_restored"
                        ],
                        "process_count": segmented["process_count"],
                        "segmented_runtime_seconds": segmented["elapsed_seconds"],
                        "final_restart_matches_continuous_window_bitwise": (
                            restart_equal
                        ),
                        "branch_tape_matches_continuous_window_bitwise": tape_equal,
                        "continuous_restart_sha256": sha256_file(
                            continuous_dir / "outhst"
                        ),
                        "segmented_restart_sha256": sha256_file(
                            segmented["final_restart"]
                        ),
                        "passed": passed,
                    }
                )

        # Reject publication if any live provenance input changed while the long
        # validation was running.  All scientific executions used the private
        # staged copies above; this second gate protects the report's producer
        # chain from concurrent replacement.
        final_build_provenance = validate_build_provenance(
            build_dir,
            generator=generator,
            builder_path=here / "build_kernel.py",
            generator_path=generator_path,
        )
        if final_build_provenance != build_provenance:
            raise RuntimeError("kernel build provenance changed during validation")
        if sha256_file(metadata_path) != metadata_sha256:
            raise RuntimeError("processed metadata changed during validation")
        if sha256_file(validator_path) != validator_sha256:
            raise RuntimeError("replay validator changed during validation")
        for checkpoint in original_checkpoint_cases:
            if sha256_file(checkpoint["path"]) != checkpoint["expected_sha256"]:
                raise RuntimeError(
                    "source checkpoint changed during validation: "
                    f"{checkpoint['label']}"
                )
        if (
            verify_staged_validation_inputs(
                source_dir=source_dir,
                reference_executable=reference_executable,
                kernel_executable=kernel_executable,
                metadata_path=staged_metadata_path,
                checkpoint_cases=checkpoint_cases,
                build_provenance=build_provenance,
                metadata_sha256=metadata_sha256,
                generator=generator,
            )
            != private_input_binding
        ):
            raise RuntimeError("private staged-input binding changed")

        report = {
            "schema_version": 2,
            "validator_version": VALIDATOR_VERSION,
            "validator_script_sha256": validator_sha256,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "passed" if all_passed else "failed",
            "build_provenance": build_provenance,
            "build_report_sha256": build_provenance["build_report_sha256"],
            "reference_executable_sha256": build_provenance[
                "reference_executable_sha256"
            ],
            "kernel_executable_sha256": build_provenance["kernel_executable_sha256"],
            "executed_staged_executable_sha256": private_input_binding[
                "executable_sha256"
            ],
            "private_staged_input_binding": private_input_binding,
            "processed_metadata": {
                "filename": metadata_path.name,
                "sha256": metadata_sha256,
                "staged_sha256": sha256_file(staged_metadata_path),
                "schema_version": metadata["schema_version"],
                "generator_script_version": metadata["script_version"],
            },
            "locked_checkpoint_contract": {
                checkpoint["label"]: {
                    "pre_nt": checkpoint["pre_nt"],
                    "sha256": checkpoint["expected_sha256"],
                }
                for checkpoint in original_checkpoint_cases
            },
            "state_file_size_bytes": STATE_FILE_SIZE,
            "cases": cases,
            "segmented_replay_cases": segmented_cases,
            "workspace_poison_cases": workspace_poison_cases,
        }
        (output_dir / "replay_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not all_passed:
            raise RuntimeError("One or more kernel replay gates failed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
