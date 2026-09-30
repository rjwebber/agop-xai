#!/usr/bin/env python3
"""Validate the full coupled Tapenade ZC tangent against centered differences."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from output_safety import transactional_output

NREAL = 59_148
NCOMPLEX = 5_280
NDOUBLE = 1
NINTEGER = 4
NTIME = 2
STATE_SIZE = NREAL * 4 + NCOMPLEX * 8 + NDOUBLE * 8 + NINTEGER * 4 + NTIME * 4
NINO3_OFFSET = 32_121
NINO3_I = range(13, 19)
NINO3_J = range(21, 32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def verify_sha256_manifest(
    root: Path,
    manifest_path: Path,
    *,
    exact_inventory: bool,
) -> int:
    root = root.resolve()
    declared: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative_path = Path(relative.strip())
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest path: {relative!r}")
        name = relative_path.as_posix()
        if name in declared:
            raise ValueError(f"duplicate manifest path: {name}")
        declared.add(name)
        target = root / relative_path
        if target.is_symlink() or not target.is_file() or sha256(target) != digest:
            raise RuntimeError(f"manifest mismatch: {target}")
    if not declared:
        raise ValueError(f"empty manifest: {manifest_path}")
    if exact_inventory:
        actual: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"snapshot contains a symbolic link: {path}")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise ValueError(f"snapshot contains a special file: {path}")
        if actual != declared:
            raise RuntimeError("exact coupled-build snapshot inventory mismatch")
    return len(declared)


def verify_audit_build(
    *,
    tangent: Path,
    primal: Path,
    source: Path,
    generation_provenance: Path,
    tangent_build_manifest: Path,
    tangent_build_inputs: Path,
    coupled_build_manifest: Path,
    coupled_build_input_manifest: Path,
    coupled_build_snapshot: Path,
    kernel_build_report: Path,
    generator_path: Path,
) -> dict[str, object]:
    """Bind the O3 tangent and forward oracle to one immutable snapshot."""

    expected_paths = {
        "source": coupled_build_snapshot / "kernel_source",
        "generation": coupled_build_snapshot / "generated/generation_provenance.txt",
        "kernel_report": coupled_build_snapshot / "kernel_parent/build_report.json",
    }
    if source != expected_paths["source"]:
        raise RuntimeError("kernel source is not the coupled-build snapshot")
    if generation_provenance != expected_paths["generation"]:
        raise RuntimeError("generation provenance is not the coupled-build snapshot")
    if kernel_build_report != expected_paths["kernel_report"]:
        raise RuntimeError("kernel build report is not the coupled-build snapshot")

    tangent_manifest = parse_key_values(tangent_build_manifest)
    coupled_manifest = parse_key_values(coupled_build_manifest)
    if tangent_manifest.get("schema") != "zc-tangent-audit-build-v1":
        raise ValueError("unsupported tangent build-manifest schema")
    if tangent_manifest.get("optimization") != "O3":
        raise RuntimeError("release evidence requires an O3 audit tangent")
    expected_bindings = {
        "one_step_executable_sha256": sha256(tangent),
        "tangent_build_inputs_sha256": sha256(tangent_build_inputs),
        "coupled_build_manifest_sha256": sha256(coupled_build_manifest),
        "coupled_build_input_manifest_sha256": sha256(coupled_build_input_manifest),
    }
    for key, expected in expected_bindings.items():
        if tangent_manifest.get(key) != expected:
            raise RuntimeError(f"tangent build-manifest mismatch for {key}")
    if coupled_manifest.get("build_input_manifest_sha256") != sha256(
        coupled_build_input_manifest
    ):
        raise RuntimeError("coupled build does not bind its input snapshot")
    if coupled_manifest.get("kernel_parent_build_report_sha256") != sha256(
        kernel_build_report
    ):
        raise RuntimeError("coupled build does not bind the kernel build report")
    kernel_report = json.loads(kernel_build_report.read_text(encoding="utf-8"))
    if kernel_report.get("kernel_executable_sha256") != sha256(primal):
        raise RuntimeError("forward executable does not match kernel build report")
    generator_hashes = kernel_report.get("builder_source_sha256", {})
    if generator_hashes.get("generate_fresh_zc_dataset.py") != sha256(generator_path):
        raise RuntimeError("dataset generator does not match kernel build report")
    snapshot_count = verify_sha256_manifest(
        coupled_build_snapshot,
        coupled_build_input_manifest,
        exact_inventory=True,
    )
    tangent_input_count = verify_sha256_manifest(
        coupled_build_snapshot,
        tangent_build_inputs,
        exact_inventory=False,
    )
    return {
        "tangent_build_manifest_sha256": sha256(tangent_build_manifest),
        "tangent_build_inputs_sha256": sha256(tangent_build_inputs),
        "coupled_build_manifest_sha256": sha256(coupled_build_manifest),
        "coupled_build_input_manifest_sha256": sha256(coupled_build_input_manifest),
        "kernel_build_report_sha256": sha256(kernel_build_report),
        "snapshot_entry_count": snapshot_count,
        "tangent_input_count": tangent_input_count,
        "optimization": tangent_manifest["optimization"],
        "dataset_generator_sha256": sha256(generator_path),
        "primal_executable_sha256": sha256(primal),
        "one_step_executable_sha256": sha256(tangent),
    }


def tree_file_hashes(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"input tree contains a symbolic link: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256(path)
        elif not path.is_dir():
            raise ValueError(f"input tree contains a special file: {path}")
    return files


def file_mapping_sha256(files: dict[str, str]) -> str:
    """Hash one sorted relative-path/SHA mapping without machine-local paths."""

    digest = hashlib.sha256()
    for name, file_hash in sorted(files.items()):
        digest.update(f"{file_hash}  {name}\n".encode())
    return digest.hexdigest()


def verify_private_staged_inputs(
    *,
    trees: dict[str, tuple[Path, dict[str, str]]],
    files: dict[str, tuple[Path, str]],
) -> dict[str, object]:
    """Reauthenticate every private input used by a long scientific run."""

    tree_bindings: dict[str, dict[str, object]] = {}
    for label, (path, expected) in trees.items():
        observed = tree_file_hashes(path)
        if observed != expected:
            raise RuntimeError(f"staged {label} changed during validation")
        tree_bindings[label] = {
            "manifest_sha256": file_mapping_sha256(observed),
            "file_count": len(observed),
        }

    file_bindings: dict[str, str] = {}
    for label, (path, expected) in files.items():
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"staged {label} changed during validation")
        file_bindings[label] = expected
    return {"trees": tree_bindings, "files": file_bindings}


def verify_replay_report_binding(
    replay_report_path: Path,
    *,
    kernel_build_report: Path,
    primal: Path,
) -> dict[str, str]:
    """Bind a passed replay gate to the exact forward build used by the audit."""

    report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 2 or report.get("status") != "passed":
        raise RuntimeError("replay-validation report is not a passed schema-2 report")
    build_hash = sha256(kernel_build_report)
    primal_hash = sha256(primal)
    if report.get("build_report_sha256") != build_hash:
        raise RuntimeError("replay-validation report names a different kernel build")
    if report.get("kernel_executable_sha256") != primal_hash:
        raise RuntimeError("replay-validation report names a different primal")
    executed = report.get("executed_staged_executable_sha256")
    if (
        not isinstance(executed, dict)
        or executed.get("zc_kernel_replay") != primal_hash
    ):
        raise RuntimeError(
            "replay-validation report does not bind the executed staged primal"
        )
    provenance = report.get("build_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("replay-validation report lacks build provenance")
    if (
        provenance.get("build_report_sha256") != build_hash
        or provenance.get("kernel_executable_sha256") != primal_hash
    ):
        raise RuntimeError("replay build-provenance cross-binding is inconsistent")
    locked = report.get("locked_checkpoint_contract")
    if not isinstance(locked, dict) or not locked:
        raise RuntimeError("replay-validation report has no checkpoint contract")
    passed_case_labels = {
        case.get("checkpoint_label")
        for case in report.get("cases", [])
        if isinstance(case, dict) and case.get("passed") is True
    }
    if not set(locked).issubset(passed_case_labels):
        raise RuntimeError(
            "replay-validation report lacks a passed case for a locked checkpoint"
        )
    return {
        "replay_report_sha256": sha256(replay_report_path),
        "kernel_build_report_sha256": build_hash,
        "primal_executable_sha256": primal_hash,
    }


def verify_replay_runtime_source(
    runtime_source: Path,
    replay_report_path: Path,
    *,
    kernel_build_report: Path,
    generator,
) -> dict[str, object]:
    """Bind the runnable source tree to the replay validator that staged it."""

    expected_path = replay_report_path.resolve().parent / "verified_inputs/source"
    if runtime_source.resolve() != expected_path:
        raise RuntimeError(
            "runtime source must be replay-validation verified_inputs/source"
        )
    report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    provenance = report.get("build_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("replay-validation report lacks build provenance")
    build_report_hash = sha256(kernel_build_report)
    if provenance.get("build_report_sha256") != build_report_hash:
        raise RuntimeError("replay runtime source names a different kernel build")
    kernel_report = json.loads(kernel_build_report.read_text(encoding="utf-8"))
    source_manifest_path = kernel_build_report.parent / "source_manifest.json"
    if (
        not source_manifest_path.is_file()
        or sha256(source_manifest_path)
        != kernel_report.get("source_manifest_file_sha256")
    ):
        raise RuntimeError("kernel source manifest does not match build report")
    original_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(original_manifest, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in original_manifest.items()
    ):
        raise ValueError("kernel source manifest must map paths to SHA-256")
    if generator.manifest_sha256(original_manifest) != kernel_report.get(
        "source_manifest_sha256"
    ):
        raise RuntimeError("kernel source-manifest digest does not match build report")
    expected_build_source = dict(original_manifest)
    for field in (
        "fresh_patch_sha256",
        "kernel_patch_sha256",
        "kernel_support_source_sha256",
    ):
        updates = kernel_report.get(field)
        if not isinstance(updates, dict) or not all(
            isinstance(name, str) and isinstance(digest, str)
            for name, digest in updates.items()
        ):
            raise ValueError(f"kernel build report has invalid {field}")
        expected_build_source.update(updates)
    expected_files = provenance.get("staged_source_file_sha256")
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError("replay report does not bind its staged runtime source")
    if expected_files != expected_build_source:
        raise RuntimeError(
            "replay runtime source manifest is not the exact kernel-build source"
        )
    actual_files = generator.source_manifest(runtime_source)
    if actual_files != expected_files:
        raise RuntimeError("replay runtime source differs from its build provenance")
    expected_digest = provenance.get("staged_source_manifest_sha256")
    actual_digest = generator.manifest_sha256(actual_files)
    if actual_digest != expected_digest:
        raise RuntimeError("replay runtime source manifest digest is inconsistent")
    complete_tree = tree_file_hashes(runtime_source)
    if complete_tree != expected_files:
        raise RuntimeError(
            "replay runtime source has an unauthenticated complete-tree inventory"
        )
    return {
        "replay_report_sha256": sha256(replay_report_path),
        "kernel_build_report_sha256": build_report_hash,
        "kernel_source_manifest_sha256": sha256(source_manifest_path),
        "runtime_source_manifest_sha256": actual_digest,
        "runtime_source_manifest_entry_count": len(actual_files),
        "runtime_source_complete_tree_sha256": file_mapping_sha256(complete_tree),
        "runtime_source_complete_tree_entry_count": len(complete_tree),
    }


def run(run_dir: Path) -> float:
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as log:
        completed = subprocess.run(
            [str(run_dir / "zeqfc1")],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"run failed; inspect {run_dir / 'run.log'}")
    return time.monotonic() - started


def state_real(path: Path) -> np.ndarray:
    if path.stat().st_size != STATE_SIZE:
        raise RuntimeError(f"unexpected explicit-state size: {path}")
    return np.fromfile(path, dtype=np.float32, count=NREAL).copy()


def replace_state_real(source: Path, destination: Path, values: np.ndarray) -> None:
    payload = bytearray(source.read_bytes())
    if len(payload) != STATE_SIZE or values.shape != (NREAL,):
        raise ValueError("invalid packed-state payload")
    payload[: NREAL * 4] = np.asarray(values, dtype=np.float32).tobytes()
    destination.write_bytes(payload)


def nino3_canonical(values: np.ndarray) -> np.float32:
    """Use the zc-v3 float64 reduction, then publish one float32 value."""
    total = 0.0
    for j in NINO3_J:
        for i in NINO3_I:
            # Fortran K=offset+I+30*(J-1), converted to zero-based Python.
            k = NINO3_OFFSET + i + 30 * (j - 1) - 1
            total += float(values[k])
    return np.float32(total / 66.0)


def prepare(
    generator,
    *,
    source_dir: Path,
    executable: Path,
    run_dir: Path,
    restart: Path,
    pre_nt: int,
    steps: int,
) -> None:
    generator.prepare_run(
        source_dir,
        executable,
        run_dir,
        nstart=3,
        tfind=generator.model_time(pre_nt),
        tzero=generator.model_time(pre_nt),
        tend=generator.model_time(pre_nt + steps),
        ntape=0,
        nrewnd=11,
        nic=0,
        write_start=generator.model_time(pre_nt + steps + 2),
        write_end=generator.model_time(pre_nt + steps + 2),
        restart=restart,
    )


def active_masks(manifest: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    active = np.zeros(NREAL, dtype=bool)
    independent = np.zeros(NREAL, dtype=bool)
    segments = manifest["arrays"]["real32"]["segments"]
    for segment in segments:
        slc = slice(segment["start"], segment["stop"])
        if segment["activity"] == "active":
            active[slc] = True
        if segment["independent_control"]:
            independent[slc] = True
    return active, independent, segments


def scaled_direction(
    x: np.ndarray, segments: list[dict], *, seed: int, independent_only: bool
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = np.zeros(NREAL, dtype=np.float32)
    for segment in segments:
        use = (
            segment["independent_control"]
            if independent_only
            else segment["activity"] == "active"
        )
        if not use:
            continue
        slc = slice(segment["start"], segment["stop"])
        sample = rng.standard_normal(segment["stop"] - segment["start"])
        sample_rms = float(np.sqrt(np.mean(sample * sample)))
        x64 = x[slc].astype(np.float64)
        scale = max(float(np.sqrt(np.mean(x64 * x64))), 1.0e-3)
        direction[slc] = np.asarray(sample * (scale / sample_rms), dtype=np.float32)
    return direction


def relative_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    denominator = max(float(np.linalg.norm(b)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(a - b) / denominator)


def cosine(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 31])
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
        "--generation-provenance",
        type=Path,
        required=True,
        help="Pinned Tapenade generation-provenance file.",
    )
    parser.add_argument(
        "--kernel-source",
        type=Path,
        required=True,
        help=(
            "Immutable coupled-build kernel_source snapshot used to bind the "
            "compiled forward oracle; this compact tree is not used as the "
            "runnable ZC source tree."
        ),
    )
    parser.add_argument(
        "--runtime-source",
        type=Path,
        help=(
            "Full runnable source tree. With --replay-validation-dir this "
            "defaults to, and if supplied must equal, "
            "verified_inputs/source from that replay validation. It is required "
            "for explicit --checkpoint inputs."
        ),
    )
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
        help=(
            "Explicit validation checkpoint; repeat this option for each case. "
            "ADVANCE is the number of model steps before the tested state."
        ),
    )
    checkpoint_group.add_argument(
        "--replay-validation-dir",
        type=Path,
        help=(
            "Passed replay-validation directory containing replay_report.json "
            "and verified_inputs/checkpoints/*.hst; uses ADVANCE=0."
        ),
    )
    parser.add_argument(
        "--case-labels",
        nargs="+",
        help="Optionally run only the named checkpoint cases.",
    )
    parser.add_argument(
        "--direction-scope",
        choices=("independent", "active"),
        default="independent",
        help="Perturb only independent controls or every active real-state segment.",
    )
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[0.02, 0.01, 0.005, 0.0025, 0.00125],
    )
    args = parser.parse_args()
    validator_path = Path(__file__).resolve()
    validator_sha256 = sha256(validator_path)

    root = args.project_root.resolve()
    output_requested = args.output_dir

    sys.path.insert(0, str(root / "scripts"))
    import generate_fresh_zc_dataset as generator  # noqa: PLC0415

    generator_path = root / "scripts/generate_fresh_zc_dataset.py"

    kernel_source = args.kernel_source.resolve()
    primal = args.primal_executable.resolve()
    tangent = args.tangent_executable.resolve()
    tangent_build_manifest = args.tangent_build_manifest.resolve()
    tangent_build_inputs = args.tangent_build_inputs.resolve()
    coupled_build_manifest = args.coupled_build_manifest.resolve()
    coupled_build_input_manifest = args.coupled_build_input_manifest.resolve()
    coupled_build_snapshot = args.coupled_build_snapshot.resolve()
    kernel_build_report = args.kernel_build_report.resolve()
    manifest_path = root / "adjoint/fortran_kernel/state_manifest.json"
    state_manifest_sha256 = sha256(manifest_path)

    replay_report_path: Path | None = None
    replay_report_sha256: str | None = None
    cases: list[dict] = []
    if args.replay_validation_dir is not None:
        replay_dir = args.replay_validation_dir.resolve()
        replay_report_path = replay_dir / "replay_report.json"
        replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
        replay_report_sha256 = sha256(replay_report_path)
        if replay_report.get("status") != "passed":
            raise RuntimeError("replay-validation report is not passed")
        locked = replay_report.get("locked_checkpoint_contract", {})
        if not isinstance(locked, dict) or not locked:
            raise RuntimeError("replay-validation report has no checkpoint contract")
        for label, specification in sorted(locked.items()):
            cases.append(
                {
                    "label": label,
                    "restart": replay_dir
                    / "verified_inputs/checkpoints"
                    / f"{label}.hst",
                    "pre_nt": int(specification["pre_nt"]),
                    "advance": 0,
                    "expected_sha256": specification["sha256"],
                }
            )
    else:
        for label, restart, pre_nt, advance in args.checkpoint or []:
            restart_path = Path(restart).resolve()
            cases.append(
                {
                    "label": label,
                    "restart": restart_path,
                    "pre_nt": int(pre_nt),
                    "advance": int(advance),
                    "expected_sha256": sha256(restart_path),
                }
            )
    if args.runtime_source is not None:
        runtime_source = args.runtime_source.resolve()
    elif replay_report_path is not None:
        runtime_source = replay_report_path.parent / "verified_inputs/source"
    else:
        parser.error("--runtime-source is required with explicit --checkpoint")
    if args.case_labels:
        requested = set(args.case_labels)
        available = {case["label"] for case in cases}
        unknown = requested - available
        if unknown:
            parser.error(f"unknown case labels: {sorted(unknown)}")
        cases = [case for case in cases if case["label"] in requested]

    generation_provenance = args.generation_provenance.resolve()
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
        if sha256(restart) != case["expected_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {case['label']}")

    build_provenance = verify_audit_build(
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
        verify_replay_report_binding(
            replay_report_path,
            kernel_build_report=kernel_build_report,
            primal=primal,
        )
        if replay_report_path is not None
        else None
    )
    runtime_source_binding = (
        verify_replay_runtime_source(
            runtime_source,
            replay_report_path,
            kernel_build_report=kernel_build_report,
            generator=generator,
        )
        if replay_report_path is not None
        else {
            "authentication": "explicit_runtime_source_not_replay_authenticated",
            "runtime_source_complete_tree_sha256": file_mapping_sha256(
                tree_file_hashes(runtime_source)
            ),
            "runtime_source_complete_tree_entry_count": len(
                tree_file_hashes(runtime_source)
            ),
        }
    )
    kernel_source_hashes = tree_file_hashes(kernel_source)
    runtime_source_hashes = tree_file_hashes(runtime_source)
    checkpoint_hashes = {case["label"]: sha256(Path(case["restart"])) for case in cases}
    bound_kernel_source = kernel_source
    bound_runtime_source = runtime_source
    bound_primal = primal
    bound_tangent = tangent
    bound_cases = [dict(case) for case in cases]

    output_transaction = transactional_output(
        output_requested,
        overwrite=args.overwrite,
        protected_paths=(
            kernel_source,
            runtime_source,
            primal,
            tangent,
            manifest_path,
            *(path for path in (replay_report_path,) if path is not None),
            generation_provenance,
            tangent_build_manifest,
            tangent_build_inputs,
            coupled_build_manifest,
            coupled_build_input_manifest,
            coupled_build_snapshot,
            kernel_build_report,
            generator_path,
            *(Path(case["restart"]) for case in cases),
        ),
        project_root=root,
        owner="validate_tangent",
    )
    with output_transaction as output:
        verified_inputs = output / "verified_inputs"
        verified_inputs.mkdir()
        staged_kernel_source = verified_inputs / "kernel_source"
        shutil.copytree(kernel_source, staged_kernel_source)
        if tree_file_hashes(staged_kernel_source) != kernel_source_hashes:
            raise RuntimeError("staged kernel source differs from verified source")
        staged_runtime_source = verified_inputs / "runtime_source"
        shutil.copytree(runtime_source, staged_runtime_source)
        if tree_file_hashes(staged_runtime_source) != runtime_source_hashes:
            raise RuntimeError("staged runtime source differs from verified source")
        staged_manifest = verified_inputs / "state_manifest.json"
        staged_manifest.write_bytes(manifest_path.read_bytes())
        if sha256(staged_manifest) != state_manifest_sha256:
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
            if sha256(staged) != expected:
                raise RuntimeError(f"executable changed while staging: {original}")
            staged.chmod(staged.stat().st_mode | 0o100)
        staged_cases: list[dict] = []
        staged_checkpoint_dir = verified_inputs / "checkpoints"
        staged_checkpoint_dir.mkdir()
        for case in cases:
            staged_restart = staged_checkpoint_dir / f"{case['label']}.hst"
            staged_restart.write_bytes(Path(case["restart"]).read_bytes())
            if sha256(staged_restart) != checkpoint_hashes[case["label"]]:
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
        private_input_binding = verify_private_staged_inputs(
            trees=private_input_trees,
            files=private_input_files,
        )
        manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
        active, independent, segments = active_masks(manifest)
        source = staged_runtime_source
        primal = staged_primal
        tangent = staged_tangent
        cases = staged_cases

        report_cases: list[dict] = []
        for case_index, case in enumerate(cases):
            case_root = output / case["label"]
            restart = Path(case["restart"])
            pre_nt = int(case["pre_nt"])
            if case["advance"]:
                advance_dir = case_root / "advance_to_validation_state"
                prepare(
                    generator,
                    source_dir=source,
                    executable=primal,
                    run_dir=advance_dir,
                    restart=restart,
                    pre_nt=pre_nt,
                    steps=int(case["advance"]),
                )
                run(advance_dir)
                restart = advance_dir / "outhst"
                state_path = advance_dir / "kernel_final_state.bin"
                pre_nt += int(case["advance"])
            else:
                capture_dir = case_root / "capture_initial_state"
                prepare(
                    generator,
                    source_dir=source,
                    executable=primal,
                    run_dir=capture_dir,
                    restart=restart,
                    pre_nt=pre_nt,
                    steps=1,
                )
                run(capture_dir)
                state_path = capture_dir / "kernel_initial_state.bin"

            x = state_real(state_path)
            direction = scaled_direction(
                x,
                segments,
                seed=4200 + case_index,
                independent_only=args.direction_scope == "independent",
            )
            output_rng = np.random.default_rng(9900 + case_index)
            output_weight = output_rng.standard_normal(NREAL)
            output_weight /= np.linalg.norm(output_weight)

            for window in args.windows:
                window_root = case_root / f"window_{window:03d}"
                baseline_dir = window_root / "baseline"
                prepare(
                    generator,
                    source_dir=source,
                    executable=primal,
                    run_dir=baseline_dir,
                    restart=restart,
                    pre_nt=pre_nt,
                    steps=window,
                )
                shutil.copy2(state_path, baseline_dir / "kernel_input_state.bin")
                baseline_runtime = run(baseline_dir)
                baseline = state_real(baseline_dir / "kernel_final_state.bin")
                baseline_tape = np.fromfile(
                    baseline_dir / "kernel_branch_tape.bin", dtype=np.int32
                )

                tangent_dir = window_root / "tangent"
                prepare(
                    generator,
                    source_dir=source,
                    executable=tangent,
                    run_dir=tangent_dir,
                    restart=restart,
                    pre_nt=pre_nt,
                    steps=window,
                )
                shutil.copy2(state_path, tangent_dir / "kernel_input_state.bin")
                direction.tofile(tangent_dir / "tangent_direction.bin")
                (tangent_dir / "tangent_nsteps.txt").write_text(f"{window}\n")
                tangent_runtime = run(tangent_dir)
                tangent_primal = np.fromfile(
                    tangent_dir / "tangent_primal_out.bin", dtype=np.float32
                )
                jv = np.fromfile(tangent_dir / "tangent_jv.bin", dtype=np.float32)
                tangent_tape = np.fromfile(
                    tangent_dir / "tangent_branch_tape.bin", dtype=np.int32
                )
                nino3_text = np.loadtxt(tangent_dir / "tangent_nino3.txt")

                step_results: list[dict] = []
                for h in args.steps:
                    side_outputs: dict[str, np.ndarray] = {}
                    side_tapes: dict[str, np.ndarray] = {}
                    side_runtime: dict[str, float] = {}
                    transformed_outputs: dict[str, np.ndarray] = {}
                    transformed_tapes: dict[str, np.ndarray] = {}
                    transformed_runtime: dict[str, float] = {}
                    for sign, name in ((1.0, "plus"), (-1.0, "minus")):
                        side_dir = window_root / f"h_{h:.8g}_{name}"
                        prepare(
                            generator,
                            source_dir=source,
                            executable=primal,
                            run_dir=side_dir,
                            restart=restart,
                            pre_nt=pre_nt,
                            steps=window,
                        )
                        perturbed = np.asarray(
                            x.astype(np.float64)
                            + sign * h * direction.astype(np.float64),
                            dtype=np.float32,
                        )
                        replace_state_real(
                            state_path, side_dir / "kernel_input_state.bin", perturbed
                        )
                        side_runtime[name] = run(side_dir)
                        side_outputs[name] = state_real(
                            side_dir / "kernel_final_state.bin"
                        )
                        side_tapes[name] = np.fromfile(
                            side_dir / "kernel_branch_tape.bin", dtype=np.int32
                        )

                        transformed_dir = (
                            window_root / f"h_{h:.8g}_{name}_tangent_primal"
                        )
                        prepare(
                            generator,
                            source_dir=source,
                            executable=tangent,
                            run_dir=transformed_dir,
                            restart=restart,
                            pre_nt=pre_nt,
                            steps=window,
                        )
                        replace_state_real(
                            state_path,
                            transformed_dir / "kernel_input_state.bin",
                            perturbed,
                        )
                        np.zeros(NREAL, dtype=np.float32).tofile(
                            transformed_dir / "tangent_direction.bin"
                        )
                        (transformed_dir / "tangent_nsteps.txt").write_text(
                            f"{window}\n"
                        )
                        transformed_runtime[name] = run(transformed_dir)
                        transformed_outputs[name] = np.fromfile(
                            transformed_dir / "tangent_primal_out.bin",
                            dtype=np.float32,
                        )
                        transformed_tapes[name] = np.fromfile(
                            transformed_dir / "tangent_branch_tape.bin",
                            dtype=np.int32,
                        )

                    fd = (
                        side_outputs["plus"].astype(np.float64)
                        - side_outputs["minus"].astype(np.float64)
                    ) / (2.0 * h)
                    transformed_fd = (
                        transformed_outputs["plus"].astype(np.float64)
                        - transformed_outputs["minus"].astype(np.float64)
                    ) / (2.0 * h)
                    jv64 = jv.astype(np.float64)
                    scalar_fd = float(np.dot(output_weight, fd))
                    scalar_tangent = float(np.dot(output_weight, jv64))
                    scalar_scale = max(abs(scalar_tangent), np.finfo(float).tiny)
                    normwise_scalar_scale = max(
                        float(np.linalg.norm(jv64)), np.finfo(float).tiny
                    )
                    nino3_jv = float(nino3_canonical(jv))
                    nino3_fd = float(
                        (
                            nino3_canonical(side_outputs["plus"])
                            - nino3_canonical(side_outputs["minus"])
                        )
                        / np.float32(2.0 * h)
                    )
                    transformed_nino3_fd = float(
                        (
                            nino3_canonical(transformed_outputs["plus"])
                            - nino3_canonical(transformed_outputs["minus"])
                        )
                        / np.float32(2.0 * h)
                    )
                    step_results.append(
                        {
                            "h": h,
                            "major_tape_matches_baseline": bool(
                                np.array_equal(side_tapes["plus"], baseline_tape)
                                and np.array_equal(side_tapes["minus"], baseline_tape)
                            ),
                            "full_relative_l2_error": relative_l2(fd, jv64),
                            "active_output_relative_l2_error": relative_l2(
                                fd, jv64, active
                            ),
                            "independent_output_relative_l2_error": relative_l2(
                                fd, jv64, independent
                            ),
                            "full_cosine": cosine(fd, jv64),
                            "active_output_cosine": cosine(fd, jv64, active),
                            "tangent_norm": float(np.linalg.norm(jv64)),
                            "full_absolute_l2_defect": float(np.linalg.norm(fd - jv64)),
                            "transformed_primal_fd_relative_l2_error": relative_l2(
                                transformed_fd, jv64
                            ),
                            "transformed_primal_fd_cosine": cosine(
                                transformed_fd, jv64
                            ),
                            "nino3_tangent": nino3_jv,
                            "nino3_forward_oracle_fd": nino3_fd,
                            "nino3_forward_oracle_fd_absolute_error": abs(
                                nino3_fd - nino3_jv
                            ),
                            "nino3_forward_oracle_fd_relative_error": abs(
                                nino3_fd - nino3_jv
                            )
                            / max(abs(nino3_jv), np.finfo(float).tiny),
                            "nino3_transformed_primal_fd": transformed_nino3_fd,
                            "nino3_transformed_primal_fd_absolute_error": abs(
                                transformed_nino3_fd - nino3_jv
                            ),
                            "nino3_transformed_primal_fd_relative_error": abs(
                                transformed_nino3_fd - nino3_jv
                            )
                            / max(abs(nino3_jv), np.finfo(float).tiny),
                            "random_output_functional_fd": scalar_fd,
                            "random_output_functional_tangent": scalar_tangent,
                            "random_output_functional_relative_error": abs(
                                scalar_fd - scalar_tangent
                            )
                            / scalar_scale,
                            "random_output_functional_normwise_defect": abs(
                                scalar_fd - scalar_tangent
                            )
                            / normwise_scalar_scale,
                            "finite_difference_runtime_seconds": sum(
                                side_runtime.values()
                            ),
                            "transformed_primal_fd_runtime_seconds": sum(
                                transformed_runtime.values()
                            ),
                            "transformed_primal_major_tape_matches_forward_sides": (
                                bool(
                                    np.array_equal(
                                        transformed_tapes["plus"], side_tapes["plus"]
                                    )
                                    and np.array_equal(
                                        transformed_tapes["minus"], side_tapes["minus"]
                                    )
                                )
                            ),
                            "transformed_plus_primal_relative_l2_difference": (
                                relative_l2(
                                    transformed_outputs["plus"].astype(np.float64),
                                    side_outputs["plus"].astype(np.float64),
                                )
                            ),
                            "transformed_minus_primal_relative_l2_difference": (
                                relative_l2(
                                    transformed_outputs["minus"].astype(np.float64),
                                    side_outputs["minus"].astype(np.float64),
                                )
                            ),
                        }
                    )

                scalar_expected = None
                scalar_derivative_expected = None
                scalar_forward_oracle = None
                if window == 31:
                    scalar_expected = float(nino3_canonical(tangent_primal))
                    scalar_derivative_expected = float(nino3_canonical(jv))
                    scalar_forward_oracle = float(nino3_canonical(baseline))
                report_cases.append(
                    {
                        "checkpoint": case["label"],
                        "window_steps": window,
                        "pre_nt": pre_nt,
                        "input_state_sha256": sha256(state_path),
                        "direction_sha256": sha256(
                            tangent_dir / "tangent_direction.bin"
                        ),
                        "direction_nonzero_count": int(np.count_nonzero(direction)),
                        "baseline_runtime_seconds": baseline_runtime,
                        "tangent_runtime_seconds_including_extra_31_step_head": (
                            tangent_runtime
                        ),
                        "tangent_is_finite": bool(np.all(np.isfinite(jv))),
                        "tangent_primal_branch_tape_matches_baseline": bool(
                            np.array_equal(tangent_tape, baseline_tape)
                        ),
                        "tangent_primal_bitwise_matches_forward_oracle": bool(
                            np.array_equal(tangent_primal, baseline)
                        ),
                        "tangent_primal_relative_l2_difference": relative_l2(
                            tangent_primal.astype(np.float64),
                            baseline.astype(np.float64),
                        ),
                        "tangent_primal_maximum_absolute_difference": float(
                            np.max(
                                np.abs(
                                    tangent_primal.astype(np.float64)
                                    - baseline.astype(np.float64)
                                )
                            )
                        ),
                        "scalar_head_nino3": float(nino3_text[0]),
                        "scalar_head_nino3_tangent": float(nino3_text[1]),
                        "scalar_head_expected_from_tangent_generic_window": (
                            scalar_expected
                        ),
                        "scalar_head_tangent_expected_from_generic_window": (
                            scalar_derivative_expected
                        ),
                        "scalar_forward_oracle_from_generic_window": (
                            scalar_forward_oracle
                        ),
                        "scalar_head_matches_tangent_generic_window_bitwise": (
                            scalar_expected is None
                            or np.float32(nino3_text[0]).tobytes()
                            == np.float32(scalar_expected).tobytes()
                        ),
                        "scalar_head_tangent_matches_generic_window_bitwise": (
                            scalar_derivative_expected is None
                            or np.float32(nino3_text[1]).tobytes()
                            == np.float32(scalar_derivative_expected).tobytes()
                        ),
                        "centered_difference_steps": step_results,
                    }
                )

        if (
            verify_audit_build(
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
        if tree_file_hashes(bound_kernel_source) != kernel_source_hashes:
            raise RuntimeError("kernel source changed during validation")
        if tree_file_hashes(bound_runtime_source) != runtime_source_hashes:
            raise RuntimeError("runtime source changed during validation")
        if (
            replay_report_path is not None
            and sha256(replay_report_path) != replay_report_sha256
        ):
            raise RuntimeError("replay-validation report changed during validation")
        if replay_report_path is not None and verify_replay_report_binding(
            replay_report_path,
            kernel_build_report=kernel_build_report,
            primal=bound_primal,
        ) != replay_binding:
            raise RuntimeError("replay-validation producer binding changed")
        current_runtime_source_binding = (
            verify_replay_runtime_source(
                bound_runtime_source,
                replay_report_path,
                kernel_build_report=kernel_build_report,
                generator=generator,
            )
            if replay_report_path is not None
            else {
                "authentication": "explicit_runtime_source_not_replay_authenticated",
                "runtime_source_complete_tree_sha256": file_mapping_sha256(
                    tree_file_hashes(bound_runtime_source)
                ),
                "runtime_source_complete_tree_entry_count": len(
                    tree_file_hashes(bound_runtime_source)
                ),
            }
        )
        if current_runtime_source_binding != runtime_source_binding:
            raise RuntimeError("runtime-source provenance changed during validation")
        for case in bound_cases:
            if sha256(Path(case["restart"])) != checkpoint_hashes[case["label"]]:
                raise RuntimeError(f"checkpoint changed: {case['label']}")
        if sha256(manifest_path) != state_manifest_sha256:
            raise RuntimeError("state manifest changed during validation")
        if sha256(validator_path) != validator_sha256:
            raise RuntimeError("tangent validator changed during validation")
        if (
            verify_private_staged_inputs(
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
                "major-tape-conditioned tangent-linear comparison for the full "
                "coupled explicit-state ZC window. The recorded tape covers only "
                "SST substeps, atmosphere iterations, and reset; it does not observe "
                "every internal sign, cap, or upwind decision. This report applies "
                "no acceptance threshold and is not an adjoint certification."
            ),
            "compiler_mode": (
                "gfortran -std=legacy; float32 model arithmetic with canonical "
                "float64 Nino-3 reduction"
            ),
            "state_real_length": NREAL,
            "direction": (
                f"segment-scaled random direction over {args.direction_scope} state"
            ),
            "executables": {
                "forward_oracle_with_reentrant_cforce_sha256": sha256(primal),
                "tapenade_tangent_sha256": sha256(tangent),
            },
            "tapenade_generation_provenance": str(generation_provenance),
            "tapenade_generation_provenance_sha256": sha256(generation_provenance),
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
            "replay_validation_report_sha256": (replay_report_sha256),
            "replay_validation_binding": replay_binding,
            "runtime_source_binding": runtime_source_binding,
            "state_manifest_sha256": state_manifest_sha256,
            "private_staged_input_binding": private_input_binding,
            "cases": report_cases,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
