#!/usr/bin/env python3
"""Test the segmented ZC adjoint against the matching tangent map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import validate_tangent as tangent_audit
from output_safety import transactional_output

NREAL = tangent_audit.NREAL
PATH_HEADER_BYTES = 7 * 4
PATH_STATES = 32
OUTPUT_WEIGHT_NORM = math.sqrt(66.0 * (1.0 / 66.0) ** 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(directory: Path) -> tuple[str, dict[str, str]]:
    entries = {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    digest = hashlib.sha256()
    for name, file_hash in entries.items():
        digest.update(f"{file_hash}  {name}\n".encode())
    return digest.hexdigest(), entries


def parse_key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def verify_producer_runtime(
    run_dir: Path,
    run_report_path: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    """Authenticate the exact case runtime declared by its producer report."""

    expected_run_dir = run_report_path.resolve().parent / "runtime"
    if run_dir.resolve() != expected_run_dir:
        raise RuntimeError("--run-dir must be the producer case's runtime directory")
    producer = json.loads(run_report_path.read_text(encoding="utf-8"))
    raw_manifest = producer.get("checkpoint_input", {}).get("staged_file_sha256")
    if not isinstance(raw_manifest, dict) or not raw_manifest:
        raise RuntimeError("producer report has no staged runtime-file manifest")
    manifest: dict[str, str] = {}
    for raw_name, raw_digest in raw_manifest.items():
        if not isinstance(raw_name, str) or not isinstance(raw_digest, str):
            raise ValueError("producer runtime manifest must map paths to SHA-256")
        relative = Path(raw_name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(raw_digest) != 64
            or any(character not in "0123456789abcdef" for character in raw_digest)
        ):
            raise ValueError(f"invalid producer runtime manifest entry: {raw_name!r}")
        name = relative.as_posix()
        if name in manifest:
            raise ValueError(f"duplicate producer runtime manifest entry: {name}")
        manifest[name] = raw_digest
    verify_staged_producer_runtime(run_dir, manifest, label="producer runtime")
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest, {
        "run_report_sha256": sha256(run_report_path),
        "runtime_manifest_sha256": digest,
        "runtime_file_count": len(manifest),
    }


def verify_staged_producer_runtime(
    run_dir: Path,
    manifest: dict[str, str],
    *,
    label: str = "staged producer runtime",
) -> dict[str, str]:
    """Reverify every bound runtime input without rejecting derivative outputs."""

    observed: dict[str, str] = {}
    for name, expected in manifest.items():
        source = run_dir
        for part in Path(name).parts:
            source /= part
            if source.is_symlink():
                raise RuntimeError(f"{label} path became a symlink: {name}")
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise RuntimeError(f"{label} manifest mismatch: {source}")
        observed[name] = expected
    return observed


def stage_producer_runtime(
    run_dir: Path,
    destination: Path,
    manifest: dict[str, str],
) -> None:
    """Copy only producer-bound runtime inputs into a private execution tree."""

    destination.mkdir(parents=True)
    for name, expected in manifest.items():
        source = run_dir / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f"producer runtime changed while staging: {name}")
        target.write_bytes(payload)
        if target.is_symlink() or sha256(target) != expected:
            raise RuntimeError(f"staged producer runtime differs: {name}")
    (destination / "EOF_data" / "to").mkdir(parents=True, exist_ok=True)


def stage_verified_file(source: Path, destination: Path, expected: str) -> Path:
    """Copy one already hashed input and reject a concurrent byte change."""

    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"audit input changed while staging: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    if destination.is_symlink() or sha256(destination) != expected:
        raise RuntimeError(f"staged audit input differs from {source}")
    return destination


def run_bound_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    label: str,
) -> None:
    """Execute one privately staged derivative binary in its bound runtime."""

    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"{label} failed: {log_path}")


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
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(f"invalid manifest entry: {line!r}")
        name = relative_path.as_posix()
        if name in declared:
            raise ValueError(f"duplicate manifest entry: {name}")
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
            raise RuntimeError("exact snapshot inventory does not match manifest")
    return len(declared)


def verify_tangent_build(
    *,
    tangent_executable: Path,
    tangent_build_manifest: Path,
    tangent_build_inputs: Path,
    coupled_build_manifest: Path,
    coupled_build_input_manifest: Path,
    coupled_build_snapshot: Path,
) -> dict[str, object]:
    """Bind the O3 audit tangent to the exact coupled-build snapshot."""

    tangent = parse_key_values(tangent_build_manifest)
    coupled = parse_key_values(coupled_build_manifest)
    if tangent.get("schema") != "zc-tangent-audit-build-v1":
        raise ValueError("unsupported tangent build-manifest schema")
    if tangent.get("optimization") != "O3":
        raise RuntimeError("release transpose evidence requires the O3 audit tangent")
    if tangent.get("path_executable_sha256") != sha256(tangent_executable):
        raise RuntimeError("tangent executable does not match its build manifest")
    if tangent.get("tangent_build_inputs_sha256") != sha256(tangent_build_inputs):
        raise RuntimeError("tangent input manifest is not bound by its build report")
    if tangent.get("coupled_build_manifest_sha256") != sha256(coupled_build_manifest):
        raise RuntimeError("tangent build does not bind the coupled build manifest")
    if tangent.get("coupled_build_input_manifest_sha256") != sha256(
        coupled_build_input_manifest
    ):
        raise RuntimeError("tangent build does not bind the coupled input manifest")
    if coupled.get("build_input_manifest_sha256") != sha256(
        coupled_build_input_manifest
    ):
        raise RuntimeError("coupled build does not bind its exact input manifest")
    exact_count = verify_sha256_manifest(
        coupled_build_snapshot,
        coupled_build_input_manifest,
        exact_inventory=True,
    )
    tangent_count = verify_sha256_manifest(
        coupled_build_snapshot,
        tangent_build_inputs,
        exact_inventory=False,
    )
    return {
        "tangent_build_manifest_sha256": sha256(tangent_build_manifest),
        "tangent_build_inputs_sha256": sha256(tangent_build_inputs),
        "coupled_build_manifest_sha256": sha256(coupled_build_manifest),
        "coupled_build_input_manifest_sha256": sha256(coupled_build_input_manifest),
        "coupled_snapshot_entry_count": exact_count,
        "tangent_compile_input_count": tangent_count,
        "optimization": tangent["optimization"],
    }


def verify_scalar_gradient_producer(
    *,
    run_report_path: Path,
    path_file: Path,
    gradient_file: Path,
    reverse_executable: Path,
    coupled_build_manifest: Path,
    coupled_build_input_manifest: Path,
) -> dict[str, object]:
    """Cross-bind a supplied scalar gradient to the reverse run that made it."""

    producer = json.loads(run_report_path.read_text(encoding="utf-8"))
    if producer.get("schema_version") != 1 or producer.get("status") != "completed":
        raise ValueError("unsupported or incomplete adjoint producer report")
    contract = producer.get("scientific_contract", {})
    if contract.get("transitions") != 31 or contract.get("real_state_length") != NREAL:
        raise RuntimeError("producer report has the wrong scientific dimensions")
    if contract.get("objective") != "canonical Nino-3 SST anomaly":
        raise RuntimeError("scalar dot test requires the canonical Nino-3 gradient")
    build = producer.get("build", {})
    if build.get("adjoint_executable_sha256") != sha256(reverse_executable):
        raise RuntimeError("producer report does not bind the reverse executable")
    executed = build.get("executed_staged_executable_sha256", {})
    if executed.get(reverse_executable.name) != sha256(reverse_executable):
        raise RuntimeError("producer report does not bind the executed reverse bytes")
    if build.get("build_manifest_sha256") != sha256(coupled_build_manifest):
        raise RuntimeError("producer report does not bind the coupled build")
    coupled = parse_key_values(coupled_build_manifest)
    input_manifest_hash = sha256(coupled_build_input_manifest)
    if (
        coupled.get("build_input_manifest_sha256") != input_manifest_hash
        or build.get("build_input_manifest_sha256") != input_manifest_hash
    ):
        raise RuntimeError(
            "producer report does not bind the exact coupled-build input manifest"
        )
    if contract.get("configuration_verified") is not True:
        raise RuntimeError("producer report does not verify an approved runtime case")
    output_hashes = producer.get("results", {}).get("output_sha256", {})
    path_hash = sha256(path_file)
    gradient_hash = sha256(gradient_file)
    if output_hashes.get("artifacts/zc_31step_path.bin") != path_hash:
        raise RuntimeError("producer report does not bind the supplied path")
    if output_hashes.get("artifacts/zc_nino3_gradient.bin") != gradient_hash:
        raise RuntimeError("producer report does not bind the supplied gradient")
    return {
        "run_report_sha256": sha256(run_report_path),
        "path_sha256": path_hash,
        "gradient_sha256": gradient_hash,
        "reverse_executable_sha256": sha256(reverse_executable),
        "coupled_build_manifest_sha256": sha256(coupled_build_manifest),
        "coupled_build_input_manifest_sha256": input_manifest_hash,
    }


def verify_path_producer(
    *,
    run_report_path: Path,
    path_file: Path,
    reverse_executable: Path,
    coupled_build_manifest: Path,
    coupled_build_input_manifest: Path,
) -> dict[str, object]:
    """Bind a bound forward path and reverse binary to one producer run report."""

    producer = json.loads(run_report_path.read_text(encoding="utf-8"))
    if producer.get("schema_version") != 1 or producer.get("status") != "completed":
        raise ValueError("unsupported or incomplete adjoint producer report")
    build = producer.get("build", {})
    if build.get("adjoint_executable_sha256") != sha256(reverse_executable):
        raise RuntimeError("producer report does not bind the reverse executable")
    executed = build.get("executed_staged_executable_sha256", {})
    if executed.get(reverse_executable.name) != sha256(reverse_executable):
        raise RuntimeError("producer report does not bind the executed reverse bytes")
    if build.get("build_manifest_sha256") != sha256(coupled_build_manifest):
        raise RuntimeError("producer report does not bind the coupled build")
    coupled = parse_key_values(coupled_build_manifest)
    input_manifest_hash = sha256(coupled_build_input_manifest)
    if (
        coupled.get("build_input_manifest_sha256") != input_manifest_hash
        or build.get("build_input_manifest_sha256") != input_manifest_hash
    ):
        raise RuntimeError(
            "producer report does not bind the exact coupled-build input manifest"
        )
    contract = producer.get("scientific_contract", {})
    if contract.get("transitions") != 31 or contract.get("real_state_length") != NREAL:
        raise RuntimeError("producer report has the wrong scientific dimensions")
    if contract.get("configuration_verified") is not True:
        raise RuntimeError("producer report does not verify an approved runtime case")
    path_hash = sha256(path_file)
    output_hashes = producer.get("results", {}).get("output_sha256", {})
    if output_hashes.get("artifacts/zc_31step_path.bin") != path_hash:
        raise RuntimeError("producer report does not bind the supplied path")
    return {
        "run_report_sha256": sha256(run_report_path),
        "path_sha256": path_hash,
        "reverse_executable_sha256": sha256(reverse_executable),
        "coupled_build_manifest_sha256": sha256(coupled_build_manifest),
        "coupled_build_input_manifest_sha256": input_manifest_hash,
    }


def read_initial_real(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=7)
    expected = np.array([31, NREAL, 5280, 1, 4, 2, 3], dtype=np.int32)
    if not np.array_equal(header, expected):
        raise RuntimeError(f"unexpected validated-path header: {header.tolist()}")
    values = np.fromfile(
        path,
        dtype=np.float32,
        count=NREAL * PATH_STATES,
        offset=PATH_HEADER_BYTES,
    )
    if values.size != NREAL * PATH_STATES:
        raise RuntimeError("truncated certified real-state path")
    return values.reshape((NREAL, PATH_STATES), order="F")[:, 0].copy()


def parse_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def precise_dot(left: np.ndarray, right: np.ndarray) -> float:
    return math.fsum(float(x) * float(y) for x, y in zip(left, right, strict=True))


def canonical_tangent_head(jv: np.ndarray) -> float:
    values = []
    for j in tangent_audit.NINO3_J:
        for i in tangent_audit.NINO3_I:
            k = tangent_audit.NINO3_OFFSET + i + 30 * (j - 1) - 1
            values.append(float(jv[k]))
    return math.fsum(values) / 66.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7301, 7302, 7303])
    parser.add_argument(
        "--direction-scope", choices=("active", "independent"), default="active"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--path-file", type=Path, required=True)
    parser.add_argument("--gradient-file", type=Path, required=True)
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
    validator_sha256 = sha256(validator_path)

    root = args.project_root.resolve()
    output_requested = args.output_dir

    run_dir = args.run_dir.resolve()
    path_file = args.path_file.resolve()
    gradient_file = args.gradient_file.resolve()
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
    if {
        "prepared": prepared_dir,
        "tangent": tangent_source_dir,
        "reverse": reverse_source_dir,
    } != {
        "prepared": coupled_build_snapshot / "prepared",
        "tangent": coupled_build_snapshot / "generated/tangent",
        "reverse": coupled_build_snapshot / "generated/reverse",
    }:
        raise RuntimeError(
            "prepared/tangent/reverse sources must come from the declared "
            "coupled-build snapshot"
        )

    manifest_path = root / "adjoint/fortran_kernel/state_manifest.json"
    for required in (
        path_file,
        gradient_file,
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

    state_manifest_sha256 = sha256(manifest_path)
    tangent_tree_hash, tangent_files = tree_manifest(tangent_source_dir)
    reverse_tree_hash, reverse_files = tree_manifest(reverse_source_dir)
    prepared_tree_hash, prepared_files = tree_manifest(prepared_dir)
    tangent_build = verify_tangent_build(
        tangent_executable=tangent_executable,
        tangent_build_manifest=tangent_build_manifest,
        tangent_build_inputs=tangent_build_inputs,
        coupled_build_manifest=coupled_build_manifest,
        coupled_build_input_manifest=coupled_build_input_manifest,
        coupled_build_snapshot=coupled_build_snapshot,
    )
    producer_binding = verify_scalar_gradient_producer(
        run_report_path=producer_run_report,
        path_file=path_file,
        gradient_file=gradient_file,
        reverse_executable=reverse_executable,
        coupled_build_manifest=coupled_build_manifest,
        coupled_build_input_manifest=coupled_build_input_manifest,
    )
    runtime_manifest, runtime_binding = verify_producer_runtime(
        run_dir, producer_run_report
    )

    output_transaction = transactional_output(
        output_requested,
        overwrite=args.overwrite,
        protected_paths=(
            run_dir,
            path_file,
            gradient_file,
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
        owner="validate_adjoint_dot",
    )
    with output_transaction as output:
        verified_inputs = output / "verified_inputs"
        verified_inputs.mkdir()
        staged_tangent_executable = verified_inputs / tangent_executable.name
        staged_reverse_executable = verified_inputs / reverse_executable.name
        staged_path_file = verified_inputs / path_file.name
        staged_gradient_file = verified_inputs / gradient_file.name
        staged_manifest_path = verified_inputs / "state_manifest.json"
        for source, destination, expected_hash in (
            (tangent_executable, staged_tangent_executable, sha256(tangent_executable)),
            (
                reverse_executable,
                staged_reverse_executable,
                producer_binding["reverse_executable_sha256"],
            ),
            (path_file, staged_path_file, producer_binding["path_sha256"]),
            (gradient_file, staged_gradient_file, producer_binding["gradient_sha256"]),
            (manifest_path, staged_manifest_path, state_manifest_sha256),
        ):
            stage_verified_file(source, destination, str(expected_hash))
        staged_tangent_executable.chmod(
            staged_tangent_executable.stat().st_mode | 0o100
        )
        staged_reverse_executable.chmod(
            staged_reverse_executable.stat().st_mode | 0o100
        )
        staged_run_dir = verified_inputs / "producer_runtime"
        stage_producer_runtime(run_dir, staged_run_dir, runtime_manifest)

        # Re-execute the bound reverse in the private producer runtime.  The
        # supplied gradient is retained as a producer artifact, but numerical
        # tests consume the regenerated bytes only after exact agreement.
        regenerated_root = output / "regenerated_reverse"
        regenerated_root.mkdir()
        regenerated_prefix = regenerated_root / "zc_nino3"
        run_bound_command(
            [
                str(staged_reverse_executable),
                str(staged_path_file),
                str(regenerated_prefix),
            ],
            cwd=staged_run_dir,
            log_path=regenerated_root / "run.log",
            label="reverse regeneration",
        )
        regenerated_gradient_file = regenerated_root / "zc_nino3_gradient.bin"
        regenerated_metadata_file = regenerated_root / "zc_nino3_gradient.txt"
        if not regenerated_gradient_file.is_file():
            raise RuntimeError("reverse regeneration did not produce a gradient")
        if regenerated_gradient_file.read_bytes() != staged_gradient_file.read_bytes():
            raise RuntimeError(
                "re-executed reverse gradient differs from producer-bound gradient"
            )
        regenerated_metadata = parse_metadata(regenerated_metadata_file)
        if (
            regenerated_metadata.get("objective")
            != "canonical Nino-3 SST anomaly"
            or regenerated_metadata.get("terminal_seed") != "canonical_nino3"
            or regenerated_metadata.get("transitions") != "31"
            or regenerated_metadata.get("real_state_length") != str(NREAL)
        ):
            raise RuntimeError("re-executed reverse metadata violates scalar contract")

        # Parse scientific arrays only from private, verified copies.
        gradient = np.fromfile(regenerated_gradient_file, dtype=np.float32)
        if gradient.shape != (NREAL,) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("invalid staged reverse gradient")
        initial = read_initial_real(staged_path_file)
        manifest = json.loads(staged_manifest_path.read_text(encoding="utf-8"))
        _, _, segments = tangent_audit.active_masks(manifest)

        records: list[dict] = []
        for seed in args.seeds:
            case_dir = output / f"seed_{seed}"
            case_dir.mkdir()
            direction = tangent_audit.scaled_direction(
                initial,
                segments,
                seed=seed,
                independent_only=args.direction_scope == "independent",
            )
            direction_file = case_dir / "direction.bin"
            direction.tofile(direction_file)
            prefix = case_dir / "tangent_path"
            run_bound_command(
                [
                    str(staged_tangent_executable),
                    str(staged_path_file),
                    str(direction_file),
                    str(prefix),
                ],
                cwd=staged_run_dir,
                log_path=case_dir / "run.log",
                label="tangent run",
            )

            jv_file = case_dir / "tangent_path_jv.bin"
            jv = np.fromfile(jv_file, dtype=np.float32)
            if jv.shape != (NREAL,) or not np.all(np.isfinite(jv)):
                raise RuntimeError(f"invalid tangent output for seed {seed}")
            metadata = parse_metadata(case_dir / "tangent_path_jv.txt")
            tangent_head = canonical_tangent_head(jv)
            adjoint_dot = precise_dot(
                direction.astype(np.float64), gradient.astype(np.float64)
            )
            absolute_defect = abs(tangent_head - adjoint_dot)
            direction_norm = float(np.linalg.norm(direction.astype(np.float64)))
            jv_norm = float(np.linalg.norm(jv.astype(np.float64)))
            gradient_norm = float(np.linalg.norm(gradient.astype(np.float64)))
            transpose_scale = (
                jv_norm * OUTPUT_WEIGHT_NORM + direction_norm * gradient_norm
            )
            symmetric_scale = max(
                abs(tangent_head), abs(adjoint_dot), np.finfo(float).tiny
            )
            bilinear_relative_defect = absolute_defect / symmetric_scale
            records.append(
                {
                    "seed": seed,
                    "direction_scope": args.direction_scope,
                    "direction_sha256": sha256(direction_file),
                    "direction_nonzero_count": int(np.count_nonzero(direction)),
                    "direction_l2_norm": direction_norm,
                    "output_weight_l2_norm": OUTPUT_WEIGHT_NORM,
                    "tangent_head": tangent_head,
                    "tangent_driver_head": float(metadata["nino3_tangent"]),
                    "adjoint_dot": adjoint_dot,
                    "absolute_defect": absolute_defect,
                    "bilinear_relative_defect": bilinear_relative_defect,
                    "symmetric_relative_defect": bilinear_relative_defect,
                    "norm_scaled_transpose_defect": absolute_defect
                    / max(transpose_scale, np.finfo(float).tiny),
                    "norm_scaled_transpose_denominator": transpose_scale,
                    "jv_l2_norm": jv_norm,
                    "tangent_cpu_seconds": float(metadata["cpu_seconds"]),
                    "generated_fwd_worst_max_abs_vs_bound_forward_path": float(
                        metadata["fwd_real_worst_max_abs"]
                    ),
                    "generated_fwd_worst_relative_l2_vs_bound_forward_path": float(
                        metadata["fwd_real_worst_rel_l2"]
                    ),
                }
            )

        if (
            verify_tangent_build(
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
            verify_scalar_gradient_producer(
                run_report_path=producer_run_report,
                path_file=path_file,
                gradient_file=gradient_file,
                reverse_executable=reverse_executable,
                coupled_build_manifest=coupled_build_manifest,
                coupled_build_input_manifest=coupled_build_input_manifest,
            )
            != producer_binding
        ):
            raise RuntimeError("reverse producer provenance changed during validation")
        if (
            tree_manifest(prepared_dir)[0] != prepared_tree_hash
            or tree_manifest(tangent_source_dir)[0] != tangent_tree_hash
            or tree_manifest(reverse_source_dir)[0] != reverse_tree_hash
        ):
            raise RuntimeError("coupled source tree changed during validation")
        if sha256(validator_path) != validator_sha256:
            raise RuntimeError("scalar dot validator changed during validation")
        if sha256(manifest_path) != state_manifest_sha256:
            raise RuntimeError("state manifest changed during validation")
        if sha256(staged_manifest_path) != state_manifest_sha256:
            raise RuntimeError("staged state manifest changed during validation")
        if sha256(staged_path_file) != producer_binding["path_sha256"]:
            raise RuntimeError("staged forward path changed during validation")
        if sha256(staged_gradient_file) != producer_binding["gradient_sha256"]:
            raise RuntimeError("staged gradient changed during validation")
        if sha256(staged_tangent_executable) != sha256(tangent_executable):
            raise RuntimeError("executed tangent changed during validation")
        if sha256(staged_reverse_executable) != producer_binding[
            "reverse_executable_sha256"
        ]:
            raise RuntimeError("executed reverse changed during validation")
        if sha256(regenerated_gradient_file) != producer_binding["gradient_sha256"]:
            raise RuntimeError("regenerated gradient changed during validation")
        if (
            verify_staged_producer_runtime(staged_run_dir, runtime_manifest)
            != runtime_manifest
        ):
            raise RuntimeError("staged producer runtime changed during validation")
        if verify_producer_runtime(run_dir, producer_run_report) != (
            runtime_manifest,
            runtime_binding,
        ):
            raise RuntimeError("producer runtime changed during validation")

        report = {
            "schema_version": 1,
            "status": "completed_not_certified",
            "acceptance_thresholds": None,
            "certification_assessed": False,
            "created_utc": datetime.now(UTC).isoformat(),
            "validator_script_sha256": validator_sha256,
            "claim": (
                "31-step tangent-reverse bilinear comparisons. The scalar reverse "
                "is re-executed in a private copy of its producer-bound runtime, and "
                "its regenerated gradient must match the supplied producer artifact "
                "bitwise. That artifact is cryptographically cross-bound to its "
                "reverse-run report, executable, coupled build, and forward path; "
                "the O3 audit tangent is independently bound to the same coupled "
                "build snapshot. This report records defects but applies no acceptance "
                "threshold and is not, by itself, derivative certification."
            ),
            "state_real_length": NREAL,
            "transitions": 31,
            "executables": {
                "tangent_path_driver": str(tangent_executable),
                "tangent_path_driver_sha256": sha256(tangent_executable),
                "reverse_driver": str(reverse_executable),
                "reverse_driver_sha256": sha256(reverse_executable),
            },
            "inputs": {
                "bound_forward_path": str(path_file),
                "bound_forward_path_sha256": sha256(path_file),
                "reverse_gradient": str(gradient_file),
                "reverse_gradient_sha256": sha256(gradient_file),
                "regenerated_reverse_gradient_sha256": sha256(
                    regenerated_gradient_file
                ),
                "regenerated_reverse_metadata_sha256": sha256(
                    regenerated_metadata_file
                ),
                "regenerated_reverse_metadata": regenerated_metadata,
                "reverse_gradient_l2_norm": float(
                    np.linalg.norm(gradient.astype(np.float64))
                ),
            },
            "source_manifests": {
                "prepared_sha256": prepared_tree_hash,
                "tangent_sha256": tangent_tree_hash,
                "reverse_sha256": reverse_tree_hash,
                "prepared_files": prepared_files,
                "tangent_files": tangent_files,
                "reverse_files": reverse_files,
            },
            "tangent_build_provenance": tangent_build,
            "reverse_gradient_producer_binding": producer_binding,
            "producer_runtime_binding": runtime_binding,
            "state_manifest_sha256": state_manifest_sha256,
            "tests": records,
            "maximum_symmetric_relative_defect": max(
                record["symmetric_relative_defect"] for record in records
            ),
            "maximum_bilinear_relative_defect": max(
                record["bilinear_relative_defect"] for record in records
            ),
            "maximum_norm_scaled_transpose_defect": max(
                record["norm_scaled_transpose_defect"] for record in records
            ),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
