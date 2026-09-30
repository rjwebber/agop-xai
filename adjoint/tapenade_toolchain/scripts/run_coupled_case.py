#!/usr/bin/env python3
"""Run one authentic ZC checkpoint through the 31-step adjoint workflow."""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

N_STEPS = 31
N_REAL = 59_148
N_COMPLEX = 5_280
N_DOUBLE = 1
N_INTEGER = 4
N_TIME = 2
TAPE_WIDTH = 3
FLOAT32_BYTES = 4
STATE_BYTES = 278_864
HISTORY_BYTES = 278_808
OUTPUT_MARKER = ".zc_adjoint_case_output"
OUTPUT_MARKER_CONTENT = "managed-zc-adjoint-case-output-v1\n"

REQUIRED_RUNTIME_FILES = (
    "fc.data",
    "zeq9fsu.hst",
    "kernel_initial_state.bin",
    "modified_means.namelist",
    "scales_EOF.namelist",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def verify_sha256_manifest(
    root: Path,
    manifest: Path,
    *,
    exact_inventory: bool = False,
) -> int:
    entries = 0
    seen: set[str] = set()
    root = root.resolve()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative_name = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"malformed SHA-256 manifest line: {line!r}") from error
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid SHA-256 digest in {manifest}: {expected!r}")
        relative_name = relative_name.strip()
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest path: {relative_name!r}")
        if relative_name in seen:
            raise ValueError(f"duplicate manifest path: {relative_name!r}")
        seen.add(relative_name)
        target = (root / relative_path).resolve()
        if root != target.parent and root not in target.parents:
            raise ValueError(f"manifest path escapes its root: {relative_name!r}")
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(f"manifest target is missing: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(f"manifest hash mismatch: {target}")
        entries += 1
    if entries == 0:
        raise ValueError(f"empty SHA-256 manifest: {manifest}")
    if exact_inventory:
        actual_paths: set[str] = set()
        for target in root.rglob("*"):
            if target.is_symlink():
                raise ValueError(
                    f"exact-inventory root contains a symbolic link: {target}"
                )
            if target.is_file():
                actual_paths.add(target.relative_to(root).as_posix())
            elif not target.is_dir():
                raise ValueError(
                    f"exact-inventory root contains a special file: {target}"
                )
        if actual_paths != seen:
            missing = sorted(seen - actual_paths)
            extra = sorted(actual_paths - seen)
            raise RuntimeError(
                f"exact manifest inventory mismatch; missing={missing}, extra={extra}"
            )
    return entries


def canonical_runtime_text(name: str, text: str) -> str:
    """Mask only case timing values; every physics value remains hash-bound."""

    if name == "fc.data":
        result, count = re.subn(
            r"^(TFIND|TZERO|TENDD)(\s*=).*$",
            r"\1\2 <dynamic>",
            text,
            flags=re.MULTILINE,
        )
        if count != 3:
            raise ValueError(f"fc.data had {count} dynamic timing fields; expected 3")
        return result
    if name == "modified_means.namelist":
        result, count = re.subn(
            r"^( time_(?:start|end)_writing_grads_data=).*$",
            r"\1<dynamic>,",
            text,
            flags=re.MULTILINE,
        )
        if count != 2:
            raise ValueError(
                "modified_means.namelist had "
                f"{count} dynamic output-time fields; expected 2"
            )
        return result
    raise ValueError(f"no runtime canonicalization rule for {name}")


def text_assignments(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for name in names:
        match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*([^,\s]+)", text)
        if match is None:
            raise ValueError(f"required runtime assignment {name} is missing in {path}")
        values[name] = match.group(1)
    return values


def verify_runtime_contract(
    checkpoint: Path,
    contract_path: Path,
    producer_manifest_path: Path | None,
) -> dict[str, object]:
    """Verify the locked physics configuration and an approved state/history pair."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    canonical_hashes: dict[str, str] = {}
    for name, expected in contract["canonical_text_sha256"].items():
        canonical = canonical_runtime_text(
            name, (checkpoint / name).read_text(encoding="utf-8")
        )
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        canonical_hashes[name] = actual
        if actual != expected:
            raise RuntimeError(f"unsupported ZC physics configuration in {name}")
    for name, expected in contract["exact_file_sha256"].items():
        if sha256_file(checkpoint / name) != expected:
            raise RuntimeError(f"unsupported ZC runtime file: {name}")

    expected_data: dict[str, str] = contract["data_files_sha256"]
    actual_data_paths = {
        path.relative_to(checkpoint).as_posix()
        for path in (checkpoint / "Data").rglob("*")
        if path.is_file()
    }
    if actual_data_paths != set(expected_data):
        missing = sorted(set(expected_data) - actual_data_paths)
        extra = sorted(actual_data_paths - set(expected_data))
        raise RuntimeError(
            f"static Data/ contract mismatch; missing={missing}, extra={extra}"
        )
    for relative_name, expected in expected_data.items():
        if sha256_file(checkpoint / relative_name) != expected:
            raise RuntimeError(f"static Data/ hash mismatch: {relative_name}")

    dynamic_names = (
        "TFIND",
        "TZERO",
        "TENDD",
        "NSTART",
        "NIC",
    )
    dynamic_values = text_assignments(checkpoint / "fc.data", dynamic_names)
    if dynamic_values["NSTART"] != "3" or dynamic_values["NIC"] != "0":
        raise RuntimeError("adjoint requires NSTART=3 and deterministic NIC=0")
    dynamic_values.update(
        text_assignments(
            checkpoint / "modified_means.namelist",
            (
                "time_start_writing_grads_data",
                "time_end_writing_grads_data",
            ),
        )
    )

    paired_names = tuple(next(iter(contract["approved_cases"].values())))
    actual_pair = {name: sha256_file(checkpoint / name) for name in paired_names}
    matching_cases = [
        case_id
        for case_id, expected in contract["approved_cases"].items()
        if actual_pair == expected
    ]
    if len(matching_cases) > 1:
        raise RuntimeError("runtime fingerprints ambiguously match multiple cases")

    if producer_manifest_path is None and matching_cases:
        producer_manifest_path = (
            contract_path.parent / "runtime_manifests" / f"{matching_cases[0]}.json"
        )
    producer_verified = False
    producer_manifest: dict[str, object] | None = None
    if producer_manifest_path is not None:
        producer_manifest = json.loads(
            producer_manifest_path.read_text(encoding="utf-8")
        )
        case_id = str(producer_manifest.get("case_id", ""))
        if producer_manifest.get("contract_id") != contract["contract_id"]:
            raise RuntimeError("producer manifest names a different runtime contract")
        expected_pair = contract["approved_cases"].get(case_id)
        if expected_pair is None or actual_pair != expected_pair:
            raise RuntimeError(
                "checkpoint/history files do not match the approved producer manifest"
            )
        producer_verified = True

    return {
        "configuration_verified": producer_verified,
        "physics_configuration_verified": True,
        "checkpoint_history_pair_verified": producer_verified,
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(contract_path),
        "canonical_text_sha256": canonical_hashes,
        "static_data_file_count": len(expected_data),
        "dynamic_values": dynamic_values,
        "runtime_file_sha256": actual_pair,
        "runtime_source_sha256": relative_manifest(
            checkpoint,
            [
                *(checkpoint / name for name in REQUIRED_RUNTIME_FILES),
                *(path for path in (checkpoint / "Data").rglob("*") if path.is_file()),
            ],
        ),
        "approved_case_id": matching_cases[0] if matching_cases else None,
        "producer_manifest": producer_manifest,
        "producer_manifest_sha256": (
            sha256_file(producer_manifest_path)
            if producer_manifest_path is not None
            else None
        ),
    }


def verify_build_provenance(build_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    compiled = build_dir / "coupled_compiled"
    snapshot = compiled / "build_input_snapshot"
    build_manifest_path = compiled / "build_manifest.txt"
    build_manifest = parse_key_values(build_manifest_path)
    expected_compile_roots = {
        "prepared_compile_inputs_root": "build_input_snapshot/prepared",
        "tangent_compile_inputs_root": "build_input_snapshot/generated/tangent",
        "reverse_compile_inputs_root": "build_input_snapshot/generated/reverse",
    }
    for key, expected in expected_compile_roots.items():
        if build_manifest.get(key) != expected:
            raise RuntimeError(f"compiled build has an unexpected {key}")
    contracts = (
        (
            "prepared_compile_inputs_sha256",
            compiled / "prepared_compile_inputs.sha256",
            snapshot / "prepared",
            True,
        ),
        (
            "tangent_compile_inputs_sha256",
            compiled / "tangent_compile_inputs.sha256",
            snapshot / "generated/tangent",
            True,
        ),
        (
            "reverse_compile_inputs_sha256",
            compiled / "reverse_compile_inputs.sha256",
            snapshot / "generated/reverse",
            True,
        ),
        (
            "prepared_source_manifest_sha256",
            snapshot / "prepared/prepared_source_manifest.sha256",
            snapshot / "prepared",
            False,
        ),
        (
            "generated_source_manifest_sha256",
            snapshot / "generated/generated_source_manifest.sha256",
            snapshot / "generated",
            False,
        ),
        (
            "build_input_manifest_sha256",
            compiled / "build_input_manifest.sha256",
            snapshot,
            True,
        ),
        (
            "toolchain_patch_manifest_sha256",
            compiled / "toolchain_patch_manifest.sha256",
            snapshot / "toolchain/patches",
            True,
        ),
    )
    counts: dict[str, int] = {}
    for key, manifest_path, manifest_root, exact_inventory in contracts:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"required provenance manifest is missing: {manifest_path}"
            )
        if sha256_file(manifest_path) != build_manifest.get(key):
            raise RuntimeError(f"{key} does not match {manifest_path.name}")
        counts[key] = verify_sha256_manifest(
            manifest_root,
            manifest_path,
            exact_inventory=exact_inventory,
        )
    direct_bindings = (
        (
            "prepared_manifest_binding_sha256",
            snapshot / "prepared/prepared_manifest_binding.txt",
        ),
        (
            "generation_manifest_binding_sha256",
            snapshot / "generated/generation_manifest_binding.txt",
        ),
        (
            "instrumentation_manifest_binding_sha256",
            snapshot / "generated/instrumentation_manifest_binding.txt",
        ),
        (
            "tapenade_install_receipt_sha256",
            snapshot / "tapenade/.zc_tapenade_install_receipt.txt",
        ),
        (
            "tapenade_tree_manifest_sha256",
            snapshot / "tapenade/.zc_tapenade_tree_manifest.sha256",
        ),
    )
    for key, path in direct_bindings:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required provenance binding is missing: {path}")
        if sha256_file(path) != build_manifest.get(key):
            raise RuntimeError(f"{key} does not match {path.name}")
        counts[key] = 1

    prepared_manifest = snapshot / "prepared/prepared_source_manifest.sha256"
    prepared_binding_path = snapshot / "prepared/prepared_manifest_binding.txt"
    generated_pre_manifest = (
        snapshot / "generated/generated_source_manifest.pre_instrument.sha256"
    )
    generated_manifest = snapshot / "generated/generated_source_manifest.sha256"
    generation_provenance = snapshot / "generated/generation_provenance.txt"
    generation_binding_path = snapshot / "generated/generation_manifest_binding.txt"
    instrumentation_provenance = snapshot / "generated/instrumentation_provenance.txt"
    instrumentation_binding_path = (
        snapshot / "generated/instrumentation_manifest_binding.txt"
    )
    prepared_binding = parse_key_values(prepared_binding_path)
    generation_binding = parse_key_values(generation_binding_path)
    instrumentation_binding = parse_key_values(instrumentation_binding_path)
    instrumentation = parse_key_values(instrumentation_provenance)
    generation = parse_key_values(generation_provenance)

    expected_bindings = (
        (
            prepared_binding,
            "prepared_source_manifest_sha256",
            prepared_manifest,
        ),
        (
            generation_binding,
            "generated_pre_manifest_sha256",
            generated_pre_manifest,
        ),
        (
            generation_binding,
            "generation_provenance_sha256",
            generation_provenance,
        ),
        (
            generation_binding,
            "prepared_manifest_binding_sha256",
            prepared_binding_path,
        ),
        (
            instrumentation_binding,
            "generated_source_manifest_sha256",
            generated_manifest,
        ),
        (
            instrumentation_binding,
            "instrumentation_provenance_sha256",
            instrumentation_provenance,
        ),
        (
            instrumentation_binding,
            "generation_manifest_binding_sha256",
            generation_binding_path,
        ),
        (
            instrumentation_binding,
            "generated_pre_manifest_sha256",
            generated_pre_manifest,
        ),
    )
    for binding, key, path in expected_bindings:
        if binding.get(key) != sha256_file(path):
            raise RuntimeError(f"cross-stage provenance mismatch for {key}")
    if instrumentation.get("generation_manifest_binding_sha256") != sha256_file(
        generation_binding_path
    ):
        raise RuntimeError("instrumentation provenance does not bind generation")
    if generation.get("prepared_manifest_binding_sha256") != sha256_file(
        prepared_binding_path
    ):
        raise RuntimeError("generation provenance does not bind preparation")
    recipe = snapshot / "manifests/compile_recipe.txt"
    if sha256_file(recipe) != build_manifest.get("compile_recipe_sha256"):
        raise RuntimeError("compile_recipe_sha256 does not match compile_recipe.txt")
    return build_manifest, counts


def relative_manifest(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)
    }


def expected_path_bytes() -> int:
    states = N_STEPS + 1
    return (
        7 * 4
        + N_REAL * states * 4
        + N_COMPLEX * states * 8
        + N_DOUBLE * states * 8
        + N_INTEGER * states * 4
        + N_TIME * states * 4
        + TAPE_WIDTH * N_STEPS * 4
    )


def read_float32_stream(path: Path, expected_count: int) -> array.array[float]:
    if path.stat().st_size != expected_count * FLOAT32_BYTES:
        raise ValueError(
            f"{path.name} has {path.stat().st_size} bytes; expected "
            f"{expected_count * FLOAT32_BYTES}"
        )
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, expected_count)
    if sys.byteorder != "little":
        raise RuntimeError("the authentic ZC binary contract is little-endian")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path.name} contains a nonfinite float32 value")
    return values


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )


def validate_runtime_source(source: Path) -> None:
    missing = [name for name in REQUIRED_RUNTIME_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "checkpoint directory is missing: " + ", ".join(sorted(missing))
        )
    if not (source / "Data").is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing Data/: {source}")
    state_size = (source / "kernel_initial_state.bin").stat().st_size
    if state_size != STATE_BYTES:
        raise ValueError(
            f"kernel_initial_state.bin has {state_size} bytes; expected {STATE_BYTES}"
        )
    history_size = (source / "zeq9fsu.hst").stat().st_size
    if history_size != HISTORY_BYTES:
        raise ValueError(
            f"zeq9fsu.hst has {history_size} bytes; expected {HISTORY_BYTES}"
        )


def stage_runtime(
    source: Path,
    destination: Path,
    *,
    expected_sha256: dict[str, str],
) -> dict[str, str]:
    validate_runtime_source(source)
    destination.mkdir(parents=True)
    copied: list[Path] = []
    for name in REQUIRED_RUNTIME_FILES:
        target = destination / name
        shutil.copy2(source / name, target)
        copied.append(target)
    shutil.copytree(source / "Data", destination / "Data")
    copied.extend(path for path in (destination / "Data").rglob("*") if path.is_file())
    (destination / "EOF_data").mkdir()
    staged = relative_manifest(destination, copied)
    if staged != expected_sha256:
        missing = sorted(set(expected_sha256) - set(staged))
        extra = sorted(set(staged) - set(expected_sha256))
        changed = sorted(
            name
            for name in set(staged) & set(expected_sha256)
            if staged[name] != expected_sha256[name]
        )
        raise RuntimeError(
            "staged runtime differs from the verified checkpoint; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return staged


def verify_staged_runtime(
    root: Path,
    expected_sha256: dict[str, str],
) -> dict[str, str]:
    """Reverify every producer-bound runtime input after derivative execution."""

    observed: dict[str, str] = {}
    for name, expected in expected_sha256.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe staged-runtime manifest path: {name!r}")
        path = root
        for part in relative.parts:
            path /= part
            if path.is_symlink():
                raise RuntimeError(f"staged runtime path became a symlink: {name}")
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"staged runtime input is missing or unsafe: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"staged runtime input changed during execution: {name}")
        observed[name] = actual
    return observed


def stage_verified_executable(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Copy one executable and prove the executed bytes match the build record."""

    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"required executable is missing: {source}")
    if sha256_file(source) != expected_sha256:
        raise RuntimeError(f"build hash does not match {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or sha256_file(destination) != expected_sha256:
        raise RuntimeError(f"staged executable differs from {source.name}")
    destination.chmod(destination.stat().st_mode | 0o100)
    return destination


def guarded_prepare_output(
    output: Path,
    *,
    overwrite: bool,
    protected_paths: tuple[Path, ...],
    project_root: Path,
) -> Path:
    raw = Path(os.path.abspath(output.expanduser()))
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
    dangerous = {Path("/").resolve(), Path.home().resolve(), project}
    if project not in resolved.parents:
        raise ValueError(f"output path must resolve inside the project root: {project}")
    if resolved in dangerous or any(resolved == item for item in protected):
        raise ValueError(f"refusing unsafe output directory: {resolved}")
    if any(
        resolved in item.parents or (item.is_dir() and item in resolved.parents)
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
                "refusing to delete a nonempty directory: invalid runner marker; "
                f"{resolved}"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / OUTPUT_MARKER).write_text(OUTPUT_MARKER_CONTENT, encoding="utf-8")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="toolchain build containing coupled_compiled/",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="validated runtime directory containing kernel_initial_state.bin",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--terminal-seed",
        type=Path,
        help="optional raw stream of exactly 59,148 little-endian float32 weights",
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        help=(
            "optional approved producer manifest; known official checkpoint "
            "fingerprints are matched automatically"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    # Preserve the requested directory entry until the overwrite guard has
    # rejected symbolic links.  Resolving it here would hide a link and could
    # redirect the later deletion to an unrelated directory.
    output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    seed_source = args.terminal_seed.resolve() if args.terminal_seed else None
    producer_manifest_source = (
        args.runtime_manifest.resolve() if args.runtime_manifest else None
    )
    toolchain_dir = Path(__file__).resolve().parents[1]
    project_root = toolchain_dir.parents[1]
    runtime_contract_path = toolchain_dir / "config/supported_runtime_config.json"
    protected_paths = [build_dir, checkpoint_dir]
    if seed_source is not None:
        protected_paths.append(seed_source)
    if producer_manifest_source is not None:
        protected_paths.append(producer_manifest_source)
    validate_runtime_source(checkpoint_dir)
    seed_sha256: str | None = None
    if seed_source is not None:
        read_float32_stream(seed_source, N_REAL)
        seed_sha256 = sha256_file(seed_source)
    runtime_verification = verify_runtime_contract(
        checkpoint_dir, runtime_contract_path, producer_manifest_source
    )

    compiled = build_dir / "coupled_compiled"
    path_executable = compiled / "zc_adjoint_make_path"
    adjoint_executable = compiled / "zc_nino3_adjoint"
    build_manifest_path = compiled / "build_manifest.txt"
    for required in (path_executable, adjoint_executable, build_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"required build artifact is missing: {required}")
    build_manifest, verified_manifest_entries = verify_build_provenance(build_dir)
    build_manifest_sha256 = sha256_file(build_manifest_path)
    runner_path = Path(__file__).resolve()
    bound_runner = (
        compiled / "build_input_snapshot/toolchain/scripts/run_coupled_case.py"
    )
    if not bound_runner.is_file():
        raise FileNotFoundError(f"build-bound case runner is missing: {bound_runner}")
    runner_sha256 = sha256_file(runner_path)
    bound_runner_sha256 = sha256_file(bound_runner)
    if runner_sha256 != bound_runner_sha256:
        raise RuntimeError(
            "the executing case runner differs from the runner bound into "
            "the compiled build snapshot"
        )
    executable_hashes: dict[str, str] = {}
    for key, executable in (
        ("path_executable_sha256", path_executable),
        ("adjoint_executable_sha256", adjoint_executable),
    ):
        actual = sha256_file(executable)
        if build_manifest.get(key) != actual:
            raise RuntimeError(f"{key} does not match {executable.name}")
        executable_hashes[key] = actual
    # Delay the only potentially destructive operation until every declared
    # input and build artifact has passed its read-only validation.
    output_dir = guarded_prepare_output(
        output_dir,
        overwrite=args.overwrite,
        protected_paths=tuple(protected_paths),
        project_root=project_root,
    )

    work_dir = output_dir / "runtime"
    runtime_manifest = stage_runtime(
        checkpoint_dir,
        work_dir,
        expected_sha256=runtime_verification["runtime_source_sha256"],
    )
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    executable_dir = output_dir / "executables"
    staged_path_executable = stage_verified_executable(
        path_executable,
        executable_dir / path_executable.name,
        expected_sha256=executable_hashes["path_executable_sha256"],
    )
    staged_adjoint_executable = stage_verified_executable(
        adjoint_executable,
        executable_dir / adjoint_executable.name,
        expected_sha256=executable_hashes["adjoint_executable_sha256"],
    )
    path_file = artifacts / "zc_31step_path.bin"
    prefix = artifacts / ("zc_custom" if seed_source else "zc_nino3")

    seed_file: Path | None = None
    if seed_source is not None:
        seed_file = artifacts / "terminal_seed.bin"
        shutil.copy2(seed_source, seed_file)
        if sha256_file(seed_file) != seed_sha256:
            raise RuntimeError("staged terminal seed differs from the verified input")

    run_logged(
        [
            str(staged_path_executable),
            str(work_dir / "kernel_initial_state.bin"),
            str(path_file),
        ],
        work_dir,
        output_dir / "path_stdout.log",
    )
    adjoint_command = [
        str(staged_adjoint_executable),
        str(path_file),
        str(prefix),
    ]
    if seed_file is not None:
        adjoint_command.append(str(seed_file))
    run_logged(
        adjoint_command,
        work_dir,
        output_dir / "adjoint_stdout.log",
    )

    path_metadata_path = Path(f"{path_file}.txt")
    gradient_path = Path(f"{prefix}_gradient.bin")
    gradient_metadata_path = Path(f"{prefix}_gradient.txt")
    replay_path = Path(f"{prefix}_forward_replay.csv")
    for required in (
        path_file,
        path_metadata_path,
        gradient_path,
        gradient_metadata_path,
        replay_path,
    ):
        if not required.is_file():
            raise RuntimeError(f"expected output is missing: {required}")
    if path_file.stat().st_size != expected_path_bytes():
        raise RuntimeError(
            f"path has {path_file.stat().st_size} bytes; expected "
            f"{expected_path_bytes()}"
        )
    gradient = read_float32_stream(gradient_path, N_REAL)
    gradient_norm = math.sqrt(math.fsum(float(value) ** 2 for value in gradient))
    gradient_max = max(abs(value) for value in gradient)
    gradient_nonzero = sum(value != 0.0 for value in gradient)

    path_metadata = parse_key_values(path_metadata_path)
    gradient_metadata = parse_key_values(gradient_metadata_path)
    if path_metadata.get("transitions") != str(N_STEPS):
        raise RuntimeError("path metadata has the wrong transition count")
    if gradient_metadata.get("transitions") != str(N_STEPS):
        raise RuntimeError("gradient metadata has the wrong transition count")
    if gradient_metadata.get("real_state_length") != str(N_REAL):
        raise RuntimeError("gradient metadata has the wrong state length")
    with replay_path.open(encoding="utf-8", newline="") as stream:
        replay_rows = list(csv.DictReader(stream))
    if len(replay_rows) != N_STEPS:
        raise RuntimeError(f"forward replay contains {len(replay_rows)} steps")

    # A case is publishable only when every mutable source still matches the
    # bytes validated before staging.  The executables actually run above are
    # private copies, while these checks detect concurrent mutation of the
    # producer inputs and prevent a misleading provenance report.
    final_runtime_verification = verify_runtime_contract(
        checkpoint_dir,
        runtime_contract_path,
        producer_manifest_source,
    )
    if final_runtime_verification != runtime_verification:
        raise RuntimeError("checkpoint or runtime contract changed during the run")
    final_staged_runtime_manifest = verify_staged_runtime(work_dir, runtime_manifest)
    if final_staged_runtime_manifest != runtime_manifest:
        raise RuntimeError("staged runtime manifest changed during the run")
    final_build_manifest, final_manifest_entries = verify_build_provenance(build_dir)
    if (
        final_build_manifest != build_manifest
        or final_manifest_entries != verified_manifest_entries
        or sha256_file(build_manifest_path) != build_manifest_sha256
    ):
        raise RuntimeError("compiled build provenance changed during the run")
    for key, executable in (
        ("path_executable_sha256", path_executable),
        ("adjoint_executable_sha256", adjoint_executable),
    ):
        if sha256_file(executable) != executable_hashes[key]:
            raise RuntimeError(f"{executable.name} changed during the run")
    staged_executable_hashes = {
        staged_path_executable.name: sha256_file(staged_path_executable),
        staged_adjoint_executable.name: sha256_file(staged_adjoint_executable),
    }
    if staged_executable_hashes != {
        path_executable.name: executable_hashes["path_executable_sha256"],
        adjoint_executable.name: executable_hashes["adjoint_executable_sha256"],
    }:
        raise RuntimeError("executed staged executable changed during the run")
    if sha256_file(runner_path) != runner_sha256:
        raise RuntimeError("case runner changed during the run")
    if seed_source is not None and sha256_file(seed_source) != seed_sha256:
        raise RuntimeError("terminal seed changed during the run")

    output_files = [
        path_file,
        path_metadata_path,
        gradient_path,
        gradient_metadata_path,
        replay_path,
        output_dir / "path_stdout.log",
        output_dir / "adjoint_stdout.log",
    ]
    if seed_file is not None:
        output_files.append(seed_file)
    report = {
        "schema_version": 1,
        "status": "completed",
        "created_utc": datetime.now(UTC).isoformat(),
        "scientific_contract": {
            "model": "authentic Zebiak-Cane Fortran implementation",
            "state_semantics": "pre-input native checkpoint",
            "transitions": N_STEPS,
            "real_state_length": N_REAL,
            "default_objective": "canonical Nino-3 SST anomaly",
            "canonical_fortran_box": "I=13:18,J=21:31",
            "objective": gradient_metadata.get("objective"),
            "certification_assessed_by_runner": False,
            "configuration_verified": runtime_verification["configuration_verified"],
        },
        "build": {
            **build_manifest,
            "build_manifest_sha256": build_manifest_sha256,
            "verified_source_manifest_entries": verified_manifest_entries,
            "case_runner_sha256": runner_sha256,
            "bound_case_runner_sha256": bound_runner_sha256,
            "executed_staged_executable_sha256": {
                **staged_executable_hashes,
            },
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "byteorder": sys.byteorder,
        },
        "checkpoint_input": {
            "source_directory_name": checkpoint_dir.name,
            "staged_file_sha256": runtime_manifest,
            "runtime_verification": runtime_verification,
        },
        "results": {
            "path_metadata": path_metadata,
            "gradient_metadata": gradient_metadata,
            "gradient_l2_norm_recomputed_float64": gradient_norm,
            "gradient_max_abs_recomputed": gradient_max,
            "gradient_nonzero_count": gradient_nonzero,
            "forward_replay_steps": len(replay_rows),
            "output_sha256": relative_manifest(output_dir, output_files),
        },
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Completed {N_STEPS}-transition ZC adjoint run.")
    print(f"Final Nino-3: {gradient_metadata['final_nino3_celsius']} C")
    print(f"Gradient: {gradient_path}")
    print(f"Gradient SHA-256: {sha256_file(gradient_path)}")
    print(f"Run report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
