#!/usr/bin/env python3
"""Regenerate only the certified native history for the fresh ZC run.

The processed release intentionally omits the large native restart history.
This utility reruns the exact certified production executable while placing the
diagnostic-output window beyond the end of the integration.  It therefore
recreates ``outhst`` without also writing the roughly 12 GB field stream.  The
result is published only after its byte count and SHA-256 hash agree with the
sanitized production provenance distributed with the processed data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import generate_fresh_zc_dataset as fresh_generator  # noqa: E402

SCRIPT_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1
STATUS = "complete_certified_native_history"
RUNTIME_INPUTS = (
    "fc.data",
    "Experiments/Standard/modified_means.namelist_1",
    "Experiments/Standard/scales_EOF.namelist_1",
    "Data/uv.zeb",
    "Data/wem.zeb",
    "Data/rcsstmn.data",
    "Data/rcdivmn.data",
    "Data/rcwindmn.data",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument(
        "--executable",
        type=Path,
        help="Certified production executable (defaults to SOURCE_DIR/zeqfc1).",
    )
    result.add_argument(
        "--production-report",
        type=Path,
        default=Path("data/processed/zc-v3/provenance/production_report.json"),
    )
    result.add_argument(
        "--preflight-report",
        type=Path,
        default=Path("data/processed/zc-v3/provenance/preflight_report.json"),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/zc_generation/zc-v3-native-history"),
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parser().parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(f"Required provenance file is missing: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")


def validate_runtime_inputs(
    source_dir: Path, expected_files: dict[str, str]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_name in RUNTIME_INPUTS:
        path = source_dir / relative_name
        require_file(path, f"ZC runtime input {relative_name}")
        actual = fresh_generator.sha256_file(path)
        expected = expected_files.get(relative_name)
        if actual != expected:
            raise RuntimeError(
                f"ZC runtime input {relative_name} does not match the certified "
                f"production source: expected {expected}, found {actual}"
            )
        hashes[relative_name] = actual
    return hashes


def validate_existing(
    output_dir: Path,
    *,
    expected_history_sha256: str,
    expected_history_size: int,
) -> bool:
    history = output_dir / "outhst"
    report_path = output_dir / "history_report.json"
    if not output_dir.exists():
        return False
    if not history.is_file() or not report_path.is_file():
        raise RuntimeError(
            "Incomplete output already exists; move it aside before retrying: "
            f"{output_dir}"
        )
    report = load_json(report_path)
    if report.get("status") != STATUS:
        raise RuntimeError(f"Existing history report is not complete: {report_path}")
    if history.stat().st_size != expected_history_size:
        raise RuntimeError("Existing native history has the wrong byte count")
    if fresh_generator.sha256_file(history) != expected_history_sha256:
        raise RuntimeError("Existing native history has the wrong SHA-256 hash")
    return True


def ensure_generation_workspace_layout(
    output_dir: Path, *, source_dir: Path, production_report: Path
) -> None:
    """Expose the regenerated history through the established driver layout."""

    local_report = output_dir / "production_report.json"
    if local_report.exists():
        if (
            not local_report.is_file()
            or fresh_generator.sha256_file(local_report)
            != fresh_generator.sha256_file(production_report)
        ):
            raise RuntimeError(
                f"Compatibility production report changed: {local_report}"
            )
    else:
        shutil.copy2(production_report, local_report)

    source_link = output_dir / "source"
    if source_link.exists() or source_link.is_symlink():
        if source_link.resolve() != source_dir:
            raise RuntimeError(f"Compatibility source link changed: {source_link}")
    else:
        source_link.symlink_to(os.path.relpath(source_dir, start=output_dir))

    production_dir = output_dir / "runs" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)
    history_link = production_dir / "outhst"
    history = output_dir / "outhst"
    if history_link.exists() or history_link.is_symlink():
        if history_link.resolve() != history:
            raise RuntimeError(f"Compatibility history link changed: {history_link}")
    else:
        history_link.symlink_to(os.path.relpath(history, start=production_dir))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = resolve_path(args.source_dir)
    executable = resolve_path(args.executable or source_dir / "zeqfc1")
    production_path = resolve_path(args.production_report)
    preflight_path = resolve_path(args.preflight_report)
    output_dir = resolve_path(args.output_dir)

    require_file(executable, "certified ZC executable")
    production = load_json(production_path)
    preflight = load_json(preflight_path)

    expected_executable_hash = preflight["build"]["executable_sha256"]
    executable_hash = fresh_generator.sha256_file(executable)
    if executable_hash != expected_executable_hash:
        raise RuntimeError(
            "Executable does not match the one recorded for the released simulation: "
            f"expected {expected_executable_hash}, found {executable_hash}"
        )
    runtime_input_hashes = validate_runtime_inputs(
        source_dir, preflight["build"]["upstream"]["local_source_files"]
    )

    configuration = production["configuration"]
    checkpoint_count = int(production["checkpoint_count"])
    checkpoint_chunk_bytes = int(production["checkpoint_chunk_bytes"])
    expected_history_size = checkpoint_count * checkpoint_chunk_bytes
    recorded_history_size = int(production["runtime"]["history_size_bytes"])
    if expected_history_size != recorded_history_size:
        raise RuntimeError(
            "Production provenance contains an inconsistent history size"
        )
    expected_history_hash = str(production["history_sha256"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if validate_existing(
        output_dir,
        expected_history_sha256=expected_history_hash,
        expected_history_size=expected_history_size,
    ):
        ensure_generation_workspace_layout(
            output_dir,
            source_dir=source_dir,
            production_report=production_path,
        )
        print(f"Certified native history already exists: {output_dir / 'outhst'}")
        return 0

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    started = time.monotonic()
    try:
        tend = float(configuration["tend_months"])
        # The strict inequality in the Fortran writer means this interval emits
        # no diagnostic records while leaving the integration itself unchanged.
        suppressed_output_time = tend + 1.0
        fresh_generator.prepare_run(
            source_dir,
            executable,
            staging,
            nstart=0,
            tfind=120.5,
            tzero=fresh_generator.TZERO_MONTHS,
            tend=tend,
            ntape=int(configuration["checkpoint_steps"]),
            nrewnd=10,
            nic=4,
            write_start=suppressed_output_time,
            write_end=suppressed_output_time,
        )
        runtime = fresh_generator.run_model(staging)
        field_stream = staging / "fresh_fields.data"
        if field_stream.stat().st_size != 0:
            raise RuntimeError("History-only run unexpectedly wrote diagnostic fields")

        history = staging / "outhst"
        if history.stat().st_size != expected_history_size:
            raise RuntimeError(
                "Regenerated history has the wrong byte count: "
                f"expected {expected_history_size}, found {history.stat().st_size}"
            )
        history_hash = fresh_generator.sha256_file(history)
        if history_hash != expected_history_hash:
            raise RuntimeError(
                "Regenerated history does not match the released simulation: "
                f"expected {expected_history_hash}, found {history_hash}"
            )

        field_stream.unlink()
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "status": STATUS,
            "completed_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "purpose": (
                "Exact native restart history for all-phase training-covariance "
                "capture; "
                "the diagnostic field stream was intentionally suppressed."
            ),
            "source": {
                "source_directory": str(source_dir),
                "executable_file": str(executable),
                "executable_sha256": executable_hash,
                "runtime_input_sha256": runtime_input_hashes,
                "production_report_file": str(production_path),
                "production_report_sha256": fresh_generator.sha256_file(
                    production_path
                ),
                "preflight_report_file": str(preflight_path),
                "preflight_report_sha256": fresh_generator.sha256_file(
                    preflight_path
                ),
            },
            "configuration": configuration,
            "runtime": runtime,
            "history": {
                "file": "outhst",
                "sha256": history_hash,
                "size_bytes": expected_history_size,
                "checkpoint_count": checkpoint_count,
                "checkpoint_chunk_bytes": checkpoint_chunk_bytes,
            },
        }
        write_json(staging / "history_report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    ensure_generation_workspace_layout(
        output_dir,
        source_dir=source_dir,
        production_report=production_path,
    )

    print(f"Published certified native history: {output_dir / 'outhst'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
