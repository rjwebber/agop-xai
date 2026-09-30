#!/usr/bin/env python3
"""Validate and package a raw ZC adjoint gradient for release.

The Tapenade driver writes a headerless stream of 59,148 Fortran ``REAL``
values.  This program checks that stream against the explicit-state manifest,
summarizes the packed gradient, and publishes a JSON report plus an NPZ archive
whose keys are the manifest segment names.  Segment arrays are reshaped with
Fortran indexing: the first logical subscript varies fastest.

The coefficients remain derivatives with respect to *raw packed model-state
coordinates*.  This program neither standardizes them nor turns carried
diagnostic/restart state into independently perturbable scientific controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.io import sha256_file, sha256_json, write_json, write_npz  # noqa: E402

SCRIPT_VERSION = "2.2.1"
REPORT_SCHEMA_VERSION = 3
PRODUCER_METADATA_SCHEMA_VERSION = 2
DEFAULT_MANIFEST = REPOSITORY_ROOT / "adjoint/fortran_kernel/state_manifest.json"
FLOAT_BYTES = np.dtype(np.float32).itemsize
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CERTIFICATION_EVIDENCE_KINDS = {
    "primal_replay",
    "tangent_reverse_dot",
    "independent_finite_difference_or_taylor",
}
BUNDLE_STAGING_MARKER = ".zc_adjoint_gradient_bundle_staging"
BUNDLE_STAGING_MARKER_CONTENT = "managed-zc-adjoint-gradient-bundle-v1\n"


@dataclass(frozen=True)
class ArtifactSnapshot:
    """Identity and digest of bytes read from one stable open file."""

    path: Path
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    data: bytes | None = None


def _snapshot_file(path: Path, *, keep_bytes: bool = False) -> ArtifactSnapshot:
    """Read and hash one regular file through a single descriptor.

    The pre/post ``fstat`` check rejects an in-place mutation during the read.
    An atomic replacement after the descriptor is opened cannot change the
    consumed bytes; commit-time revalidation detects that pathname change.
    """

    resolved = path.expanduser().resolve()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Artifact is not a regular file: {resolved}")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if keep_bytes else None
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"Artifact changed while it was being read: {resolved}")
    return ArtifactSnapshot(
        path=resolved,
        sha256=digest.hexdigest(),
        size=int(after.st_size),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns),
        data=b"".join(chunks) if chunks is not None else None,
    )


def _reverify_artifacts(artifacts: dict[Path, ArtifactSnapshot]) -> None:
    """Require every producer input to retain the exact verified bytes."""

    for path, expected in artifacts.items():
        observed = _snapshot_file(path)
        if observed.sha256 != expected.sha256 or observed.size != expected.size:
            raise RuntimeError(
                "A verified gradient-packaging input changed before commit: "
                f"{path}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gradient",
        type=Path,
        help="Headerless stream written as STATE_R_INB by the Fortran driver.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Explicit-state manifest defining all packed REAL segments.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help=(
            "Output path without an extension. By default, remove .bin from "
            "the gradient path and write adjacent .json and .npz files."
        ),
    )
    parser.add_argument(
        "--byte-order",
        choices=("native", "little", "big"),
        default="little",
        help="Byte order used by the machine that wrote the Fortran stream.",
    )
    parser.add_argument(
        "--producer-metadata",
        type=Path,
        help=(
            "Optional strict provenance document. A producer may declare a "
            "gradient certified only through this document, which must then bind "
            "the gradient itself, objective, path, sources, executable, Tapenade "
            "version, and the three required validation-evidence classes."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_manifest(
    payload: bytes, *, label: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly validate the packed-real portion of manifest bytes."""

    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid state manifest JSON: {label}") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected a JSON object in {label}.")

    try:
        specification = manifest["arrays"]["real32"]
        length = int(specification["length"])
        segments = specification["segments"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Manifest has no valid arrays.real32 contract.") from error
    if length <= 0 or not isinstance(segments, list) or not segments:
        raise ValueError("Manifest real32 length and segments must be nonempty.")

    cursor = 0
    names: set[str] = set()
    reserved_archive_keys = {"packed_gradient", "independent_control_mask"}
    required = {
        "name",
        "start",
        "stop",
        "shape",
        "order",
        "role",
        "activity",
        "independent_control",
    }
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not required <= segment.keys():
            raise ValueError(f"Manifest real32 segment {index} is incomplete.")
        name = segment["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or name in reserved_archive_keys
        ):
            raise ValueError(f"Invalid or duplicate segment name at index {index}.")
        names.add(name)
        start = int(segment["start"])
        stop = int(segment["stop"])
        shape = tuple(int(value) for value in segment["shape"])
        if start != cursor or stop <= start:
            raise ValueError(f"Segment {name} is not contiguous at offset {cursor}.")
        if not shape or any(value <= 0 for value in shape):
            raise ValueError(f"Segment {name} has an invalid shape.")
        if math.prod(shape) != stop - start:
            raise ValueError(f"Segment {name} shape does not match its slice.")
        if segment["order"] != "F":
            raise ValueError(f"Segment {name} does not use Fortran order.")
        if segment["activity"] not in {"active", "diagnostic", "passive"}:
            raise ValueError(f"Segment {name} has an invalid activity class.")
        if not isinstance(segment["independent_control"], bool):
            raise ValueError(f"Segment {name} has an invalid control flag.")
        if segment["activity"] != "active" and segment["independent_control"]:
            raise ValueError(
                f"Non-active segment {name} cannot be an independent control."
            )
        cursor = stop
    if cursor != length:
        raise ValueError(
            f"Manifest segments end at {cursor}, not declared length {length}."
        )
    return manifest, segments


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate a manifest from one stable file snapshot."""

    snapshot = _snapshot_file(path, keep_bytes=True)
    assert snapshot.data is not None
    return _parse_manifest(snapshot.data, label=str(snapshot.path))


def _input_dtype(byte_order: str) -> np.dtype[Any]:
    prefix = {"native": "=", "little": "<", "big": ">"}[byte_order]
    return np.dtype(f"{prefix}f4")


def read_raw_gradient(
    path: Path,
    *,
    expected_length: int,
    byte_order: str,
) -> np.ndarray:
    """Read an exact-length headerless REAL stream from one stable snapshot."""

    snapshot = _snapshot_file(path, keep_bytes=True)
    assert snapshot.data is not None
    return _decode_raw_gradient(
        snapshot.data,
        expected_length=expected_length,
        byte_order=byte_order,
        label=str(snapshot.path),
    )


def _decode_raw_gradient(
    payload: bytes,
    *,
    expected_length: int,
    byte_order: str,
    label: str,
) -> np.ndarray:
    """Decode already verified gradient bytes into native float32."""

    expected_bytes = expected_length * FLOAT_BYTES
    actual_bytes = len(payload)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Gradient file has {actual_bytes:,} bytes; expected exactly "
            f"{expected_bytes:,} bytes ({expected_length:,} float32 values)."
        )
    values = np.frombuffer(
        payload, dtype=_input_dtype(byte_order), count=expected_length
    ).copy()
    if values.size != expected_length:
        raise OSError(f"Short read from {label}: obtained {values.size} values.")
    values = np.asarray(values, dtype=np.float32)
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        preview = ", ".join(str(int(index)) for index in bad[:8])
        suffix = " ..." if bad.size > 8 else ""
        raise ValueError(
            f"Gradient contains {bad.size} nonfinite coefficient(s) at "
            f"zero-based packed indices {preview}{suffix}."
        )
    return values


def _canonical_float32_hash(values: np.ndarray, *, order: str) -> str:
    """Hash shape plus canonical little-endian float32 bytes in one order."""

    array = np.asarray(values, dtype="<f4")
    payload = np.ravel(array, order=order).tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(b"zc-adjoint-float32-array-v1\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(order.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _canonical_bool_hash(values: np.ndarray) -> str:
    """Hash a boolean mask with an explicit schema and byte representation."""

    array = np.asarray(values, dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(b"zc-adjoint-bool-mask-v1\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(np.asarray(array, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Producer metadata field {field} must be a nonempty string.")
    return value.strip()


def _sha256(value: Any, *, field: str) -> str:
    digest = _nonempty_string(value, field=field).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"Producer metadata field {field} is not a SHA-256 digest.")
    return digest


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Producer metadata field {field} must be a JSON object.")
    return value


def _verified_artifact(
    value: Any,
    *,
    field: str,
    metadata_dir: Path,
    verified_paths: set[Path] | None = None,
    verified_artifacts: dict[Path, ArtifactSnapshot] | None = None,
) -> dict[str, Any]:
    """Verify and sanitize one path/SHA artifact reference for publication."""

    reference = _mapping(value, field=field)
    raw_path = _nonempty_string(reference.get("path"), field=f"{field}.path")
    expected_sha = _sha256(reference.get("sha256"), field=f"{field}.sha256")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = metadata_dir / path
    path = path.resolve()
    snapshot = _snapshot_file(path)
    if snapshot.sha256 != expected_sha:
        raise ValueError(
            f"Producer metadata hash mismatch for {field}: expected "
            f"{expected_sha}, observed {snapshot.sha256}."
        )
    if verified_paths is not None:
        verified_paths.add(path)
    if verified_artifacts is not None:
        verified_artifacts[path] = snapshot
    return {
        "filename": path.name,
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
    }


def load_producer_metadata(
    path: Path,
    *,
    verified_artifact_paths: set[Path] | None = None,
    verified_artifacts: dict[Path, ArtifactSnapshot] | None = None,
) -> dict[str, Any]:
    """Validate, verify, and sanitize optional producer certification metadata."""

    path = path.expanduser().resolve()
    metadata_snapshot = _snapshot_file(path, keep_bytes=True)
    if verified_artifact_paths is not None:
        verified_artifact_paths.add(path)
    if verified_artifacts is not None:
        verified_artifacts[path] = metadata_snapshot
    assert metadata_snapshot.data is not None
    try:
        document = json.loads(metadata_snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid producer metadata JSON: {path}") from error
    metadata = _mapping(document, field="root")
    if metadata.get("schema_version") != PRODUCER_METADATA_SCHEMA_VERSION:
        raise ValueError(
            "Producer metadata schema_version must equal "
            f"{PRODUCER_METADATA_SCHEMA_VERSION}."
        )

    objective = _mapping(metadata.get("objective"), field="objective")
    clean_objective = {
        key: _nonempty_string(objective.get(key), field=f"objective.{key}")
        for key in ("name", "definition", "units", "terminal_seed")
    }
    transitions = metadata.get("transitions")
    if isinstance(transitions, bool) or not isinstance(transitions, int):
        raise ValueError("Producer metadata field transitions must be an integer.")
    if transitions <= 0:
        raise ValueError("Producer metadata field transitions must be positive.")

    checkpoint = _mapping(
        metadata.get("initial_checkpoint"), field="initial_checkpoint"
    )
    clean_checkpoint = {
        "label": _nonempty_string(
            checkpoint.get("label"), field="initial_checkpoint.label"
        ),
        "artifact": _verified_artifact(
            checkpoint,
            field="initial_checkpoint",
            metadata_dir=path.parent,
            verified_paths=verified_artifact_paths,
            verified_artifacts=verified_artifacts,
        ),
    }
    clean_path = _verified_artifact(
        metadata.get("certified_path"),
        field="certified_path",
        metadata_dir=path.parent,
        verified_paths=verified_artifact_paths,
        verified_artifacts=verified_artifacts,
    )
    clean_gradient = _verified_artifact(
        metadata.get("gradient"),
        field="gradient",
        metadata_dir=path.parent,
        verified_paths=verified_artifact_paths,
        verified_artifacts=verified_artifacts,
    )

    source_manifests = _mapping(
        metadata.get("source_manifests"), field="source_manifests"
    )
    clean_source_manifests = {
        kind: _verified_artifact(
            source_manifests.get(kind),
            field=f"source_manifests.{kind}",
            metadata_dir=path.parent,
            verified_paths=verified_artifact_paths,
            verified_artifacts=verified_artifacts,
        )
        for kind in ("prepared", "tangent", "reverse")
    }
    clean_executable = _verified_artifact(
        metadata.get("reverse_executable"),
        field="reverse_executable",
        metadata_dir=path.parent,
        verified_paths=verified_artifact_paths,
        verified_artifacts=verified_artifacts,
    )

    tapenade = _mapping(metadata.get("tapenade"), field="tapenade")
    clean_tapenade = {
        "version": _nonempty_string(
            tapenade.get("version"), field="tapenade.version"
        ),
        "revision": _nonempty_string(
            tapenade.get("revision"), field="tapenade.revision"
        ),
        "archive_sha256": _sha256(
            tapenade.get("archive_sha256"), field="tapenade.archive_sha256"
        ),
    }

    certification = _mapping(metadata.get("certification"), field="certification")
    status = certification.get("status")
    if status not in {"uncertified", "provisional", "certified"}:
        raise ValueError(
            "Producer metadata certification.status must be uncertified, "
            "provisional, or certified."
        )
    scope = _nonempty_string(certification.get("scope"), field="certification.scope")
    raw_evidence = certification.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("Producer metadata certification.evidence must be a list.")
    clean_evidence: list[dict[str, Any]] = []
    evidence_kinds: set[str] = set()
    for index, item in enumerate(raw_evidence):
        entry = _mapping(item, field=f"certification.evidence[{index}]")
        kind = _nonempty_string(
            entry.get("kind"), field=f"certification.evidence[{index}].kind"
        )
        evidence_kinds.add(kind)
        clean_evidence.append(
            {
                "kind": kind,
                "artifact": _verified_artifact(
                    entry,
                    field=f"certification.evidence[{index}]",
                    metadata_dir=path.parent,
                    verified_paths=verified_artifact_paths,
                    verified_artifacts=verified_artifacts,
                ),
            }
        )
    missing = CERTIFICATION_EVIDENCE_KINDS - evidence_kinds
    if status == "certified" and missing:
        listing = ", ".join(sorted(missing))
        raise ValueError(
            "Certified producer metadata is missing required validation evidence: "
            f"{listing}."
        )

    sanitized = {
        "schema_version": PRODUCER_METADATA_SCHEMA_VERSION,
        "metadata_file": {
            "filename": path.name,
            "sha256": metadata_snapshot.sha256,
            "canonical_json_sha256": sha256_json(metadata),
        },
        "objective": clean_objective,
        "transitions": transitions,
        "initial_checkpoint": clean_checkpoint,
        "certified_path": clean_path,
        "gradient": clean_gradient,
        "source_manifests": clean_source_manifests,
        "reverse_executable": clean_executable,
        "tapenade": clean_tapenade,
        "certification": {
            "producer_declared_status": status,
            "scope": scope,
            "evidence": clean_evidence,
            "required_evidence_kinds_for_certified_status": sorted(
                CERTIFICATION_EVIDENCE_KINDS
            ),
        },
    }
    return sanitized


def _norm_summary(values: np.ndarray) -> dict[str, Any]:
    work = np.asarray(values, dtype=np.float64)
    absolute = np.abs(work)
    squared_l2 = float(np.dot(work, work))
    nonzero_count = int(np.count_nonzero(work))
    if work.size and nonzero_count:
        max_index = int(np.argmax(absolute))
        max_abs = float(absolute[max_index])
        max_value: float | None = float(work[max_index])
    else:
        max_index = None
        max_abs = 0.0
        max_value = None
    return {
        "coefficient_count": int(work.size),
        "nonzero_count": nonzero_count,
        "zero_count": int(work.size - nonzero_count),
        "l1_norm": float(np.sum(absolute, dtype=np.float64)),
        "l2_norm": math.sqrt(squared_l2),
        "squared_l2_norm": squared_l2,
        "max_abs": max_abs,
        "max_abs_flat_index": max_index,
        "coefficient_at_max_abs": max_value,
    }


def _find_segment(
    packed_index: int | None,
    segments: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int | None]:
    if packed_index is None:
        return None, None
    for segment in segments:
        start = int(segment["start"])
        stop = int(segment["stop"])
        if start <= packed_index < stop:
            return segment, packed_index - start
    raise AssertionError(f"Packed index {packed_index} is outside the manifest.")


def build_package(
    gradient: np.ndarray,
    *,
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    input_path: Path,
    manifest_path: Path,
    input_snapshot: ArtifactSnapshot,
    manifest_snapshot: ArtifactSnapshot,
    byte_order: str,
    producer_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the machine-readable report and named-array NPZ payload."""

    packed = np.asarray(gradient, dtype="<f4")
    expected_length = int(manifest["arrays"]["real32"]["length"])
    if packed.shape != (expected_length,):
        raise ValueError(
            f"Expected gradient shape ({expected_length},), got {packed.shape}."
        )
    if not np.isfinite(packed).all():
        raise ValueError("Gradient must be finite before packaging.")

    arrays: dict[str, np.ndarray] = {"packed_gradient": packed}
    independent_mask = np.zeros(expected_length, dtype=np.bool_)
    segment_reports: list[dict[str, Any]] = []
    global_summary = _norm_summary(packed)
    global_squared = float(global_summary["squared_l2_norm"])

    for segment in segments:
        name = str(segment["name"])
        start = int(segment["start"])
        stop = int(segment["stop"])
        shape = tuple(int(value) for value in segment["shape"])
        shaped = np.asfortranarray(packed[start:stop].reshape(shape, order="F"))
        arrays[name] = shaped
        if segment["independent_control"]:
            independent_mask[start:stop] = True

        summary = _norm_summary(packed[start:stop])
        local_flat = summary.pop("max_abs_flat_index")
        local_python = (
            [int(value) for value in np.unravel_index(local_flat, shape, order="F")]
            if local_flat is not None
            else None
        )
        segment_reports.append(
            {
                "name": name,
                "python_slice": [start, stop],
                "fortran_slice": [start + 1, stop],
                "shape": list(shape),
                "order": "F",
                "role": segment["role"],
                "activity": segment["activity"],
                "independent_control": segment["independent_control"],
                **summary,
                "squared_l2_fraction_of_packed_gradient": (
                    float(summary["squared_l2_norm"] / global_squared)
                    if global_squared > 0.0
                    else None
                ),
                "max_abs_local_python_index": local_python,
                "max_abs_local_fortran_index": (
                    [value + 1 for value in local_python]
                    if local_python is not None
                    else None
                ),
                "max_abs_global_python_index": (
                    start + local_flat if local_flat is not None else None
                ),
                "max_abs_global_fortran_index": (
                    start + local_flat + 1 if local_flat is not None else None
                ),
                "canonical_little_endian_fortran_order_sha256": (
                    _canonical_float32_hash(shaped, order="F")
                ),
            }
        )

    arrays["independent_control_mask"] = independent_mask
    activity_summaries: dict[str, Any] = {}
    for activity in ("active", "diagnostic", "passive"):
        mask = np.zeros(expected_length, dtype=np.bool_)
        for segment in segments:
            if segment["activity"] == activity:
                mask[int(segment["start"]) : int(segment["stop"])] = True
        activity_summaries[activity] = _norm_summary(packed[mask])

    control_summary = _norm_summary(packed[independent_mask])
    control_summary.pop("max_abs_flat_index")
    excluded_control_summary = _norm_summary(packed[~independent_mask])
    excluded_control_summary.pop("max_abs_flat_index")
    global_max_segment, local_flat = _find_segment(
        global_summary["max_abs_flat_index"], segments
    )
    if global_max_segment is None:
        global_max_location = None
    else:
        shape = tuple(int(value) for value in global_max_segment["shape"])
        local_python = [
            int(value) for value in np.unravel_index(local_flat, shape, order="F")
        ]
        packed_index = int(global_summary["max_abs_flat_index"])
        global_max_location = {
            "segment": global_max_segment["name"],
            "global_python_index": packed_index,
            "global_fortran_index": packed_index + 1,
            "local_python_index": local_python,
            "local_fortran_index": [value + 1 for value in local_python],
        }

    global_summary.pop("max_abs_flat_index")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "format_validation": {
            "status": "passed",
            "scope": (
                "Exact byte count, declared byte order, finite coefficients, "
                "and state-manifest layout only. This is not validation of the "
                "adjoint calculation."
            ),
        },
        "adjoint_certification": {
            "assessed_by_packager": False,
            "producer_declared_status": (
                producer_metadata["certification"]["producer_declared_status"]
                if producer_metadata is not None
                else "not_provided"
            ),
            "warning": (
                "The packager does not certify tangent/reverse correctness. Any "
                "status is a producer declaration bound to separately hashed "
                "validation evidence."
            ),
        },
        "input": {
            "filename": input_path.name,
            "byte_order_requested": byte_order,
            "resolved_numpy_dtype": _input_dtype(byte_order).str,
            "bytes": input_snapshot.size,
            "sha256": input_snapshot.sha256,
        },
        "state_manifest": {
            "filename": manifest_path.name,
            "schema_version": manifest.get("schema_version"),
            "sha256": manifest_snapshot.sha256,
            "supported_configuration": manifest.get("supported_configuration"),
        },
        "coordinate_semantics": {
            "coordinate_system": "raw packed Fortran model-state coordinates",
            "rescaled_or_standardized_by_this_program": False,
            "gradient_statement": (
                "Each coefficient is the derivative of the producer's scalar "
                "objective with respect to one raw packed REAL state coordinate."
            ),
            "norm_warning": (
                "The packed state mixes physical variables and units; raw norms "
                "must not be interpreted as standardized variable importance."
            ),
            "control_warning": (
                "Only segments with independent_control=true belong to the "
                "manifest's conservative direct-control mask. Active carried "
                "state, diagnostic, and passive segments marked false are not "
                "independent scientific intervention controls."
            ),
        },
        "packed_gradient": {
            **global_summary,
            "max_abs_location": global_max_location,
            "canonical_little_endian_packed_sha256": (
                _canonical_float32_hash(packed, order="C")
            ),
        },
        "independent_control_mask": {
            "coefficient_count": int(independent_mask.size),
            "true_count": int(np.count_nonzero(independent_mask)),
            "false_count": int(
                independent_mask.size - np.count_nonzero(independent_mask)
            ),
            "canonical_bool_mask_sha256": _canonical_bool_hash(independent_mask),
        },
        "independent_control_gradient_summary": {
            **control_summary,
            "excluded_coordinate_summary": excluded_control_summary,
        },
        "by_activity": activity_summaries,
        "segments": segment_reports,
    }
    if producer_metadata is not None:
        report["producer_provenance"] = producer_metadata
    return report, arrays


def _same_existing_file(left: Path, right: Path) -> bool:
    """Return whether two existing paths name the same inode."""

    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return False


def _validate_output_bundle(
    report_path: Path,
    archive_path: Path,
    *,
    protected_inputs: set[Path],
    overwrite: bool,
) -> None:
    """Reject output aliases to inputs or to the other bundle member."""

    outputs = (report_path, archive_path)
    if report_path == archive_path or _same_existing_file(report_path, archive_path):
        raise ValueError("JSON and NPZ output paths collide with each other.")
    for output in outputs:
        if output.is_symlink():
            raise ValueError(f"Output path may not be a symbolic link: {output}")
        if output.exists() and not output.is_file():
            raise ValueError(f"Output path exists and is not a file: {output}")
        resolved_output = output.resolve()
        for protected in protected_inputs:
            if resolved_output == protected or _same_existing_file(output, protected):
                raise ValueError(
                    "Output path collides with a gradient-packaging input: "
                    f"{output}"
                )
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            listing = "\n  ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output already exists:\n  {listing}\nUse --overwrite to replace it."
            )


def _path_matches_snapshot(path: Path, snapshot: ArtifactSnapshot) -> bool:
    """Authenticate a transaction path without following symbolic links."""

    try:
        observed = path.lstat()
    except FileNotFoundError:
        return False
    metadata_matches = bool(
        stat.S_ISREG(observed.st_mode)
        and not path.is_symlink()
        and observed.st_dev == snapshot.device
        and observed.st_ino == snapshot.inode
        and observed.st_size == snapshot.size
        and observed.st_mtime_ns == snapshot.mtime_ns
    )
    return metadata_matches and sha256_file(path) == snapshot.sha256


def _path_matches_optional_snapshot(
    path: Path,
    snapshot: ArtifactSnapshot | None,
) -> bool:
    if snapshot is None:
        return not path.exists() and not path.is_symlink()
    return _path_matches_snapshot(path, snapshot)


class BundleRollbackError(RuntimeError):
    """A bundle rollback needs manual recovery from its preserved staging tree."""


def _remove_bundle_staging(
    path: Path,
    *,
    device: int,
    inode: int,
) -> None:
    """Remove only the exact private bundle directory created by this process."""

    observed = path.lstat()
    marker = path / BUNDLE_STAGING_MARKER
    if (
        not stat.S_ISDIR(observed.st_mode)
        or path.is_symlink()
        or observed.st_dev != device
        or observed.st_ino != inode
        or marker.is_symlink()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8") != BUNDLE_STAGING_MARKER_CONTENT
    ):
        raise BundleRollbackError(
            f"refusing to remove replaced or unauthenticated staging: {path}"
        )
    shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_output_bundle(
    *,
    staged_report: Path,
    staged_archive: Path,
    report_path: Path,
    archive_path: Path,
    expected_outputs: dict[Path, ArtifactSnapshot | None],
    commit_check: Callable[[], None],
    overwrite: bool,
) -> None:
    """Publish two staged files with authenticated signal-safe rollback."""

    # The archive is installed first and the JSON commit record last. Consumers
    # must verify report["archive"]["sha256"] before accepting the pair.
    pairs = ((staged_archive, archive_path), (staged_report, report_path))
    entries: list[dict[str, Any]] = []
    for index, (staged, destination) in enumerate(pairs):
        staged_snapshot = _snapshot_file(staged)
        if destination.is_symlink():
            raise ValueError(
                f"Output became a symbolic link before publication: {destination}"
            )
        original_snapshot = expected_outputs[destination]
        if not _path_matches_optional_snapshot(destination, original_snapshot):
            raise RuntimeError(
                f"Output changed concurrently before publication: {destination}"
            )
        if original_snapshot is not None and not overwrite:
            raise FileExistsError(
                f"Output appeared during publication: {destination}"
            )
        entries.append(
            {
                "staged": staged,
                "destination": destination,
                "staged_snapshot": staged_snapshot,
                "original_snapshot": original_snapshot,
                # Backups live inside the private staging directory. Recording
                # the intended name before a move closes the signal gap between
                # rename success and Python transaction-state bookkeeping.
                "backup": staged.parent / f"prior-{index}-{destination.name}",
            }
        )
    try:
        for entry in entries:
            destination = entry["destination"]
            backup = entry["backup"]
            original_snapshot = entry["original_snapshot"]
            if original_snapshot is None:
                if destination.exists() or destination.is_symlink():
                    raise RuntimeError(
                        "Output changed concurrently before publication: "
                        f"{destination}"
                    )
                continue
            if not _path_matches_snapshot(destination, original_snapshot):
                raise RuntimeError(
                    "Existing output changed concurrently before backup: "
                    f"{destination}"
                )
            os.replace(destination, backup)
            if not _path_matches_snapshot(backup, original_snapshot):
                raise RuntimeError(
                    "Backed-up output identity changed during publication: "
                    f"{destination}"
                )

        for entry in entries:
            staged = entry["staged"]
            destination = entry["destination"]
            staged_snapshot = entry["staged_snapshot"]
            if destination.exists() or destination.is_symlink():
                raise RuntimeError(
                    "Output appeared after backup and before publication: "
                    f"{destination}"
                )
            if not _path_matches_snapshot(staged, staged_snapshot):
                raise RuntimeError(
                    f"Staged bundle member changed before publication: {staged}"
                )
            if overwrite:
                os.replace(staged, destination)
            else:
                os.link(staged, destination)
            if not _path_matches_snapshot(destination, staged_snapshot):
                raise RuntimeError(
                    f"Published bundle member has the wrong identity: {destination}"
                )
        # This executes while all rollback backups remain available. It closes
        # the interval between the pre-publication check and installation of
        # the JSON commit record.
        commit_check()
        _fsync_directory(report_path.parent)
    except BaseException as publication_error:
        rollback_errors: list[BaseException] = []
        for entry in reversed(entries):
            staged = entry["staged"]
            destination = entry["destination"]
            staged_snapshot = entry["staged_snapshot"]
            if _path_matches_snapshot(destination, staged_snapshot):
                try:
                    if staged.exists():
                        # Non-overwrite publication uses a hard link, so the
                        # staged name still authenticates the destination.
                        destination.unlink()
                    else:
                        os.replace(destination, staged)
                except BaseException as error:  # pragma: no cover
                    rollback_errors.append(error)
            elif destination.exists() or destination.is_symlink():
                original_snapshot = entry["original_snapshot"]
                original_is_still_present = bool(
                    original_snapshot is not None
                    and _path_matches_snapshot(destination, original_snapshot)
                    and not entry["backup"].exists()
                )
                if not original_is_still_present:
                    rollback_errors.append(
                        RuntimeError(
                            "refusing to remove an unauthenticated concurrent "
                            f"output: {destination}"
                        )
                    )

        for entry in reversed(entries):
            backup = entry["backup"]
            destination = entry["destination"]
            original_snapshot = entry["original_snapshot"]
            if backup.exists():
                if original_snapshot is None or not _path_matches_snapshot(
                    backup, original_snapshot
                ):
                    rollback_errors.append(
                        RuntimeError(
                            f"refusing unauthenticated rollback backup: {backup}"
                        )
                    )
                    continue
                if destination.exists() or destination.is_symlink():
                    rollback_errors.append(
                        RuntimeError(
                            "cannot restore prior output over an unexpected path: "
                            f"{destination}"
                        )
                    )
                    continue
                try:
                    os.replace(backup, destination)
                except BaseException as error:  # pragma: no cover
                    rollback_errors.append(error)
            elif original_snapshot is not None and not _path_matches_snapshot(
                destination, original_snapshot
            ):
                rollback_errors.append(
                    RuntimeError(f"prior output could not be restored: {destination}")
                )
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise BundleRollbackError(
                "Gradient bundle publication failed and rollback was incomplete: "
                f"{details}. Recovery files remain in {staged_report.parent}"
            ) from publication_error
        _fsync_directory(report_path.parent)
        raise


def _write_output_bundle(
    *,
    report_path: Path,
    archive_path: Path,
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    protected_inputs: set[Path],
    verified_artifacts: dict[Path, ArtifactSnapshot],
    expected_outputs: dict[Path, ArtifactSnapshot | None],
    overwrite: bool,
) -> None:
    """Stage a complete NPZ/JSON pair, revalidate, and publish it together."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{report_path.stem}.bundle-incomplete-",
            dir=report_path.parent,
        )
    )
    (staging_root / BUNDLE_STAGING_MARKER).write_text(
        BUNDLE_STAGING_MARKER_CONTENT,
        encoding="utf-8",
    )
    staging_status = staging_root.stat(follow_symlinks=False)
    staged_report = staging_root / report_path.name
    staged_archive = staging_root / archive_path.name
    preserve_staging = False
    try:
        write_npz(staged_archive, overwrite=False, **arrays)
        report["archive"] = {
            "filename": archive_path.name,
            "sha256": sha256_file(staged_archive),
            "keys": list(arrays),
        }
        write_json(staged_report, report, overwrite=False)
        _validate_output_bundle(
            report_path,
            archive_path,
            protected_inputs=protected_inputs,
            overwrite=overwrite,
        )
        for output_path, expected in expected_outputs.items():
            if not _path_matches_optional_snapshot(output_path, expected):
                raise RuntimeError(
                    f"Output changed while the package was staged: {output_path}"
                )
        _reverify_artifacts(verified_artifacts)
        try:
            _publish_output_bundle(
                staged_report=staged_report,
                staged_archive=staged_archive,
                report_path=report_path,
                archive_path=archive_path,
                expected_outputs=expected_outputs,
                commit_check=lambda: _reverify_artifacts(verified_artifacts),
                overwrite=overwrite,
            )
        except BundleRollbackError:
            preserve_staging = True
            raise
    finally:
        if staging_root.exists() and not preserve_staging:
            try:
                _remove_bundle_staging(
                    staging_root,
                    device=int(staging_status.st_dev),
                    inode=int(staging_status.st_ino),
                )
            except (OSError, BundleRollbackError) as error:
                warnings.warn(str(error), RuntimeWarning, stacklevel=2)


def package_gradient(
    gradient_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_prefix: Path | None = None,
    byte_order: str = "little",
    producer_metadata_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Format-check one binary and publish a transactional JSON/NPZ bundle."""

    gradient_path = gradient_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if byte_order not in {"native", "little", "big"}:
        raise ValueError(f"Unsupported byte order: {byte_order}")
    verified_artifacts: dict[Path, ArtifactSnapshot] = {}
    manifest_snapshot = _snapshot_file(manifest_path, keep_bytes=True)
    verified_artifacts[manifest_path] = manifest_snapshot
    assert manifest_snapshot.data is not None
    manifest, segments = _parse_manifest(
        manifest_snapshot.data, label=str(manifest_path)
    )
    protected_inputs = {gradient_path, manifest_path}
    producer_metadata = None
    if producer_metadata_path is not None:
        producer_metadata = load_producer_metadata(
            producer_metadata_path,
            verified_artifact_paths=protected_inputs,
            verified_artifacts=verified_artifacts,
        )
    gradient_snapshot = _snapshot_file(gradient_path, keep_bytes=True)
    verified_artifacts[gradient_path] = gradient_snapshot
    input_sha256 = gradient_snapshot.sha256
    if (
        producer_metadata is not None
        and producer_metadata["gradient"]["sha256"] != input_sha256
    ):
        raise ValueError(
            "Producer metadata gradient hash does not match the gradient being "
            f"packaged: expected {producer_metadata['gradient']['sha256']}, "
            f"observed {input_sha256}."
        )
    length = int(manifest["arrays"]["real32"]["length"])
    assert gradient_snapshot.data is not None
    gradient = _decode_raw_gradient(
        gradient_snapshot.data,
        expected_length=length,
        byte_order=byte_order,
        label=str(gradient_path),
    )
    if output_prefix is None:
        prefix = gradient_path.with_suffix("")
    else:
        prefix = output_prefix.expanduser().resolve()
    report_path = Path(f"{prefix}.json")
    archive_path = Path(f"{prefix}.npz")
    _validate_output_bundle(
        report_path,
        archive_path,
        protected_inputs=protected_inputs,
        overwrite=overwrite,
    )
    expected_outputs = {
        path: (_snapshot_file(path) if path.exists() else None)
        for path in (report_path, archive_path)
    }

    report, arrays = build_package(
        gradient,
        manifest=manifest,
        segments=segments,
        input_path=gradient_path,
        manifest_path=manifest_path,
        input_snapshot=gradient_snapshot,
        manifest_snapshot=manifest_snapshot,
        byte_order=byte_order,
        producer_metadata=producer_metadata,
    )
    _write_output_bundle(
        report_path=report_path,
        archive_path=archive_path,
        report=report,
        arrays=arrays,
        protected_inputs=protected_inputs,
        verified_artifacts=verified_artifacts,
        expected_outputs=expected_outputs,
        overwrite=overwrite,
    )
    return report_path, archive_path, report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path, archive_path, report = package_gradient(
        args.gradient,
        manifest_path=args.manifest,
        output_prefix=args.output_prefix,
        byte_order=args.byte_order,
        producer_metadata_path=args.producer_metadata,
        overwrite=args.overwrite,
    )
    summary = report["packed_gradient"]
    print(f"Format-checked {summary['coefficient_count']:,} finite coefficients.")
    print("This packaging check does not certify the adjoint calculation.")
    print(f"L2 norm: {summary['l2_norm']:.9g}")
    print(f"Maximum absolute coefficient: {summary['max_abs']:.9g}")
    print(f"JSON report: {report_path}")
    print(f"Named-array archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
