"""Validation helpers for completed, transferable experiment bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file


@dataclass(frozen=True)
class CompletedBundle:
    directory: Path
    metadata_path: Path
    completion_path: Path
    metadata: dict[str, Any]
    files: dict[str, Path]
    sha256: dict[str, str]


def _safe_basename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label} must be a safe filename basename.")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def load_completed_bundle(
    directory: Path,
    *,
    expected_artifact: str,
    required_files: tuple[str, ...],
) -> CompletedBundle:
    """Load a bundle only after validating its completion marker and file hashes."""

    resolved = directory.expanduser().resolve()
    metadata_path = resolved / "metadata.json"
    completion_path = resolved / "completed.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Bundle metadata not found: {metadata_path}")
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"Bundle completion marker not found: {completion_path}"
        )
    metadata = load_json(metadata_path)
    completion = load_json(completion_path)
    if metadata.get("artifact") != expected_artifact:
        raise ValueError("Bundle metadata describes a different artifact.")
    if completion.get("artifact") != expected_artifact:
        raise ValueError("Bundle completion marker describes a different artifact.")
    if metadata.get("schema_version") != completion.get("schema_version"):
        raise ValueError("Bundle metadata and completion schemas disagree.")
    completion_files = completion.get("files")
    output_records = metadata.get("output_files")
    if not isinstance(completion_files, dict) or not isinstance(output_records, dict):
        raise ValueError("Bundle file manifests must be JSON objects.")
    expected_metadata_hash = _sha256(
        completion_files.get("metadata"),
        "completed.files.metadata",
    )
    if sha256_file(metadata_path) != expected_metadata_hash:
        raise ValueError("Bundle metadata hash does not match completed.json.")

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for key in required_files:
        record = output_records.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Bundle metadata is missing output_files.{key}.")
        filename = _safe_basename(record.get("file"), f"output_files.{key}.file")
        recorded_hash = _sha256(
            record.get("sha256"),
            f"output_files.{key}.sha256",
        )
        completion_hash = _sha256(
            completion_files.get(key),
            f"completed.files.{key}",
        )
        if completion_hash != recorded_hash:
            raise ValueError(f"Bundle manifests disagree about {key}.")
        path = resolved / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundle file not found: {path}")
        if sha256_file(path) != recorded_hash:
            raise ValueError(f"Bundle file hash mismatch: {path}")
        paths[key] = path
        hashes[key] = recorded_hash
    return CompletedBundle(
        directory=resolved,
        metadata_path=metadata_path,
        completion_path=completion_path,
        metadata=metadata,
        files=paths,
        sha256=hashes,
    )
