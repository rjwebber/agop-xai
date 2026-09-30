"""Portable validated plot-data bundles for ZC steering Figures 10--12."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .io import load_json, sha256_file

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_ARTIFACT = "zc-manuscript-figures-10-12-plot-data"


def sha256_array_contract(value: np.ndarray) -> str:
    """Hash an array together with its dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_public_document(value: Any, *, location: str = "document") -> None:
    """Reject host-specific paths from a document intended for release."""

    if isinstance(value, dict):
        for key, item in value.items():
            validate_public_document(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_public_document(item, location=f"{location}[{index}]")
        return
    if not isinstance(value, str):
        return
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ValueError(f"Public bundle contains an absolute path at {location}")
    if normalized == "scratch" or normalized.startswith("scratch/"):
        raise ValueError(f"Public bundle exposes a scratch path at {location}")


@dataclass(frozen=True)
class SteeringPlotBundle:
    """Validated, portable arrays and provenance for Figures 10--12."""

    directory: Path
    manifest_path: Path
    archive_path: Path
    reports_path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def array(self, key: str) -> np.ndarray:
        try:
            return self.arrays[key]
        except KeyError as error:
            raise KeyError(f"Steering plot bundle has no array {key!r}") from error


def load_steering_plot_bundle(directory: Path) -> SteeringPlotBundle:
    """Load and authenticate a compact public steering plot-data bundle."""

    root = directory.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("artifact") != BUNDLE_ARTIFACT
        or manifest.get("portable_paths") is not True
    ):
        raise ValueError("Unsupported steering plot bundle contract")
    validate_public_document(manifest, location="manifest")

    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise ValueError("Steering bundle archive record is missing")
    filename = archive.get("file")
    if not isinstance(filename, str) or PurePosixPath(filename).name != filename:
        raise ValueError("Steering bundle archive filename must be local")
    archive_path = root / filename
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != archive.get("size_bytes")
        or sha256_file(archive_path) != archive.get("sha256")
    ):
        raise ValueError("Steering bundle archive failed integrity validation")

    report_archive = manifest.get("accepted_reports")
    if not isinstance(report_archive, dict):
        raise ValueError("Accepted-report archive record is missing")
    report_filename = report_archive.get("file")
    if (
        not isinstance(report_filename, str)
        or PurePosixPath(report_filename).name != report_filename
    ):
        raise ValueError("Accepted-report archive filename must be local")
    reports_path = root / report_filename
    if (
        not reports_path.is_file()
        or reports_path.stat().st_size != report_archive.get("size_bytes")
        or sha256_file(reports_path) != report_archive.get("sha256")
    ):
        raise ValueError("Accepted-report archive failed integrity validation")
    expected_report_hashes: dict[str, str] = {}
    figures = manifest.get("figures")
    if not isinstance(figures, dict):
        raise ValueError("Steering bundle figure metadata are missing")
    for figure_record in figures.values():
        stack = [figure_record]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                member = item.get("accepted_report_file")
                digest = item.get("report_sha256")
                if member is not None or digest is not None:
                    if (
                        not isinstance(member, str)
                        or not isinstance(digest, str)
                        or len(digest) != 64
                    ):
                        raise ValueError("Accepted-report reference is malformed")
                    expected_report_hashes[member] = digest
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    if len(expected_report_hashes) != report_archive.get("member_count"):
        raise ValueError("Accepted-report member count does not match the manifest")
    with zipfile.ZipFile(reports_path, mode="r") as reports:
        if set(reports.namelist()) != set(expected_report_hashes):
            raise ValueError(
                "Accepted-report archive members do not match the manifest"
            )
        for member, expected_digest in expected_report_hashes.items():
            raw = reports.read(member)
            if hashlib.sha256(raw).hexdigest() != expected_digest:
                raise ValueError(f"Accepted report failed hash validation: {member}")
            validate_public_document(
                json.loads(raw), location=f"accepted_reports.{member}"
            )

    contract = manifest.get("arrays")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("Steering bundle array contract is missing")
    with np.load(archive_path, allow_pickle=False) as source:
        if set(source.files) != set(contract):
            raise ValueError("Steering bundle archive keys do not match its manifest")
        arrays = {key: np.asarray(source[key]) for key in source.files}
    for key, array in arrays.items():
        record = contract.get(key)
        finite = array.dtype.kind not in {"f", "c"} or np.isfinite(array).all()
        if (
            not isinstance(record, dict)
            or list(array.shape) != record.get("shape")
            or array.dtype.str != record.get("dtype")
            or not finite
            or sha256_array_contract(array) != record.get("sha256")
        ):
            raise ValueError(f"Steering bundle array failed validation: {key}")
    return SteeringPlotBundle(
        directory=root,
        manifest_path=manifest_path,
        archive_path=archive_path,
        reports_path=reports_path,
        manifest=manifest,
        arrays=arrays,
    )
