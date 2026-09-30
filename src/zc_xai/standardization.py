"""Read-only validation of fit-only standardization artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import Standardizer
from .io import load_json, sha256_file


@dataclass(frozen=True)
class RecordedStandardizer:
    """A standardizer tied to one completed experiment manifest."""

    standardizer: Standardizer
    artifact_directory: Path
    normalization_path: Path
    normalization_sha256: str
    indices_sha256: str
    completion_path: Path
    completion_sha256: str
    completion_schema_version: int
    generation_id: str
    spec: dict[str, Any]
    data_metadata_sha256: str


def _contained_directory(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact directory must be a nonempty relative path.")
    candidate_path = Path(relative)
    if candidate_path.is_absolute():
        raise ValueError("Artifact directory must be relative to --artifacts-dir.")
    resolved_root = root.expanduser().resolve()
    resolved = (resolved_root / candidate_path).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError("Artifact directory escapes --artifacts-dir.")
    return resolved


def load_recorded_standardizer(
    artifact_root: Path,
    artifact_directory: str,
    *,
    expected_data_metadata_sha256: str,
    expected_input_shape: tuple[int, ...],
    expected_normalization_sha256: str | None = None,
    expected_spec: dict[str, Any] | None = None,
) -> RecordedStandardizer:
    """Validate and load normalization without loading a model checkpoint.

    The completion manifest must bind both ``normalization.npz`` and
    ``indices.npz``.  The latter is checked so the stored normalization count can
    be verified against the exact population used to fit it.  Fresh zc-v3
    experiments record that population as ``standardization_inputs``; older
    experiments used ``fit_inputs`` for both fitting and standardization.
    """

    directory = _contained_directory(artifact_root, artifact_directory)
    completion_path = directory / "completed.json"
    normalization_path = directory / "normalization.npz"
    indices_path = directory / "indices.npz"
    for path in (completion_path, normalization_path, indices_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed normalization artifact missing: {path}")

    # Read the commit marker before and after the bound files so a concurrent
    # training overwrite cannot silently combine two artifact generations.
    completion_sha256 = sha256_file(completion_path)
    completion = load_json(completion_path)
    schema_version = completion.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"Invalid training schema in {completion_path}.")
    if completion.get("data_metadata_sha256") != expected_data_metadata_sha256:
        raise ValueError(f"Processed-data provenance mismatch: {directory}")
    spec = completion.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid experiment specification: {completion_path}")
    if expected_spec is not None and spec != expected_spec:
        raise ValueError(f"Experiment specification mismatch: {directory}")
    generation_id = completion.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError(f"Invalid generation id: {completion_path}")
    files = completion.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Invalid completion file manifest: {completion_path}")

    normalization_sha256 = sha256_file(normalization_path)
    indices_sha256 = sha256_file(indices_path)
    if files.get("normalization.npz") != normalization_sha256:
        raise ValueError(f"Completion hash mismatch: {normalization_path}")
    if files.get("indices.npz") != indices_sha256:
        raise ValueError(f"Completion hash mismatch: {indices_path}")
    if (
        expected_normalization_sha256 is not None
        and normalization_sha256 != expected_normalization_sha256
    ):
        raise ValueError(f"Recorded normalization hash mismatch: {normalization_path}")

    with np.load(normalization_path, allow_pickle=False) as archive:
        standardizer = Standardizer(
            mean=np.asarray(archive["mean"], dtype=np.float32),
            scale=np.asarray(archive["scale"], dtype=np.float32),
            scale_floor=float(archive["scale_floor"]),
            count=int(archive["count"]),
        )
    if standardizer.mean.shape != expected_input_shape:
        raise ValueError(
            f"Normalization mean has the wrong shape: {normalization_path}"
        )
    if standardizer.scale.shape != expected_input_shape:
        raise ValueError(
            f"Normalization scale has the wrong shape: {normalization_path}"
        )
    if not np.isfinite(standardizer.mean).all():
        raise ValueError(f"Normalization mean is nonfinite: {normalization_path}")
    if not np.isfinite(standardizer.scale).all() or np.any(standardizer.scale <= 0):
        raise ValueError(
            "Normalization scale must be finite and positive: "
            f"{normalization_path}"
        )
    if not math.isfinite(standardizer.scale_floor) or standardizer.scale_floor <= 0:
        raise ValueError(f"Normalization floor is invalid: {normalization_path}")
    if standardizer.count <= 0:
        raise ValueError(f"Normalization count is invalid: {normalization_path}")

    with np.load(indices_path, allow_pickle=False) as archive:
        population_name = (
            "standardization_inputs"
            if "standardization_inputs" in archive.files
            else "fit_inputs"
        )
        raw_population = np.asarray(archive[population_name])
    population_label = (
        "recorded standardization indices"
        if population_name == "standardization_inputs"
        else "optimization-fit indices"
    )
    if raw_population.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"{population_label.capitalize()} are not integers: {indices_path}"
        )
    population = np.asarray(raw_population, dtype=np.int64)
    if population.ndim != 1 or population.size != standardizer.count:
        raise ValueError(
            f"Normalization count does not match the {population_label}: "
            f"{directory}"
        )
    if population.size and (
        np.any(population < 0)
        or not np.array_equal(population, np.unique(population))
    ):
        raise ValueError(
            f"{population_label.capitalize()} must be nonnegative, sorted, "
            "and unique: "
            f"{indices_path}"
        )
    if sha256_file(completion_path) != completion_sha256:
        raise RuntimeError(
            "Completed normalization artifact changed while it was being read: "
            f"{directory}"
        )

    return RecordedStandardizer(
        standardizer=standardizer,
        artifact_directory=directory,
        normalization_path=normalization_path,
        normalization_sha256=normalization_sha256,
        indices_sha256=indices_sha256,
        completion_path=completion_path,
        completion_sha256=completion_sha256,
        completion_schema_version=schema_version,
        generation_id=generation_id,
        spec=spec,
        data_metadata_sha256=expected_data_metadata_sha256,
    )
