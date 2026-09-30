#!/usr/bin/env python3
"""Fail closed unless a fresh Figure 5 result grid is complete and comparable."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

ARCHITECTURES = ("mlp", "cnn", "vit")
TRAINING_YEARS = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
RIGHT_YEARS = (50.0, 10000.0)
LEADS = tuple(range(1, 13))
BASE_SEED = 42
DEFAULT_LEFT_REPETITIONS = 3
DEFAULT_RIGHT_REPETITIONS = 1
RESULT_ROWS = (
    len(ARCHITECTURES) * len(TRAINING_YEARS) * DEFAULT_LEFT_REPETITIONS
    + len(ARCHITECTURES)
    * len(RIGHT_YEARS)
    * len(LEADS)
    * DEFAULT_RIGHT_REPETITIONS
)
RESULT_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 1
INPUT_PROFILE = "core4"
COMMON_MAXIMUM_LEAD = max(LEADS)
ARTIFACT_FILES = (
    "checkpoint.pt",
    "normalization.npz",
    "indices.npz",
    "metrics.json",
)
EXPECTED_TRAINING_CONFIG = {
    "batch_size": 256,
    "maximum_epochs": 100,
    "patience": 10,
    "minimum_improvement": 1.0e-4,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "statistics_batch_size": 1024,
    "deterministic": True,
}


@dataclass(frozen=True)
class GridExpectation:
    """Complete Figure 5 grid encoded by a validated result sidecar."""

    left_repetitions: int
    right_repetitions: int
    result_rows: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the completed fresh zc-v3 Figure 5 grid."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-left-repetitions",
        type=int,
        help=(
            "Require this many training-size repetitions. If omitted, use the "
            "strictly validated result sidecar (three in the legacy grid)."
        ),
    )
    parser.add_argument(
        "--expected-right-repetitions",
        type=int,
        help=(
            "Require this many lead-time repetitions. If omitted, use the "
            "strictly validated result sidecar (one in the legacy grid)."
        ),
    )
    return parser


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(indices: np.ndarray, block: tuple[int, int], label: str) -> None:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{label} must be a nonempty one-dimensional array.")
    if int(indices.min()) < block[0] or int(indices.max()) >= block[1]:
        raise ValueError(f"{label} escapes its fixed chronological block.")


def _finite_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Figure 5 row has an invalid {key!r} value.") from error
    if not math.isfinite(value):
        raise ValueError(f"Figure 5 row has a nonfinite {key!r} value.")
    return value


def _integer(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Figure 5 row has an invalid {key!r} value.") from error


def _year_label(years: float) -> str:
    if float(years).is_integer():
        return str(int(years))
    return f"{years:.6g}".replace(".", "p")


def _expected_relative_directory(row: dict[str, str]) -> PurePosixPath:
    architecture = row["architecture"]
    lead = _integer(row, "lead_months")
    years = _finite_float(row, "train_years_requested")
    seed = _integer(row, "seed")
    path = (
        PurePosixPath("figure5")
        / INPUT_PROFILE
        / architecture
        / f"lead-{lead:02d}m"
        / f"years-{_year_label(years)}"
    )
    if row["panel"] == "lead_time":
        path /= f"common-max-lead-{COMMON_MAXIMUM_LEAD:02d}m"
    return path / f"seed-{seed:06d}"


def _validated_relative_directory(row: dict[str, str]) -> PurePosixPath:
    try:
        value = PurePosixPath(row["artifact_directory"])
    except KeyError as error:
        raise ValueError("Figure 5 row is missing artifact_directory.") from error
    if value.is_absolute() or ".." in value.parts or value == PurePosixPath("."):
        raise ValueError(f"Unsafe Figure 5 artifact_directory: {value}")
    expected = _expected_relative_directory(row)
    if value != expected:
        raise ValueError(
            "Figure 5 artifact_directory does not match its row specification: "
            f"{value} versus {expected}."
        )
    return value


def _target_period(row: dict[str, str]) -> tuple[int, int]:
    period = (
        _integer(row, "test_target_start_step"),
        _integer(row, "test_target_stop_step_exclusive"),
    )
    if period[1] <= period[0]:
        raise ValueError(f"Invalid Figure 5 test target period: {period}")
    return period


def _positive_repetition_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _resolve_repetition_count(
    sidecar_value: object,
    explicit_value: int | None,
    label: str,
) -> int:
    sidecar_count = _positive_repetition_count(sidecar_value, label)
    if explicit_value is None:
        return sidecar_count
    expected_count = _positive_repetition_count(explicit_value, f"expected {label}")
    if sidecar_count != expected_count:
        raise ValueError(
            f"The Figure 5 sidecar declares {sidecar_count} {label}, but "
            f"{expected_count} were explicitly required."
        )
    return expected_count


def _result_row_count(left_repetitions: int, right_repetitions: int) -> int:
    return (
        len(ARCHITECTURES) * len(TRAINING_YEARS) * left_repetitions
        + len(ARCHITECTURES)
        * len(RIGHT_YEARS)
        * len(LEADS)
        * right_repetitions
    )


def _validate_sidecar(
    path: Path,
    *,
    expected_left_repetitions: int | None = None,
    expected_right_repetitions: int | None = None,
) -> tuple[dict[str, Any], GridExpectation]:
    sidecar = _load_object(path, "Figure 5 result sidecar")
    if sidecar.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Unexpected Figure 5 result-sidecar schema version.")
    if sidecar.get("figure") != 5:
        raise ValueError("The result sidecar does not identify Figure 5.")
    if sidecar.get("base_seed") != BASE_SEED:
        raise ValueError(f"The canonical Figure 5 base seed must be {BASE_SEED}.")
    data = sidecar.get("data")
    if not isinstance(data, dict) or data.get("input_profile") != INPUT_PROFILE:
        raise ValueError(
            f"The Figure 5 sidecar must describe the {INPUT_PROFILE!r} data view."
        )
    metadata_sha256 = data.get("metadata_sha256")
    if (
        not isinstance(metadata_sha256, str)
        or len(metadata_sha256) != 64
        or any(character not in "0123456789abcdef" for character in metadata_sha256)
    ):
        raise ValueError("The Figure 5 sidecar has an invalid data metadata digest.")
    if sidecar.get("training_config") != EXPECTED_TRAINING_CONFIG:
        raise ValueError(
            "The Figure 5 sidecar does not use the canonical training configuration."
        )
    left = sidecar.get("left_panel")
    right = sidecar.get("right_panel")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("The Figure 5 sidecar is missing panel definitions.")
    left_repetitions = _resolve_repetition_count(
        left.get("repetitions"),
        expected_left_repetitions,
        "left-panel repetitions",
    )
    right_repetitions = _resolve_repetition_count(
        right.get("repetitions"),
        expected_right_repetitions,
        "right-panel repetitions",
    )
    if (
        left.get("lead_months") != 10
        or tuple(float(value) for value in left.get("training_years", ()))
        != TRAINING_YEARS
    ):
        raise ValueError("The Figure 5 sidecar has an unexpected left-panel grid.")
    if (
        tuple(int(value) for value in right.get("lead_months", ())) != LEADS
        or tuple(float(value) for value in right.get("training_years", ()))
        != RIGHT_YEARS
        or right.get("common_period_max_lead_months") != COMMON_MAXIMUM_LEAD
    ):
        raise ValueError("The Figure 5 sidecar has an unexpected right-panel grid.")
    result_rows = _result_row_count(left_repetitions, right_repetitions)
    if sidecar.get("result_rows") != result_rows:
        raise ValueError(
            "The result-sidecar row count does not match its declared complete grid."
        )
    return sidecar, GridExpectation(
        left_repetitions=left_repetitions,
        right_repetitions=right_repetitions,
        result_rows=result_rows,
    )


def _expected_spec(row: dict[str, str]) -> dict[str, Any]:
    return {
        "architecture": row["architecture"],
        "lead_months": _integer(row, "lead_months"),
        "train_years": _finite_float(row, "train_years_requested"),
        "seed": _integer(row, "seed"),
        "train_fraction": 0.9,
        "validation_fraction": 0.2,
        "common_period_max_lead_months": (
            COMMON_MAXIMUM_LEAD if row["panel"] == "lead_time" else None
        ),
        "input_profile": INPUT_PROFILE,
    }


def _validate_artifact(
    row: dict[str, str],
    directory: Path,
    *,
    sidecar: dict[str, Any],
) -> None:
    completion_path = directory / "completed.json"
    indices_path = directory / "indices.npz"
    completion = _load_object(completion_path, "Figure 5 completion manifest")
    if completion.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unexpected Figure 5 artifact schema: {directory}")
    expected_spec = _expected_spec(row)
    if completion.get("spec") != expected_spec:
        raise ValueError(f"Artifact specification does not match its row: {directory}")
    if completion.get("training_config") != sidecar["training_config"]:
        raise ValueError(
            f"Artifact training configuration does not match the sidecar: {directory}"
        )
    if completion.get("data_metadata_sha256") != sidecar["data"]["metadata_sha256"]:
        raise ValueError(f"Artifact data provenance does not match: {directory}")

    files = completion.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES):
        raise ValueError(f"Invalid Figure 5 file manifest: {completion_path}")
    for filename in ARTIFACT_FILES:
        path = directory / filename
        expected_digest = files.get(filename)
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise ValueError(f"Corrupt Figure 5 cache file: {path}")

    row_checkpoint = row.get("checkpoint_sha256")
    if row_checkpoint != files["checkpoint.pt"]:
        raise ValueError(f"CSV checkpoint digest does not match: {directory}")

    metrics = _load_object(directory / "metrics.json", "Figure 5 metrics")
    if metrics.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unexpected Figure 5 metrics schema: {directory}")
    if metrics.get("spec") != expected_spec:
        raise ValueError(f"Metrics specification does not match its row: {directory}")
    if metrics.get("training_config") != sidecar["training_config"]:
        raise ValueError(f"Metrics training configuration does not match: {directory}")
    if metrics.get("model", {}).get("checkpoint_sha256") != row_checkpoint:
        raise ValueError(f"Metrics checkpoint digest does not match: {directory}")
    if not math.isclose(
        float(metrics.get("test_r2", math.nan)),
        _finite_float(row, "test_r2"),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"Metrics test R2 does not match its CSV row: {directory}")
    evaluation = metrics.get("evaluation")
    if not isinstance(evaluation, dict) or (
        evaluation.get("test_target_start_step"),
        evaluation.get("test_target_stop_step_exclusive"),
    ) != _target_period(row):
        raise ValueError(
            f"Metrics target period does not match its CSV row: {directory}"
        )

    selection = metrics.get("selection")
    if not isinstance(selection, dict) or not isinstance(
        selection.get("fixed_blocks"), dict
    ):
        raise ValueError(f"Metrics are missing fixed chronological blocks: {directory}")
    blocks = selection["fixed_blocks"]
    try:
        train_block = tuple(int(value) for value in blocks["train"])
        validation_block = tuple(int(value) for value in blocks["validation"])
        test_block = tuple(int(value) for value in blocks["test"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid fixed chronological blocks: {directory}") from error
    all_blocks = (train_block, validation_block, test_block)
    if any(len(block) != 2 or block[1] <= block[0] for block in all_blocks):
        raise ValueError(f"Invalid fixed chronological blocks: {directory}")
    if not (
        train_block[1] <= validation_block[0]
        and validation_block[1] <= test_block[0]
    ):
        raise ValueError(f"Fixed chronological blocks are not ordered: {directory}")
    with np.load(indices_path, allow_pickle=False) as archive:
        required_arrays = {
            "fit_inputs",
            "standardization_inputs",
            "validation_inputs",
            "test_inputs",
        }
        if not required_arrays.issubset(archive.files):
            raise ValueError(f"Figure 5 index archive is incomplete: {indices_path}")
        _inside(archive["fit_inputs"], train_block, "fit inputs")
        _inside(
            archive["standardization_inputs"],
            train_block,
            "standardization inputs",
        )
        _inside(
            archive["validation_inputs"],
            validation_block,
            "validation inputs",
        )
        _inside(archive["test_inputs"], test_block, "test inputs")


def audit_results(
    result_path: Path,
    artifact_root: Path,
    *,
    expected_left_repetitions: int | None = None,
    expected_right_repetitions: int | None = None,
) -> int:
    """Validate the canonical result grid and return its artifact count."""

    result_path = result_path.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    rows = _rows(result_path)
    sidecar, grid = _validate_sidecar(
        result_path.with_suffix(".json"),
        expected_left_repetitions=expected_left_repetitions,
        expected_right_repetitions=expected_right_repetitions,
    )
    if len(rows) != grid.result_rows:
        raise ValueError(
            f"Expected {grid.result_rows} Figure 5 rows, found {len(rows)}."
        )

    left_counts: Counter[tuple[str, float, int]] = Counter()
    right_counts: Counter[tuple[str, float, int, int]] = Counter()
    left_dates: set[tuple[int, int]] = set()
    right_dates: set[tuple[int, int]] = set()
    artifact_rows: dict[PurePosixPath, dict[str, str]] = {}
    for row in rows:
        architecture = row.get("architecture")
        years = _finite_float(row, "train_years_requested")
        lead = _integer(row, "lead_months")
        repetition = _integer(row, "repetition")
        seed = _integer(row, "seed")
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unexpected architecture {architecture!r}.")
        _finite_float(row, "test_r2")
        if seed != BASE_SEED + repetition:
            raise ValueError("Figure 5 seed does not match its repetition.")
        panel = row.get("panel")
        if panel == "training_size":
            left_counts[(architecture, years, repetition)] += 1
            if (
                lead != 10
                or years not in TRAINING_YEARS
                or repetition not in range(grid.left_repetitions)
            ):
                raise ValueError("Unexpected left-panel configuration.")
            left_dates.add(_target_period(row))
        elif panel == "lead_time":
            right_counts[(architecture, years, lead, repetition)] += 1
            if (
                years not in RIGHT_YEARS
                or lead not in LEADS
                or repetition not in range(grid.right_repetitions)
            ):
                raise ValueError("Unexpected right-panel configuration.")
            right_dates.add(_target_period(row))
        else:
            raise ValueError(f"Unexpected panel {panel!r}.")
        relative_directory = _validated_relative_directory(row)
        if relative_directory in artifact_rows:
            raise ValueError(
                f"Figure 5 rows repeat artifact directory {relative_directory}."
            )
        artifact_rows[relative_directory] = row

    expected_left = {
        (architecture, years, repetition)
        for architecture in ARCHITECTURES
        for years in TRAINING_YEARS
        for repetition in range(grid.left_repetitions)
    }
    expected_right = {
        (architecture, years, lead, repetition)
        for architecture in ARCHITECTURES
        for years in RIGHT_YEARS
        for lead in LEADS
        for repetition in range(grid.right_repetitions)
    }
    if set(left_counts) != expected_left or any(
        value != 1 for value in left_counts.values()
    ):
        raise ValueError("Left-panel grid is incomplete or duplicated.")
    if set(right_counts) != expected_right or any(
        value != 1 for value in right_counts.values()
    ):
        raise ValueError("Right-panel grid is incomplete or duplicated.")
    if len(left_dates) != 1:
        raise ValueError("Left-panel target dates are not shared by every run.")
    if len(right_dates) != 1:
        raise ValueError("Right-panel target dates are not identical across all runs.")
    if len(artifact_rows) != grid.result_rows:
        raise ValueError(
            f"Expected {grid.result_rows} distinct artifacts, "
            f"found {len(artifact_rows)}."
        )

    for relative_directory, row in sorted(
        artifact_rows.items(), key=lambda item: str(item[0])
    ):
        _validate_artifact(
            row,
            artifact_root.joinpath(*relative_directory.parts),
            sidecar=sidecar,
        )
    return len(artifact_rows)


def main() -> int:
    args = build_parser().parse_args()
    artifact_count = audit_results(
        args.results,
        args.artifacts_dir,
        expected_left_repetitions=args.expected_left_repetitions,
        expected_right_repetitions=args.expected_right_repetitions,
    )
    print(
        f"Figure 5 audit passed: {artifact_count} rows, {artifact_count} distinct "
        "complete artifacts, fixed blocks, and shared panel target dates verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
