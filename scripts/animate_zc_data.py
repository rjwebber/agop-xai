#!/usr/bin/env python3
"""Animate standardized fields from the public ``zc-v3`` data set.

Arrays are memory-mapped, so this script does not load the multi-gigabyte
fields into RAM.
By default the arrows show the depth-averaged ocean currents written by the
Fortran variables ``U1``/``V1``. Use ``--vector-kind surface-wind`` to show the
atmospheric surface-wind anomalies written as ``UO``/``VO`` instead. The data
use explicit semantic filenames, so the two vector pairs
cannot be confused by their abbreviated source names.  Each displayed field is
transformed with the exact training-block mean and scale from a validated
completed ``zc-v3`` training artifact; validation and test statistics are never
used.

Examples
--------
Display an animation interactively::

    python scripts/animate_zc_data.py \
        --data-dir data/processed/zc-v3 \
        --artifacts-dir artifacts/zc-v3

Save a short MP4::

    python scripts/animate_zc_data.py \
        --data-dir data/processed/zc-v3 \
        --artifacts-dir artifacts/zc-v3 \
        --frames 300 \
        --output outputs/video/zc_standardized_ocean_currents.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.colors import TwoSlopeNorm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import (  # noqa: E402
    FRESH_FIELD_IDENTITIES,
    FRESH_INPUT_PROFILES,
    FRESH_SCHEMA_VERSION,
)
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    sha256_array,
    sha256_file,
    write_json,
)
from zc_xai.standardization import (  # noqa: E402
    RecordedStandardizer,
    load_recorded_standardizer,
)

LOGGER = logging.getLogger(__name__)

AUTO_QUIVER_SAMPLE_FRAMES = 200
AUTO_QUIVER_REFERENCE_PERCENTILE = 95.0
AUTO_QUIVER_REFERENCE_WIDTH_FRACTION = 0.025
COLORMAP_LOOKUP_TABLE_SIZE = 256
COLORMAP_NAME = (
    "Matplotlib bwr sampled at 256 colors (even; no exact-white lookup entry)"
)


def _even_bwr():
    """Use the same balanced blue-white-red palette as Figures 6--8."""

    return plt.get_cmap("bwr").resampled(COLORMAP_LOOKUP_TABLE_SIZE)


COLORMAP = _even_bwr()

# Coordinate fallback for format-version-1 metadata produced by the original
# extractor. Later format versions store these coordinates explicitly.
LEGACY_LATITUDES = -19.0 + 2.0 * np.arange(20, dtype=np.float64)
LEGACY_LONGITUDES = 129.375 + 5.625 * np.arange(27, dtype=np.float64)

VECTOR_FIELDS = {
    "ocean-current": {
        "format_v3": ("zonal_ocean_current", "meridional_ocean_current"),
        "legacy": ("u1", "v1"),
        "label": "Depth-averaged ocean-current anomaly (source U1/V1)",
    },
    "surface-wind": {
        "format_v3": ("zonal_surface_wind", "meridional_surface_wind"),
        "legacy": ("uo", "vo"),
        "label": "Atmospheric surface-wind anomaly (source UO/VO)",
    },
}

FORMAT_V3_FIELD_PROVENANCE = {
    "sst_anomaly": ("SST", 1),
    "zonal_wind_stress": ("TAUX", 2),
    "meridional_wind_stress": ("TAUY", 3),
    "zonal_surface_wind": ("UO", 4),
    "meridional_surface_wind": ("VO", 5),
    "thermocline_depth": ("H1", 6),
    "zonal_ocean_current": ("U1", 7),
    "meridional_ocean_current": ("V1", 8),
    "atmospheric_heating": ("QF", 9),
    "total_sst": ("TT", 10),
}


@dataclass(frozen=True)
class PreparedData:
    """Memory-mapped fields and metadata needed by the animation."""

    metadata: dict[str, Any]
    sst: Any
    u_component: Any
    v_component: Any
    sst_field_name: str
    u_field_name: str
    v_field_name: str
    vector_label: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    steps_per_month: float
    steps_per_year: float
    normalization_artifact: RecordedStandardizer | None = None


@dataclass(frozen=True)
class StandardizedField:
    """Lazy feature-wise standardization over a memory-mapped time series."""

    source: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(np.float32)

    def __getitem__(self, key: Any) -> np.ndarray:
        keys = key if isinstance(key, tuple) else (key,)
        spatial_keys = list(keys[1:])
        while len(spatial_keys) < 2:
            spatial_keys.append(slice(None))
        if len(spatial_keys) != 2 or any(value is Ellipsis for value in spatial_keys):
            raise IndexError(
                "StandardizedField expects time, latitude, longitude keys."
            )
        spatial = tuple(spatial_keys)
        values = np.asarray(self.source[key], dtype=np.float32)
        return (values - self.mean[spatial]) / self.scale[spatial]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Animate extracted Zebiak-Cane SST with depth-averaged ocean "
            "currents or atmospheric surface winds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/zc-v3"),
        help="Public data/processed/zc-v3 directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/zc-v3"),
        help="Root containing the completed training-only normalization artifact.",
    )
    parser.add_argument(
        "--input-profile",
        choices=tuple(sorted(FRESH_INPUT_PROFILES)),
        default="core4",
        help="Fresh-data field profile used by the normalization artifact.",
    )
    parser.add_argument(
        "--architecture",
        choices=("mlp", "cnn", "vit"),
        default="cnn",
        help="Trained experiment whose fit-only normalizer defines model inputs.",
    )
    parser.add_argument(
        "--lead-months",
        type=positive_int,
        default=10,
        help="Forecast lead of the normalization-source experiment.",
    )
    parser.add_argument(
        "--train-years",
        type=positive_float,
        default=10000.0,
        help="Training-window length of the normalization-source experiment.",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=42,
        help="Seed of the normalization-source experiment.",
    )
    parser.add_argument(
        "--vector-kind",
        choices=tuple(VECTOR_FIELDS),
        default="ocean-current",
        help=(
            "Vector pair to draw. The model writer identifies U1/V1 as "
            "depth-averaged ocean current and UO/VO as atmospheric surface wind."
        ),
    )
    parser.add_argument(
        "--start-step",
        type=nonnegative_int,
        default=0,
        help="First simulation-step index to display.",
    )
    parser.add_argument(
        "--frames",
        type=positive_int,
        default=1000,
        help="Maximum number of animation frames.",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=1,
        help="Simulation steps between consecutive frames.",
    )
    parser.add_argument(
        "--interval-ms",
        type=positive_int,
        default=100,
        help="Delay between interactive frames in milliseconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/video/zc_standardized_ocean_currents.mp4"),
        help="Output .mp4 or .gif path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing animation at --output.",
    )
    speed_group = parser.add_mutually_exclusive_group()
    speed_group.add_argument(
        "--fps",
        type=positive_float,
        default=10.0,
        help="Frames per second when saving.",
    )
    speed_group.add_argument(
        "--duration-seconds",
        type=positive_float,
        help=(
            "Target MP4/GIF duration. When supplied, derive the frame rate from "
            "the selected frame count instead of using --fps."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=150,
        help="Resolution when saving.",
    )
    parser.add_argument(
        "--quiver-stride",
        type=positive_int,
        default=2,
        help="Plot every nth vector in both spatial directions.",
    )
    parser.add_argument(
        "--quiver-scale",
        type=positive_float,
        default=None,
        help=(
            "Fixed Matplotlib scale in vector-units per axes-width unit; smaller "
            "values draw longer arrows. Omit it to estimate a stable scale from "
            "nonzero vectors across the selected animation."
        ),
    )
    parser.add_argument(
        "--sst-min",
        type=float,
        default=-8.0,
        help="Lower standardized SST-anomaly color limit in fit SD units.",
    )
    parser.add_argument(
        "--sst-max",
        type=float,
        default=8.0,
        help="Upper standardized SST-anomaly color limit in fit SD units.",
    )
    parser.add_argument(
        "--aspect",
        choices=("grid", "geographic"),
        default="grid",
        help=(
            "Use square model-grid cells (paper-style) or preserve the "
            "latitude/longitude degree aspect ratio."
        ),
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "bilinear", "bicubic"),
        default="bilinear",
        help="Spatial interpolation used only for rendering the SST image.",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Do not loop the animation in an interactive window.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not math.isfinite(args.sst_min) or not math.isfinite(args.sst_max):
        parser.error("--sst-min and --sst-max must be finite")
    if not args.sst_min < 0.0 < args.sst_max:
        parser.error("SST color limits must satisfy --sst-min < 0 < --sst-max")
    if args.duration_seconds is not None and args.output is None:
        parser.error("--duration-seconds requires --output")

    return args


def read_metadata(data_dir: Path) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Run generate_fresh_zc_dataset.py before making the animation."
        )

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected a JSON object in {metadata_path}.")
    return metadata


def load_field(
    data_dir: Path,
    metadata: dict[str, Any],
    field_name: str,
) -> np.ndarray:
    try:
        if metadata.get("schema_version") == FRESH_SCHEMA_VERSION:
            file_name = metadata["field_files"][field_name]["file"]
        else:
            file_name = metadata["clean_data"]["field_files"][field_name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Field {field_name!r} is not listed correctly in metadata.json."
        ) from error

    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise ValueError(
            f"Unsafe or invalid file name for field {field_name!r}: {file_name!r}."
        )

    path = data_dir / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Extracted field file not found: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def metadata_format_version(metadata: dict[str, Any]) -> int:
    if metadata.get("schema_version") == FRESH_SCHEMA_VERSION:
        return 4
    raw_format_version = metadata.get("format_version", 1)
    if isinstance(raw_format_version, bool) or not isinstance(raw_format_version, int):
        raise TypeError(
            "metadata format_version must be 1, 2, 3, or the fresh zc-v3 "
            "schema identifier; received "
            f"{raw_format_version!r}."
        )
    format_version = raw_format_version
    if format_version not in {1, 2, 3}:
        raise ValueError(
            "Unsupported metadata format_version "
            f"{format_version}; expected 1, 2, or 3."
        )
    return format_version


def validate_fresh_field_semantics(metadata: dict[str, Any]) -> None:
    """Verify names and native-variable labels in the public zc-v3 release."""

    raw_fields = metadata.get("fields")
    if not isinstance(raw_fields, list):
        raise TypeError("Fresh zc-v3 metadata must contain a fields list.")
    by_name = {
        field.get("name"): field
        for field in raw_fields
        if isinstance(field, dict) and isinstance(field.get("name"), str)
    }
    expected = {name: fortran for name, fortran, _group in FRESH_FIELD_IDENTITIES}
    if set(by_name) != set(expected):
        raise ValueError("Fresh zc-v3 field names do not match the release schema.")
    for name, fortran in expected.items():
        if by_name[name].get("fortran") != fortran:
            raise ValueError(
                f"Field {name!r} must map to native variable {fortran!r}."
            )


def validate_format_v3_field_semantics(metadata: dict[str, Any]) -> None:
    """Reject format-v3 metadata whose semantic names do not match raw records."""

    raw_fields = metadata.get("fields")
    if not isinstance(raw_fields, list):
        raise TypeError("Format-version-3 metadata must contain a fields list.")

    fields: dict[str, dict[str, Any]] = {}
    for raw_field in raw_fields:
        if not isinstance(raw_field, dict) or not isinstance(
            raw_field.get("name"), str
        ):
            raise TypeError(
                "Every format-version-3 fields entry must be an object with a name."
            )
        name = raw_field["name"]
        if name in fields:
            raise ValueError(f"Duplicate field metadata for {name!r}.")
        fields[name] = raw_field

    for public_name, (source_name, record_number) in FORMAT_V3_FIELD_PROVENANCE.items():
        try:
            field = fields[public_name]
        except KeyError as error:
            raise ValueError(
                f"Format-version-3 metadata is missing field {public_name!r}."
            ) from error
        if field.get("source_name") != source_name:
            raise ValueError(
                f"Field {public_name!r} must map to source variable {source_name}, "
                f"not {field.get('source_name')!r}."
            )
        if field.get("source_record_number") != record_number:
            raise ValueError(
                f"Field {public_name!r} must map to source record {record_number}, "
                f"not {field.get('source_record_number')!r}."
            )


def coordinates_from_metadata(
    metadata: dict[str, Any],
    spatial_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    format_version = metadata_format_version(metadata)

    try:
        if format_version == 4:
            grid = metadata["grid"]
        else:
            grid_name = "active_domain" if format_version == 3 else "extracted"
            grid = metadata["grid"][grid_name]
        latitudes = np.asarray(grid["latitude_degrees_north"], dtype=np.float64)
        longitudes = np.asarray(grid["longitude_degrees_east"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        if format_version != 1:
            raise ValueError(
                "Modern metadata must contain valid active-grid "
                "latitude and longitude arrays."
            ) from error
        if spatial_shape != (20, 27):
            raise ValueError(
                "metadata.json has no valid coordinates, and the array shape "
                f"{spatial_shape} is not the known legacy (20, 27) grid."
            ) from None
        LOGGER.warning(
            "Using standard ZC coordinates because this legacy metadata does "
            "not store latitude/longitude arrays. Re-extract with the current "
            "generate_fresh_zc_dataset.py before publication."
        )
        return LEGACY_LATITUDES.copy(), LEGACY_LONGITUDES.copy()

    expected_latitudes, expected_longitudes = spatial_shape
    if latitudes.shape != (expected_latitudes,):
        raise ValueError(
            "Latitude-coordinate length does not match the arrays: "
            f"{latitudes.shape} versus {(expected_latitudes,)}."
        )
    if longitudes.shape != (expected_longitudes,):
        raise ValueError(
            "Longitude-coordinate length does not match the arrays: "
            f"{longitudes.shape} versus {(expected_longitudes,)}."
        )
    for name, coordinates in (
        ("latitude", latitudes),
        ("longitude", longitudes),
    ):
        if not np.isfinite(coordinates).all():
            raise ValueError(f"The {name} coordinates contain nonfinite values.")
        spacing = np.diff(coordinates)
        if not np.all(spacing > 0):
            raise ValueError(f"The {name} coordinates must be strictly increasing.")
        if not np.allclose(spacing, spacing[0], rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                f"The {name} coordinates must be uniformly spaced for imshow."
            )

    return latitudes, longitudes


def positive_sampling_value(metadata: dict[str, Any], name: str) -> float:
    try:
        container = (
            metadata["integration"]
            if metadata_format_version(metadata) == 4
            else metadata["sampling"]
        )
        value = float(container[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"metadata.json does not contain a numeric sampling.{name}."
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"metadata sampling.{name} must be positive and finite.")
    return value


def validate_structural_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Cross-check cheap structural metadata against memory-mapped headers."""

    try:
        if metadata_format_version(metadata) == 4:
            raw_time_steps = metadata["integration"]["retained_steps"]
            raw_shape = [raw_time_steps, *metadata["grid"]["shape"]]
            raw_dtype = "<f4"
        else:
            raw_shape = metadata["clean_data"]["field_shape"]
            raw_dtype = metadata["clean_data"]["dtype"]
            raw_time_steps = metadata["sampling"]["n_time_steps"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Modern metadata is missing field_shape, dtype, or sampling.n_time_steps."
        ) from error

    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw_shape
        )
    ):
        raise ValueError(
            "clean_data.field_shape must contain exactly three integer dimensions."
        )
    expected_shape = tuple(raw_shape)

    try:
        expected_dtype = np.dtype(raw_dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid clean_data.dtype: {raw_dtype!r}.") from error

    if isinstance(raw_time_steps, bool) or not isinstance(raw_time_steps, int):
        raise TypeError("sampling.n_time_steps must be an integer.")
    if raw_time_steps != expected_shape[0]:
        raise ValueError(
            "sampling.n_time_steps and clean_data.field_shape disagree: "
            f"{raw_time_steps} versus {expected_shape[0]}."
        )

    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name}.npy has shape {array.shape}, but metadata declares "
                f"{expected_shape}."
            )
        if array.dtype != expected_dtype:
            raise ValueError(
                f"{name}.npy has dtype {array.dtype}, but metadata declares "
                f"{expected_dtype}."
            )


def load_prepared_data(
    data_dir: Path,
    vector_kind: str = "ocean-current",
) -> PreparedData:
    resolved = data_dir.expanduser().resolve()
    metadata = read_metadata(resolved)
    format_version = metadata_format_version(metadata)

    try:
        vector_spec = VECTOR_FIELDS[vector_kind]
    except KeyError as error:
        raise ValueError(
            f"Unknown vector kind {vector_kind!r}; choose from {tuple(VECTOR_FIELDS)}."
        ) from error

    if format_version == 4:
        validate_fresh_field_semantics(metadata)
        sst_name = "sst_anomaly"
        u_name, v_name = vector_spec["format_v3"]
    elif format_version == 3:
        validate_format_v3_field_semantics(metadata)
        sst_name = "sst_anomaly"
        u_name, v_name = vector_spec["format_v3"]
    else:
        sst_name = "sst"
        u_name, v_name = vector_spec["legacy"]
    vector_label = vector_spec["label"]

    sst = load_field(resolved, metadata, sst_name)
    u_component = load_field(resolved, metadata, u_name)
    v_component = load_field(resolved, metadata, v_name)

    if sst.shape != u_component.shape or sst.shape != v_component.shape:
        raise ValueError(
            f"{sst_name}, {u_name}, and {v_name} arrays must have identical shapes; "
            f"received {sst.shape}, {u_component.shape}, and {v_component.shape}."
        )
    if sst.ndim != 3:
        raise ValueError(
            "Expected arrays with shape (time, latitude, longitude); received "
            f"{sst.shape}."
        )
    if sst.shape[0] == 0:
        raise ValueError("The extracted arrays contain no time steps.")
    if any(array.dtype.kind != "f" for array in (sst, u_component, v_component)):
        raise ValueError("SST and vector arrays must have floating-point dtype.")

    if format_version >= 2:
        validate_structural_metadata(
            metadata,
            {sst_name: sst, u_name: u_component, v_name: v_component},
        )

    latitudes, longitudes = coordinates_from_metadata(metadata, sst.shape[1:])
    steps_per_month = positive_sampling_value(metadata, "steps_per_month")
    try:
        steps_per_year = positive_sampling_value(metadata, "steps_per_year")
    except ValueError:
        if format_version != 1:
            raise
        steps_per_year = 12.0 * steps_per_month
        LOGGER.warning(
            "sampling.steps_per_year is missing or invalid; using 12 times "
            "steps_per_month = %s.",
            steps_per_year,
        )
    expected_steps_per_year = 12.0 * steps_per_month
    if not math.isclose(
        steps_per_year,
        expected_steps_per_year,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "metadata sampling values disagree: steps_per_year must equal "
            f"12 * steps_per_month ({expected_steps_per_year:g}), received "
            f"{steps_per_year:g}."
        )

    return PreparedData(
        metadata=metadata,
        sst=sst,
        u_component=u_component,
        v_component=v_component,
        sst_field_name=sst_name,
        u_field_name=u_name,
        v_field_name=v_name,
        vector_label=vector_label,
        latitudes=latitudes,
        longitudes=longitudes,
        steps_per_month=steps_per_month,
        steps_per_year=steps_per_year,
    )


def _year_label(years: float) -> str:
    if float(years).is_integer():
        return str(int(years))
    return f"{years:.6g}".replace(".", "p")


def normalization_artifact_directory(
    input_profile: str,
    architecture: str,
    lead_months: int,
    train_years: float,
    seed: int,
) -> str:
    return str(
        Path("models")
        / input_profile
        / architecture
        / f"lead-{lead_months:02d}m"
        / f"years-{_year_label(train_years)}"
        / f"seed-{seed:06d}"
    )


def apply_fit_standardization(
    data: PreparedData,
    artifact_root: Path,
    *,
    input_profile: str,
    architecture: str,
    lead_months: int,
    train_years: float,
    seed: int,
    data_metadata_path: Path,
) -> PreparedData:
    """Wrap displayed fields in the selected checkpoint's lazy input transform."""

    if metadata_format_version(data.metadata) != 4:
        raise ValueError(
            "Checkpoint standardization requires the public fresh zc-v3 data set."
        )
    field_names = FRESH_INPUT_PROFILES[input_profile]
    required_fields = {
        data.sst_field_name,
        data.u_field_name,
        data.v_field_name,
    }
    missing_fields = sorted(required_fields.difference(field_names))
    if missing_fields:
        raise ValueError(
            f"Input profile {input_profile!r} does not contain the displayed "
            f"fields: {missing_fields!r}."
        )
    spatial_shape = tuple(int(value) for value in data.sst.shape[1:])
    spatial_size = int(np.prod(spatial_shape, dtype=np.int64))
    input_shape = (len(field_names) * spatial_size + 2,)
    relative_directory = normalization_artifact_directory(
        input_profile,
        architecture,
        lead_months,
        train_years,
        seed,
    )
    artifact = load_recorded_standardizer(
        artifact_root,
        relative_directory,
        expected_data_metadata_sha256=sha256_file(data_metadata_path),
        expected_input_shape=input_shape,
    )
    selected = {
        "input_profile": input_profile,
        "architecture": architecture,
        "lead_months": lead_months,
        "train_years": train_years,
        "seed": seed,
    }
    for key, expected in selected.items():
        actual = artifact.spec.get(key)
        if key == "train_years":
            matches = (
                not isinstance(actual, bool)
                and isinstance(actual, (int, float))
                and math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Selected normalization experiment has {key}={actual!r}, "
                f"not {expected!r}."
            )

    standardizer = artifact.standardizer
    spatial_mean = standardizer.mean[:-2].reshape(len(field_names), *spatial_shape)
    spatial_scale = standardizer.scale[:-2].reshape(len(field_names), *spatial_shape)

    def standardized(field: Any, name: str) -> StandardizedField:
        index = field_names.index(name)
        return StandardizedField(
            source=field,
            mean=spatial_mean[index],
            scale=spatial_scale[index],
        )

    return PreparedData(
        metadata=data.metadata,
        sst=standardized(data.sst, data.sst_field_name),
        u_component=standardized(data.u_component, data.u_field_name),
        v_component=standardized(data.v_component, data.v_field_name),
        sst_field_name=data.sst_field_name,
        u_field_name=data.u_field_name,
        v_field_name=data.v_field_name,
        # The colorbar and provenance identify the standardized coordinate
        # system.  Keeping the concise physical field name prevents the movie
        # title from running across the entire frame.
        vector_label=data.vector_label,
        latitudes=data.latitudes,
        longitudes=data.longitudes,
        steps_per_month=data.steps_per_month,
        steps_per_year=data.steps_per_year,
        normalization_artifact=artifact,
    )


def longitude_label(longitude_east: float) -> str:
    value = longitude_east % 360.0
    if math.isclose(value, 0.0, abs_tol=1.0e-9):
        return "0°"
    if math.isclose(value, 180.0, abs_tol=1.0e-9):
        return "180°E/W"
    if value < 180.0:
        return f"{value:g}°E"
    return f"{360.0 - value:g}°W"


def latitude_label(latitude: float) -> str:
    if math.isclose(latitude, 0.0, abs_tol=1.0e-9):
        return "0°"
    hemisphere = "N" if latitude > 0 else "S"
    return f"{abs(latitude):g}°{hemisphere}"


def coordinate_positions(coordinates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = np.arange(coordinates.size, dtype=np.float64)
    return np.interp(targets, coordinates, indices)


def choose_longitude_ticks(longitudes: np.ndarray) -> np.ndarray:
    first = math.ceil(float(longitudes[0]) / 30.0) * 30.0
    last = math.floor(float(longitudes[-1]) / 30.0) * 30.0
    if first <= last:
        return np.arange(first, last + 0.1, 30.0)
    return np.linspace(longitudes[0], longitudes[-1], num=3)


def choose_latitude_ticks(latitudes: np.ndarray) -> np.ndarray:
    preferred = np.asarray([-15.0, -5.0, 5.0, 15.0])
    inside = preferred[(preferred >= latitudes[0]) & (preferred <= latitudes[-1])]
    if inside.size >= 2:
        return inside
    return np.linspace(latitudes[0], latitudes[-1], num=5)


def configure_axes(
    ax: plt.Axes,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    aspect: str,
) -> None:
    n_lat = latitudes.size
    n_lon = longitudes.size
    ax.set_xlim(-0.5, n_lon - 0.5)
    ax.set_ylim(-0.5, n_lat - 0.5)

    lon_ticks = choose_longitude_ticks(longitudes)
    lat_ticks = choose_latitude_ticks(latitudes)
    ax.set_xticks(coordinate_positions(longitudes, lon_ticks))
    ax.set_xticklabels([longitude_label(value) for value in lon_ticks])
    ax.set_yticks(coordinate_positions(latitudes, lat_ticks))
    ax.set_yticklabels([latitude_label(value) for value in lat_ticks])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if aspect == "grid":
        ax.set_aspect("equal")
    else:
        latitude_spacing = float(np.median(np.diff(latitudes)))
        longitude_spacing = float(np.median(np.diff(longitudes)))
        ax.set_aspect(latitude_spacing / longitude_spacing)


def frame_steps(
    n_time_steps: int,
    *,
    start_step: int,
    requested_frames: int,
    stride: int,
) -> np.ndarray:
    if start_step >= n_time_steps:
        raise IndexError(
            f"--start-step={start_step} is outside a dataset with "
            f"{n_time_steps} time steps."
        )

    maximum_frames = ((n_time_steps - 1 - start_step) // stride) + 1
    n_frames = min(requested_frames, maximum_frames)
    if n_frames < requested_frames:
        LOGGER.warning(
            "Requested %s frames, but only %s are available.",
            f"{requested_frames:,}",
            f"{n_frames:,}",
        )
    return start_step + stride * np.arange(n_frames, dtype=np.int64)


def automatic_quiver_scale(
    data: PreparedData,
    *,
    steps: np.ndarray,
    quiver_stride: int,
) -> float:
    """Choose a stable arrow scale from nonzero vectors across the animation."""

    n_samples = min(AUTO_QUIVER_SAMPLE_FRAMES, steps.size)
    sample_positions = np.unique(
        np.linspace(0, steps.size - 1, num=n_samples, dtype=np.int64)
    )
    sample_steps = steps[sample_positions]
    vector_slice = slice(None, None, quiver_stride)
    u_sample = np.asarray(
        data.u_component[sample_steps, vector_slice, vector_slice],
        dtype=np.float64,
    )
    v_sample = np.asarray(
        data.v_component[sample_steps, vector_slice, vector_slice],
        dtype=np.float64,
    )
    magnitudes = np.hypot(u_sample, v_sample)
    usable = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]

    if usable.size == 0:
        LOGGER.warning(
            "All selected %s vectors are zero; using a harmless fixed arrow scale.",
            data.vector_label,
        )
        return 1.0

    reference = float(np.percentile(usable, AUTO_QUIVER_REFERENCE_PERCENTILE))
    scale = reference / AUTO_QUIVER_REFERENCE_WIDTH_FRACTION
    LOGGER.info(
        "Arrow scale: %.6g (%.0fth-percentile magnitude %.6g draws as %.1f%% "
        "of axes width)",
        scale,
        AUTO_QUIVER_REFERENCE_PERCENTILE,
        reference,
        100.0 * AUTO_QUIVER_REFERENCE_WIDTH_FRACTION,
    )
    return scale


def build_animation(
    data: PreparedData,
    *,
    steps: np.ndarray,
    interval_ms: int,
    quiver_stride: int,
    quiver_scale: float | None,
    sst_min: float,
    sst_max: float,
    aspect: str,
    interpolation: str,
    repeat: bool,
) -> tuple[plt.Figure, FuncAnimation, float]:
    """Build and return the figure and live animation object."""

    n_lat, n_lon = data.sst.shape[1:]
    x = np.arange(n_lon)
    y = np.arange(n_lat)
    grid_x, grid_y = np.meshgrid(x, y)
    vector_slice = (
        slice(None, None, quiver_stride),
        slice(None, None, quiver_stride),
    )
    resolved_quiver_scale = (
        quiver_scale
        if quiver_scale is not None
        else automatic_quiver_scale(
            data,
            steps=steps,
            quiver_stride=quiver_stride,
        )
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    first_step = int(steps[0])
    color_norm = TwoSlopeNorm(vmin=sst_min, vcenter=0.0, vmax=sst_max)
    image = ax.imshow(
        data.sst[first_step],
        origin="lower",
        interpolation=interpolation,
        cmap=COLORMAP,
        norm=color_norm,
        extent=(-0.5, n_lon - 0.5, -0.5, n_lat - 0.5),
    )
    # imshow applies its own default aspect, so configure the scientific axes
    # after creating the image.
    configure_axes(
        ax,
        data.latitudes,
        data.longitudes,
        aspect=aspect,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.03)
    colorbar.set_label("Standardized SST anomaly (fit SD units)")

    quiver = ax.quiver(
        grid_x[vector_slice],
        grid_y[vector_slice],
        data.u_component[first_step][vector_slice],
        data.v_component[first_step][vector_slice],
        angles="uv",
        pivot="middle",
        scale=resolved_quiver_scale,
        scale_units="width",
        units="width",
        color="black",
        width=0.0025,
    )
    title = ax.set_title("")

    def update_title(step: int) -> None:
        elapsed_months = step / data.steps_per_month
        elapsed_years = step / data.steps_per_year
        title.set_text(
            f"{data.vector_label}\n"
            f"Simulation time = {elapsed_years:.2f} years "
            f"({elapsed_months:.1f} months; step {step:,})"
        )

    update_title(first_step)

    def update(frame_number: int) -> tuple[Any, ...]:
        step = int(steps[frame_number])
        image.set_data(data.sst[step])
        quiver.set_UVC(
            data.u_component[step][vector_slice],
            data.v_component[step][vector_slice],
        )
        update_title(step)
        return image, quiver, title

    animation = FuncAnimation(
        fig,
        update,
        frames=steps.size,
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )
    # Keep a strong reference for GUI backends where plt.show() is nonblocking.
    fig._zc_animation = animation
    fig.tight_layout()
    return fig, animation, resolved_quiver_scale


def validated_output_path(
    output: Path | None,
    *,
    overwrite: bool,
) -> tuple[Path | None, str | None]:
    if output is None:
        return None, None

    path = output.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise ValueError(f"--output must end in .mp4 or .gif; received {path.name!r}.")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\nUse --overwrite to replace it."
        )

    writer_name = "ffmpeg" if suffix == ".mp4" else "pillow"
    if not writers.is_available(writer_name):
        dependency = "ffmpeg" if writer_name == "ffmpeg" else "Pillow"
        raise RuntimeError(
            f"Matplotlib cannot find the {writer_name!r} writer required for "
            f"{suffix} output. Install {dependency}, then rerun the command."
        )
    return path, writer_name


def save_animation_atomically(
    animation: FuncAnimation,
    output_path: Path,
    *,
    writer: FFMpegWriter | PillowWriter,
    dpi: int,
    overwrite: bool,
) -> None:
    """Save beside the destination, then atomically publish the complete file."""

    with atomic_output_path(output_path, overwrite=overwrite) as temporary_path:
        animation.save(temporary_path, writer=writer, dpi=dpi)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_path, writer_name = validated_output_path(
        args.output,
        overwrite=args.overwrite,
    )
    sidecar_path = output_path.with_suffix(".json") if output_path else None
    if sidecar_path is not None and sidecar_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Animation provenance already exists: {sidecar_path}\n"
            "Use --overwrite to replace it."
        )
    metadata_path = args.data_dir.expanduser().resolve() / "metadata.json"
    data = load_prepared_data(args.data_dir, vector_kind=args.vector_kind)
    data = apply_fit_standardization(
        data,
        args.artifacts_dir,
        input_profile=args.input_profile,
        architecture=args.architecture,
        lead_months=args.lead_months,
        train_years=args.train_years,
        seed=args.seed,
        data_metadata_path=metadata_path,
    )
    steps = frame_steps(
        data.sst.shape[0],
        start_step=args.start_step,
        requested_frames=args.frames,
        stride=args.stride,
    )
    model_year_span = (int(steps[-1]) - int(steps[0])) / data.steps_per_year
    if steps.size > 1:
        model_months_per_frame = (int(steps[1]) - int(steps[0])) / data.steps_per_month
    else:
        model_months_per_frame = 0.0
    LOGGER.info(
        "Animation selection: %s frames spanning %.2f model years "
        "(%.3g model months between frames)",
        f"{steps.size:,}",
        model_year_span,
        model_months_per_frame,
    )

    style = {
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.linewidth": 1.0,
    }
    with plt.rc_context(style):
        fig, animation, resolved_quiver_scale = build_animation(
            data,
            steps=steps,
            interval_ms=args.interval_ms,
            quiver_stride=args.quiver_stride,
            quiver_scale=args.quiver_scale,
            sst_min=args.sst_min,
            sst_max=args.sst_max,
            aspect=args.aspect,
            interpolation=args.interpolation,
            repeat=not args.no_repeat,
        )

        if output_path is None:
            LOGGER.info(
                "Displaying %s frames. If Spyder shows only a static image, "
                "select its Qt/Automatic graphics backend or save an MP4/GIF.",
                f"{steps.size:,}",
            )
            plt.show(block=True)
            return 0

        resolved_fps = args.fps
        if args.duration_seconds is not None:
            resolved_fps = steps.size / args.duration_seconds
            LOGGER.info(
                "Video duration %.1f seconds gives %.3f frames per second",
                args.duration_seconds,
                resolved_fps,
            )

        if writer_name == "ffmpeg":
            writer = FFMpegWriter(
                fps=resolved_fps,
                metadata={"artist": "Matplotlib"},
            )
        else:
            writer = PillowWriter(fps=resolved_fps)

        LOGGER.info("Saving %s frames to %s", f"{steps.size:,}", output_path)
        try:
            save_animation_atomically(
                animation,
                output_path,
                writer=writer,
                dpi=args.dpi,
                overwrite=args.overwrite,
            )
        finally:
            plt.close(fig)
        LOGGER.info("Animation saved: %s", output_path)
        artifact = data.normalization_artifact
        if artifact is None or sidecar_path is None:
            raise RuntimeError("Saved animation is missing normalization provenance.")
        standardizer = artifact.standardizer
        write_json(
            sidecar_path,
            {
                "schema_version": 1,
                "artifact": "standardized_zc_animation",
                "data": {
                    "metadata_file": metadata_path.name,
                    "metadata_sha256": sha256_file(metadata_path),
                    "schema_version": data.metadata.get("schema_version"),
                },
                "normalization": {
                    "coordinate_system": (
                        "featurewise z=(x-training_mean)/training_scale"
                    ),
                    "population": "fixed 10000-year training block only",
                    "artifact_directory": str(
                        artifact.artifact_directory.relative_to(
                            args.artifacts_dir.expanduser().resolve()
                        )
                    ),
                    "normalization_file": artifact.normalization_path.name,
                    "normalization_sha256": artifact.normalization_sha256,
                    "indices_sha256": artifact.indices_sha256,
                    "mean_sha256": sha256_array(standardizer.mean),
                    "scale_sha256": sha256_array(standardizer.scale),
                    "count": standardizer.count,
                    "completion_schema_version": (
                        artifact.completion_schema_version
                    ),
                    "completion_sha256": artifact.completion_sha256,
                    "generation_id": artifact.generation_id,
                    "experiment_spec": artifact.spec,
                },
                "fields": {
                    "shading": data.sst_field_name,
                    "zonal_vector": data.u_field_name,
                    "meridional_vector": data.v_field_name,
                    "vector_kind": args.vector_kind,
                },
                "frames": {
                    "start_step": int(steps[0]),
                    "stop_step": int(steps[-1]),
                    "count": int(steps.size),
                    "stride": args.stride,
                    "standardized_sst_limits": [args.sst_min, args.sst_max],
                    "quiver_stride": args.quiver_stride,
                    "quiver_scale": resolved_quiver_scale,
                    "quiver_scale_source": (
                        "explicit" if args.quiver_scale is not None else "automatic"
                    ),
                    "colormap": COLORMAP_NAME,
                    "colormap_lookup_table_size": COLORMAP_LOOKUP_TABLE_SIZE,
                },
                "output": {
                    "file": output_path.name,
                    "sha256": sha256_file(output_path),
                    "fps": resolved_fps,
                    "dpi": args.dpi,
                },
            },
            overwrite=args.overwrite,
        )
        LOGGER.info("Animation provenance saved: %s", sidecar_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
