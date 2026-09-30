"""Memory-efficient access to the release-oriented Zebiak-Cane arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from .io import load_json, sha256_file

CANONICAL_FIELDS = (
    "sst_anomaly",
    "zonal_wind_stress",
    "meridional_wind_stress",
    "zonal_surface_wind",
    "meridional_surface_wind",
    "thermocline_depth",
    "zonal_ocean_current",
    "meridional_ocean_current",
    "atmospheric_heating",
    "total_sst",
)

FRESH_PROCESS_FIELDS = (
    "zonal_mixed_layer_current",
    "meridional_mixed_layer_current",
    "upwelling_anomaly",
)
FRESH_INPUT_PROFILES = {
    "core4": (
        "sst_anomaly",
        "thermocline_depth",
        "zonal_ocean_current",
        "meridional_ocean_current",
    ),
    "core4_wp": (
        "sst_anomaly",
        "thermocline_depth",
        "zonal_ocean_current",
        "meridional_ocean_current",
        "upwelling_anomaly",
    ),
    "legacy10": CANONICAL_FIELDS,
    "all13": CANONICAL_FIELDS + FRESH_PROCESS_FIELDS,
}

SOURCE_TO_PUBLIC_FIELDS = (
    ("SST", "sst_anomaly"),
    ("TAUX", "zonal_wind_stress"),
    ("TAUY", "meridional_wind_stress"),
    ("UO", "zonal_surface_wind"),
    ("VO", "meridional_surface_wind"),
    ("H1", "thermocline_depth"),
    ("U1", "zonal_ocean_current"),
    ("V1", "meridional_ocean_current"),
    ("QF", "atmospheric_heating"),
    ("TT", "total_sst"),
)

EXPECTED_CLEAN_DTYPE = np.dtype("<f4")
EXPECTED_NINO3_LATITUDES = np.arange(-5.0, 5.1, 2.0, dtype=np.float64)
EXPECTED_NINO3_LONGITUDES = 101.25 + 5.625 * np.arange(20, 31, dtype=np.float64)
EXPECTED_NINO3_TARGET_DEFINITION_VERSION = "center-inclusive-conventional-nino3-v1"
FRESH_SCHEMA_VERSION = "fresh-zc-interpretability-v1"
FRESH_LATITUDES = np.arange(-19.0, 20.0, 2.0, dtype=np.float64)
FRESH_LONGITUDES = 129.375 + 5.625 * np.arange(27, dtype=np.float64)
FRESH_PHASE_FORMULA = (
    "angle=2*pi*mod(native_time_months-0.5,12)/12; columns sin(angle),cos(angle)"
)
FRESH_PHASE_STORAGE = "two scalars per time, never duplicated over the spatial grid"
FRESH_NINO3_DEFINITION = (
    "unweighted mean of model-grid SST-anomaly centers inside or on 5S-5N, 150W-90W"
)
FRESH_SPLIT_SEMANTICS = (
    "half-open chronological indices before lead-specific pair construction"
)
FRESH_NORMALIZATION_SEMANTICS = (
    "fit means/scales for every selected spatial coordinate and both phase "
    "scalars on all raw states in the 10000-year train block only"
)
FRESH_FIELD_IDENTITIES = (
    ("sst_anomaly", "TO", "core"),
    ("zonal_wind_stress", "HTAU(:,:,1)", "legacy"),
    ("meridional_wind_stress", "HTAU(:,:,2)", "legacy"),
    ("zonal_surface_wind", "UO", "legacy"),
    ("meridional_surface_wind", "VO", "legacy"),
    ("thermocline_depth", "H1", "core"),
    ("zonal_ocean_current", "U1", "core"),
    ("meridional_ocean_current", "V1", "core"),
    ("atmospheric_heating", "QF", "legacy"),
    ("total_sst", "TT", "legacy"),
    ("zonal_mixed_layer_current", "US", "process"),
    ("meridional_mixed_layer_current", "VS", "process"),
    ("upwelling_anomaly", "WP", "process"),
)


def _metadata_error(message: str) -> NoReturn:
    raise ValueError(f"Invalid format-version-3 processed-data metadata: {message}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _metadata_error(f"{label} must be a JSON object.")
    return value


def _required_mapping(parent: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    if key not in parent:
        _metadata_error(f"{label}.{key} is required.")
    return _mapping(parent[key], f"{label}.{key}")


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _metadata_error(f"{label} must be a positive integer.")
    return value


def _integer_pair(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        _metadata_error(f"{label} must contain exactly two integers.")
    return int(value[0]), int(value[1])


def _positive_shape(value: Any, length: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        _metadata_error(f"{label} must contain exactly {length} dimensions.")
    return tuple(
        _positive_integer(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _float_coordinates(value: Any, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        _metadata_error(f"{label} must contain exactly {length} coordinates.")
    try:
        coordinates = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        _metadata_error(f"{label} must contain numeric coordinates ({error}).")
    if not np.isfinite(coordinates).all():
        _metadata_error(f"{label} contains a nonfinite coordinate.")
    if coordinates.size > 1 and not np.all(np.diff(coordinates) > 0.0):
        _metadata_error(f"{label} must be strictly increasing.")
    return coordinates


def _safe_basename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _metadata_error(f"{label} must be a nonempty filename string.")
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).is_absolute()
        or Path(value).name != value
    ):
        _metadata_error(f"{label} must be a safe relative basename, not {value!r}.")
    return value


def _safe_npy_basename(value: Any, label: str) -> str:
    value = _safe_basename(value, label)
    if Path(value).suffix != ".npy":
        _metadata_error(f"{label} must name a .npy file, not {value!r}.")
    return value


def _sha256_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        _metadata_error(f"{label} must be a 64-character hexadecimal SHA-256.")
    return value.lower()


def _validate_source_provenance(metadata: dict[str, Any]) -> None:
    source = _required_mapping(metadata, "source", "metadata")
    if "file_name" in source:
        _safe_basename(source["file_name"], "source.file_name")
    generator_value = metadata.get("generator")
    if generator_value is not None:
        generator = _mapping(generator_value, "generator")
        if "script" in generator:
            _safe_basename(generator["script"], "generator.script")
    expected_sources = tuple(source_name for source_name, _ in SOURCE_TO_PUBLIC_FIELDS)
    evidence_found = False

    if "field_order" in source:
        field_order = source["field_order"]
        if not isinstance(field_order, list) or tuple(field_order) != expected_sources:
            _metadata_error(
                "source.field_order must be "
                f"{list(expected_sources)!r}; this order distinguishes UO/VO "
                "surface winds from U1/V1 ocean currents."
            )
        evidence_found = True

    if "field_records" in source:
        records = source["field_records"]
        if not isinstance(records, list) or len(records) != len(
            SOURCE_TO_PUBLIC_FIELDS
        ):
            _metadata_error(
                "source.field_records must contain "
                f"{len(SOURCE_TO_PUBLIC_FIELDS)} records."
            )
        by_number: dict[int, dict[str, Any]] = {}
        for index, value in enumerate(records):
            record = _mapping(value, f"source.field_records[{index}]")
            number = record.get("record_number_within_time_step")
            if isinstance(number, bool) or not isinstance(number, int):
                _metadata_error(
                    "each source.field_records entry needs an integer "
                    "record_number_within_time_step."
                )
            if number in by_number:
                _metadata_error(f"source.field_records repeats record number {number}.")
            by_number[number] = record
        for number, (source_name, public_name) in enumerate(
            SOURCE_TO_PUBLIC_FIELDS,
            start=1,
        ):
            record = by_number.get(number)
            if record is None:
                _metadata_error(
                    f"source.field_records is missing record number {number}."
                )
            actual = (record.get("source_name"), record.get("public_name"))
            if actual != (source_name, public_name):
                _metadata_error(
                    f"source record {number} must map {source_name} to {public_name}; "
                    f"metadata instead maps {actual[0]!r} to {actual[1]!r}."
                )
        evidence_found = True

    if "fields" in metadata:
        fields = metadata["fields"]
        if not isinstance(fields, list) or len(fields) != len(SOURCE_TO_PUBLIC_FIELDS):
            _metadata_error(
                f"fields must contain {len(SOURCE_TO_PUBLIC_FIELDS)} descriptors."
            )
        descriptors: dict[int, dict[str, Any]] = {}
        for index, value in enumerate(fields):
            descriptor = _mapping(value, f"fields[{index}]")
            number = descriptor.get("source_record_number")
            if isinstance(number, bool) or not isinstance(number, int):
                _metadata_error(
                    "each fields entry needs an integer source_record_number."
                )
            if number in descriptors:
                _metadata_error(f"fields repeats source record number {number}.")
            descriptors[number] = descriptor
        for number, (source_name, public_name) in enumerate(
            SOURCE_TO_PUBLIC_FIELDS,
            start=1,
        ):
            descriptor = descriptors.get(number)
            if descriptor is None:
                _metadata_error(f"fields is missing source record number {number}.")
            actual = (descriptor.get("source_name"), descriptor.get("name"))
            if actual != (source_name, public_name):
                _metadata_error(
                    f"fields record {number} must identify {source_name} as "
                    f"{public_name}; found {actual!r}."
                )
        evidence_found = True

    if not evidence_found:
        _metadata_error(
            "source provenance must include canonical field_order, field_records, "
            "or fields descriptors."
        )

    semantics_value = metadata.get("field_semantics")
    if semantics_value is not None:
        semantics = _mapping(semantics_value, "field_semantics")
        expected_semantics = {
            "atmospheric_surface_wind_source_variables": ["UO", "VO"],
            "depth_averaged_ocean_current_source_variables": ["U1", "V1"],
        }
        for key, expected in expected_semantics.items():
            if semantics.get(key) != expected:
                _metadata_error(
                    f"field_semantics.{key} must be {expected!r}; "
                    f"found {semantics.get(key)!r}."
                )
        semantic_filenames = semantics.get(
            "public_filenames_are_semantic_not_raw_source_names"
        )
        if semantic_filenames is not None and semantic_filenames is not True:
            _metadata_error(
                "field_semantics.public_filenames_are_semantic_not_raw_source_names "
                "must be true when present."
            )


def _validate_grid(
    metadata: dict[str, Any],
    field_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    grid = _required_mapping(metadata, "grid", "metadata")
    active = _required_mapping(grid, "active_domain", "grid")
    active_shape = _positive_shape(active.get("shape"), 2, "grid.active_domain.shape")
    if active_shape != field_shape[1:]:
        _metadata_error(
            "grid.active_domain.shape does not match clean_data.field_shape's "
            f"spatial dimensions: {active_shape!r} versus {field_shape[1:]!r}."
        )
    latitudes = _float_coordinates(
        active.get("latitude_degrees_north"),
        active_shape[0],
        "grid.active_domain.latitude_degrees_north",
    )
    longitudes = _float_coordinates(
        active.get("longitude_degrees_east"),
        active_shape[1],
        "grid.active_domain.longitude_degrees_east",
    )

    row_slice = active.get("source_row_slice")
    column_slice = active.get("source_column_slice")
    parsed_row_slice = None
    parsed_column_slice = None
    if row_slice is not None:
        parsed_row_slice = _integer_pair(
            row_slice, "grid.active_domain.source_row_slice"
        )
        if (
            parsed_row_slice[0] < 0
            or parsed_row_slice[1] - parsed_row_slice[0] != active_shape[0]
        ):
            _metadata_error(
                "grid.active_domain.source_row_slice is inconsistent with its shape."
            )
    if column_slice is not None:
        parsed_column_slice = _integer_pair(
            column_slice,
            "grid.active_domain.source_column_slice",
        )
        if (
            parsed_column_slice[0] < 0
            or parsed_column_slice[1] - parsed_column_slice[0] != active_shape[1]
        ):
            _metadata_error(
                "grid.active_domain.source_column_slice is inconsistent with its shape."
            )

    raw_value = grid.get("raw")
    raw = None if raw_value is None else _mapping(raw_value, "grid.raw")
    if raw is not None:
        raw_shape = _positive_shape(raw.get("shape"), 2, "grid.raw.shape")
        raw_latitudes = _float_coordinates(
            raw.get("latitude_degrees_north"),
            raw_shape[0],
            "grid.raw.latitude_degrees_north",
        )
        raw_longitudes = _float_coordinates(
            raw.get("longitude_degrees_east"),
            raw_shape[1],
            "grid.raw.longitude_degrees_east",
        )
        if parsed_row_slice is not None:
            start, stop = parsed_row_slice
            if stop > raw_shape[0] or not np.array_equal(
                latitudes,
                raw_latitudes[start:stop],
            ):
                _metadata_error(
                    "active latitude coordinates do not equal the declared "
                    "raw-grid slice."
                )
        if parsed_column_slice is not None:
            start, stop = parsed_column_slice
            if stop > raw_shape[1] or not np.array_equal(
                longitudes,
                raw_longitudes[start:stop],
            ):
                _metadata_error(
                    "active longitude coordinates do not equal the declared "
                    "raw-grid slice."
                )
    return latitudes, longitudes, raw


def _validate_nino3_metadata(
    metadata: dict[str, Any],
    active_latitudes: np.ndarray,
    active_longitudes: np.ndarray,
    raw_grid: dict[str, Any] | None,
) -> None:
    value = metadata.get("nino3")
    if value is None:
        _metadata_error(
            "metadata.nino3 is required so the forecast-target definition can "
            "be validated."
        )
    nino3 = _mapping(value, "nino3")
    if (
        nino3.get("target_definition_version")
        != EXPECTED_NINO3_TARGET_DEFINITION_VERSION
    ):
        _metadata_error(
            "nino3.target_definition_version must be "
            f"{EXPECTED_NINO3_TARGET_DEFINITION_VERSION!r}; migrate the old "
            "west-shifted target with scripts/migrate_nino3_target.py."
        )
    if nino3.get("source_field") != "sst_anomaly":
        _metadata_error("nino3.source_field must be 'sst_anomaly'.")
    row_slice = _integer_pair(nino3.get("source_row_slice"), "nino3.source_row_slice")
    column_slice = _integer_pair(
        nino3.get("source_column_slice"),
        "nino3.source_column_slice",
    )
    if row_slice != (12, 18) or column_slice != (20, 31):
        _metadata_error(
            "the center-inclusive Nino-3 target must use raw row slice [12, 18] "
            "and column slice [20, 31]."
        )
    centers = _required_mapping(nino3, "selected_grid_centers", "nino3")
    latitudes = _float_coordinates(
        centers.get("latitude_degrees_north"),
        EXPECTED_NINO3_LATITUDES.size,
        "nino3.selected_grid_centers.latitude_degrees_north",
    )
    longitudes = _float_coordinates(
        centers.get("longitude_degrees_east"),
        EXPECTED_NINO3_LONGITUDES.size,
        "nino3.selected_grid_centers.longitude_degrees_east",
    )
    if latitudes.size * longitudes.size != 66:
        _metadata_error(
            "the center-inclusive Nino-3 target must select exactly 66 grid centers."
        )
    if not np.array_equal(latitudes, EXPECTED_NINO3_LATITUDES) or not np.array_equal(
        longitudes,
        EXPECTED_NINO3_LONGITUDES,
    ):
        _metadata_error(
            "nino3.selected_grid_centers does not match the center-inclusive "
            "6 x 11 Nino-3 grid (213.75 E through 270 E)."
        )
    if (
        not np.isin(latitudes, active_latitudes).all()
        or not np.isin(
            longitudes,
            active_longitudes,
        ).all()
    ):
        _metadata_error(
            "the center-inclusive Nino-3 centers must lie on the active grid."
        )
    if raw_grid is not None:
        raw_shape = _positive_shape(raw_grid.get("shape"), 2, "grid.raw.shape")
        raw_latitudes = _float_coordinates(
            raw_grid.get("latitude_degrees_north"),
            raw_shape[0],
            "grid.raw.latitude_degrees_north",
        )
        raw_longitudes = _float_coordinates(
            raw_grid.get("longitude_degrees_east"),
            raw_shape[1],
            "grid.raw.longitude_degrees_east",
        )
        if not np.array_equal(
            latitudes,
            raw_latitudes[slice(*row_slice)],
        ) or not np.array_equal(
            longitudes,
            raw_longitudes[slice(*column_slice)],
        ):
            _metadata_error(
                "center-inclusive Nino-3 centers do not equal the declared "
                "raw-grid slices."
            )

    conventional_value = nino3.get("nominal_conventional_region")
    if conventional_value is not None:
        conventional = _mapping(
            conventional_value,
            "nino3.nominal_conventional_region",
        )
        expected_bounds = {
            "latitude_bounds_degrees_north": [-5.0, 5.0],
            "longitude_bounds_degrees_east": [210.0, 270.0],
            "longitude_bounds_degrees_west": [150.0, 90.0],
        }
        for key, expected in expected_bounds.items():
            if conventional.get(key) != expected:
                _metadata_error(
                    f"nino3.nominal_conventional_region.{key} must be {expected!r}."
                )


def _validate_manifest(
    clean: dict[str, Any],
    paths: dict[str, Path],
    *,
    verify_checksums: bool,
) -> None:
    manifest_value = clean.get("files")
    if manifest_value is None:
        if verify_checksums:
            _metadata_error("clean_data.files is required when verify_checksums=True.")
        return
    manifest = _mapping(manifest_value, "clean_data.files")
    missing = sorted(set(paths) - set(manifest))
    if missing:
        _metadata_error(f"clean_data.files is missing entries for {missing!r}.")
    extra = sorted(set(manifest) - set(paths))
    if extra:
        _metadata_error(f"clean_data.files has unexpected entries: {extra!r}.")
    for filename, path in paths.items():
        entry = _mapping(manifest[filename], f"clean_data.files[{filename!r}]")
        recorded_size = entry.get("size_bytes")
        if isinstance(recorded_size, bool) or not isinstance(recorded_size, int):
            _metadata_error(
                f"clean_data.files[{filename!r}].size_bytes must be an integer."
            )
        actual_size = path.stat().st_size
        if recorded_size != actual_size:
            _metadata_error(
                f"{filename!r} has {actual_size} bytes, not the recorded "
                f"{recorded_size} bytes."
            )
        digest = entry.get("sha256")
        if digest is not None:
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef" for character in digest.lower()
                )
            ):
                _metadata_error(
                    f"clean_data.files[{filename!r}].sha256 is not a SHA-256 digest."
                )
            if verify_checksums and sha256_file(path) != digest.lower():
                raise ValueError(f"Checksum mismatch for processed-data file: {path}")
        elif verify_checksums:
            _metadata_error(
                f"no SHA-256 digest is recorded for {filename!r}; cannot verify it."
            )


@dataclass(frozen=True)
class Standardizer:
    """Feature-wise training statistics with a recorded numerical floor."""

    mean: np.ndarray
    scale: np.ndarray
    scale_floor: float
    count: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


@dataclass(frozen=True)
class SupervisedSplit:
    """Chronological forecast input indices that never cross the split boundary."""

    train_inputs: np.ndarray
    test_inputs: np.ndarray
    split_step: int
    lead_steps: int


@dataclass(frozen=True)
class FixedSupervisedSplit:
    """Canonical 10,000/1,000/1,000-year forecast-pair blocks."""

    train_inputs: np.ndarray
    validation_inputs: np.ndarray
    test_inputs: np.ndarray
    lead_steps: int
    train_block: tuple[int, int]
    validation_block: tuple[int, int]
    test_block: tuple[int, int]


@dataclass(frozen=True)
class DevelopmentSelection:
    """A reproducible window with chronological fit, embargo, and validation blocks."""

    development_inputs: np.ndarray
    fit_inputs: np.ndarray
    embargo_inputs: np.ndarray
    validation_inputs: np.ndarray
    validation_embargo_steps: int
    requested_years: float
    actual_years: float
    window_start_step: int
    window_stop_step_exclusive: int


class ZCData:
    """Validated memory maps for one format-version-3 processed dataset.

    File sizes and NumPy headers are always checked. Large array payloads are
    hashed only when ``verify_checksums=True``; that opt-in requires a recorded
    SHA-256 digest for every field and the target.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        verify_checksums: bool = False,
        input_profile: str | None = None,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Processed-data metadata not found: {metadata_path}"
            )
        self.metadata = load_json(metadata_path)
        if self.metadata.get("schema_version") == FRESH_SCHEMA_VERSION:
            self._initialize_fresh(
                metadata_path,
                verify_checksums=verify_checksums,
                input_profile=input_profile or "core4",
            )
            return
        if self.metadata.get("format_version") != 3:
            raise ValueError(
                "These experiments require format-version-3 processed data. "
                "Rerun prepare_zc_data.py with the current extractor."
            )

        _validate_source_provenance(self.metadata)

        clean = _required_mapping(self.metadata, "clean_data", "metadata")
        dtype_value = clean.get("dtype")
        try:
            metadata_dtype = np.dtype(dtype_value)
        except (TypeError, ValueError) as error:
            _metadata_error(f"clean_data.dtype is invalid ({error}).")
        if metadata_dtype.str != EXPECTED_CLEAN_DTYPE.str:
            _metadata_error(
                f"clean_data.dtype must be {EXPECTED_CLEAN_DTYPE.str!r}, "
                f"not {dtype_value!r}."
            )
        layout = clean.get("array_layout")
        if layout is not None and layout != ["time", "latitude", "longitude"]:
            _metadata_error(
                "clean_data.array_layout must be ['time', 'latitude', 'longitude']."
            )
        expected_shape = _positive_shape(
            clean.get("field_shape"),
            3,
            "clean_data.field_shape",
        )

        field_files = _required_mapping(clean, "field_files", "clean_data")
        missing_names = [name for name in CANONICAL_FIELDS if name not in field_files]
        if missing_names:
            _metadata_error(
                f"clean_data.field_files is missing fields: {missing_names!r}."
            )
        extra_names = sorted(set(field_files) - set(CANONICAL_FIELDS))
        if extra_names:
            _metadata_error(
                f"clean_data.field_files has unexpected fields: {extra_names!r}."
            )

        filenames = {
            name: _safe_npy_basename(
                field_files[name],
                f"clean_data.field_files[{name!r}]",
            )
            for name in CANONICAL_FIELDS
        }
        target_filename = _safe_npy_basename(
            clean.get("nino3_file"),
            "clean_data.nino3_file",
        )
        all_filenames = [*filenames.values(), target_filename]
        if len(set(all_filenames)) != len(all_filenames):
            _metadata_error(
                "every processed field and target must use a distinct file."
            )
        canonical_filename_owners = {
            f"{field_name}.npy": field_name for field_name in CANONICAL_FIELDS
        }
        for field_name, filename in filenames.items():
            other_owner = canonical_filename_owners.get(filename)
            if other_owner is not None and other_owner != field_name:
                _metadata_error(
                    f"field {field_name!r} cannot point to the canonical file for "
                    f"{other_owner!r}: {filename!r}."
                )

        active_latitudes, active_longitudes, raw_grid = _validate_grid(
            self.metadata,
            expected_shape,
        )
        _validate_nino3_metadata(
            self.metadata,
            active_latitudes,
            active_longitudes,
            raw_grid,
        )

        source = _required_mapping(self.metadata, "source", "metadata")
        sampling = _required_mapping(self.metadata, "sampling", "metadata")
        for container, label in ((source, "source"), (sampling, "sampling")):
            recorded_steps = container.get("n_time_steps")
            if recorded_steps is not None:
                recorded_steps = _positive_integer(
                    recorded_steps,
                    f"{label}.n_time_steps",
                )
                if recorded_steps != expected_shape[0]:
                    _metadata_error(
                        f"{label}.n_time_steps does not match clean_data.field_shape."
                    )
        steps_per_month = _positive_integer(
            sampling.get("steps_per_month"),
            "sampling.steps_per_month",
        )
        steps_per_year = _positive_integer(
            sampling.get("steps_per_year"),
            "sampling.steps_per_year",
        )
        if steps_per_year != 12 * steps_per_month:
            _metadata_error(
                "sampling.steps_per_year must equal 12 * sampling.steps_per_month."
            )

        paths = {filename: self.data_dir / filename for filename in all_filenames}
        missing_files = [str(path) for path in paths.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "Processed-data files referenced by metadata were not found: "
                + ", ".join(missing_files)
            )
        _validate_manifest(clean, paths, verify_checksums=verify_checksums)

        self.field_names = CANONICAL_FIELDS
        self.arrays = tuple(
            np.load(
                paths[filenames[name]],
                mmap_mode="r",
                allow_pickle=False,
            )
            for name in self.field_names
        )
        self.target = np.load(
            paths[target_filename],
            mmap_mode="r",
            allow_pickle=False,
        )
        if any(array.shape != expected_shape for array in self.arrays):
            raise ValueError("At least one processed field has an unexpected shape.")
        if self.target.shape != (expected_shape[0],):
            raise ValueError(
                "The Nino-3 target length does not match the input fields."
            )
        if any(array.dtype.str != EXPECTED_CLEAN_DTYPE.str for array in self.arrays):
            raise ValueError("Processed fields must be little-endian float32 arrays.")
        if self.target.dtype.str != EXPECTED_CLEAN_DTYPE.str:
            raise ValueError("The Nino-3 target must be a little-endian float32 array.")

        self.n_time_steps, self.n_latitudes, self.n_longitudes = expected_shape
        self.n_fields = len(self.field_names)
        self.spatial_field_names = self.field_names
        self.spatial_input_shape = (
            self.n_fields,
            self.n_latitudes,
            self.n_longitudes,
        )
        self.n_phase_features = 0
        self.input_shape = (self.n_fields, self.n_latitudes, self.n_longitudes)
        self.n_features = int(np.prod(self.input_shape))
        self.steps_per_month = steps_per_month
        self.steps_per_year = steps_per_year
        self.latitudes = active_latitudes
        self.longitudes = active_longitudes
        self.metadata_sha256 = sha256_file(metadata_path)
        self.input_profile = "legacy10"
        self.checksum_verified = bool(verify_checksums)

    def _initialize_fresh(
        self,
        metadata_path: Path,
        *,
        verify_checksums: bool,
        input_profile: str,
    ) -> None:
        """Initialize the interpretation-first 13-field release format."""

        if input_profile not in FRESH_INPUT_PROFILES:
            raise ValueError(
                f"Unknown fresh-data input profile {input_profile!r}; choose from "
                f"{sorted(FRESH_INPUT_PROFILES)!r}."
            )
        integration = _required_mapping(self.metadata, "integration", "metadata")
        grid = _required_mapping(self.metadata, "grid", "metadata")
        field_files = _required_mapping(self.metadata, "field_files", "metadata")
        phase = _required_mapping(self.metadata, "phase", "metadata")
        nino3 = _required_mapping(self.metadata, "nino3", "metadata")
        split = _required_mapping(self.metadata, "chronological_split", "metadata")
        retained_steps = _positive_integer(
            integration.get("retained_steps"), "integration.retained_steps"
        )
        steps_per_month = _positive_integer(
            integration.get("steps_per_month"), "integration.steps_per_month"
        )
        steps_per_year = _positive_integer(
            integration.get("steps_per_year"), "integration.steps_per_year"
        )
        if steps_per_year != 12 * steps_per_month:
            _metadata_error(
                "integration.steps_per_year must equal 12 times "
                "integration.steps_per_month."
            )
        is_fixture = split.get("testing_fixture") is True
        if retained_steps != 432_000 and not is_fixture:
            _metadata_error(
                "non-production fresh datasets must explicitly set "
                "chronological_split.testing_fixture to true."
            )
        if retained_steps == 432_000:
            expected_integration = {
                "retained_years": 12_000,
                "spinup_years_discarded": 100,
                "steps_per_month": 3,
                "steps_per_year": 36,
            }
            for key, expected in expected_integration.items():
                if integration.get(key) != expected:
                    _metadata_error(
                        f"integration.{key} must be {expected!r} for zc-v3."
                    )
        native_time_filename = _safe_npy_basename(
            integration.get("native_time_file"),
            "integration.native_time_file",
        )
        if native_time_filename != "native_time_months.npy":
            _metadata_error(
                "integration.native_time_file must be native_time_months.npy."
            )
        if (
            integration.get("native_time_units")
            != "months; 0.5 is mid-January of nominal year 1960"
        ):
            _metadata_error("integration.native_time_units has changed semantics.")

        shape = _positive_shape(grid.get("shape"), 2, "grid.shape")
        if shape != (20, 27):
            _metadata_error("fresh grid.shape must be [20, 27].")
        latitudes = _float_coordinates(
            grid.get("latitude_degrees_north"), shape[0], "grid.latitude_degrees_north"
        )
        longitudes = _float_coordinates(
            grid.get("longitude_degrees_east"), shape[1], "grid.longitude_degrees_east"
        )
        if not np.array_equal(latitudes, FRESH_LATITUDES):
            _metadata_error(
                "fresh latitude centers must be -19 through 19 by 2 degrees."
            )
        if not np.array_equal(longitudes, FRESH_LONGITUDES):
            _metadata_error(
                "fresh longitude centers must be 129.375E through 275.625E "
                "by 5.625 degrees."
            )
        expected_fortran_grid = {
            "ordinary_fields": {
                "latitude": [25, 6, -1],
                "longitude": [6, 32, 1],
            },
            "HTAU_latitude_mapping": "HTAU longitude J, latitude 31-I",
        }
        if grid.get("active_fortran_indices_one_based") != expected_fortran_grid:
            _metadata_error("fresh active Fortran grid mapping is not canonical.")

        all_names = CANONICAL_FIELDS + FRESH_PROCESS_FIELDS
        if set(field_files) != set(all_names):
            _metadata_error(
                "fresh field_files must contain exactly the ten legacy and "
                "three process fields."
            )
        descriptors = self.metadata.get("fields")
        if not isinstance(descriptors, list) or len(descriptors) != len(all_names):
            _metadata_error("fresh fields must contain exactly 13 descriptors.")
        for position, (name, source, group) in enumerate(FRESH_FIELD_IDENTITIES):
            descriptor = _mapping(descriptors[position], f"fields[{position}]")
            if (
                descriptor.get("name"),
                descriptor.get("fortran"),
                descriptor.get("group"),
            ) != (name, source, group):
                _metadata_error(
                    f"fields[{position}] must map {source} to {name} in group {group}."
                )
            for text_key in ("role", "units"):
                if (
                    not isinstance(descriptor.get(text_key), str)
                    or not descriptor[text_key]
                ):
                    _metadata_error(
                        f"fields[{position}].{text_key} must be a nonempty string."
                    )

        paths: dict[str, Path] = {}
        recorded_digests: dict[Path, str] = {}
        for name in all_names:
            entry = _required_mapping(field_files, name, "field_files")
            filename = _safe_npy_basename(entry.get("file"), f"field_files.{name}.file")
            if filename != f"{name}.npy":
                _metadata_error(
                    f"field_files.{name}.file must be the semantic filename {name}.npy."
                )
            path = self.data_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            recorded_size = entry.get("size_bytes")
            if recorded_size != path.stat().st_size:
                _metadata_error(
                    f"field_files.{name}.size_bytes does not match {filename}."
                )
            recorded_digests[path] = _sha256_digest(
                entry.get("sha256"), f"field_files.{name}.sha256"
            )
            paths[name] = path

        phase_path = self.data_dir / _safe_npy_basename(phase.get("file"), "phase.file")
        target_path = self.data_dir / _safe_npy_basename(
            nino3.get("file"), "nino3.file"
        )
        if phase_path.name != "annual_phase_sin_cos.npy":
            _metadata_error("phase.file must be annual_phase_sin_cos.npy.")
        if target_path.name != "nino3_index.npy":
            _metadata_error("nino3.file must be nino3_index.npy.")
        if not phase_path.is_file() or not target_path.is_file():
            raise FileNotFoundError("Fresh phase or Nino-3 file is missing.")
        if phase.get("shape") != [retained_steps, 2]:
            _metadata_error("phase.shape must be [retained_steps, 2].")
        if phase.get("formula") != FRESH_PHASE_FORMULA:
            _metadata_error("phase.formula does not match the annual phase definition.")
        if phase.get("storage") != FRESH_PHASE_STORAGE:
            _metadata_error("phase.storage must declare two non-spatial scalars.")

        expected_nino3 = {
            "definition": FRESH_NINO3_DEFINITION,
            "grid_point_count": 66,
            "includes_270E_90W_center": True,
            "latitude_degrees_north": EXPECTED_NINO3_LATITUDES.tolist(),
            "longitude_degrees_east": EXPECTED_NINO3_LONGITUDES.tolist(),
        }
        for key, expected in expected_nino3.items():
            if nino3.get(key) != expected:
                _metadata_error(f"nino3.{key} does not match the canonical definition.")

        expected_auxiliary = {
            phase_path.name,
            target_path.name,
            native_time_filename,
        }
        auxiliary = _required_mapping(self.metadata, "auxiliary_files", "metadata")
        if set(auxiliary) != expected_auxiliary:
            _metadata_error(
                "auxiliary_files must contain exactly phase, Nino-3, and native time."
            )
        auxiliary_paths: dict[str, Path] = {}
        for filename in sorted(expected_auxiliary):
            _safe_npy_basename(filename, "auxiliary_files key")
            path = self.data_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            entry = _mapping(auxiliary[filename], f"auxiliary_files.{filename}")
            if entry.get("size_bytes") != path.stat().st_size:
                _metadata_error(
                    f"auxiliary_files.{filename}.size_bytes does not match the file."
                )
            recorded_digests[path] = _sha256_digest(
                entry.get("sha256"), f"auxiliary_files.{filename}.sha256"
            )
            auxiliary_paths[filename] = path

        blocks = {
            name: _integer_pair(split.get(name), f"chronological_split.{name}")
            for name in ("train", "validation", "test")
        }
        for name, (start, stop) in blocks.items():
            if start < 0 or stop > retained_steps or stop <= start:
                _metadata_error(f"chronological_split.{name} is outside the data set.")
        if not (
            blocks["train"][0] == 0
            and blocks["train"][1] == blocks["validation"][0]
            and blocks["validation"][1] == blocks["test"][0]
            and blocks["test"][1] == retained_steps
        ):
            _metadata_error(
                "chronological split blocks must be contiguous, nonoverlapping, "
                "and cover every retained step."
            )
        if retained_steps == 432_000 and blocks != {
            "train": (0, 360_000),
            "validation": (360_000, 396_000),
            "test": (396_000, 432_000),
        }:
            _metadata_error("zc-v3 must use the canonical 10000/1000/1000-year split.")
        if split.get("semantics") != FRESH_SPLIT_SEMANTICS:
            _metadata_error("chronological_split.semantics has changed.")
        if split.get("normalization") != FRESH_NORMALIZATION_SEMANTICS:
            _metadata_error("chronological_split.normalization has changed.")

        arrays_by_name: dict[str, np.ndarray] = {}
        expected_shape = (retained_steps, *shape)
        for name, path in paths.items():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                array.shape != expected_shape
                or array.dtype.str != EXPECTED_CLEAN_DTYPE.str
            ):
                raise ValueError(
                    f"Fresh field {name} must have shape {expected_shape} "
                    "and dtype <f4."
                )
            arrays_by_name[name] = array
        selected_names = FRESH_INPUT_PROFILES[input_profile]
        self.arrays = tuple(arrays_by_name[name] for name in selected_names)
        self.phase = np.load(phase_path, mmap_mode="r", allow_pickle=False)
        self.target = np.load(target_path, mmap_mode="r", allow_pickle=False)
        if self.phase.shape != (retained_steps, 2):
            raise ValueError("Fresh annual phase must have shape (time, 2).")
        if self.target.shape != (retained_steps,):
            raise ValueError("Fresh Nino-3 target must have shape (time,).")
        if self.phase.dtype.str != EXPECTED_CLEAN_DTYPE.str:
            raise ValueError("Fresh annual phase must be little-endian float32.")
        if self.target.dtype.str != EXPECTED_CLEAN_DTYPE.str:
            raise ValueError("Fresh Nino-3 target must be little-endian float32.")
        native_time = np.load(
            auxiliary_paths[native_time_filename], mmap_mode="r", allow_pickle=False
        )
        if native_time.shape != (retained_steps,) or native_time.dtype.str != "<f8":
            raise ValueError("Fresh native time must have shape (time,) and dtype <f8.")

        if verify_checksums:
            for path, digest in recorded_digests.items():
                if sha256_file(path) != digest:
                    raise ValueError(
                        f"Checksum mismatch for processed-data file: {path}"
                    )

        self.input_profile = input_profile
        self.field_names = selected_names
        self.spatial_field_names = selected_names
        self.n_time_steps = retained_steps
        self.n_latitudes, self.n_longitudes = shape
        self.n_fields = len(selected_names)
        self.spatial_input_shape = (self.n_fields, *shape)
        self.n_phase_features = 2
        self.input_shape = (self.n_fields * shape[0] * shape[1] + 2,)
        self.n_features = self.input_shape[0]
        self.steps_per_month = steps_per_month
        self.steps_per_year = steps_per_year
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.metadata_sha256 = sha256_file(metadata_path)
        self.checksum_verified = bool(verify_checksums)

    def fixed_supervised_split(self, lead_months: int) -> FixedSupervisedSplit:
        """Return pairs wholly contained in each canonical chronological block."""

        if self.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
            raise ValueError("fixed_supervised_split requires the fresh ZC data set.")
        if lead_months <= 0:
            raise ValueError("lead_months must be positive")
        split = _required_mapping(self.metadata, "chronological_split", "metadata")
        blocks: dict[str, tuple[int, int]] = {}
        lead_steps = lead_months * self.steps_per_month
        arrays: dict[str, np.ndarray] = {}
        for name in ("train", "validation", "test"):
            start, stop = _integer_pair(split.get(name), f"chronological_split.{name}")
            if start < 0 or stop > self.n_time_steps or stop <= start:
                _metadata_error(f"chronological_split.{name} is outside the data set.")
            if stop - start <= lead_steps:
                raise ValueError(f"Lead time leaves no usable {name} pairs.")
            blocks[name] = (start, stop)
            arrays[name] = np.arange(start, stop - lead_steps, dtype=np.int64)
        if not (
            blocks["train"][1] <= blocks["validation"][0]
            and blocks["validation"][1] <= blocks["test"][0]
        ):
            _metadata_error(
                "chronological split blocks must be ordered and nonoverlapping."
            )
        return FixedSupervisedSplit(
            train_inputs=arrays["train"],
            validation_inputs=arrays["validation"],
            test_inputs=arrays["test"],
            lead_steps=lead_steps,
            train_block=blocks["train"],
            validation_block=blocks["validation"],
            test_block=blocks["test"],
        )

    def supervised_split(
        self,
        lead_months: int,
        *,
        train_fraction: float = 0.9,
    ) -> SupervisedSplit:
        if lead_months <= 0:
            raise ValueError("lead_months must be positive")
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between 0 and 1")
        lead_steps = lead_months * self.steps_per_month
        split_step = int(math.floor(train_fraction * self.n_time_steps))
        train_stop = split_step - lead_steps
        test_stop = self.n_time_steps - lead_steps
        if train_stop <= 0 or test_stop <= split_step:
            raise ValueError("The requested lead leaves no usable train or test pairs.")
        return SupervisedSplit(
            train_inputs=np.arange(train_stop, dtype=np.int64),
            test_inputs=np.arange(split_step, test_stop, dtype=np.int64),
            split_step=split_step,
            lead_steps=lead_steps,
        )

    def development_selection(
        self,
        available_inputs: np.ndarray,
        *,
        train_years: float,
        validation_fraction: float,
        validation_embargo_steps: int,
        seed: int,
    ) -> DevelopmentSelection:
        if train_years <= 0.0 or not math.isfinite(train_years):
            raise ValueError("train_years must be finite and positive")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly between 0 and 1")
        if validation_embargo_steps < 0:
            raise ValueError("validation_embargo_steps cannot be negative")
        available_inputs = np.asarray(available_inputs, dtype=np.int64)
        if available_inputs.ndim != 1 or available_inputs.size == 0:
            raise ValueError(
                "available_inputs must be a nonempty one-dimensional array"
            )
        if not np.array_equal(
            available_inputs,
            np.arange(available_inputs[0], available_inputs[-1] + 1),
        ):
            raise ValueError("available_inputs must be sorted, unique, and contiguous")
        requested_count = max(2, int(round(train_years * self.steps_per_year)))
        count = min(requested_count, available_inputs.size)
        rng = np.random.default_rng(seed)
        maximum_start = available_inputs.size - count
        offset = 0 if maximum_start == 0 else int(rng.integers(maximum_start + 1))
        development = np.asarray(
            available_inputs[offset : offset + count],
            dtype=np.int64,
        )

        validation_count = max(1, int(round(validation_fraction * count)))
        validation_count = min(validation_count, count - 1)
        fit_count = count - validation_count - validation_embargo_steps
        if fit_count < 1:
            raise ValueError(
                "The requested development window is too short for the validation "
                "block and embargo. Increase train_years or reduce the lead time."
            )
        fit = development[:fit_count]
        embargo = development[fit_count : count - validation_count]
        validation = development[count - validation_count :]
        return DevelopmentSelection(
            development_inputs=development,
            fit_inputs=fit,
            embargo_inputs=embargo,
            validation_inputs=validation,
            validation_embargo_steps=int(validation_embargo_steps),
            requested_years=float(train_years),
            actual_years=count / self.steps_per_year,
            window_start_step=int(development[0]),
            window_stop_step_exclusive=int(development[-1]) + 1,
        )

    def load_inputs(
        self,
        indices: np.ndarray,
        *,
        standardizer: Standardizer | None = None,
    ) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if self.n_phase_features:
            values = np.empty((indices.size, self.n_features), dtype=np.float32)
            spatial_size = self.n_features - self.n_phase_features
            spatial = values[:, :spatial_size].reshape(
                indices.size, *self.spatial_input_shape
            )
            for field_index, array in enumerate(self.arrays):
                spatial[:, field_index] = array[indices]
            values[:, spatial_size:] = self.phase[indices]
        else:
            values = np.empty((indices.size, *self.input_shape), dtype=np.float32)
            for field_index, array in enumerate(self.arrays):
                values[:, field_index] = array[indices]
        if standardizer is not None:
            np.subtract(values, standardizer.mean, out=values)
            np.divide(values, standardizer.scale, out=values)
        return values

    def load_targets(self, input_indices: np.ndarray, *, lead_steps: int) -> np.ndarray:
        indices = np.asarray(input_indices, dtype=np.int64) + int(lead_steps)
        return np.asarray(self.target[indices], dtype=np.float32)

    def compute_standardizer(
        self,
        indices: np.ndarray,
        *,
        batch_size: int = 1024,
        relative_floor: float = 1.0e-6,
    ) -> Standardizer:
        """Compute population moments in float64 over the specified inputs."""

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("At least one one-dimensional input index is required.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if relative_floor <= 0.0:
            raise ValueError("relative_floor must be positive")

        ordered = np.sort(indices)
        total = np.zeros(self.input_shape, dtype=np.float64)
        total_squares = np.zeros(self.input_shape, dtype=np.float64)
        for start in range(0, ordered.size, batch_size):
            batch_indices = ordered[start : start + batch_size]
            values = self.load_inputs(batch_indices).astype(np.float64, copy=False)
            total += values.sum(axis=0, dtype=np.float64)
            total_squares += np.square(values).sum(axis=0, dtype=np.float64)

        mean = total / ordered.size
        variance = np.maximum(total_squares / ordered.size - np.square(mean), 0.0)
        raw_scale = np.sqrt(variance)
        positive = raw_scale[raw_scale > 0.0]
        typical_scale = float(np.median(positive)) if positive.size else 1.0
        scale_floor = max(np.finfo(np.float32).eps, relative_floor * typical_scale)
        scale = np.maximum(raw_scale, scale_floor)
        return Standardizer(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            scale_floor=float(scale_floor),
            count=int(ordered.size),
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "data_directory_name": self.data_dir.name,
            "metadata_file": "metadata.json",
            "metadata_sha256": self.metadata_sha256,
            "checksum_verified": self.checksum_verified,
            "format_version": self.metadata.get(
                "format_version", self.metadata.get("schema_version")
            ),
            "field_names": list(self.field_names),
            "input_profile": self.input_profile,
            "spatial_input_shape": list(self.spatial_input_shape),
            "n_phase_features": self.n_phase_features,
            "input_shape": list(self.input_shape),
            "n_time_steps": self.n_time_steps,
            "steps_per_month": self.steps_per_month,
            "steps_per_year": self.steps_per_year,
        }
