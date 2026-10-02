#!/usr/bin/env python3
"""Reproduce Figure 3 with NOAA ERSSTv5 and the fresh ZC grid regions.

The background is the annual mean of NOAA's fixed 1991--2020 ERSSTv5
monthly climatology. The blue rectangle uses the half-grid cell-edge
footprint inferred from the centers of the 20 x 27 active Zebiak--Cane grid.
The black rectangle is the conventional continuous Nino-3 region.

The NOAA file is downloaded once to ``data/external/noaa`` and verified by
SHA-256. Neither the private ``grads_1.data`` file nor the large extracted
model arrays are read.

Run from the repository root::

    python scripts/make_figure3.py \
        --data-dir data/processed/zc-v3 \
        --output outputs/manuscript/figures/figure3_noaa_ersstv5.pdf \
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import scipy
from scipy.io import netcdf_file

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

LOGGER = logging.getLogger(__name__)

SCRIPT_VERSION = "3.1.0"
EXPECTED_FORMAT_VERSION = 3
FRESH_SCHEMA_VERSION = "fresh-zc-interpretability-v1"
SUPPORTED_OUTPUT_SUFFIXES = {".pdf", ".png"}

NOAA_ERSST_URL = (
    "https://www.psl.noaa.gov/thredds/fileServer/Datasets/"
    "noaa.ersst.v5/sst.mon.ltm.1991-2020.nc"
)
NOAA_ERSST_SHA256 = (
    "b4f20ecfbf5e4c9276ff9e72adbf6205663e9efd64a3b1e74a0fcd2b3c72e99f"
)
NOAA_ERSST_SIZE_BYTES = 1_160_004
NOAA_CLIMATOLOGY_PERIOD = "1991-2020"

EXPECTED_DATA_LATITUDE_EDGES = (-20.0, 20.0)
EXPECTED_DATA_LONGITUDE_EDGES = (126.5625, 278.4375)
EXPECTED_NINO3_LATITUDE_BOUNDS = (-5.0, 5.0)
EXPECTED_NINO3_LONGITUDE_BOUNDS = (210.0, 270.0)
EXPECTED_TARGET_LATITUDES = np.arange(-5.0, 5.1, 2.0, dtype=np.float64)
EXPECTED_TARGET_LONGITUDES = np.arange(
    213.75,
    270.0 + 0.1,
    5.625,
    dtype=np.float64,
)
EXPECTED_ACTIVE_LATITUDES = np.arange(-19.0, 20.0, 2.0, dtype=np.float64)
EXPECTED_ACTIVE_LONGITUDES = 129.375 + 5.625 * np.arange(27, dtype=np.float64)
EXPECTED_FRESH_NINO3_DEFINITION = (
    "unweighted mean of model-grid SST-anomaly centers inside or on "
    "5S-5N, 150W-90W"
)

DATA_REGION_COLOR = "#0000ff"
NINO3_COLOR = "#111111"
LAND_COLOR = "#f3f3f3"
COASTLINE_COLOR = "#2f2f2f"
COLORBAR_PAD_FRACTION = 0.075

FIGURE_STYLE: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 10.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
COLORBAR_LABEL_FONT_SIZE = 11.0
COLORBAR_TICK_FONT_SIZE = 9.0


@dataclass(frozen=True)
class RegionMetadata:
    """Validated geographic regions read from processed-data metadata."""

    data_latitude_edges: tuple[float, float]
    data_longitude_edges: tuple[float, float]
    nino3_latitude_bounds: tuple[float, float]
    nino3_longitude_bounds: tuple[float, float]
    metadata_sha256: str
    metadata_schema: str


@dataclass(frozen=True)
class ErsstClimatology:
    """Validated annual NOAA ERSSTv5 climatology."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    annual_sst: np.ma.MaskedArray
    file_sha256: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the NOAA-backed global Figure 3 with the inferred ZC "
            "active-grid footprint and corrected Nino-3 definition."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/zc-v3"),
        help="Fresh zc-v3 (or corrected legacy format-version-3) data directory.",
    )
    parser.add_argument(
        "--noaa-file",
        type=Path,
        default=Path("data/external/noaa/sst.mon.ltm.1991-2020.nc"),
        help="Local cache for the fixed NOAA ERSSTv5 climatology.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading the NOAA file when it is absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manuscript/figures/figure3_noaa_ersstv5.pdf"),
        help="Output path ending in .pdf or .png.",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=600,
        help="Resolution for PNG and rasterized SST inside PDF output.",
    )
    parser.add_argument(
        "--coastline-resolution",
        choices=("110m", "50m", "10m"),
        default="50m",
        help="Natural Earth land/coastline resolution used by Cartopy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the figure and its JSON provenance sidecar.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_pair(value: object, *, label: str) -> tuple[float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata.json has invalid {label}.") from error
    if array.shape != (2,) or not np.isfinite(array).all() or not array[0] < array[1]:
        raise ValueError(f"metadata.json {label} must be two increasing numbers.")
    return float(array[0]), float(array[1])


def numeric_vector(value: object, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata.json has invalid {label}.") from error
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"metadata.json has invalid {label}.")
    return array


def require_exact_pair(
    actual: tuple[float, float],
    expected: tuple[float, float],
    *,
    label: str,
) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"Expected {label} {expected}, but metadata declares {actual}."
        )


def outer_edges_from_centers(
    centers: np.ndarray,
    *,
    label: str,
) -> tuple[float, float]:
    """Infer the plotted cell-edge footprint from uniform grid centers."""

    if centers.ndim != 1 or centers.size < 2:
        raise ValueError(f"{label} must contain at least two grid centers.")
    spacings = np.diff(centers)
    if not np.allclose(spacings, spacings[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{label} must be uniformly spaced.")
    half_spacing = 0.5 * float(spacings[0])
    return float(centers[0] - half_spacing), float(centers[-1] + half_spacing)


def _load_fresh_region_metadata(
    metadata: dict[str, Any],
    *,
    raw_document: bytes,
) -> RegionMetadata:
    try:
        grid = metadata["grid"]
        nino3 = metadata["nino3"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Fresh metadata.json is missing the grid or Nino-3 definition."
        ) from error

    active_latitudes = numeric_vector(
        grid.get("latitude_degrees_north"),
        label="fresh active-grid latitude centers",
    )
    active_longitudes = numeric_vector(
        grid.get("longitude_degrees_east"),
        label="fresh active-grid longitude centers",
    )
    if grid.get("shape") != [20, 27]:
        raise ValueError("Fresh metadata grid.shape must be [20, 27].")
    if not np.array_equal(active_latitudes, EXPECTED_ACTIVE_LATITUDES):
        raise ValueError("Fresh active-grid latitudes are not the canonical centers.")
    if not np.array_equal(active_longitudes, EXPECTED_ACTIVE_LONGITUDES):
        raise ValueError("Fresh active-grid longitudes are not the canonical centers.")

    target_latitudes = numeric_vector(
        nino3.get("latitude_degrees_north"),
        label="fresh model-grid Nino-3 target latitudes",
    )
    target_longitudes = numeric_vector(
        nino3.get("longitude_degrees_east"),
        label="fresh model-grid Nino-3 target longitudes",
    )
    if not np.array_equal(target_latitudes, EXPECTED_TARGET_LATITUDES):
        raise ValueError("Fresh Nino-3 target latitudes are not canonical.")
    if not np.array_equal(target_longitudes, EXPECTED_TARGET_LONGITUDES):
        raise ValueError(
            "Fresh Nino-3 must exclude 151.875 W and include the 90 W center."
        )
    if nino3.get("definition") != EXPECTED_FRESH_NINO3_DEFINITION:
        raise ValueError("Fresh metadata has an unexpected Nino-3 definition.")
    if nino3.get("grid_point_count") != 66 or nino3.get(
        "includes_270E_90W_center"
    ) is not True:
        raise ValueError("Fresh Nino-3 grid-point metadata is not canonical.")

    data_latitudes = outer_edges_from_centers(
        active_latitudes,
        label="fresh active-grid latitude centers",
    )
    data_longitudes = outer_edges_from_centers(
        active_longitudes,
        label="fresh active-grid longitude centers",
    )
    require_exact_pair(
        data_latitudes,
        EXPECTED_DATA_LATITUDE_EDGES,
        label="fresh active-domain latitude outer edges",
    )
    require_exact_pair(
        data_longitudes,
        EXPECTED_DATA_LONGITUDE_EDGES,
        label="fresh active-domain longitude outer edges",
    )
    return RegionMetadata(
        data_latitude_edges=data_latitudes,
        data_longitude_edges=data_longitudes,
        nino3_latitude_bounds=EXPECTED_NINO3_LATITUDE_BOUNDS,
        nino3_longitude_bounds=EXPECTED_NINO3_LONGITUDE_BOUNDS,
        metadata_sha256=hashlib.sha256(raw_document).hexdigest(),
        metadata_schema=FRESH_SCHEMA_VERSION,
    )


def load_region_metadata(data_dir: Path) -> RegionMetadata:
    """Load fresh or corrected legacy metadata and validate plotted regions."""

    metadata_path = data_dir.expanduser().resolve() / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Prepared-data metadata not found: {metadata_path}\n"
            "Create the fresh zc-v3 data set first."
        )
    raw_document = metadata_path.read_bytes()
    metadata = json.loads(raw_document)
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected a JSON object in {metadata_path}.")
    if metadata.get("schema_version") == FRESH_SCHEMA_VERSION:
        return _load_fresh_region_metadata(metadata, raw_document=raw_document)
    if metadata.get("format_version") != EXPECTED_FORMAT_VERSION:
        raise ValueError(
            "Figure 3 requires format-version-3 prepared data; received "
            f"{metadata.get('format_version')!r}."
        )

    try:
        active = metadata["grid"]["active_domain"]
        outer_edges = active["outer_cell_edge_bounds"]
        nino3 = metadata["nino3"]
        conventional = nino3["nominal_conventional_region"]
        centers = nino3["selected_grid_centers"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "metadata.json is missing active-domain or Nino-3 definitions."
        ) from error

    data_latitudes = numeric_pair(
        outer_edges.get("latitude_degrees_north"),
        label="active-domain latitude outer edges",
    )
    data_longitudes = numeric_pair(
        outer_edges.get("longitude_degrees_east"),
        label="active-domain longitude outer edges",
    )
    nino3_latitudes = numeric_pair(
        conventional.get("latitude_bounds_degrees_north"),
        label="Nino-3 latitude bounds",
    )
    nino3_longitudes = numeric_pair(
        conventional.get("longitude_bounds_degrees_east"),
        label="Nino-3 longitude bounds",
    )
    require_exact_pair(
        data_latitudes,
        EXPECTED_DATA_LATITUDE_EDGES,
        label="active-domain latitude outer edges",
    )
    require_exact_pair(
        data_longitudes,
        EXPECTED_DATA_LONGITUDE_EDGES,
        label="active-domain longitude outer edges",
    )
    require_exact_pair(
        nino3_latitudes,
        EXPECTED_NINO3_LATITUDE_BOUNDS,
        label="conventional Nino-3 latitude bounds",
    )
    require_exact_pair(
        nino3_longitudes,
        EXPECTED_NINO3_LONGITUDE_BOUNDS,
        label="conventional Nino-3 longitude bounds",
    )

    target_latitudes = numeric_vector(
        centers.get("latitude_degrees_north"),
        label="model-grid Nino-3 target latitudes",
    )
    target_longitudes = numeric_vector(
        centers.get("longitude_degrees_east"),
        label="model-grid Nino-3 target longitudes",
    )
    latitudes_match = np.array_equal(target_latitudes, EXPECTED_TARGET_LATITUDES)
    longitudes_match = np.array_equal(target_longitudes, EXPECTED_TARGET_LONGITUDES)
    if not latitudes_match or not longitudes_match:
        raise ValueError(
            "This data set still uses the superseded model-grid Nino-3 centers. "
            "The corrected target must exclude 151.875 W and include 90 W "
            "(longitudes 213.75 E through 270 E). Regenerate the public "
            "zc-v3 data set with scripts/generate_fresh_zc_dataset.py."
        )
    if nino3.get("source_row_slice") != [12, 18] or nino3.get(
        "source_column_slice"
    ) != [20, 31]:
        raise ValueError(
            "The corrected target must use raw NumPy slices rows [12, 18] and "
            "columns [20, 31]."
        )

    return RegionMetadata(
        data_latitude_edges=data_latitudes,
        data_longitude_edges=data_longitudes,
        nino3_latitude_bounds=nino3_latitudes,
        nino3_longitude_bounds=nino3_longitudes,
        metadata_sha256=hashlib.sha256(raw_document).hexdigest(),
        metadata_schema=f"legacy-format-version-{EXPECTED_FORMAT_VERSION}",
    )


def verify_noaa_file(path: Path) -> str:
    if path.stat().st_size != NOAA_ERSST_SIZE_BYTES:
        raise ValueError(
            f"NOAA cache has size {path.stat().st_size:,} bytes; expected "
            f"{NOAA_ERSST_SIZE_BYTES:,}: {path}"
        )
    digest = sha256_file(path)
    if digest != NOAA_ERSST_SHA256:
        raise ValueError(
            "NOAA cache checksum does not match the fixed ERSSTv5 file. "
            f"Expected {NOAA_ERSST_SHA256}, received {digest}: {path}"
        )
    return digest


def obtain_noaa_file(path: Path, *, download: bool) -> tuple[Path, str]:
    """Return a verified local copy, downloading atomically when necessary."""

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved, verify_noaa_file(resolved)
    if resolved.exists():
        raise ValueError(f"NOAA cache path is not a regular file: {resolved}")
    if not download:
        raise FileNotFoundError(
            f"NOAA ERSSTv5 cache not found: {resolved}\n"
            f"Download {NOAA_ERSST_URL} there or omit --no-download."
        )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent,
        prefix=f".{resolved.name}.incomplete-",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    LOGGER.info("Downloading fixed NOAA ERSSTv5 climatology: %s", NOAA_ERSST_URL)
    try:
        request = urllib.request.Request(
            NOAA_ERSST_URL,
            headers={"User-Agent": "zc-xai-figure3/2.0"},
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1 << 20)
                stream.flush()
                os.fsync(stream.fileno())
        except (TimeoutError, urllib.error.URLError) as error:
            raise ConnectionError(
                "Could not download NOAA ERSSTv5. Check the network connection, "
                "or download the URL shown above in a browser and pass "
                "--noaa-file."
            ) from error
        digest = verify_noaa_file(temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, resolved)
        return resolved, digest
    finally:
        temporary.unlink(missing_ok=True)


def load_ersst_climatology(path: Path, digest: str) -> ErsstClimatology:
    """Read and validate NOAA's 12 monthly long-term means."""

    with netcdf_file(path, mode="r", mmap=False) as dataset:
        try:
            latitudes = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            longitudes = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
            sst_variable = dataset.variables["sst"]
            monthly_sst = np.asarray(sst_variable[:], dtype=np.float64)
        except KeyError as error:
            raise ValueError(
                f"NOAA file is missing variable {error.args[0]!r}."
            ) from error
        units = getattr(sst_variable, "units", b"")
        if isinstance(units, bytes):
            units = units.decode("ascii", errors="replace")
        missing_value = float(getattr(sst_variable, "missing_value", -9.96921e36))

    if latitudes.shape != (89,) or longitudes.shape != (180,):
        raise ValueError(
            "NOAA ERSSTv5 coordinates must have 89 latitudes and 180 longitudes."
        )
    if monthly_sst.shape != (12, 89, 180):
        raise ValueError(
            "NOAA ERSSTv5 climatology must have shape (12, 89, 180); received "
            f"{monthly_sst.shape}."
        )
    if units.strip().lower() not in {"degc", "degree_celsius", "degrees c"}:
        raise ValueError(f"Unexpected NOAA SST units: {units!r}.")
    if not np.allclose(latitudes, np.arange(88.0, -88.1, -2.0)):
        raise ValueError("Unexpected NOAA ERSSTv5 latitude coordinates.")
    if not np.allclose(longitudes, np.arange(0.0, 360.0, 2.0)):
        raise ValueError("Unexpected NOAA ERSSTv5 longitude coordinates.")

    invalid = monthly_sst <= missing_value / 2.0
    invalid |= ~np.isfinite(monthly_sst)
    masked_monthly = np.ma.array(monthly_sst, mask=invalid)
    annual_sst = np.ma.mean(masked_monthly, axis=0)
    if annual_sst.count() == 0:
        raise ValueError("NOAA ERSSTv5 contains no valid SST values.")
    valid_values = annual_sst.compressed()
    if valid_values.min() < -2.1 or valid_values.max() > 45.0:
        raise ValueError(
            "NOAA ERSSTv5 SST values fall outside expected physical bounds."
        )

    # NOAA stores latitude north-to-south; reverse it once for plotting.
    return ErsstClimatology(
        latitudes=latitudes[::-1].copy(),
        longitudes=longitudes.copy(),
        annual_sst=annual_sst[::-1].copy(),
        file_sha256=digest,
    )


def add_region_rectangle(
    axis: plt.Axes,
    *,
    longitude_bounds: tuple[float, float],
    latitude_bounds: tuple[float, float],
    color: str,
    linewidth: float,
) -> None:
    axis.add_patch(
        Rectangle(
            (longitude_bounds[0], latitude_bounds[0]),
            longitude_bounds[1] - longitude_bounds[0],
            latitude_bounds[1] - latitude_bounds[0],
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
    )


def build_figure(
    regions: RegionMetadata,
    ersst: ErsstClimatology,
    *,
    coastline_resolution: str,
) -> plt.Figure:
    """Construct a high-resolution counterpart of the manuscript map."""

    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.PlateCarree(central_longitude=180.0)

    # Duplicate Greenwich at 360 degrees to remove the map-edge seam.
    cyclic_sst = np.ma.concatenate(
        [ersst.annual_sst, ersst.annual_sst[:, :1]],
        axis=1,
    )
    cyclic_longitudes = np.append(ersst.longitudes, 360.0)
    displayed_sst = np.ma.clip(cyclic_sst, -2.0, 32.0)

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(12.0, 7.0), constrained_layout=False)
        axis = figure.add_subplot(1, 1, 1, projection=map_crs)
        mesh = axis.pcolormesh(
            cyclic_longitudes,
            ersst.latitudes,
            displayed_sst,
            cmap="Reds",
            vmin=-2.0,
            vmax=32.0,
            shading="nearest",
            transform=data_crs,
            rasterized=True,
            zorder=1,
        )

        land = cfeature.NaturalEarthFeature(
            category="physical",
            name="land",
            scale=coastline_resolution,
            facecolor=LAND_COLOR,
            edgecolor="none",
        )
        axis.add_feature(land, zorder=3)
        axis.coastlines(
            resolution=coastline_resolution,
            color=COASTLINE_COLOR,
            linewidth=0.7,
            zorder=4,
        )

        axis.set_global()
        axis.set_xticks([0, 60, 120, 180, 240, 300], crs=data_crs)
        axis.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W"])
        axis.set_yticks([-60, -30, 0, 30, 60], crs=data_crs)
        axis.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N"])
        axis.tick_params(direction="out", length=3.0, width=0.75, pad=3.5)
        gridlines = axis.gridlines(
            crs=data_crs,
            xlocs=[-180, -120, -60, 0, 60, 120, 180],
            ylocs=[-60, -30, 0, 30, 60],
            draw_labels=False,
            color="#818181",
            linewidth=0.65,
            linestyle="--",
            alpha=0.42,
            zorder=5,
        )
        gridlines.n_steps = 100
        axis.spines["geo"].set_color("#444444")
        axis.spines["geo"].set_linewidth(0.8)

        add_region_rectangle(
            axis,
            longitude_bounds=regions.data_longitude_edges,
            latitude_bounds=regions.data_latitude_edges,
            color=DATA_REGION_COLOR,
            linewidth=2.2,
        )
        add_region_rectangle(
            axis,
            longitude_bounds=regions.nino3_longitude_bounds,
            latitude_bounds=regions.nino3_latitude_bounds,
            color=NINO3_COLOR,
            linewidth=1.7,
        )

        legend = axis.legend(
            handles=[
                Patch(
                    facecolor="none",
                    edgecolor=DATA_REGION_COLOR,
                    linewidth=2.2,
                    label="Data Region",
                ),
                Patch(
                    facecolor="none",
                    edgecolor=NINO3_COLOR,
                    linewidth=1.7,
                    label="Niño3",
                ),
            ],
            loc="lower left",
            frameon=True,
            framealpha=0.94,
            facecolor="white",
            edgecolor="#c4c4c4",
            borderpad=0.5,
            handlelength=2.5,
        )
        legend.get_frame().set_linewidth(0.7)

        colorbar = figure.colorbar(
            mesh,
            ax=axis,
            orientation="horizontal",
            ticks=np.arange(0.0, 31.0, 5.0),
            pad=COLORBAR_PAD_FRACTION,
            fraction=0.15,
            shrink=0.80,
            aspect=20,
        )
        colorbar.set_label(
            "SST (°C)", fontsize=COLORBAR_LABEL_FONT_SIZE, labelpad=5.0
        )
        colorbar.ax.tick_params(
            labelsize=COLORBAR_TICK_FONT_SIZE,
            length=3.0,
            width=0.7,
        )
        colorbar.outline.set_linewidth(0.75)

    return figure


def validated_output_paths(
    output_path: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    output = output_path.expanduser().resolve()
    if output.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        raise ValueError(f"Output extension must be one of {supported}: {output}")
    sidecar = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, sidecar):
        if path.exists() and not path.is_file():
            raise ValueError(f"Output path exists but is not a regular file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}\nUse --overwrite to replace it."
            )
    return output, sidecar


def replace_atomically(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    os.chmod(temporary, 0o644)
    if overwrite:
        os.replace(temporary, destination)
    else:
        os.link(temporary, destination)
        temporary.unlink()


def save_figure_atomically(
    figure: plt.Figure,
    output: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}.incomplete-",
        suffix=output.suffix.lower(),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        metadata: dict[str, Any] = {
            "Title": "NOAA ERSSTv5 climatology and Zebiak-Cane model regions",
            "Creator": f"make_figure3.py {SCRIPT_VERSION}",
            "Subject": "Figure 3",
        }
        if output.suffix.lower() == ".pdf":
            metadata["CreationDate"] = None
        figure.savefig(
            temporary,
            dpi=dpi,
            metadata=metadata,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        if temporary.stat().st_size == 0:
            raise OSError(f"Matplotlib created an empty figure: {temporary}")
        replace_atomically(temporary, output, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def save_json_atomically(
    document: dict[str, Any],
    output: Path,
    *,
    overwrite: bool,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}.incomplete-",
        suffix=".json",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace_atomically(temporary, output, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def provenance_document(
    regions: RegionMetadata,
    ersst: ErsstClimatology,
    *,
    noaa_file: Path,
    output: Path,
    coastline_resolution: str,
    dpi: int,
) -> dict[str, Any]:
    return {
        "figure": "Figure 3",
        "generator": {
            "script": "scripts/make_figure3.py",
            "version": SCRIPT_VERSION,
        },
        "output": {
            "file": output.name,
            "sha256": sha256_file(output),
        },
        "background": {
            "product": "NOAA Extended Reconstructed SST version 5 (ERSSTv5)",
            "variable": "sst",
            "units": "degrees C",
            "climatology_period": NOAA_CLIMATOLOGY_PERIOD,
            "annual_aggregation": "unweighted mean of 12 monthly long-term means",
            "source_url": NOAA_ERSST_URL,
            "local_file_name": noaa_file.name,
            "size_bytes": NOAA_ERSST_SIZE_BYTES,
            "sha256": ersst.file_sha256,
            "dataset_doi": "10.7289/V5T72FNM",
        },
        "regions": {
            "zc_active_domain_inferred_cell_edge_footprint": {
                "latitude_degrees_north": list(regions.data_latitude_edges),
                "longitude_degrees_east": list(regions.data_longitude_edges),
                "derivation": (
                    "Half a grid spacing beyond the retained active-grid centers."
                ),
            },
            "conventional_nino3": {
                "latitude_degrees_north": list(regions.nino3_latitude_bounds),
                "longitude_degrees_east": list(regions.nino3_longitude_bounds),
            },
            "processed_metadata_sha256": regions.metadata_sha256,
            "processed_metadata_schema": regions.metadata_schema,
        },
        "rendering": {
            "projection": "PlateCarree, central_longitude=180 degrees",
            "coastline_source": "Natural Earth via Cartopy",
            "coastline_resolution": coastline_resolution,
            "dpi": dpi,
            "color_limits_degrees_c": [-2.0, 32.0],
            "colorbar_pad_fraction": COLORBAR_PAD_FRACTION,
            "font_size_points": FIGURE_STYLE["font.size"],
            "tick_label_size_points": FIGURE_STYLE["xtick.labelsize"],
            "legend_font_size_points": FIGURE_STYLE["legend.fontsize"],
            "colorbar_label_size_points": COLORBAR_LABEL_FONT_SIZE,
            "colorbar_tick_size_points": COLORBAR_TICK_FONT_SIZE,
            "bounding_box": "tight",
            "outer_padding_inches": 0.03,
        },
        "software": {
            "cartopy": cartopy.__version__,
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output, sidecar = validated_output_paths(args.output, overwrite=args.overwrite)
    regions = load_region_metadata(args.data_dir)
    noaa_file, noaa_digest = obtain_noaa_file(
        args.noaa_file,
        download=not args.no_download,
    )
    ersst = load_ersst_climatology(noaa_file, noaa_digest)
    LOGGER.info(
        "Loaded NOAA ERSSTv5 %s climatology (annual range %.2f to %.2f C).",
        NOAA_CLIMATOLOGY_PERIOD,
        float(ersst.annual_sst.min()),
        float(ersst.annual_sst.max()),
    )

    figure = build_figure(
        regions,
        ersst,
        coastline_resolution=args.coastline_resolution,
    )
    try:
        save_figure_atomically(
            figure,
            output,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
    finally:
        plt.close(figure)
    save_json_atomically(
        provenance_document(
            regions,
            ersst,
            noaa_file=noaa_file,
            output=output,
            coastline_resolution=args.coastline_resolution,
            dpi=args.dpi,
        ),
        sidecar,
        overwrite=args.overwrite,
    )
    LOGGER.info("Figure 3 saved: %s", output)
    LOGGER.info("Figure provenance saved: %s", sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
