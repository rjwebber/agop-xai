"""Map renderer shared by the fresh-ZC explanation figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm

from .data import ZCData
from .fresh_xai_outputs import split_spatial_phase
from .io import atomic_output_path

LOGGER = logging.getLogger(__name__)

ARCHITECTURE_LABELS = (("MLP", "mlp"), ("CNN", "cnn"), ("ViT", "vit"))
METHOD_LABELS = (
    ("AGOP", "agop"),
    ("GradientSHAP", "gradientshap"),
    ("IG", "ig"),
    ("GRAD", "grad"),
)
COLOR_LEVELS = np.linspace(-1.0, 1.0, 19, dtype=np.float64)
COLORMAP_LOOKUP_TABLE_SIZE = 256
COLORMAP = plt.get_cmap("bwr").resampled(COLORMAP_LOOKUP_TABLE_SIZE)
COLORMAP_DESCRIPTION = (
    "Matplotlib bwr sampled at 256 colors; zero is a level boundary and the "
    "lookup table has no exact-white entry"
)
COLORBAR_LABEL = "Normalized coefficient"
LONGITUDE_TICKS = np.asarray((150.0, 180.0, 210.0, 240.0, 270.0))
LONGITUDE_LABELS = ("150°E", "180°", "150°W", "120°W", "90°W")
LATITUDE_TICKS = np.asarray((-15.0, -5.0, 5.0, 15.0))
LATITUDE_LABELS = ("15°S", "5°S", "5°N", "15°N")
QUIVER_SCALE = 40.0
QUIVER_WIDTH = 0.0024
PANEL_BOX_ASPECT = 2.0 / 3.0
# Shared physical horizontal geometry for every three-column map figure.
# A common ``wspace`` alone is insufficient when fixed-aspect axes are allowed
# to shrink inside differently sized GridSpec cells.
THREE_COLUMN_FIGURE_WIDTH_INCHES = 10.0
THREE_COLUMN_GRID_LEFT = 0.10
THREE_COLUMN_GRID_RIGHT = 0.985
THREE_COLUMN_WSPACE = 0.05


def cell_edge_bounds(coordinates: np.ndarray) -> tuple[float, float]:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError("Map coordinates must be a strictly increasing vector.")
    return (
        float(values[0] - 0.5 * (values[1] - values[0])),
        float(values[-1] + 0.5 * (values[-1] - values[-2])),
    )


def _normalized_scalar(values: np.ndarray, label: str) -> np.ndarray:
    maximum = float(np.max(np.abs(values)))
    if maximum <= np.finfo(np.float64).eps:
        LOGGER.warning("%s has zero thermocline coefficients", label)
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / maximum


def _normalized_vectors(
    zonal: np.ndarray,
    meridional: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = float(np.max(np.hypot(zonal, meridional)))
    if maximum <= np.finfo(np.float64).eps:
        LOGGER.warning("%s has zero ocean-current coefficients", label)
        zeros = np.zeros_like(zonal, dtype=np.float64)
        return zeros, zeros.copy()
    return (
        np.asarray(zonal, dtype=np.float64) / maximum,
        np.asarray(meridional, dtype=np.float64) / maximum,
    )


def draw_ocean_panel(
    axis: plt.Axes,
    values: np.ndarray,
    data: ZCData,
    *,
    label: str,
    show_latitude: bool,
) -> None:
    """Draw thermocline shading and current arrows, excluding SST and phase."""

    spatial, _ = split_spatial_phase(values, data)
    thermocline = _normalized_scalar(
        spatial[data.spatial_field_names.index("thermocline_depth")], label
    )
    zonal, meridional = _normalized_vectors(
        spatial[data.spatial_field_names.index("zonal_ocean_current")],
        spatial[data.spatial_field_names.index("meridional_ocean_current")],
        label,
    )
    longitude_bounds = cell_edge_bounds(data.longitudes)
    latitude_bounds = cell_edge_bounds(data.latitudes)
    contour_longitudes = np.concatenate(
        ([longitude_bounds[0]], data.longitudes, [longitude_bounds[1]])
    )
    contour_latitudes = np.concatenate(
        ([latitude_bounds[0]], data.latitudes, [latitude_bounds[1]])
    )
    contour_values = np.pad(thermocline, ((1, 1), (1, 1)), mode="edge")
    axis.contourf(
        contour_longitudes,
        contour_latitudes,
        contour_values,
        levels=COLOR_LEVELS,
        cmap=COLORMAP,
        extend="both",
        antialiased=True,
    )
    axis.quiver(
        data.longitudes,
        data.latitudes,
        zonal,
        meridional,
        angles="uv",
        scale_units="width",
        scale=QUIVER_SCALE,
        width=QUIVER_WIDTH,
        headwidth=4.5,
        headlength=5.4,
        headaxislength=4.8,
        minlength=0.0,
        pivot="middle",
        color="0.06",
    )
    axis.set_box_aspect(PANEL_BOX_ASPECT)
    axis.set_xlim(*longitude_bounds)
    axis.set_ylim(*latitude_bounds)
    axis.set_xticks(LONGITUDE_TICKS, LONGITUDE_LABELS)
    if show_latitude:
        axis.set_yticks(LATITUDE_TICKS, LATITUDE_LABELS)
    else:
        axis.set_yticks([])
    axis.tick_params(axis="both", which="major", length=3.0, pad=2.0, labelsize=8.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.25")


def add_horizontal_colorbar(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    label: str = COLORBAR_LABEL,
) -> None:
    colorbar = figure.colorbar(
        ScalarMappable(
            norm=BoundaryNorm(COLOR_LEVELS, ncolors=COLORMAP.N),
            cmap=COLORMAP,
        ),
        cax=axis,
        orientation="horizontal",
        extend="both",
        ticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    colorbar.set_label(label, labelpad=3.0)
    colorbar.ax.tick_params(labelsize=8.5, length=2.5, pad=2.0)


def figure_style() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }


def save_figure(
    figure: plt.Figure,
    output: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> None:
    with atomic_output_path(output, overwrite=overwrite) as temporary:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
