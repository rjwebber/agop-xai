#!/usr/bin/env python3
"""Render target-locked multi-lead El Nino and La Nina Figures 8 and 9.

Each panel contains three rows: the held-out event input, an event-equal
10-percent composite, and its CNN AGOP explanation. The 10-month column shows
thermocline-depth anomaly; the 5- and 1-month columns show SST anomaly.
Depth-averaged horizontal ocean currents are overlaid throughout.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    BoundaryNorm,
    LinearSegmentedColormap,
    to_hex,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from generate_fresh_figures8_9_data import (  # noqa: E402
    ARTIFACT,
    COLD_CASES,
    LEADS,
    LEGACY_ARTIFACT,
    WARM_CASES,
)

from zc_xai.artifacts import load_completed_bundle  # noqa: E402
from zc_xai.composites import strongest_fraction, threshold_event_peaks  # noqa: E402
from zc_xai.data import ZCData  # noqa: E402
from zc_xai.fresh_case_studies import fixed_test_extreme  # noqa: E402
from zc_xai.fresh_xai_outputs import (  # noqa: E402
    FRESH_XAI_BUNDLE_SCHEMA_VERSION,
    FRESH_XAI_TRAINING_POPULATION,
    split_spatial_phase,
)
from zc_xai.fresh_xai_plotting import (  # noqa: E402
    LATITUDE_LABELS,
    LATITUDE_TICKS,
    LONGITUDE_LABELS,
    LONGITUDE_TICKS,
    PANEL_BOX_ASPECT,
    QUIVER_SCALE,
    QUIVER_WIDTH,
    THREE_COLUMN_FIGURE_WIDTH_INCHES,
    THREE_COLUMN_GRID_LEFT,
    THREE_COLUMN_GRID_RIGHT,
    THREE_COLUMN_WSPACE,
    cell_edge_bounds,
    figure_style,
)
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    sha256_array,
    sha256_file,
    write_json,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 8
EVENT_THRESHOLD_C = 1.0
TOP_PERCENT = 10.0
ROWS = ("Input", "Composite", "AGOP")
THERMOCLINE_COLOR_LEVELS = np.linspace(-1.0, 1.0, 19, dtype=np.float64)
SST_COLOR_LEVELS = THERMOCLINE_COLOR_LEVELS.copy()
COLORMAP_SIZE = 256
FIGURE_HEIGHT_INCHES = 8.0
GRID_BOTTOM = 0.15
GRID_TOP = 0.955
GRID_HSPACE = 0.18
COLORBAR_BOTTOM = 0.068
PANEL_B_COLORBAR_BOTTOM = 0.082
THERMOCLINE_COLORMAP = plt.get_cmap("bwr").resampled(COLORMAP_SIZE)
SST_COLORMAP = LinearSegmentedColormap.from_list(
    "ibm_sst",
    ("#648FFF", "#FFFFFF", "#FE6100"),
    N=SST_COLOR_LEVELS.size - 1,
)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render matched 10-, 5-, and 1-month CNN input, AGOP, and "
            "top-event-composite panels for one fixed El Nino event and one "
            "fixed La Nina event."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/zc-v3/manuscript/figures8_9"),
    )
    parser.add_argument(
        "--output-a",
        type=Path,
        default=Path("outputs/manuscript/figures/figure8_el_nino_multilead.pdf"),
    )
    parser.add_argument(
        "--output-b",
        type=Path,
        default=Path("outputs/manuscript/figures/figure9_la_nina_multilead.pdf"),
    )
    parser.add_argument("--dpi", type=positive_int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _load_cases(
    bundle: Any,
    data: ZCData,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    """Load the rows retained in manuscript Figures 8 and 9."""

    metadata = bundle.metadata
    expected_panels = {"8": list(WARM_CASES), "9": list(COLD_CASES)}
    configuration = _mapping(
        metadata.get("xai_configuration"), "metadata.xai_configuration"
    )
    numbering_is_valid = metadata.get("figure_numbers") == [8, 9] or (
        metadata.get("figure") == 8 and "figure_numbers" not in metadata
    )
    if (
        metadata.get("schema_version") != FRESH_XAI_BUNDLE_SCHEMA_VERSION
        or not numbering_is_valid
        or configuration.get("expected_gradients_candidate_population")
        != FRESH_XAI_TRAINING_POPULATION
    ):
        raise ValueError("Figures 8-9 XAI references are not training-only.")
    recorded_panels = metadata.get("figure_panels")
    if recorded_panels is None:
        legacy_panels = metadata.get("figure8_panels")
        recorded_panels = (
            {"8": legacy_panels.get("a"), "9": legacy_panels.get("b")}
            if isinstance(legacy_panels, dict)
            else None
        )
    if recorded_panels != expected_panels:
        raise ValueError(
            "Figures 8-9 panel membership or lead order is inconsistent."
        )
    recorded_data = _mapping(metadata.get("data"), "metadata.data")
    if (
        recorded_data.get("metadata_sha256") != data.metadata_sha256
        or metadata.get("input_profile") != "core4"
        or metadata.get("phase_features") != 2
    ):
        raise ValueError("Figures 8-9 bundle uses another processed-data view.")

    records = _mapping(metadata.get("cases"), "metadata.cases")
    experiments = _mapping(metadata.get("experiments"), "metadata.experiments")
    normalizers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with np.load(bundle.files["references"], allow_pickle=False) as archive:
        for lead in LEADS:
            prefix = f"lead_{lead:02d}"
            mean = np.asarray(archive[f"{prefix}__normalization_mean"])
            scale = np.asarray(archive[f"{prefix}__normalization_scale"])
            record = _mapping(experiments.get(prefix), f"experiments.{prefix}")
            if (
                mean.shape != data.input_shape
                or scale.shape != data.input_shape
                or record.get("normalization_mean_sha256") != sha256_array(mean)
                or record.get("normalization_scale_sha256") != sha256_array(scale)
                or np.any(scale <= 0.0)
            ):
                raise ValueError(f"{prefix} normalization is invalid.")
            normalizers[lead] = (mean, scale)
    reference_mean, reference_scale = normalizers[LEADS[0]]
    for lead in LEADS[1:]:
        mean, scale = normalizers[lead]
        if not (
            np.array_equal(mean, reference_mean)
            and np.array_equal(scale, reference_scale)
        ):
            raise ValueError(
                "Figures 8-9 leads use different standardizers; a shared composite "
                "comparison would be ambiguous."
            )

    expected = {
        "el_nino": fixed_test_extreme(data, lead_months=10, kind="maximum"),
        "la_nina": fixed_test_extreme(data, lead_months=10, kind="minimum"),
    }
    cases: dict[str, dict[str, np.ndarray]] = {}
    with np.load(bundle.files["explanations"], allow_pickle=False) as archive:
        for phase, order in (("el_nino", WARM_CASES), ("la_nina", COLD_CASES)):
            target_steps: set[int] = set()
            target_values: list[float] = []
            anchor = expected[phase]
            for case_id, expected_lead in zip(order, LEADS, strict=True):
                record = _mapping(records.get(case_id), f"cases.{case_id}")
                lead = int(record["lead_months"])
                input_step = int(record["input_step"])
                target_step = int(record["target_step"])
                target_value = float(record["target_nino3_c"])
                if lead != expected_lead:
                    raise ValueError(f"{case_id} is stored under the wrong lead.")
                if target_step != input_step + lead * data.steps_per_month:
                    raise ValueError(f"{case_id} input/target relationship is invalid.")
                target_steps.add(target_step)
                target_values.append(target_value)
                if lead == 10 and (input_step, target_step) != (
                    anchor.input_step,
                    anchor.target_step,
                ):
                    raise ValueError(f"{phase} does not use its fixed-test anchor.")

                event_input = np.asarray(
                    archive[f"{case_id}__event_input_standardized"],
                    dtype=np.float32,
                )
                physical_input = np.asarray(
                    archive[f"{case_id}__event_input_physical"],
                    dtype=np.float32,
                )
                agop = np.asarray(archive[f"{case_id}__agop"], dtype=np.float32)
                for label, values in (("Input", event_input), ("AGOP", agop)):
                    if (
                        values.shape != data.input_shape
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError(
                            f"{case_id} {label} has an invalid shape or value."
                        )
                if not math.isclose(
                    float(np.linalg.norm(agop.astype(np.float64))),
                    1.0,
                    rel_tol=2.0e-5,
                    abs_tol=2.0e-6,
                ):
                    raise ValueError(f"{case_id} AGOP is not a unit explanation.")
                mean, scale = normalizers[lead]
                if not np.allclose(
                    event_input,
                    (physical_input - mean) / scale,
                    rtol=1.0e-6,
                    atol=1.0e-6,
                ):
                    raise ValueError(f"{case_id} input does not match its normalizer.")
                cases[case_id] = {"Input": event_input, "AGOP": agop}
            if target_steps != {anchor.target_step} or not np.allclose(
                target_values,
                anchor.target_nino3_c,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError(f"{phase} columns are not locked to one target.")
    return cases, records, reference_mean, reference_scale


def select_composite_peaks(
    target_values: np.ndarray,
    target_steps: np.ndarray,
    full_target: np.ndarray,
    *,
    kind: Literal["warm", "cold"],
    threshold_c: float = EVENT_THRESHOLD_C,
    top_percent: float = TOP_PERCENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Select complete event peaks and the event-equal strongest fraction.

    The cold definition is the exact sign reversal of the warm definition.
    Consequently it finds one minimum within each complete interval at or
    below ``-threshold_c`` and ranks those minima by cold magnitude.
    """

    sign = 1.0 if kind == "warm" else -1.0
    signed_values = sign * np.asarray(target_values, dtype=np.float64)
    signed_full_target = sign * np.asarray(full_target, dtype=np.float64)
    peaks = threshold_event_peaks(
        signed_values,
        target_steps,
        threshold_c=threshold_c,
    )
    selected = strongest_fraction(
        peaks,
        signed_full_target,
        percent=top_percent,
    )
    return peaks, selected


def _build_composites(
    data: ZCData,
    cases: dict[str, dict[str, np.ndarray]],
    records: dict[str, Any],
    normalization_mean: np.ndarray,
    normalization_scale: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Build warm and cold event-equal composites at the three leads."""

    split = data.fixed_supervised_split(10)
    target_steps = split.test_inputs + split.lead_steps
    target_values = np.asarray(data.target[target_steps], dtype=np.float64)
    outputs: dict[str, dict[str, Any]] = {}
    for phase, kind, case_order in (
        ("el_nino", "warm", WARM_CASES),
        ("la_nina", "cold", COLD_CASES),
    ):
        peaks, selected = select_composite_peaks(
            target_values,
            target_steps,
            data.target,
            kind=kind,
        )
        anchor_target_step = int(records[case_order[0]]["target_step"])
        if anchor_target_step not in set(map(int, selected)):
            raise ValueError(f"The {phase} target is absent from its top 10% events.")
        composite_vectors: dict[str, np.ndarray] = {}
        for case_id, lead in zip(case_order, LEADS, strict=True):
            input_steps = selected - lead * data.steps_per_month
            lead_split = data.fixed_supervised_split(lead)
            if np.any(input_steps < lead_split.test_inputs[0]) or np.any(
                input_steps > lead_split.test_inputs[-1]
            ):
                raise ValueError(f"The {phase} composite crosses the test boundary.")
            physical = data.load_inputs(input_steps)
            standardized = (
                np.asarray(physical, dtype=np.float64) - normalization_mean
            ) / normalization_scale
            composite = standardized.mean(axis=0, dtype=np.float64).astype(np.float32)
            if composite.shape != data.input_shape or not np.isfinite(composite).all():
                raise ValueError(f"The {phase} {lead}-month composite is invalid.")
            cases[case_id]["Composite"] = composite
            composite_vectors[case_id] = composite
        selected_values = np.asarray(data.target[selected], dtype=np.float64)
        outputs[phase] = {
            "episode_count": int(peaks.size),
            "selected_count": int(selected.size),
            "all_complete_event_peak_steps": peaks.tolist(),
            "top_event_peak_steps_ranked": selected.tolist(),
            "top_event_peak_nino3_c_ranked": selected_values.tolist(),
            "cutoff_peak_nino3_c": float(selected_values[-1]),
            "composite_sha256": {
                case_id: sha256_array(values)
                for case_id, values in composite_vectors.items()
            },
        }
    return outputs


def _event_label(case_id: str, record: dict[str, Any]) -> str:
    phase = "La Niña" if case_id.startswith("la_nina") else "El Niño"
    value = f"{float(record['target_nino3_c']):.2f}".replace("-", "−")
    return f"{phase} ({value} °C)"


def _lead_title(record: dict[str, Any]) -> str:
    return f"{int(record['lead_months'])}-month lead"


def _display_row_labels(
    case_order: tuple[str, ...], records: dict[str, Any]
) -> tuple[str, str, str]:
    anchor = case_order[0]
    return (_event_label(anchor, records[anchor]), "10% Composite", "AGOP XAI")


def _normalized_scalar(values: np.ndarray, label: str) -> np.ndarray:
    maximum = float(np.max(np.abs(values)))
    if maximum <= np.finfo(np.float64).eps:
        LOGGER.warning("%s has a zero scalar field", label)
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


def _draw_panel(
    axis: plt.Axes,
    values: np.ndarray,
    data: ZCData,
    *,
    scalar_field: Literal["thermocline_depth", "sst_anomaly"],
    colormap: Any,
    color_levels: np.ndarray,
    label: str,
    show_latitude: bool,
) -> None:
    spatial, _ = split_spatial_phase(values, data)
    scalar = _normalized_scalar(
        spatial[data.spatial_field_names.index(scalar_field)], label
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
    axis.contourf(
        contour_longitudes,
        contour_latitudes,
        np.pad(scalar, ((1, 1), (1, 1)), mode="edge"),
        levels=color_levels,
        cmap=colormap,
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
        zorder=4,
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


def _add_colorbar(
    figure: plt.Figure,
    bounds: tuple[float, float, float, float],
    *,
    colormap: Any,
    color_levels: np.ndarray,
    field_label: str,
) -> None:
    axis = figure.add_axes(bounds)
    colorbar = figure.colorbar(
        ScalarMappable(
            norm=BoundaryNorm(color_levels, ncolors=colormap.N),
            cmap=colormap,
        ),
        cax=axis,
        orientation="horizontal",
        extend="both",
        ticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    colorbar.set_label(field_label, fontsize=9.5, labelpad=2.0)
    colorbar.ax.tick_params(labelsize=8.0, length=2.0, pad=1.5)


def _colorbar_bottom(panel: str) -> float:
    if panel == "a":
        return COLORBAR_BOTTOM
    if panel == "b":
        return PANEL_B_COLORBAR_BOTTOM
    raise ValueError(f"Unknown Figures 8-9 panel: {panel!r}")


def _plot_panel(
    data: ZCData,
    cases: dict[str, dict[str, np.ndarray]],
    records: dict[str, Any],
    case_order: tuple[str, ...],
    output: Path,
    *,
    panel: str,
    dpi: int,
    overwrite: bool,
) -> None:
    with plt.rc_context(figure_style()):
        figure = plt.figure(
            figsize=(THREE_COLUMN_FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
            constrained_layout=False,
        )
        grid = figure.add_gridspec(
            3,
            3,
            left=THREE_COLUMN_GRID_LEFT,
            right=THREE_COLUMN_GRID_RIGHT,
            bottom=GRID_BOTTOM,
            top=GRID_TOP,
            wspace=THREE_COLUMN_WSPACE,
            hspace=GRID_HSPACE,
        )
        display_rows = _display_row_labels(case_order, records)
        for row, (row_key, row_label) in enumerate(
            zip(ROWS, display_rows, strict=True)
        ):
            for column, case_id in enumerate(case_order):
                axis = figure.add_subplot(grid[row, column])
                scalar_field = "thermocline_depth" if column == 0 else "sst_anomaly"
                colormap = THERMOCLINE_COLORMAP if column == 0 else SST_COLORMAP
                color_levels = (
                    THERMOCLINE_COLOR_LEVELS if column == 0 else SST_COLOR_LEVELS
                )
                _draw_panel(
                    axis,
                    cases[case_id][row_key],
                    data,
                    scalar_field=scalar_field,
                    colormap=colormap,
                    color_levels=color_levels,
                    label=f"{case_id} / {row_key}",
                    show_latitude=column == 0,
                )
                if row == 0:
                    axis.set_title(
                        _lead_title(records[case_id]), pad=6.0, fontsize=11.5
                    )
                if column == 0:
                    axis.text(
                        -0.18,
                        0.5,
                        row_label,
                        transform=axis.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                    )
        colorbar_bottom = _colorbar_bottom(panel)
        _add_colorbar(
            figure,
            (0.16, colorbar_bottom, 0.21, 0.017),
            colormap=THERMOCLINE_COLORMAP,
            color_levels=THERMOCLINE_COLOR_LEVELS,
            field_label="Thermocline-depth anomaly",
        )
        _add_colorbar(
            figure,
            (0.55, colorbar_bottom, 0.33, 0.017),
            colormap=SST_COLORMAP,
            color_levels=SST_COLOR_LEVELS,
            field_label="SST anomaly",
        )
        with atomic_output_path(output, overwrite=overwrite) as temporary:
            figure.savefig(temporary, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        plt.close(figure)


def _write_sidecar(
    path: Path,
    *,
    figure_number: int,
    panel: str,
    phase: str,
    cases: tuple[str, ...],
    records: dict[str, Any],
    composites: dict[str, Any],
    normalization_mean: np.ndarray,
    normalization_scale: np.ndarray,
    data: ZCData,
    bundle: Any,
    overwrite: bool,
) -> None:
    event = records[cases[0]]
    write_json(
        path.with_suffix(".json"),
        {
            "schema_version": SCHEMA_VERSION,
            "figure": figure_number,
            "phase": phase,
            "data": data.provenance(),
            "case_order": list(cases),
            "event_target_step": int(event["target_step"]),
            "event_target_nino3_c": float(event["target_nino3_c"]),
            "row_keys": list(ROWS),
            "rows": list(_display_row_labels(cases, records)),
            "target_lock_policy": (
                "all three columns show predictors of the same physical target date"
            ),
            "composite": {
                "candidate_population": (
                    "target dates associated with 10-month predictors in the fixed "
                    "1,000-year held-out test block"
                ),
                "episode_rule": (
                    "one extremum per complete contiguous threshold episode; "
                    "boundary-truncated episodes excluded"
                ),
                "threshold_nino3_c": (
                    EVENT_THRESHOLD_C if phase == "el_nino" else -EVENT_THRESHOLD_C
                ),
                "ranking_rule": (
                    "strongest ceil(10 percent) of event extrema, with equal weight "
                    "per event"
                ),
                "lead_policy": (
                    "each column averages standardized states at its stated lead "
                    "relative to every selected event peak"
                ),
                **composites[phase],
            },
            "coordinate_policy": {
                "input_and_composite": (
                    "featurewise training-standardized coordinates using all "
                    "360,000 states in the fixed 10,000-year training block"
                ),
                "agop": (
                    "unit M^(1/2)x explanation in the same standardized coordinates"
                ),
                "phase": "two annual-phase scalars omitted from maps",
                "normalization_mean_sha256": sha256_array(normalization_mean),
                "normalization_scale_sha256": sha256_array(normalization_scale),
            },
            "visualization": {
                "scalar_fields_by_lead_months": {
                    "10": "thermocline_depth",
                    "5": "sst_anomaly",
                    "1": "sst_anomaly",
                },
                "thermocline_colormap": "Matplotlib bwr resampled to 256 colors",
                "sst_colormap": (
                    "18-color IBM blue-white-orange (#648FFF, #FFFFFF, #FE6100); "
                    "even lookup table with no exactly white entry"
                ),
                "color_levels_by_scalar_field": {
                    "thermocline_depth": THERMOCLINE_COLOR_LEVELS.tolist(),
                    "sst_anomaly": SST_COLOR_LEVELS.tolist(),
                },
                "filled_color_intervals_by_scalar_field": {
                    "thermocline_depth": int(THERMOCLINE_COLOR_LEVELS.size - 1),
                    "sst_anomaly": int(SST_COLOR_LEVELS.size - 1),
                },
                "sst_zero_boundary": {
                    "value": float(SST_COLOR_LEVELS[9]),
                    "exact_white_lookup_entry": False,
                    "negative_adjacent_color": to_hex(SST_COLORMAP(8)),
                    "positive_adjacent_color": to_hex(SST_COLORMAP(9)),
                },
                "colorbar_labels_by_scalar_field": {
                    "thermocline_depth": "Thermocline-depth anomaly",
                    "sst_anomaly": "SST anomaly",
                },
                "colorbar_label_position": "below",
                "colorbar_bottom": _colorbar_bottom(panel),
                "scalar_normalization": "within-panel maximum absolute value",
                "vector_fields": [
                    "zonal_ocean_current",
                    "meridional_ocean_current",
                ],
                "vector_interpretation": "depth-averaged ocean-current fields",
                "vector_normalization": "within-panel maximum vector magnitude",
                "quiver_scale": QUIVER_SCALE,
                "quiver_width": QUIVER_WIDTH,
                "horizontal_geometry_reference": "fresh Figure 6",
                "figure_width_inches": THREE_COLUMN_FIGURE_WIDTH_INCHES,
                "horizontal_grid_left": THREE_COLUMN_GRID_LEFT,
                "horizontal_grid_right": THREE_COLUMN_GRID_RIGHT,
                "horizontal_panel_wspace": THREE_COLUMN_WSPACE,
                "longitude_cell_edges": list(cell_edge_bounds(data.longitudes)),
                "latitude_cell_edges": list(cell_edge_bounds(data.latitudes)),
            },
            "source": {
                "bundle_metadata_sha256": sha256_file(bundle.metadata_path),
                "bundle_completion_sha256": sha256_file(bundle.completion_path),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "output": {"file": path.name, "sha256": sha256_file(path)},
        },
        overwrite=overwrite,
    )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    output_a = args.output_a.expanduser().resolve()
    output_b = args.output_b.expanduser().resolve()
    for output in (output_a, output_b):
        if output.suffix.lower() not in {".pdf", ".png"}:
            raise ValueError("Figure outputs must end in .pdf or .png")
    if output_a == output_b:
        raise ValueError("--output-a and --output-b must be different files.")
    if not args.overwrite:
        existing = [
            path
            for output in (output_a, output_b)
            for path in (output, output.with_suffix(".json"))
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Figure output already exists; use --overwrite:\n  "
                + "\n  ".join(str(path) for path in existing)
            )
    bundle = None
    errors: list[Exception] = []
    for artifact in (ARTIFACT, LEGACY_ARTIFACT):
        try:
            bundle = load_completed_bundle(
                args.bundle_dir,
                expected_artifact=artifact,
                required_files=("explanations", "references"),
            )
            break
        except (FileNotFoundError, ValueError) as error:
            errors.append(error)
    if bundle is None:
        raise ValueError(
            "The Figures 8-9 bundle is neither a canonical bundle nor a "
            "validated legacy Figure 8 bundle."
        ) from errors[-1]
    data = ZCData(args.data_dir, input_profile="core4")
    cases, records, normalization_mean, normalization_scale = _load_cases(bundle, data)
    composites = _build_composites(
        data,
        cases,
        records,
        normalization_mean,
        normalization_scale,
    )
    for figure_number, panel, phase, case_order, output in (
        (8, "a", "el_nino", WARM_CASES, output_a),
        (9, "b", "la_nina", COLD_CASES, output_b),
    ):
        _plot_panel(
            data,
            cases,
            records,
            case_order,
            output,
            panel=panel,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
        _write_sidecar(
            output,
            figure_number=figure_number,
            panel=panel,
            phase=phase,
            cases=case_order,
            records=records,
            composites=composites,
            normalization_mean=normalization_mean,
            normalization_scale=normalization_scale,
            data=data,
            bundle=bundle,
            overwrite=args.overwrite,
        )
        LOGGER.info("Fresh Figure %d saved: %s", figure_number, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
