#!/usr/bin/env python3
"""Render Figure 6 from the completed fresh core4 Table I artifact."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import ZCData  # noqa: E402
from zc_xai.fresh_xai_outputs import (  # noqa: E402
    FRESH_TABLE_ARCHITECTURES,
    FRESH_TABLE_METHODS,
    FRESH_TABLE_SCHEMA_VERSION,
    FRESH_XAI_TRAINING_POPULATION,
    phase_coefficient_summary,
    validate_unit_explanation,
)
from zc_xai.fresh_xai_plotting import (  # noqa: E402
    ARCHITECTURE_LABELS,
    COLOR_LEVELS,
    COLORMAP_DESCRIPTION,
    COLORMAP_LOOKUP_TABLE_SIZE,
    METHOD_LABELS,
    PANEL_BOX_ASPECT,
    QUIVER_SCALE,
    QUIVER_WIDTH,
    THREE_COLUMN_FIGURE_WIDTH_INCHES,
    THREE_COLUMN_GRID_LEFT,
    THREE_COLUMN_GRID_RIGHT,
    THREE_COLUMN_WSPACE,
    add_horizontal_colorbar,
    cell_edge_bounds,
    draw_ocean_panel,
    figure_style,
    save_figure,
)
from zc_xai.io import load_json, sha256_array, sha256_file, write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)

THERMOCLINE_COLORBAR_LABEL = "Thermocline-depth anomaly"
COLORBAR_VERTICAL_FRACTION = 0.48


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the fresh core4 Figure 6 explanation grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("artifacts/zc-v3/manuscript/table1_figure6"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manuscript/figures/figure6_fresh_core4.pdf"),
    )
    parser.add_argument("--dpi", type=positive_int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_figure_inputs(
    data: ZCData,
    table_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray], Path]:
    """Load and bind a completed Table I explanation artifact to fresh data."""

    metadata_path = table_dir / "table1_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Table I metadata not found: {metadata_path}")
    metadata = load_json(metadata_path)
    identity = metadata.get("run_identity")
    if (
        metadata.get("schema_version") != FRESH_TABLE_SCHEMA_VERSION
        or metadata.get("status") != "complete"
        or not isinstance(identity, dict)
    ):
        raise ValueError("Fresh Table I metadata is incomplete or invalid.")
    robustness = identity.get("robustness")
    methods = identity.get("methods")
    gradient_shap = methods.get("gradient_shap") if isinstance(methods, dict) else None
    if (
        not isinstance(robustness, dict)
        or robustness.get("candidate_population")
        != FRESH_XAI_TRAINING_POPULATION
        or not isinstance(gradient_shap, dict)
        or gradient_shap.get("reference_population")
        != FRESH_XAI_TRAINING_POPULATION
    ):
        raise ValueError("Fresh Table I XAI references are not training-only.")
    recorded_data = identity.get("data")
    if (
        not isinstance(recorded_data, dict)
        or recorded_data.get("metadata_sha256") != data.metadata_sha256
        or recorded_data.get("input_profile") != "core4"
    ):
        raise ValueError("Table I belongs to another processed-data view.")
    output_files = metadata.get("output_files")
    if not isinstance(output_files, dict):
        raise ValueError("Table I has no output manifest.")
    target_record = output_files.get("target_explanations")
    if not isinstance(target_record, dict):
        raise ValueError("Table I has no target-explanation record.")
    filename = target_record.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Unsafe target-explanation filename in Table I metadata.")
    explanations_path = table_dir / filename
    if sha256_file(explanations_path) != target_record.get("sha256"):
        raise ValueError("Target-explanation artifact failed its checksum.")

    explanations: dict[str, np.ndarray] = {}
    with np.load(explanations_path, allow_pickle=False) as archive:
        event_input = np.asarray(archive["event_input_standardized"], dtype=np.float64)
        if event_input.shape != data.input_shape or not np.isfinite(event_input).all():
            raise ValueError("The standardized event input has an invalid schema.")
        event_input_step = int(np.asarray(archive["event_input_step"]).item())
        event_target_step = int(np.asarray(archive["event_target_step"]).item())
        event_target_nino3 = float(
            np.asarray(archive["event_target_nino3_c"]).item()
        )
        event = identity.get("event")
        if not isinstance(event, dict) or (
            event_input_step != event.get("input_index")
            or event_target_step != event.get("target_index")
            or not math.isclose(
                event_target_nino3,
                float(event.get("target_nino3_c")),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise ValueError("Embedded event values disagree with Table I metadata.")
        for architecture in FRESH_TABLE_ARCHITECTURES:
            for method in FRESH_TABLE_METHODS:
                key = f"{architecture}_{method.lower()}"
                explanations[key] = validate_unit_explanation(
                    archive[key], data.n_features
                )
    return metadata, event_input, explanations, explanations_path


def _plot(
    data: ZCData,
    metadata: dict[str, Any],
    event_input: np.ndarray,
    explanations: dict[str, np.ndarray],
    output: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> None:
    event = metadata["run_identity"]["event"]
    target = float(event["target_nino3_c"])
    phase_label = "El Niño" if target >= 0.0 else "La Niña"
    with plt.rc_context(figure_style()):
        figure = plt.figure(
            figsize=(THREE_COLUMN_FIGURE_WIDTH_INCHES, 12.0),
            constrained_layout=False,
        )
        grid = figure.add_gridspec(
            6,
            3,
            left=THREE_COLUMN_GRID_LEFT,
            right=THREE_COLUMN_GRID_RIGHT,
            bottom=0.055,
            top=0.97,
            height_ratios=(1.0, 0.06, 1.0, 1.0, 1.0, 1.0),
            wspace=THREE_COLUMN_WSPACE,
            hspace=0.16,
        )
        input_axis = figure.add_subplot(grid[0, 0])
        draw_ocean_panel(
            input_axis,
            event_input,
            data,
            label="Input",
            show_latitude=True,
        )
        input_axis.set_title(
            f"{phase_label} ({target:.2f} °C)\n10-month lead",
            pad=6.0,
            fontsize=12.5,
        )
        input_axis.text(
            -0.18,
            0.5,
            "Input",
            transform=input_axis.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12.5,
        )
        colorbar_host = figure.add_subplot(grid[0, 1:])
        colorbar_host.set_axis_off()
        box = colorbar_host.get_position()
        colorbar_axis = figure.add_axes(
            [
                box.x0 + 0.08 * box.width,
                box.y0 + COLORBAR_VERTICAL_FRACTION * box.height,
                0.84 * box.width,
                0.13 * box.height,
            ]
        )
        add_horizontal_colorbar(
            figure,
            colorbar_axis,
            label=THERMOCLINE_COLORBAR_LABEL,
        )

        for method_index, (method_label, method_key) in enumerate(METHOD_LABELS):
            for column, (architecture_label, architecture) in enumerate(
                ARCHITECTURE_LABELS
            ):
                axis = figure.add_subplot(grid[method_index + 2, column])
                draw_ocean_panel(
                    axis,
                    explanations[f"{architecture}_{method_key}"],
                    data,
                    label=f"{architecture_label} / {method_label}",
                    show_latitude=column == 0,
                )
                if method_index == 0:
                    axis.set_title(architecture_label, pad=6.0, fontsize=12.5)
                if column == 0:
                    axis.text(
                        -0.18,
                        0.5,
                        method_label,
                        transform=axis.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12.5,
                    )
        save_figure(figure, output, dpi=dpi, overwrite=overwrite)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    output = args.output.expanduser().resolve()
    if output.suffix.lower() not in {".pdf", ".png"}:
        raise ValueError("--output must end in .pdf or .png")
    sidecar = output.with_suffix(".json")
    if not args.overwrite and (output.exists() or sidecar.exists()):
        raise FileExistsError("Figure output exists; pass --overwrite to replace it.")
    data = ZCData(args.data_dir, input_profile="core4")
    table_dir = args.table_dir.expanduser().resolve()
    metadata, event_input, explanations, explanations_path = load_figure_inputs(
        data, table_dir
    )
    _plot(
        data,
        metadata,
        event_input,
        explanations,
        output,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    phase_summaries = {
        key: phase_coefficient_summary(value, data)
        for key, value in explanations.items()
    }
    write_json(
        sidecar,
        {
            "schema_version": 1,
            "figure": 6,
            "data": data.provenance(),
            "table1_metadata_sha256": sha256_file(
                table_dir / "table1_metadata.json"
            ),
            "target_explanations_file": explanations_path.name,
            "target_explanations_sha256": sha256_file(explanations_path),
            "event": metadata["run_identity"]["event"],
            "input_coordinate_system": "training-standardized coordinates",
            "input_sha256": sha256_array(event_input.astype(np.float32)),
            "plotted_spatial_fields": [
                "thermocline_depth",
                "zonal_ocean_current",
                "meridional_ocean_current",
            ],
            "unplotted_spatial_field": "sst_anomaly",
            "unplotted_phase_features": [
                "annual_phase_sin",
                "annual_phase_cos",
            ],
            "phase_coefficients": phase_summaries,
            "visualization": {
                "colorbar_label": THERMOCLINE_COLORBAR_LABEL,
                "colorbar_vertical_fraction_within_host": (
                    COLORBAR_VERTICAL_FRACTION
                ),
                "colormap": COLORMAP_DESCRIPTION,
                "color_levels": COLOR_LEVELS.tolist(),
                "filled_color_intervals": int(COLOR_LEVELS.size - 1),
                "zero_is_level_boundary": True,
                "scalar_normalization": "within-panel maximum absolute value",
                "vector_normalization": "within-panel maximum vector magnitude",
                "quiver_scale": QUIVER_SCALE,
                "quiver_width": QUIVER_WIDTH,
                "panel_box_aspect_height_over_width": PANEL_BOX_ASPECT,
                "longitude_cell_edges": list(cell_edge_bounds(data.longitudes)),
                "latitude_cell_edges": list(cell_edge_bounds(data.latitudes)),
                "colormap_lookup_table_size": COLORMAP_LOOKUP_TABLE_SIZE,
            },
            "output": {
                "file": output.name,
                "sha256": sha256_file(output),
                "dpi": args.dpi,
            },
        },
        overwrite=args.overwrite,
    )
    LOGGER.info("Fresh Figure 6 saved: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
