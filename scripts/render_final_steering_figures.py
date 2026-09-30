#!/usr/bin/env python3
"""Render manuscript Figures 10--12 from the compact public plot bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    sha256_array,
    sha256_file,
    write_json,
)
from zc_xai.steering_figures import (  # noqa: E402
    FIGURE10_EVENT_ORDER,
    FIGURE12_EVENT_ORDER,
    FIGURE12_METHOD_KEYS,
    FIGURE12_METHOD_ORDER,
    build_figure10,
    build_figure11,
    build_figure12,
)
from zc_xai.steering_plot_bundle import (  # noqa: E402
    SteeringPlotBundle,
    load_steering_plot_bundle,
)

SCRIPT_VERSION = "1.0.0"
DEFAULT_BUNDLE_DIR = Path("artifacts/zc-v3/manuscript/steering/final_plot_data")
DEFAULT_OUTPUT_DIR = Path("outputs/manuscript/figures")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (REPOSITORY_ROOT / expanded).resolve()
    )


def _output_record(output: Path) -> dict[str, Any]:
    return {
        "file": output.name,
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
    }


def _bundle_record(bundle: SteeringPlotBundle) -> dict[str, Any]:
    return {
        "manifest_file": bundle.manifest_path.name,
        "manifest_sha256": sha256_file(bundle.manifest_path),
        "archive_file": bundle.archive_path.name,
        "archive_sha256": sha256_file(bundle.archive_path),
        "accepted_reports_file": bundle.reports_path.name,
        "accepted_reports_sha256": sha256_file(bundle.reports_path),
    }


def _save_figure(
    figure: plt.Figure,
    output: Path,
    *,
    dpi: int,
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    with atomic_output_path(output, overwrite=overwrite) as temporary:
        figure.savefig(temporary, dpi=dpi, metadata=metadata)
    plt.close(figure)


def _render_figure10(
    bundle: SteeringPlotBundle,
    output: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> None:
    figure = build_figure10(bundle)
    _save_figure(
        figure,
        output,
        dpi=dpi,
        overwrite=overwrite,
        metadata={
            "Subject": (
                "Authentic extreme El Nino and La Nina trajectories under "
                "matched signed nonlinear pooled-covariance-action doses"
            ),
            "Creator": "render_final_steering_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    results = {}
    for event in FIGURE10_EVENT_ORDER:
        prefix = f"figure10__{event}"
        coefficients = bundle.array(f"{prefix}__coefficients")
        trajectories = bundle.array(f"{prefix}__nudged_nino3_c")
        record = bundle.manifest["figures"]["figure10"]["events"][event]
        results[event] = {
            "event_input_index": record["event_input_index"],
            "event_target_index": record["event_target_index"],
            "original_target_nino3_c": record["original_target_nino3_c"],
            "coefficients": coefficients.tolist(),
            "terminal_nino3_c": trajectories[:, -1].tolist(),
            "trajectory_sha256": [
                sha256_array(trajectory) for trajectory in trajectories
            ],
        }
    write_json(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact": "zc-extreme-event-agop-dose-response-figure",
            "renderer_version": SCRIPT_VERSION,
            "rendered_from": _bundle_record(bundle),
            "output": _output_record(output),
            "results": results,
        },
        overwrite=overwrite,
    )


def _render_figure11(
    bundle: SteeringPlotBundle,
    output: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> None:
    figure = build_figure11(bundle)
    _save_figure(
        figure,
        output,
        dpi=dpi,
        overwrite=overwrite,
        metadata={
            "Title": "Figure 11: matched AGOP XAI El Nino and La Nina trajectories",
            "Subject": (
                "Ten continuous unnudged controls and every accepted warm "
                "and cold AGOP intervention"
            ),
            "Creator": "render_final_steering_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    control = bundle.array("figure11__control_nino3_c")
    warm = bundle.array("figure11__warm_nudged_nino3_c")
    cold = bundle.array("figure11__cold_nudged_nino3_c")
    write_json(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact": "manuscript-figure-11-matched-agop-xai-trajectories",
            "renderer_version": SCRIPT_VERSION,
            "rendered_from": _bundle_record(bundle),
            "output": _output_record(output),
            "members": bundle.manifest["figures"]["figure11"],
            "results": {
                "continuous_control_sha256": sha256_array(control),
                "warm_nudged_sha256": sha256_array(warm),
                "cold_nudged_sha256": sha256_array(cold),
                "warm_nudged_terminal_nino3_c": warm[:, -1].tolist(),
                "cold_nudged_terminal_nino3_c": cold[:, -1].tolist(),
            },
        },
        overwrite=overwrite,
    )


def _render_figure12(
    bundle: SteeringPlotBundle,
    output: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> None:
    figure, y_limits = build_figure12(bundle)
    _save_figure(
        figure,
        output,
        dpi=dpi,
        overwrite=overwrite,
        metadata={
            "Title": "Mean ZC responses to XAI and composite nudging",
            "Subject": (
                "Mean paired Nino-3 responses for completed strict-gate "
                "nudging experiments"
            ),
            "Creator": "render_final_steering_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plotted_hashes = {
        event: {
            method: sha256_array(
                bundle.array(
                    f"figure12__{event}__{FIGURE12_METHOD_KEYS[method]}"
                    "__plotted_mean_nino3_c"
                )
            )
            for method in FIGURE12_METHOD_ORDER
        }
        for event in FIGURE12_EVENT_ORDER
    }
    write_json(
        output.with_suffix(".json"),
        {
            "schema_version": 1,
            "artifact": "method-specific-mean-paired-zc-steering-figure",
            "renderer_version": SCRIPT_VERSION,
            "rendered_from": _bundle_record(bundle),
            "output": _output_record(output),
            "display": {
                "method_order": list(FIGURE12_METHOD_ORDER),
                "panel_order": list(FIGURE12_EVENT_ORDER),
                "y_limits_c": list(y_limits),
            },
            "results": bundle.manifest["figures"]["figure12"],
            "plotted_mean_sha256": plotted_hashes,
        },
        overwrite=overwrite,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    bundle = load_steering_plot_bundle(_resolve(args.bundle_dir))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "figure10_extreme_agop_dose_response.pdf",
        output_dir / "figure11_agop_warm_cold_paired_trajectories.pdf",
        output_dir / "figure12_xai_method_mean_trajectories.pdf",
    )
    if not args.overwrite:
        for output in outputs:
            if output.exists() or output.with_suffix(".json").exists():
                raise FileExistsError(f"Output exists; use --overwrite: {output}")
    _render_figure10(bundle, outputs[0], dpi=args.dpi, overwrite=args.overwrite)
    _render_figure11(bundle, outputs[1], dpi=args.dpi, overwrite=args.overwrite)
    _render_figure12(bundle, outputs[2], dpi=args.dpi, overwrite=args.overwrite)
    print(f"Rendered Figures 10--12 from {bundle.directory}")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
