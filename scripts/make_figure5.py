#!/usr/bin/env python3
"""Plot revised Figure 5 from the cached tidy experiment CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.io import atomic_output_path, sha256_file, write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "5.1.0"
COLORS = {"mlp": "#fe6100", "cnn": "#648fff", "vit": "#ffb000"}
LABELS = {"mlp": "MLP", "cnn": "CNN", "vit": "ViT"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create revised Figure 5 from run_figure5_experiments.py output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "outputs/manuscript/source_data/figure5/figure5_zcv3_results.csv"
        ),
        help="Audited five-seed Figure 5 results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manuscript/figures/figure5_predictive_skill.pdf"),
        help="PNG or PDF figure path.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_rows(path: Path) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    **raw,
                    "lead_months": int(raw["lead_months"]),
                    "train_years_requested": float(raw["train_years_requested"]),
                    "repetition": int(raw["repetition"]),
                    "seed": int(raw["seed"]),
                    "test_r2": float(raw["test_r2"]),
                }
            )
    if not rows:
        raise ValueError(f"No experiment rows found in {path}.")
    return rows


def _means_and_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), standard_deviation


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results_path = args.results.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("--output must end in .png or .pdf")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    metadata_path = output.with_suffix(".json")
    if not args.overwrite:
        existing = [str(path) for path in (output, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist:\n  "
                + "\n  ".join(existing)
                + "\nUse --overwrite to replace them."
            )
    rows = _read_rows(results_path)

    left: dict[tuple[str, float], list[float]] = defaultdict(list)
    right: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for row in rows:
        architecture = str(row["architecture"])
        years = float(row["train_years_requested"])
        score = float(row["test_r2"])
        if row["panel"] == "training_size":
            left[(architecture, years)].append(score)
        elif row["panel"] == "lead_time":
            right[(architecture, years, int(row["lead_months"]))].append(score)

    style = {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with plt.rc_context(style):
        fig, (left_ax, right_ax) = plt.subplots(
            1,
            2,
            figsize=(10.5, 4.5),
            sharey=True,
            constrained_layout=True,
        )
        architectures = [
            name
            for name in ("mlp", "cnn", "vit")
            if any(key[0] == name for key in left)
        ]
        for architecture in architectures:
            years = sorted(key[1] for key in left if key[0] == architecture)
            summary = [_means_and_sd(left[(architecture, value)]) for value in years]
            means = np.asarray([item[0] for item in summary])
            left_ax.plot(
                years,
                means,
                color=COLORS[architecture],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                zorder=2,
            )
        left_ax.set_xscale("log")
        all_years = sorted({key[1] for key in left})
        left_ax.set_xticks(all_years, [f"{value:g}" for value in all_years])
        left_ax.set_xlabel("Training data (model years)")
        left_ax.set_ylabel(r"$R^2$")

        right_years = sorted({key[1] for key in right}, reverse=True)
        line_styles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^", "D"]
        for architecture in architectures:
            for style_index, years in enumerate(right_years):
                leads = sorted(
                    key[2]
                    for key in right
                    if key[0] == architecture and math.isclose(key[1], years)
                )
                summaries = [
                    _means_and_sd(right[(architecture, years, lead)])
                    for lead in leads
                ]
                means = np.asarray([item[0] for item in summaries])
                right_ax.plot(
                    leads,
                    means,
                    color=COLORS[architecture],
                    linestyle=line_styles[style_index % len(line_styles)],
                    marker=markers[style_index % len(markers)],
                    markersize=4,
                    linewidth=1.7,
                    zorder=2,
                )
        right_ax.set_xlabel("Lead time (months)")
        right_ax.set_xticks(sorted({key[2] for key in right}))

        for axis in (left_ax, right_ax):
            axis.grid(alpha=0.22, linewidth=0.6)
            axis.set_ylim(min(0.35, axis.get_ylim()[0]), 1.02)
        architecture_handles = [
            Line2D([0], [0], color=COLORS[name], linewidth=2, label=LABELS[name])
            for name in architectures
        ]
        fig.legend(
            handles=architecture_handles,
            frameon=False,
            loc="outside lower center",
            ncol=len(architecture_handles),
        )
        with atomic_output_path(output, overwrite=args.overwrite) as temporary_path:
            fig.savefig(temporary_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    write_json(
        metadata_path,
        {
            "schema_version": 4,
            "figure": 5,
            "generator": {
                "script": "scripts/make_figure5.py",
                "version": SCRIPT_VERSION,
            },
            "palette": {
                "name": "IBM color-blind-safe",
                "architecture_colors": COLORS,
            },
            "results_file": results_path.name,
            "results_sha256": sha256_file(results_path),
            "left_uncertainty": "not displayed",
            "right_uncertainty": "not displayed",
            "right_repetition_counts": sorted(
                {len(values) for values in right.values()}
            ),
            "left_repetition_counts": sorted(
                {len(values) for values in left.values()}
            ),
            "y_axis_label": "$R^2$",
            "panel_titles": [],
            "legend": {
                "architecture_labels": [LABELS[name] for name in architectures],
                "data_regime_labels": [],
                "numeric_training_years_shown": False,
            },
            "left_x_axis": (
                "Requested contiguous training years selected inside the fixed "
                "10000-year training block. Validation and testing use their "
                "separate fixed 1000-year blocks."
            ),
            "right_panel_target_dates": (
                "Identical across leads within each architecture/training-size series."
            ),
            "output": {
                "file": output.name,
                "sha256": sha256_file(output),
            },
        },
        overwrite=args.overwrite,
    )
    LOGGER.info("Figure 5 saved: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
