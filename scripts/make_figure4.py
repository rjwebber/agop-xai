#!/usr/bin/env python3
"""Plot aligned 5- and 10-month CNN forecasts over fresh held-out ZC data."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.io import atomic_output_path, sha256_file, write_json  # noqa: E402
from zc_xai.training import (  # noqa: E402
    ExperimentSpec,
    TrainingConfig,
    load_experiment,
    predict,
    train_experiment,
)

LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "4.1.0"
COLORS = {
    "nino3": "#648fff",
    "forecast_5m": "#fe6100",
    "forecast_10m": "#ffb000",
}
LEGEND_LOCATION = "outside upper center"
LEGEND_COLUMNS = 3


def positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create revised Figure 4 using CNN rather than MLP forecasts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/zc-v3")
    )
    parser.add_argument(
        "--input-profile",
        choices=("core4",),
        default="core4",
        help="Interpretation-first fresh-data input profile.",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts/zc-v3")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manuscript/figures/figure4_cnn_forecasts.pdf"),
        help="PNG or PDF figure path.",
    )
    parser.add_argument("--data-output", type=Path, help="Optional forecast CSV path.")
    parser.add_argument("--train-years", type=positive_float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-start-year", type=float, default=0.0)
    parser.add_argument("--segment-years", type=positive_float, default=100.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--training-batch-size", type=int, default=256)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_or_train(
    data: ZCData,
    args: argparse.Namespace,
    lead_months: int,
):
    spec = ExperimentSpec(
        architecture="cnn",
        lead_months=lead_months,
        train_years=args.train_years,
        seed=args.seed,
        input_profile=(
            args.input_profile
            if data.metadata.get("schema_version") == FRESH_SCHEMA_VERSION
            else None
        ),
    )
    try:
        return load_experiment(data, args.artifacts_dir, spec, device="cpu")
    except FileNotFoundError:
        if not args.train_if_missing:
            raise FileNotFoundError(
                f"CNN checkpoint for lead {lead_months} months is missing. "
                "Run scripts/train_zc_model.py first or add --train-if-missing."
            ) from None
        config = TrainingConfig(
            batch_size=args.training_batch_size,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
        )
        return train_experiment(
            data,
            args.artifacts_dir,
            spec,
            config,
            device=args.device,
        )


def select_common_test_target_steps(
    data: ZCData,
    *,
    lead_months: tuple[int, ...],
    segment_start_year: float,
    segment_years: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Choose target dates available at every lead in the held-out test block."""

    if not lead_months or any(lead <= 0 for lead in lead_months):
        raise ValueError("lead_months must contain positive values")
    if data.metadata.get("schema_version") == FRESH_SCHEMA_VERSION:
        splits = [data.fixed_supervised_split(lead) for lead in lead_months]
        test_blocks = {split.test_block for split in splits}
        if len(test_blocks) != 1:
            raise ValueError("Fresh lead-time splits do not share one test block.")
        common_start = max(
            split.test_block[0] + split.lead_steps for split in splits
        )
        common_stop = min(split.test_block[1] for split in splits)
    else:
        splits = [data.supervised_split(lead) for lead in lead_months]
        common_start = max(split.split_step + split.lead_steps for split in splits)
        common_stop = data.n_time_steps

    offset_steps = int(round(segment_start_year * data.steps_per_year))
    segment_steps = int(round(segment_years * data.steps_per_year))
    first_target = common_start + offset_steps
    stop_target = first_target + segment_steps
    if first_target < common_start or stop_target > common_stop:
        available_years = (common_stop - common_start) / data.steps_per_year
        raise ValueError(
            "The requested Figure 4 segment lies outside the common held-out "
            f"target period of {available_years:.6g} years."
        )
    return (
        np.arange(first_target, stop_target, dtype=np.int64),
        (common_start, common_stop),
    )


def _write_forecast_csv(
    path: Path,
    *,
    times: np.ndarray,
    target_steps: np.ndarray,
    truth: np.ndarray,
    forecast_5m: np.ndarray,
    forecast_10m: np.ndarray,
    overwrite: bool,
) -> None:
    with (
        atomic_output_path(path, overwrite=overwrite) as temporary_path,
        temporary_path.open("w", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.writer(stream)
        writer.writerow(
            ("time_years", "target_step", "nino3_c", "cnn_5m_c", "cnn_10m_c")
        )
        writer.writerows(
            zip(times, target_steps, truth, forecast_5m, forecast_10m, strict=True)
        )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.segment_start_year < 0.0 or not np.isfinite(args.segment_start_year):
        raise ValueError("--segment-start-year must be finite and nonnegative")
    if args.dpi <= 0 or args.prediction_batch_size <= 0:
        raise ValueError("--dpi and --prediction-batch-size must be positive")
    output = args.output.expanduser().resolve()
    if output.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("--output must end in .png or .pdf")
    data_output = (
        args.data_output.expanduser().resolve()
        if args.data_output is not None
        else output.with_suffix(".csv")
    )
    if data_output.suffix.lower() != ".csv":
        raise ValueError("--data-output must end in .csv")
    metadata_path = output.with_suffix(".json")
    output_paths = (output, data_output, metadata_path)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Figure, forecast-data, and metadata paths must be distinct.")
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist:\n  "
                + "\n  ".join(existing)
                + "\nUse --overwrite to replace them."
            )

    data = ZCData(args.data_dir, input_profile=args.input_profile)
    is_fresh = data.metadata.get("schema_version") == FRESH_SCHEMA_VERSION
    if is_fresh and (args.train_years != 10_000.0 or data.input_profile != "core4"):
        raise ValueError(
            "Fresh Figure 4 requires --train-years 10000 and --input-profile core4."
        )
    experiments = {lead: _load_or_train(data, args, lead) for lead in (5, 10)}
    target_steps, common_target_block = select_common_test_target_steps(
        data,
        lead_months=(5, 10),
        segment_start_year=args.segment_start_year,
        segment_years=args.segment_years,
    )
    truth = np.asarray(data.target[target_steps], dtype=np.float32)
    forecasts: dict[int, np.ndarray] = {}
    for lead, experiment in experiments.items():
        input_steps = target_steps - experiment.lead_steps
        if is_fresh:
            expected_test = data.fixed_supervised_split(lead).test_inputs
            if not np.array_equal(experiment.test_inputs, expected_test):
                raise ValueError(
                    f"The cached {lead}-month CNN does not use the canonical "
                    "fresh test block."
                )
        if input_steps[0] < experiment.test_inputs[0] or input_steps[-1] > (
            experiment.test_inputs[-1]
        ):
            raise ValueError(
                f"The common target dates leave the {lead}-month held-out inputs."
            )
        forecasts[lead] = predict(
            experiment,
            data,
            input_steps,
            batch_size=args.prediction_batch_size,
            device=args.device,
        )
    times = np.arange(target_steps.size, dtype=np.float64) / data.steps_per_year

    style = {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10.2, 4.5), constrained_layout=True)
        ax.plot(
            times,
            truth,
            color=COLORS["nino3"],
            linewidth=1.35,
            label="Niño-3 anomaly",
        )
        ax.plot(
            times,
            forecasts[5],
            color=COLORS["forecast_5m"],
            linewidth=1.05,
            alpha=0.95,
            label="5-month lead forecast",
        )
        ax.plot(
            times,
            forecasts[10],
            color=COLORS["forecast_10m"],
            linewidth=1.05,
            alpha=0.95,
            label="10-month lead forecast",
        )
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8, zorder=0)
        ax.set(
            xlabel="Time (years)",
            ylabel="Niño-3 anomaly (°C)",
        )
        ax.set_xlim(0.0, args.segment_years)
        ax.grid(alpha=0.18, linewidth=0.6)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=LEGEND_COLUMNS,
            loc=LEGEND_LOCATION,
        )
        with atomic_output_path(output, overwrite=args.overwrite) as temporary_path:
            fig.savefig(temporary_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    _write_forecast_csv(
        data_output,
        times=times,
        target_steps=target_steps,
        truth=truth,
        forecast_5m=forecasts[5],
        forecast_10m=forecasts[10],
        overwrite=args.overwrite,
    )
    metadata = {
        "schema_version": 3,
        "figure": 4,
        "generator": {
            "script": "scripts/make_figure4.py",
            "version": SCRIPT_VERSION,
        },
        "description": "Aligned held-out CNN forecasts plotted at their target times.",
        "data": data.provenance(),
        "preprocessing": {
            "inputs": (
                "Each selected field-grid coordinate and both annual-phase "
                "coordinates are centered and scaled using all raw states in the "
                "fixed 10,000-year training block only."
            ),
            "target": (
                "The plotted target and forecasts remain in the original Nino-3 "
                "SST-anomaly units; no training-period target mean is subtracted."
            ),
        },
        "segment": {
            "selection": "common target dates available at both 5- and 10-month leads",
            "start_year_within_test": args.segment_start_year,
            "duration_years": args.segment_years,
            "first_target_step": int(target_steps[0]),
            "last_target_step": int(target_steps[-1]),
            "common_target_block_start_step": common_target_block[0],
            "common_target_block_stop_step_exclusive": common_target_block[1],
            "sample_interval_model_months": 1.0 / data.steps_per_month,
        },
        "plot": {
            "palette": {
                "name": "IBM color-blind-safe",
                "series_colors": COLORS,
            },
            "title": None,
            "x_axis": "Time (years)",
            "y_axis": "Nino-3 anomaly (degrees C)",
            "legend_location": LEGEND_LOCATION,
            "legend_rows": 1,
            "legend_columns": LEGEND_COLUMNS,
        },
        "models": {
            str(lead): {
                "artifact_directory": str(
                    experiment.artifact_dir.relative_to(
                        args.artifacts_dir.expanduser().resolve()
                    )
                ),
                "checkpoint_sha256": experiment.checkpoint_sha256,
                "test_r2": experiment.metrics["test_r2"],
            }
            for lead, experiment in experiments.items()
        },
        "figure_output": output.name,
        "figure_output_sha256": sha256_file(output),
        "data_output": data_output.name,
        "data_output_sha256": sha256_file(data_output),
    }
    write_json(metadata_path, metadata, overwrite=args.overwrite)
    LOGGER.info("Figure 4 saved: %s", output)
    LOGGER.info("Forecast data saved: %s", data_output)
    LOGGER.info("Metadata saved: %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
