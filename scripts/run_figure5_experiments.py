#!/usr/bin/env python3
"""Train/cache the fresh core-four model grid for Figure 5."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.figure5 import train_figure5_experiment  # noqa: E402
from zc_xai.io import atomic_output_path, write_json  # noqa: E402
from zc_xai.models import ARCHITECTURES  # noqa: E402
from zc_xai.training import ExperimentSpec, TrainingConfig  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_TRAINING_YEARS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
DEFAULT_LEAD_MONTHS = tuple(range(1, 13))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the complete fresh-data core-four experiment grid for "
            "revised Figure 5."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--input-profile",
        choices=("core4",),
        default="core4",
        help="Primary interpretation-first input profile.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output", type=Path, required=True, help="Tidy CSV output path."
    )
    parser.add_argument(
        "--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES
    )
    parser.add_argument(
        "--training-years",
        nargs="+",
        type=float,
        default=DEFAULT_TRAINING_YEARS,
        help="Training sizes for the left panel at a 10-month lead.",
    )
    parser.add_argument(
        "--lead-months",
        nargs="+",
        type=int,
        default=DEFAULT_LEAD_MONTHS,
        help="Forecast leads for the right panel.",
    )
    parser.add_argument(
        "--right-panel-training-years",
        nargs="+",
        type=float,
        default=(50.0, 10000.0),
    )
    parser.add_argument("--left-repetitions", type=int, default=3)
    parser.add_argument("--right-repetitions", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        help="Early-stopping override; defaults to 10 for every architecture.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the distinct run count without training or writing outputs.",
    )
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--statistics-batch-size", type=int, default=1024)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--overwrite-models", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the result CSV/JSON."
    )
    return parser


def _run_key(spec: ExperimentSpec) -> tuple[str, int, float, int, int | None]:
    return (
        spec.architecture,
        spec.lead_months,
        spec.train_years,
        spec.seed,
        spec.common_period_max_lead_months,
    )


def _reject_duplicates(values: list[object] | tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} cannot contain duplicate values.")


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.left_repetitions <= 0 or args.right_repetitions <= 0:
        raise ValueError("Repetition counts must be positive.")
    if any(
        years <= 0.0
        for years in (*args.training_years, *args.right_panel_training_years)
    ):
        raise ValueError("All training-set sizes must be positive.")
    if any(
        years > 10_000.0
        for years in (*args.training_years, *args.right_panel_training_years)
    ):
        raise ValueError("Fresh Figure 5 training sizes cannot exceed 10000 years.")
    if any(lead <= 0 for lead in args.lead_months):
        raise ValueError("All lead times must be positive.")
    if args.base_seed < 0:
        raise ValueError("--base-seed must be nonnegative.")
    _reject_duplicates(args.architectures, "--architectures")
    _reject_duplicates(args.training_years, "--training-years")
    _reject_duplicates(args.lead_months, "--lead-months")
    _reject_duplicates(
        args.right_panel_training_years,
        "--right-panel-training-years",
    )

    config = TrainingConfig(
        batch_size=args.batch_size,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        minimum_improvement=args.minimum_improvement,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        statistics_batch_size=args.statistics_batch_size,
        deterministic=not args.non_deterministic,
    )
    specifications: dict[tuple[str, int, float, int, int | None], ExperimentSpec] = {}
    rows_to_spec: list[tuple[str, int, ExperimentSpec]] = []
    common_maximum_lead = max(args.lead_months)
    for architecture in args.architectures:
        for years in args.training_years:
            for repetition in range(args.left_repetitions):
                spec = ExperimentSpec(
                    architecture=architecture,
                    lead_months=10,
                    train_years=float(years),
                    seed=args.base_seed + repetition,
                    input_profile=args.input_profile,
                )
                specifications[_run_key(spec)] = spec
                rows_to_spec.append(("training_size", repetition, spec))
        for years in args.right_panel_training_years:
            for lead in args.lead_months:
                for repetition in range(args.right_repetitions):
                    spec = ExperimentSpec(
                        architecture=architecture,
                        lead_months=int(lead),
                        train_years=float(years),
                        seed=args.base_seed + repetition,
                        common_period_max_lead_months=common_maximum_lead,
                        input_profile=args.input_profile,
                    )
                    specifications[_run_key(spec)] = spec
                    rows_to_spec.append(("lead_time", repetition, spec))

    LOGGER.info(
        "Figure 5 grid: %s distinct model fits and %s result rows.",
        len(specifications),
        len(rows_to_spec),
    )
    data = ZCData(
        args.data_dir,
        input_profile=args.input_profile,
        verify_checksums=True,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError(
            "Figure 5 now requires the fresh zc-v3 data set. The historical "
            "zc-v2 workflow is intentionally not mixed with this result grid."
        )
    if args.dry_run:
        LOGGER.info(
            "Dry run only. The right panel uses common periods valid through "
            "the maximum %s-month lead.",
            common_maximum_lead,
        )
        return 0

    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("--output must end in .csv")
    metadata_path = output.with_suffix(".json")
    if not args.overwrite:
        existing = [str(path) for path in (output, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist:\n  "
                + "\n  ".join(existing)
                + "\nUse --overwrite to replace them."
            )

    completed = {}
    for run_number, spec in enumerate(specifications.values(), start=1):
        LOGGER.info(
            "Figure 5 run %s/%s: %s",
            run_number,
            len(specifications),
            spec,
        )
        completed[_run_key(spec)] = train_figure5_experiment(
            data,
            args.artifacts_dir,
            spec,
            config,
            device=args.device,
            overwrite=args.overwrite_models,
        )

    result_rows: list[dict[str, str | int | float]] = []
    for panel, repetition, spec in rows_to_spec:
        experiment = completed[_run_key(spec)]
        result_rows.append(
            {
                "panel": panel,
                "architecture": spec.architecture,
                "lead_months": spec.lead_months,
                "train_years_requested": spec.train_years,
                "development_years_actual": experiment.metrics["selection"][
                    "actual_years"
                ],
                "fit_years_actual": experiment.metrics["selection"]["fit_years"],
                "embargo_years_actual": 0.0,
                "validation_years_actual": experiment.metrics["selection"][
                    "validation_years"
                ],
                "repetition": repetition,
                "seed": spec.seed,
                "test_r2": experiment.metrics["test_r2"],
                "test_target_start_step": experiment.metrics["evaluation"][
                    "test_target_start_step"
                ],
                "test_target_stop_step_exclusive": experiment.metrics["evaluation"][
                    "test_target_stop_step_exclusive"
                ],
                "checkpoint_sha256": experiment.checkpoint_sha256,
                "artifact_directory": str(
                    experiment.artifact_dir.relative_to(
                        args.artifacts_dir.expanduser().resolve()
                    )
                ),
            }
        )

    fieldnames = list(result_rows[0])
    with (
        atomic_output_path(output, overwrite=args.overwrite) as temporary_path,
        temporary_path.open("w", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
    write_json(
        metadata_path,
        {
            "schema_version": 2,
            "figure": 5,
            "data": data.provenance(),
            "training_config": asdict(config),
            "base_seed": args.base_seed,
            "left_panel": {
                "lead_months": 10,
                "training_years": list(args.training_years),
                "repetitions": args.left_repetitions,
                "error_bar": "sample standard deviation across repetitions",
                "validation": "fixed canonical 1000-year validation block",
            },
            "right_panel": {
                "lead_months": list(args.lead_months),
                "training_years": list(args.right_panel_training_years),
                "repetitions": args.right_repetitions,
                "common_period_max_lead_months": common_maximum_lead,
                "comparison_rule": (
                    "All leads use equal-length training, validation, and test "
                    "target windows and identical target dates valid at the "
                    "maximum lead."
                ),
            },
            "r2_definition": "1 - sum((y-yhat)^2) / sum((y-mean(y_test))^2)",
            "chronological_test_target_period_shared_within_each_panel": True,
            "training_size_axis_definition": (
                "Requested number of contiguous model years selected from the "
                "fixed 10000-year training block. The canonical 1000-year "
                "validation and 1000-year test blocks are never sampled for fit "
                "or normalization."
            ),
            "result_rows": len(result_rows),
        },
        overwrite=args.overwrite,
    )
    LOGGER.info("Figure 5 numerical results saved: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
