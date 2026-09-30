#!/usr/bin/env python3
"""Train and cache one reproducible PyTorch Zebiak-Cane forecaster."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import ZCData  # noqa: E402
from zc_xai.models import ARCHITECTURES  # noqa: E402
from zc_xai.training import (  # noqa: E402
    ExperimentSpec,
    TrainingConfig,
    train_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one cached Zebiak-Cane forecasting model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--input-profile",
        choices=("core4", "core4_wp", "legacy10", "all13"),
        default="core4",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--lead-months", type=int, required=True)
    parser.add_argument("--train-years", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        help="Early-stopping patience; defaults to 10 for every architecture.",
    )
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--statistics-batch-size", type=int, default=1024)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    data = ZCData(args.data_dir, input_profile=args.input_profile)
    artifact_profile = args.input_profile if data.n_phase_features else None
    spec = ExperimentSpec(
        architecture=args.architecture,
        lead_months=args.lead_months,
        train_years=args.train_years,
        seed=args.seed,
        input_profile=artifact_profile,
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
    experiment = train_experiment(
        data,
        args.artifacts_dir,
        spec,
        config,
        device=args.device,
        overwrite=args.overwrite,
    )
    logging.info("Experiment: %s", experiment.artifact_dir)
    logging.info("Held-out test R2: %.6f", experiment.metrics["test_r2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
