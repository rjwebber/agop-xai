#!/usr/bin/env python3
"""Run the pooled-covariance ZC controller toward a 10% event composite.

This wrapper changes only the unit target direction used by the established
four-method sufficiency experiment.  Cohort selection, extreme-event target
projection, nine controls over three months, pooled native covariance, solver
gates, and direct-first/adaptive optimizer settings remain fixed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_zc_direct_agop_covariance_action as direct  # noqa: E402

from zc_xai.direct_composite_target import (  # noqa: E402
    COMPOSITE_METHOD,
    build_direct_composite_target,
)

EVENT_SLUGS = {
    "extreme_el_nino": "extreme-el-nino",
    "extreme_la_nina": "extreme-la-nina",
}
FORBIDDEN_FORWARD_FLAGS = {
    "--case",
    "--continuation-fractions",
    "--direct-exact-target-first",
    "--maximum-adaptive-subdivisions",
    "--maximum-iterations",
    "--output-dir",
    "--projection-change-multiple",
    "--selection-seed",
    "--target-event",
    "--trajectory",
    "--xai-method",
}


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    result.add_argument(
        "--target-event",
        required=True,
        choices=tuple(EVENT_SLUGS),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/zc_pooled_composite_covariance_action/"
            "core4-cnn-lead10-seed42-direct-first-adaptive-v1"
        ),
    )
    result.add_argument(
        "--covariance-centered-cache-gib",
        type=positive_float,
        default=24.0,
        help=(
            "RAM budget for the exact centered pooled-covariance cache. The "
            "current nine-phase cache requires 19.17 GiB."
        ),
    )
    result.add_argument(
        "--member",
        action="append",
        type=int,
        dest="members",
        help="Run one fixed seed-42 cohort member; repeat as needed.",
    )
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--fixed-batch-rescue-policy",
        action="store_true",
        help=(
            "Run the bounded rescue after the batch supervisor has authenticated "
            "an eligible primary rejection: use a midpoint bridge and at most "
            "three additional adaptive bisections. This permits at most eight "
            "additional 55-iteration SQP solves without changing any publication "
            "tolerance."
        ),
    )
    result.add_argument(
        "--hard-batch-rescue-policy",
        action="store_true",
        help=(
            "Use the versioned final-rescue schedule authorized only by the "
            "frozen all-phase hard-rescue supervisor: at most twelve SQP stage "
            "attempts of at most 55 iterations each, "
            "without changing any publication tolerance."
        ),
    )
    result.add_argument(
        "--primary-exact-only",
        action="store_true",
        help=(
            "Run only the canonical 40-iteration exact-target solve. The "
            "fail-forward batch supervisor separately authenticates any "
            "rejection before launching an isolated rescue."
        ),
    )
    return result


def _reject_conflicting_forwarded_flags(values: Sequence[str]) -> None:
    for value in values:
        name = value.split("=", 1)[0]
        if name in FORBIDDEN_FORWARD_FLAGS:
            raise ValueError(f"{name} is fixed by the composite wrapper")


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded = parser().parse_known_args(argv)
    _reject_conflicting_forwarded_flags(forwarded)
    preset_count = sum(
        bool(value)
        for value in (
            args.primary_exact_only,
            args.fixed_batch_rescue_policy,
            args.hard_batch_rescue_policy,
        )
    )
    if preset_count > 1:
        raise ValueError(
            "the primary, fixed-rescue, and hard-rescue presets are mutually "
            "exclusive"
        )

    # Register the composite provider only inside this process.  The mature
    # direct driver remains byte-for-byte unchanged, preserving the identities
    # and resumability of all existing AGOP/GRAD/IG/GradientSHAP runs.
    original_methods = direct.XAI_METHODS
    original_builder = direct.build_direct_xai_target
    original_sources = direct.SCIENTIFIC_SOURCE_PATHS
    direct.XAI_METHODS = (*original_methods, COMPOSITE_METHOD)

    def build_target(
        method: str,
        *,
        data: object,
        experiment: object,
        event_standardized: object,
        agop_factor: object = None,
        device: str = "cpu",
    ) -> object:
        return build_direct_composite_target(
            method,
            data=data,  # type: ignore[arg-type]
            experiment=experiment,  # type: ignore[arg-type]
            event_standardized=event_standardized,  # type: ignore[arg-type]
            target_event_label=args.target_event,
            agop_factor=agop_factor,
            device=device,
        )

    direct.build_direct_xai_target = build_target
    for source in (
        Path("scripts/run_zc_pooled_composite_covariance_action.py"),
        Path("src/zc_xai/composites.py"),
        Path("src/zc_xai/direct_composite_target.py"),
    ):
        if source not in direct.SCIENTIFIC_SOURCE_PATHS:
            direct.SCIENTIFIC_SOURCE_PATHS += (source,)

    output_dir = args.output_root / EVENT_SLUGS[args.target_event]
    if args.primary_exact_only:
        continuation_fractions = "1"
        maximum_adaptive_subdivisions = None
        maximum_iterations = "40"
    elif args.hard_batch_rescue_policy:
        continuation_fractions = "0.5,1.0"
        maximum_adaptive_subdivisions = "10"
        maximum_iterations = "55"
    elif args.fixed_batch_rescue_policy:
        continuation_fractions = "0.5,1.0"
        maximum_adaptive_subdivisions = "3"
        maximum_iterations = "55"
    else:
        continuation_fractions = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
        maximum_adaptive_subdivisions = "32"
        maximum_iterations = "40"

    delegated = [
        "--xai-method",
        COMPOSITE_METHOD,
        "--target-event",
        args.target_event,
        "--trajectory",
        "uniform-cohort",
        "--selection-seed",
        "42",
        "--case",
        "pooled",
        "--output-dir",
        str(output_dir),
        "--maximum-iterations",
        maximum_iterations,
        "--continuation-fractions",
        continuation_fractions,
        "--covariance-centered-cache-gib",
        str(args.covariance_centered_cache_gib),
    ]
    if not args.primary_exact_only:
        delegated.extend(
            (
                "--scale-continuation-warm-start",
                "--adaptive-continuation",
                "--adaptive-minimum-fraction-step",
                "0.00625",
                "--maximum-adaptive-subdivisions",
                str(maximum_adaptive_subdivisions),
            )
        )
    if (
        not args.fixed_batch_rescue_policy
        and not args.hard_batch_rescue_policy
        and not args.primary_exact_only
    ):
        delegated.append("--zero-dual-restart-after-warm-rejection")
        delegated.append("--direct-exact-target-first")
    for member in args.members or ():
        delegated.extend(("--member", str(member)))
    if args.overwrite:
        delegated.append("--overwrite")
    delegated.extend(forwarded)
    try:
        return direct.main(delegated)
    finally:
        direct.XAI_METHODS = original_methods
        direct.build_direct_xai_target = original_builder
        direct.SCIENTIFIC_SOURCE_PATHS = original_sources


if __name__ == "__main__":
    raise SystemExit(main())
