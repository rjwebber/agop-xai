#!/usr/bin/env python3
"""Run one pooled-covariance ZC sufficiency experiment for a selected XAI method.

This is the canonical comparison wrapper.  It deliberately fixes the CNN,
lead, cohort seed, cohort size, native covariance arm, and event checkpoints
through the mature direct-action driver; only the XAI direction and warm/cold
event are selected here.
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

from zc_xai.direct_xai_target import XAI_METHOD_KEYS, XAI_METHODS  # noqa: E402

EVENT_SLUGS = {
    "extreme_el_nino": "extreme-el-nino",
    "extreme_la_nina": "extreme-la-nina",
}
COHERENT_ADAPTIVE_PRESET = {
    "maximum_iterations": 40,
    "continuation_fractions": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    "scale_continuation_warm_start": True,
    "adaptive_continuation": True,
    "zero_dual_restart_after_warm_rejection": True,
    "adaptive_minimum_fraction_step": 0.00625,
    "maximum_adaptive_subdivisions": 32,
}
COHERENT_DIRECT_FIRST_ADAPTIVE_PRESET = {
    **COHERENT_ADAPTIVE_PRESET,
    "direct_exact_target_first": True,
}
FIXED_BATCH_RESCUE_PRESET = {
    # This preset is used only after the fail-forward supervisor authenticates
    # an eligible 40-iteration primary rejection with the full intermediate
    # gate.  The rescue has at most eight solves (0.5 and 1.0 plus three
    # adaptive bisections and endpoint retries), each capped at 55 iterations;
    # primary plus rescue is therefore bounded by 480 optimizer iterations.
    # Publication tolerances are deliberately absent and remain unchanged.
    "maximum_iterations": 55,
    "continuation_fractions": "0.5,1.0",
    "scale_continuation_warm_start": True,
    "adaptive_continuation": True,
    "zero_dual_restart_after_warm_rejection": False,
    "adaptive_minimum_fraction_step": 0.00625,
    "maximum_adaptive_subdivisions": 3,
    "direct_exact_target_first": False,
}
HARD_BATCH_RESCUE_PRESET = {
    # Versioned final rescue for the frozen 16-case all-phase allowlist.  With
    # zero-dual retries disabled, the two requested stages plus at most ten
    # midpoint insertions give a hard pre-solve ceiling of twelve SQP solves.
    # At 55 iterations per solve this is at most 660 optimizer iterations.
    "maximum_iterations": 55,
    "continuation_fractions": "0.5,1.0",
    "scale_continuation_warm_start": True,
    "adaptive_continuation": True,
    "zero_dual_restart_after_warm_rejection": False,
    "adaptive_minimum_fraction_step": 0.00625,
    "maximum_adaptive_subdivisions": 10,
    "direct_exact_target_first": False,
}
COHERENT_ADAPTIVE_FORWARD_FLAGS = {
    "--maximum-iterations",
    "--continuation-fractions",
    "--scale-continuation-warm-start",
    "--adaptive-continuation",
    "--zero-dual-restart-after-warm-rejection",
    "--adaptive-minimum-fraction-step",
    "--maximum-adaptive-subdivisions",
    "--direct-exact-target-first",
}
FORBIDDEN_FORWARD_FLAGS = {
    "--case",
    "--output-dir",
    "--projection-change-multiple",
    "--selection-seed",
    "--target-event",
    "--trajectory",
    "--xai-method",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    result.add_argument("--xai-method", required=True, choices=XAI_METHODS)
    result.add_argument(
        "--target-event",
        required=True,
        choices=tuple(EVENT_SLUGS),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/zc_pooled_xai_method_comparison/core4-cnn-lead10-seed42"),
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
        "--coherent-adaptive",
        action="store_true",
        help=(
            "Use the documented robust rerun preset: 40 SQP iterations per "
            "stage; target fractions 0.1,...,1.0; scaled accepted-stage warm "
            "starts; adaptive bisection to a minimum full-target fraction "
            "step of 0.00625; one zero-dual retry after a rejected warm start; "
            "and at most 32 inserted stages. Durable resume "
            "remains enabled by default. The final exact-target gates are "
            "unchanged."
        ),
    )
    result.add_argument(
        "--coherent-direct-first-adaptive",
        action="store_true",
        help=(
            "Use the coherent adaptive settings, but first attempt the exact "
            "fraction-1 target from canonical zero dual under the unchanged "
            "strict publication gates. Only a finite strict-gate rejection "
            "activates the 0.1,...,1.0 adaptive fallback. Rejected zero-dual "
            "solves are not recomputed at an identical target."
        ),
    )
    result.add_argument(
        "--fixed-batch-rescue-policy",
        action="store_true",
        help=(
            "Run the bounded rescue after the batch supervisor has authenticated "
            "an eligible primary rejection: use a midpoint bridge and at most "
            "three further adaptive bisections. Rejected iterates are never "
            "reused, at most eight additional 55-iteration SQP solves are "
            "possible, and the original strict publication gates remain unchanged."
        ),
    )
    result.add_argument(
        "--hard-batch-rescue-policy",
        action="store_true",
        help=(
            "Use the versioned final-rescue schedule authorized only by the "
            "frozen all-phase hard-rescue supervisor: start at fractions 0.5 "
            "and 1.0, allow at most ten adaptive midpoint insertions, never "
            "reuse a rejected iterate, and cap each of the at most twelve SQP "
            "solves at 55 iterations. Publication gates remain unchanged."
        ),
    )
    result.add_argument(
        "--primary-exact-only",
        action="store_true",
        help=(
            "Run only the canonical 40-iteration exact-target solve. The "
            "fail-forward supervisor authenticates a rejection before any "
            "isolated fixed-budget rescue."
        ),
    )
    return result


def _reject_conflicting_forwarded_flags(values: Sequence[str]) -> None:
    for value in values:
        name = value.split("=", 1)[0]
        if name in FORBIDDEN_FORWARD_FLAGS:
            raise ValueError(f"{name} is fixed by the pooled comparison wrapper")


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded = parser().parse_known_args(argv)
    _reject_conflicting_forwarded_flags(forwarded)
    preset_count = sum(
        bool(value)
        for value in (
            args.coherent_adaptive,
            args.coherent_direct_first_adaptive,
            args.fixed_batch_rescue_policy,
            args.hard_batch_rescue_policy,
            args.primary_exact_only,
        )
    )
    if preset_count > 1:
        raise ValueError(
            "the coherent and fixed batch rescue presets are mutually exclusive"
        )
    coherent_preset_requested = bool(
        args.coherent_adaptive or args.coherent_direct_first_adaptive
    )
    any_preset_requested = bool(
        coherent_preset_requested
        or args.fixed_batch_rescue_policy
        or args.hard_batch_rescue_policy
    )
    conflict_guard_requested = bool(any_preset_requested or args.primary_exact_only)
    if conflict_guard_requested:
        conflicts = sorted(
            {value.split("=", 1)[0] for value in forwarded}
            & COHERENT_ADAPTIVE_FORWARD_FLAGS
        )
        if conflicts:
            raise ValueError(
                "the coherent adaptive preset cannot be combined with overrides for: "
                + ", ".join(conflicts)
            )
    output_dir = (
        args.output_root
        / XAI_METHOD_KEYS[args.xai_method]
        / EVENT_SLUGS[args.target_event]
    )
    wrapper_source = Path("scripts/run_zc_pooled_xai_covariance_action.py")
    if wrapper_source not in direct.SCIENTIFIC_SOURCE_PATHS:
        direct.SCIENTIFIC_SOURCE_PATHS += (wrapper_source,)
    delegated = [
        "--xai-method",
        args.xai_method,
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
    ]
    for member in args.members or ():
        delegated.extend(("--member", str(member)))
    if args.overwrite:
        delegated.append("--overwrite")
    selected_preset = (
        HARD_BATCH_RESCUE_PRESET
        if args.hard_batch_rescue_policy
        else FIXED_BATCH_RESCUE_PRESET
        if args.fixed_batch_rescue_policy
        else COHERENT_ADAPTIVE_PRESET
    )
    if args.primary_exact_only:
        delegated.extend(
            (
                "--maximum-iterations",
                "40",
                "--continuation-fractions",
                "1",
            )
        )
    if any_preset_requested:
        delegated.extend(
            (
                "--maximum-iterations",
                str(selected_preset["maximum_iterations"]),
                "--continuation-fractions",
                str(selected_preset["continuation_fractions"]),
                "--scale-continuation-warm-start",
                "--adaptive-continuation",
                "--adaptive-minimum-fraction-step",
                str(selected_preset["adaptive_minimum_fraction_step"]),
                "--maximum-adaptive-subdivisions",
                str(selected_preset["maximum_adaptive_subdivisions"]),
            )
        )
        if selected_preset["zero_dual_restart_after_warm_rejection"]:
            delegated.append("--zero-dual-restart-after-warm-rejection")
    if args.coherent_direct_first_adaptive:
        delegated.append("--direct-exact-target-first")
    delegated.extend(forwarded)
    return direct.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
