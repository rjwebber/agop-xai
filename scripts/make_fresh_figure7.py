#!/usr/bin/env python3
"""Aggregate five trained CNN seeds and render revised manuscript Figure 7."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from generate_fresh_figure7_data import (  # noqa: E402
    ARTIFACT as FIGURE7_SEED_ARTIFACT,
)

from zc_xai.artifacts import CompletedBundle, load_completed_bundle  # noqa: E402
from zc_xai.fresh_case_studies import (  # noqa: E402
    METHOD_ORDER,
    FreshXAISettings,
    settings_metadata,
)
from zc_xai.fresh_xai_outputs import (  # noqa: E402
    FRESH_XAI_BUNDLE_SCHEMA_VERSION,
    FRESH_XAI_TRAINING_POPULATION,
)
from zc_xai.io import atomic_output_path, sha256_file, write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)
LEGACY_FIGURE9_SEED_ARTIFACT = (
    "fresh zc-v3 Figure 9 common-target core4 CNN XAI bundle"
)
CANONICAL_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_LEADS = tuple(range(1, 13))
REQUIRED_BUNDLE_FILES = ("scores", "explanations", "neighbors", "references")
METRICS = (
    ("sensitivity", "Sensitivity ($S_s$)"),
    ("attribution", "Attribution ($S_a$)"),
    ("robustness", "Robustness ($S_r$)"),
    ("coherence", "Coherence ($S_c$)"),
)
METHOD_STYLES = {
    "AGOP": {"color": "#648fff", "marker": "o", "linewidth": 2.2},
    "GradientSHAP": {"color": "#fe6100", "marker": "s", "linewidth": 1.7},
    "IG": {"color": "#ffb000", "marker": "^", "linewidth": 1.7},
    "GRAD": {"color": "#dc267f", "marker": "D", "linewidth": 1.7},
}
PALETTE_NAME = "IBM color-blind-safe"
SCORE_Y_LIMITS = (-0.1, 1.1)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the complete seeds 42--46 Figure 7 XAI bundles and "
            "plot equal-weight seed means."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("artifacts/zc-v3/manuscript/figure7"),
        help="Contains seed-000042 through seed-000046 completed bundles.",
    )
    parser.add_argument(
        "--legacy-seed42-bundle",
        type=Path,
        help=(
            "Explicit compatibility input for the old, otherwise equivalent "
            "single-seed Figure 9 bundle. Canonical runs should omit this."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manuscript/figures/figure7_zcv3_core4.pdf"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Tidy mean/SD/SEM CSV; defaults beside --output.",
    )
    parser.add_argument("--dpi", type=positive_int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _case_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    cases = metadata["cases"]
    return {
        case_id: {
            key: record[key]
            for key in (
                "case_id",
                "lead_months",
                "input_step",
                "target_step",
                "target_nino3_c",
                "selection_rule",
                "selection_split",
                "event_input_standardized_sha256",
            )
        }
        for case_id, record in sorted(cases.items())
    }


def _experiment_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    for lead_key, record in sorted(metadata["experiments"].items()):
        spec = dict(record["spec"])
        spec.pop("seed")
        contract[lead_key] = {
            "spec_without_seed": spec,
            "training_config": record["training_config"],
            "common_target_dates": record["common_target_dates"],
            "fit_indices_sha256": record["fit_indices_sha256"],
            "standardization_indices_sha256": record[
                "standardization_indices_sha256"
            ],
            "validation_indices_sha256": record["validation_indices_sha256"],
            "test_indices_sha256": record["test_indices_sha256"],
            "normalization_count": record["normalization_count"],
            "normalization_mean_sha256": record["normalization_mean_sha256"],
            "normalization_scale_sha256": record["normalization_scale_sha256"],
            "spatial_input_shape": record["spatial_input_shape"],
            "phase_features": record["phase_features"],
        }
    return contract


def _comparison_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract every seed-invariant scientific choice from one bundle."""

    return {
        "data": metadata["data"],
        "input_profile": metadata["input_profile"],
        "spatial_fields": metadata["spatial_fields"],
        "phase_features": metadata["phase_features"],
        "lead_months": metadata["lead_months"],
        "case_order": metadata["case_order"],
        "cases": _case_contract(metadata),
        "experiments": _experiment_contract(metadata),
        "anchor": metadata["anchor"],
        "comparison_period": metadata["comparison_period"],
        "xai_configuration": metadata["xai_configuration"],
    }


def _validate_seed_metadata(
    metadata: dict[str, Any],
    expected_seed: int,
    *,
    legacy_figure9: bool,
) -> tuple[dict[str, Any], dict[int, str]]:
    expected_figure = 9 if legacy_figure9 else 7
    _require(
        metadata.get("schema_version") == FRESH_XAI_BUNDLE_SCHEMA_VERSION,
        f"Seed {expected_seed} has an unexpected bundle schema.",
    )
    _require(
        metadata.get("figure") == expected_figure,
        f"Seed {expected_seed} bundle has the wrong source figure number.",
    )
    _require(
        metadata.get("input_profile") == "core4",
        f"Seed {expected_seed} does not use core4 inputs.",
    )
    _require(
        metadata.get("lead_months") == list(EXPECTED_LEADS),
        f"Seed {expected_seed} does not contain leads 1--12 exactly.",
    )
    expected_cases = [f"el_nino_lead_{lead:02d}" for lead in EXPECTED_LEADS]
    _require(
        metadata.get("case_order") == expected_cases,
        f"Seed {expected_seed} has an invalid case order.",
    )
    comparison = metadata.get("comparison_period")
    _require(
        isinstance(comparison, dict)
        and comparison.get("common_period_max_lead_months") == 12
        and all(
            comparison.get(key) is True
            for key in (
                "shared_fit_target_dates",
                "shared_validation_target_dates",
                "shared_test_target_dates",
                "shared_normalization",
            )
        ),
        f"Seed {expected_seed} lacks the common-target comparison contract.",
    )
    configuration = metadata.get("xai_configuration")
    _require(
        configuration == settings_metadata(FreshXAISettings())
        and configuration.get("neighbor_sampling")
        == "none; exhaustive finite population"
        and configuration.get("robustness_candidate_population")
        == FRESH_XAI_TRAINING_POPULATION
        and configuration.get("expected_gradients_candidate_population")
        == FRESH_XAI_TRAINING_POPULATION,
        f"Seed {expected_seed} does not use the canonical training-only XAI setup.",
    )
    cases = metadata.get("cases")
    experiments = metadata.get("experiments")
    _require(
        isinstance(cases, dict) and set(cases) == set(expected_cases),
        f"Seed {expected_seed} case records are incomplete.",
    )
    expected_experiments = {f"lead_{lead:02d}" for lead in EXPECTED_LEADS}
    _require(
        isinstance(experiments, dict) and set(experiments) == expected_experiments,
        f"Seed {expected_seed} experiment records are incomplete.",
    )
    checkpoint_hashes: dict[int, str] = {}
    target_steps: set[int] = set()
    for lead in EXPECTED_LEADS:
        lead_key = f"lead_{lead:02d}"
        case_id = f"el_nino_lead_{lead:02d}"
        experiment = experiments[lead_key]
        case = cases[case_id]
        _require(
            isinstance(experiment, dict) and isinstance(case, dict),
            f"Seed {expected_seed}, lead {lead} has malformed records.",
        )
        spec = experiment.get("spec")
        _require(
            isinstance(spec, dict)
            and spec.get("seed") == expected_seed
            and spec.get("architecture") == "cnn"
            and spec.get("input_profile") == "core4"
            and spec.get("lead_months") == lead
            and spec.get("train_years") == 10_000.0
            and spec.get("common_period_max_lead_months") == 12,
            f"Seed {expected_seed}, lead {lead} has the wrong model specification.",
        )
        checkpoint = experiment.get("checkpoint_sha256")
        _require(
            isinstance(checkpoint, str)
            and len(checkpoint) == 64
            and case.get("checkpoint_sha256") == checkpoint,
            f"Seed {expected_seed}, lead {lead} checkpoint provenance disagrees.",
        )
        _require(
            case.get("case_id") == case_id
            and case.get("lead_months") == lead
            and case.get("target_step") == case.get("input_step") + 3 * lead,
            f"Seed {expected_seed}, lead {lead} event timing is invalid.",
        )
        checkpoint_hashes[lead] = checkpoint
        target_steps.add(int(case["target_step"]))
    _require(
        len(target_steps) == 1,
        f"Seed {expected_seed} cases are not locked to one target date.",
    )
    _require(
        len(set(checkpoint_hashes.values())) == len(EXPECTED_LEADS),
        f"Seed {expected_seed} reuses a checkpoint across different leads.",
    )
    try:
        contract = _comparison_contract(metadata)
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Seed {expected_seed} scientific provenance is incomplete."
        ) from error
    return contract, checkpoint_hashes


def _load_seed_bundles(
    bundle_root: Path,
    legacy_seed42_bundle: Path | None,
) -> tuple[
    dict[int, CompletedBundle],
    dict[int, dict[int, str]],
    dict[str, Any],
    bool,
]:
    bundles: dict[int, CompletedBundle] = {}
    checkpoints: dict[int, dict[int, str]] = {}
    reference_contract: dict[str, Any] | None = None
    for seed in CANONICAL_SEEDS:
        legacy = seed == 42 and legacy_seed42_bundle is not None
        directory = legacy_seed42_bundle if legacy else bundle_root / f"seed-{seed:06d}"
        artifact = LEGACY_FIGURE9_SEED_ARTIFACT if legacy else FIGURE7_SEED_ARTIFACT
        bundle = load_completed_bundle(
            directory,
            expected_artifact=artifact,
            required_files=REQUIRED_BUNDLE_FILES,
        )
        contract, seed_checkpoints = _validate_seed_metadata(
            bundle.metadata,
            seed,
            legacy_figure9=legacy,
        )
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise ValueError(
                f"Seed {seed} scientific provenance differs from the other seeds."
            )
        bundles[seed] = bundle
        checkpoints[seed] = seed_checkpoints
    if reference_contract is None:  # pragma: no cover
        raise ValueError("No Figure 7 seed bundles were loaded.")
    for lead in EXPECTED_LEADS:
        lead_hashes = [checkpoints[seed][lead] for seed in CANONICAL_SEEDS]
        if len(set(lead_hashes)) != len(CANONICAL_SEEDS):
            raise ValueError(
                f"Figure 7 seed bundles do not contain five distinct CNN "
                f"checkpoints at lead {lead}."
            )
    return bundles, checkpoints, reference_contract, legacy_seed42_bundle is not None


def _parse_float(row: dict[str, str], key: str, context: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context} has an invalid {key} value.") from error
    if not math.isfinite(value):
        raise ValueError(f"{context} has a nonfinite {key} value.")
    return value


def _parse_int(row: dict[str, str], key: str, context: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context} has an invalid {key} value.") from error


def _read_seed_scores(
    bundle: CompletedBundle,
    seed: int,
    checkpoints: dict[int, str],
) -> dict[tuple[int, str], dict[str, float]]:
    values: dict[tuple[int, str], dict[str, float]] = {}
    cases = bundle.metadata["cases"]
    with bundle.files["scores"].open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            context = f"Seed {seed} score row"
            lead = _parse_int(raw, "lead_months", context)
            method = raw.get("method")
            key = (lead, str(method))
            if lead not in EXPECTED_LEADS or method not in METHOD_ORDER:
                raise ValueError(f"Seed {seed} has an unexpected score row: {key}")
            if key in values:
                raise ValueError(f"Seed {seed} has a duplicate score row: {key}")
            case_id = f"el_nino_lead_{lead:02d}"
            case = cases[case_id]
            if (
                raw.get("case_id") != case_id
                or raw.get("architecture") != "cnn"
                or raw.get("checkpoint_sha256") != checkpoints[lead]
                or _parse_int(raw, "event_input_step", context) != case["input_step"]
                or _parse_int(raw, "event_target_step", context) != case["target_step"]
                or not math.isclose(
                    _parse_float(raw, "event_target_nino3_c", context),
                    float(case["target_nino3_c"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
            ):
                raise ValueError(f"Seed {seed} score provenance disagrees: {key}")
            neighborhood_count = _parse_int(raw, "neighborhood_count", context)
            if (
                raw.get("robustness_exhaustive") != "True"
                or not math.isclose(
                    _parse_float(raw, "neighbor_percent", context),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or neighborhood_count <= 0
                or _parse_int(raw, "robustness_evaluated_count", context)
                != neighborhood_count
                or not math.isclose(
                    _parse_float(raw, "robustness_finite_population_se", context),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"Seed {seed} score is not exhaustive over the nearest 1%: {key}"
                )
            values[key] = {
                metric: _parse_float(raw, metric, context) for metric, _ in METRICS
            }
    expected = {(lead, method) for lead in EXPECTED_LEADS for method in METHOD_ORDER}
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(
            f"Seed {seed} score grid mismatch; missing={missing}, extra={extra}."
        )
    return values


def _aggregate_scores(
    seed_scores: dict[int, dict[tuple[int, str], dict[str, float]]],
) -> list[dict[str, Any]]:
    if tuple(sorted(seed_scores)) != CANONICAL_SEEDS:
        raise ValueError("Figure 7 requires exactly one score bundle for seeds 42--46.")
    expected = {(lead, method) for lead in EXPECTED_LEADS for method in METHOD_ORDER}
    for seed, scores in seed_scores.items():
        if set(scores) != expected:
            raise ValueError(f"Seed {seed} has an incomplete lead/method score grid.")
    rows: list[dict[str, Any]] = []
    seed_label = ";".join(str(seed) for seed in CANONICAL_SEEDS)
    for lead in EXPECTED_LEADS:
        for method in METHOD_ORDER:
            row: dict[str, Any] = {
                "lead_months": lead,
                "method": method,
                "seed_count": len(CANONICAL_SEEDS),
                "seeds": seed_label,
            }
            for metric, _ in METRICS:
                array = np.asarray(
                    [
                        seed_scores[seed][(lead, method)][metric]
                        for seed in CANONICAL_SEEDS
                    ],
                    dtype=np.float64,
                )
                sample_sd = float(array.std(ddof=1))
                row[f"{metric}_mean"] = float(array.mean())
                row[f"{metric}_sample_sd"] = sample_sd
                row[f"{metric}_sem"] = sample_sd / math.sqrt(array.size)
                row[f"{metric}_min"] = float(array.min())
                row[f"{metric}_max"] = float(array.max())
            rows.append(row)
    return rows


def _write_summary(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if not rows:
        raise ValueError("Cannot write an empty Figure 7 summary.")
    with (
        atomic_output_path(path, overwrite=overwrite) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_arrays(rows: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    indexed = {(int(row["lead_months"]), str(row["method"])): row for row in rows}
    return {
        method: {
            metric: np.asarray(
                [indexed[(lead, method)][f"{metric}_mean"] for lead in EXPECTED_LEADS],
                dtype=np.float64,
            )
            for metric, _ in METRICS
        }
        for method in METHOD_ORDER
    }


def _plot(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    dpi: int,
    overwrite: bool,
) -> None:
    means = _mean_arrays(rows)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            2, 2, figsize=(9.1, 6.7), sharex=True, constrained_layout=True
        )
        for axis, (metric, title) in zip(axes.flat, METRICS, strict=True):
            for method in METHOD_ORDER:
                axis.plot(
                    EXPECTED_LEADS,
                    means[method][metric],
                    label=method,
                    markersize=4.5,
                    markeredgewidth=0.0,
                    **METHOD_STYLES[method],
                )
            axis.set_title(title)
            axis.set_xticks(EXPECTED_LEADS)
            axis.grid(True, color="0.88", linewidth=0.7)
            axis.axhline(0.0, color="0.45", linewidth=0.7, zorder=0)
            axis.set_ylim(SCORE_Y_LIMITS)
        axes[1, 0].set_xlabel("Forecast lead (months)")
        axes[1, 1].set_xlabel("Forecast lead (months)")
        axes[0, 0].set_ylabel("Score")
        axes[1, 0].set_ylabel("Score")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="outside upper center", ncol=4, frameon=False
        )
        with atomic_output_path(output, overwrite=overwrite) as temporary:
            figure.savefig(temporary, dpi=dpi)
        plt.close(figure)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    bundle_root = args.bundle_root.expanduser().resolve()
    legacy = (
        args.legacy_seed42_bundle.expanduser().resolve()
        if args.legacy_seed42_bundle is not None
        else None
    )
    output = args.output.expanduser().resolve()
    if output.suffix.lower() not in {".pdf", ".png"}:
        raise ValueError("--output must end in .pdf or .png")
    summary = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else output.with_name(f"{output.stem}_scores.csv")
    )
    if summary.suffix.lower() != ".csv":
        raise ValueError("--summary-output must end in .csv")
    sidecar = output.with_suffix(".json")
    if not args.overwrite:
        existing = [path for path in (output, summary, sidecar) if path.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist:\n  "
                + "\n  ".join(str(path) for path in existing)
                + "\nUse --overwrite to replace them."
            )

    bundles, checkpoints, contract, legacy_used = _load_seed_bundles(
        bundle_root, legacy
    )
    seed_scores = {
        seed: _read_seed_scores(bundles[seed], seed, checkpoints[seed])
        for seed in CANONICAL_SEEDS
    }
    rows = _aggregate_scores(seed_scores)
    _write_summary(summary, rows, overwrite=args.overwrite)
    _plot(rows, output, dpi=args.dpi, overwrite=args.overwrite)

    sources = []
    for seed in CANONICAL_SEEDS:
        bundle = bundles[seed]
        sources.append(
            {
                "seed": seed,
                "bundle_directory": bundle.directory.name,
                "legacy_figure9_compatibility_input": seed == 42 and legacy_used,
                "metadata_sha256": sha256_file(bundle.metadata_path),
                "completion_sha256": sha256_file(bundle.completion_path),
                "files": dict(bundle.sha256),
                "checkpoint_sha256_by_lead": {
                    str(lead): checkpoints[seed][lead] for lead in EXPECTED_LEADS
                },
            }
        )
    write_json(
        sidecar,
        {
            "schema_version": 1,
            "figure": 7,
            "description": (
                "CNN XAI scores averaged equally across training seeds 42--46 "
                "at each forecast lead."
            ),
            "lead_months": list(EXPECTED_LEADS),
            "methods": list(METHOD_ORDER),
            "model_seeds": list(CANONICAL_SEEDS),
            "seed_count": len(CANONICAL_SEEDS),
            "aggregation": {
                "center": (
                    "unweighted arithmetic mean of the five seed-level scores "
                    "at each method and lead"
                ),
                "sample_standard_deviation": "computed with denominator n-1",
                "standard_error": "sample standard deviation divided by sqrt(5)",
                "uncertainty_displayed": False,
                "uncertainty_storage": summary.name,
                "robustness_note": (
                    "Across-seed SD and SEM are distinct from the within-neighborhood "
                    "robustness_population_sd stored in each source bundle."
                ),
            },
            "palette": {
                "name": PALETTE_NAME,
                "method_colors": {
                    method: style["color"] for method, style in METHOD_STYLES.items()
                },
            },
            "score_y_limits": list(SCORE_Y_LIMITS),
            "scientific_contract": contract,
            "source_seed_bundles": sources,
            "compatibility": {
                "legacy_seed42_bundle_used": legacy_used,
                "canonical_source_figure": 7,
            },
            "summary": {"file": summary.name, "sha256": sha256_file(summary)},
            "output": {
                "file": output.name,
                "sha256": sha256_file(output),
                "dpi": args.dpi,
            },
        },
        overwrite=args.overwrite,
    )
    LOGGER.info("Five-seed Figure 7 saved: %s", output)
    LOGGER.info("Five-seed statistics saved: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
