#!/usr/bin/env python3
"""Audit cross-lead AGOP squared mass by physical input field.

The audit loads the released common-target CNNs at leads 1--12 and applies
their exact full-rank AGOP factors to the target-locked extreme El Nino and La
Nina inputs.  Only the compact eigensystem cache is retained: no Figure 7/9
score bundle, robustness calculation, or dense AGOP matrix is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.fresh_case_studies import common_fixed_test_targets  # noqa: E402
from zc_xai.fresh_figure7 import (  # noqa: E402
    load_common_target_cnn,
    load_or_build_exact_factor,
)
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    sha256_array,
    sha256_file,
    write_json,
)
from zc_xai.training import resolve_device  # noqa: E402

SCHEMA_VERSION = 2
SPATIAL_FIELDS = (
    "sst_anomaly",
    "thermocline_depth",
    "zonal_ocean_current",
    "meridional_ocean_current",
)
PHASE_FIELDS = ("annual_phase_sin", "annual_phase_cos")
EXPECTED_LEADS = tuple(range(1, 13))
CSV_FIELDS = (
    "scope",
    "case",
    "lead_months",
    "architecture",
    "method",
    "input_step",
    "target_step",
    "target_nino3_c",
    "sst_anomaly_mass_percent",
    "thermocline_depth_mass_percent",
    "zonal_ocean_current_mass_percent",
    "meridional_ocean_current_mass_percent",
    "ocean_currents_mass_percent",
    "thermocline_and_currents_mass_percent",
    "spatial_mass_percent",
    "phase_mass_percent",
    "explanation_l2_norm",
    "source_artifact",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit AGOP squared-entry mass for zc-v3/core4 artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/zc-v3"))
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=Path("artifacts/zc-v3/manuscript/field_mass_crosslead"),
        help="Compact exact-AGOP factor cache, organized by forecast lead.",
    )
    parser.add_argument(
        "--build-missing-factors",
        action="store_true",
        help=(
            "Build absent exact factors from all fit inputs. This is the only "
            "expensive operation; dense AGOP matrices are not retained."
        ),
    )
    parser.add_argument(
        "--gradient-batch-size",
        type=positive_int,
        default=1024,
        help=(
            "Gradient batch size used to construct and validate exact-factor "
            "cache identities."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument(
        "--skip-data-checksums",
        action="store_true",
        help="Development only; production verifies the released data checksums.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/manuscript/source_data/diagnostics/agop_field_mass_audit.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/manuscript/source_data/diagnostics/agop_field_mass_audit.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def portable_path(path: Path) -> str:
    """Use repository-relative paths when possible for stable report bytes."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def field_mass(
    values: np.ndarray, spatial_shape: tuple[int, int, int]
) -> dict[str, Any]:
    """Return full-vector squared-entry mass, retaining phase in the denominator."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    spatial_count = int(np.prod(spatial_shape))
    _require(spatial_shape[0] == len(SPATIAL_FIELDS), "Unexpected field count.")
    _require(
        vector.size == spatial_count + len(PHASE_FIELDS),
        "Explanation does not contain four spatial fields followed by two phases.",
    )
    _require(np.isfinite(vector).all(), "Explanation contains nonfinite values.")
    squared = np.square(vector)
    total = float(squared.sum())
    _require(total > 0.0 and math.isfinite(total), "Explanation has zero norm.")
    blocks = squared[:spatial_count].reshape(spatial_shape).sum(axis=(1, 2))
    fractions = blocks / total
    phase_fraction = float(squared[spatial_count:].sum() / total)
    spatial_fraction = float(fractions.sum())
    _require(
        math.isclose(spatial_fraction + phase_fraction, 1.0, abs_tol=1.0e-12),
        "Field and phase masses do not sum to one.",
    )
    result = {
        name: float(value)
        for name, value in zip(SPATIAL_FIELDS, fractions, strict=True)
    }
    result.update(
        {
            "ocean_currents": result["zonal_ocean_current"]
            + result["meridional_ocean_current"],
            "thermocline_and_currents": result["thermocline_depth"]
            + result["zonal_ocean_current"]
            + result["meridional_ocean_current"],
            "spatial": spatial_fraction,
            "phase": phase_fraction,
            "l2_norm": math.sqrt(total),
        }
    )
    return result


def crossover_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize where thermocline mass first overtakes SST mass."""

    ordered = sorted(rows, key=lambda row: int(row["lead_months"]))
    leads = [int(row["lead_months"]) for row in ordered]
    _require(leads == list(EXPECTED_LEADS), "Crossover requires leads 1 through 12.")
    thermocline_larger = [
        lead
        for lead, row in zip(leads, ordered, strict=True)
        if row["mass_fraction"]["thermocline_depth"]
        > row["mass_fraction"]["sst_anomaly"]
    ]
    sst_larger = [lead for lead in leads if lead not in thermocline_larger]
    first = min(thermocline_larger) if thermocline_larger else None
    near_ties = []
    for row in ordered:
        difference = 100.0 * (
            row["mass_fraction"]["sst_anomaly"]
            - row["mass_fraction"]["thermocline_depth"]
        )
        if abs(difference) < 0.1:
            near_ties.append(
                {
                    "lead_months": int(row["lead_months"]),
                    "sst_minus_thermocline_percentage_points": difference,
                }
            )
    return {
        "first_lead_thermocline_exceeds_sst": first,
        "leads_sst_exceeds_thermocline": sst_larger,
        "leads_thermocline_exceeds_sst": thermocline_larger,
        "near_ties_within_0.1_percentage_point": near_ties,
    }


def _agop_explanation(
    standardized_input: np.ndarray,
    basis: np.ndarray,
    root_eigenvalues: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(standardized_input, dtype=np.float64).reshape(-1)
    transformed = ((vector @ basis) * root_eigenvalues) @ basis.T
    norm = float(np.linalg.norm(transformed))
    _require(norm > 0.0 and math.isfinite(norm), "AGOP explanation has zero norm.")
    return transformed / norm


def _mass_row(
    *,
    scope: str,
    case: str,
    lead_months: int,
    architecture: str,
    input_step: int,
    target_step: int,
    target_nino3_c: float,
    mass: dict[str, Any],
    source_artifact: str,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "case": case,
        "lead_months": lead_months,
        "architecture": architecture,
        "method": "AGOP",
        "input_step": input_step,
        "target_step": target_step,
        "target_nino3_c": target_nino3_c,
        "mass_fraction": {name: mass[name] for name in SPATIAL_FIELDS},
        "ocean_currents_mass_fraction": mass["ocean_currents"],
        "thermocline_and_currents_mass_fraction": mass["thermocline_and_currents"],
        "spatial_mass_fraction": mass["spatial"],
        "phase_mass_fraction": mass["phase"],
        "explanation_l2_norm": mass["l2_norm"],
        "source_artifact": source_artifact,
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    masses = row["mass_fraction"]
    return {
        "scope": row["scope"],
        "case": row["case"],
        "lead_months": row["lead_months"],
        "architecture": row["architecture"],
        "method": row["method"],
        "input_step": row["input_step"],
        "target_step": row["target_step"],
        "target_nino3_c": row["target_nino3_c"],
        "sst_anomaly_mass_percent": 100.0 * masses["sst_anomaly"],
        "thermocline_depth_mass_percent": 100.0 * masses["thermocline_depth"],
        "zonal_ocean_current_mass_percent": 100.0 * masses["zonal_ocean_current"],
        "meridional_ocean_current_mass_percent": 100.0
        * masses["meridional_ocean_current"],
        "ocean_currents_mass_percent": 100.0 * row["ocean_currents_mass_fraction"],
        "thermocline_and_currents_mass_percent": 100.0
        * row["thermocline_and_currents_mass_fraction"],
        "spatial_mass_percent": 100.0 * row["spatial_mass_fraction"],
        "phase_mass_percent": 100.0 * row["phase_mass_fraction"],
        "explanation_l2_norm": row["explanation_l2_norm"],
        "source_artifact": row["source_artifact"],
    }


def _validate_data(data: ZCData) -> None:
    _require(
        data.metadata.get("schema_version") == FRESH_SCHEMA_VERSION,
        "The audit requires fresh zc-v3 data.",
    )
    _require(data.input_profile == "core4", "The audit requires core4 inputs.")
    _require(
        tuple(data.spatial_field_names) == SPATIAL_FIELDS,
        "The core4 field order has changed.",
    )
    _require(data.spatial_input_shape == (4, 20, 27), "Unexpected spatial shape.")
    _require(data.input_shape == (2162,), "Unexpected core4 feature count.")
    _require(data.n_phase_features == 2, "Expected two annual-phase coordinates.")


def analyze(
    *,
    data_dir: Path,
    artifacts_dir: Path,
    factor_dir: Path,
    build_missing_factors: bool,
    gradient_batch_size: int,
    device: str,
    seed: int,
    verify_data_checksums: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = ZCData(
        data_dir,
        verify_checksums=verify_data_checksums,
        input_profile="core4",
    )
    _validate_data(data)
    resolved_device = str(resolve_device(device))
    common_targets = common_fixed_test_targets(data, list(EXPECTED_LEADS))
    el_target = int(common_targets[np.argmax(data.target[common_targets])])
    la_target = int(common_targets[np.argmin(data.target[common_targets])])
    targets = {
        "el_nino": (el_target, float(data.target[el_target])),
        "la_nina": (la_target, float(data.target[la_target])),
    }

    cross_rows: list[dict[str, Any]] = []
    factor_records: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    normalization_records: list[dict[str, Any]] = []
    reference_mean: np.ndarray | None = None
    reference_scale: np.ndarray | None = None
    reference_fit_targets: np.ndarray | None = None
    reference_validation_targets: np.ndarray | None = None
    reference_test_targets: np.ndarray | None = None
    for lead in EXPECTED_LEADS:
        experiment = load_common_target_cnn(
            data,
            artifacts_dir,
            lead_months=lead,
            seed=seed,
            maximum_lead_months=max(EXPECTED_LEADS),
            device=resolved_device,
        )
        fit_targets = experiment.fit_inputs + experiment.lead_steps
        validation_targets = experiment.validation_inputs + experiment.lead_steps
        test_targets = experiment.test_inputs + experiment.lead_steps
        if reference_mean is None:
            reference_mean = np.asarray(experiment.standardizer.mean, dtype=np.float64)
            reference_scale = np.asarray(
                experiment.standardizer.scale,
                dtype=np.float64,
            )
            reference_fit_targets = fit_targets
            reference_validation_targets = validation_targets
            reference_test_targets = test_targets
        else:
            _require(
                np.array_equal(experiment.standardizer.mean, reference_mean)
                and np.array_equal(experiment.standardizer.scale, reference_scale),
                "Cross-lead CNN checkpoints do not share exact normalization.",
            )
            _require(
                np.array_equal(fit_targets, reference_fit_targets)
                and np.array_equal(validation_targets, reference_validation_targets)
                and np.array_equal(test_targets, reference_test_targets),
                "Cross-lead CNN checkpoints do not share target-date splits.",
            )

        lead_factor_dir = factor_dir / f"lead-{lead:02d}m-seed-{seed:06d}"
        factor_path = lead_factor_dir / "full_eigensystem.npz"
        if not factor_path.is_file() and not build_missing_factors:
            raise FileNotFoundError(
                f"Missing exact AGOP factor: {factor_path}\n"
                "Rerun with --build-missing-factors to construct all absent "
                "lead factors from the released training inputs."
            )
        factor, factor_record = load_or_build_exact_factor(
            data,
            experiment,
            lead_factor_dir,
            gradient_batch_size=gradient_batch_size,
            device=resolved_device,
            retain_matrix_cache=False,
        )
        factor_record = {
            "lead_months": lead,
            "file": portable_path(factor_path),
            "file_sha256": factor_record["factor_sha256"],
            "manifest": portable_path(factor_path.with_suffix(".npz.json")),
            "manifest_sha256": sha256_file(
                factor_path.with_suffix(".npz.json")
            ),
            "identity_sha256": factor_record["identity_sha256"],
            "reference_count": factor_record["reference_count"],
            "rank": factor_record["rank"],
        }
        factor_records.append(factor_record)
        checkpoint_path = experiment.artifact_dir / "checkpoint.pt"
        normalization_path = experiment.artifact_dir / "normalization.npz"
        model_records.append(
            {
                "lead_months": lead,
                "artifact_directory": portable_path(experiment.artifact_dir),
                "checkpoint": portable_path(checkpoint_path),
                "checkpoint_sha256": experiment.checkpoint_sha256,
                "fit_indices_sha256": sha256_array(experiment.fit_inputs),
                "test_indices_sha256": sha256_array(experiment.test_inputs),
            }
        )
        normalization_records.append(
            {
                "lead_months": lead,
                "file": portable_path(normalization_path),
                "file_sha256": sha256_file(normalization_path),
                "mean_sha256": sha256_array(experiment.standardizer.mean),
                "scale_sha256": sha256_array(experiment.standardizer.scale),
            }
        )
        for case, (target_step, target_value) in targets.items():
            input_step = target_step - experiment.lead_steps
            _require(
                bool(np.any(experiment.test_inputs == input_step)),
                f"{case} input is outside the test split at lead {lead}.",
            )
            physical = data.load_inputs(np.asarray([input_step], dtype=np.int64))[0]
            standardized = (
                physical.astype(np.float64)
                - np.asarray(experiment.standardizer.mean, dtype=np.float64)
            ) / np.asarray(experiment.standardizer.scale, dtype=np.float64)
            explanation = _agop_explanation(
                standardized,
                factor.basis,
                factor.root_eigenvalues,
            )
            cross_rows.append(
                _mass_row(
                    scope="cross_lead",
                    case=case,
                    lead_months=lead,
                    architecture="cnn",
                    input_step=input_step,
                    target_step=target_step,
                    target_nino3_c=target_value,
                    mass=field_mass(explanation, data.spatial_input_shape),
                    source_artifact=str(factor_record["file"]),
                )
            )

    assert reference_fit_targets is not None
    assert reference_validation_targets is not None
    assert reference_test_targets is not None

    el_rows = [row for row in cross_rows if row["case"] == "el_nino"]
    la_rows = [row for row in cross_rows if row["case"] == "la_nina"]
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "zc-v3/core4 AGOP squared-entry field-mass audit",
        "methodology": {
            "field_order": list(SPATIAL_FIELDS),
            "field_shape": [20, 27],
            "phase_order": list(PHASE_FIELDS),
            "coordinate_system": "training-standardized model-input coordinates",
            "agop_explanation": "unit_l2(G^(1/2) @ standardized_input)",
            "agop_matrix": (
                "exact empirical gradient outer-product matrix over every fixed "
                "training predictor"
            ),
            "mass_denominator": "all 2162 coordinates, including both phase scalars",
            "map_policy": "Figures 8--9 omit the two phase scalars from maps",
        },
        "data": data.provenance(),
        "sources": {
            "script_sha256": sha256_file(Path(__file__)),
            "figure5_models": model_records,
            "exact_agop_factors": factor_records,
            "normalization": normalization_records,
        },
        "cross_lead": {
            "experiment": {
                "architecture": "cnn",
                "model_seed": seed,
                "training_years": 10_000,
                "common_period_max_lead_months": 12,
                "lead_months": list(EXPECTED_LEADS),
                "resolved_device": resolved_device,
                "gradient_batch_size": gradient_batch_size,
            },
            "common_test_target_count": int(common_targets.size),
            "common_test_targets_sha256": sha256_array(common_targets),
            "common_fit_targets_sha256": sha256_array(reference_fit_targets),
            "common_validation_targets_sha256": sha256_array(
                reference_validation_targets
            ),
            "common_split_test_targets_sha256": sha256_array(reference_test_targets),
            "events": {
                case: {
                    "target_step": target_step,
                    "target_nino3_c": target_value,
                    "selection": ("maximum" if case == "el_nino" else "minimum")
                    + " true Nino-3 over target dates shared by all 12 leads",
                }
                for case, (target_step, target_value) in targets.items()
            },
            "rows": cross_rows,
            "crossover": {
                "el_nino": crossover_summary(el_rows),
                "la_nina": crossover_summary(la_rows),
                "interpretation": (
                    "Thermocline mass first exceeds SST mass at 9 months for both "
                    "events. La Nina has an effectively tied, 0.015-percentage-point "
                    "SST reversal at 11 months, so its transition is not monotone."
                ),
            },
        },
    }
    return cross_rows, document


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    with (
        atomic_output_path(path, overwrite=overwrite) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)


def main() -> int:
    args = build_parser().parse_args()
    existing = [
        path
        for path in (args.output_csv, args.output_json)
        if path.expanduser().resolve().exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist:\n  "
            + "\n  ".join(str(path) for path in existing)
            + "\nUse --overwrite to replace both audit files."
        )
    rows, document = analyze(
        data_dir=args.data_dir.expanduser().resolve(),
        artifacts_dir=args.artifacts_dir.expanduser().resolve(),
        factor_dir=args.factor_dir.expanduser().resolve(),
        build_missing_factors=args.build_missing_factors,
        gradient_batch_size=args.gradient_batch_size,
        device=args.device,
        seed=args.seed,
        verify_data_checksums=not args.skip_data_checksums,
    )
    write_csv(args.output_csv, rows, overwrite=args.overwrite)
    document["output_csv"] = {
        "file": portable_path(args.output_csv),
        "sha256": sha256_file(args.output_csv.expanduser().resolve()),
        "row_count": len(rows),
    }
    write_json(args.output_json, document, overwrite=args.overwrite)
    print(f"Wrote {args.output_csv} ({len(rows)} rows)")
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
