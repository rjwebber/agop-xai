#!/usr/bin/env python3
"""Generate target-only multi-lead CNN XAI data for Figures 8 and 9.

The warm and cold rows are each target locked: their 10-, 5-, and 1-month
inputs lead to exactly the same held-out ZC target date.  This figure-only
bundle intentionally omits nearest-neighbor robustness calculations, which are
unnecessary for rendering the six target explanations and much more costly.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.fresh_case_studies import (  # noqa: E402
    METHOD_ARCHIVE_KEYS,
    METHOD_ORDER,
    FreshEvent,
    FreshXAISettings,
    build_explainers,
    experiment_metadata,
    fixed_test_extreme,
    target_locked_event,
    validate_primary_experiment,
)
from zc_xai.fresh_xai_outputs import (  # noqa: E402
    FRESH_XAI_BUNDLE_SCHEMA_VERSION,
    FRESH_XAI_TRAINING_POPULATION,
)
from zc_xai.io import sha256_array, sha256_file, write_json, write_npz  # noqa: E402
from zc_xai.training import (  # noqa: E402
    ExperimentSpec,
    load_experiment,
    predict,
    resolve_device,
)
from zc_xai.xai import build_exact_dense_agop_factor  # noqa: E402

LOGGER = logging.getLogger(__name__)
ARTIFACT = "fresh zc-v3 Figures 8-9 target-locked core4 CNN multi-lead XAI bundle"
LEGACY_ARTIFACT = (
    "fresh zc-v3 Figure 8 target-locked core4 CNN multi-lead XAI bundle"
)
LEADS = (10, 5, 1)
WARM_CASES = tuple(f"el_nino_{lead}m" for lead in LEADS)
COLD_CASES = tuple(f"la_nina_{lead}m" for lead in LEADS)
CASE_ORDER = (*WARM_CASES, *COLD_CASES)
OUTPUT_FILENAMES = {
    "explanations": "case_explanations.npz",
    "references": "xai_references.npz",
    "metadata": "metadata.json",
    "completion": "completed.json",
}


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate target-locked 10-, 5-, and 1-month CNN explanations for "
            "the fixed-test extreme El Nino and La Nina events."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/zc-v3")
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts/zc-v3")
    )
    parser.add_argument(
        "--agop-root", type=Path, default=Path("scratch/figures8_9_exact_agop")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/zc-v3/manuscript/figures8_9"),
    )
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--skip-data-checksums",
        action="store_true",
        help="Development only; production verifies the complete data release.",
    )
    parser.add_argument(
        "--overwrite-agop",
        action="store_true",
        help="Rebuild the small streamed exact-AGOP matrix caches.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _agop_directory(root: Path, lead_months: int, seed: int) -> Path:
    return root / (
        f"core4-cnn-lead-{lead_months:02d}m-seed-{seed:06d}-"
        "refs-all-streamed-batch-1024"
    )


def _build_or_load_exact_agop(
    root: Path,
    *,
    data: ZCData,
    experiment: Any,
    references: np.ndarray,
    lead_months: int,
    seed: int,
    device: str,
    overwrite: bool,
) -> tuple[Any, dict[str, Any]]:
    """Build exact AGOP without retaining the enormous gradient-row cache."""

    directory = _agop_directory(root, lead_months, seed)
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / "dense_agop_matrix.npy"
    started = time.perf_counter()
    factor = build_exact_dense_agop_factor(
        experiment.model,
        data,
        experiment.standardizer,
        references,
        gradient_batch_size=1024,
        device=device,
        matrix_cache_path=matrix_path,
        overwrite_matrix_cache=overwrite,
        cache_identity={
            "purpose": "fresh Figures 8-9 exact AGOP",
            "architecture": "cnn",
            "input_profile": "core4",
            "lead_months": lead_months,
            "seed": seed,
            "checkpoint_sha256": experiment.checkpoint_sha256,
        },
    )
    manifest_path = matrix_path.with_name(matrix_path.name + ".json")
    if not matrix_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Exact-AGOP matrix cache was not published.")
    return factor, {
        "method": "streamed exact dense empirical AGOP",
        "gradient_rows_retained": False,
        "gradient_batch_size": 1024,
        "matrix_file": matrix_path.name,
        "matrix_sha256": sha256_file(matrix_path),
        "matrix_manifest_file": manifest_path.name,
        "matrix_manifest_sha256": sha256_file(manifest_path),
        "directory_name": directory.name,
        "build_or_load_and_eigendecompose_seconds": time.perf_counter() - started,
        "solver_metadata": factor.solver_metadata,
    }


def _prefix(destination: dict[str, np.ndarray], prefix: str, values: dict) -> None:
    for key, value in values.items():
        destination[f"{prefix}__{key}"] = np.asarray(value)


def build_target_locked_events(data: ZCData) -> dict[str, FreshEvent]:
    """Return two physical events, each viewed at three forecast leads."""

    warm_anchor = fixed_test_extreme(data, lead_months=10, kind="maximum")
    cold_anchor = fixed_test_extreme(data, lead_months=10, kind="minimum")
    events: dict[str, FreshEvent] = {}
    for phase, anchor in (("el_nino", warm_anchor), ("la_nina", cold_anchor)):
        for lead in LEADS:
            case_id = f"{phase}_{lead}m"
            if lead == 10:
                events[case_id] = FreshEvent(
                    **{
                        **asdict(anchor),
                        "case_id": case_id,
                    }
                )
            else:
                events[case_id] = target_locked_event(
                    data,
                    case_id=case_id,
                    lead_months=lead,
                    target_step=anchor.target_step,
                    selection_rule=(
                        "target locked to the corresponding fixed-test 10-month "
                        f"{phase.replace('_', ' ')} extreme"
                    ),
                )
    return events


def _validated_target_explanations(
    data: ZCData,
    experiment: Any,
    explainers: dict[str, Any],
    event: FreshEvent,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validate_primary_experiment(data, experiment, lead_months=event.lead_months)
    if event.input_step not in experiment.test_inputs:
        raise ValueError(f"{event.case_id} input is outside the fixed test pairs.")
    if event.target_step != event.input_step + experiment.lead_steps:
        raise ValueError(f"{event.case_id} input and target are not lead consistent.")
    if not math.isclose(
        event.target_nino3_c,
        float(data.target[event.target_step]),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(f"{event.case_id} target value disagrees with the data.")

    query = data.load_inputs(
        np.asarray([event.input_step], dtype=np.int64),
        standardizer=experiment.standardizer,
    )
    arrays: dict[str, np.ndarray] = {
        "event_input_standardized": query[0].astype(np.float32),
        "event_input_physical": data.load_inputs(
            np.asarray([event.input_step], dtype=np.int64)
        )[0].astype(np.float32),
        "event_input_step": np.asarray(event.input_step, dtype=np.int64),
        "event_target_step": np.asarray(event.target_step, dtype=np.int64),
        "event_target_nino3_c": np.asarray(event.target_nino3_c, dtype=np.float32),
    }
    method_seconds: dict[str, float] = {}
    for method in METHOD_ORDER:
        started = time.perf_counter()
        explanation = np.asarray(explainers[method].explain(query)[0], dtype=np.float64)
        method_seconds[method] = time.perf_counter() - started
        if explanation.shape != (data.n_features,) or not np.isfinite(
            explanation
        ).all():
            raise ValueError(f"{event.case_id} {method} explanation is invalid.")
        norm = float(np.linalg.norm(explanation))
        if not math.isclose(norm, 1.0, rel_tol=2.0e-5, abs_tol=2.0e-6):
            raise ValueError(
                f"{event.case_id} {method} explanation is not unit norm: {norm}."
            )
        arrays[METHOD_ARCHIVE_KEYS[method]] = explanation.astype(np.float32)

    forecast = float(
        predict(
            experiment,
            data,
            np.asarray([event.input_step], dtype=np.int64),
            batch_size=1,
            device=device,
        )[0]
    )
    record = {
        **asdict(event),
        "selection_split": "fixed 1,000-year test block",
        "input_nino3_c": float(data.target[event.input_step]),
        "cnn_forecast_c": forecast,
        "checkpoint_sha256": experiment.checkpoint_sha256,
        "event_input_standardized_sha256": sha256_array(
            arrays["event_input_standardized"]
        ),
        "event_input_physical_sha256": sha256_array(arrays["event_input_physical"]),
        "target_explanation_seconds": method_seconds,
    }
    return arrays, record


def _xai_configuration(settings: FreshXAISettings) -> dict[str, Any]:
    return {
        "scope": "target explanations only; no robustness neighborhood evaluated",
        "methods": list(METHOD_ORDER),
        "integrated_gradients_steps": settings.integrated_gradients_steps,
        "expected_gradients_samples": settings.expected_gradients_samples,
        "expected_gradients_seed": settings.expected_gradients_seed,
        "gradient_batch_size": settings.gradient_batch_size,
        "expected_gradients_definition": (
            "distinct empirical fixed-training-predictor baselines without "
            "replacement, with one independent Uniform(0,1) interpolation "
            "coefficient per baseline"
        ),
        "expected_gradients_candidate_population": FRESH_XAI_TRAINING_POPULATION,
        "agop_definition": (
            "exact dense empirical gradient outer-product matrix over every fixed "
            "training predictor, with a complete float64 eigendecomposition"
        ),
        "coordinate_policy": (
            "all standardized model-input coordinates are explained; the two "
            "phase scalars are excluded only when maps are rendered"
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    data_dir = args.data_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    agop_root = args.agop_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = {key: output_dir / name for key, name in OUTPUT_FILENAMES.items()}
    if not args.overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist:\n  "
                + "\n  ".join(str(path) for path in existing)
                + "\nUse --overwrite to replace the complete bundle."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        paths["completion"].unlink(missing_ok=True)

    started = time.perf_counter()
    device = str(resolve_device(args.device))
    data = ZCData(
        data_dir,
        input_profile="core4",
        verify_checksums=not args.skip_data_checksums,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("This generator accepts only the fresh zc-v3 data set.")
    settings = FreshXAISettings()
    settings.validate()

    experiments: dict[int, Any] = {}
    explainers: dict[int, dict[str, Any]] = {}
    experiment_records: dict[str, Any] = {}
    references_archive: dict[str, np.ndarray] = {}
    first_mean: np.ndarray | None = None
    first_scale: np.ndarray | None = None
    for lead in sorted(LEADS):
        spec = ExperimentSpec(
            architecture="cnn",
            lead_months=lead,
            train_years=10_000.0,
            seed=args.seed,
            input_profile="core4",
        )
        experiment = load_experiment(data, artifacts_dir, spec, device="cpu")
        validate_primary_experiment(data, experiment, lead_months=lead)
        if first_mean is None:
            first_mean = experiment.standardizer.mean
            first_scale = experiment.standardizer.scale
        elif not np.array_equal(
            first_mean, experiment.standardizer.mean
        ) or not np.array_equal(first_scale, experiment.standardizer.scale):
            raise ValueError("Lead-specific CNNs use different input normalizers.")
        fixed = data.fixed_supervised_split(lead)
        factor, agop_record = _build_or_load_exact_agop(
            agop_root,
            data=data,
            experiment=experiment,
            references=fixed.train_inputs,
            lead_months=lead,
            seed=args.seed,
            device=device,
            overwrite=args.overwrite_agop,
        )
        method_explainers, references = build_explainers(
            data,
            experiment,
            factor,
            settings,
            device=device,
        )
        experiments[lead] = experiment
        explainers[lead] = method_explainers
        lead_key = f"lead_{lead:02d}"
        experiment_records[lead_key] = experiment_metadata(
            data,
            experiment,
            artifact_root=artifacts_dir,
            agop_provenance=agop_record,
        )
        _prefix(references_archive, lead_key, references)

    events = build_target_locked_events(data)
    explanations_archive: dict[str, np.ndarray] = {}
    case_records: dict[str, Any] = {}
    for case_id in CASE_ORDER:
        event = events[case_id]
        LOGGER.info("Evaluating target explanations for %s", case_id)
        arrays, record = _validated_target_explanations(
            data,
            experiments[event.lead_months],
            explainers[event.lead_months],
            event,
            device=device,
        )
        _prefix(explanations_archive, case_id, arrays)
        case_records[case_id] = record

    write_npz(paths["explanations"], overwrite=True, **explanations_archive)
    write_npz(paths["references"], overwrite=True, **references_archive)
    output_records = {
        key: {"file": paths[key].name, "sha256": sha256_file(paths[key])}
        for key in ("explanations", "references")
    }
    metadata = {
        "schema_version": FRESH_XAI_BUNDLE_SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "figure_numbers": [8, 9],
        "data": data.provenance(),
        "input_profile": "core4",
        "spatial_fields": list(data.spatial_field_names),
        "phase_features": data.n_phase_features,
        "case_order": list(CASE_ORDER),
        "figure_panels": {
            "8": list(WARM_CASES),
            "9": list(COLD_CASES),
        },
        "target_lock_policy": (
            "within each panel, the 10-, 5-, and 1-month predictors share one "
            "physical target step chosen as the 10-month fixed-test extreme"
        ),
        "cases": case_records,
        "experiments": experiment_records,
        "xai_configuration": _xai_configuration(settings),
        "runtime": {
            "clock": "time.perf_counter wall time",
            "total_seconds": time.perf_counter() - started,
            "resolved_device": device,
        },
        "source_sha256": {
            "generator": sha256_file(Path(__file__)),
            "fresh_case_studies.py": sha256_file(
                REPOSITORY_ROOT / "src" / "zc_xai" / "fresh_case_studies.py"
            ),
            "xai.py": sha256_file(REPOSITORY_ROOT / "src" / "zc_xai" / "xai.py"),
        },
        "output_files": output_records,
    }
    write_json(paths["metadata"], metadata, overwrite=True)
    write_json(
        paths["completion"],
        {
            "schema_version": FRESH_XAI_BUNDLE_SCHEMA_VERSION,
            "artifact": ARTIFACT,
            "files": {
                **{key: record["sha256"] for key, record in output_records.items()},
                "metadata": sha256_file(paths["metadata"]),
            },
            "generator_sha256": metadata["source_sha256"]["generator"],
        },
        overwrite=True,
    )
    for experiment in experiments.values():
        experiment.model.cpu()
    LOGGER.info("Fresh Figures 8-9 bundle saved: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
