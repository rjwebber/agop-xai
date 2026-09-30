#!/usr/bin/env python3
"""Generate fresh-ZC Table I data and Figure 6/7 explanation artifacts.

The scientific configuration is intentionally narrow: the four-ocean-field
``core4`` model plus two annual-phase scalars, a ten-month forecast lead, exact
dense all-training-reference AGOP, 1,024-gradient IG and expected gradients,
and exhaustive evaluation of every state in the nearest one percent of the
training-predictor population. Neighbor explanation chunks are content-addressed and
resumable, so an interrupted long ViT run can continue without recomputing
completed chunks.
"""

from __future__ import annotations

import argparse
import csv
import logging
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import benchmark_fresh_xai_methods as benchmark  # noqa: E402

from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.fresh_xai_outputs import (  # noqa: E402
    FRESH_EXPECTED_GRADIENT_COUNT,
    FRESH_IG_GRADIENT_COUNT,
    FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
    FRESH_TABLE_ARCHITECTURES,
    FRESH_TABLE_METHODS,
    FRESH_TABLE_SCHEMA_VERSION,
    FRESH_XAI_GRADIENT_BATCH_SIZE,
    FRESH_XAI_TRAINING_POPULATION,
    exact_robustness_summary,
    phase_coefficient_summary,
    spatial_coherence,
    validate_unit_explanation,
)
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    load_json,
    sha256_array,
    sha256_file,
    sha256_json,
    write_json,
    write_npz,
)
from zc_xai.training import (  # noqa: E402
    ExperimentSpec,
    LoadedExperiment,
    load_experiment,
    resolve_device,
)
from zc_xai.xai import (  # noqa: E402
    AgopExplainer,
    GradientExplainer,
    GradientShapExplainer,
    IntegratedGradientsExplainer,
    attribution_score,
    input_gradients,
    sample_empirical_neighbors,
    sensitivity_score,
)

LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "1.1.0"
DEFAULT_OUTPUT_DIR = Path("outputs/fresh_table1")
DEFAULT_AGOP_ROOT = Path("outputs/fresh_agop_benchmark")


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate resumable Table I XAI data for the fresh core4 ZC models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/zc-v3"))
    parser.add_argument("--agop-benchmark-root", type=Path, default=DEFAULT_AGOP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=FRESH_TABLE_ARCHITECTURES,
        default=FRESH_TABLE_ARCHITECTURES,
        help=(
            "Architectures to compute in this invocation. Multiple invocations may "
            "fill the same resumable output directory."
        ),
    )
    parser.add_argument("--lead-months", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--distance-batch-size", type=positive_int, default=8192)
    parser.add_argument(
        "--neighbor-chunk-size",
        type=positive_int,
        default=128,
        help="Number of exhaustive neighbors in each resumable explanation chunk.",
    )
    parser.add_argument(
        "--skip-data-checksums",
        action="store_true",
        help="Development only; recorded prominently in the output metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute selected architectures' caches with the same identity.",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _software(device: str) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
    }


def _agop_directory(root: Path, architecture: str, lead: int, seed: int) -> Path:
    stem = f"core4-{architecture}-lead-{lead:02d}m-seed-{seed:06d}-refs-all"
    batch_named = root / f"{stem}-batch-{FRESH_XAI_GRADIENT_BATCH_SIZE}"
    # New caches name the gradient batch explicitly.  Retain compatibility with
    # already-completed MLP/ViT caches whose report (checked below) records the
    # same production batch even though their directory name predates the suffix.
    return batch_named if batch_named.is_dir() else root / stem


def _validate_agop_gradient_batch(directory: Path) -> None:
    report = load_json(directory / "report.json")
    if report.get("run_identity", {}).get("numeric_method", {}).get(
        "gradient_batch_size"
    ) != FRESH_XAI_GRADIENT_BATCH_SIZE:
        raise ValueError(
            "The exact AGOP cache was not accumulated with the production "
            f"gradient batch size {FRESH_XAI_GRADIENT_BATCH_SIZE}."
        )


def _validate_experiments(
    data: ZCData,
    artifacts_dir: Path,
    *,
    lead_months: int,
    seed: int,
) -> tuple[dict[str, LoadedExperiment], Any]:
    fixed = data.fixed_supervised_split(lead_months)
    experiments: dict[str, LoadedExperiment] = {}
    first: LoadedExperiment | None = None
    for architecture in FRESH_TABLE_ARCHITECTURES:
        experiment = load_experiment(
            data,
            artifacts_dir,
            ExperimentSpec(
                architecture=architecture,
                lead_months=lead_months,
                train_years=10_000.0,
                seed=seed,
                input_profile="core4",
            ),
            device="cpu",
        )
        if not np.array_equal(experiment.fit_inputs, fixed.train_inputs):
            raise ValueError(f"{architecture} does not use the fixed training block.")
        if not np.array_equal(experiment.validation_inputs, fixed.validation_inputs):
            raise ValueError(f"{architecture} does not use the fixed validation block.")
        if not np.array_equal(experiment.test_inputs, fixed.test_inputs):
            raise ValueError(f"{architecture} does not use the fixed test block.")
        if first is not None:
            if not np.array_equal(
                experiment.development_inputs, first.development_inputs
            ):
                raise ValueError("Architectures use different development inputs.")
            if not np.array_equal(
                experiment.standardizer.mean, first.standardizer.mean
            ) or not np.array_equal(
                experiment.standardizer.scale, first.standardizer.scale
            ):
                raise ValueError("Architectures use different input normalizers.")
        else:
            first = experiment
        experiments[architecture] = experiment
    assert first is not None
    return experiments, fixed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty score table.")
    with (
        atomic_output_path(path, overwrite=True) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_stage_explanations(path: Path, feature_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        explanations = np.asarray(archive["explanations"], dtype=np.float32)
    if explanations.ndim != 2 or explanations.shape[1] != feature_count:
        raise ValueError(f"Invalid explanation cache schema: {path}")
    if not np.isfinite(explanations).all():
        raise ValueError(f"Non-finite explanation cache: {path}")
    return explanations


def _method_explainer(
    method: str,
    *,
    experiment: LoadedExperiment,
    backgrounds: np.ndarray,
    alphas: np.ndarray,
    agop_factor: Any,
    device: str,
) -> Any:
    if method == "AGOP":
        return AgopExplainer(agop_factor)
    if method == "GradientSHAP":
        return GradientShapExplainer(
            experiment.model,
            backgrounds=backgrounds,
            alphas=alphas,
            gradient_batch_size=FRESH_XAI_GRADIENT_BATCH_SIZE,
            device=device,
        )
    if method == "IG":
        return IntegratedGradientsExplainer(
            experiment.model,
            n_steps=FRESH_IG_GRADIENT_COUNT,
            gradient_batch_size=FRESH_XAI_GRADIENT_BATCH_SIZE,
            device=device,
        )
    if method == "GRAD":
        return GradientExplainer(
            experiment.model,
            batch_size=FRESH_XAI_GRADIENT_BATCH_SIZE,
            device=device,
        )
    raise ValueError(f"Unknown Table I method: {method!r}")


def _explanation_payload(explainer: Any, inputs: np.ndarray) -> dict[str, np.ndarray]:
    explanations = np.asarray(explainer.explain(inputs), dtype=np.float32)
    if explanations.shape != (inputs.shape[0], inputs.shape[1]):
        raise ValueError("An XAI explainer returned an unexpected shape.")
    for row in explanations:
        validate_unit_explanation(row, inputs.shape[1])
    return {"explanations": explanations}


def _base_report(
    run_identity: dict[str, Any],
    existing: dict[str, Any] | None,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    identity_sha256 = sha256_json(run_identity)
    if existing is not None:
        mismatch = (
            existing.get("schema_version") != FRESH_TABLE_SCHEMA_VERSION
            or existing.get("run_identity") != run_identity
            or existing.get("run_identity_sha256") != identity_sha256
        )
        if mismatch and not overwrite:
            raise ValueError(
                "Existing fresh Table I output has another identity; choose a new "
                "--output-dir or pass --overwrite to replace the bundle."
            )
        if not mismatch:
            return existing
    return {
        "schema_version": FRESH_TABLE_SCHEMA_VERSION,
        "status": "in_progress",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "run_identity": run_identity,
        "run_identity_sha256": identity_sha256,
        "architectures": {},
        "output_files": {},
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    report["updated_at_utc"] = _utc_now()
    write_json(path, report, overwrite=True)


def _finalize_outputs(
    output_dir: Path,
    report: dict[str, Any],
    *,
    data: ZCData,
    query_index: int,
    target_index: int,
    target_nino3: float,
    standardizer: Any,
) -> None:
    architecture_records = report["architectures"]
    if set(architecture_records) != set(FRESH_TABLE_ARCHITECTURES):
        report["status"] = "in_progress"
        return

    rows: list[dict[str, Any]] = []
    target_arrays: dict[str, np.ndarray] = {
        "event_input_physical": data.load_inputs(
            np.asarray([query_index], dtype=np.int64)
        )[0],
        "event_input_standardized": data.load_inputs(
            np.asarray([query_index], dtype=np.int64), standardizer=standardizer
        )[0],
        "event_input_step": np.asarray(query_index, dtype=np.int64),
        "event_target_step": np.asarray(target_index, dtype=np.int64),
        "event_target_nino3_c": np.asarray(target_nino3, dtype=np.float32),
    }
    overlap_arrays: dict[str, np.ndarray] = {}
    for architecture in FRESH_TABLE_ARCHITECTURES:
        record = architecture_records[architecture]
        summary_path = output_dir / record["summary_file"]
        if sha256_file(summary_path) != record["summary_sha256"]:
            raise ValueError(f"Architecture summary is corrupted: {summary_path}")
        with np.load(summary_path, allow_pickle=False) as archive:
            for method in FRESH_TABLE_METHODS:
                key = method.lower()
                target_arrays[f"{architecture}_{key}"] = np.asarray(
                    archive[f"target_{key}"], dtype=np.float32
                )
                overlap_arrays[f"{architecture}_{key}"] = np.asarray(
                    archive[f"overlaps_{key}"], dtype=np.float32
                )
        rows.extend(record["scores"])

    scores_path = output_dir / "table1_scores.csv"
    targets_path = output_dir / "target_explanations.npz"
    overlaps_path = output_dir / "robustness_overlaps.npz"
    _write_csv(scores_path, rows)
    write_npz(targets_path, overwrite=True, compressed=False, **target_arrays)
    write_npz(overlaps_path, overwrite=True, compressed=False, **overlap_arrays)
    report["status"] = "complete"
    report["completed_at_utc"] = _utc_now()
    report["output_files"] = {
        "scores": {"file": scores_path.name, "sha256": sha256_file(scores_path)},
        "target_explanations": {
            "file": targets_path.name,
            "sha256": sha256_file(targets_path),
        },
        "robustness_overlaps": {
            "file": overlaps_path.name,
            "sha256": sha256_file(overlaps_path),
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if len(args.architectures) != len(set(args.architectures)):
        raise ValueError("--architectures cannot contain duplicates")
    if args.lead_months != 10:
        raise ValueError("The current manuscript Table I is fixed at a 10-month lead.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    agop_root = args.agop_benchmark_root.expanduser().resolve()
    device = str(resolve_device(args.device))
    data = ZCData(
        args.data_dir,
        input_profile="core4",
        verify_checksums=not args.skip_data_checksums,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("This generator requires the fresh zc-v3 data set.")
    experiments, fixed = _validate_experiments(
        data,
        artifacts_dir,
        lead_months=args.lead_months,
        seed=args.seed,
    )
    reference = experiments[FRESH_TABLE_ARCHITECTURES[0]]
    target_values = data.load_targets(
        reference.test_inputs, lead_steps=reference.lead_steps
    )
    event_position = int(np.argmax(target_values))
    query_index = int(reference.test_inputs[event_position])
    target_index = query_index + reference.lead_steps
    target_nino3 = float(target_values[event_position])
    candidates = np.asarray(reference.fit_inputs, dtype=np.int64)

    run_identity = {
        "purpose": "fresh core4 manuscript Table I and Figures 6/7 XAI data",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__)),
        "xai_source_sha256": sha256_file(REPOSITORY_ROOT / "src/zc_xai/xai.py"),
        "helper_source_sha256": sha256_file(
            REPOSITORY_ROOT / "src/zc_xai/fresh_xai_outputs.py"
        ),
        "data": data.provenance(),
        "processed_checksums_verified": not args.skip_data_checksums,
        "input_profile": "core4",
        "spatial_fields": list(data.spatial_field_names),
        "phase_features": ["annual_phase_sin", "annual_phase_cos"],
        "lead_months": args.lead_months,
        "model_seed": args.seed,
        "expected_architectures": list(FRESH_TABLE_ARCHITECTURES),
        "event": {
            "selection": "maximum true Nino-3 target among fixed test predictors",
            "input_index": query_index,
            "target_index": target_index,
            "target_nino3_c": target_nino3,
        },
        "robustness": {
            "distance": "RMS Euclidean distance over all 2162 standardized features",
            "candidate_population": FRESH_XAI_TRAINING_POPULATION,
            "candidate_count": int(candidates.size),
            "candidate_indices_sha256": sha256_array(candidates),
            "nearest_percent": FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
            "sampling": "none: every member of the nearest-percent population",
        },
        "methods": {
            "integrated_gradients": {
                "gradient_count_per_explanation": FRESH_IG_GRADIENT_COUNT,
                "baseline": "zero vector in training-standardized coordinates",
                "quadrature": "right-endpoint uniform path grid",
            },
            "gradient_shap": {
                "scientific_name": "expected gradients / GradientSHAP approximation",
                "distinct_empirical_backgrounds": FRESH_EXPECTED_GRADIENT_COUNT,
                "reference_population": FRESH_XAI_TRAINING_POPULATION,
                "reference_population_count": int(candidates.size),
                "reference_population_indices_sha256": sha256_array(candidates),
                "background_seed": args.seed,
                "independent_uniform_alpha_seed": args.seed + 1,
            },
            "agop": {
                "references": "every fixed training predictor",
                "solver": (
                    "exact float64 dense accumulation and full eigendecomposition"
                ),
            },
            "gradient_batch_size": FRESH_XAI_GRADIENT_BATCH_SIZE,
        },
        "normalizer": {
            "population": "all raw states in the fixed 10000-year training block",
            "mean_sha256": sha256_array(reference.standardizer.mean),
            "scale_sha256": sha256_array(reference.standardizer.scale),
            "count": reference.standardizer.count,
        },
        "software": _software(device),
    }
    report_path = output_dir / "table1_metadata.json"
    existing = load_json(report_path) if report_path.exists() else None
    report = _base_report(run_identity, existing, overwrite=args.overwrite)

    neighbor_path = output_dir / "exact_nearest_1pct.npz"
    neighbor_identity = {
        "stage": "exact_nearest_percent_population",
        "data_metadata_sha256": data.metadata_sha256,
        "query_index": query_index,
        "candidate_indices_sha256": sha256_array(candidates),
        "normalizer_mean_sha256": sha256_array(reference.standardizer.mean),
        "normalizer_scale_sha256": sha256_array(reference.standardizer.scale),
        "nearest_percent": FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
        "distance_batch_size": args.distance_batch_size,
    }

    def compute_neighbors() -> dict[str, np.ndarray]:
        selection = sample_empirical_neighbors(
            data,
            reference.standardizer,
            query_index=query_index,
            candidate_indices=candidates,
            neighbor_percent=FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
            n_samples=1,
            seed=args.seed,
            distance_batch_size=args.distance_batch_size,
        )
        return {
            "query_index": np.asarray(selection.query_index, dtype=np.int64),
            "candidate_count": np.asarray(selection.candidate_count, dtype=np.int64),
            "neighbor_percent": np.asarray(selection.neighbor_percent),
            "neighborhood_indices": np.asarray(
                selection.neighborhood_indices, dtype=np.int64
            ),
            "neighborhood_rms_distances": np.asarray(
                selection.neighborhood_rms_distances, dtype=np.float64
            ),
        }

    neighbor_arrays, neighbor_record = benchmark.load_or_compute_npz(
        neighbor_path,
        stage="exact_nearest_percent_population",
        identity=neighbor_identity,
        compute=compute_neighbors,
        overwrite=args.overwrite,
    )
    neighbor_indices = np.asarray(
        neighbor_arrays["neighborhood_indices"], dtype=np.int64
    )
    neighbor_distances = np.asarray(
        neighbor_arrays["neighborhood_rms_distances"], dtype=np.float64
    )
    eligible_candidate_count = candidates.size - int(query_index in candidates)
    expected_neighbor_count = max(
        1,
        int(
            np.ceil(
                FRESH_ROBUSTNESS_NEIGHBOR_PERCENT
                * eligible_candidate_count
                / 100.0
            )
        ),
    )
    if neighbor_indices.size != expected_neighbor_count:
        raise ValueError(
            "The canonical lead-10 training population should yield exactly "
            f"{expected_neighbor_count:,} nearest-one-percent states, not "
            f"{neighbor_indices.size:,}."
        )

    reference_path = output_dir / "expected_gradient_references.npz"
    reference_identity = {
        "stage": "fixed_expected_gradient_references",
        "candidate_indices_sha256": sha256_array(candidates),
        "count": FRESH_EXPECTED_GRADIENT_COUNT,
        "background_seed": args.seed,
        "alpha_seed": args.seed + 1,
    }
    reference_arrays, reference_record = benchmark.load_or_compute_npz(
        reference_path,
        stage="fixed_expected_gradient_references",
        identity=reference_identity,
        compute=lambda: benchmark._fixed_expected_gradient_references(
            candidates,
            count=FRESH_EXPECTED_GRADIENT_COUNT,
            seed=args.seed,
        ),
        overwrite=args.overwrite,
    )
    background_indices = np.asarray(
        reference_arrays["background_indices"], dtype=np.int64
    )
    alphas = np.asarray(reference_arrays["alphas"], dtype=np.float32)
    query = data.load_inputs(
        np.asarray([query_index], dtype=np.int64),
        standardizer=reference.standardizer,
    )
    neighbors = data.load_inputs(
        neighbor_indices, standardizer=reference.standardizer
    )
    backgrounds = data.load_inputs(
        background_indices, standardizer=reference.standardizer
    )
    report["shared_setup"] = {
        "neighbors": {
            **neighbor_record,
            "population_count": int(neighbor_indices.size),
            "indices_sha256": sha256_array(neighbor_indices),
            "distances_sha256": sha256_array(neighbor_distances),
            "minimum_rms_distance": float(neighbor_distances.min()),
            "maximum_rms_distance": float(neighbor_distances.max()),
        },
        "expected_gradient_references": {
            **reference_record,
            "background_indices_sha256": sha256_array(background_indices),
            "alphas_sha256": sha256_array(alphas),
        },
    }
    _write_report(report_path, report)

    for architecture in args.architectures:
        if architecture in report["architectures"] and not args.overwrite:
            LOGGER.info("Skipping completed %s result", architecture.upper())
            continue
        experiment = experiments[architecture]
        agop_dir = _agop_directory(
            agop_root, architecture, args.lead_months, args.seed
        )
        _validate_agop_gradient_batch(agop_dir)
        agop_factor, agop_provenance = benchmark.load_validated_full_agop_factor(
            agop_dir,
            data=data,
            experiment=experiment,
            fixed_training_inputs=fixed.train_inputs,
        )
        input_gradients(
            experiment.model,
            query,
            batch_size=1,
            device=device,
        )
        gradient_started = time.perf_counter()
        query_gradient = input_gradients(
            experiment.model,
            query,
            batch_size=FRESH_XAI_GRADIENT_BATCH_SIZE,
            device=device,
        )[0].reshape(-1)
        query_gradient_seconds = time.perf_counter() - gradient_started

        architecture_dir = output_dir / architecture
        target_explanations: dict[str, np.ndarray] = {}
        overlap_arrays: dict[str, np.ndarray] = {}
        score_rows: list[dict[str, Any]] = []
        method_runtime: dict[str, Any] = {}
        for method in FRESH_TABLE_METHODS:
            LOGGER.info("%s / %s target", architecture.upper(), method)
            explainer = _method_explainer(
                method,
                experiment=experiment,
                backgrounds=backgrounds,
                alphas=alphas,
                agop_factor=agop_factor,
                device=device,
            )
            method_key = method.lower()
            base_stage_identity = {
                "run_identity_sha256": report["run_identity_sha256"],
                "architecture": architecture,
                "checkpoint_sha256": experiment.checkpoint_sha256,
                "method": method,
                "gradient_batch_size": FRESH_XAI_GRADIENT_BATCH_SIZE,
                "ig_steps": FRESH_IG_GRADIENT_COUNT,
                "expected_gradient_samples": FRESH_EXPECTED_GRADIENT_COUNT,
                "background_indices_sha256": sha256_array(background_indices),
                "alphas_sha256": sha256_array(alphas),
                "agop_eigensystem_sha256": agop_provenance["eigensystem_sha256"],
                "resolved_device": device,
            }
            target_path = architecture_dir / f"{method_key}_target.npz"
            target_values, target_record = benchmark.load_or_compute_npz(
                target_path,
                stage=f"{architecture}_{method}_target_explanation",
                identity={
                    **base_stage_identity,
                    "population": "target",
                    "indices_sha256": sha256_array(
                        np.asarray([query_index], dtype=np.int64)
                    ),
                },
                compute=lambda explainer=explainer: _explanation_payload(
                    explainer, query
                ),
                overwrite=args.overwrite,
            )
            target_explanation = validate_unit_explanation(
                target_values["explanations"][0], data.n_features
            )
            target_explanations[method_key] = target_explanation.astype(np.float32)

            overlaps: list[np.ndarray] = []
            chunk_records: list[dict[str, Any]] = []
            for start in range(0, neighbor_indices.size, args.neighbor_chunk_size):
                stop = min(start + args.neighbor_chunk_size, neighbor_indices.size)
                chunk_path = architecture_dir / (
                    f"{method_key}_neighbors_{start:04d}_{stop:04d}.npz"
                )
                chunk_indices = neighbor_indices[start:stop]
                chunk_values, chunk_record = benchmark.load_or_compute_npz(
                    chunk_path,
                    stage=f"{architecture}_{method}_neighbor_explanations",
                    identity={
                        **base_stage_identity,
                        "population": "exact_nearest_one_percent",
                        "chunk_start": start,
                        "chunk_stop": stop,
                        "indices_sha256": sha256_array(chunk_indices),
                    },
                    compute=lambda explainer=explainer, start=start, stop=stop: (
                        _explanation_payload(explainer, neighbors[start:stop])
                    ),
                    overwrite=args.overwrite,
                )
                explanations = np.asarray(
                    chunk_values["explanations"], dtype=np.float64
                )
                overlaps.append(
                    np.clip(explanations @ target_explanation, -1.0, 1.0)
                )
                chunk_records.append(
                    {
                        "start": start,
                        "stop": stop,
                        "file": chunk_record["file"],
                        "file_sha256": chunk_record["file_sha256"],
                        "compute_seconds": chunk_record["compute_seconds"],
                        "cache_hit": chunk_record["cache_hit"],
                    }
                )
            method_overlaps = np.concatenate(overlaps).astype(np.float32)
            robustness = exact_robustness_summary(method_overlaps)
            overlap_arrays[method_key] = method_overlaps
            attribution, attribution_ratio, attribution_alpha = attribution_score(
                experiment.model,
                query[0],
                target_explanation,
                device=device,
            )
            phase = phase_coefficient_summary(target_explanation, data)
            row = {
                "method": method,
                "architecture": architecture,
                "sensitivity": sensitivity_score(
                    query_gradient, target_explanation
                ),
                "attribution": attribution,
                "attribution_ratio": attribution_ratio,
                "attribution_alpha": attribution_alpha,
                **robustness,
                "coherence_spatial_only": spatial_coherence(
                    target_explanation, data
                ),
                **phase,
                "neighbor_percent": FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
                "neighborhood_count": int(neighbor_indices.size),
                "event_input_step": query_index,
                "event_target_step": target_index,
                "event_target_nino3_c": target_nino3,
                "checkpoint_sha256": experiment.checkpoint_sha256,
            }
            score_rows.append(row)
            method_runtime[method] = {
                "target": target_record,
                "neighbor_chunks": chunk_records,
                "neighbor_compute_seconds": float(
                    sum(record["compute_seconds"] for record in chunk_records)
                ),
                "gradient_rows": (
                    0
                    if method == "AGOP"
                    else (
                        1 + neighbor_indices.size
                        if method == "GRAD"
                        else (1 + neighbor_indices.size) * 1024
                    )
                ),
            }

        summary_path = architecture_dir / "summary.npz"
        summary_arrays: dict[str, np.ndarray] = {}
        for method in FRESH_TABLE_METHODS:
            key = method.lower()
            summary_arrays[f"target_{key}"] = target_explanations[key]
            summary_arrays[f"overlaps_{key}"] = overlap_arrays[key]
        write_npz(
            summary_path,
            overwrite=True,
            compressed=False,
            **summary_arrays,
        )
        report["architectures"][architecture] = {
            "artifact_directory": str(
                experiment.artifact_dir.relative_to(artifacts_dir)
            ),
            "checkpoint_sha256": experiment.checkpoint_sha256,
            "normalization_sha256": sha256_file(
                experiment.artifact_dir / "normalization.npz"
            ),
            "test_r2": float(experiment.metrics["test_r2"]),
            "parameter_count": int(experiment.metrics["model"]["parameter_count"]),
            "experiment_spec": asdict(experiment.spec),
            "training_config": asdict(experiment.training_config),
            "agop": agop_provenance,
            "query_gradient_seconds": query_gradient_seconds,
            "methods": method_runtime,
            "scores": score_rows,
            "summary_file": str(summary_path.relative_to(output_dir)),
            "summary_sha256": sha256_file(summary_path),
        }
        experiment.model.cpu()
        _finalize_outputs(
            output_dir,
            report,
            data=data,
            query_index=query_index,
            target_index=target_index,
            target_nino3=target_nino3,
            standardizer=reference.standardizer,
        )
        _write_report(report_path, report)
        LOGGER.info("Completed %s", architecture.upper())

    _finalize_outputs(
        output_dir,
        report,
        data=data,
        query_index=query_index,
        target_index=target_index,
        target_nino3=target_nino3,
        standardizer=reference.standardizer,
    )
    _write_report(report_path, report)
    LOGGER.info("Fresh Table I status: %s", report["status"])
    LOGGER.info("Metadata: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
