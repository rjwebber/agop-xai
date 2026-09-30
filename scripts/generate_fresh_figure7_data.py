#!/usr/bin/env python3
"""Generate one canonical CNN-seed data bundle for manuscript Figure 7."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.artifacts import load_completed_bundle  # noqa: E402
from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.fresh_case_studies import (  # noqa: E402
    FreshXAISettings,
    build_explainers,
    common_fixed_test_targets,
    evaluate_fresh_case,
    experiment_metadata,
    fixed_test_extreme,
    settings_metadata,
    target_locked_event,
)
from zc_xai.fresh_figure7 import (  # noqa: E402
    load_common_target_cnn,
    load_or_build_exact_factor,
)
from zc_xai.fresh_xai_outputs import FRESH_XAI_BUNDLE_SCHEMA_VERSION  # noqa: E402
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    sha256_array,
    sha256_file,
    write_json,
    write_npz,
)
from zc_xai.training import resolve_device  # noqa: E402

LOGGER = logging.getLogger(__name__)
ARTIFACT = "fresh zc-v3 Figure 7 per-seed common-target core4 CNN XAI bundle"
FIGURE_NUMBER = 7
REQUIRED_FILES = ("scores", "explanations", "neighbors", "references")
LEGACY_FRESH_FIGURE9_MODULE_SHA256 = (
    "8ed7f9e48390b7f8eff12c7eab63b563bbae1132359a69251594a5ad14ceeabb"
)
LEGACY_FIGURE7_WRAPPER_SHA256 = (
    "7a745040688940f8e98c0dbd4ca80eecf75cbc84a498d405335bb77b652846a9"
)
LEGACY_FIGURE7_PIPELINE_SHA256 = (
    "6dce8d987a98d33b8fc18432b9ea102c48c80355b75dd97299b5575f2200b45b"
)
OUTPUT_FILENAMES = {
    "scores": "scores.csv",
    "explanations": "case_explanations.npz",
    "neighbors": "case_neighbors.npz",
    "references": "xai_references.npz",
    "metadata": "metadata.json",
    "completion": "completed.json",
}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one trained CNN seed for the five-seed Figure 7 XAI-score "
            "average."
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
        "--agop-cache-root", type=Path, default=Path("scratch/figure7_exact_agop")
    )
    parser.add_argument(
        "--output-dir", type=Path
    )
    parser.add_argument(
        "--lead-months", nargs="+", type=positive_int, default=list(range(1, 13))
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
        help="Rebuild exact AGOP matrices and eigensystems intentionally.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-complete",
        action="store_true",
        help=(
            "Return successfully when the requested output is already a fully "
            "hashed, configuration-matching completed bundle. With --overwrite, "
            "an invalid or partial bundle is regenerated."
        ),
    )
    return parser


def _prefix(destination: dict[str, np.ndarray], prefix: str, values: dict) -> None:
    for key, value in values.items():
        destination[f"{prefix}__{key}"] = np.asarray(value)


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


def _neighbor_arrays(result: Any) -> dict[str, np.ndarray]:
    sample = result.neighbors
    return {
        "query_index": np.asarray(sample.query_index, dtype=np.int64),
        "candidate_count": np.asarray(sample.candidate_count, dtype=np.int64),
        "neighbor_percent": np.asarray(sample.neighbor_percent, dtype=np.float64),
        "neighborhood_count": np.asarray(sample.neighborhood_count, dtype=np.int64),
        "neighborhood_indices": sample.neighborhood_indices.astype(np.int64),
        "neighborhood_rms_distances": sample.neighborhood_rms_distances.astype(
            np.float64
        ),
        "exhaustive": np.asarray(True),
    }


def generate_bundle(
    args: argparse.Namespace,
    *,
    artifact: str = ARTIFACT,
    figure_number: int = FIGURE_NUMBER,
    generator_path: Path | None = None,
) -> int:
    """Generate one seed bundle for a numbered lead-time XAI figure."""

    if not isinstance(artifact, str) or not artifact:
        raise ValueError("artifact must be a nonempty string")
    if figure_number <= 0:
        raise ValueError("figure_number must be positive")
    source_path = Path(__file__) if generator_path is None else generator_path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    leads = sorted(args.lead_months)
    if len(leads) != len(set(leads)):
        raise ValueError("--lead-months cannot contain duplicates.")
    maximum_lead = max(leads)
    data_dir = args.data_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    agop_root = args.agop_cache_root.expanduser().resolve()
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
    common_targets = common_fixed_test_targets(data, leads)
    canonical_extreme = fixed_test_extreme(data, lead_months=10, kind="maximum")
    if canonical_extreme.target_step not in common_targets:
        raise ValueError(
            "The canonical 10-month fixed-test El Nino extreme is outside the "
            "target dates shared by every requested lead."
        )

    experiments: dict[int, Any] = {}
    explainers: dict[int, dict[str, Any]] = {}
    experiment_records: dict[str, Any] = {}
    references_archive: dict[str, np.ndarray] = {}
    reference_fit_targets: np.ndarray | None = None
    reference_validation_targets: np.ndarray | None = None
    reference_test_targets: np.ndarray | None = None
    reference_mean: np.ndarray | None = None
    reference_scale: np.ndarray | None = None
    for lead in leads:
        LOGGER.info("Loading common-target Figure 5 CNN at lead %s", lead)
        experiment = load_common_target_cnn(
            data,
            artifacts_dir,
            lead_months=lead,
            seed=args.seed,
            maximum_lead_months=maximum_lead,
            device="cpu",
        )
        fit_targets = experiment.fit_inputs + experiment.lead_steps
        validation_targets = experiment.validation_inputs + experiment.lead_steps
        test_targets = experiment.test_inputs + experiment.lead_steps
        if reference_fit_targets is None:
            reference_fit_targets = fit_targets
            reference_validation_targets = validation_targets
            reference_test_targets = test_targets
            reference_mean = experiment.standardizer.mean
            reference_scale = experiment.standardizer.scale
        elif not (
            np.array_equal(fit_targets, reference_fit_targets)
            and np.array_equal(validation_targets, reference_validation_targets)
            and np.array_equal(test_targets, reference_test_targets)
            and np.array_equal(experiment.standardizer.mean, reference_mean)
            and np.array_equal(experiment.standardizer.scale, reference_scale)
        ):
            raise ValueError(
                "Figure 7 checkpoints do not share target dates and normalization."
            )
        factor, agop_record = load_or_build_exact_factor(
            data,
            experiment,
            agop_root / f"lead-{lead:02d}m-seed-{args.seed:06d}",
            gradient_batch_size=settings.gradient_batch_size,
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
        record = experiment_metadata(
            data,
            experiment,
            artifact_root=artifacts_dir,
            agop_provenance=agop_record,
        )
        record["common_target_dates"] = {
            "maximum_lead_months": maximum_lead,
            "fit_target_indices_sha256": sha256_array(fit_targets),
            "validation_target_indices_sha256": sha256_array(validation_targets),
            "test_target_indices_sha256": sha256_array(test_targets),
        }
        experiment_records[lead_key] = record
        _prefix(references_archive, lead_key, references)

    score_rows: list[dict[str, Any]] = []
    explanations_archive: dict[str, np.ndarray] = {}
    neighbors_archive: dict[str, np.ndarray] = {}
    case_records: dict[str, Any] = {}
    case_order = []
    for lead in leads:
        case_id = f"el_nino_lead_{lead:02d}"
        case_order.append(case_id)
        event = target_locked_event(
            data,
            case_id=case_id,
            lead_months=lead,
            target_step=canonical_extreme.target_step,
            selection_rule=(
                "target locked to the canonical 10-month fixed-test El Nino "
                "extreme and shared across every forecast lead"
            ),
        )
        if event.input_step not in experiments[lead].test_inputs:
            raise ValueError(f"{case_id} is outside the common-target test cache.")
        LOGGER.info(
            "Evaluating %s with all nearest-one-percent neighbors", case_id
        )
        result = evaluate_fresh_case(
            data,
            experiments[lead],
            explainers[lead],
            event,
            settings,
            device=device,
            validate_primary=False,
        )
        score_rows.extend(result.rows)
        _prefix(explanations_archive, case_id, result.arrays)
        _prefix(neighbors_archive, case_id, _neighbor_arrays(result))
        case_records[case_id] = {
            **asdict(event),
            "selection_split": "common target dates in fixed test block",
            "checkpoint_sha256": experiments[lead].checkpoint_sha256,
            "event_input_standardized_sha256": sha256_array(
                result.arrays["event_input_standardized"]
            ),
            "runtime": result.runtime,
        }

    _write_csv(paths["scores"], score_rows)
    write_npz(paths["explanations"], overwrite=True, **explanations_archive)
    write_npz(paths["neighbors"], overwrite=True, **neighbors_archive)
    write_npz(paths["references"], overwrite=True, **references_archive)
    output_records = {
        key: {"file": paths[key].name, "sha256": sha256_file(paths[key])}
        for key in ("scores", "explanations", "neighbors", "references")
    }
    metadata = {
        "schema_version": FRESH_XAI_BUNDLE_SCHEMA_VERSION,
        "artifact": artifact,
        "figure": figure_number,
        "data": data.provenance(),
        "input_profile": "core4",
        "spatial_fields": list(data.spatial_field_names),
        "phase_features": data.n_phase_features,
        "lead_months": leads,
        "case_order": case_order,
        "cases": case_records,
        "experiments": experiment_records,
        "anchor": {
            **asdict(canonical_extreme),
            "eligible_common_test_target_count": common_targets.size,
            "common_test_target_indices_sha256": sha256_array(common_targets),
        },
        "comparison_period": {
            "common_period_max_lead_months": maximum_lead,
            "shared_fit_target_dates": True,
            "shared_validation_target_dates": True,
            "shared_test_target_dates": True,
            "shared_normalization": True,
            "normalization_population": (
                "all raw states in the fixed 10,000-year training block"
            ),
            "checkpoint_reuse": (
                "full-10000-year core4 CNN checkpoints generated for the "
                "Figure 5 common-target lead-time panel"
            ),
        },
        "xai_configuration": settings_metadata(settings),
        "runtime": {
            "clock": "time.perf_counter wall time",
            "total_seconds": time.perf_counter() - started,
            "resolved_device": device,
        },
        "source_sha256": {
            "generator": sha256_file(source_path),
            "fresh_case_studies.py": sha256_file(
                REPOSITORY_ROOT / "src" / "zc_xai" / "fresh_case_studies.py"
            ),
            "fresh_figure7.py": sha256_file(
                REPOSITORY_ROOT / "src" / "zc_xai" / "fresh_figure7.py"
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
            "artifact": artifact,
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
    LOGGER.info("Fresh Figure %s bundle saved: %s", figure_number, output_dir)
    return 0


def _has_compatible_source_provenance(source_sha256: Any) -> bool:
    """Accept current sources and immutable bundles made under the old filename.

    Figure 7 was originally generated by a thin Figure-7 wrapper around a file
    then named ``generate_fresh_figure9_data.py``.  Completed bundles retain
    those historical keys and hashes.  Their own completion manifests protect
    those records, so accepting that layout does not rewrite provenance.
    """

    if not isinstance(source_sha256, dict):
        return False
    shared_sources_match = (
        source_sha256.get("fresh_case_studies.py")
        == sha256_file(REPOSITORY_ROOT / "src" / "zc_xai" / "fresh_case_studies.py")
        and source_sha256.get("xai.py")
        == sha256_file(REPOSITORY_ROOT / "src" / "zc_xai" / "xai.py")
    )
    canonical = (
        source_sha256.get("generator") == sha256_file(Path(__file__))
        and source_sha256.get("fresh_figure7.py")
        == sha256_file(REPOSITORY_ROOT / "src" / "zc_xai" / "fresh_figure7.py")
    )
    legacy = (
        source_sha256.get("generator") == LEGACY_FIGURE7_WRAPPER_SHA256
        and source_sha256.get("numerical_pipeline")
        == LEGACY_FIGURE7_PIPELINE_SHA256
        and source_sha256.get("fresh_figure9.py")
        == LEGACY_FRESH_FIGURE9_MODULE_SHA256
    )
    return shared_sources_match and (canonical or legacy)


def _is_matching_completed_bundle(args: argparse.Namespace) -> bool:
    """Return whether ``args.output_dir`` is a valid bundle for this request."""

    try:
        bundle = load_completed_bundle(
            args.output_dir,
            expected_artifact=ARTIFACT,
            required_files=REQUIRED_FILES,
        )
    except (FileNotFoundError, ValueError):
        return False
    metadata = bundle.metadata
    leads = sorted(args.lead_months)
    experiments = metadata.get("experiments")
    if (
        metadata.get("figure") != FIGURE_NUMBER
        or metadata.get("lead_months") != leads
        or metadata.get("input_profile") != "core4"
        or metadata.get("xai_configuration")
        != settings_metadata(FreshXAISettings())
        or not _has_compatible_source_provenance(metadata.get("source_sha256"))
        or not isinstance(experiments, dict)
        or set(experiments) != {f"lead_{lead:02d}" for lead in leads}
    ):
        return False
    for lead in leads:
        record = experiments[f"lead_{lead:02d}"]
        spec = record.get("spec") if isinstance(record, dict) else None
        if (
            not isinstance(spec, dict)
            or spec.get("seed") != args.seed
            or spec.get("architecture") != "cnn"
            or spec.get("input_profile") != "core4"
            or spec.get("lead_months") != lead
            or spec.get("train_years") != 10_000.0
            or spec.get("common_period_max_lead_months") != max(leads)
        ):
            return False
    return True


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = Path(
            f"artifacts/zc-v3/manuscript/figure7/seed-{args.seed:06d}"
        )
    if (
        args.resume_complete
        and not args.overwrite_agop
        and _is_matching_completed_bundle(args)
    ):
        LOGGER.info(
            "Validated completed Figure 7 seed %s bundle; nothing to do: %s",
            args.seed,
            args.output_dir,
        )
        return 0
    return generate_bundle(args)


if __name__ == "__main__":
    raise SystemExit(main())
