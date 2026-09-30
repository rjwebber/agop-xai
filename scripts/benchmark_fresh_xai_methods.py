#!/usr/bin/env python3
"""Benchmark fresh-ZC XAI explanations at the canonical robustness sample.

The default experiment uses the core4 CNN, the most extreme held-out El Nino
target at a 10-month lead, and 256 uniformly sampled states from the nearest
one percent of fixed training predictors in standardized coordinates. Target
and neighbor explanations are cached and timed separately.

AGOP explanation timing uses the validated complete eigensystem created by
``benchmark_fresh_agop.py``. Its cold gradient, dense-accumulation, and EVD
timings are imported as separate provenance and are never charged to the
cached AGOP explanation application.
"""

from __future__ import annotations

import argparse
import logging
import math
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import benchmark_fresh_agop as fresh_agop  # noqa: E402

from zc_xai.data import FRESH_INPUT_PROFILES, FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.fresh_xai_outputs import FRESH_XAI_TRAINING_POPULATION  # noqa: E402
from zc_xai.io import (  # noqa: E402
    load_json,
    sha256_array,
    sha256_file,
    sha256_json,
    write_json,
    write_npz,
)
from zc_xai.models import ARCHITECTURES  # noqa: E402
from zc_xai.training import (  # noqa: E402
    ExperimentSpec,
    LoadedExperiment,
    load_experiment,
    resolve_device,
)
from zc_xai.xai import (  # noqa: E402
    AgopExplainer,
    AgopFactor,
    GradientExplainer,
    GradientShapExplainer,
    IntegratedGradientsExplainer,
    NeighborSample,
    input_gradients,
    sample_empirical_neighbors,
    unit_rows,
)

LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "1.1.0"
REPORT_SCHEMA_VERSION = 2
CACHE_SCHEMA_VERSION = 2
DEFAULT_OUTPUT_ROOT = Path("outputs/fresh_xai_method_benchmark")


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


def percent(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise argparse.ArgumentTypeError("percentage must lie in (0, 100]")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Time GRAD, integrated gradients, expected gradients/GradientSHAP, "
            "and cached exact AGOP on one fresh-ZC event and its robustness sample."
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
        "--agop-benchmark-dir",
        type=Path,
        help=(
            "Completed all-reference exact-AGOP benchmark directory. The default "
            "is inferred from profile, architecture, lead, and seed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Method caches and report directory; a parameterized default is used.",
    )
    parser.add_argument(
        "--input-profile",
        choices=tuple(FRESH_INPUT_PROFILES),
        default="core4",
    )
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="cnn")
    parser.add_argument("--lead-months", type=positive_int, default=10)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--neighbor-percent", type=percent, default=1.0)
    parser.add_argument("--neighbor-samples", type=positive_int, default=256)
    parser.add_argument("--neighbor-seed", type=nonnegative_int, default=42)
    parser.add_argument("--distance-batch-size", type=positive_int, default=8192)
    parser.add_argument(
        "--ig-steps",
        type=positive_int,
        default=1024,
        help=(
            "Equally spaced right-endpoint path gradients per IG explanation. "
            "The manuscript production rule is exactly 1024."
        ),
    )
    parser.add_argument(
        "--gradient-shap-samples",
        type=positive_int,
        default=1024,
        help=(
            "Distinct empirical baseline/independent-alpha pairs per expected-"
            "gradients explanation. The manuscript production rule is exactly 1024."
        ),
    )
    parser.add_argument(
        "--gradient-shap-seed", type=nonnegative_int, default=42
    )
    parser.add_argument("--gradient-batch-size", type=positive_int, default=64)
    parser.add_argument(
        "--fused-pair-batch-size",
        type=positive_int,
        default=256,
        help=(
            "Canonical streaming batch of query-sample pairs for fused IG and "
            "expected gradients."
        ),
    )
    parser.add_argument(
        "--execution-modes",
        nargs="+",
        choices=("current", "fused"),
        default=("current", "fused"),
        help=(
            "Implementations timed for IG and expected gradients. Very high-sample "
            "runs can select fused alone to avoid the slower per-query baseline."
        ),
    )
    parser.add_argument(
        "--skip-data-checksums",
        action="store_true",
        help="Development only; the omission is recorded in the report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically recompute every setup and explanation cache.",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_name(args: argparse.Namespace) -> str:
    return (
        f"{args.input_profile}-{args.architecture}-lead-{args.lead_months:02d}m-"
        f"seed-{args.seed:06d}-neighbors-{args.neighbor_samples}-"
        f"nearest-{args.neighbor_percent:g}pct-ig-{args.ig_steps}-"
        f"eg-{args.gradient_shap_samples}-modes-{'-'.join(args.execution_modes)}"
    )


def _canonicalize_execution_modes(args: argparse.Namespace) -> None:
    # Canonicalize duplicate modes while retaining user-visible order.
    args.execution_modes = tuple(dict.fromkeys(args.execution_modes))


def method_variants(execution_modes: tuple[str, ...]) -> tuple[str, ...]:
    variants = ["GRAD"]
    variants.extend(f"IG-{mode}" for mode in execution_modes)
    variants.extend(f"GradientSHAP-{mode}" for mode in execution_modes)
    variants.append("AGOP")
    return tuple(variants)


def _variant_parts(variant: str) -> tuple[str, str]:
    if variant in {"GRAD", "AGOP"}:
        return variant, "native"
    for method in ("IG", "GradientSHAP"):
        prefix = method + "-"
        if variant.startswith(prefix):
            mode = variant[len(prefix) :]
            if mode in {"current", "fused"}:
                return method, mode
    raise ValueError(f"Unknown method variant: {variant!r}")


def _default_agop_benchmark_dir(args: argparse.Namespace) -> Path:
    return fresh_agop.DEFAULT_OUTPUT_ROOT / (
        f"{args.input_profile}-{args.architecture}-lead-{args.lead_months:02d}m-"
        f"seed-{args.seed:06d}-refs-all"
    )


def _safe_basename(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).is_absolute()
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{label} is not a safe portable basename: {value!r}")
    return value


def _artifact_id(artifact_dir: Path, artifact_root: Path) -> str:
    try:
        return artifact_dir.resolve().relative_to(artifact_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "The model artifact is outside --artifacts-dir; portable provenance "
            "cannot be constructed."
        ) from error


def _software(device: str) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
    }


def _machine() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def _validate_cached_arrays(arrays: dict[str, np.ndarray]) -> None:
    if not arrays:
        raise ValueError("A benchmark cache cannot be empty.")
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Cached array {name!r} contains non-finite values.")


def load_or_compute_npz(
    path: Path,
    *,
    stage: str,
    identity: dict[str, Any],
    compute: Callable[[], dict[str, np.ndarray]],
    overwrite: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Atomically compute or content-validate one resumable benchmark stage."""

    identity_sha256 = sha256_json(identity)
    manifest_path = _manifest_path(path)
    if path.exists() and not overwrite:
        started = time.perf_counter()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Incomplete {stage} cache: {manifest_path.name} is missing. "
                "Use --overwrite to rebuild it."
            )
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("stage") != stage
            or manifest.get("identity_sha256") != identity_sha256
            or not isinstance(manifest.get("identity"), dict)
            or sha256_json(manifest["identity"]) != identity_sha256
        ):
            raise ValueError(
                f"The existing {stage} cache has another identity. Choose a new "
                "--output-dir or pass --overwrite."
            )
        digest = sha256_file(path)
        if manifest.get("file_sha256") != digest:
            raise ValueError(f"The {stage} cache is incomplete or corrupted: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["identity_sha256"].item()) != identity_sha256:
                raise ValueError(f"The {stage} payload identity is corrupted.")
            arrays = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "identity_sha256"
            }
        _validate_cached_arrays(arrays)
        return arrays, {
            "cache_hit": True,
            "cache_validation_seconds_this_invocation": (
                time.perf_counter() - started
            ),
            "compute_seconds": float(manifest["compute_seconds"]),
            "file": path.name,
            "file_sha256": digest,
        }

    started = time.perf_counter()
    arrays = compute()
    compute_seconds = time.perf_counter() - started
    _validate_cached_arrays(arrays)
    write_npz(
        path,
        overwrite=True,
        compressed=False,
        **arrays,
        identity_sha256=np.asarray(identity_sha256),
    )
    digest = sha256_file(path)
    write_json(
        manifest_path,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage": stage,
            "created_at_utc": _utc_now(),
            "identity": identity,
            "identity_sha256": identity_sha256,
            "compute_seconds": compute_seconds,
            "file": path.name,
            "file_sha256": digest,
        },
        overwrite=True,
    )
    return arrays, {
        "cache_hit": False,
        "cache_validation_seconds_this_invocation": 0.0,
        "compute_seconds": compute_seconds,
        "file": path.name,
        "file_sha256": digest,
    }


def method_workload(
    method: str,
    *,
    execution_mode: str,
    query_count: int,
    gradient_batch_size: int,
    fused_pair_batch_size: int,
    ig_steps: int,
    gradient_shap_samples: int,
) -> dict[str, Any]:
    """Describe the actual loop/batch structure used by each production explainer."""

    if (
        query_count <= 0
        or gradient_batch_size <= 0
        or fused_pair_batch_size <= 0
    ):
        raise ValueError("query and batch counts must be positive")
    if method == "GRAD":
        if execution_mode != "native":
            raise ValueError("GRAD uses only its native batched implementation.")
        rows = query_count
        calls = 1
        batches = math.ceil(query_count / gradient_batch_size)
        rows_per_query = 1
        structure = "one input_gradients call for the complete query population"
    elif method == "IG":
        rows = query_count * ig_steps
        if execution_mode == "current":
            calls = query_count
            batches = query_count * math.ceil(ig_steps / gradient_batch_size)
            batch_size = gradient_batch_size
            structure = "one interpolation path and input_gradients call per query"
        elif execution_mode == "fused":
            batches = math.ceil(rows / fused_pair_batch_size)
            calls = batches
            batch_size = fused_pair_batch_size
            structure = (
                "canonical query-major pair stream; online per-query reduction "
                "without a cartesian input array"
            )
        else:
            raise ValueError("IG execution_mode must be current or fused.")
        rows_per_query = ig_steps
    elif method == "GradientSHAP":
        rows = query_count * gradient_shap_samples
        if execution_mode == "current":
            calls = query_count
            batches = query_count * math.ceil(
                gradient_shap_samples / gradient_batch_size
            )
            batch_size = gradient_batch_size
            structure = (
                "one fixed-background interpolation set and input_gradients call "
                "per query"
            )
        elif execution_mode == "fused":
            batches = math.ceil(rows / fused_pair_batch_size)
            calls = batches
            batch_size = fused_pair_batch_size
            structure = (
                "canonical query-major pair stream; online per-query reduction "
                "without a cartesian input array"
            )
        else:
            raise ValueError(
                "GradientSHAP execution_mode must be current or fused."
            )
        rows_per_query = gradient_shap_samples
    elif method == "AGOP":
        if execution_mode != "native":
            raise ValueError("AGOP uses only its native cached-factor application.")
        return {
            "queries": query_count,
            "gradient_rows": 0,
            "input_gradients_calls": 0,
            "gradient_batches": 0,
            "gradient_rows_per_query": 0,
            "effective_pair_batch_size": None,
            "nominal_gradient_batches_per_query": 0,
            "gradient_row_definition": "not applicable; AGOP uses no query gradients",
            "structure": (
                "cached full-basis M^(1/2) application; two dense basis "
                "multiplications per query population"
            ),
        }
    else:
        raise ValueError(f"Unknown XAI method: {method!r}")
    return {
        "queries": query_count,
        "gradient_rows": rows,
        "input_gradients_calls": calls,
        "gradient_batches": batches,
        "gradient_rows_per_query": rows_per_query,
        "effective_pair_batch_size": (
            gradient_batch_size if method == "GRAD" else batch_size
        ),
        "nominal_gradient_batches_per_query": (
            None
            if method == "GRAD"
            else math.ceil(rows_per_query / batch_size)
        ),
        "gradient_row_definition": (
            "one scalar forecast gradient with respect to one interpolated "
            "standardized input"
        ),
        "structure": structure,
    }


def _pair_stream_inputs(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    inputs = np.asarray(values, dtype=np.float32)
    if inputs.ndim < 2 or inputs.shape[0] == 0 or inputs[0].size == 0:
        raise ValueError("Fused XAI requires a nonempty query batch.")
    if not np.isfinite(inputs).all():
        raise ValueError("Fused XAI inputs contain non-finite values.")
    return inputs, inputs.shape[0], inputs[0].size


def _accumulate_canonical_rows(
    destination: np.ndarray,
    query_indices: np.ndarray,
    rows: np.ndarray,
) -> None:
    """Accumulate contiguous query-major row segments with float64 reductions."""

    indices = np.asarray(query_indices, dtype=np.int64)
    values = np.asarray(rows)
    if (
        indices.ndim != 1
        or values.ndim != 2
        or values.shape[0] != indices.size
        or indices.size == 0
        or np.any(np.diff(indices) < 0)
    ):
        raise ValueError("Canonical pair rows must be nonempty and query-major.")
    segment_starts = np.concatenate(
        (np.asarray([0]), np.flatnonzero(np.diff(indices)) + 1)
    )
    segment_stops = np.concatenate((segment_starts[1:], [indices.size]))
    for start, stop in zip(segment_starts, segment_stops, strict=True):
        destination[indices[start]] += values[start:stop].sum(
            axis=0,
            dtype=np.float64,
        )


def fused_integrated_gradients(
    model: torch.nn.Module,
    inputs: np.ndarray,
    *,
    n_steps: int,
    pair_batch_size: int,
    device: str,
) -> np.ndarray:
    """Stream a canonical query-by-path grid and reduce IG online per query."""

    values, query_count, feature_count = _pair_stream_inputs(inputs)
    if n_steps <= 0 or pair_batch_size <= 0:
        raise ValueError("n_steps and pair_batch_size must be positive")
    total_pairs = query_count * n_steps
    gradient_sums = np.zeros((query_count, feature_count), dtype=np.float64)
    alpha_grid = np.linspace(
        1.0 / n_steps,
        1.0,
        num=n_steps,
        dtype=np.float32,
    )
    for start in range(0, total_pairs, pair_batch_size):
        stop = min(start + pair_batch_size, total_pairs)
        pair_indices = np.arange(start, stop, dtype=np.int64)
        query_indices = pair_indices // n_steps
        sample_indices = pair_indices % n_steps
        alphas = alpha_grid[sample_indices]
        alpha_shape = (-1,) + (1,) * (values.ndim - 1)
        interpolated = values[query_indices] * alphas.reshape(alpha_shape)
        gradients = input_gradients(
            model,
            interpolated,
            batch_size=pair_batch_size,
            device=device,
        ).reshape(stop - start, feature_count)
        _accumulate_canonical_rows(gradient_sums, query_indices, gradients)
    attributions = values.reshape(query_count, feature_count).astype(
        np.float64
    ) * (gradient_sums / n_steps)
    return unit_rows(attributions, label="fused IG")


def fused_expected_gradients(
    model: torch.nn.Module,
    inputs: np.ndarray,
    *,
    backgrounds: np.ndarray,
    alphas: np.ndarray,
    pair_batch_size: int,
    device: str,
) -> np.ndarray:
    """Stream query-background pairs and reduce expected gradients online."""

    values, query_count, feature_count = _pair_stream_inputs(inputs)
    background_values = np.asarray(backgrounds, dtype=np.float32)
    alpha_values = np.asarray(alphas, dtype=np.float32)
    if (
        background_values.ndim != values.ndim
        or background_values.shape[1:] != values.shape[1:]
        or background_values.shape[0] == 0
        or alpha_values.shape != (background_values.shape[0],)
        or pair_batch_size <= 0
        or not np.isfinite(background_values).all()
        or not np.isfinite(alpha_values).all()
        or np.any((alpha_values < 0.0) | (alpha_values > 1.0))
    ):
        raise ValueError("Fused expected-gradient backgrounds or alphas are invalid.")
    sample_count = background_values.shape[0]
    total_pairs = query_count * sample_count
    attribution_sums = np.zeros((query_count, feature_count), dtype=np.float64)
    for start in range(0, total_pairs, pair_batch_size):
        stop = min(start + pair_batch_size, total_pairs)
        pair_indices = np.arange(start, stop, dtype=np.int64)
        query_indices = pair_indices // sample_count
        sample_indices = pair_indices % sample_count
        differences = values[query_indices] - background_values[sample_indices]
        alpha_shape = (-1,) + (1,) * (values.ndim - 1)
        interpolated = background_values[sample_indices] + alpha_values[
            sample_indices
        ].reshape(alpha_shape) * differences
        gradients = input_gradients(
            model,
            interpolated,
            batch_size=pair_batch_size,
            device=device,
        )
        # Match the current explainer's arithmetic: pairwise products are
        # float32, while the Monte Carlo reduction is accumulated in float64.
        contributions = np.multiply(differences, gradients).reshape(
            stop - start, feature_count
        ).astype(np.float64)
        _accumulate_canonical_rows(
            attribution_sums,
            query_indices,
            contributions,
        )
    attributions = attribution_sums / sample_count
    return unit_rows(attributions, label="fused expected gradients")


@dataclass
class FusedIntegratedGradientsExplainer:
    model: torch.nn.Module
    n_steps: int
    pair_batch_size: int
    device: str

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        return fused_integrated_gradients(
            self.model,
            inputs,
            n_steps=self.n_steps,
            pair_batch_size=self.pair_batch_size,
            device=self.device,
        )


@dataclass
class FusedGradientShapExplainer:
    model: torch.nn.Module
    backgrounds: np.ndarray
    alphas: np.ndarray
    pair_batch_size: int
    device: str

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        return fused_expected_gradients(
            self.model,
            inputs,
            backgrounds=self.backgrounds,
            alphas=self.alphas,
            pair_batch_size=self.pair_batch_size,
            device=self.device,
        )


def _extreme_test_event(
    data: ZCData,
    experiment: LoadedExperiment,
) -> tuple[int, int, float]:
    targets = data.load_targets(
        experiment.test_inputs,
        lead_steps=experiment.lead_steps,
    )
    position = int(np.argmax(targets))
    input_step = int(experiment.test_inputs[position])
    target_step = input_step + experiment.lead_steps
    return input_step, target_step, float(targets[position])


def _neighbor_arrays(sample: NeighborSample) -> dict[str, np.ndarray]:
    return {
        "query_index": np.asarray(sample.query_index, dtype=np.int64),
        "candidate_count": np.asarray(sample.candidate_count, dtype=np.int64),
        "neighbor_percent": np.asarray(sample.neighbor_percent, dtype=np.float64),
        "neighborhood_count": np.asarray(sample.neighborhood_count, dtype=np.int64),
        "sampled_indices": np.asarray(sample.sampled_indices, dtype=np.int64),
        "sampled_rms_distances": np.asarray(
            sample.sampled_rms_distances, dtype=np.float64
        ),
        "neighborhood_indices": np.asarray(
            sample.neighborhood_indices, dtype=np.int64
        ),
        "neighborhood_rms_distances": np.asarray(
            sample.neighborhood_rms_distances, dtype=np.float64
        ),
        "seed": np.asarray(sample.seed, dtype=np.int64),
    }


def _fixed_expected_gradient_references(
    candidates: np.ndarray,
    *,
    count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if count > candidates.size:
        raise ValueError(
            f"Expected-gradients requested {count:,} backgrounds from only "
            f"{candidates.size:,} training predictors."
        )
    rng = np.random.default_rng(seed)
    indices = np.asarray(
        rng.choice(candidates, size=count, replace=False),
        dtype=np.int64,
    )
    alphas = np.random.default_rng(seed + 1).uniform(
        0.0,
        1.0,
        size=count,
    ).astype(np.float32)
    return {"background_indices": indices, "alphas": alphas}


def load_validated_full_agop_factor(
    benchmark_dir: Path,
    *,
    data: ZCData,
    experiment: LoadedExperiment,
    fixed_training_inputs: np.ndarray,
) -> tuple[AgopFactor, dict[str, Any]]:
    """Load a full exact factor and bind it to this data/model/normalizer."""

    started = time.perf_counter()
    directory = benchmark_dir.expanduser().resolve()
    report_path = directory / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Complete the exact-AGOP benchmark first; missing {report_path}."
        )
    report = load_json(report_path)
    run = report.get("run_identity")
    if (
        report.get("schema_version") != fresh_agop.REPORT_SCHEMA_VERSION
        or report.get("status") != "complete"
        or not isinstance(run, dict)
        or report.get("run_identity_sha256") != sha256_json(run)
    ):
        raise ValueError("The exact-AGOP benchmark report is incomplete or invalid.")
    if run.get("script_sha256") != sha256_file(
        REPOSITORY_ROOT / "scripts" / "benchmark_fresh_agop.py"
    ):
        raise ValueError(
            "The exact-AGOP report was built by another benchmark source version."
        )
    expected_model = {
        "architecture": experiment.spec.architecture,
        "input_profile": data.input_profile,
        "lead_months": experiment.spec.lead_months,
        "seed": experiment.spec.seed,
        "checkpoint_sha256": experiment.checkpoint_sha256,
    }
    recorded_model = run.get("model")
    if not isinstance(recorded_model, dict) or any(
        recorded_model.get(key) != value for key, value in expected_model.items()
    ):
        raise ValueError("The exact-AGOP eigensystem belongs to another model.")
    recorded_data = run.get("data")
    if (
        not isinstance(recorded_data, dict)
        or recorded_data.get("metadata_sha256") != data.metadata_sha256
        or recorded_data.get("input_profile") != data.input_profile
        or recorded_data.get("input_shape") != list(data.input_shape)
    ):
        raise ValueError("The exact-AGOP eigensystem belongs to another data view.")
    recorded_standardizer = run.get("standardizer")
    expected_standardizer = {
        "mean_sha256": sha256_array(experiment.standardizer.mean),
        "scale_sha256": sha256_array(experiment.standardizer.scale),
        "count": experiment.standardizer.count,
    }
    if not isinstance(recorded_standardizer, dict) or any(
        recorded_standardizer.get(key) != value
        for key, value in expected_standardizer.items()
    ):
        raise ValueError("The exact-AGOP eigensystem uses another normalizer.")
    recorded_references = run.get("references")
    fixed = np.asarray(fixed_training_inputs, dtype=np.int64)
    if (
        not isinstance(recorded_references, dict)
        or recorded_references.get("requested_count_zero_means_all") != 0
        or recorded_references.get("count") != fixed.size
        or recorded_references.get("indices_sha256") != sha256_array(fixed)
    ):
        raise ValueError(
            "Method timing requires the exact AGOP fitted to every fixed training "
            "predictor, not a prefix benchmark."
        )
    numeric = run.get("numeric_method")
    if (
        not isinstance(numeric, dict)
        or numeric.get("full_basis") is not True
        or numeric.get("matrix_dtype") != np.dtype(np.float64).str
        or numeric.get("eigendecomposition_dtype") != np.dtype(np.float64).str
    ):
        raise ValueError("The AGOP benchmark is not the required full float64 solve.")

    output_files = report.get("output_files")
    cold_stages = report.get("stages")
    if not isinstance(output_files, dict) or not isinstance(cold_stages, dict):
        raise ValueError("The AGOP report has no valid stage/output manifest.")
    eigensystem_name = _safe_basename(
        output_files.get("full_eigensystem_cache"),
        "full_eigensystem_cache",
    )
    eigensystem_path = directory / eigensystem_name
    manifest_path = _manifest_path(eigensystem_path)
    if not eigensystem_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("The AGOP eigensystem cache is incomplete.")
    eigensystem_manifest = load_json(manifest_path)
    identity = eigensystem_manifest.get("identity")
    digest = sha256_file(eigensystem_path)
    if (
        eigensystem_manifest.get("schema_version")
        != fresh_agop.CACHE_SCHEMA_VERSION
        or eigensystem_manifest.get("stage") != "full_eigensystem"
        or not isinstance(identity, dict)
        or eigensystem_manifest.get("identity_sha256") != sha256_json(identity)
        or eigensystem_manifest.get("file_sha256") != digest
        or not isinstance(cold_stages.get("eigensystem"), dict)
        or cold_stages["eigensystem"].get("file_sha256") != digest
    ):
        raise ValueError("The AGOP eigensystem cache failed identity validation.")
    with np.load(eigensystem_path, allow_pickle=False) as archive:
        raw_basis = archive["basis"]
        raw_root = archive["root_eigenvalues"]
        if raw_basis.dtype != np.float64 or raw_root.dtype != np.float64:
            raise ValueError("The AGOP eigensystem payload is not float64.")
        basis = np.asarray(raw_basis).copy()
        root_eigenvalues = np.asarray(raw_root).copy()
    dimension = data.n_features
    if (
        basis.shape != (dimension, dimension)
        or root_eigenvalues.shape != (dimension,)
        or not np.isfinite(basis).all()
        or not np.isfinite(root_eigenvalues).all()
        or np.any(root_eigenvalues < 0.0)
    ):
        raise ValueError("The AGOP eigensystem payload has an invalid schema.")
    factor = AgopFactor(
        basis=basis,
        root_eigenvalues=root_eigenvalues,
        center=np.zeros(dimension, dtype=np.float64),
        reference_indices=fixed,
        approximation_rank=dimension,
        solver_metadata={
            "method": "exact_dense_empirical_agop",
            "source_report_sha256": sha256_file(report_path),
        },
    )
    if not all(
        isinstance(cold_stages.get(stage), dict)
        for stage in ("input_gradients", "dense_matrix", "eigensystem")
    ):
        raise ValueError("The AGOP report has incomplete cold-build stage timing.")
    provenance = {
        "benchmark_directory_name": directory.name,
        "report_file": report_path.name,
        "report_sha256": sha256_file(report_path),
        "eigensystem_file": eigensystem_path.name,
        "eigensystem_sha256": digest,
        "reference_count": fixed.size,
        "rank": dimension,
        "load_and_validation_seconds_this_invocation": (
            time.perf_counter() - started
        ),
        "cold_build_stages": {
            stage: {
                "build_timing": cold_stages[stage].get("build_timing", {}),
                "cache_hit_when_report_last_written": cold_stages[stage].get(
                    "cache_hit"
                ),
            }
            for stage in ("input_gradients", "dense_matrix", "eigensystem")
        },
        "cold_measured_component_seconds": report.get(
            "measured_component_seconds", {}
        ),
        "timing_policy": (
            "cold build is imported provenance and excluded from cached AGOP "
            "target/neighbor explanation timings"
        ),
    }
    return factor, provenance


def _explanation_arrays(explainer: Any, inputs: np.ndarray) -> dict[str, np.ndarray]:
    explanations = np.asarray(explainer.explain(inputs), dtype=np.float32)
    if explanations.shape != (inputs.shape[0], inputs[0].size):
        raise ValueError("An XAI method returned an unexpected explanation shape.")
    norms = np.linalg.norm(explanations.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=2.0e-6, atol=2.0e-6):
        raise ValueError("An XAI method returned non-unit explanations.")
    return {"explanations": explanations}


def _method_parameters(
    method: str,
    *,
    execution_mode: str,
    args: argparse.Namespace,
    background_indices: np.ndarray,
    alphas: np.ndarray,
    agop_provenance: dict[str, Any],
) -> dict[str, Any]:
    if method == "GRAD":
        return {
            "definition": "input gradient at standardized input",
            "execution_mode": execution_mode,
        }
    if method == "IG":
        return {
            "steps": args.ig_steps,
            "baseline": "zero vector in training-standardized coordinates",
            "quadrature": "right-endpoint uniform path grid",
            "execution_mode": execution_mode,
            "current_gradient_batch_size": args.gradient_batch_size,
            "fused_pair_batch_size": args.fused_pair_batch_size,
        }
    if method == "GradientSHAP":
        return {
            "definition": "expected gradients with fixed backgrounds and alphas",
            "samples": args.gradient_shap_samples,
            "baseline_sampling": (
                "distinct empirical fixed-training predictors sampled without "
                "replacement"
            ),
            "alpha_sampling": (
                "one independent Uniform(0,1) alpha paired with each baseline"
            ),
            "background_seed": args.gradient_shap_seed,
            "alpha_seed": args.gradient_shap_seed + 1,
            "background_indices_sha256": sha256_array(background_indices),
            "alphas_sha256": sha256_array(alphas),
            "reuse": (
                "identical paired backgrounds/alphas for target plus all "
                f"{args.neighbor_samples} neighbors (common random numbers)"
            ),
            "execution_mode": execution_mode,
            "current_gradient_batch_size": args.gradient_batch_size,
            "fused_pair_batch_size": args.fused_pair_batch_size,
        }
    if method == "AGOP":
        return {
            "definition": "full exact empirical M^(1/2) times standardized input",
            "factor_sha256": agop_provenance["eigensystem_sha256"],
            "reference_count": agop_provenance["reference_count"],
            "rank": agop_provenance["rank"],
            "execution_mode": execution_mode,
        }
    raise ValueError(f"Unknown method: {method!r}")


def _explainer(
    method: str,
    *,
    execution_mode: str,
    experiment: LoadedExperiment,
    args: argparse.Namespace,
    backgrounds: np.ndarray,
    alphas: np.ndarray,
    agop_factor: AgopFactor,
    device: str,
) -> Any:
    if method == "GRAD":
        return GradientExplainer(
            experiment.model,
            batch_size=args.gradient_batch_size,
            device=device,
        )
    if method == "IG":
        if execution_mode == "fused":
            return FusedIntegratedGradientsExplainer(
                experiment.model,
                n_steps=args.ig_steps,
                pair_batch_size=args.fused_pair_batch_size,
                device=device,
            )
        return IntegratedGradientsExplainer(
            experiment.model,
            n_steps=args.ig_steps,
            gradient_batch_size=args.gradient_batch_size,
            device=device,
        )
    if method == "GradientSHAP":
        if execution_mode == "fused":
            return FusedGradientShapExplainer(
                experiment.model,
                backgrounds=backgrounds,
                alphas=alphas,
                pair_batch_size=args.fused_pair_batch_size,
                device=device,
            )
        return GradientShapExplainer(
            experiment.model,
            backgrounds=backgrounds,
            alphas=alphas,
            gradient_batch_size=args.gradient_batch_size,
            device=device,
        )
    if method == "AGOP":
        return AgopExplainer(agop_factor)
    raise ValueError(f"Unknown method: {method!r}")


def _report_payload(
    *,
    run_identity: dict[str, Any],
    run_identity_sha256: str,
    setup: dict[str, Any],
    methods: dict[str, Any],
    agop_provenance: dict[str, Any],
) -> dict[str, Any]:
    expected_variants = run_identity.get("method_variants", [])
    neighbor_design = run_identity.get("neighbor_design", {})
    baseline_count = int(agop_provenance.get("reference_count", 0))
    total_queries = 1 + int(
        neighbor_design.get("uniform_sample_without_replacement", 0)
    )
    complete = all(
        variant in methods
        and "target" in methods[variant]
        and "neighbors" in methods[variant]
        for variant in expected_variants
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "complete" if complete else "in_progress",
        "updated_at_utc": _utc_now(),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "machine": _machine(),
        "setup": setup,
        "methods": methods,
        "agop_cold_build": agop_provenance,
        "interpretation": {
            "coordinates": "training-standardized input coordinates throughout",
            "target_and_neighbors": "timed and cached separately for every method",
            "gradient_shap_name": (
                "expected gradients approximation with fixed paired backgrounds "
                "and interpolation alphas; not exact Shapley values"
            ),
            "current_vs_fused": (
                "current reproduces the production per-query explainers; fused "
                "streams query-sample pairs in canonical order and reduces online"
            ),
            "gradient_row": (
                "one row means one scalar-output model gradient at one "
                "query-specific interpolated standardized input"
            ),
            "common_random_numbers": (
                "expected gradients reuses baseline indices and alpha values, "
                "not gradients, across the target and neighbors"
            ),
            "full_training_baseline_counterfactual": {
                "baselines_per_query": baseline_count,
                "queries": total_queries,
                "gradient_rows": baseline_count * total_queries,
                "why_rows_are_not_globally_shareable": (
                    "for baseline b_i and reused alpha_i, the interpolated point "
                    "b_i + alpha_i * (x_j - b_i) depends on query x_j, so its "
                    "model gradient must be evaluated again for every query"
                ),
            },
            "agop_timing": (
                "cached factor application only; cold gradient/M/EVD timings are "
                "reported separately under agop_cold_build"
            ),
            "robustness_scope": (
                f"{neighbor_design.get('uniform_sample_without_replacement')} states "
                "sampled uniformly without replacement from the nearest "
                f"{neighbor_design.get('nearest_percent')} percent population, "
                "not all members of that population"
            ),
        },
    }


def _write_progress(
    path: Path,
    *,
    run_identity: dict[str, Any],
    run_identity_sha256: str,
    setup: dict[str, Any],
    methods: dict[str, Any],
    agop_provenance: dict[str, Any],
) -> None:
    write_json(
        path,
        _report_payload(
            run_identity=run_identity,
            run_identity_sha256=run_identity_sha256,
            setup=setup,
            methods=methods,
            agop_provenance=agop_provenance,
        ),
        overwrite=True,
    )


def _implementation_comparisons(
    output_dir: Path,
    execution_modes: tuple[str, ...],
) -> dict[str, Any]:
    if not {"current", "fused"}.issubset(execution_modes):
        return {
            "performed": False,
            "reason": "both current and fused execution modes were not requested",
        }
    comparisons: dict[str, Any] = {"performed": True, "methods": {}}
    for method in ("IG", "GradientSHAP"):
        method_result: dict[str, Any] = {}
        for population in ("target", "neighbors"):
            arrays = []
            for mode in ("current", "fused"):
                path = output_dir / f"{method.lower()}-{mode}_{population}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    arrays.append(
                        np.asarray(archive["explanations"], dtype=np.float64)
                    )
            current, fused = arrays
            overlaps = np.sum(current * fused, axis=1, dtype=np.float64)
            method_result[population] = {
                "minimum_row_cosine": float(np.min(overlaps)),
                "mean_row_cosine": float(np.mean(overlaps)),
                "maximum_absolute_coefficient_difference": float(
                    np.max(np.abs(current - fused))
                ),
                "interpretation": (
                    "same estimator and fixed samples; differences arise only "
                    "from backend batch shape and floating-point reduction order"
                ),
            }
        comparisons["methods"][method] = method_result
    return comparisons


def main() -> int:
    args = build_parser().parse_args()
    _canonicalize_execution_modes(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    data_dir = args.data_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    device = str(resolve_device(args.device))
    data = ZCData(
        data_dir,
        input_profile=args.input_profile,
        verify_checksums=not args.skip_data_checksums,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("This benchmark accepts only the fresh zc-v3 data set.")
    spec = ExperimentSpec(
        architecture=args.architecture,
        lead_months=args.lead_months,
        train_years=10_000.0,
        seed=args.seed,
        input_profile=args.input_profile,
    )
    experiment = load_experiment(data, artifacts_dir, spec, device="cpu")
    fixed = data.fixed_supervised_split(args.lead_months)
    if not np.array_equal(experiment.fit_inputs, fixed.train_inputs):
        raise ValueError("The model artifact does not use the fixed training block.")
    if not np.array_equal(experiment.validation_inputs, fixed.validation_inputs):
        raise ValueError("The model artifact does not use the fixed validation block.")

    event_started = time.perf_counter()
    query_index, target_index, target_nino3 = _extreme_test_event(data, experiment)
    event_selection_seconds = time.perf_counter() - event_started
    candidates = np.asarray(experiment.fit_inputs, dtype=np.int64)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / _default_run_name(args)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    neighbor_path = output_dir / "robustness_neighbors.npz"
    references_path = output_dir / "gradientshap_references.npz"
    agop_dir = (
        args.agop_benchmark_dir
        if args.agop_benchmark_dir is not None
        else _default_agop_benchmark_dir(args)
    )
    agop_factor, agop_provenance = load_validated_full_agop_factor(
        agop_dir,
        data=data,
        experiment=experiment,
        fixed_training_inputs=fixed.train_inputs,
    )

    base_identity = {
        "purpose": "fresh XAI target and empirical-neighbor method timing",
        "script_sha256": sha256_file(Path(__file__)),
        "xai_source_sha256": sha256_file(
            REPOSITORY_ROOT / "src" / "zc_xai" / "xai.py"
        ),
        "data": data.provenance(),
        "processed_checksums_verified": not args.skip_data_checksums,
        "model": {
            "spec": asdict(spec),
            "artifact_id": _artifact_id(experiment.artifact_dir, artifacts_dir),
            "checkpoint_sha256": experiment.checkpoint_sha256,
            "generation_id": experiment.metrics.get("generation_id"),
            "training_source_sha256": experiment.metrics.get(
                "training_source_sha256"
            ),
        },
        "standardizer": {
            "mean_sha256": sha256_array(experiment.standardizer.mean),
            "scale_sha256": sha256_array(experiment.standardizer.scale),
            "count": experiment.standardizer.count,
        },
        "event": {
            "selection": "maximum true Nino-3 target among fixed test predictors",
            "input_index": query_index,
            "target_index": target_index,
            "target_nino3_c": target_nino3,
        },
        "neighbor_design": {
            "candidate_population": FRESH_XAI_TRAINING_POPULATION,
            "candidate_count": candidates.size,
            "candidate_indices_sha256": sha256_array(candidates),
            "distance": "RMS Euclidean distance in standardized coordinates",
            "nearest_percent": args.neighbor_percent,
            "uniform_sample_without_replacement": args.neighbor_samples,
            "seed": args.neighbor_seed,
        },
        "method_parameters": {
            "ig_steps": args.ig_steps,
            "gradient_shap_samples": args.gradient_shap_samples,
            "gradient_shap_seed": args.gradient_shap_seed,
            "gradient_batch_size": args.gradient_batch_size,
            "fused_pair_batch_size": args.fused_pair_batch_size,
            "execution_modes": list(args.execution_modes),
        },
        "method_variants": list(method_variants(args.execution_modes)),
        "agop_factor_sha256": agop_provenance["eigensystem_sha256"],
        "software": _software(device),
    }
    run_identity_sha256 = sha256_json(base_identity)
    methods: dict[str, Any] = {}
    if report_path.exists() and not args.overwrite:
        existing = load_json(report_path)
        if (
            existing.get("run_identity_sha256") != run_identity_sha256
            or existing.get("run_identity") != base_identity
        ):
            raise ValueError(
                "The existing method report has another identity. Choose another "
                "--output-dir or pass --overwrite."
            )
        if isinstance(existing.get("methods"), dict):
            methods = dict(existing["methods"])

    neighbor_identity = {
        "stage": "robustness_neighbor_selection",
        "data_metadata_sha256": data.metadata_sha256,
        "normalizer_mean_sha256": sha256_array(experiment.standardizer.mean),
        "normalizer_scale_sha256": sha256_array(experiment.standardizer.scale),
        "query_index": query_index,
        "candidate_indices_sha256": sha256_array(candidates),
        "neighbor_percent": args.neighbor_percent,
        "sample_count": args.neighbor_samples,
        "seed": args.neighbor_seed,
        "distance_batch_size": args.distance_batch_size,
        "coordinates": "training-standardized",
    }
    neighbor_arrays, neighbor_record = load_or_compute_npz(
        neighbor_path,
        stage="robustness_neighbor_selection",
        identity=neighbor_identity,
        compute=lambda: _neighbor_arrays(
            sample_empirical_neighbors(
                data,
                experiment.standardizer,
                query_index=query_index,
                candidate_indices=candidates,
                neighbor_percent=args.neighbor_percent,
                n_samples=args.neighbor_samples,
                seed=args.neighbor_seed,
                distance_batch_size=args.distance_batch_size,
            )
        ),
        overwrite=args.overwrite,
    )
    sampled_indices = np.asarray(neighbor_arrays["sampled_indices"], dtype=np.int64)
    if sampled_indices.shape != (args.neighbor_samples,):
        raise ValueError(
            "The nearest-percent population is too small for the requested sample."
        )

    reference_identity = {
        "stage": "fixed_expected_gradient_references",
        "candidate_indices_sha256": sha256_array(candidates),
        "count": args.gradient_shap_samples,
        "background_seed": args.gradient_shap_seed,
        "alpha_seed": args.gradient_shap_seed + 1,
    }
    reference_arrays, reference_record = load_or_compute_npz(
        references_path,
        stage="fixed_expected_gradient_references",
        identity=reference_identity,
        compute=lambda: _fixed_expected_gradient_references(
            candidates,
            count=args.gradient_shap_samples,
            seed=args.gradient_shap_seed,
        ),
        overwrite=args.overwrite,
    )
    background_indices = np.asarray(
        reference_arrays["background_indices"], dtype=np.int64
    )
    alphas = np.asarray(reference_arrays["alphas"], dtype=np.float32)
    if (
        background_indices.shape != (args.gradient_shap_samples,)
        or alphas.shape != (args.gradient_shap_samples,)
        or np.unique(background_indices).size != background_indices.size
        or not np.all(np.isin(background_indices, candidates))
        or np.any((alphas < 0.0) | (alphas > 1.0))
    ):
        raise ValueError("Expected-gradient reference cache is invalid.")

    timer = time.perf_counter()
    query = data.load_inputs(
        np.asarray([query_index], dtype=np.int64),
        standardizer=experiment.standardizer,
    )
    query_load_seconds = time.perf_counter() - timer
    timer = time.perf_counter()
    neighbors = data.load_inputs(
        sampled_indices,
        standardizer=experiment.standardizer,
    )
    neighbor_load_seconds = time.perf_counter() - timer
    timer = time.perf_counter()
    backgrounds = data.load_inputs(
        background_indices,
        standardizer=experiment.standardizer,
    )
    background_load_seconds = time.perf_counter() - timer
    if not (
        np.isfinite(query).all()
        and np.isfinite(neighbors).all()
        and np.isfinite(backgrounds).all()
    ):
        raise ValueError("Standardized benchmark inputs contain non-finite values.")

    warmup_started = time.perf_counter()
    input_gradients(
        experiment.model,
        query,
        batch_size=1,
        device=device,
    )
    warmup_seconds = time.perf_counter() - warmup_started
    setup = {
        "event_selection_seconds": event_selection_seconds,
        "neighbor_selection": neighbor_record,
        "expected_gradient_references": reference_record,
        "standardized_input_loading_seconds": {
            "target": query_load_seconds,
            "neighbors": neighbor_load_seconds,
            "gradient_shap_backgrounds": background_load_seconds,
        },
        "excluded_single_gradient_device_warmup_seconds": warmup_seconds,
        "event": base_identity["event"],
        "neighbors": {
            "candidate_count": int(neighbor_arrays["candidate_count"]),
            "neighborhood_count": int(neighbor_arrays["neighborhood_count"]),
            "sample_count": sampled_indices.size,
            "sampled_indices_sha256": sha256_array(sampled_indices),
            "sampled_rms_distance_min": float(
                np.min(neighbor_arrays["sampled_rms_distances"])
            ),
            "sampled_rms_distance_max": float(
                np.max(neighbor_arrays["sampled_rms_distances"])
            ),
        },
        "gradient_shap_references": {
            "count": background_indices.size,
            "background_indices_sha256": sha256_array(background_indices),
            "alphas_sha256": sha256_array(alphas),
        },
    }
    _write_progress(
        report_path,
        run_identity=base_identity,
        run_identity_sha256=run_identity_sha256,
        setup=setup,
        methods=methods,
        agop_provenance=agop_provenance,
    )

    base_method_identity = {
        "script_sha256": base_identity["script_sha256"],
        "xai_source_sha256": base_identity["xai_source_sha256"],
        "checkpoint_sha256": experiment.checkpoint_sha256,
        "data_metadata_sha256": data.metadata_sha256,
        "normalizer_mean_sha256": sha256_array(experiment.standardizer.mean),
        "normalizer_scale_sha256": sha256_array(experiment.standardizer.scale),
        "resolved_device": device,
        "gradient_batch_size": args.gradient_batch_size,
    }
    for variant in method_variants(args.execution_modes):
        method, execution_mode = _variant_parts(variant)
        LOGGER.info("Benchmarking %s target and neighbors", variant)
        explainer = _explainer(
            method,
            execution_mode=execution_mode,
            experiment=experiment,
            args=args,
            backgrounds=backgrounds,
            alphas=alphas,
            agop_factor=agop_factor,
            device=device,
        )
        parameters = _method_parameters(
            method,
            execution_mode=execution_mode,
            args=args,
            background_indices=background_indices,
            alphas=alphas,
            agop_provenance=agop_provenance,
        )
        method_record = dict(methods.get(variant, {}))
        for population, inputs, indices in (
            (
                "target",
                query,
                np.asarray([query_index], dtype=np.int64),
            ),
            ("neighbors", neighbors, sampled_indices),
        ):
            cache_path = output_dir / f"{variant.lower()}_{population}.npz"
            identity = {
                **base_method_identity,
                "method": method,
                "variant": variant,
                "execution_mode": execution_mode,
                "parameters": parameters,
                "population": population,
                "input_indices_sha256": sha256_array(indices),
                "standardized_inputs_sha256": sha256_array(inputs),
            }
            arrays, record = load_or_compute_npz(
                cache_path,
                stage=f"{variant}_{population}_explanations",
                identity=identity,
                compute=lambda explainer=explainer, inputs=inputs: (
                    _explanation_arrays(explainer, inputs)
                ),
                overwrite=args.overwrite,
            )
            explanations = np.asarray(arrays["explanations"])
            if explanations.shape != (inputs.shape[0], data.n_features):
                raise ValueError(
                    f"Cached {variant} {population} shape is invalid."
                )
            method_record[population] = {
                **record,
                "workload": method_workload(
                    method,
                    execution_mode=execution_mode,
                    query_count=inputs.shape[0],
                    gradient_batch_size=args.gradient_batch_size,
                    fused_pair_batch_size=args.fused_pair_batch_size,
                    ig_steps=args.ig_steps,
                    gradient_shap_samples=args.gradient_shap_samples,
                ),
            }
            methods[variant] = method_record
            _write_progress(
                report_path,
                run_identity=base_identity,
                run_identity_sha256=run_identity_sha256,
                setup=setup,
                methods=methods,
                agop_provenance=agop_provenance,
            )
        method_record["parameters"] = parameters
        method_record["total_explanation_seconds"] = sum(
            float(method_record[population]["compute_seconds"])
            for population in ("target", "neighbors")
        )
        method_record["total_gradient_rows"] = sum(
            int(method_record[population]["workload"]["gradient_rows"])
            for population in ("target", "neighbors")
        )
        method_record["total_gradient_batches"] = sum(
            int(method_record[population]["workload"]["gradient_batches"])
            for population in ("target", "neighbors")
        )
        methods[variant] = method_record
        _write_progress(
            report_path,
            run_identity=base_identity,
            run_identity_sha256=run_identity_sha256,
            setup=setup,
            methods=methods,
            agop_provenance=agop_provenance,
        )
    setup["implementation_comparisons"] = _implementation_comparisons(
        output_dir,
        args.execution_modes,
    )
    _write_progress(
        report_path,
        run_identity=base_identity,
        run_identity_sha256=run_identity_sha256,
        setup=setup,
        methods=methods,
        agop_provenance=agop_provenance,
    )
    experiment.model.cpu()
    LOGGER.info("Fresh XAI method timing report complete: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
