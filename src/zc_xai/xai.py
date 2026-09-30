"""XAI methods and quantitative scores used for the revised Table I."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.linalg import blas, eigh

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - dependency guidance
    raise ModuleNotFoundError(
        "PyTorch is required for XAI experiments. Install the project with "
        "`python -m pip install -e .`."
    ) from error

from .data import Standardizer, ZCData
from .io import (
    atomic_output_path,
    load_json,
    sha256_array,
    sha256_file,
    sha256_json,
    write_json,
)
from .training import resolve_device

LOGGER = logging.getLogger(__name__)
XAI_CACHE_SCHEMA_VERSION = 3
EXACT_DENSE_AGOP_CACHE_SCHEMA_VERSION = 2
DEFAULT_ROBUSTNESS_NEIGHBOR_PERCENT = 1.0
DEFAULT_AGOP_POWER_ITERATIONS = 4


def unit_rows(values: np.ndarray, *, label: str) -> np.ndarray:
    """Return row-wise L2-normalized vectors, rejecting undefined explanations."""

    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError(f"{label} explanations must be finite two-dimensional rows.")
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        positions = np.flatnonzero(norms <= np.finfo(np.float64).eps).tolist()
        raise ValueError(
            f"{label} produced zero-norm explanations at rows {positions}."
        )
    return rows / norms[:, None]


def input_gradients(
    model: nn.Module,
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Evaluate gradients of scalar forecasts with respect to standardized inputs."""

    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    model.eval()
    values = np.asarray(inputs, dtype=np.float32)
    output = np.empty_like(values)
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        tensor = torch.from_numpy(values[start:stop]).to(resolved_device)
        tensor.requires_grad_(True)
        with torch.enable_grad():
            forecasts = model(tensor)
            # Supplying the output cotangent explicitly is mathematically
            # equivalent to differentiating ``forecasts.sum()``.  It also
            # avoids an Apple MPS reduction bug that can silently retain only
            # every sixteenth row of a batched input gradient.
            gradients = torch.autograd.grad(
                forecasts,
                tensor,
                grad_outputs=torch.ones_like(forecasts),
            )[0]
        output[start:stop] = gradients.detach().cpu().numpy()
    return output


class Explainer(Protocol):
    name: str

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        """Return one unit-norm flattened explanation per standardized input."""


@dataclass
class GradientExplainer:
    model: nn.Module
    batch_size: int
    device: str
    name: str = "GRAD"

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        gradients = input_gradients(
            self.model,
            inputs,
            batch_size=self.batch_size,
            device=self.device,
        )
        return unit_rows(gradients.reshape(gradients.shape[0], -1), label=self.name)


@dataclass
class IntegratedGradientsExplainer:
    model: nn.Module
    n_steps: int
    gradient_batch_size: int
    device: str
    name: str = "IG"

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        if self.n_steps <= 0:
            raise ValueError("Integrated Gradients requires at least one step.")
        values = np.asarray(inputs, dtype=np.float32)
        alphas = np.linspace(
            1.0 / self.n_steps,
            1.0,
            num=self.n_steps,
            dtype=np.float32,
        )
        explanations = np.empty((values.shape[0], values[0].size), dtype=np.float64)
        for position, query in enumerate(values):
            path = alphas.reshape((-1,) + (1,) * query.ndim) * query[None, ...]
            gradients = input_gradients(
                self.model,
                path,
                batch_size=self.gradient_batch_size,
                device=self.device,
            )
            attribution = query * gradients.mean(axis=0, dtype=np.float64)
            explanations[position] = attribution.reshape(-1)
        return unit_rows(explanations, label=self.name)


@dataclass
class GradientShapExplainer:
    """Expected-gradients implementation with fixed backgrounds and alphas."""

    model: nn.Module
    backgrounds: np.ndarray
    alphas: np.ndarray
    gradient_batch_size: int
    device: str
    name: str = "GradientSHAP"

    def __post_init__(self) -> None:
        self.backgrounds = np.asarray(self.backgrounds, dtype=np.float32)
        self.alphas = np.asarray(self.alphas, dtype=np.float32)
        if self.backgrounds.shape[0] != self.alphas.size:
            raise ValueError(
                "GradientSHAP needs one interpolation alpha per background."
            )
        if self.alphas.ndim != 1 or np.any((self.alphas < 0) | (self.alphas > 1)):
            raise ValueError("GradientSHAP interpolation alphas must lie in [0, 1].")

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        explanations = np.empty((values.shape[0], values[0].size), dtype=np.float64)
        alpha_shape = (-1,) + (1,) * (values.ndim - 1)
        alphas = self.alphas.reshape(alpha_shape)
        for position, query in enumerate(values):
            differences = query[None, ...] - self.backgrounds
            interpolated = self.backgrounds + alphas * differences
            gradients = input_gradients(
                self.model,
                interpolated,
                batch_size=self.gradient_batch_size,
                device=self.device,
            )
            attribution = np.multiply(differences, gradients).mean(
                axis=0,
                dtype=np.float64,
            )
            explanations[position] = attribution.reshape(-1)
        return unit_rows(explanations, label=self.name)


@dataclass(frozen=True)
class AgopFactor:
    """Low-rank representation of the positive-semidefinite AGOP square root."""

    basis: np.ndarray
    root_eigenvalues: np.ndarray
    center: np.ndarray
    reference_indices: np.ndarray
    approximation_rank: int
    solver_metadata: dict[str, object] | None = None


@dataclass
class AgopExplainer:
    factor: AgopFactor
    name: str = "AGOP"

    def explain(self, inputs: np.ndarray) -> np.ndarray:
        flattened = np.asarray(inputs, dtype=np.float64).reshape(inputs.shape[0], -1)
        basis = np.asarray(self.factor.basis, dtype=np.float64)
        weights = np.asarray(self.factor.root_eigenvalues, dtype=np.float64)
        center = np.asarray(self.factor.center, dtype=np.float64)
        transformed = (((flattened - center) @ basis) * weights[None, :]) @ basis.T
        return unit_rows(transformed, label=self.name)


def build_agop_factor(
    model: nn.Module,
    data: ZCData,
    standardizer: Standardizer,
    reference_indices: np.ndarray,
    *,
    rank: int,
    gradient_batch_size: int,
    device: str,
    gradient_matrix_path: Path | None = None,
    overwrite_gradient_matrix: bool = False,
    svd_seed: int = 42,
    power_iterations: int = DEFAULT_AGOP_POWER_ITERATIONS,
    gradient_cache_identity: dict[str, object] | None = None,
) -> AgopFactor:
    """Build M^(1/2) from an optional low-rank SVD of reference gradients."""

    references = np.asarray(reference_indices, dtype=np.int64)
    if references.ndim != 1 or references.size == 0:
        raise ValueError("AGOP requires at least one reference input.")
    if gradient_batch_size <= 0:
        raise ValueError("gradient_batch_size must be positive")
    if power_iterations < 0:
        raise ValueError("power_iterations must be nonnegative")
    gradient_shape = (references.size, data.n_features)
    gradient_matrix: np.ndarray
    if gradient_matrix_path is not None:
        path = gradient_matrix_path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = path.with_name(path.name + ".json")
        cache_identity: dict[str, object] = {
            "schema_version": XAI_CACHE_SCHEMA_VERSION,
            "data_metadata_sha256": data.metadata_sha256,
            "reference_indices_sha256": sha256_array(references),
            "standardizer_mean_sha256": sha256_array(standardizer.mean),
            "standardizer_scale_sha256": sha256_array(standardizer.scale),
            "standardizer_count": standardizer.count,
            "gradient_shape": list(gradient_shape),
            "gradient_dtype": np.dtype(np.float32).str,
        }
        if gradient_cache_identity is not None:
            cache_identity.update(gradient_cache_identity)
        identity_sha256 = sha256_json(cache_identity)
        if path.exists() and not overwrite_gradient_matrix:
            if not manifest_path.is_file():
                raise ValueError(
                    f"AGOP gradient cache has no completion manifest: {path}. "
                    "Rebuild the AGOP cache explicitly."
                )
            manifest = load_json(manifest_path)
            if manifest.get("identity_sha256") != identity_sha256:
                raise ValueError(
                    f"AGOP gradient cache identity does not match: {path}. "
                    "Rebuild the AGOP cache explicitly."
                )
            if manifest.get("file_sha256") != sha256_file(path):
                raise ValueError(
                    f"AGOP gradient cache is incomplete or corrupted: {path}. "
                    "Rebuild the AGOP cache explicitly."
                )
            loaded = np.load(path, mmap_mode="c", allow_pickle=False)
            if loaded.shape != gradient_shape or loaded.dtype != np.float32:
                raise ValueError(
                    f"Cached AGOP gradient matrix has wrong schema: {path}"
                )
            gradient_matrix = loaded
        else:
            with atomic_output_path(path, overwrite=True) as temporary_path:
                writable_matrix = np.lib.format.open_memmap(
                    temporary_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=gradient_shape,
                )
                _fill_gradient_matrix(
                    writable_matrix,
                    model,
                    data,
                    standardizer,
                    references,
                    gradient_batch_size=gradient_batch_size,
                    device=device,
                )
                writable_matrix.flush()
                del writable_matrix
            write_json(
                manifest_path,
                {
                    "schema_version": XAI_CACHE_SCHEMA_VERSION,
                    "identity": cache_identity,
                    "identity_sha256": identity_sha256,
                    "file": path.name,
                    "file_sha256": sha256_file(path),
                },
                overwrite=True,
            )
            gradient_matrix = np.load(path, mmap_mode="c", allow_pickle=False)
    else:
        gradient_matrix = np.empty(gradient_shape, dtype=np.float32)
        _fill_gradient_matrix(
            gradient_matrix,
            model,
            data,
            standardizer,
            references,
            gradient_batch_size=gradient_batch_size,
            device=device,
        )

    maximum_rank = min(gradient_shape)
    if rank < 0:
        raise ValueError("AGOP rank cannot be negative.")
    full_decomposition = rank == 0 or rank >= maximum_rank
    effective_rank = maximum_rank if full_decomposition else rank
    LOGGER.info(
        "Computing AGOP factor from %s gradients at rank %s",
        f"{references.size:,}",
        effective_rank,
    )
    torch.manual_seed(svd_seed)
    matrix = torch.from_numpy(np.asarray(gradient_matrix))
    if full_decomposition:
        _, singular_values, right_vectors_t = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
        basis = right_vectors_t.T
    else:
        _, singular_values, basis = torch.pca_lowrank(
            matrix,
            q=effective_rank,
            center=False,
            niter=power_iterations,
        )
    root_eigenvalues = singular_values / math.sqrt(references.size)
    return AgopFactor(
        basis=basis.cpu().numpy().astype(np.float32, copy=False),
        root_eigenvalues=root_eigenvalues.cpu().numpy().astype(np.float32, copy=False),
        # The zero vector is the empirical optimization-fit mean in standardized
        # coordinates and is used as the common XAI baseline.
        center=np.zeros(data.n_features, dtype=np.float32),
        reference_indices=references,
        approximation_rank=effective_rank,
    )


def build_exact_dense_agop_factor(
    model: nn.Module,
    data: ZCData,
    standardizer: Standardizer,
    reference_indices: np.ndarray,
    *,
    gradient_batch_size: int,
    device: str,
    matrix_cache_path: Path | None = None,
    overwrite_matrix_cache: bool = False,
    cache_identity: dict[str, object] | None = None,
) -> AgopFactor:
    """Build the exact empirical AGOP square root through dense accumulation.

    Each standardized gradient batch contributes ``G_batch.T @ G_batch`` to a
    float64 matrix. Individual gradient rows are never retained after their
    batch has been accumulated. The final symmetric matrix is diagonalized in
    float64 with the full eigenbasis retained.
    """

    references = _validated_dense_agop_inputs(
        data,
        standardizer,
        reference_indices,
        gradient_batch_size=gradient_batch_size,
    )
    identity = _exact_dense_agop_identity(
        model,
        data,
        standardizer,
        references,
        cache_identity=cache_identity,
    )
    identity_sha256 = sha256_json(identity)
    if matrix_cache_path is None:
        matrix = _accumulate_dense_agop(
            model,
            data,
            standardizer,
            references,
            gradient_batch_size=gradient_batch_size,
            device=device,
        )
        matrix_cache_metadata: dict[str, object] = {
            "used": False,
            "identity_sha256": identity_sha256,
            "accumulation_gradient_batch_size": int(gradient_batch_size),
        }
    else:
        matrix, matrix_cache_metadata = _load_or_build_dense_agop_matrix(
            model,
            data,
            standardizer,
            references,
            gradient_batch_size=gradient_batch_size,
            device=device,
            path=matrix_cache_path.expanduser().resolve(),
            overwrite=overwrite_matrix_cache,
            identity=identity,
            identity_sha256=identity_sha256,
        )

    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Dense AGOP cache or accumulation is not exactly symmetric.")
    eigenvalues, basis = eigh(
        matrix,
        lower=False,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    eigenvalues, clipping = _clip_numerical_negative_eigenvalues(eigenvalues)
    order = np.arange(eigenvalues.size - 1, -1, -1)
    eigenvalues = eigenvalues[order]
    basis = basis[:, order]
    root_eigenvalues = np.sqrt(eigenvalues)
    metadata: dict[str, object] = {
        "method": "exact_dense_empirical_agop",
        "reference_count": int(references.size),
        "feature_count": int(data.n_features),
        "requested_gradient_batch_size": int(gradient_batch_size),
        "accumulation_dtype": np.dtype(np.float64).str,
        "eigendecomposition_dtype": np.dtype(np.float64).str,
        "accumulation_kernel": "scipy.linalg.blas.dsyrk upper triangle",
        "eigendecomposition_driver": "scipy.linalg.eigh driver=evd",
        "full_basis": True,
        "matrix_definition": "sum_i grad_i grad_i^T / reference_count",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "matrix_cache": matrix_cache_metadata,
        "eigenvalue_clipping": clipping,
        "largest_eigenvalue": float(eigenvalues[0]),
        "trace": float(eigenvalues.sum()),
    }
    return AgopFactor(
        basis=np.asarray(basis, dtype=np.float64),
        root_eigenvalues=np.asarray(root_eigenvalues, dtype=np.float64),
        center=np.zeros(data.n_features, dtype=np.float64),
        reference_indices=references,
        approximation_rank=data.n_features,
        solver_metadata=metadata,
    )


def _validated_dense_agop_inputs(
    data: ZCData,
    standardizer: Standardizer,
    reference_indices: np.ndarray,
    *,
    gradient_batch_size: int,
) -> np.ndarray:
    raw = np.asarray(reference_indices)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("Exact dense AGOP requires nonempty one-dimensional indices.")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Exact dense AGOP reference indices must be integers.")
    references = np.asarray(raw, dtype=np.int64)
    if np.unique(references).size != references.size:
        raise ValueError("Exact dense AGOP reference indices must be unique.")
    if np.any(references < 0) or np.any(references >= data.n_time_steps):
        raise ValueError("Exact dense AGOP reference indices are outside the data set.")
    # The empirical matrix is order-invariant mathematically. Canonical sorting
    # also makes the floating-point accumulation and cache identity reproducible.
    references = np.sort(references)
    if gradient_batch_size <= 0:
        raise ValueError("gradient_batch_size must be positive")
    if data.n_features <= 0:
        raise ValueError("Exact dense AGOP requires at least one input feature.")
    if standardizer.mean.shape != data.input_shape:
        raise ValueError("Standardizer mean does not match the model input shape.")
    if standardizer.scale.shape != data.input_shape:
        raise ValueError("Standardizer scale does not match the model input shape.")
    if not np.isfinite(standardizer.mean).all():
        raise ValueError("Standardizer mean contains non-finite values.")
    if not np.isfinite(standardizer.scale).all() or np.any(standardizer.scale <= 0):
        raise ValueError("Standardizer scale must be finite and strictly positive.")
    return references


def _model_state_identity(model: nn.Module) -> dict[str, str]:
    identity = {}
    for name, value in sorted(model.state_dict().items()):
        identity[name] = sha256_array(value.detach().cpu().numpy())
    return identity


def _exact_dense_agop_identity(
    model: nn.Module,
    data: ZCData,
    standardizer: Standardizer,
    references: np.ndarray,
    *,
    cache_identity: dict[str, object] | None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": EXACT_DENSE_AGOP_CACHE_SCHEMA_VERSION,
        "method": "exact_dense_empirical_agop",
        "data_metadata_sha256": data.metadata_sha256,
        "input_profile": getattr(data, "input_profile", None),
        "input_shape": list(data.input_shape),
        "reference_indices_sha256": sha256_array(references),
        "reference_count": int(references.size),
        "standardizer_mean_sha256": sha256_array(standardizer.mean),
        "standardizer_scale_sha256": sha256_array(standardizer.scale),
        "standardizer_count": int(standardizer.count),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_state": _model_state_identity(model),
        "matrix_shape": [data.n_features, data.n_features],
        "matrix_dtype": np.dtype(np.float64).str,
    }
    if cache_identity is not None:
        identity["caller_identity"] = cache_identity
    return identity


def _accumulate_dense_agop(
    model: nn.Module,
    data: ZCData,
    standardizer: Standardizer,
    references: np.ndarray,
    *,
    gradient_batch_size: int,
    device: str,
) -> np.ndarray:
    # Fortran-order output lets the BLAS wrapper update the destination without
    # a hidden layout conversion. Only the upper triangle is accumulated.
    matrix = np.zeros(
        (data.n_features, data.n_features),
        dtype=np.float64,
        order="F",
    )
    for start in range(0, references.size, gradient_batch_size):
        stop = min(start + gradient_batch_size, references.size)
        inputs = data.load_inputs(
            references[start:stop],
            standardizer=standardizer,
        )
        gradients = input_gradients(
            model,
            inputs,
            batch_size=gradient_batch_size,
            device=device,
        )
        flattened = np.asfortranarray(
            np.asarray(gradients, dtype=np.float64).reshape(
                stop - start, data.n_features
            )
        )
        if not np.isfinite(flattened).all():
            raise ValueError("Exact dense AGOP gradients contain non-finite values.")
        matrix = blas.dsyrk(
            alpha=1.0,
            a=flattened,
            beta=1.0,
            c=matrix,
            trans=1,
            lower=0,
            overwrite_c=1,
        )
        LOGGER.info(
            "Dense AGOP gradients: %s / %s",
            f"{stop:,}",
            f"{references.size:,}",
        )
    matrix /= references.size
    # DSYRK intentionally leaves the lower triangle untouched. Mirror the
    # accumulated upper triangle once to publish a conventional symmetric M.
    upper = np.triu(matrix)
    return upper + np.triu(upper, k=1).T


def _load_or_build_dense_agop_matrix(
    model: nn.Module,
    data: ZCData,
    standardizer: Standardizer,
    references: np.ndarray,
    *,
    gradient_batch_size: int,
    device: str,
    path: Path,
    overwrite: bool,
    identity: dict[str, object],
    identity_sha256: str,
) -> tuple[np.ndarray, dict[str, object]]:
    manifest_path = path.with_name(path.name + ".json")
    if path.exists() and not overwrite:
        if not manifest_path.is_file():
            raise ValueError(f"Dense AGOP cache has no completion manifest: {path}")
        manifest = load_json(manifest_path)
        if manifest.get("identity_sha256") != identity_sha256:
            raise ValueError(f"Dense AGOP cache identity does not match: {path}")
        if manifest.get("file_sha256") != sha256_file(path):
            raise ValueError(f"Dense AGOP cache is incomplete or corrupted: {path}")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (data.n_features, data.n_features)
        if matrix.shape != expected_shape or matrix.dtype != np.float64:
            raise ValueError(f"Dense AGOP cache has the wrong schema: {path}")
        build_metadata = manifest.get("build")
        accumulation_batch_size = (
            build_metadata.get("gradient_batch_size")
            if isinstance(build_metadata, dict)
            else None
        )
        return matrix, {
            "used": True,
            "loaded": True,
            "path": str(path),
            "file_sha256": manifest["file_sha256"],
            "identity_sha256": identity_sha256,
            "accumulation_gradient_batch_size": accumulation_batch_size,
        }
    matrix = _accumulate_dense_agop(
        model,
        data,
        standardizer,
        references,
        gradient_batch_size=gradient_batch_size,
        device=device,
    )
    with (
        atomic_output_path(path, overwrite=True) as temporary_path,
        temporary_path.open("wb") as stream,
    ):
        np.save(stream, matrix, allow_pickle=False)
    file_sha256 = sha256_file(path)
    write_json(
        manifest_path,
        {
            "schema_version": EXACT_DENSE_AGOP_CACHE_SCHEMA_VERSION,
            "identity": identity,
            "identity_sha256": identity_sha256,
            "file": path.name,
            "file_sha256": file_sha256,
            "build": {"gradient_batch_size": int(gradient_batch_size)},
        },
        overwrite=True,
    )
    return matrix, {
        "used": True,
        "loaded": False,
        "path": str(path),
        "file_sha256": file_sha256,
        "identity_sha256": identity_sha256,
        "accumulation_gradient_batch_size": int(gradient_batch_size),
    }


def _clip_numerical_negative_eigenvalues(
    eigenvalues: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Dense AGOP eigensolver returned invalid eigenvalues.")
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    tolerance = 64.0 * np.finfo(np.float64).eps * values.size * scale
    minimum = float(values.min())
    if minimum < -tolerance:
        raise ValueError(
            "Dense empirical AGOP has a materially negative eigenvalue: "
            f"{minimum:.6g} below tolerance {-tolerance:.6g}."
        )
    negative_count = int(np.count_nonzero(values < 0.0))
    clipped = values.copy()
    clipped[clipped < 0.0] = 0.0
    return clipped, {
        "negative_count": negative_count,
        "minimum_before_clipping": minimum,
        "absolute_tolerance": float(tolerance),
        "policy": "clip only eigenvalues within roundoff tolerance; reject others",
    }


def _fill_gradient_matrix(
    destination: np.ndarray,
    model: nn.Module,
    data: ZCData,
    standardizer: Standardizer,
    reference_indices: np.ndarray,
    *,
    gradient_batch_size: int,
    device: str,
) -> None:
    for start in range(0, reference_indices.size, gradient_batch_size):
        stop = min(start + gradient_batch_size, reference_indices.size)
        LOGGER.info(
            "AGOP gradients: %s / %s",
            f"{stop:,}",
            f"{reference_indices.size:,}",
        )
        inputs = data.load_inputs(
            reference_indices[start:stop],
            standardizer=standardizer,
        )
        gradients = input_gradients(
            model,
            inputs,
            batch_size=gradient_batch_size,
            device=device,
        )
        if not np.isfinite(gradients).all():
            raise ValueError("AGOP gradients contain non-finite values.")
        destination[start:stop] = gradients.reshape(stop - start, -1)


@dataclass(frozen=True)
class NeighborSample:
    query_index: int
    neighbor_percent: float
    candidate_count: int
    neighborhood_count: int
    sampled_indices: np.ndarray
    sampled_rms_distances: np.ndarray
    neighborhood_indices: np.ndarray
    neighborhood_rms_distances: np.ndarray
    seed: int


def sample_empirical_neighbors(
    data: ZCData,
    standardizer: Standardizer,
    *,
    query_index: int,
    candidate_indices: np.ndarray,
    neighbor_percent: float = DEFAULT_ROBUSTNESS_NEIGHBOR_PERCENT,
    n_samples: int = 256,
    seed: int = 42,
    distance_batch_size: int = 512,
) -> NeighborSample:
    """Uniformly sample actual states from the nearest X% in standardized space."""

    if not 0.0 < neighbor_percent <= 100.0:
        raise ValueError("neighbor_percent must lie in (0, 100]")
    if n_samples <= 0 or distance_batch_size <= 0:
        raise ValueError("n_samples and distance_batch_size must be positive")
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("candidate_indices must be a nonempty one-dimensional array")
    query = data.load_inputs(
        np.asarray([query_index], dtype=np.int64),
        standardizer=standardizer,
    )[0]
    squared_rms = np.empty(candidates.size, dtype=np.float64)
    for start in range(0, candidates.size, distance_batch_size):
        stop = min(start + distance_batch_size, candidates.size)
        values = data.load_inputs(
            candidates[start:stop],
            standardizer=standardizer,
        )
        differences = (values - query).reshape(stop - start, -1)
        squared_rms[start:stop] = np.square(
            differences, dtype=np.float64
        ).mean(axis=1)
        LOGGER.info(
            "Neighbor distances: %s / %s",
            f"{stop:,}",
            f"{candidates.size:,}",
        )

    eligible = candidates != query_index
    eligible_candidates = candidates[eligible]
    eligible_distances = np.sqrt(squared_rms[eligible])
    if eligible_candidates.size == 0:
        raise ValueError("No neighbor candidates remain after excluding the query.")
    neighborhood_count = max(
        1,
        int(math.ceil(neighbor_percent * eligible_candidates.size / 100.0)),
    )
    order = np.lexsort((eligible_candidates, eligible_distances))
    neighborhood_positions = order[:neighborhood_count]
    neighborhood_indices = eligible_candidates[neighborhood_positions]
    neighborhood_distances = eligible_distances[neighborhood_positions]
    sample_count = min(n_samples, neighborhood_count)
    rng = np.random.default_rng(seed)
    sampled_positions = rng.choice(
        neighborhood_count,
        size=sample_count,
        replace=False,
    )
    return NeighborSample(
        query_index=int(query_index),
        neighbor_percent=float(neighbor_percent),
        candidate_count=int(eligible_candidates.size),
        neighborhood_count=int(neighborhood_count),
        sampled_indices=neighborhood_indices[sampled_positions],
        sampled_rms_distances=neighborhood_distances[sampled_positions],
        neighborhood_indices=neighborhood_indices,
        neighborhood_rms_distances=neighborhood_distances,
        seed=int(seed),
    )


def sensitivity_score(gradient: np.ndarray, explanation: np.ndarray) -> float:
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    explanation = np.asarray(explanation, dtype=np.float64).reshape(-1)
    if not np.isclose(
        np.linalg.norm(explanation),
        1.0,
        rtol=1.0e-6,
        atol=1.0e-8,
    ):
        raise ValueError("Sensitivity requires a unit-norm explanation.")
    denominator = np.linalg.norm(gradient)
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Sensitivity is undefined for a zero input gradient.")
    value = abs(float(np.dot(gradient, explanation))) / denominator
    return float(np.clip(value, 0.0, 1.0))


def attribution_score(
    model: nn.Module,
    standardized_input: np.ndarray,
    explanation: np.ndarray,
    *,
    device: str,
) -> tuple[float, float, float]:
    """Return the attribution score, raw ratio R, and anomaly projection alpha."""

    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    model.eval()
    query = np.asarray(standardized_input, dtype=np.float32)
    direction = np.asarray(explanation, dtype=np.float32).reshape(-1)
    if not np.isclose(
        np.linalg.norm(direction),
        1.0,
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise ValueError("Attribution requires a unit-norm explanation.")
    alpha = float(np.dot(query.reshape(-1), direction))
    perturbed = (alpha * direction).reshape(query.shape)
    stacked = np.stack((np.zeros_like(query), query, perturbed), axis=0)
    with torch.inference_mode():
        forecasts = model(torch.from_numpy(stacked).to(resolved_device)).cpu().numpy()
    model.cpu()
    denominator = float(forecasts[1] - forecasts[0])
    if abs(denominator) <= np.finfo(np.float32).eps:
        raise ValueError("Attribution is undefined because f(x) equals f(baseline).")
    ratio = float((forecasts[2] - forecasts[0]) / denominator)
    score = ratio if -1.0 <= ratio <= 1.0 else 1.0 / ratio
    return float(np.clip(score, -1.0, 1.0)), ratio, alpha


def coherence_score(
    explanation: np.ndarray, input_shape: tuple[int, int, int]
) -> float:
    fields = np.asarray(explanation, dtype=np.float64).reshape(input_shape)
    horizontal = np.multiply(fields[:, :, :-1], fields[:, :, 1:]).sum()
    vertical = np.multiply(fields[:, :-1, :], fields[:, 1:, :]).sum()
    numerator = 2.0 * float(horizontal + vertical)
    degree = np.full(input_shape[1:], 4.0, dtype=np.float64)
    degree[0, :] -= 1.0
    degree[-1, :] -= 1.0
    degree[:, 0] -= 1.0
    degree[:, -1] -= 1.0
    denominator = float(np.multiply(np.square(fields), degree[None, :, :]).sum())
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Coherence is undefined for this explanation.")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


@dataclass(frozen=True)
class RobustnessSummary:
    score: float
    sample_standard_deviation: float
    finite_population_standard_error: float
    overlaps: np.ndarray


def robustness_score(
    target_explanation: np.ndarray,
    neighbor_explanations: np.ndarray,
    *,
    neighborhood_count: int,
) -> RobustnessSummary:
    target = unit_rows(target_explanation, label="target")[0]
    neighbors = unit_rows(neighbor_explanations, label="neighbor")
    overlaps = np.clip(neighbors @ target, -1.0, 1.0)
    sample_count = overlaps.size
    if sample_count > neighborhood_count:
        raise ValueError("Robustness sample exceeds the neighborhood population.")
    if sample_count > 1:
        sample_sd = float(np.std(overlaps, ddof=1))
        # For simple random sampling without replacement,
        # Var(sample mean) = (1 - n/N) S^2 / n.  The Bessel-corrected sample
        # variance (ddof=1) is unbiased for the finite-population S^2.
        standard_error = (
            math.sqrt(
                max(0.0, neighborhood_count - sample_count) / neighborhood_count
            )
            * sample_sd
            / math.sqrt(sample_count)
        )
    elif neighborhood_count == 1:
        sample_sd = 0.0
        standard_error = 0.0
    else:
        sample_sd = float("nan")
        standard_error = float("nan")
    return RobustnessSummary(
        score=float(overlaps.mean()),
        sample_standard_deviation=sample_sd,
        finite_population_standard_error=float(standard_error),
        overlaps=overlaps,
    )
