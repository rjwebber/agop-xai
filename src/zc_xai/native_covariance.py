"""Exact empirical covariance operators for native ZC states.

The native Zebiak--Cane state has many more coordinates than a practical
sample covariance has samples.  Materializing its square covariance is both
unnecessary and, for the fresh 28,591-coordinate independent state, wasteful.
This module retains the centered sample matrix as a rectangular factor.

For samples ``X`` with row mean ``mu`` and ``n > 1``, define

``F = (X - mu).T / sqrt(n - 1)``.

Then the empirical covariance is exactly ``C = F @ F.T``.  Consequently,

``min ||u||^2 subject to F @ u = a``

equals ``a.T @ pinv(C) @ a`` for every ``a`` in ``range(C)``.  The classes
below expose ``F`` and ``F.T`` without constructing either ``F`` or ``C``;
the source sample arrays remain memory mapped.  A completed covariance may
also be loaded as a read-only dense memory map when repeated covariance
products justify compiling the square matrix once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import scipy.sparse.linalg

from .io import load_json, sha256_array, sha256_file

NATIVE_COVARIANCE_SCHEMA_VERSION = 1
NATIVE_COVARIANCE_STATUS = "complete_exact_centered_sample_covariance_factors"
DENSE_NATIVE_COVARIANCE_SCHEMA_VERSION = 1
DENSE_NATIVE_COVARIANCE_STATUS = "complete_dense_native_covariance"
DENSE_NATIVE_COVARIANCE_DTYPE = np.dtype("<f8")


class CovarianceFactor(Protocol):
    """Matrix-free covariance square-root contract."""

    @property
    def state_size(self) -> int: ...

    @property
    def coefficient_size(self) -> int: ...

    def apply(self, coefficients: np.ndarray) -> np.ndarray: ...

    def transpose(self, native_covector: np.ndarray) -> np.ndarray: ...

    def project_coefficients(self, coefficients: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MinimumActionResult:
    """Numerical minimum-coefficient-norm representation of an increment."""

    coefficients: np.ndarray
    reconstructed_increment: np.ndarray
    squared_action: float
    residual_l2: float
    relative_residual: float
    iterations: int
    stop_code: int
    operator_norm_estimate: float
    condition_estimate: float


@dataclass(frozen=True)
class CenteredSampleCovarianceFactor:
    """Implicit ``(X - mean(X)).T / sqrt(n - 1)`` for one annual phase."""

    samples: np.ndarray
    mean: np.ndarray
    phase_offset: int
    source_path: Path | None = None
    _centered_float64_samples: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        samples = np.asanyarray(self.samples)
        mean = np.asarray(self.mean, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] == 0:
            raise ValueError("samples must have shape (n >= 2, d >= 1)")
        if mean.shape != (samples.shape[1],):
            raise ValueError("mean does not match the native-state dimension")
        if not np.issubdtype(samples.dtype, np.floating):
            raise TypeError("native samples must have a floating dtype")
        if not np.isfinite(mean).all():
            raise ValueError("native sample mean contains nonfinite values")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "phase_offset", int(self.phase_offset))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def state_size(self) -> int:
        return int(self.samples.shape[1])

    @property
    def estimated_centered_cache_bytes(self) -> int:
        """Bytes required by an exact contiguous float64 centered cache."""

        return int(self.sample_count * self.state_size * np.dtype(np.float64).itemsize)

    @property
    def centered_cache_bytes(self) -> int:
        """Bytes currently held by the optional exact centered cache."""

        cached = self._centered_float64_samples
        return 0 if cached is None else int(cached.nbytes)

    def with_contiguous_float64_cache(
        self,
        *,
        maximum_bytes: int,
    ) -> CenteredSampleCovarianceFactor:
        """Return an equivalent factor with an exact centered sample cache.

        This is an opt-in memory/time tradeoff, not an approximation.  A byte
        budget is mandatory so callers cannot accidentally materialize a very
        large native-state cache.  The source array and provenance path remain
        attached to the returned factor.
        """

        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
            raise TypeError("maximum_bytes must be an integer")
        if maximum_bytes < 0:
            raise ValueError("maximum_bytes must be nonnegative")
        if self._centered_float64_samples is not None:
            return self
        required = self.estimated_centered_cache_bytes
        if required > maximum_bytes:
            raise MemoryError(
                "Exact centered covariance cache requires "
                f"{required} bytes, exceeding the {maximum_bytes}-byte budget"
            )
        cached = np.array(self.samples, dtype=np.float64, order="C", copy=True)
        cached -= self.mean
        cached.setflags(write=False)
        result = CenteredSampleCovarianceFactor(
            samples=self.samples,
            mean=self.mean,
            phase_offset=self.phase_offset,
            source_path=self.source_path,
        )
        object.__setattr__(result, "_centered_float64_samples", cached)
        return result

    @property
    def coefficient_size(self) -> int:
        return self.sample_count

    @property
    def normalization(self) -> float:
        return math.sqrt(self.sample_count - 1)

    def project_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        """Remove the exact all-ones null direction introduced by centering."""

        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.coefficient_size,):
            raise ValueError(
                f"coefficients must have shape ({self.coefficient_size},)"
            )
        if not np.isfinite(values).all():
            raise ValueError("coefficients contain nonfinite values")
        return values - np.mean(values, dtype=np.float64)

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply the exact centered-sample factor to sample coefficients."""

        projected = self.project_coefficients(coefficients)
        # Since sum(projected) is zero, Xc.T @ projected == X.T @ projected.
        result = np.asarray(self.samples.T @ projected, dtype=np.float64)
        return result / self.normalization

    def transpose(self, native_covector: np.ndarray) -> np.ndarray:
        """Apply the algebraic transpose of :meth:`apply`."""

        values = np.asarray(native_covector, dtype=np.float64)
        if values.shape != (self.state_size,):
            raise ValueError(f"native_covector must have shape ({self.state_size},)")
        if not np.isfinite(values).all():
            raise ValueError("native_covector contains nonfinite values")
        mean_projection = float(self.mean @ values)
        result = np.asarray(self.samples @ values, dtype=np.float64)
        result -= mean_projection
        # Enforce the analytically exact centered range after floating arithmetic.
        result -= np.mean(result, dtype=np.float64)
        return result / self.normalization

    def covariance_apply(self, native_vector: np.ndarray) -> np.ndarray:
        """Apply the exact empirical covariance ``F @ F.T``."""

        return self.apply(self.transpose(native_vector))

    def _covariance_apply_matrix_prevalidated(
        self,
        vectors: np.ndarray,
        *,
        block_rows: int,
        sample_workspace: np.ndarray | None,
        projection_workspace: np.ndarray,
        product_workspace: np.ndarray,
        result: np.ndarray,
    ) -> np.ndarray:
        """Apply this factor with caller-owned buffers after validation."""

        result.fill(0.0)
        cached = self._centered_float64_samples
        for start in range(0, self.sample_count, block_rows):
            stop = min(start + block_rows, self.sample_count)
            rows = stop - start
            if cached is None:
                if sample_workspace is None:  # pragma: no cover - caller invariant
                    raise RuntimeError("missing covariance sample workspace")
                centered = sample_workspace[:rows]
                np.copyto(centered, self.samples[start:stop], casting="same_kind")
                centered -= self.mean
            else:
                centered = cached[start:stop]
            projected = projection_workspace[:rows]
            np.matmul(centered, vectors, out=projected)
            np.matmul(centered.T, projected, out=product_workspace)
            result += product_workspace
        result /= self.sample_count - 1
        return result

    def covariance_apply_matrix(
        self, native_vectors: np.ndarray, *, block_rows: int = 128
    ) -> np.ndarray:
        """Apply the covariance to many native vectors in one sample pass.

        ``native_vectors`` has shape ``(state_size, vector_count)``.  Blocking
        bounds temporary memory while ensuring that all vectors share the same
        read of a potentially multi-gigabyte memory-mapped sample array.
        """

        vectors = np.asarray(native_vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != self.state_size:
            raise ValueError(
                "native_vectors must have shape "
                f"({self.state_size}, vector_count)"
            )
        if vectors.shape[1] == 0:
            raise ValueError("native_vectors must contain at least one column")
        if not np.isfinite(vectors).all():
            raise ValueError("native_vectors contain nonfinite values")
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        workspace_rows = min(block_rows, self.sample_count)
        sample_workspace = (
            None
            if self._centered_float64_samples is not None
            else np.empty((workspace_rows, self.state_size), dtype=np.float64)
        )
        projection_workspace = np.empty(
            (workspace_rows, vectors.shape[1]), dtype=np.float64
        )
        product_workspace = np.empty(
            (self.state_size, vectors.shape[1]), dtype=np.float64
        )
        result = np.empty((self.state_size, vectors.shape[1]), dtype=np.float64)
        return self._covariance_apply_matrix_prevalidated(
            vectors,
            block_rows=block_rows,
            sample_workspace=sample_workspace,
            projection_workspace=projection_workspace,
            product_workspace=product_workspace,
            result=result,
        )


@dataclass(frozen=True)
class PooledCenteredSampleCovarianceFactor:
    """Equal-phase pooled factor for ``mean_k(F_k @ F_k.T)``."""

    phase_factors: tuple[CenteredSampleCovarianceFactor, ...]

    def __post_init__(self) -> None:
        factors = tuple(self.phase_factors)
        if not factors:
            raise ValueError("pooled covariance needs at least one phase factor")
        state_sizes = {factor.state_size for factor in factors}
        if len(state_sizes) != 1:
            raise ValueError("all pooled phase factors must share a state size")
        object.__setattr__(self, "phase_factors", factors)

    @property
    def state_size(self) -> int:
        return self.phase_factors[0].state_size

    @property
    def estimated_centered_cache_bytes(self) -> int:
        """Bytes required to cache every currently uncached phase exactly."""

        return int(
            sum(
                factor.estimated_centered_cache_bytes
                for factor in self.phase_factors
                if factor.centered_cache_bytes == 0
            )
        )

    @property
    def centered_cache_bytes(self) -> int:
        """Bytes held by exact centered caches across all phase factors."""

        return int(sum(factor.centered_cache_bytes for factor in self.phase_factors))

    def with_contiguous_float64_cache(
        self,
        *,
        maximum_bytes: int,
    ) -> PooledCenteredSampleCovarianceFactor:
        """Return a pooled factor with exact contiguous caches for all phases."""

        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
            raise TypeError("maximum_bytes must be an integer")
        if maximum_bytes < 0:
            raise ValueError("maximum_bytes must be nonnegative")
        required = self.estimated_centered_cache_bytes
        if required > maximum_bytes:
            raise MemoryError(
                "Exact pooled centered covariance cache requires "
                f"{required} bytes, exceeding the {maximum_bytes}-byte budget"
            )
        return PooledCenteredSampleCovarianceFactor(
            phase_factors=tuple(
                factor.with_contiguous_float64_cache(
                    maximum_bytes=factor.estimated_centered_cache_bytes
                )
                for factor in self.phase_factors
            )
        )

    @property
    def coefficient_size(self) -> int:
        return sum(factor.coefficient_size for factor in self.phase_factors)

    @property
    def phase_count(self) -> int:
        return len(self.phase_factors)

    def _coefficient_blocks(self, coefficients: np.ndarray) -> tuple[np.ndarray, ...]:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.coefficient_size,):
            raise ValueError(
                f"coefficients must have shape ({self.coefficient_size},)"
            )
        if not np.isfinite(values).all():
            raise ValueError("coefficients contain nonfinite values")
        blocks: list[np.ndarray] = []
        start = 0
        for factor in self.phase_factors:
            stop = start + factor.coefficient_size
            blocks.append(values[start:stop])
            start = stop
        return tuple(blocks)

    def project_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        blocks = self._coefficient_blocks(coefficients)
        return np.concatenate(
            [
                factor.project_coefficients(block)
                for factor, block in zip(self.phase_factors, blocks, strict=True)
            ]
        )

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        blocks = self._coefficient_blocks(coefficients)
        result = np.zeros(self.state_size, dtype=np.float64)
        phase_scale = 1.0 / math.sqrt(self.phase_count)
        for factor, block in zip(self.phase_factors, blocks, strict=True):
            result += phase_scale * factor.apply(block)
        return result

    def transpose(self, native_covector: np.ndarray) -> np.ndarray:
        values = np.asarray(native_covector, dtype=np.float64)
        if values.shape != (self.state_size,):
            raise ValueError(f"native_covector must have shape ({self.state_size},)")
        phase_scale = 1.0 / math.sqrt(self.phase_count)
        return np.concatenate(
            [phase_scale * factor.transpose(values) for factor in self.phase_factors]
        )

    def covariance_apply(self, native_vector: np.ndarray) -> np.ndarray:
        return self.apply(self.transpose(native_vector))

    def covariance_apply_matrix(
        self, native_vectors: np.ndarray, *, block_rows: int = 128
    ) -> np.ndarray:
        """Apply the pooled covariance with one shared set of BLAS buffers."""

        vectors = np.asarray(native_vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != self.state_size:
            raise ValueError(
                "native_vectors must have shape "
                f"({self.state_size}, vector_count)"
            )
        if vectors.shape[1] == 0:
            raise ValueError("native_vectors must contain at least one column")
        if not np.isfinite(vectors).all():
            raise ValueError("native_vectors contain nonfinite values")
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")

        workspace_rows = min(
            block_rows,
            max(factor.sample_count for factor in self.phase_factors),
        )
        sample_workspace = (
            None
            if all(
                factor._centered_float64_samples is not None
                for factor in self.phase_factors
            )
            else np.empty((workspace_rows, self.state_size), dtype=np.float64)
        )
        projection_workspace = np.empty(
            (workspace_rows, vectors.shape[1]), dtype=np.float64
        )
        product_workspace = np.empty(
            (self.state_size, vectors.shape[1]), dtype=np.float64
        )
        phase_result = np.empty(
            (self.state_size, vectors.shape[1]), dtype=np.float64
        )
        result = np.zeros((self.state_size, vectors.shape[1]), dtype=np.float64)
        for factor in self.phase_factors:
            factor._covariance_apply_matrix_prevalidated(
                vectors,
                block_rows=block_rows,
                sample_workspace=sample_workspace,
                projection_workspace=projection_workspace,
                product_workspace=product_workspace,
                result=phase_result,
            )
            result += phase_result
        result /= self.phase_count
        return result


@dataclass(frozen=True)
class DenseNativeCovarianceOperator:
    """Read-only memory-mapped native covariance matrix.

    The completion manifest has this minimal contract; additional provenance
    fields are allowed::

        {
          "schema_version": 1,
          "status": "complete_dense_native_covariance",
          "covariance": {
            "file": "covariance.npy",
            "sha256": "<sha256 of the complete .npy file>",
            "shape": [state_size, state_size],
            "dtype": "<f8",
            "order": "C" | "F"
          }
        }

    ``order`` describes the on-disk NumPy array order and is preserved by the
    memory map.  The production compiler uses ``F`` so a column-major BLAS
    rank-k update can write directly into the final array.
    """

    covariance: np.memmap = field(repr=False)
    order: str
    manifest: dict[str, Any]
    manifest_path: Path
    covariance_path: Path

    def __post_init__(self) -> None:
        matrix = self.covariance
        if not isinstance(matrix, np.memmap):
            raise TypeError("dense native covariance must be a memory map")
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
        ):
            raise ValueError("dense native covariance must be a nonempty square matrix")
        if matrix.dtype != DENSE_NATIVE_COVARIANCE_DTYPE:
            raise TypeError("dense native covariance must have dtype <f8")
        if matrix.flags.writeable:
            raise ValueError("dense native covariance memory map must be read-only")
        if self.order not in {"C", "F"}:
            raise ValueError("dense native covariance order must be 'C' or 'F'")
        expected_contiguous = (
            matrix.flags.c_contiguous
            if self.order == "C"
            else matrix.flags.f_contiguous
        )
        if not expected_contiguous:
            raise ValueError(
                "dense native covariance storage does not match its declared order"
            )
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).resolve())
        object.__setattr__(
            self, "covariance_path", Path(self.covariance_path).resolve()
        )

    @property
    def state_size(self) -> int:
        return int(self.covariance.shape[0])

    @staticmethod
    def _validate_block_rows(block_rows: int) -> None:
        if isinstance(block_rows, bool) or not isinstance(
            block_rows, (int, np.integer)
        ):
            raise TypeError("block_rows must be an integer")
        if int(block_rows) <= 0:
            raise ValueError("block_rows must be positive")

    def covariance_apply(self, native_vector: np.ndarray) -> np.ndarray:
        """Apply the compiled covariance to one native vector."""

        vector = np.asarray(native_vector, dtype=np.float64)
        if vector.shape != (self.state_size,):
            raise ValueError(f"native_vector must have shape ({self.state_size},)")
        if not np.isfinite(vector).all():
            raise ValueError("native_vector contains nonfinite values")
        return np.asarray(self.covariance @ vector, dtype=np.float64)

    def covariance_apply_matrix(
        self, native_vectors: np.ndarray, *, block_rows: int = 128
    ) -> np.ndarray:
        """Apply the compiled covariance to vectors arranged by column.

        ``block_rows`` is accepted for compatibility with sample-factor
        operators.  Dense BLAS performs the complete product, so the value is
        validated but otherwise intentionally ignored.
        """

        self._validate_block_rows(block_rows)
        vectors = np.asarray(native_vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != self.state_size:
            raise ValueError(
                "native_vectors must have shape "
                f"({self.state_size}, vector_count)"
            )
        if vectors.shape[1] == 0:
            raise ValueError("native_vectors must contain at least one column")
        if not np.isfinite(vectors).all():
            raise ValueError("native_vectors contain nonfinite values")
        return np.asarray(self.covariance @ vectors, dtype=np.float64)

    @classmethod
    def load(
        cls,
        manifest_path: Path | str,
        *,
        verify_hashes: bool = True,
    ) -> DenseNativeCovarianceOperator:
        """Validate a completion manifest and open its covariance read-only."""

        path = Path(manifest_path).expanduser().resolve()
        document = load_json(path)
        if (
            type(document.get("schema_version")) is not int
            or document["schema_version"] != DENSE_NATIVE_COVARIANCE_SCHEMA_VERSION
            or document.get("status") != DENSE_NATIVE_COVARIANCE_STATUS
        ):
            raise ValueError(
                "dense native covariance manifest has an unsupported schema"
            )

        record = document.get("covariance")
        if not isinstance(record, dict):
            raise ValueError("dense native covariance manifest lacks covariance data")

        filename = record.get("file")
        if not isinstance(filename, str) or not filename:
            raise ValueError("dense native covariance manifest has an invalid file")
        covariance_path = Path(filename).expanduser()
        if covariance_path.is_absolute():
            raise ValueError("dense native covariance file must be manifest-relative")
        covariance_path = (path.parent / covariance_path).resolve()
        if covariance_path.suffix != ".npy":
            raise ValueError("dense native covariance file must be a .npy array")
        if not covariance_path.is_file():
            raise FileNotFoundError(
                f"dense native covariance file is missing: {covariance_path}"
            )

        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("dense native covariance manifest has an invalid hash")
        if verify_hashes and sha256_file(covariance_path) != digest:
            raise ValueError("dense native covariance file hash mismatch")

        shape_record = record.get("shape")
        if (
            not isinstance(shape_record, list)
            or len(shape_record) != 2
            or any(type(value) is not int or value <= 0 for value in shape_record)
            or shape_record[0] != shape_record[1]
        ):
            raise ValueError(
                "dense native covariance manifest shape must be "
                "[state_size, state_size]"
            )
        expected_shape = tuple(shape_record)
        if record.get("dtype") != DENSE_NATIVE_COVARIANCE_DTYPE.str:
            raise ValueError("dense native covariance manifest dtype must be <f8")
        order = record.get("order")
        if order not in {"C", "F"}:
            raise ValueError(
                "dense native covariance manifest order must be 'C' or 'F'"
            )

        matrix = np.load(covariance_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(matrix, np.memmap):
            if isinstance(matrix, np.lib.npyio.NpzFile):
                matrix.close()
            raise ValueError(
                "dense native covariance file is not a memory-mappable .npy"
            )
        if matrix.shape != expected_shape:
            raise ValueError(
                "dense native covariance shape does not match its manifest"
            )
        if matrix.dtype != DENSE_NATIVE_COVARIANCE_DTYPE:
            raise ValueError(
                "dense native covariance dtype does not match its manifest"
            )
        expected_contiguous = (
            matrix.flags.c_contiguous if order == "C" else matrix.flags.f_contiguous
        )
        if not expected_contiguous:
            raise ValueError(
                "dense native covariance storage does not match its declared order"
            )
        return cls(
            covariance=matrix,
            order=order,
            manifest=document,
            manifest_path=path,
            covariance_path=covariance_path,
        )


@dataclass(frozen=True)
class NativeCovarianceBundle:
    """Validated phase-specific and pooled native covariance factors."""

    phase_offsets: np.ndarray
    phase_factors: tuple[CenteredSampleCovarianceFactor, ...]
    coordinate_variances: np.ndarray
    manifest: dict[str, Any]
    manifest_path: Path

    def __post_init__(self) -> None:
        offsets = np.asarray(self.phase_offsets, dtype=np.int64)
        factors = tuple(self.phase_factors)
        variances = np.asarray(self.coordinate_variances, dtype=np.float64)
        if offsets.ndim != 1 or offsets.size != len(factors):
            raise ValueError("phase offsets do not match phase factors")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("phase offsets must be strictly increasing")
        if not factors:
            raise ValueError("native covariance bundle has no factors")
        if variances.shape != (len(factors), factors[0].state_size):
            raise ValueError("coordinate variances have the wrong shape")
        if np.any(variances < 0.0) or not np.isfinite(variances).all():
            raise ValueError("coordinate variances must be finite and nonnegative")
        object.__setattr__(self, "phase_offsets", offsets)
        object.__setattr__(self, "phase_factors", factors)
        object.__setattr__(self, "coordinate_variances", variances)
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).resolve())

    @property
    def pooled_factor(self) -> PooledCenteredSampleCovarianceFactor:
        return PooledCenteredSampleCovarianceFactor(self.phase_factors)

    @property
    def estimated_centered_cache_bytes(self) -> int:
        """Bytes needed to cache every currently uncached phase exactly."""

        return self.pooled_factor.estimated_centered_cache_bytes

    @property
    def centered_cache_bytes(self) -> int:
        """Bytes held by exact centered caches across this bundle."""

        return self.pooled_factor.centered_cache_bytes

    def with_contiguous_float64_cache(
        self,
        *,
        maximum_bytes: int,
    ) -> NativeCovarianceBundle:
        """Return an equivalent bundle with exact centered phase caches."""

        pooled = self.pooled_factor.with_contiguous_float64_cache(
            maximum_bytes=maximum_bytes
        )
        return NativeCovarianceBundle(
            phase_offsets=self.phase_offsets,
            phase_factors=pooled.phase_factors,
            coordinate_variances=self.coordinate_variances,
            manifest=self.manifest,
            manifest_path=self.manifest_path,
        )

    def phase_factor(self, phase_offset: int) -> CenteredSampleCovarianceFactor:
        matches = np.flatnonzero(self.phase_offsets == int(phase_offset))
        if matches.size != 1:
            raise KeyError(f"phase offset is not present: {phase_offset}")
        return self.phase_factors[int(matches[0])]

    @classmethod
    def load(
        cls, manifest_path: Path | str, *, verify_hashes: bool = True
    ) -> NativeCovarianceBundle:
        path = Path(manifest_path).expanduser().resolve()
        document = load_json(path)
        if (
            document.get("schema_version") != NATIVE_COVARIANCE_SCHEMA_VERSION
            or document.get("status") != NATIVE_COVARIANCE_STATUS
        ):
            raise ValueError("native covariance manifest has an unsupported schema")
        means_record = document.get("means_and_variances")
        if not isinstance(means_record, dict):
            raise ValueError("native covariance manifest lacks mean/variance data")
        means_path = (path.parent / str(means_record["file"])).resolve()
        if verify_hashes and sha256_file(means_path) != means_record.get("sha256"):
            raise ValueError("native covariance means/variances hash mismatch")
        with np.load(means_path, allow_pickle=False) as archive:
            offsets = np.asarray(archive["phase_offsets"], dtype=np.int64)
            means = np.asarray(archive["means"], dtype=np.float64)
            variances = np.asarray(archive["coordinate_variances"], dtype=np.float64)
        if means_record.get("means_sha256") != sha256_array(means):
            raise ValueError("native covariance mean array hash mismatch")
        if means_record.get("coordinate_variances_sha256") != sha256_array(variances):
            raise ValueError("native covariance variance array hash mismatch")

        records = document.get("phase_factors")
        if not isinstance(records, list) or len(records) != offsets.size:
            raise ValueError("native covariance phase-factor records are incomplete")
        factors: list[CenteredSampleCovarianceFactor] = []
        for position, (offset, record) in enumerate(zip(offsets, records, strict=True)):
            if (
                not isinstance(record, dict)
                or int(record.get("phase_offset", -1)) != offset
            ):
                raise ValueError("native covariance phase record ordering changed")
            sample_path = (path.parent / str(record["file"])).resolve()
            if verify_hashes and sha256_file(sample_path) != record.get("sha256"):
                raise ValueError(
                    f"native covariance sample hash mismatch: {sample_path}"
                )
            samples = np.load(sample_path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(int(value) for value in record["shape"])
            if samples.shape != expected_shape or samples.dtype.str != record["dtype"]:
                raise ValueError(
                    f"native covariance sample schema changed: {sample_path}"
                )
            factors.append(
                CenteredSampleCovarianceFactor(
                    samples=samples,
                    mean=means[position],
                    phase_offset=int(offset),
                    source_path=sample_path,
                )
            )
        return cls(
            phase_offsets=offsets,
            phase_factors=tuple(factors),
            coordinate_variances=variances,
            manifest=document,
            manifest_path=path,
        )


def solve_minimum_action(
    factor: CovarianceFactor,
    native_increment: np.ndarray,
    *,
    atol: float = 1.0e-10,
    btol: float = 1.0e-10,
    iteration_limit: int | None = None,
    required_relative_residual: float | None = 1.0e-8,
) -> MinimumActionResult:
    """Find the minimum-norm factor coefficient for a native increment.

    When the increment lies in ``range(F)``, the returned squared coefficient
    norm is the empirical pseudoinverse quadratic ``a.T @ pinv(C) @ a`` for
    ``C = F @ F.T``.  A nonzero residual diagnoses an increment outside the
    empirical covariance range or an unconverged iterative solve.
    """

    target = np.asarray(native_increment, dtype=np.float64)
    if target.shape != (factor.state_size,):
        raise ValueError(f"native_increment must have shape ({factor.state_size},)")
    if not np.isfinite(target).all():
        raise ValueError("native_increment contains nonfinite values")
    if atol < 0.0 or btol < 0.0:
        raise ValueError("LSQR tolerances must be nonnegative")
    if required_relative_residual is not None and required_relative_residual < 0.0:
        raise ValueError("required_relative_residual must be nonnegative")
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        coefficients = np.zeros(factor.coefficient_size, dtype=np.float64)
        return MinimumActionResult(
            coefficients=coefficients,
            reconstructed_increment=target.copy(),
            squared_action=0.0,
            residual_l2=0.0,
            relative_residual=0.0,
            iterations=0,
            stop_code=0,
            operator_norm_estimate=0.0,
            condition_estimate=0.0,
        )

    operator = scipy.sparse.linalg.LinearOperator(
        shape=(factor.state_size, factor.coefficient_size),
        matvec=factor.apply,
        rmatvec=factor.transpose,
        dtype=np.float64,
    )
    result = scipy.sparse.linalg.lsqr(
        operator,
        target,
        atol=float(atol),
        btol=float(btol),
        iter_lim=iteration_limit,
        show=False,
    )
    coefficients = factor.project_coefficients(np.asarray(result[0], dtype=np.float64))
    reconstructed = factor.apply(coefficients)
    residual = float(np.linalg.norm(reconstructed - target))
    relative = residual / target_norm
    if required_relative_residual is not None and relative > required_relative_residual:
        raise ValueError(
            "native increment is outside the certified covariance range or the "
            f"minimum-action solve did not converge: relative residual {relative:g}"
        )
    return MinimumActionResult(
        coefficients=coefficients,
        reconstructed_increment=reconstructed,
        squared_action=float(coefficients @ coefficients),
        residual_l2=residual,
        relative_residual=relative,
        iterations=int(result[2]),
        stop_code=int(result[1]),
        operator_norm_estimate=float(result[5]),
        condition_estimate=float(result[6]),
    )
