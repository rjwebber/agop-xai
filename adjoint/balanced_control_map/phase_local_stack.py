"""Coherent phase-local covariance controls for the authentic ZC state.

Separate principal-component fits at different annual phases have arbitrary
signs and rotations, so their columns cannot be interpreted as one evolving
control coordinate.  This module instead finds one sample-space basis from
the average standardized-observation covariance over all requested phases.
It then regresses each phase's native and observed anomalies on those same
sample coefficients.

For phase ``k`` and a common orthonormal sample basis ``U`` the factors are

``B_k = Xc_k.T @ U / sqrt(n - 1)`` and
``G_k = Zc_k.T @ U / sqrt(n - 1)``.

The reduced vector therefore has the same empirical meaning at every phase:
it weights the same training trajectories through the same columns of ``U``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

from .balanced_map import ControlSegment, IndependentControlLayout


@dataclass(frozen=True)
class PhaseLocalControlStack:
    """Phase-specific regressions sharing one whitened sample coordinate."""

    B: np.ndarray
    G: np.ndarray
    pooled_eigenvalues: np.ndarray
    native_mean: np.ndarray
    native_scale: np.ndarray
    observation_mean: np.ndarray
    sample_vectors: np.ndarray
    phase_offsets: np.ndarray
    layout: IndependentControlLayout
    sample_count: int
    pooled_observation_variance_trace: float

    @property
    def phase_count(self) -> int:
        return self.B.shape[0]

    @property
    def rank(self) -> int:
        return self.B.shape[2]

    @property
    def observation_size(self) -> int:
        return self.G.shape[1]

    def phase_position(self, phase_offset: int) -> int:
        matches = np.flatnonzero(self.phase_offsets == int(phase_offset))
        if matches.size != 1:
            raise KeyError(f"phase offset is not present: {phase_offset}")
        return int(matches[0])

    def packed_control_stack(self) -> np.ndarray:
        """Return all explicit ``E @ B_k`` maps with shape ``(p,59148,r)``."""

        compact = np.transpose(self.B, (0, 2, 1))
        return np.transpose(self.layout.compact_to_packed(compact), (0, 2, 1))

    def apply_B(
        self, control: np.ndarray, *, phase_offset: int, packed: bool = True
    ) -> np.ndarray:
        """Map common reduced coordinates to one phase's native increment."""

        values = np.asarray(control, dtype=np.float64)
        if values.shape[-1:] != (self.rank,):
            raise ValueError(f"control must end in length {self.rank}")
        position = self.phase_position(phase_offset)
        compact = values @ self.B[position].T
        return self.layout.compact_to_packed(compact) if packed else compact

    def apply_BT(
        self,
        native_covector: np.ndarray,
        *,
        phase_offset: int,
        packed: bool = True,
    ) -> np.ndarray:
        """Apply the exact Euclidean transpose of one phase's native map."""

        values = np.asarray(native_covector, dtype=np.float64)
        compact = self.layout.packed_to_compact(values) if packed else values
        if compact.shape[-1:] != (self.layout.compact_size,):
            raise ValueError(
                f"native covector must end in length {self.layout.compact_size}"
            )
        return compact @ self.B[self.phase_position(phase_offset)]

    def apply_G(self, control: np.ndarray, *, phase_offset: int) -> np.ndarray:
        """Predict one phase's standardized core4 displacement."""

        values = np.asarray(control, dtype=np.float64)
        if values.shape[-1:] != (self.rank,):
            raise ValueError(f"control must end in length {self.rank}")
        return values @ self.G[self.phase_position(phase_offset)].T

    def apply_GT(
        self, observation_covector: np.ndarray, *, phase_offset: int
    ) -> np.ndarray:
        """Apply the exact transpose of one phase's observation map."""

        values = np.asarray(observation_covector, dtype=np.float64)
        if values.shape[-1:] != (self.observation_size,):
            raise ValueError(
                f"observation covector must end in length {self.observation_size}"
            )
        return values @ self.G[self.phase_position(phase_offset)]

    def validate(self) -> None:
        """Validate dimensions and the pooled EOF identity."""

        if self.B.ndim != 3 or self.G.ndim != 3:
            raise ValueError("B and G must be phase-indexed rank-three arrays")
        phases, compact_size, rank = self.B.shape
        if phases == 0 or compact_size != self.layout.compact_size or rank == 0:
            raise ValueError("B has invalid phase, native, or rank dimensions")
        if self.G.shape[0] != phases or self.G.shape[2] != rank:
            raise ValueError("G phase/rank dimensions disagree with B")
        if self.pooled_eigenvalues.shape != (rank,):
            raise ValueError("pooled eigenvalue count disagrees with the rank")
        if self.native_mean.shape != (phases, compact_size):
            raise ValueError("native_mean has the wrong shape")
        if self.native_scale.shape != self.native_mean.shape:
            raise ValueError("native_scale has the wrong shape")
        if self.observation_mean.shape != self.G.shape[:2]:
            raise ValueError("observation_mean has the wrong shape")
        if self.sample_vectors.shape != (self.sample_count, rank):
            raise ValueError("sample_vectors has the wrong shape")
        if self.phase_offsets.shape != (phases,):
            raise ValueError("phase_offsets has the wrong shape")
        if np.any(np.diff(self.phase_offsets) <= 0):
            raise ValueError("phase_offsets must be strictly increasing")
        arrays = (
            self.B,
            self.G,
            self.pooled_eigenvalues,
            self.native_mean,
            self.native_scale,
            self.observation_mean,
            self.sample_vectors,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("phase-local control artifact contains nonfinite values")
        if np.any(self.pooled_eigenvalues <= 0.0) or np.any(
            np.diff(self.pooled_eigenvalues) > 0.0
        ):
            raise ValueError("pooled eigenvalues must be positive and nonincreasing")
        if not math.isfinite(self.pooled_observation_variance_trace) or (
            self.pooled_observation_variance_trace <= 0.0
        ):
            raise ValueError("pooled observation variance trace must be positive")

        orthogonality = self.sample_vectors.T @ self.sample_vectors
        orthogonality_error = float(
            np.max(np.abs(orthogonality - np.eye(rank)), initial=0.0)
        )
        if orthogonality_error > 5.0e-10:
            raise ValueError(
                f"common sample vectors are not orthonormal: {orthogonality_error}"
            )
        pooled_gram = np.einsum("pdr,pds->rs", self.G, self.G) / phases
        gram_error = float(
            np.max(
                np.abs(pooled_gram - np.diag(self.pooled_eigenvalues)),
                initial=0.0,
            )
        )
        tolerance = 5.0e-9 * max(1.0, float(self.pooled_eigenvalues[0]))
        if gram_error > tolerance:
            raise ValueError(f"pooled G Gram identity fails: {gram_error}")

    def save(self, path: Path | str, *, metadata: dict[str, Any] | None = None) -> None:
        """Serialize the stack without pickle-dependent arrays."""

        descriptor = {
            "schema_version": 1,
            "sample_count": self.sample_count,
            "pooled_observation_variance_trace": (
                self.pooled_observation_variance_trace
            ),
            "packed_size": self.layout.packed_size,
            "manifest_schema_version": self.layout.manifest_schema_version,
            "segments": [
                {
                    "name": segment.name,
                    "packed_start": segment.packed_start,
                    "packed_stop": segment.packed_stop,
                    "compact_start": segment.compact_start,
                    "compact_stop": segment.compact_stop,
                    "shape": list(segment.shape),
                    "order": segment.order,
                    "role": segment.role,
                    "restart_record": segment.restart_record,
                }
                for segment in self.layout.segments
            ],
            "metadata": metadata or {},
        }
        with Path(path).open("wb") as stream:
            np.savez(
                stream,
                B=np.asarray(self.B, dtype=np.float64),
                G=np.asarray(self.G, dtype=np.float64),
                pooled_eigenvalues=np.asarray(
                    self.pooled_eigenvalues, dtype=np.float64
                ),
                native_mean=np.asarray(self.native_mean, dtype=np.float64),
                native_scale=np.asarray(self.native_scale, dtype=np.float64),
                observation_mean=np.asarray(
                    self.observation_mean, dtype=np.float64
                ),
                sample_vectors=np.asarray(self.sample_vectors, dtype=np.float64),
                phase_offsets=np.asarray(self.phase_offsets, dtype=np.int64),
                descriptor_json=np.asarray(json.dumps(descriptor, sort_keys=True)),
            )

    @classmethod
    def load(
        cls, path: Path | str
    ) -> tuple[PhaseLocalControlStack, dict[str, Any]]:
        """Load and validate an artifact written by :meth:`save`."""

        with np.load(path, allow_pickle=False) as archive:
            descriptor = json.loads(str(archive["descriptor_json"].item()))
            segments = tuple(
                ControlSegment(
                    name=str(raw["name"]),
                    packed_start=int(raw["packed_start"]),
                    packed_stop=int(raw["packed_stop"]),
                    compact_start=int(raw["compact_start"]),
                    compact_stop=int(raw["compact_stop"]),
                    shape=tuple(int(value) for value in raw["shape"]),
                    order=str(raw["order"]),
                    role=str(raw["role"]),
                    restart_record=(
                        None
                        if raw["restart_record"] is None
                        else int(raw["restart_record"])
                    ),
                )
                for raw in descriptor["segments"]
            )
            layout = IndependentControlLayout(
                packed_size=int(descriptor["packed_size"]),
                segments=segments,
                manifest_schema_version=int(descriptor["manifest_schema_version"]),
            )
            result = cls(
                B=np.asarray(archive["B"], dtype=np.float64),
                G=np.asarray(archive["G"], dtype=np.float64),
                pooled_eigenvalues=np.asarray(
                    archive["pooled_eigenvalues"], dtype=np.float64
                ),
                native_mean=np.asarray(archive["native_mean"], dtype=np.float64),
                native_scale=np.asarray(archive["native_scale"], dtype=np.float64),
                observation_mean=np.asarray(
                    archive["observation_mean"], dtype=np.float64
                ),
                sample_vectors=np.asarray(
                    archive["sample_vectors"], dtype=np.float64
                ),
                phase_offsets=np.asarray(archive["phase_offsets"], dtype=np.int64),
                layout=layout,
                sample_count=int(descriptor["sample_count"]),
                pooled_observation_variance_trace=float(
                    descriptor["pooled_observation_variance_trace"]
                ),
            )
        result.validate()
        return result, dict(descriptor.get("metadata", {}))


def fit_phase_local_control_stack(
    native_samples_by_phase: Sequence[np.ndarray],
    standardized_observations_by_phase: Sequence[np.ndarray],
    *,
    phase_offsets: np.ndarray,
    layout: IndependentControlLayout,
    rank: int,
    relative_eigenvalue_floor: float = 1.0e-10,
) -> PhaseLocalControlStack:
    """Fit coherent phase-specific maps from paired authentic trajectories."""

    native_sequence = tuple(native_samples_by_phase)
    observation_sequence = tuple(standardized_observations_by_phase)
    offsets = np.asarray(phase_offsets, dtype=np.int64)
    phase_count = len(native_sequence)
    if phase_count == 0 or len(observation_sequence) != phase_count:
        raise ValueError("native and observation phase sequences must be nonempty")
    if offsets.shape != (phase_count,) or np.any(np.diff(offsets) <= 0):
        raise ValueError("phase_offsets must be strictly increasing and phase-matched")

    first_native = np.asarray(native_sequence[0])
    first_observation = np.asarray(observation_sequence[0])
    if first_native.ndim != 2 or first_observation.ndim != 2:
        raise ValueError("phase samples must be matrices")
    sample_count = first_native.shape[0]
    observation_size = first_observation.shape[1]
    if (
        sample_count < 2
        or first_native.shape[1] != layout.compact_size
        or first_observation.shape[0] != sample_count
    ):
        raise ValueError("phase sample dimensions are incompatible")
    if not isinstance(rank, int) or rank <= 0 or rank >= sample_count:
        raise ValueError("rank must be positive and smaller than sample count")
    if not math.isfinite(relative_eigenvalue_floor) or not (
        0.0 <= relative_eigenvalue_floor < 1.0
    ):
        raise ValueError("relative_eigenvalue_floor must lie in [0,1)")

    denominator = sample_count - 1
    pooled_gram = np.zeros((sample_count, sample_count), dtype=np.float64)
    observation_means = np.empty(
        (phase_count, observation_size), dtype=np.float64
    )
    pooled_trace = 0.0
    centered_observations: list[np.ndarray] = []
    for position, raw in enumerate(observation_sequence):
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (sample_count, observation_size):
            raise ValueError("observation phase matrices have inconsistent shapes")
        if not np.isfinite(values).all():
            raise ValueError("observation phase samples contain nonfinite values")
        mean = np.mean(values, axis=0, dtype=np.float64)
        centered = values - mean
        observation_means[position] = mean
        centered_observations.append(centered)
        pooled_gram += centered @ centered.T
        pooled_trace += float(np.sum(centered * centered, dtype=np.float64))
    pooled_gram /= phase_count * denominator
    pooled_gram = 0.5 * (pooled_gram + pooled_gram.T)
    pooled_trace /= phase_count * denominator

    eigenvalues, sample_vectors = scipy.linalg.eigh(
        pooled_gram,
        subset_by_index=(sample_count - rank, sample_count - 1),
        driver="evr",
        check_finite=True,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    sample_vectors = np.asarray(sample_vectors[:, order], dtype=np.float64)
    if eigenvalues[-1] <= relative_eigenvalue_floor * eigenvalues[0]:
        raise ValueError("requested rank reaches an unresolved pooled mode")

    normalization = math.sqrt(denominator)
    B = np.empty(
        (phase_count, layout.compact_size, rank), dtype=np.float64
    )
    G = np.empty((phase_count, observation_size, rank), dtype=np.float64)
    native_means = np.empty(
        (phase_count, layout.compact_size), dtype=np.float64
    )
    native_scales = np.empty_like(native_means)
    for position, raw in enumerate(native_sequence):
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (sample_count, layout.compact_size):
            raise ValueError("native phase matrices have inconsistent shapes")
        if not np.isfinite(values).all():
            raise ValueError("native phase samples contain nonfinite values")
        mean = np.mean(values, axis=0, dtype=np.float64)
        centered = values - mean
        native_means[position] = mean
        native_scales[position] = np.std(centered, axis=0, ddof=1)
        B[position] = centered.T @ sample_vectors / normalization
        G[position] = (
            centered_observations[position].T @ sample_vectors / normalization
        )

    # One sign choice per common mode, based on all observable phases together.
    for column in range(rank):
        flattened = G[:, :, column].reshape(-1)
        pivot = int(np.argmax(np.abs(flattened)))
        if flattened[pivot] < 0.0:
            sample_vectors[:, column] *= -1.0
            B[:, :, column] *= -1.0
            G[:, :, column] *= -1.0

    realized = np.diag(np.einsum("pdr,pds->rs", G, G) / phase_count).copy()
    result = PhaseLocalControlStack(
        B=B,
        G=G,
        pooled_eigenvalues=realized,
        native_mean=native_means,
        native_scale=native_scales,
        observation_mean=observation_means,
        sample_vectors=sample_vectors,
        phase_offsets=offsets,
        layout=layout,
        sample_count=sample_count,
        pooled_observation_variance_trace=pooled_trace,
    )
    result.validate()
    return result
