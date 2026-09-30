"""Low-rank native-state controls learned from authentic ZC checkpoints.

The construction intentionally does not invert ``ZAVG``.  Given paired,
same-phase samples of independent native controls ``X`` and fresh core4
standardized observations ``Z``, it diagonalizes the observation-space sample
covariance in sample space.  If ``U_r`` contains the retained sample-space
eigenvectors, the two factors are

    B = X_c.T @ U_r / sqrt(n - 1)
    G = Z_c.T @ U_r / sqrt(n - 1).

Thus a dimensionless, covariance-whitened reduced control ``a`` produces the
raw native increment ``B a`` and the paired linear observation prediction
``G a``.  ``B.T`` is the exact algebraic transpose needed to project a packed
native adjoint gradient.  A requested core4 direction is fit by Tikhonov-
regularized least squares in the retained range of ``G``; it is not claimed to
have an exact inverse outside that range.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

CORE4_FIELDS = (
    "sst_anomaly",
    "thermocline_depth",
    "zonal_ocean_current",
    "meridional_ocean_current",
)


@dataclass(frozen=True)
class ControlSegment:
    """One manifest segment in compact independent-control order."""

    name: str
    packed_start: int
    packed_stop: int
    compact_start: int
    compact_stop: int
    shape: tuple[int, ...]
    order: str
    role: str
    restart_record: int | None

    @property
    def size(self) -> int:
        return self.packed_stop - self.packed_start


@dataclass(frozen=True)
class IndependentControlLayout:
    """Mapping between compact controls and the 59,148-value packed state."""

    packed_size: int
    segments: tuple[ControlSegment, ...]
    manifest_schema_version: int

    @property
    def compact_size(self) -> int:
        return sum(segment.size for segment in self.segments)

    @property
    def packed_indices(self) -> np.ndarray:
        return np.concatenate(
            [
                np.arange(segment.packed_start, segment.packed_stop, dtype=np.int64)
                for segment in self.segments
            ]
        )

    def compact_to_packed(self, values: np.ndarray) -> np.ndarray:
        compact = np.asarray(values)
        if compact.shape[-1:] != (self.compact_size,):
            raise ValueError(
                f"compact values must end in length {self.compact_size}, "
                f"not {compact.shape}"
            )
        packed = np.zeros((*compact.shape[:-1], self.packed_size), dtype=compact.dtype)
        packed[..., self.packed_indices] = compact
        return packed

    def packed_to_compact(self, values: np.ndarray) -> np.ndarray:
        packed = np.asarray(values)
        if packed.shape[-1:] != (self.packed_size,):
            raise ValueError(
                f"packed values must end in length {self.packed_size}, "
                f"not {packed.shape}"
            )
        return packed[..., self.packed_indices]


def load_independent_control_layout(path: Path | str) -> IndependentControlLayout:
    """Load and validate the conservative independent-control mask."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    real32 = document.get("arrays", {}).get("real32", {})
    packed_size = int(real32.get("length", -1))
    if packed_size <= 0:
        raise ValueError("state manifest has no positive real32 packed length")
    raw_segments = real32.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("state manifest real32.segments must be a list")
    segments: list[ControlSegment] = []
    compact_start = 0
    prior_stop = -1
    for raw in raw_segments:
        if not isinstance(raw, dict) or not raw.get("independent_control", False):
            continue
        start = int(raw["start"])
        stop = int(raw["stop"])
        shape = tuple(int(value) for value in raw["shape"])
        if not 0 <= start < stop <= packed_size:
            raise ValueError(f"invalid packed interval for {raw.get('name')!r}")
        if start < prior_stop:
            raise ValueError("independent-control segments are not strictly ordered")
        if math.prod(shape) != stop - start:
            raise ValueError(f"manifest shape mismatch for {raw.get('name')!r}")
        segment = ControlSegment(
            name=str(raw["name"]),
            packed_start=start,
            packed_stop=stop,
            compact_start=compact_start,
            compact_stop=compact_start + stop - start,
            shape=shape,
            order=str(raw["order"]),
            role=str(raw["role"]),
            restart_record=(
                None
                if raw.get("restart_record") is None
                else int(raw["restart_record"])
            ),
        )
        segments.append(segment)
        compact_start = segment.compact_stop
        prior_stop = stop
    if not segments:
        raise ValueError("state manifest contains no independent controls")
    return IndependentControlLayout(
        packed_size=packed_size,
        segments=tuple(segments),
        manifest_schema_version=int(document.get("schema_version", -1)),
    )


def _fortran_records(payload: bytes) -> tuple[memoryview, ...]:
    """Split a little-endian, four-byte-marker sequential Fortran file."""

    view = memoryview(payload)
    position = 0
    records: list[memoryview] = []
    while position < len(view):
        if position + 4 > len(view):
            raise ValueError("truncated leading Fortran record marker")
        count = struct.unpack_from("<I", view, position)[0]
        position += 4
        stop = position + count
        if stop + 4 > len(view):
            raise ValueError("truncated Fortran record payload")
        trailing = struct.unpack_from("<I", view, stop)[0]
        if trailing != count:
            raise ValueError("Fortran record markers disagree")
        records.append(view[position:stop])
        position = stop + 4
    return tuple(records)


@dataclass(frozen=True)
class RestartSample:
    """Independent controls and clock extracted from one complete restart."""

    native_controls: np.ndarray
    nt: int
    time_months: float


def read_restart_sample(
    path: Path | str,
    layout: IndependentControlLayout,
) -> RestartSample:
    """Read a complete fresh restart file from disk."""

    return read_restart_payload(Path(path).read_bytes(), layout)


def read_restart_payload(
    payload: bytes,
    layout: IndependentControlLayout,
) -> RestartSample:
    """Read independent controls without interpreting diagnostic work arrays.

    The fresh restart contains four Fortran records.  The first 80 bytes of
    record 2 are the historical character header.  Independent controls in the
    supported manifest occur only in records 2 and 3; the hidden ``AK`` state
    omitted by the legacy restart is not an independent control.
    """

    records = _fortran_records(payload)
    if len(records) != 4:
        raise ValueError(f"expected four fresh restart records, found {len(records)}")
    if len(records[0]) < 8:
        raise ValueError("restart clock record is too short")
    time_months = float(struct.unpack_from("<f", records[0], 0)[0])
    nt = int(struct.unpack_from("<i", records[0], 4)[0])

    record2_segments = [
        segment for segment in layout.segments if segment.restart_record == 2
    ]
    record3_segments = [
        segment for segment in layout.segments if segment.restart_record == 3
    ]
    # The last stored real32 offset in each record includes non-control segments.
    record2_stop = max(
        segment.packed_stop
        for segment in _manifest_record_segments(layout, 2)
    )
    record3_all = _manifest_record_segments(layout, 3)
    record3_start = min(segment.packed_start for segment in record3_all)
    record3_stop = max(segment.packed_stop for segment in record3_all)
    header_bytes = len(records[1]) - 4 * record2_stop
    if header_bytes != 80:
        raise ValueError(
            f"fresh restart record-2 header is {header_bytes} bytes, expected 80"
        )
    record2 = np.frombuffer(records[1][header_bytes:], dtype="<f4")
    record3 = np.frombuffer(records[2], dtype="<f4")
    if record2.size != record2_stop:
        raise ValueError("restart record 2 disagrees with the state manifest")
    if record3.size != record3_stop - record3_start:
        raise ValueError("restart record 3 disagrees with the state manifest")

    compact = np.empty(layout.compact_size, dtype=np.float64)
    for segment in record2_segments:
        compact[segment.compact_start : segment.compact_stop] = record2[
            segment.packed_start : segment.packed_stop
        ]
    for segment in record3_segments:
        local_start = segment.packed_start - record3_start
        local_stop = segment.packed_stop - record3_start
        compact[segment.compact_start : segment.compact_stop] = record3[
            local_start:local_stop
        ]
    if not np.isfinite(compact).all():
        raise ValueError("restart independent controls contain a nonfinite value")
    return RestartSample(native_controls=compact, nt=nt, time_months=time_months)


def _manifest_record_segments(
    layout: IndependentControlLayout,
    record: int,
) -> tuple[ControlSegment, ...]:
    """Return known record bounds, including noncontrols implied by this layout.

    The supported state contract has fixed record boundaries.  They are kept
    here rather than inferred from the selected controls because record 3 also
    contains non-independent diagnostics before and after ``TO``.
    """

    if layout.packed_size != 59_148:
        raise ValueError("restart reader supports only the 59,148-value ZC contract")
    if record == 2:
        bounds = ((0, 28_041),)
    elif record == 3:
        bounds = ((28_041, 36_201),)
    else:
        raise ValueError(f"unsupported restart record {record}")
    return tuple(
        ControlSegment(
            name=f"record_{record}",
            packed_start=start,
            packed_stop=stop,
            compact_start=0,
            compact_stop=stop - start,
            shape=(stop - start,),
            order="F",
            role="record bound",
            restart_record=record,
        )
        for start, stop in bounds
    )


@dataclass(frozen=True)
class LiftSolution:
    """One regularized core4-to-native reduced lift."""

    control: np.ndarray
    native_increment: np.ndarray
    predicted_observation: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class BalancedControlMap:
    """Explicit paired native/observation covariance factors ``B`` and ``G``."""

    B: np.ndarray
    G: np.ndarray
    eigenvalues: np.ndarray
    native_mean: np.ndarray
    native_scale: np.ndarray
    observation_mean: np.ndarray
    layout: IndependentControlLayout
    sample_count: int
    observation_variance_trace: float

    @property
    def rank(self) -> int:
        return self.B.shape[1]

    @property
    def observation_size(self) -> int:
        return self.G.shape[0]

    def packed_control_matrix(self) -> np.ndarray:
        """Return the explicit packed-state map ``E @ B``.

        ``B`` is stored economically in the 28,591 independent coordinates,
        whereas the differentiated ZC step acts on all 59,148 packed real
        coordinates.  This helper inserts the structural zeros required by
        that state contract and returns shape ``(packed_size, rank)``.
        """

        return self.layout.compact_to_packed(self.B.T).T

    def apply_B(self, control: np.ndarray, *, packed: bool = True) -> np.ndarray:
        """Apply ``B``; controls are dimensionless one-sigma coordinates."""

        values = np.asarray(control, dtype=np.float64)
        if values.shape[-1:] != (self.rank,):
            raise ValueError(f"control must end in length {self.rank}")
        compact = values @ self.B.T
        return self.layout.compact_to_packed(compact) if packed else compact

    def apply_BT(
        self, native_covector: np.ndarray, *, packed: bool = True
    ) -> np.ndarray:
        """Apply the exact Euclidean transpose ``B.T``."""

        values = np.asarray(native_covector, dtype=np.float64)
        compact = self.layout.packed_to_compact(values) if packed else values
        if compact.shape[-1:] != (self.layout.compact_size,):
            raise ValueError(
                f"native covector must end in length {self.layout.compact_size}"
            )
        return compact @ self.B

    def apply_G(self, control: np.ndarray) -> np.ndarray:
        """Predict the standardized core4 image of a reduced control."""

        values = np.asarray(control, dtype=np.float64)
        if values.shape[-1:] != (self.rank,):
            raise ValueError(f"control must end in length {self.rank}")
        return values @ self.G.T

    def apply_GT(self, observation_covector: np.ndarray) -> np.ndarray:
        """Apply the exact observation-factor transpose ``G.T``."""

        values = np.asarray(observation_covector, dtype=np.float64)
        if values.shape[-1:] != (self.observation_size,):
            raise ValueError(
                f"observation covector must end in length {self.observation_size}"
            )
        return values @ self.G

    def lift_observation(
        self,
        target: np.ndarray,
        *,
        ridge_fraction: float = 1.0e-3,
        packed: bool = True,
    ) -> LiftSolution:
        """Fit a core4 increment in ``range(G)`` with a covariance-action prior.

        ``ridge_fraction`` multiplies the leading retained observation
        eigenvalue.  The minimized nondimensional objective is

        ``0.5 ||G a - d||_2^2 + 0.5 alpha ||a||_2^2``.
        """

        direction = np.asarray(target, dtype=np.float64)
        if direction.shape != (self.observation_size,):
            raise ValueError(
                f"target must have shape ({self.observation_size},), "
                f"not {direction.shape}"
            )
        if not np.isfinite(direction).all():
            raise ValueError("target contains a nonfinite value")
        if not math.isfinite(ridge_fraction) or ridge_fraction < 0.0:
            raise ValueError("ridge_fraction must be finite and nonnegative")
        gram = self.G.T @ self.G
        alpha = float(ridge_fraction * self.eigenvalues[0])
        control = scipy.linalg.solve(
            gram + alpha * np.eye(self.rank),
            self.G.T @ direction,
            assume_a="pos",
            check_finite=True,
        )
        prediction = self.apply_G(control)
        native = self.apply_B(control, packed=packed)
        projection_control = scipy.linalg.solve(
            gram,
            self.G.T @ direction,
            assume_a="pos",
            check_finite=True,
        )
        projection = self.apply_G(projection_control)
        residual = prediction - direction
        target_norm = float(np.linalg.norm(direction))
        prediction_norm = float(np.linalg.norm(prediction))
        cosine = None
        if target_norm > 0.0 and prediction_norm > 0.0:
            cosine = float(direction @ prediction / (target_norm * prediction_norm))
        diagnostics: dict[str, Any] = {
            "ridge_fraction_of_leading_eigenvalue": float(ridge_fraction),
            "ridge_absolute": alpha,
            "control_l2_mahalanobis_action_sqrt": float(np.linalg.norm(control)),
            "half_squared_control_action": float(0.5 * control @ control),
            "target_l2": target_norm,
            "predicted_l2": prediction_norm,
            "residual_l2": float(np.linalg.norm(residual)),
            "relative_residual_l2": (
                None
                if target_norm == 0.0
                else float(np.linalg.norm(residual) / target_norm)
            ),
            "cosine_with_target": cosine,
            "maximum_absolute_control": float(np.max(np.abs(control), initial=0.0)),
            "representable_squared_fraction": (
                None
                if target_norm == 0.0
                else float(np.linalg.norm(projection) ** 2 / target_norm**2)
            ),
            "mode_filter_factors": (
                self.eigenvalues / (self.eigenvalues + alpha)
            ).tolist(),
            "native_segment_rms": self._native_segment_rms(
                self.apply_B(control, packed=False)
            ),
        }
        if self.observation_size % len(CORE4_FIELDS) == 0:
            field_size = self.observation_size // len(CORE4_FIELDS)
            diagnostics["fieldwise_standardized_rms"] = {
                field: {
                    "target": float(
                        np.sqrt(
                            np.mean(
                                direction[i * field_size : (i + 1) * field_size]
                                ** 2
                            )
                        )
                    ),
                    "predicted": float(
                        np.sqrt(
                            np.mean(
                                prediction[i * field_size : (i + 1) * field_size]
                                ** 2
                            )
                        )
                    ),
                    "residual": float(
                        np.sqrt(
                            np.mean(
                                residual[i * field_size : (i + 1) * field_size]
                                ** 2
                            )
                        )
                    ),
                }
                for i, field in enumerate(CORE4_FIELDS)
            }
        return LiftSolution(
            control=np.asarray(control),
            native_increment=np.asarray(native),
            predicted_observation=np.asarray(prediction),
            diagnostics=diagnostics,
        )

    def _native_segment_rms(self, compact: np.ndarray) -> dict[str, float]:
        return {
            segment.name: float(
                np.sqrt(
                    np.mean(
                        compact[segment.compact_start : segment.compact_stop] ** 2
                    )
                )
            )
            for segment in self.layout.segments
        }

    def save(self, path: Path | str, *, metadata: dict[str, Any] | None = None) -> None:
        """Serialize the explicit factors without pickle-dependent arrays."""

        destination = Path(path)
        segment_metadata = [
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
        ]
        descriptor = {
            "schema_version": 1,
            "sample_count": self.sample_count,
            "observation_variance_trace": self.observation_variance_trace,
            "packed_size": self.layout.packed_size,
            "manifest_schema_version": self.layout.manifest_schema_version,
            "segments": segment_metadata,
            "metadata": metadata or {},
        }
        np.savez(
            destination,
            B=np.asarray(self.B, dtype=np.float64),
            G=np.asarray(self.G, dtype=np.float64),
            eigenvalues=np.asarray(self.eigenvalues, dtype=np.float64),
            native_mean=np.asarray(self.native_mean, dtype=np.float64),
            native_scale=np.asarray(self.native_scale, dtype=np.float64),
            observation_mean=np.asarray(self.observation_mean, dtype=np.float64),
            descriptor_json=np.asarray(json.dumps(descriptor, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path | str) -> tuple[BalancedControlMap, dict[str, Any]]:
        """Load a map saved by :meth:`save` with ``allow_pickle=False``."""

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
                eigenvalues=np.asarray(archive["eigenvalues"], dtype=np.float64),
                native_mean=np.asarray(archive["native_mean"], dtype=np.float64),
                native_scale=np.asarray(archive["native_scale"], dtype=np.float64),
                observation_mean=np.asarray(
                    archive["observation_mean"], dtype=np.float64
                ),
                layout=layout,
                sample_count=int(descriptor["sample_count"]),
                observation_variance_trace=float(
                    descriptor["observation_variance_trace"]
                ),
            )
        result.validate()
        return result, dict(descriptor.get("metadata", {}))

    def validate(self) -> None:
        """Validate dimensions, finiteness, and the paired EOF algebra."""

        if self.B.ndim != 2 or self.G.ndim != 2 or self.B.shape[1] != self.G.shape[1]:
            raise ValueError("B and G must be rank-matched matrices")
        if self.B.shape[0] != self.layout.compact_size:
            raise ValueError("B row count disagrees with the independent-control mask")
        if self.eigenvalues.shape != (self.rank,):
            raise ValueError("eigenvalue count disagrees with map rank")
        if self.native_mean.shape != (self.layout.compact_size,):
            raise ValueError("native mean shape disagrees with B")
        if self.native_scale.shape != self.native_mean.shape:
            raise ValueError("native scale shape disagrees with B")
        if self.observation_mean.shape != (self.observation_size,):
            raise ValueError("observation mean shape disagrees with G")
        arrays = (
            self.B,
            self.G,
            self.eigenvalues,
            self.native_mean,
            self.native_scale,
            self.observation_mean,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("balanced-map artifact contains nonfinite values")
        if np.any(self.eigenvalues <= 0.0) or np.any(np.diff(self.eigenvalues) > 0.0):
            raise ValueError("eigenvalues must be positive and nonincreasing")
        gram_error = np.max(
            np.abs(self.G.T @ self.G - np.diag(self.eigenvalues)), initial=0.0
        )
        tolerance = 5.0e-10 * max(1.0, float(self.eigenvalues[0]))
        if gram_error > tolerance:
            raise ValueError(f"G columns violate the EOF Gram identity: {gram_error}")


def fit_balanced_control_map(
    native_samples: np.ndarray,
    standardized_observation_samples: np.ndarray,
    *,
    layout: IndependentControlLayout,
    rank: int,
    relative_eigenvalue_floor: float = 1.0e-10,
) -> BalancedControlMap:
    """Fit paired covariance factors from authentic, aligned samples."""

    native = np.asarray(native_samples, dtype=np.float64)
    observations = np.asarray(standardized_observation_samples, dtype=np.float64)
    if native.ndim != 2 or observations.ndim != 2:
        raise ValueError("native and observation samples must be matrices")
    if native.shape[0] != observations.shape[0] or native.shape[0] < 2:
        raise ValueError("paired sample matrices must have the same n >= 2")
    if native.shape[1] != layout.compact_size:
        raise ValueError("native sample width disagrees with the control layout")
    if not np.isfinite(native).all() or not np.isfinite(observations).all():
        raise ValueError("paired samples contain nonfinite values")
    if not isinstance(rank, int) or rank <= 0 or rank >= native.shape[0]:
        raise ValueError("rank must be positive and smaller than sample count")
    if not math.isfinite(relative_eigenvalue_floor) or not (
        0.0 <= relative_eigenvalue_floor < 1.0
    ):
        raise ValueError("relative_eigenvalue_floor must lie in [0, 1)")

    native_mean = np.mean(native, axis=0, dtype=np.float64)
    observation_mean = np.mean(observations, axis=0, dtype=np.float64)
    native_centered = native - native_mean
    observation_centered = observations - observation_mean
    denominator = native.shape[0] - 1
    sample_gram = observation_centered @ observation_centered.T / denominator
    sample_gram = 0.5 * (sample_gram + sample_gram.T)
    eigenvalues, sample_vectors = scipy.linalg.eigh(
        sample_gram,
        subset_by_index=(native.shape[0] - rank, native.shape[0] - 1),
        driver="evr",
        check_finite=True,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    sample_vectors = np.asarray(sample_vectors[:, order], dtype=np.float64)
    floor = relative_eigenvalue_floor * eigenvalues[0]
    if eigenvalues[-1] <= floor:
        raise ValueError(
            "requested rank reaches an unresolved observation mode: "
            f"lambda_min={eigenvalues[-1]:.6g}, floor={floor:.6g}"
        )
    normalization = math.sqrt(denominator)
    B = native_centered.T @ sample_vectors / normalization
    G = observation_centered.T @ sample_vectors / normalization

    # Fix the otherwise arbitrary sign using the observable mode, not a mixed-unit
    # native coordinate.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(G[:, column])))
        if G[pivot, column] < 0.0:
            B[:, column] *= -1.0
            G[:, column] *= -1.0
    # Use the realized Gram values so serialization and transpose diagnostics are
    # tied to the explicitly stored G rather than to the eigensolver residual.
    realized_eigenvalues = np.diag(G.T @ G).copy()
    model = BalancedControlMap(
        B=np.asarray(B),
        G=np.asarray(G),
        eigenvalues=np.asarray(realized_eigenvalues),
        native_mean=np.asarray(native_mean),
        native_scale=np.std(native_centered, axis=0, ddof=1),
        observation_mean=np.asarray(observation_mean),
        layout=layout,
        sample_count=native.shape[0],
        observation_variance_trace=float(
            np.sum(observation_centered**2) / denominator
        ),
    )
    model.validate()
    return model


@dataclass(frozen=True)
class PairedReconstruction:
    """Held-out reconstruction arrays and compact summary."""

    controls: np.ndarray
    predicted_native_anomalies: np.ndarray
    predicted_observation_anomalies: np.ndarray
    summary: dict[str, Any]


def paired_reconstruction_diagnostics(
    model: BalancedControlMap,
    native_samples: np.ndarray,
    standardized_observation_samples: np.ndarray,
    *,
    ridge_fraction: float = 1.0e-3,
) -> PairedReconstruction:
    """Evaluate paired samples not used to fit ``model`` when possible."""

    native = np.asarray(native_samples, dtype=np.float64)
    observations = np.asarray(standardized_observation_samples, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != model.layout.compact_size:
        raise ValueError("native evaluation samples have the wrong shape")
    if observations.shape != (native.shape[0], model.observation_size):
        raise ValueError("observation evaluation samples have the wrong shape")
    if not np.isfinite(native).all() or not np.isfinite(observations).all():
        raise ValueError("evaluation samples contain nonfinite values")
    if not math.isfinite(ridge_fraction) or ridge_fraction < 0.0:
        raise ValueError("ridge_fraction must be finite and nonnegative")
    native_anomalies = native - model.native_mean
    observation_anomalies = observations - model.observation_mean
    gram = model.G.T @ model.G
    alpha = ridge_fraction * model.eigenvalues[0]
    controls = scipy.linalg.solve(
        gram + alpha * np.eye(model.rank),
        model.G.T @ observation_anomalies.T,
        assume_a="pos",
        check_finite=True,
    ).T
    observation_prediction = model.apply_G(controls)
    native_prediction = model.apply_B(controls, packed=False)
    observation_residual = observation_prediction - observation_anomalies
    observation_norm = np.linalg.norm(observation_anomalies, axis=1)
    prediction_norm = np.linalg.norm(observation_prediction, axis=1)
    denominator = observation_norm * prediction_norm
    cosine = np.divide(
        np.sum(observation_prediction * observation_anomalies, axis=1),
        denominator,
        out=np.full(native.shape[0], np.nan),
        where=denominator > 0.0,
    )
    observation_rms = np.sqrt(np.mean(observation_residual**2, axis=1))
    action = np.linalg.norm(controls, axis=1)
    segment_summary: dict[str, Any] = {}
    for segment in model.layout.segments:
        block = slice(segment.compact_start, segment.compact_stop)
        scale = model.native_scale[block]
        positive = scale > max(float(np.max(scale, initial=0.0)) * 1.0e-12, 0.0)
        if not np.any(positive):
            segment_summary[segment.name] = {
                "variable_coordinate_count": 0,
                "median_empirical_sigma_normalized_rms": None,
                "p90_empirical_sigma_normalized_rms": None,
            }
            continue
        normalized = (
            native_prediction[:, block][:, positive]
            - native_anomalies[:, block][:, positive]
        ) / scale[positive]
        rms = np.sqrt(np.mean(normalized**2, axis=1))
        segment_summary[segment.name] = {
            "variable_coordinate_count": int(np.count_nonzero(positive)),
            "median_empirical_sigma_normalized_rms": float(np.median(rms)),
            "p90_empirical_sigma_normalized_rms": float(np.quantile(rms, 0.9)),
        }
    summary = {
        "sample_count": int(native.shape[0]),
        "ridge_fraction_of_leading_eigenvalue": float(ridge_fraction),
        "median_observation_standardized_rms_residual": float(
            np.median(observation_rms)
        ),
        "p90_observation_standardized_rms_residual": float(
            np.quantile(observation_rms, 0.9)
        ),
        "median_observation_cosine": float(np.nanmedian(cosine)),
        "p10_observation_cosine": float(np.nanquantile(cosine, 0.1)),
        "median_control_l2": float(np.median(action)),
        "maximum_control_l2": float(np.max(action, initial=0.0)),
        "native_reconstruction_by_segment": segment_summary,
    }
    return PairedReconstruction(
        controls=np.asarray(controls),
        predicted_native_anomalies=np.asarray(native_prediction),
        predicted_observation_anomalies=np.asarray(observation_prediction),
        summary=summary,
    )
