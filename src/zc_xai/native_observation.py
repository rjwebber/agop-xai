"""Time-aligned observation of explicit Zebiak--Cane boundary states.

The fresh ZC stream is written inside ``SSTA``: SST has already advanced, but
the coarse thermocline and current diagnostics still belong to the preceding
coupled-step boundary.  Consequently one fresh input is an observation of a
*pair* of explicit states, not a slice of either state alone.

This module implements that selection, the two annual-phase coordinates, the
frozen training standardization, its tangent action, and its exact transpose.
It deliberately does not implement a balanced lift from observations to an
independent native-state control.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .data import FRESH_INPUT_PROFILES, Standardizer, ZCData
from .io import sha256_file
from .standardization import RecordedStandardizer, load_recorded_standardizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_MANIFEST = (
    REPOSITORY_ROOT / "adjoint" / "fortran_kernel" / "state_manifest.json"
)

CORE4_FIELDS = FRESH_INPUT_PROFILES["core4"]
CORE4_NATIVE_FIELDS = ("TO", "H1", "U1", "V1")
GRID_SHAPE = (20, 27)
SPATIAL_FEATURES_PER_FIELD = math.prod(GRID_SHAPE)
CORE4_FEATURES = len(CORE4_FIELDS) * SPATIAL_FEATURES_PER_FIELD + 2
PHASE_ORIGIN_MONTHS = 0.5
PHASE_PERIOD_MONTHS = 12.0
PHASE_ANGULAR_FREQUENCY = 2.0 * np.pi / PHASE_PERIOD_MONTHS
EFFECTIVE_DTD_MONTHS = np.float32(1.0 / 3.0)

# The compact writer uses ((FIELD(II,JJ),JJ=6,32),II=25,6,-1).
_LATITUDE_SELECTION = slice(24, 4, -1)
_LONGITUDE_SELECTION = slice(5, 32)
_NINO3_NATIVE_LATITUDES = range(12, 18)
_NINO3_NATIVE_LONGITUDES = range(20, 31)


@dataclass(frozen=True)
class StateSegment:
    """One validated segment of the packed real state."""

    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    order: str


@dataclass(frozen=True)
class NativeStateLayout:
    """Typed lengths and scientific slices bound to one state manifest."""

    manifest_path: Path
    manifest_sha256: str
    real32_length: int
    complex64_length: int
    real64_length: int
    integer_length: int
    passive_time_length: int
    real_segments: Mapping[str, StateSegment]
    integer_segments: Mapping[str, StateSegment]
    passive_time_segments: Mapping[str, StateSegment]

    @property
    def file_size_bytes(self) -> int:
        return (
            4 * self.real32_length
            + 8 * self.complex64_length
            + 8 * self.real64_length
            + 4 * self.integer_length
            + 4 * self.passive_time_length
        )


@dataclass(frozen=True)
class PackedZCState:
    """One explicit typed state emitted by ``zc_kernel_replay``."""

    real32: np.ndarray
    complex64: np.ndarray
    real64: np.ndarray
    integers: np.ndarray
    passive_time: np.ndarray


@dataclass(frozen=True)
class Core4ObservationTranspose:
    """Transpose result at the two state boundaries and passive clocks.

    ``previous_real32`` and ``post_step_real32`` are the scientific-state
    result used by default.  Clock adjoints are separate because the kernel
    manifest classifies time as passive: a caller must not inject
    ``post_step_passive_time`` as a native-state perturbation merely because it
    is useful for checking the complete mathematical transpose.
    """

    previous_real32: np.ndarray
    post_step_real32: np.ndarray
    previous_passive_time: np.ndarray
    post_step_passive_time: np.ndarray

    @property
    def phase_td_cotangent(self) -> float:
        """Cotangent of the post-step absolute model time ``TD``."""

        return float(self.post_step_passive_time[0])

    def scientific_control_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the real-state transpose with passive phase time held fixed."""

        return self.previous_real32, self.post_step_real32

    def compose_post_step_pullback(
        self, post_step_pullback: np.ndarray
    ) -> np.ndarray:
        """Combine a dynamic pullback with the direct previous-state seed.

        If ``post_step = F(previous)``, the caller obtains
        ``post_step_pullback = F'(previous).T @ post_step_real32`` from the
        matching model reverse.  This method adds the observation terms that
        act directly on the previous boundary.
        """

        pullback = np.asarray(post_step_pullback, dtype=np.float64)
        if pullback.shape != self.previous_real32.shape:
            raise ValueError("Post-step pullback has the wrong packed-state shape.")
        if not np.isfinite(pullback).all():
            raise ValueError("Post-step pullback must be finite.")
        return self.previous_real32 + pullback


def _positive_length(specification: Any, name: str) -> int:
    if not isinstance(specification, dict):
        raise ValueError(f"State manifest arrays.{name} must be an object.")
    value = specification.get("length")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"State manifest arrays.{name}.length is invalid.")
    return value


def _segments(specification: dict[str, Any], name: str) -> Mapping[str, StateSegment]:
    raw_segments = specification.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"State manifest arrays.{name}.segments is invalid.")
    expected_start = 0
    result: dict[str, StateSegment] = {}
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError(f"State manifest arrays.{name} has a non-object segment.")
        segment_name = raw.get("name")
        start = raw.get("start")
        stop = raw.get("stop")
        shape = raw.get("shape")
        order = raw.get("order")
        if (
            not isinstance(segment_name, str)
            or not segment_name
            or segment_name in result
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
            or order != "F"
        ):
            raise ValueError(f"State manifest arrays.{name} has an invalid segment.")
        dimensions = tuple(shape)
        if (
            start != expected_start
            or stop <= start
            or math.prod(dimensions) != stop - start
        ):
            raise ValueError(
                f"State manifest segment {segment_name!r} is not contiguous or "
                "does not match its shape."
            )
        result[segment_name] = StateSegment(
            name=segment_name,
            start=start,
            stop=stop,
            shape=dimensions,
            order=order,
        )
        expected_start = stop
    if expected_start != specification["length"]:
        raise ValueError(
            f"State manifest arrays.{name} segments do not fill its length."
        )
    return MappingProxyType(result)


def load_native_state_layout(
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
) -> NativeStateLayout:
    """Load and strictly validate the explicit-state layout used here."""

    path = manifest_path.expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read a valid ZC state manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("ZC state manifest root must be an object.")
    if manifest.get("schema_version") != 2:
        raise ValueError("ZC state manifest schema_version must equal 2.")
    supported = manifest.get("supported_configuration")
    canonical_configuration = {
        "NX": 78,
        "NY": 115,
        "NXP": 79,
        "NYP": 116,
        "NSEG": 1,
        "NS": 2,
        "NATM": 2,
        "NIC": 0,
        "westerly_wind_bursts": False,
    }
    if supported != canonical_configuration:
        raise ValueError("ZC state manifest configuration is not canonical.")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("ZC state manifest arrays contract is missing.")
    required = ("real32", "complex64", "real64", "integer", "passive_time_real32")
    if any(name not in arrays for name in required):
        raise ValueError("ZC state manifest is missing a typed state array.")

    lengths = {name: _positive_length(arrays[name], name) for name in required}
    all_segments = {name: _segments(arrays[name], name) for name in required}
    real_segments = all_segments["real32"]
    integer_segments = all_segments["integer"]
    time_segments = all_segments["passive_time_real32"]

    if lengths != {
        "real32": 59_148,
        "complex64": 5_280,
        "real64": 1,
        "integer": 4,
        "passive_time_real32": 2,
    }:
        raise ValueError(
            "ZC state manifest typed lengths are not the canonical layout."
        )
    for field in CORE4_NATIVE_FIELDS:
        segment = real_segments.get(field)
        if segment is None or segment.shape != (30, 34):
            raise ValueError(f"ZC state manifest has no canonical {field} segment.")
    if integer_segments.get("NT") is None or time_segments.get("TD") is None:
        raise ValueError("ZC state manifest lacks canonical NT or TD coordinates.")
    named = manifest.get("named_offsets_for_target_and_attribution")
    if not isinstance(named, dict):
        raise ValueError("ZC state manifest lacks named scientific offsets.")
    for field in CORE4_NATIVE_FIELDS:
        segment = real_segments[field]
        expected_slice = f"{segment.start}:{segment.stop}"
        entry = named.get(field)
        if not isinstance(entry, dict) or entry.get("python_slice") != expected_slice:
            raise ValueError(
                f"ZC state manifest named offset for {field} is inconsistent."
            )

    return NativeStateLayout(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        real32_length=lengths["real32"],
        complex64_length=lengths["complex64"],
        real64_length=lengths["real64"],
        integer_length=lengths["integer"],
        passive_time_length=lengths["passive_time_real32"],
        real_segments=real_segments,
        integer_segments=integer_segments,
        passive_time_segments=time_segments,
    )


def _byte_prefix(byte_order: str) -> str:
    try:
        return {"little": "<", "big": ">", "native": "="}[byte_order]
    except KeyError as error:
        raise ValueError("byte_order must be 'little', 'big', or 'native'.") from error


def read_packed_zc_state(
    path: Path,
    *,
    layout: NativeStateLayout | None = None,
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
    byte_order: str = "little",
) -> PackedZCState:
    """Read one exact headerless typed state written by the kernel driver."""

    resolved_layout = layout or load_native_state_layout(manifest_path)
    state_path = path.expanduser().resolve()
    payload = state_path.read_bytes()
    if len(payload) != resolved_layout.file_size_bytes:
        raise ValueError(
            f"Packed ZC state has {len(payload):,} bytes; expected exactly "
            f"{resolved_layout.file_size_bytes:,}: {state_path}"
        )
    prefix = _byte_prefix(byte_order)
    offset = 0

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        parsed_dtype = np.dtype(f"{prefix}{dtype}")
        result = np.frombuffer(payload, dtype=parsed_dtype, count=count, offset=offset)
        offset += parsed_dtype.itemsize * count
        return result.astype(np.dtype(f"={dtype}"), copy=True)

    state = PackedZCState(
        real32=take("f4", resolved_layout.real32_length),
        complex64=take("c8", resolved_layout.complex64_length),
        real64=take("f8", resolved_layout.real64_length),
        integers=take("i4", resolved_layout.integer_length),
        passive_time=take("f4", resolved_layout.passive_time_length),
    )
    if offset != len(payload):  # defensive check against a changed dtype contract
        raise RuntimeError("Packed ZC state reader did not consume the complete file.")
    for label, values in (
        ("real32", state.real32),
        ("complex64", state.complex64),
        ("real64", state.real64),
        ("passive_time", state.passive_time),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                f"Packed ZC state {label} values are nonfinite: {state_path}"
            )
    return state


def _validated_state_arrays(
    state: PackedZCState, layout: NativeStateLayout
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    specifications = (
        (state.real32, layout.real32_length, np.float32, "real32"),
        (state.complex64, layout.complex64_length, np.complex64, "complex64"),
        (state.real64, layout.real64_length, np.float64, "real64"),
        (state.integers, layout.integer_length, np.int32, "integers"),
        (state.passive_time, layout.passive_time_length, np.float32, "passive_time"),
    )
    result: list[np.ndarray] = []
    for raw, length, dtype, label in specifications:
        values = np.asarray(raw)
        if values.shape != (length,):
            raise ValueError(f"Packed ZC state {label} has the wrong shape.")
        if label != "integers" and not np.isfinite(values).all():
            raise ValueError(f"Packed ZC state {label} values must be finite.")
        if label == "integers" and values.dtype.kind not in {"i", "u"}:
            raise ValueError("Packed ZC state integers must use an integer dtype.")
        result.append(np.asarray(values, dtype=dtype))
    return tuple(result)  # type: ignore[return-value]


def write_packed_zc_state(
    path: Path,
    state: PackedZCState,
    *,
    layout: NativeStateLayout | None = None,
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
    byte_order: str = "little",
    overwrite: bool = False,
) -> Path:
    """Atomically write one exact kernel-compatible typed state stream."""

    resolved_layout = layout or load_native_state_layout(manifest_path)
    prefix = _byte_prefix(byte_order)
    arrays = _validated_state_arrays(state, resolved_layout)
    requested = path.expanduser()
    parent = requested.parent.resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"Packed-state output directory is missing: {parent}")
    target = parent / requested.name
    if target.is_symlink():
        raise ValueError(f"Packed-state output may not be a symbolic link: {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Packed-state output already exists: {target}")
    if target.exists() and not target.is_file():
        raise ValueError(f"Packed-state output is not a regular file: {target}")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            for values, dtype in zip(
                arrays, ("f4", "c8", "f8", "i4", "f4"), strict=True
            ):
                stream.write(np.asarray(values, dtype=f"{prefix}{dtype}").tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != resolved_layout.file_size_bytes:
            raise RuntimeError("Packed-state writer produced an invalid byte count.")
        if not overwrite:
            # Linking within one directory is an atomic create-if-absent commit.
            os.link(temporary, target, follow_symlinks=False)
            temporary.unlink()
        else:
            if target.is_symlink():
                raise ValueError(
                    f"Packed-state output became a symbolic link: {target}"
                )
            os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def core_field_from_real_state(
    real_state: np.ndarray,
    name: str,
    *,
    layout: NativeStateLayout | None = None,
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
) -> np.ndarray:
    """Select one public 20 x 27 core field from a packed real state."""

    resolved_layout = layout or load_native_state_layout(manifest_path)
    if name not in CORE4_NATIVE_FIELDS:
        raise ValueError(f"Unknown native core4 field {name!r}.")
    values = np.asarray(real_state)
    if values.shape != (resolved_layout.real32_length,):
        raise ValueError("Packed real state must have shape (59148,).")
    segment = resolved_layout.real_segments[name]
    native = values[segment.start : segment.stop].reshape(segment.shape, order="F")
    selected = native[_LATITUDE_SELECTION, _LONGITUDE_SELECTION]
    if selected.shape != GRID_SHAPE:
        raise RuntimeError("Canonical native-grid selection has changed shape.")
    return np.array(selected, copy=True)


def nino3_from_real_state(
    real_state: np.ndarray,
    *,
    layout: NativeStateLayout | None = None,
    manifest_path: Path = DEFAULT_STATE_MANIFEST,
) -> np.float32:
    """Evaluate the canonical 66-cell Niño-3 head on packed-state ``TO``."""

    resolved_layout = layout or load_native_state_layout(manifest_path)
    values = np.asarray(real_state)
    if values.shape != (resolved_layout.real32_length,):
        raise ValueError("Packed real state must have shape (59148,).")
    segment = resolved_layout.real_segments["TO"]
    native = values[segment.start : segment.stop].reshape(segment.shape, order="F")
    total = 0.0
    # Match the final3 scalar head: longitude outermost, then native latitude.
    for longitude in _NINO3_NATIVE_LONGITUDES:
        for latitude in _NINO3_NATIVE_LATITUDES:
            total += float(native[latitude, longitude])
    return np.float32(total / 66.0)


def canonical_model_time(nt: int) -> np.float32:
    """Reproduce the fresh generator's default-real ``TD`` arithmetic."""

    if isinstance(nt, bool) or not isinstance(nt, (int, np.integer)):
        raise TypeError("NT must be an integer.")
    return np.float32(
        np.float32(PHASE_ORIGIN_MONTHS) + np.float32(nt) * EFFECTIVE_DTD_MONTHS
    )


def annual_phase(native_time_months: float) -> np.ndarray:
    """Return release-compatible sine/cosine phase in float32."""

    time = float(native_time_months)
    if not math.isfinite(time):
        raise ValueError("Native time must be finite.")
    angle = PHASE_ANGULAR_FREQUENCY * np.mod(
        time - PHASE_ORIGIN_MONTHS, PHASE_PERIOD_MONTHS
    )
    return np.asarray((np.sin(angle), np.cos(angle)), dtype=np.float32)


def annual_phase_tangent(native_time_months: float) -> np.ndarray:
    """Return the smooth periodic phase derivative with respect to ``TD``."""

    time = float(native_time_months)
    if not math.isfinite(time):
        raise ValueError("Native time must be finite.")
    angle = PHASE_ANGULAR_FREQUENCY * np.mod(
        time - PHASE_ORIGIN_MONTHS, PHASE_PERIOD_MONTHS
    )
    return PHASE_ANGULAR_FREQUENCY * np.asarray(
        (np.cos(angle), -np.sin(angle)), dtype=np.float64
    )


class FrozenCore4ObservationChain:
    """Aligned core4+phase observation with frozen feature standardization."""

    def __init__(
        self,
        standardizer: Standardizer,
        *,
        manifest_path: Path = DEFAULT_STATE_MANIFEST,
        recorded_standardizer: RecordedStandardizer | None = None,
    ) -> None:
        self.layout = load_native_state_layout(manifest_path)
        mean = np.array(standardizer.mean, dtype=np.float32, copy=True)
        scale = np.array(standardizer.scale, dtype=np.float32, copy=True)
        if mean.shape != (CORE4_FEATURES,) or scale.shape != (CORE4_FEATURES,):
            raise ValueError("Core4 observation normalization must have shape (2162,).")
        if not np.isfinite(mean).all():
            raise ValueError("Core4 observation normalization mean is nonfinite.")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("Core4 observation normalization scale must be positive.")
        if not math.isfinite(standardizer.scale_floor) or standardizer.scale_floor <= 0:
            raise ValueError("Core4 observation normalization floor is invalid.")
        if standardizer.count <= 0:
            raise ValueError("Core4 observation normalization count is invalid.")
        mean.setflags(write=False)
        scale.setflags(write=False)
        self.standardizer = Standardizer(
            mean=mean,
            scale=scale,
            scale_floor=float(standardizer.scale_floor),
            count=int(standardizer.count),
        )
        self.recorded_standardizer = recorded_standardizer

    @classmethod
    def from_recorded_artifact(
        cls,
        data_dir: Path,
        artifact_root: Path,
        artifact_directory: str,
        *,
        manifest_path: Path = DEFAULT_STATE_MANIFEST,
        expected_normalization_sha256: str | None = None,
        verify_data_checksums: bool = False,
    ) -> FrozenCore4ObservationChain:
        """Bind the chain to a completed fresh core4 normalization artifact."""

        data = ZCData(
            data_dir,
            input_profile="core4",
            verify_checksums=verify_data_checksums,
        )
        if data.field_names != CORE4_FIELDS or data.input_shape != (CORE4_FEATURES,):
            raise ValueError("Processed data do not use the canonical core4 profile.")
        recorded = load_recorded_standardizer(
            artifact_root,
            artifact_directory,
            expected_data_metadata_sha256=data.metadata_sha256,
            expected_input_shape=(CORE4_FEATURES,),
            expected_normalization_sha256=expected_normalization_sha256,
        )
        if recorded.spec.get("input_profile") != "core4":
            raise ValueError("Recorded normalization is not for input_profile='core4'.")
        return cls(
            recorded.standardizer,
            manifest_path=manifest_path,
            recorded_standardizer=recorded,
        )

    def read_state(self, path: Path, *, byte_order: str = "little") -> PackedZCState:
        """Read a state against this chain's already validated manifest."""

        return read_packed_zc_state(path, layout=self.layout, byte_order=byte_order)

    def write_state(
        self,
        path: Path,
        state: PackedZCState,
        *,
        byte_order: str = "little",
        overwrite: bool = False,
    ) -> Path:
        """Write a state against this chain's already validated manifest."""

        return write_packed_zc_state(
            path,
            state,
            layout=self.layout,
            byte_order=byte_order,
            overwrite=overwrite,
        )

    def native_time_months(self, state: PackedZCState) -> float:
        """Return the state's scalar absolute model time ``TD``."""

        self._validate_state(state, "state")
        return float(self._coordinate(state, "TD", kind="time"))

    def _validate_state(self, state: PackedZCState, label: str) -> None:
        expected = (
            (state.real32, self.layout.real32_length, "real32"),
            (state.complex64, self.layout.complex64_length, "complex64"),
            (state.real64, self.layout.real64_length, "real64"),
            (state.integers, self.layout.integer_length, "integers"),
            (state.passive_time, self.layout.passive_time_length, "passive_time"),
        )
        for values, length, component in expected:
            if np.asarray(values).shape != (length,):
                raise ValueError(f"{label} {component} has the wrong shape.")

    def _coordinate(self, state: PackedZCState, name: str, *, kind: str) -> Any:
        if kind == "integer":
            segment = self.layout.integer_segments[name]
            values = state.integers
        else:
            segment = self.layout.passive_time_segments[name]
            values = state.passive_time
        if segment.stop - segment.start != 1:
            raise RuntimeError(f"State coordinate {name} is not scalar.")
        return values[segment.start]

    def _validate_transition(
        self, previous: PackedZCState, post_step: PackedZCState
    ) -> None:
        self._validate_state(previous, "previous state")
        self._validate_state(post_step, "post-step state")
        previous_nt = int(self._coordinate(previous, "NT", kind="integer"))
        post_nt = int(self._coordinate(post_step, "NT", kind="integer"))
        if post_nt != previous_nt + 1:
            raise ValueError(
                "Fresh observation alignment requires consecutive boundary states."
            )
        for state, nt, label in (
            (previous, previous_nt, "previous"),
            (post_step, post_nt, "post-step"),
        ):
            td = np.float32(self._coordinate(state, "TD", kind="time"))
            expected_td = canonical_model_time(nt)
            if td.view(np.uint32) != expected_td.view(np.uint32):
                raise ValueError(
                    f"{label} state TD does not match canonical fresh-run NT timing."
                )

    def _field(self, real_state: np.ndarray, name: str) -> np.ndarray:
        return core_field_from_real_state(
            real_state,
            name,
            layout=self.layout,
        )

    def raw_features(
        self, previous: PackedZCState, post_step: PackedZCState
    ) -> np.ndarray:
        """Return the unstandardized fresh row at the SSTA output boundary."""

        self._validate_transition(previous, post_step)
        values = np.empty(CORE4_FEATURES, dtype=np.float32)
        spatial = values[:-2].reshape(len(CORE4_FIELDS), *GRID_SHAPE)
        spatial[0] = self._field(post_step.real32, "TO")
        for index, name in enumerate(("H1", "U1", "V1"), start=1):
            spatial[index] = self._field(previous.real32, name)
        td = float(self._coordinate(post_step, "TD", kind="time"))
        values[-2:] = annual_phase(td)
        return values

    def forward(
        self, previous: PackedZCState, post_step: PackedZCState
    ) -> np.ndarray:
        """Return the bit-compatible frozen-standardized 2,162-vector."""

        values = self.raw_features(previous, post_step)
        np.subtract(values, self.standardizer.mean, out=values)
        np.divide(values, self.standardizer.scale, out=values)
        return values

    def tangent(
        self,
        previous_real32_tangent: np.ndarray,
        post_step_real32_tangent: np.ndarray,
        *,
        post_step_native_time_months: float,
        post_step_td_tangent: float = 0.0,
    ) -> np.ndarray:
        """Apply the observation/standardization Jacobian.

        The default holds annual phase fixed, matching the scientific-control
        convention.  Supplying ``post_step_td_tangent`` includes the smooth
        periodic phase derivative for a full mathematical transpose check.
        """

        previous_tangent = np.asarray(previous_real32_tangent)
        post_tangent = np.asarray(post_step_real32_tangent)
        expected = (self.layout.real32_length,)
        if previous_tangent.shape != expected or post_tangent.shape != expected:
            raise ValueError("Packed real-state tangents must have shape (59148,).")
        if not np.isfinite(previous_tangent).all() or not np.isfinite(
            post_tangent
        ).all():
            raise ValueError("Packed real-state tangents must be finite.")
        if not math.isfinite(float(post_step_td_tangent)):
            raise ValueError("Post-step TD tangent must be finite.")

        raw = np.empty(CORE4_FEATURES, dtype=np.float64)
        spatial = raw[:-2].reshape(len(CORE4_FIELDS), *GRID_SHAPE)
        spatial[0] = self._field(post_tangent, "TO")
        for index, name in enumerate(("H1", "U1", "V1"), start=1):
            spatial[index] = self._field(previous_tangent, name)
        raw[-2:] = annual_phase_tangent(post_step_native_time_months) * float(
            post_step_td_tangent
        )
        return raw / self.standardizer.scale.astype(np.float64)

    def transpose(
        self,
        standardized_cotangent: np.ndarray,
        *,
        post_step_native_time_months: float,
    ) -> Core4ObservationTranspose:
        """Apply the exact transpose of :meth:`tangent`.

        The phase/TD term is returned only in the separate passive-time array.
        Calling :meth:`Core4ObservationTranspose.scientific_control_pair`
        therefore applies the default fixed-phase scientific convention.
        """

        cotangent = np.asarray(standardized_cotangent, dtype=np.float64)
        if cotangent.shape != (CORE4_FEATURES,):
            raise ValueError("Standardized cotangent must have shape (2162,).")
        if not np.isfinite(cotangent).all():
            raise ValueError("Standardized cotangent must be finite.")
        physical = cotangent / self.standardizer.scale.astype(np.float64)
        blocks = physical[:-2].reshape(len(CORE4_FIELDS), *GRID_SHAPE)
        previous = np.zeros(self.layout.real32_length, dtype=np.float64)
        post_step = np.zeros(self.layout.real32_length, dtype=np.float64)

        def scatter(target: np.ndarray, name: str, block: np.ndarray) -> None:
            segment = self.layout.real_segments[name]
            native = target[segment.start : segment.stop].reshape(
                segment.shape, order="F"
            )
            native[_LATITUDE_SELECTION, _LONGITUDE_SELECTION] += block

        scatter(post_step, "TO", blocks[0])
        for index, name in enumerate(("H1", "U1", "V1"), start=1):
            scatter(previous, name, blocks[index])

        previous_time = np.zeros(self.layout.passive_time_length, dtype=np.float64)
        post_time = np.zeros(self.layout.passive_time_length, dtype=np.float64)
        td_segment = self.layout.passive_time_segments["TD"]
        post_time[td_segment.start] = float(
            np.dot(
                physical[-2:],
                annual_phase_tangent(post_step_native_time_months),
            )
        )
        return Core4ObservationTranspose(
            previous_real32=previous,
            post_step_real32=post_step,
            previous_passive_time=previous_time,
            post_step_passive_time=post_time,
        )
