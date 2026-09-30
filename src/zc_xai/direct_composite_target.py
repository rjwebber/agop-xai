"""Event-equal composite directions for direct ZC sufficiency experiments.

The target follows the composite convention used by Figure 8: identify one
peak in every complete held-out ENSO episode, retain the strongest ten percent
of those peaks, and average the training-standardized core4 states ten months
before the retained peaks.  The passive annual-phase coordinates are removed
before the spatial direction is normalized.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agop_control import unit_fixed_phase_direction
from .composites import strongest_fraction, threshold_event_peaks
from .data import ZCData
from .direct_xai_target import DirectXAITarget
from .io import sha256_array
from .training import LoadedExperiment

COMPOSITE_METHOD = "Composite"
COMPOSITE_METHOD_KEY = "event_composite_top10"
EVENT_THRESHOLD_C = 1.0
TOP_PERCENT = 10.0
LEAD_MONTHS = 10


def _unit(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} has zero or nonfinite norm")
    return array / norm


def _event_kind(target_event_label: str) -> tuple[str, float]:
    if target_event_label == "extreme_el_nino":
        return "warm", 1.0
    if target_event_label == "extreme_la_nina":
        return "cold", -1.0
    raise ValueError(f"unknown target event: {target_event_label!r}")


def build_direct_composite_target(
    method: str,
    *,
    data: ZCData,
    experiment: LoadedExperiment,
    event_standardized: np.ndarray,
    target_event_label: str,
    agop_factor: Any = None,
    device: str = "cpu",
) -> DirectXAITarget:
    """Build the Figure 8 top/bottom-decile event-composite direction.

    The returned object has the same interface as an XAI target so it can use
    the already-audited native pooled-covariance controller.  Its release
    threshold remains the projection of the held-out extreme event, exactly as
    in the four XAI-method experiments; only the unit spatial direction changes.
    """

    del device
    if method != COMPOSITE_METHOD:
        raise ValueError(f"expected method {COMPOSITE_METHOD!r}, received {method!r}")
    if agop_factor is not None:
        raise ValueError("the composite target must not receive an AGOP factor")
    if data.n_phase_features != 2:
        raise ValueError("direct ZC controls require exactly two phase features")
    event = np.asarray(event_standardized, dtype=np.float64)
    if event.shape != data.input_shape or not np.isfinite(event).all():
        raise ValueError("event_standardized has the wrong shape or is nonfinite")

    kind, sign = _event_kind(target_event_label)
    split = data.fixed_supervised_split(LEAD_MONTHS)
    target_steps = np.asarray(split.test_inputs + split.lead_steps, dtype=np.int64)
    target_values = np.asarray(data.target[target_steps], dtype=np.float64)
    signed_target = sign * np.asarray(data.target, dtype=np.float64)
    all_peaks = threshold_event_peaks(
        sign * target_values,
        target_steps,
        threshold_c=EVENT_THRESHOLD_C,
    )
    selected_peaks = strongest_fraction(
        all_peaks,
        signed_target,
        percent=TOP_PERCENT,
    )
    selected_inputs = selected_peaks - split.lead_steps
    if not np.isin(selected_inputs, split.test_inputs).all():
        raise ValueError(
            "a selected composite precursor leaves the held-out test block"
        )

    checkpoint = next(
        (
            record
            for record in data.metadata["event_restart_checkpoints"]
            if record["label"] == target_event_label
        ),
        None,
    )
    if checkpoint is None:
        raise ValueError(f"missing restart checkpoint for {target_event_label}")
    extreme_target_step = int(checkpoint["target_index"])
    if extreme_target_step not in set(map(int, selected_peaks)):
        raise ValueError("the fixed extreme event is absent from its 10% composite")

    selected_physical = np.asarray(data.load_inputs(selected_inputs), dtype=np.float64)
    selected_states = (
        selected_physical - experiment.standardizer.mean
    ) / experiment.standardizer.scale
    # Figure 8 publishes its composite fields as float32.  Round at the same
    # point so this controller uses exactly those displayed composite values.
    composite_published = selected_states.mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    composite = np.asarray(composite_published, dtype=np.float64)
    if composite.shape != data.input_shape or not np.isfinite(composite).all():
        raise ValueError("the event composite has the wrong shape or is nonfinite")
    full_unit = _unit(composite, label="event composite")
    fixed_phase = unit_fixed_phase_direction(full_unit, phase_features=2)
    raw_spatial = np.asarray(fixed_phase[:-2], dtype=np.float64)
    raw_projection = float(raw_spatial @ event[:-2])
    if not np.isfinite(raw_projection) or abs(raw_projection) <= np.finfo(float).eps:
        raise ValueError("the extreme event has zero composite-direction projection")
    orientation = 1.0 if raw_projection > 0.0 else -1.0
    oriented = orientation * raw_spatial
    oriented_projection = orientation * raw_projection

    selected_values = np.asarray(data.target[selected_peaks], dtype=np.float64)
    precursor_phases, phase_counts = np.unique(
        selected_inputs % data.steps_per_year,
        return_counts=True,
    )
    reference_arrays = {
        "composite_all_complete_event_peak_steps": all_peaks,
        "composite_selected_peak_steps_ranked": selected_peaks,
        "composite_selected_input_steps_ranked": selected_inputs,
        "composite_selected_peak_nino3_c_ranked": selected_values,
        "composite_raw_standardized": composite_published,
        "composite_precursor_phases": precursor_phases.astype(np.int64),
        "composite_precursor_phase_counts": phase_counts.astype(np.int64),
    }
    provenance: dict[str, Any] = {
        "key": COMPOSITE_METHOD_KEY,
        "label": "10% event composite",
        "parameters": {
            "definition": (
                "equal-event mean of standardized states 10 months before the "
                "strongest 10% of complete held-out ENSO episode peaks"
            ),
            "event_kind": kind,
            "candidate_population": (
                "target dates associated with 10-month predictors in the fixed "
                "1,000-year held-out test block"
            ),
            "episode_threshold_nino3_c": sign * EVENT_THRESHOLD_C,
            "episode_rule": (
                "one signed extremum per complete contiguous threshold episode; "
                "boundary-truncated episodes excluded; earliest ties retained"
            ),
            "ranking_rule": "strongest ceil(10 percent) of episode extrema",
            "event_weighting": "equal",
            "all_complete_episode_count": int(all_peaks.size),
            "selected_event_count": int(selected_peaks.size),
            "lead_months": LEAD_MONTHS,
            "standardization_population": (
                "all states in the fixed 10,000-year training block"
            ),
            "phase_matching": (
                "event locked, not restricted to the extreme event's exact phase"
            ),
        },
        "explanation_sha256": sha256_array(full_unit),
        "fixed_phase_explanation_sha256": sha256_array(fixed_phase),
        "raw_spatial_direction_sha256": sha256_array(raw_spatial),
        "oriented_spatial_direction_sha256": sha256_array(oriented),
        "event_standardized_sha256": sha256_array(event),
        "composite_raw_standardized_sha256": sha256_array(composite_published),
        "all_complete_event_peak_steps_sha256": sha256_array(all_peaks),
        "selected_event_peak_steps_sha256": sha256_array(selected_peaks),
        "phase_coordinate_policy": (
            "drop the final sine/cosine coordinates, then renormalize"
        ),
        "orientation_policy": (
            "multiply the fixed-phase composite direction by the sign of its "
            "projection on the corresponding held-out extreme event"
        ),
        "target_amplitude_policy": (
            "use the corresponding extreme event's projection, matching the four "
            "XAI-method sufficiency experiments"
        ),
        "orientation_multiplier": orientation,
        "raw_event_projection": raw_projection,
        "oriented_event_projection": oriented_projection,
    }
    return DirectXAITarget(
        method=COMPOSITE_METHOD,
        explanation=full_unit,
        fixed_phase_explanation=fixed_phase,
        raw_spatial_direction=raw_spatial,
        oriented_spatial_direction=oriented,
        raw_event_projection=raw_projection,
        oriented_event_projection=oriented_projection,
        orientation_multiplier=orientation,
        provenance=provenance,
        reference_arrays=reference_arrays,
    )
