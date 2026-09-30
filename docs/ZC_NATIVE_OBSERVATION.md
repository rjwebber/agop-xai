# Native-state to fresh-core4 observation chain

`zc_xai.native_observation` connects the explicit Zebiak--Cane boundary state
to the exact 2,162-coordinate input used by the fresh `zc-v3` core4 models. It
implements the observation and frozen normalization maps in both tangent and
transpose form. It does not alter or regenerate the Tapenade final3 sources or
the curated run23 evidence.

## Why two boundary states are required

The fresh output record is written near the end of `SSTA`. At that instruction
boundary, SST anomaly `TO` has advanced through the current SST update, while
the coarse-ocean fields `H1`, `U1`, and `V1` still describe the preceding
boundary. For consecutive explicit states `s0` and `s1`, the raw model input is

```text
x = [P TO(s1), P H1(s0), P U1(s0), P V1(s0), sin(theta), cos(theta)]
theta = 2 pi mod(TD(s1) - 0.5, 12) / 12
```

`P` is exactly the compact writer's grid selection: Fortran latitude indices
25 down to 6, longitude indices 6 through 32, then NumPy C-order flattening.
Thus the public field order is SST anomaly, thermocline depth, zonal ocean
current, and meridional ocean current, followed by the two scalar phase
coordinates. No phase value is duplicated over the grid.

The standardized input is `z = (x - mean) / scale`, using immutable float32
arrays from a completion-manifest-bound `normalization.npz`. The loader checks
`standardization_inputs` for fresh artifacts; it retains the older
`fit_inputs` convention for legacy artifacts.

## Tangent and transpose

For perturbations of the two packed real boundary states and post-step `TD`,
the tangent is

```text
dz = diag(1 / scale) [P dTO(s1), P dH1(s0), P dU1(s0),
                      P dV1(s0), phi'(TD) dTD]
phi'(TD) = (2 pi / 12) [cos(theta), -sin(theta)].
```

The transpose first divides a standardized cotangent by the same frozen scale,
then scatters the four spatial blocks into the exact manifest slices and grid
indices. The phase contribution is returned separately in
`post_step_passive_time[TD]`. This separation is intentional: the state
manifest classifies `TD` and `TY` as passive. The default scientific-control
pair contains only packed-real state cotangents and therefore holds annual
phase fixed. The separate `TD` value exists for a complete dot-product identity
and must not be treated as permission to actuate model time.

The sine/cosine phase is a smooth periodic function. The `mod` operation only
chooses its 12-month representative; at a representation wrap the composed
sine/cosine value and analytic derivative agree on both sides (apart from
ordinary floating-point rounding).

## API

```python
from pathlib import Path

from zc_xai.native_observation import FrozenCore4ObservationChain

chain = FrozenCore4ObservationChain.from_recorded_artifact(
    Path("data/processed/zc-v3"),
    Path("artifacts/zc-v3"),
    "models/core4/cnn/lead-10m/years-10000/seed-000042",
)
previous = chain.read_state(Path("kernel_initial_state.bin"))
post_step = chain.read_state(Path("kernel_final_state.bin"))

raw_input = chain.raw_features(previous, post_step)
standardized_input = chain.forward(previous, post_step)
chain.write_state(Path("patched_state.bin"), previous)
```

The two state files must come from the same one-step replay of an authentic
checkpoint. The chain requires consecutive `NT` values and verifies both `TD`
values against the canonical fresh-run default-real clock. This prevents a
single unstaggered boundary, unrelated states, or an off-by-one seasonal phase
from silently producing a plausible vector.

Use `tangent(...)` for the Jacobian action and `transpose(...)` for its exact
transpose. `transpose(...).scientific_control_pair()` returns the fixed-phase
packed-real pair; `phase_td_cotangent` exposes the separate passive diagnostic.
For a single-boundary map with `s1 = F(s0)`, propagate `post_step_real32`
through the matching one-step model reverse and add it to `previous_real32`.
`compose_post_step_pullback(...)` performs that final addition. The observation
module does not substitute for, rebuild, or revalidate the final3 dynamic
reverse.

The standalone `core_field_from_real_state(...)` exposes the same canonical
20-by-27 selector, and `nino3_from_real_state(...)` reproduces the final3
66-cell double-accumulation head. `write_packed_zc_state(...)` and
`chain.write_state(...)` publish exact typed state streams atomically and
refuse an existing output unless `overwrite=True` is explicit.

## Validation

`tests/test_native_observation.py` covers typed state-stream parsing, exact grid
and field ordering, consecutive-time guards, frozen standardization, the full
tangent/transpose dot-product identity (including the separately reported
`TD` term), and bitwise forward equality for both retained extreme-event
checkpoints. For each event, both the raw and standardized vectors match the
released processed-data row bit for bit.

This observation transpose is not a balanced native-state lift. In particular,
`H1`, `U1`, and `V1` are carried diagnostics rather than independent controls
under `state_manifest.json`. A scientific AGOP intervention still needs an
explicit admissible control/lift map and its transpose.
