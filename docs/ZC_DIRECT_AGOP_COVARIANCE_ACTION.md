# Covariance-action steering experiments

This document records the steering method used for manuscript Figures 10--12.
It deliberately excludes pilot interventions and legacy
covariance formulations from the primary protocol.

## Scientific question

For a fixed explanation direction `e`, can small changes to the native
Zebiak--Cane (ZC) state make a later El Nino or La Nina outcome more likely?
The experiment changes only the scalar release coordinate `e.T @ x`; it does
not require the full native-state displacement to be parallel to `e`.

The four-field CNN input has 2,160 spatial entries on the `20 x 27` grid plus
two annual-phase coordinates. The intervention holds phase fixed. A native ZC
restart instead has 28,591 independently controllable entries:

```text
AKB (Kelvin-wave amplitude):       79
UB  (non-Kelvin zonal current): 9,164
HB  (thermocline depth):        9,164
V   (meridional current):       9,164
TO  (SST anomaly):              1,020
total:                          28,591
```

Redundant boundary, diagnostic, passive, and clock entries are preserved by
the model but are not independent controls. Because the CNN and restart
representations differ, an explanation cannot be added directly to a
checkpoint.

## Minimum-action problem

Nine native perturbations `a_1,...,a_9` are inserted before successive ZC
steps during the three months before release. The model then evolves freely
for ten months. For a requested release coordinate `q`, the calculation
approximately solves

```text
minimize    sum_i a_i.T @ C^+ @ a_i
subject to  e.T @ x_release(a_1,...,a_9) = q.
```

Here, `x_release` is the standardized four-field CNN input observed from the
nonlinear ZC release state. The canonical `C` is one common covariance for
all nine controls, both event seasons, and every manuscript method. It uses
all 360,000 states in the 10,000-year training block: 10,000 states at each of
36 annual phases. A separate mean is removed at each phase, and the 36 sample
covariances are averaged with equal weight:

```text
C = (1/36) sum_p [(X_p - 1 mu_p^T)^T (X_p - 1 mu_p^T) / 9,999].
```

Thus common within-phase ZC variations are inexpensive, unusual combinations
are expensive, and the seasonal displacement among phase means is excluded.
The warm and cold experiments no longer select different covariance
populations.

The implementation parameterizes every perturbation as `a_i = C @ v_i`. This
keeps controls in the empirical covariance range and evaluates covariance
action without constructing an inverse. The exact pooled `C` is compiled once
as a read-only Fortran-order float64 `28,591 x 28,591` array (about 6.09 GiB)
and memory mapped. A covariance application is one dense matrix product; the
360,000 source states are not scanned during optimization.

The former warm/cold nine-phase factor geometries remain available only under
the explicit `legacy-event-window-nine-phase` policy for historical
reproduction and phase-specific sensitivity analysis. They are not pooled
manuscript runs and their outputs must be labelled separately.

## Nonlinear solver and acceptance gates

Each major sequential-quadratic-programming update uses:

1. a nonlinear ZC integration through the intervention window;
2. an exact reverse-adjoint gradient of the release coordinate with respect
   to all nine controls;
3. applications of the single annual covariance;
4. the minimum-action solution of the scalar linearized constraint, limited
   by a covariance-metric trust region; and
5. a forward nonlinear line search.

Hard targets use adaptive continuation: the requested coordinate is approached
through intermediate fractions, but only the exact final target can become a
published result. Accepted continuation stages are written atomically and can
be resumed after interruption. Rejected iterates are never reported as
scientific outcomes.

The release result is accepted only if the requested coordinate is attained,
the relative covariance-metric stationarity residual is at most `0.02`, and
relative complementarity is at most `1e-4`. The final controls are then
replayed with both the differentiable and canonical optimized Fortran
executables.

## Manuscript experiments

Figure 10 starts from the selected extreme event and independently optimizes
release coordinates `(1 + c) q_ext`, where

```text
c = -1.0, -0.9, ..., -0.1, 0.1, 0.2, 0.3.
```

The unmodified event is `c=0`. The resulting free trajectories form the warm
and cold dose--response panels.

Figure 11 uses ten uniformly sampled, nonoverlapping test trajectories at the
appropriate annual phases. Selection does not depend on initial conditions or
future Nino-3. The warm and cold release points lie on the same underlying
control trajectories, shifted by three months to match the phases of the
selected events. Each optimization matches the selected extreme event's
event-oriented AGOP coordinate.

Figure 12 compares the same annual pooled-covariance protocol for AGOP XAI,
GRAD, Integrated Gradients, GradientSHAP, and the strongest-10-percent
composite direction. Each colored curve is the fixed ten-member control mean
plus the mean paired nudged-minus-control response over every completed case
that passes the prespecified numerical gates. Its sidecar reports the completed
sample size for each method and event.

## Reproduction

After generating the public `zc-v3` data and trained core-four-field CNN,
capture all annual native phases, publish the exact factors, and compile the
single dense covariance:

```bash
python scripts/capture_zc_annual_native_phases.py \
  --independent-overlap-replay
python scripts/build_zc_native_covariance_factors.py
python scripts/build_zc_all_phase_dense_covariance.py
```

The direct driver defaults to
`outputs/zc_native_covariance/training-years10000-phases00-35/dense_manifest.json`,
requires phases `0,...,35`, 10,000 samples per phase, state size 28,591, the
full training interval, and matching data/state-layout provenance. Pooled
warm and cold wrappers inherit that same default. See
`docs/ZC_NATIVE_COVARIANCE.md` for the artifact contract and storage details.

The production drivers are:

```text
scripts/run_zc_extreme_agop_nonlinear_dose_response.py   Figure 10 data
scripts/run_zc_pooled_xai_covariance_action.py           XAI steering cases
scripts/run_zc_pooled_composite_covariance_action.py     composite cases
scripts/render_final_steering_figures.py                 Figures 10--12
```

The final checked inputs are organized under
`artifacts/zc-v3/manuscript/steering/`; final PDFs are under
`outputs/manuscript/`. Portable commands are listed in
`docs/FRESH_FIGURE_RUNBOOK.md`.

## Interpretation limits

- The SQP output is a local constrained solution, not a certificate of the
  global nonlinear minimum.
- Empirical covariance action is a plausibility geometry, not proof that a
  perturbation lies on an exact balanced manifold.
- Matching one explanation coordinate permits covariance-compatible companion
  changes. The experiment tests the coordinate under its minimum-action native
  completion, not a raw coarse-grid vector in isolation.
- Results from failed numerical gates are retained for diagnosis but excluded
  from manuscript outcome statistics.
