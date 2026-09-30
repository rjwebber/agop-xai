# Zebiak--Cane tangent-linear and adjoint design and status

## Decision and scope

A validated discrete adjoint of the production Zebiak--Cane (ZC) model is the
goal.  The primary object is a source-level, branchwise linearization of the
real arithmetic in the numerical algorithm used to generate `zc-v3`, rather
than a differentiable rewrite or surrogate.  It follows the comparisons and
iteration counts executed by the primal run.  It does not differentiate
floating-point rounding or assign derivatives to discrete branch selectors,
and it is not a globally smooth derivative of the model.

The first supported configuration is deliberately narrow and matches the
current experiments:

- deterministic forcing (`NIC=0`, no stochastic westerly wind bursts);
- the classic, nonsmoothed model switches;
- the production default-real build used by `zc-v3`;
- authentic restart checkpoints and exact model time/seasonal phase;
- a ten-day coupled step and windows of up to 40 steps;
- terminal objectives defined on an explicit model state, not implicitly on
  the time-staggered output-file record.

A result for another compile-time or namelist configuration requires a fresh
forward-equivalence and derivative validation run.

## Implemented status (6 September 2026)

The source-free, same-generation run23 final3 producer and evidence chain is
curated under `adjoint/tapenade_toolchain/results/run23`. Its precise status is
**`provisional_major_tape_conditioned`**. This means an operational branchwise
research derivative for the three locked trajectories and configuration; it is
not global derivative certification. The curated index binds the original and
sanitized report hashes, three compile-input inventories, build and executable
hashes, v2 producer records, and three named-array gradient packages.

- the explicit-state primal kernel has passed pack/unpack, reference-restart,
  fresh-process segmented-replay, and omitted-workspace poison tests;
- the final3 full coupled tangent was evaluated with one-step centered finite-
  difference checks at selected El Niño, La Niña, and neutral checkpoints.
  Directions restricted to the conservative independent-control mask pass the
  provisional `1e-4` complete-step gate, with relative full-output errors
  `2.07e-6`--`2.31e-6`. Broader directions that also perturb active carried
  state give `8.18e-5`--`1.70e-4`; the warm and neutral cases fail that
  gate, while the cold case passes;
- the final3 full coupled reverse passed same-generation
  31-transition tangent/reverse dot tests.  The maximum norm-product-scaled
  transpose defects over
  three directions per checkpoint are `6.39e-10`--`2.29e-9` for the canonical
  scalar Niño-3 head and `3.04e-8`--`3.70e-8` for dense random terminal-state
  seeds.  The more cancellation-sensitive bilinear-relative maxima were
  `5.58e-4` and `6.89e-5`, respectively; both normalizations must be reported;
- independent 31-transition finite differences corroborate the scalar
  gradient but do not exhibit the predeclared clean `h^2` Taylor regime.  The
  best recorded major-tape-stable directional-derivative errors are `0.0336%`
  for the extreme El Niño aligned check and `1.11%` for the neutral check.
  The extreme La Niña sweep has no two-sided major-tape-stable step size because
  the minus perturbation changes `NSST` at transition 15 for every tested
  step size. Random full-state composed checks are retained as diagnostics,
  not passes, because the warm and cold perturbations change the recorded tape;
  and
- the centered finite-difference reduced-control oracle is an independent
  reference calculation, not an adjoint implementation.

These checks support the stated operational, configuration-specific,
major-tape-conditioned research adjoint but do not certify a globally smooth
long-window derivative. Only a production-compatible default-real derivative
build exists; the real64 twin below remains a design target. Run21/run22
reports remain historical audit records and are not substituted for the
same-generation final3 evidence.

Raw coefficient norms need special caution. The packed 59,148-coordinate
gradients mix units and include carried atmospheric memory. The largest segment
accounts for 61.5% of squared norm in the warm case (`Q0O`), 93.7% in the cold
case (`UBAR`), and 50.1% in the neutral case (`Q0O`). Only 28,591 native entries
belong to the conservative independent-control mask. The raw full-state vector
must not be interpreted as physical feature importance.  The differentiated
observation/time-alignment and frozen-normalization maps are now implemented in
`zc_xai.native_observation`. A phase-local empirical balanced control map and
its exact algebraic transpose are implemented in `adjoint/balanced_control_map`;
their finite-rank and nonlinear-manifold limitations remain part of the
scientific qualification.

## Why this is practical

One coupled ten-day step is

```text
SSTA -> ZATMC -> STRESS -> CFORCE -> MLOOP
```

and a ten-month forecast from the neural-network input is only 30 steps.  The
retained event checkpoint is immediately before that input update, so the
checkpoint-to-target operator used here contains 31 transitions.  The closed
explicit typed state is 278,864 bytes.  The 278,808-byte legacy file contains
four payloads of 172, 112,244, 32,640, and 133,720 bytes plus two four-byte
record markers around each payload (32 marker bytes total).  The payloads of
legacy records 2--4 total 278,604 bytes.  The explicit state removes their
80-byte legacy `HEADER`, then adds 316 bytes for 79 hidden `AK` values, 16 bytes
for four integers, and 8 bytes for two clock reals:
`278,604 - 80 + 316 + 16 + 8 = 278,864` bytes.  Retaining all step-boundary
states for the 31-transition reverse pass costs under 9 MB.
Memory and optimal checkpoint scheduling are not limiting factors.

The old source contains an unfinished `run_CZ_as_subroutine` path and a
39,019-element state-vector declaration that were intended for Newton
optimization.  The required serializer is absent, its calls are commented
out, and parts of its scaling code are stale.  It is useful evidence that the
model was once considered for derivative-based optimization, but it should
not be revived directly.

## State at a coupled-step boundary

The derivative state must be the complete state required to reproduce the
next step, not merely the four fields shown to the neural network.  It
includes at least:

- fine ocean state `AKB`, `UB`, `HB`, and `V`;
- ocean boundary memories `UBNDY` and `HBNDY`;
- native SST anomaly `TO`;
- deliberately lagged coarse fields `H1`, `U1`, and `V1`;
- atmospheric incremental/iteration memory `Q0O`, `UO`, `VO`, `DO`, and
  `DIF`;
- atmospheric-reset time, absolute model time, and seasonal phase;
- `HTAU` when preserving the exact existing instruction boundary.

The implemented explicit state packs the numeric payload of the validated
restart together with the hidden Kelvin-amplitude memory needed to reproduce
the next step.  `adjoint/fortran_kernel/state_manifest.json` classifies each
packed segment as active, diagnostic, or passive and records which active
segments are valid independent controls.  Arrays omitted as overwritten
timestep workspace are named in that manifest and are covered by fresh-process
poison tests; they were not omitted merely because they looked diagnostic.

## Forward interface

The production model's `COMMON`, `SAVE`, `ENTRY`, initialization, and file-I/O
behavior is isolated behind an explicit interface:

```fortran
subroutine zc_step(state_in, forcing_in, time_in, &
                   state_out, observations_out, branch_tape)
```

Initialization, restart I/O, plotting, and long-run output stay outside this
active kernel.  Packing is bitwise round-trip exact, and one-step and
multi-step windows reproduce the untouched executable's final legacy restart
bit for bit at the tested checkpoints.  Fresh-process segmented replay tests
closure of the full explicit state.  The untouched reference executable does
not emit an independently packed copy of every internal value, so this evidence
must not be described as a direct field-by-field comparison of every internal
state element or all 13 released fields.

The released field stream is time-staggered.  In particular, `H1`, `U1`, and
`V1` in path/state 0 describe the coarse ocean before the coupling stage,
whereas `TO` in path/state 1 has advanced through that stage.  Every derivative
objective must therefore name its exact instruction boundary.  A final Niño-3
objective seeds the final `TO` array directly.  The raw 59,148-coordinate
gradient is not coordinatewise comparable with the standardized 2,162-feature
AGOP vector. The paired-boundary observation/time-alignment map and frozen-
normalization transpose are now implemented in `zc_xai.native_observation`; a
comparison can now use the empirical phase-local native-state control map in
`adjoint/balanced_control_map`, subject to its finite-rank and nonlinear-
balance qualifications.

## Tangent and adjoint generation

The implementation route is source-transformation automatic differentiation
with [Tapenade](https://tapenade.gitlabpages.inria.fr/tapenade/), after making
the active kernel deterministic, reentrant, and I/O-free.  Tapenade has
generated tangent-linear and reverse sources; their generation alone is not a
validation result.  Small troublesome primitives may receive audited hand
adjoints:

- real tridiagonal ocean solves use transpose solves;
- complex atmospheric tridiagonal solves use conjugate-transpose solves;
- FFTs require the exact conjugation and normalization transpose;
- averaging, interpolation, and packing use explicit transpose operators where
  they enter the implemented native-state derivative. The paired-boundary
  observation and frozen-standardization chain is documented in
  [`ZC_NATIVE_OBSERVATION.md`](ZC_NATIVE_OBSERVATION.md).

The untouched production executable remains the independent finite-
difference oracle.  A fully hand-written adjoint, Enzyme/Flang, or a JAX
rewrite is not the first implementation path.

## Piecewise physics and branch tape

This is a piecewise-smooth model.  The derivative follows the branch and the
number of iterations actually executed; it does not invent a derivative for
the branch selector.

The currently implemented boundary tape has width three and records only:

- the adaptive SST substep count (`NSST`);
- the executed atmospheric feedback-iteration count; and
- one atmosphere-reset indicator.

The reset indicator does not distinguish the threshold reset from the
five-year fallback.  The current tape does **not** record the thermocline,
upwelling, heating, or upwind sign masks; the SST-cap mask; the controlling CFL
cell; or distances to switching thresholds.  Generated derivative code follows
the control flow of its primal sweep internally, but the compact three-value
boundary tape alone cannot audit all of those fine-grained decisions.

A more diagnostic branch tape remains a design target.  It would record:

- the `abs(Niño-3) <= 0.1` and five-year atmosphere-reset decisions;
- the number of atmospheric feedback iterations;
- thermocline, upwelling, heating, and upwind sign masks;
- the SST-cap mask;
- the adaptive SST substep count and controlling cell;
- the minimum distance from each switching threshold.

Recomputation during the reverse pass must at minimum reproduce the implemented
three-value tape bit for bit.  Taylor tests that cross a known switch are
reported as nonsmooth cases, not silently counted as derivative failures.  A
future optimization should use a trust region and authentic-forward line
search, with every detectable branch change reported.

## Checkpoint and precision policy

All ten-day boundary states are retained.  For each reverse step, the code
restores its boundary state, reruns that step while retaining its intrastep
values, and immediately reverses it.  Revolve-style checkpoint scheduling is
unnecessary for the present 10--40-step windows.

The operational derivative build retains the production model's default
single-precision `REAL` arithmetic.  A second, explicitly kind-refactored
real64 verification twin is planned for more stringent dot and Taylor tests;
it has not yet been implemented or validated.

The intended two-build policy is therefore:

1. an operational real32 adjoint generated from an AD-ready primal that passes
   bitwise reference-replay tests; generated derivative sweeps are compared
   with and, for a window reverse, reanchored to exact authentic boundary
   states rather than presumed bitwise identical;
2. an explicitly kind-refactored real64 verification twin for stringent dot
   and Taylor tests.

The production source must not be converted with a blanket
`-fdefault-real-8`, because that changes binary layouts and some complex
declarations.  Inner products and norms should use real64 accumulation in both
builds.

## Required verification and current coverage

The numbered items and numerical gates in this section began as acceptance
targets. The status summary and curated final3 evidence record which have been
satisfied, which remain diagnostic, and which remain future work. In
particular, the ten-seed/block-impulse expansion and real64 twin are not done.

### Forward gates

1. State pack/unpack is bitwise exact.
2. One-step and multi-step zero-control trajectories match the untouched
   executable bit for bit.
3. All locked neutral, extreme El Niño, and extreme La Niña restarts pass.
4. The supported runtime/compile configuration is asserted and hashed.

### Operator transpose tests

For every map with

```text
delta_y = A delta_x + B delta_u,
```

verify

```text
<A delta_x + B delta_u, w>
    = <delta_x, A^T w> + <delta_u, B^T w>.
```

The completed validation suite should cover the covariance lift,
observation/standardization map, `ZAVG`, both tridiagonal solvers, FFT, local
piecewise physics, each coupled component, one complete step, and 10--40-step
windows.  The target is ten fixed random seeds and block impulses at each
level.  Present component and one-step tangent tests cover only a subset of
this list.

Initial relative-defect gates are:

| Scope | real64 verification | production-compatible real32 |
|---|---:|---:|
| Linear/unit operator | `1e-12` | `1e-5` |
| Complete ten-day step | `1e-10` | `1e-4` |
| 10--40 steps | `1e-8` | `1e-3` |

Failures are diagnosed rather than accommodated by automatically weakening
the gates.

### Taylor and finite-difference tests

For a scalar objective `J`, require first-order behavior of

```text
abs(J(x + h v) - J(x))
```

and second-order behavior of

```text
abs(J(x + h v) - J(x) - h grad(J)^T v)
```

over at least three consecutive major-tape-stable step sizes.  Because the
three-value tape does not record every internal switch, this condition alone
does not prove that every fine branch stayed fixed.  Tests span annual phase,
neutral states, the retained extreme warm and cold events, and
directions concentrated in SST, ocean wave memory, thermocline, currents, and
boundary memory.

For the first 36-dimensional reduced control, every coordinate is also
checked with centered finite differences from the untouched executable,
along with at least 20 random directions.  At the current roughly 0.23-second
13-month runtime, a complete 73-run centered-difference gradient costs only
about 17 seconds, making it both a useful bridge and a strong independent
oracle.

## Development order and deliverables

1. **State contract and primal kernel -- implemented:** state manifest,
   pack/unpack routines, the current three-value branch logger, and bitwise
   replay reports.
2. **Tangent model -- provisional final3 evidence complete:** one-step
   full-coupled `J v` checks pass strongly on the independent-control mask;
   broader active-carried-state checks are less accurate and retain the stated
   real32 caveat.
3. **Adjoint model -- provisional final3 evidence complete:** same-generation
   full-coupled `J^T w` results include three scalar-head and three dense-
   terminal transpose tests at each locked checkpoint. Ten seeds, block
   impulses, and the real64 twin remain future targets.
4. **Window adjoint -- provisional package complete:** the 31-transition
   canonical Niño-3 producer, three v2 producer records, and three gradient
   packages are curated. Real32 Taylor evidence lacks a clean quadratic regime
   and every tested aligned La Niña perturbation crosses an `NSST` switch at
   transition 15. A hash-bound runtime-variable wrapper reproduces the
   canonical 31-step path and gradient byte for byte. Its 40-step dense-seed
   tangent/reverse check on an unforced retained trajectory has relative dot
   defect `6.97e-6`, within the predeclared production-real32 window gate.
5. **Observation bridge -- implemented:** the exact staggered core4 selection,
   annual phase, frozen checkpoint standardization, tangent action, and
   transpose pass bitwise released-event and dot-product tests. Passive `TD`
   cotangents are reported separately and are not scientific actuations.
6. **Scientific application -- implemented and archived:** the staggered
   observation, pooled covariance control, nine native-state interventions,
   tangent/reverse boundary recurrences, and constrained nonlinear solves
   produce the dose-response, matched-cohort, and method-comparison results in
   Figures 10--12. The final protocol is documented in
   [`ZC_DIRECT_AGOP_COVARIANCE_ACTION.md`](ZC_DIRECT_AGOP_COVARIANCE_ACTION.md);
   exploratory neutral-state and fixed-template pilots are not release results.
7. **Release hardening -- curated evidence complete:** the recipe pins Tapenade,
   separates the authentic `-O3` primal from debug `-O0` derivative
   compilation, refuses stale targets and uninstrumented reverse code, and
   records source/patch/executable hashes. Curated evidence and packages carry
   provisional v2 producer metadata. The project-authored ZC-derived patches
   and coupled drivers are released with the repository; a portable
   container/compiler digest remains a possible reproducibility enhancement.

## Release boundary checklist

- **Configuration:** Standard grid, deterministic `NIC=0`, no stochastic WWB,
  classic switches, default-real arithmetic, and the exact recorded time and
  seasonal phase only.
- **Objective:** canonical center-inclusive Niño-3 after exactly 31 coupled
  ten-day transitions; the event checkpoint is immediately before the first
  transition.
- **Coordinates:** derivatives are with respect to 59,148 raw packed Fortran
  `REAL` coordinates.  They are not standardized feature importance and not
  every carried-state coordinate is an independent physical control.
- **Validation label:** the curated gradients are labeled
  `provisional_major_tape_conditioned` because their exact producer and evidence
  chains are present. Do not abbreviate that to a globally validated or
  certified ZC adjoint.
- **Branch audit:** state that the external tape records only `NSST`, atmosphere
  iteration count, and one reset indicator; it does not certify every fine
  physics mask.
- **Precision:** state that there is no real64 verification twin and no clean
  31-step `h^2` Taylor regime in the authentic real32 trajectory.
- **Separation of methods:** this is neither an adjoint of the CNN/AGOP
  pipeline nor the reduced finite-difference oracle.  No balanced-state
  optimization result is implied by the derivative build.
- **Provenance:** retain the exact prepared, generated, executable, checkpoint,
  path, gradient, and evidence hashes in each package.  A format-valid NPZ/JSON
  file alone is not derivative certification.
- **Distribution:** release the deliberately curated evidence, packages, and
  project-authored ZC-derived patches and coupled drivers. Keep downloaded
  upstream source copies, generated derivatives, executables, scratch reports,
  and local paths out of the public deposit.

Redistribution permission has been confirmed for the project-authored
ZC-derived modifications included in this repository. The downloaded upstream
ZC source is still obtained separately, and generated adjoint source and
compiler products remain excluded from the public deposit.
