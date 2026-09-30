# Authentic ZC explicit-state forward kernel

This directory is the forward-model gate for a discrete tangent-linear and
adjoint Zebiak--Cane implementation. It does **not** modify files under
`data/raw/CZ_model_share`. The builder copies that tree into an isolated build
directory, applies the same output/restart patch used for `zc-v3`, and builds:

- `zc_reference`: the untouched-equations forward reference executable
  equations;
- `zc_kernel_replay`: a driver for the explicit-state timestep/window kernel.

The callable forward sequence is exactly

```text
SSTA -> ZATMC -> STRESS -> CFORCE -> MLOOP
```

after the driver advances the ten-day clock. `ZC_KERNEL_STEP` applies this map
once; `ZC_KERNEL_WINDOW` applies it repeatedly. File writes are disabled while
the active kernel executes. The initial restart/configuration setup remains
outside the active map.

`CFORCE` contains saved interpolation geometry and a saved upper component
index.  Consequently, each new process must make the authentic initialization
call with `ISTART=0` before calling either explicit-state entry point.  The
provided driver does this by calling `STRESS` and `CFORCE` before setting
`ISTART=1`; this is part of the fixed model context, not part of the evolving
state.  Calling `ZC_KERNEL_STEP` or `ZC_KERNEL_WINDOW` in a new host program
without reproducing that initialization is unsupported.

## State contract

The supported first configuration is the public Standard grid used for
`zc-v3`: `NXP=79`, `NYP=116`, `NSEG=1`, full atmosphere, `NIC=0`, and no
compiled westerly-wind-burst option. The wrapper refuses a different grid.

The old `ndim_X_ic.h` value of 39,019 is not used. It predates the complete
restart state required by the present source. The new interface contains the
numeric payload of the authentic restart records 2--4 plus one hidden piece of
timestep memory that the legacy restart omitted:

- 59,148 `REAL` values (59,069 legacy-restart values plus the 79-value
  Kelvin-wave memory `AK` omitted by the legacy restart format);
- 5,280 `COMPLEX` values;
- one `REAL*8` value;
- four discrete integers and two passive clock values.

The complex `Q`, `E1`, and `E2` arrays are retained solely for byte-exact
restart fidelity. `ZATMC` overwrites every element that it later reads, so
they are diagnostic workspace rather than active trajectory state in the
supported configuration. The disabled random-generator state is likewise
passive when `NIC=0`.

The state packs only actual Standard-grid extents. It deliberately excludes
2,827 unused elements arising from the maximum Fortran array declarations;
packing those elements would read uninitialized padding. It excludes `U`,
`H`, `F`, `G`, and `BK`, which are step workspace overwritten before future
use. It does not exclude `AK`: the western-boundary update reads prior-step
`AK(2)` before rebuilding the array, so all 79 active values are appended to
the explicit state. This also documents a closure defect in the historical
restart format, which did not serialize `AK`. It is differentiated as carried
state but is not exposed as an independent scientific intervention control.

[`state_manifest.json`](state_manifest.json) gives zero-based Python slices,
one-based Fortran slices, logical shapes, column-major storage order, and
active/diagnostic/passive classifications. It separates native
prognostic/boundary state, atmosphere iteration memory, lagged coarse
diagnostics, restart/output workspace, and passive branch metadata. Workspace
values are preserved for exact replay but must not be treated as independent
physical controls in an intervention or adjoint interpretation. In particular,
the exact Python slices are `TO[32121:33141]`, `U1[33141:34161]`,
`V1[34161:35181]`, and `H1[35181:36201]`; each reshapes to `(30, 34)` with
Fortran order.

The released diagnostic stream is time-staggered at a coupled-step boundary.
In path/state 0, `H1`, `U1`, and `V1` describe the coarse ocean before the
coupling stage, while `TO` in path/state 1 has advanced through that stage.
The scalar adjoint seeds terminal `TO` directly. Its 59,148 raw packed
coefficients therefore cannot be compared coordinate by coordinate with the
standardized 2,162-feature neural-network AGOP. The explicit paired-boundary
observation/time-alignment map and frozen-normalization transpose are now in
`zc_xai.native_observation` and documented in
[`../../docs/ZC_NATIVE_OBSERVATION.md`](../../docs/ZC_NATIVE_OBSERVATION.md).
A balanced lift or declared native-state control map is still required for a
scientific coordinatewise comparison.

## Build and validate

From the repository root:

```bash
python adjoint/fortran_kernel/build_kernel.py \
  --source-dir data/raw/CZ_model_share \
  --build-dir outputs/zc_adjoint/kernel_build \
  --overwrite

python adjoint/fortran_kernel/validate_replay.py \
  --build-dir outputs/zc_adjoint/kernel_build \
  --data-dir data/processed/zc-v3 \
  --output-dir outputs/zc_adjoint/kernel_replay_validation \
  --overwrite
```

Both programs validate their inputs before preparing output. They reject root,
home, repository-root, symbolic-link, and input-overlapping targets. A
nonempty directory is replaced by `--overwrite` only when it contains that
program's ownership marker. A directory created by an older, unmarked version
must be inspected and removed manually or replaced with a new output path.

The validator uses the locked extreme El Nino and La Nina native checkpoints.
For each it tests 1-, 10-, 30-, 31-, and 40-step windows. The 31-step case is
included because the released event checkpoint is immediately before the
10-month-lead input update: step 1 creates the input state and step 31 creates
the associated target state. It requires:

1. byte-identical `pack -> unpack -> pack` state files;
2. a byte-identical final four-record legacy restart from the kernel and
   untouched reference executables;
3. correctly sized and valid branch summaries; and
4. a byte-identical result when every ten-day boundary is restored in a fresh
   process, which exposes omitted `COMMON` or local `SAVE` memory.

It also executes two one-step adversarial workspace tests per checkpoint.  A
test-only sentinel makes the driver fill every declared element of omitted
`U`, `H`, `F`, `G`, and `BK` with distinct finite values after unpacking and
before advancing.  The final packed state, legacy restart, and branch tape must
remain byte-identical to a clean fresh-process run.  These runs also exercise
the required `CFORCE` static-context initialization independently in every
process.  No sentinel is present in production runs.

On the recorded macOS/gfortran 13.1 toolchain, the historical requested cases
passed bit for bit. A fresh report is written to the requested output directory
as `replay_report.json`; an ignored local output is not release evidence until
a sanitized copy and its producer chain are included in the tracked results
manifest.

The current branch tape records the adaptive SST substep count, executed
atmospheric iteration count, and atmosphere-reset decision. Masks for every
piecewise physics branch are not exposed on this external diagnostic tape.
The reverse therefore remains a derivative on its recorded branch path;
perturbations that cross an unrecorded sign or cap switch require separate
diagnosis rather than being covered by the three tape entries.

The untouched reference executable emits its historical restart, not the
kernel's complete explicit-state vector.  The reference comparison therefore
establishes byte-identical legacy-restart evolution.  Full-state closure is
tested separately by pack/unpack, fresh-process segmented replay, and omitted-
workspace poison tests; it is not a direct independent comparison of every
packed internal value or every released diagnostic field.

Additional checkpoints can be included without changing the validator. The
recommended derivative-validation state is locked neutral member 04 (release
index 402767). Unlike an earlier neutral candidate, it takes no atmosphere
reset branch anywhere in its 40-step baseline. Validate it with

```bash
python adjoint/fortran_kernel/validate_replay.py \
  --build-dir outputs/zc_adjoint/kernel_build \
  --data-dir data/processed/zc-v3 \
  --output-dir outputs/zc_adjoint/kernel_replay_validation \
  --extra-checkpoint neutral_member_04 \
    outputs/zc_agop_neutral_ensemble/core4-cnn-lead-10m-seed-000042/checkpoints/member_04_i402767_preconditioning_start.hst \
    406358 \
  --overwrite
```

The checkpoint SHA-256 is
`62e392ff707836a3683422db580dbbd1268a4baa79555df2401a715acdef7c74`.
The recorded 1-, 10-, 30-, 31-, and 40-step neutral replays are all bitwise
exact and use two SST substeps, two atmospheric iterations, and zero
atmosphere resets at every step.

## Present boundary

This directory is a validated **explicit-state primal kernel**.  It is not, by
itself, an adjoint.  Separate work under `../tapenade_toolchain`,
`../ssta_tapenade`, and `../full_tangent_audit` contains:

- validated component reverse tests, which are not a full coupled adjoint;
- a full coupled tangent that passes selected one-step centered finite-
  difference tests, with relative full-output errors of
  `2.07e-6`--`2.31e-6` for independent-control directions and
  `6.36e-5`--`1.70e-4` for broader active-carried-state directions;
- a historical run22 full coupled reverse whose maximum
  norm-product-scaled 31-transition transpose defects were `2.29e-9` for the
  canonical scalar Niño-3 head and `3.70e-8` for dense random terminal-state
  seeds, while the corresponding cancellation-sensitive bilinear-relative
  maxima were `5.58e-4` and `6.89e-5`; and
- independent 31-transition scalar finite differences that corroborate the El
  Niño and neutral gradients but do not show a clean single-precision `h^2`
  Taylor regime.  The tested La Niña perturbations cross an `NSST` switch.

These run22 numbers are historical cross-checks, not current release evidence.
The implementation is a configuration-specific, major-tape-conditioned
research-adjoint candidate, not a globally smooth or certified derivative.
Only three transpose directions per checkpoint were tested, broader
active-state one-step accuracy was weaker than the independent-control result,
detailed physics masks are absent from the external tape, and no real64 twin
exists.  A fresh same-generation run must be fully cross-bound under
`../tapenade_toolchain/results/run23` before any packaged gradient is described
as current.

The current derivative target is configuration-specific and branchwise: it
linearizes the real arithmetic executed for the locked Standard-grid,
deterministic configuration while treating clocks, discrete decisions, and the
disabled stochastic state as passive.  It is neither a globally smooth
derivative, an adjoint of the neural-network/AGOP pipeline, nor the existing
reduced-control finite-difference oracle.  No full coupled gradient should be
described as certified solely because the primal kernel, a transpose identity,
or an individual component test passes.
