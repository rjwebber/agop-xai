# Independent coupled ZC tangent/adjoint audit

This directory contains numerical audits for the Tapenade-generated tangent
and reverse maps of the coupled Zebiak--Cane kernel. The implemented reverse
is a derivative of the discrete 31-step program on one recorded branch path.
It is not a claim that the model is globally differentiable across adaptive
substep, iteration, sign, reset, or cap changes.

## Current release evidence: run23 final3

The complete, same-generation final3 chain is curated under
`../tapenade_toolchain/results/run23/` with status
**`provisional_major_tape_conditioned`**. It replaces the former forward-only
run23 development bundle, which is preserved separately. This status means a
usable research derivative for the exact locked trajectories and recorded
major tape; it does not mean global derivative certification.

The final3 forward evidence records:

- 15 bitwise comparisons with the authentic ZC executable: 1, 10, 30, 31,
  and 40 steps from the extreme El Niño, extreme La Niña, and neutral
  checkpoints;
- three longest-window replays that restart a fresh process at every step;
  and
- six one-step tests that poison arrays classified as overwritten workspace.

Every recorded forward case passed. The curated bundle also contains the fresh
tangent/reverse, finite-difference, Tapenade warning, and per-case gradient
records, with original and sanitized SHA-256 digests.

## State, objective, and branch contract

- The packed real state has length `59148`; the final 79 entries carry the
  active Kelvin-amplitude vector `AK`.
- The clock, branch choices, and disabled stochastic forcing are passive.
- The external tape has width three: adaptive SST substep count (`NSST`),
  atmospheric iteration count, and atmospheric-reset occurrence. It does not
  contain every fine-grained sign or cap mask.
- The scalar objective is the center-inclusive Niño-3 mean after 31 ten-day
  transitions: Fortran `I=13:18`, `J=21:31` (66 cells). The primal sums the
  float32 cells in Fortran order using a float64 accumulator, divides by 66,
  and publishes one float32 scalar. Reverse mode seeds the selected cells by
  `1/66`.
- Each segmented tangent/reverse step is re-anchored on an independently
  generated authentic-primal boundary state. This avoids treating the
  transformed forward sweep as the independently validated trajectory.
- At a boundary, path/state 0 `H1`, `U1`, and `V1` precede the coupling stage,
  whereas path/state 1 `TO` is advanced. The objective and every derivative
  comparison must state this exact instruction boundary.

Tapenade is pinned at version 3.16, revision
`0449e37b7da896cb2c38b63c8a34f10c87a84534`. The final run23 results index
binds the prepared source, generated source, instrumentation patches,
Tapenade diagnostics, build inputs, and executables.

## Compilation modes used by the audit

The optimization labels refer to different roles and should not be collapsed
into a single statement:

- The normalized primal and immutable setup/support routines are compiled at
  `-O3`, matching the authentic ZC build. In this single-precision model, an
  `-O0` setup build can change last bits and therefore the later trajectory.
- The reverse executable used for release validation is built in the
  conservative debug mode:
  generated derivative objects use `-O0 -g -fcheck=all -fbacktrace`; primal
  and setup/support objects linked into that executable remain at `-O3`.
- The independent tangent executable used for the numerical transpose and
  finite-difference audit is separately compiled at `-O3` from the exact same
  run23 generated tangent source. The bound final3 executable is named
  `zc_tangent_path_driver_run23_final3_o3`.

Thus the run23 transpose reports compare an O3 audit tangent with the O0 debug
reverse generated from the same prepared program. They do not imply that the
reverse itself was compiled at O3.

## Run23 numerical evidence

The final results index contains sanitized, detailed reports rather than links
into ignored build directories:

1. scalar-head tangent/reverse transpose tests for three independent input
   directions at each checkpoint;
2. dense generic-terminal transpose tests for three independent direction and
   terminal-seed pairs at each checkpoint;
3. one-step centered differences against the independently compiled,
   authentic-equation primal for both independent-control and broader
   active-carried directions;
4. 31-step major-tape-stable finite-difference/Taylor spot checks, including
   every observed change in the three-value external tape and the real32
   roundoff limitation; this does not establish stability of unrecorded
   internal signs, caps, or upwind masks; and
5. a Tapenade warnings summary, source/build manifests, executable hashes, and
   a top-level SHA-256 manifest.

All five evidence classes are present. The independent-control one-step
full-output relative L2 errors are `2.07e-6`--`2.31e-6`; broader active-carried
directions give `8.18e-5`--`1.70e-4`. The aligned 31-transition scalar check is
best at `3.36e-4` for the extreme warm case and `1.11e-2` for the neutral case.
Every tested aligned cold-case perturbation changes `NSST` at transition 15.
The random composed 31-step full-state test is retained as a diagnostic, not a
pass, because the warm and cold perturbations change the recorded major tape
and the neutral result is non-asymptotic in real32 arithmetic.

The audit reports two different transpose normalizations:

```text
bilinear-relative = |w^T J v - v^T J^T w| / max(|w^T J v|, |v^T J^T w|)
norm-product      = |w^T J v - v^T J^T w|
                    / (||Jv|| ||w|| + ||v|| ||J^T w||).
```

The norm-product value can be tiny when the two bilinear forms themselves are
small because of cancellation, so it must not be presented as the only
relative-error measure. Run23's maxima are `2.29e-9` (scalar) and `3.70e-8`
(generic terminal seed) under the norm-product definition, but `5.58e-4` and
`6.89e-5`, respectively, under the bilinear-relative definition. Those are
consistent with a real32 implementation, not near-double-precision equality.

## Rebuild and audit

After completing the run23 toolchain build, construct the independent O3
tangent driver with:

```bash
adjoint/full_tangent_audit/build_run_tangent.sh \
  adjoint/tapenade_toolchain/build/coupled_run23_final3 O3 \
  adjoint/full_tangent_audit/build/zc_tangent_path_driver_run23_final3_o3
```

The validators require explicit executable, path/gradient producer,
prepared/generated-source, build-manifest, compile-input inventory, checkpoint
or replay-report, and output arguments. They stage the exact inputs they
consume and reverify the live producer chain before publication. Validation is
computed in a unique sibling directory and published transactionally. A
nonempty output can be replaced only when it carries the matching ownership
marker; roots, symbolic-link traversals, input overlaps, and concurrent target
changes are rejected.

Two source trees have deliberately different roles. `--kernel-source` must be
the compact immutable `coupled_compiled/build_input_snapshot/kernel_source`
tree and is used only to authenticate the compiled forward oracle. It is not a
complete runnable model checkout. The tangent, composed-tangent, and scalar
Taylor validators run from a separate full `--runtime-source`; for release
evidence this must be the exact `verified_inputs/source` tree produced by the
passed schema-2 replay validation (and it defaults to that location when a
replay directory is given). The replay report binds every file in that runtime
tree to the forward build, and the derivative validators copy and reverify the
complete tree before use.

The scalar and generic transpose validators require `--run-dir` to name the
`runtime/` directory immediately beside the supplied producer
`run_report.json`. They verify its files against the report's
`checkpoint_input.staged_file_sha256`, copy only those bound inputs to a
private execution directory, and run the derivative binaries there. Both also
stage the state manifest and forward path before parsing them. The scalar test
additionally re-executes the reverse and requires its regenerated gradient to
match the supplied producer gradient bit-for-bit; the generic test directly
executes both tangent and reverse for each terminal seed. All staged input and
executable hashes are rechecked before the report is published.

## Interpretation and release boundary

- A passing full-coupled transpose audit establishes that the reverse is the
  transpose of its matching tangent on the tested paths. Component-level
  adjoints alone do not establish this claim.
- The 31-step authentic-primal finite-difference checks are expected to reach
  a real32 plateau and may cross adaptive branch boundaries. Absence of a
  clean multi-level quadratic regime must be reported rather than hidden by a
  tuned step size.
- No real64 verification twin has been implemented.
- The raw gradient is with respect to 59,148 packed/differentiated native REAL
  entries, not the four standardized neural-network channels; only 38,315 are
  classified as active carried state, and fewer are conservative independent
  controls. Because the released fields are also time-staggered, an AGOP
  comparison needs the differentiated observation/time-alignment map, the
  frozen normalization transpose, and a balanced native-state lift/control
  map.
- Only restart-mode execution for the locked Standard configuration is in
  scope. The initialization path is not a supported differentiated API.
- Redistribution permission has been confirmed for the project-authored
  ZC-derived patches and coupled drivers, which are published in
  `../tapenade_toolchain/`. The downloaded upstream ZC source,
  Tapenade-generated source, executables, and local audit/build products remain
  outside the public release.
