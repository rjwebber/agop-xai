# Controlled-path ZC reverse bridge

This directory contains a thin one-step driver for the frozen Tapenade
reverse kernel.  It does not change the Zebiak--Cane equations.

The matched primal/tangent/reverse build used for the direct-control experiment
is generated locally as:

```
build-v5/zc_one_step_primal
build-v5/zc_one_step_tangent
build-v5/zc_one_step_adjoint
```

The public release contains the builder and validation code, not this generated
directory or its executables. A local build records hashes and compiler
provenance in `build-v5/build_provenance.txt`.

This matched build is necessary because the canonical data-generation
primal is compiled at `-O3`, whereas the Tapenade derivative was compiled at
`-O0`.  For ordinary ZC states their answers differ only at roundoff scale,
but a large controlled state can lie close enough to the steady-atmosphere
convergence threshold that the two floating-point instruction streams select
different iteration counts.  That makes an `-O0` split forward sweep an
invalid replay of that particular `-O3` step.  Build v5 compiles the unchanged
prepared ZC primal source at the same `-O0` settings as the tangent and
reverse.  Its primal and split-forward replay are bitwise identical.  The
experiment still runs the canonical `-O3` executable separately and requires
it to match processed zc-v3 bitwise; the matched `-O0` zero path is audited
against that canonical path before optimization.

The adjoint driver accepts an input state, an output REAL cotangent, and the
independently generated matched-`-O0` output state.  It reruns Tapenade's split
FWD sweep, uses the reference output's passive D/I/time values in BWD, and
writes the input REAL cotangent.  The Python bridge rejects the result unless
the generated active replay and branch tape agree with that independent
primal replay.

Build it into a new immutable directory (the builder refuses to overwrite):

```
adjoint/controlled_reverse/build_one_step_adjoint.sh \
  adjoint/tapenade_toolchain/build/coupled_run23_final3 \
  adjoint/controlled_reverse/build-v5
```

Earlier unmatched and no-reference-output builds were rejected during
development and are not distributed. They must not be substituted for the
matched build described above.

The public Python entry point is
`zc_xai.zc_controlled_adjoint_bridge.reverse_controlled_release_scalar`.
For the nine-intervention experiment it performs ten one-step reverse calls
and returns all nine native intervention gradients in one backward sweep.
