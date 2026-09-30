# Runtime-variable final3 window

This directory provides isolated, runtime-configurable path and reverse drivers
for the authentic run23 final3 Zebiak--Cane kernel. The scientific and
Tapenade-generated routines are not copied, regenerated, or edited. The build
links the new drivers against the existing final3 compiled objects only after
verifying an exact 45-file SHA-256 allowlist and the compiler recorded by the
canonical build.

The driver restriction was narrower than the kernel restriction: final3's
`ZC_KERNEL_WINDOW`, one-step `FWD`/`BWD`, and tangent routines already accept a
runtime step count. The canonical user drivers alone fixed their allocations
and path headers at 31 transitions. These versions allocate from the requested
or recorded path length, bounded to 1 through 10,000 transitions. Practical
memory use grows linearly by about 279 kB per retained boundary.

## Build

From the repository root:

```bash
bash adjoint/variable_window/build_variable_window.sh
```

The default input is
`adjoint/tapenade_toolchain/build/coupled_run23_final3`, and the default output
is `adjoint/variable_window/build`. Optional positional arguments replace those
two directories. The output directory must not already exist. The builder:

1. rejects symbolic or missing canonical inputs;
2. checks `canonical_final3_objects.sha256`, including the final3 build
   manifest, input inventory, immutable includes, canonical executables, every
   reused primal/reverse/support object, and the tangent archive;
3. checks the internal hashes recorded by the final3 build manifest;
4. requires the exact recorded compiler and derivative flags;
5. compiles and links in a sibling staging directory, reverifies every input,
   and publishes the completed directory with one rename.

This intentionally binds the local final3 generation, not any object tree that
happens to expose compatible symbol names. A legitimate new canonical build
will require a new allowlist and its own numerical validation.

## Run

Run inside a copied, validated ZC runtime directory containing `fc.data`, the
restart, namelists, `Data/`, and other final3 runtime files. The path interface
is:

```bash
adjoint/variable_window/build/zc_variable_make_path \
  kernel_initial_state.bin zc_40step_path.bin 40
```

The third argument is the number of coupled ten-day transitions. Omitting it
retains the canonical 31-step default. The stream begins with seven native
32-bit integers (steps and the six state/tape dimensions), then the complete
typed state at every boundary and the three-integer branch tape at every step.

The reverse reads its length from that authenticated path header:

```bash
adjoint/variable_window/build/zc_variable_adjoint \
  zc_40step_path.bin zc_40step
```

With no third argument, the terminal objective is the canonical 66-cell
Niño-3 mean. Supplying a 59,148-value float32 stream as the third argument
computes the transpose for that arbitrary terminal real-state cotangent. The
result is `<prefix>_gradient.bin`; the replay audit and metadata are written
beside it.

`zc_variable_tangent` is included as a validation companion:

```bash
adjoint/variable_window/build/zc_variable_tangent \
  zc_40step_path.bin initial_direction.bin zc_40step_tangent
```

It composes the unchanged generated one-step tangent on the same retained
boundaries and writes `<prefix>_jv.bin`. It is useful for checking
`w dot (J v) = v dot (J^T w)` for a chosen path and terminal seed.

## Validation and scope

`tests/test_zc_variable_window.py` checks source isolation and the exact object
allowlist. When the local final3 build, compiler, and extreme-warm runtime are
available, it also performs the following executable tests in temporary copied
runtime directories:

- the variable driver's 31-step path is byte-for-byte identical to final3
  (SHA-256 `5537a3f8b7e94b5341ca0a8897ff80ec18338d41428a2bd45850808c2b86be3d`);
- its canonical Niño-3 gradient is byte-for-byte identical to final3 (SHA-256
  `da5b61f288b25d31f4113c8df0ac7698aed26af4c6b68a668e78461500f4f9a1`);
- a 10-step path has the exact typed-header-dependent byte count; and
- a 40-step path, tangent, and custom-terminal reverse remain finite and give a
  relative transpose defect of `6.97e-6` for the fixed test vectors. The
  final Niño-3 value in that smoke path is `3.1939349174499512` °C.

These are branchwise, real32 results under the same Standard-configuration and
major-tape qualifications as final3. A longer runtime choice is not new global
derivative certification.

The sibling `adjoint/controlled_window` tools use the same packed-state ABI and
canonical one-step objects for trajectories with externally inserted controls.
They are not modified here. This directory handles an uninterrupted retained
path and its initial-state/terminal-state derivative. A controlled recurrence
must retain or assemble its own consecutive boundaries and branch tapes before
using the corresponding reverse composition.
