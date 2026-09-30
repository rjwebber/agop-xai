# Tapenade tangent and adjoint of the coupled Zebiak--Cane model

This directory builds a source-transformed tangent and reverse model for the
authentic Fortran Zebiak--Cane (ZC) implementation. The research executable
used to produce the release evidence computes

```text
J(s0)^T w,
```

where `J` is the branchwise Jacobian of 31 coupled ten-day transitions, `s0`
is a complete native checkpoint state, and `w` is a terminal cotangent on the
59,148-element real state. With no custom `w`, the program returns the
gradient of the canonical final Niño-3 SST anomaly.

This is a discrete adjoint of the implemented algorithm. It is not an
adjoint of a JAX rewrite, a neural-network surrogate, or smoothed ZC
equations.

## Scientific and binary contract

- The supported physics is the deterministic Standard ZC configuration used
  to create `zc-v3`: restart mode (`NSTART=3`/`ISTART=1`), `NIC=0`, no
  stochastic westerly-wind bursts, and the recorded classic model switches.
- The input is the **pre-input native checkpoint**. Advancing from that
  checkpoint through the released neural-network input state to its
  ten-month target takes 31 ten-day transitions. Starting at the released
  neural-network input itself would take 30 transitions.
- The canonical terminal seed averages SST anomaly over the east-inclusive
  `zc-v3` Niño-3 box, Fortran `I=13:18, J=21:31`, with weight `1/66`.
- The packed and differentiated real-state interface has 59,148 entries. A
  dependency audit classifies 38,315 as active carried entries; the most
  conservative intervention interface has 28,591 independent controls after
  overwritten diagnostics/work arrays are excluded. Entries 59,070--59,148
  in one-based Fortran indexing are `AK(1:79)`, Kelvin-wave memory omitted by
  the historical restart format. Complex atmospheric work state and
  clock/discrete state are carried but passive.
- State, path, terminal-seed, and gradient streams are little-endian. A
  terminal seed is exactly 59,148 float32 values; the raw gradient has the
  same layout.
- The API accepts an arbitrary 59,148-coordinate terminal `REAL`-state linear-
  functional seed, so the implementation is a reusable window `J^T w`, not
  only a special Niño-3 gradient.

The state order is defined by the upstream-derived `zc_kernel_state.inc` and
the pack/unpack order in `zc_kernel_api.F`. The publication packager at
`scripts/package_zc_adjoint_gradient.py` maps a raw gradient into named state
segments.

The Python case runner enforces the complete supported runtime contract. It
canonicalizes only the three case-time values in `fc.data` and the two
diagnostic-output times in `modified_means.namelist`; it checksum-verifies all
remaining settings, `scales_EOF.namelist`, and every static `Data/` file
against `config/supported_runtime_config.json`. The three official
state/history pairs also have checked-in producer manifests. An arbitrary
checkpoint with matching physics may be run, but its report sets
`configuration_verified=false` unless its exact checkpoint/history pair is
approved. The bare Fortran executables are low-level interfaces and do not
perform this full runtime-configuration and provenance check.

## Why the reverse is segmented

The legacy code uses global `COMMON` and `SAVE` state. A monolithic
checkpointed reverse would assume reentrant recomputation that the original
program cannot safely promise. The implementation instead:

1. uses the normalized authentic primal kernel to store all 32 exact boundary
   states and the three-entry major branch tape;
2. works backward through 31 separately taped one-step Tapenade `FWD/BWD`
   pairs;
3. reanchors every pair at its exact authentic boundary state; and
4. rejects a changed major branch tape or a generated-forward discrepancy
   outside the recorded tolerances.

The complete path is 8,924,048 bytes, so sophisticated checkpoint scheduling
is unnecessary. Generation uses Tapenade `-nocheckpoint`.

The canonical drivers and evidence remain fixed at 31 transitions. For local
experiments requiring a runtime-selected uninterrupted path, the isolated
[`../variable_window`](../variable_window/README.md) drivers verify and reuse
the exact final3 objects, preserve this segmented schedule, and have a bitwise
31-step compatibility test plus a 40-step transpose smoke test.

## Pinned dependencies

`VERSION.env` pins:

- Tapenade 3.16 develop, revision
  `0449e37b7da896cb2c38b63c8a34f10c87a84534`;
- archive SHA-256
  `534b83598c508a3202880016403f84e352e4b00bdc4c6b934bb39ccbdddb0963`;
- Java 17 for source transformation; and
- a legacy-compatible GNU Fortran compiler for compilation.

The installer checks the downloaded archive hash, extracts a fresh tree, and
writes both a complete tree manifest and a receipt containing the observed
archive hash. Generation and compilation verify every tree entry and the
receipt before using that installation; an added or changed JAR, parser,
runtime file, or stack header is rejected. The generation stage then checks
the live Tapenade version and revision.

Preparation, generation, instrumentation, and compilation form a chained set
of content manifests and non-circular binding records. Each standalone stage
verifies its complete predecessor before it may replace an existing stage.
Preparation also binds all active and passive inputs to one kernel-source
tree, preventing an active-source-A/passive-source-B link. Compilation records
the resolved Fortran compiler, C compiler, and archiver identities, performs
all compile/link commands from a unique initially empty directory, and reads
its final provenance summary only from the immutable checksum-bound snapshot
that it actually compiled.

Other prerequisites are Bash, Python 3.11 or newer, a C compiler, `patch`,
`ar`, `curl`, and `tar`. Compilation intentionally avoids fast-math. The
official Tapenade archive includes a Linux Fortran parser but not a native
macOS parser, so source transformation must run on Linux; the generated
Fortran can then be compiled on macOS.

## Complete Linux build

From the repository root:

```bash
KERNEL_SOURCE=/absolute/path/to/validated/kernel-build/source
BUILD="$PWD/adjoint/tapenade_toolchain/build/coupled_release"
TOOLS="$(mktemp -d)"
TAPENADE_HOME="$(adjoint/tapenade_toolchain/scripts/install_tapenade_linux.sh \
  "$TOOLS" | tail -n 1)"

adjoint/tapenade_toolchain/scripts/run_coupled_toolchain_linux.sh \
  "$KERNEL_SOURCE" "$TAPENADE_HOME" "$BUILD"
```

Every build stage is confined to a marked child of
`adjoint/tapenade_toolchain/build/`. A nonempty stage is preserved by
default. Set `ADJOINT_OVERWRITE=1` only to replace that exact managed build.
All tools and inputs are checked before a requested replacement. The default
derivative build is conservative (`-O0`, bounds checking, and a backtrace),
whereas the independently replayed primal remains at the authentic ZC `-O3`.
Set `ADJOINT_BUILD_MODE=release` only after separately validating that binary.

## Linux generation and macOS compilation

Prepare on the Mac:

```bash
KERNEL_SOURCE=/absolute/path/to/validated/kernel-build/source
BUILD="$PWD/adjoint/tapenade_toolchain/build/coupled_release"

adjoint/tapenade_toolchain/scripts/prepare_coupled_sources.sh \
  "$KERNEL_SOURCE" "$BUILD"
```

Copy the repository release files and prepared build directory to the UCSD
Research Cluster. Run the transformation inside a CPU pod that has both
Python 3 and Java 17. A bare `eclipse-temurin:17-jre` image is insufficient
because it lacks Python. For example, with a Java 17 runtime installed in the
persistent home directory, the cluster command is:

```bash
/opt/launch-sh/bin/launch.sh -f -p low -N zc-tapenade -c 1 -m 2 \
  bash -lc 'export JAVA_HOME="$HOME/jre17_tapenade"; \
    export PATH="$JAVA_HOME/bin:$PATH"; \
    cd "$HOME/agop-xai"; \
    python3 --version; java -version; \
    adjoint/tapenade_toolchain/scripts/generate_coupled_on_cluster.sh \
      "$HOME/agop-xai" coupled_release'
```

The cluster helper checksum-verifies `~/tapenade_3.16.tar` and unpacks it in
the fresh pod. Copy `coupled_generated/` back into the same local `BUILD`,
then run:

```bash
adjoint/tapenade_toolchain/scripts/instrument_generated_reverse.sh \
  "$BUILD/coupled_generated/reverse"

adjoint/tapenade_toolchain/scripts/compile_coupled_adjoint.sh \
  /path/to/extracted/tapenade_3.16 "$KERNEL_SOURCE" "$BUILD"
```

The local Tapenade tree must come from the same receipt-writing pinned
installer; it supplies the `ADFirstAidKit` stack source and headers. Running
that installer independently on Linux and macOS with the same verified
archive produces the same host-independent receipt and complete tree
manifest. Java identity is intentionally recorded in generation provenance,
where Java is actually used, rather than in this cross-host installation
receipt.
`coupled_compiled/build_input_snapshot/` contains every declared
compile/link input; `build_input_manifest.sha256`, `compile_recipe.txt`, and
`build_manifest.txt` bind the snapshot, patches, tools, flags, and output
artifacts. This is exact declared-input provenance, not a promise that
executables are bitwise identical across operating systems. In particular,
macOS relinking may insert a different Mach-O UUID.

## Run one checkpoint

The user-facing runner copies immutable runtime inputs into a managed output,
constructs an independent authentic path, executes the reverse, checks the
configuration, sizes, finiteness, and manifests, and writes `run_report.json`:

```bash
python adjoint/tapenade_toolchain/scripts/run_coupled_case.py \
  --build-dir adjoint/tapenade_toolchain/build/coupled_release \
  --checkpoint-dir /path/to/checkpoint-runtime \
  --output-dir outputs/zc_adjoint/my_case
```

The checkpoint directory must contain `kernel_initial_state.bin`, `fc.data`,
`zeq9fsu.hst`, `modified_means.namelist`, `scales_EOF.namelist`, and `Data/`.
It is never modified. `--overwrite` can replace only an output carrying this
runner's ownership marker; symlinked, broad, overlapping, and unmarked targets
are rejected. The runner executes private copies of both compiled binaries and
the report-bound runtime files, then rechecks every staged executable and
runtime-input hash after the path and reverse runs before writing the report.

Outputs include:

- `artifacts/zc_31step_path.bin`: authentic boundary path;
- `artifacts/zc_nino3_gradient.bin`: 59,148 little-endian float32 values;
- scalar/path metadata and a 31-row transformed-forward replay report; and
- `run_report.json`: runtime verification, hashes, and build provenance.

For a generic `J^T w`, provide exactly 59,148 finite little-endian float32
terminal weights:

```bash
python adjoint/tapenade_toolchain/scripts/run_coupled_case.py \
  --build-dir adjoint/tapenade_toolchain/build/coupled_release \
  --checkpoint-dir /path/to/checkpoint-runtime \
  --terminal-seed /path/to/terminal_seed.bin \
  --output-dir outputs/zc_adjoint/my_functional
```

The runner reports execution and provenance but deliberately does not declare
a derivative certified. Certification additionally requires independent
primal replay, same-generation tangent/reverse transpose tests, and an
independent finite-difference or Taylor check.

## Validation status

The tracked, portable, source-free evidence is now under `results/run23/`.
Its precise status is **`provisional_major_tape_conditioned`**, not globally
certified. The bundle contains the exact original and sanitized report hashes,
same-generation source/build inventories, v2 producer records, three named
gradient packages, and a complete SHA-256 manifest. Its README is the concise
runbook for authorized users who possess the upstream ZC source.

The final3 explicit primal passed all 15 bitwise replay cases: 1, 10, 30, 31,
and 40 steps for extreme El Niño, extreme La Niña, and a neutral checkpoint.
All three 40-step fresh-process segmented replays and all six omitted-workspace
poison tests also passed. Here and below, tape stability refers only to `NSST`,
atmosphere iteration count, and reset occurrence; it does not cover every
internal sign, cap, upwind, or threshold branch.

The same-generation run23 reverse and independently compiled `-O3` audit
tangent passed three scalar-head and three dense-terminal transpose comparisons
at each checkpoint. Worst norm-product-scaled defects are `2.29e-9` for the
scalar head and `3.70e-8` for a generic terminal seed; the corresponding worst,
more cancellation-sensitive bilinear-relative defects are `5.58e-4` and
`6.89e-5`. Both normalizations are retained in the evidence.

One-step centered finite differences against the authentic-equation primal
have full-output relative L2 errors of `2.07e-6`--`2.31e-6` on the conservative
independent-control mask and `8.18e-5`--`1.70e-4` on broader active-carried
directions. The useful 31-transition aligned scalar checks have best recorded
major-tape-stable directional defects `3.36e-4` at `h=0.0025` for the extreme
warm case and `1.11e-2` at `h=0.05` for the neutral case. Every tested centered
cold-case perturbation changes `NSST` at transition 15. The composed random
full-state test changes the recorded tape for the warm and cold cases and is
explicitly classified as diagnostic, not a pass. No clean multi-level
quadratic Taylor regime was observed.

The final run23 scalar gradients are:

| Checkpoint | Final Niño-3 | Gradient SHA-256 | Typical reverse time on Mac |
|---|---:|---|---:|
| extreme El Niño | `+4.4466548 C` | `da5b61f288b25d31f4113c8df0ac7698aed26af4c6b68a668e78461500f4f9a1` | `0.365 s` |
| extreme La Niña | `-2.1435037 C` | `a8c57ea8111fa14f919973aafc0e4fd46b6dcf27addb406412e31a63a7b02231` | `0.368 s` |
| neutral member 04 | `-0.46085355 C` | `4df158b4bc5669d302a0ad2d7f52a0a3d3a43c2b8bc1662dc334a95ff37609dd` | `0.315 s` |

The raw gradients mix units and are dominated by carried atmospheric memory:
`Q0O` contributes 61.5% and 50.1% of squared norm in the warm and neutral cases,
while `UBAR` contributes 93.7% in the cold case. Only 28,591 entries belong to
the conservative independent-control mask. The full raw vector must not be
plotted or interpreted as physical feature importance; the balanced,
standardized observation lift needed for a direct AGOP comparison is absent.

The older MLOOP-only scripts are diagnostic development artifacts and are
quarantined by default. They are not a substitute for the coupled window
adjoint or part of its release recipe.

## Limitations and release boundary

- Derivatives are local to the branch sequence followed by the reference
  state. They can jump at atmospheric resets, adaptive SST substep changes,
  upwind/sign switches, and the SST cap.
- The compact boundary tape records `NSST`, atmospheric iteration count, and
  reset occurrence. It does not expose every internal sign/cap mask or
  distance to every switch.
- Only restart-mode execution and the locked Standard configuration are in
  scope. The generated initialization path is not a supported API.
- The generated one-step primal differs slightly from the authentic primal
  because source transformation changes some float32 evaluation details, so
  every step is reanchored. Across the three recorded 31-step cases, worst
  real-state relative L2 discrepancy is below `2.9e-8`; major branch tapes
  match exactly.
- The raw gradient is with respect to the complete native real state, not the
  four standardized neural-network channels. In particular, path/state 0
  `H1`, `U1`, and `V1` precede the coupling stage, while path/state 1 `TO` is
  advanced. Comparing 59,148 packed coefficients with the 2,162-feature AGOP
  requires a differentiated observation/time-alignment map, the frozen
  normalization transpose, and a balanced native-state lift/control map.
- Redistribution permission has been confirmed for the project-authored
  ZC-derived patches in `patches/` and coupled drivers in `coupled/`; both are
  part of the source release. The downloaded upstream ZC source,
  Tapenade-generated source, binaries, and local build trees remain excluded.
  Reproduction starts from the checksum-pinned upstream download and applies
  the released patches locally.
