# Packaging a ZC adjoint gradient

The reverse driver writes a headerless stream containing 59,148
single-precision coefficients in the byte order of its host. The current
validated hosts are little-endian, and the packager defaults explicitly to
little-endian input. Package a local result from the repository root with

```bash
python scripts/package_zc_adjoint_gradient.py \
  outputs/zc_adjoint/example/zc_nino3_gradient.bin \
  --byte-order little
```

This creates adjacent `zc_nino3_gradient.json` and
`zc_nino3_gradient.npz` files. Existing output is protected unless
`--overwrite` is supplied. Use `--byte-order big` only for a stream known to
have been written by a big-endian host; `native` remains available for local
diagnostics but is not recommended in a portable release command.

The two files are built from descriptor-stable snapshots, staged as one bundle,
and individually installed with atomic filesystem operations. Every bound
producer artifact is reverified immediately before commit. If either
installation fails or the process receives an exception-like signal, both
prior files are restored. The NPZ is installed first and the JSON commit record
last; consumers must verify `archive.sha256` in the JSON before accepting a
pair. The packager
also refuses to replace the gradient, state manifest, producer metadata, or any
checkpoint, path, executable, or evidence file referenced by that metadata,
including aliases through symbolic or hard links.

The program requires an exact 236,592-byte input, rejects every NaN or
infinity, records input/manifest/archive SHA-256 hashes, and reports global,
activity-class, independent-control, and per-segment norms, maxima, and
nonzero counts. The NPZ contains `packed_gradient`,
`independent_control_mask`, and one array under each exact state-manifest name,
including `TO`, `U1`, `V1`, and `H1`. Multidimensional arrays use their logical
Fortran shapes and column-major index convention. The JSON keeps mask metadata
under `independent_control_mask` and the coefficients selected by that mask
under `independent_control_gradient_summary`.

Passing this format check does **not** validate the derivative. The report uses
`format_validation.status = "passed"` only for byte count, byte order, finite
values, and manifest layout. It separately records
`adjoint_certification.assessed_by_packager = false`. In particular, a finite
all-zero array can be packaged successfully and must not be called a validated
adjoint.

## Binding producer and validation provenance

For a result intended to carry a producer-declared validation status, pass a
strict metadata document:

```bash
python scripts/package_zc_adjoint_gradient.py \
  outputs/zc_adjoint/example/zc_nino3_gradient.bin \
  --byte-order little \
  --producer-metadata outputs/zc_adjoint/example/producer_metadata.json
```

The metadata schema is version 2 and has this structure (hashes abbreviated
here only; real inputs require complete lowercase SHA-256 digests):

```json
{
  "schema_version": 2,
  "objective": {
    "name": "canonical_nino3",
    "definition": "Mean of the 66 terminal TO cells in the locked Nino-3 box.",
    "units": "degrees Celsius",
    "terminal_seed": "uniform 1/66 on the 66 terminal TO cells"
  },
  "transitions": 31,
  "initial_checkpoint": {
    "label": "extreme_el_nino",
    "path": "kernel_initial_state.bin",
    "sha256": "..."
  },
  "certified_path": {
    "path": "zc_31step_path.bin",
    "sha256": "..."
  },
  "gradient": {
    "path": "zc_nino3_gradient.bin",
    "sha256": "..."
  },
  "source_manifests": {
    "prepared": {
      "path": "prepared_compile_inputs.sha256",
      "sha256": "..."
    },
    "tangent": {
      "path": "tangent_compile_inputs.sha256",
      "sha256": "..."
    },
    "reverse": {
      "path": "reverse_compile_inputs.sha256",
      "sha256": "..."
    }
  },
  "reverse_executable": {
    "path": "zc_nino3_adjoint",
    "sha256": "..."
  },
  "tapenade": {
    "version": "3.16",
    "revision": "0449e37b7da896cb2c38b63c8a34f10c87a84534",
    "archive_sha256": "534b83598c508a3202880016403f84e352e4b00bdc4c6b934bb39ccbdddb0963"
  },
  "certification": {
    "status": "provisional",
    "scope": "Locked real32 31-transition scalar objective.",
    "evidence": [
      {"kind": "primal_replay", "path": "replay_report.json", "sha256": "..."}
    ]
  }
}
```

Artifact paths are resolved relative to the metadata document unless absolute.
Prepared, tangent, and reverse source hashes must be supplied as actual
manifest-inventory artifacts, not as unverified digest strings. The packager
verifies every referenced file against its declared digest,
requires the declared gradient digest to match the binary being packaged, then
publishes only its basename, size, and hash so machine-local absolute paths do
not leak into the package. It also hashes the metadata file and its canonical
JSON content.

The allowed status values are `uncertified`, `provisional`, and `certified`.
The last is rejected unless the metadata supplies and hash-verifies all three
evidence classes: `primal_replay`, `tangent_reverse_dot`, and
`independent_finite_difference_or_taylor`. Even then, `certified` remains the
producer's declaration; the packager binds that declaration to evidence but
does not adjudicate whether the scientific acceptance criteria were sensible
or met. For the present ZC derivative, use `provisional`: the external tape
does not expose all piecewise-physics masks, no real64 twin exists, and the
historical long-window checks did not show a clean multi-level quadratic Taylor
regime.

These are derivatives with respect to the **raw packed Fortran state**, with
no standardization or physical-unit conversion. Since the packed vector mixes
variables and units, its raw per-segment norms are not standardized variable
importance scores. Moreover, only manifest segments marked
`independent_control=true` are in the conservative direct-intervention mask.
Other active fields are carried state needed for replay; diagnostic and passive
fields are not independent scientific controls, even if a numerical reverse
coefficient is present.
