# Reproducing the fresh Zebiak--Cane data set

The processed data release does not redistribute the upstream Zebiak--Cane
source. The public source used here is hosted on Eli Tziperman's Harvard
download page:

- Download page: <https://groups.seas.harvard.edu/climate/eli/Downloads/>
- Source archive: <https://groups.seas.harvard.edu/climate/eli/Downloads/CZ_model_share.zip>
- Expected archive SHA-256: `4b737051d7c9f27c54531ef565c19a99c7edf47a07fffdb9fbc976327bf318ee`

One reproducible macOS download sequence is:

```bash
mkdir -p data/raw
curl -L \
  https://groups.seas.harvard.edu/climate/eli/Downloads/CZ_model_share.zip \
  -o data/raw/CZ_model_share.zip
shasum -a 256 data/raw/CZ_model_share.zip
unzip data/raw/CZ_model_share.zip -d data/raw
```

Proceed only if `shasum` prints the expected digest above.

## Build prerequisites and tested environment

The generator requires Python 3 with NumPy, GNU `make`, and GNU Fortran
(`gfortran`). On macOS it also requires Apple's Command Line Tools so that
`xcrun --show-sdk-path` resolves an SDK. A typical installation is:

```bash
xcode-select --install
brew install gcc
python -m pip install numpy
```

Confirm the programs that will actually be used before starting:

```bash
gfortran --version
make --version
xcrun --show-sdk-path
python -c "import numpy; print(numpy.__version__)"
```

The published reference stream was generated on arm64 macOS 26.6.2 with
`GNU Fortran (Homebrew GCC 13.1.0) 13.1.0`, little-endian storage, and flags
`-std=legacy -O3 -I. -ffixed-line-length-none`. The metadata records the
compiler, platform, flags, patched-source hashes, executable hash, and source
manifest. A different compiler, architecture, or optimization may produce a
scientifically equivalent stream that is not bit-for-bit identical.

After downloading and extracting the archive, run the preflight. It compiles
an isolated copy, adds output and restart code without changing the model
equations, checks the original ten fields bit-for-bit against the model's
legacy output, and runs the 150-year restart certification described below.

```bash
cd /path/to/AGOP-XAI

python scripts/generate_fresh_zc_dataset.py preflight \
  --source-dir data/raw/CZ_model_share \
  --workspace outputs/zc_generation/zc-v3 \
  --output-dir data/processed/zc-v3
```

Only after the preflight passes, run the 100-year spinup plus 12,000 retained
years. This is intentionally a separate command so a failed preflight cannot
launch the expensive integration.

```bash
python scripts/generate_fresh_zc_dataset.py generate \
  --source-dir data/raw/CZ_model_share \
  --workspace outputs/zc_generation/zc-v3 \
  --output-dir data/processed/zc-v3
```

Finally, convert the compact native stream into one NumPy file per field,
compute the center-inclusive Nino-3 target, and select and verify complete
native restart checkpoints for the strongest El Nino and La Nina events.

```bash
python scripts/generate_fresh_zc_dataset.py finalize \
  --source-dir data/raw/CZ_model_share \
  --workspace outputs/zc_generation/zc-v3 \
  --output-dir data/processed/zc-v3
```

Use exactly the same `--spinup-years`, `--retained-years`,
`--checkpoint-years`, and `--event-lead-months` values for all three commands.
The generator refuses to continue if the current script, source tree,
executable, configuration, preflight report, production report, compact stream,
or history file differs from the object certified at the preceding stage.

Finalization verifies the native stream and checkpoint history in full,
including SHA-256, byte counts, layout, shape, checkpoint count, and checkpoint
record size. It then builds the release in a sibling staging directory,
validates every staged output, and atomically publishes it. An existing valid
output directory is left untouched if any earlier step fails. Replacing one
requires `--overwrite`; even then, the old directory remains in place until the
new staged release has passed validation.

The released spatial data have shape `(time, 20, 27)` and contain 13 fields:
the four primary ocean fields, the six remaining legacy fields, and mixed-layer
zonal current, mixed-layer meridional current, and upwelling anomaly. Annual
phase is stored once as two scalar columns (sine and cosine), not duplicated
over the grid. `metadata.json` records field meanings, exact grid centers,
source and executable fingerprints, run parameters, chronological splits, and
restart verification results. `generation_provenance.json`, intended to
accompany the arrays on Zenodo, contains sanitized, relocatable generator,
metadata, report, and upstream-source hashes. It contains no
workstation-specific absolute paths.

Sparse complete native checkpoints are stored every ten simulated years only
in the temporary workspace. After the extreme events are known, the finalizer
replays from the nearest sparse checkpoint and retains an exact checkpoint
immediately before each ten-month-lead input. Restarting from either retained
checkpoint reproduces all 13 saved fields bit-for-bit through its event target.

The restart preflight is deliberately longer than a smoke test. It runs 150
model years, confirms that the atmosphere's forced reinitialization guard is
actually encountered, restarts from an annual checkpoint immediately before
that event and continues through it, and separately restarts after it. Both
continuations must reproduce all 13 fields and the forced-reset timestamps
bit-for-bit. On the recorded reference toolchain, the complete 12,000-year
compact stream must also match SHA-256
`46ca95ead9db93ef023d35ea4a447f105926d74a5a9431ac92655b3e7d16bb9f`,
proving that the added restart bookkeeping did not alter the uninterrupted
model trajectory. This exact digest is scoped to the recorded toolchain. On a
portable build with a different toolchain, all schema, finite-value,
legacy-field, forced-reset, and bitwise restart-continuity checks must still
pass, and the release records the new stream hash. Scientific equivalence
should additionally be assessed from the recorded Nino-3 distribution and
summary diagnostics; the reference digest is not claimed to be
compiler-independent.

This certification applies to the public Standard configuration used for the
data release: no WWB compile option, `mask_heating` false, seasonal-background
freezing disabled, mid-run SST-dissipation changes disabled, and `NIC=0` for a
continuation. Other optional branches have additional persistent state and are
not claimed to be restart-certified by this pipeline.

For modeling, the first 10,000 retained years are the training block, the next
1,000 years are validation, and the last 1,000 years are test. Lead-specific
input/target pairs must stay within their own block. Normalization statistics
for every selected spatial coordinate and the two phase scalars are fit on all
raw states in the training block only. The retained extreme El Nino and La Nina
restart cases are selected only from valid ten-month-lead targets in the test
block.
