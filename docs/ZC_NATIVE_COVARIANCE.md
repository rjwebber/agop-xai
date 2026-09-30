# Native ZC covariance

## Canonical annual geometry

The steering experiments use one covariance geometry for the 28,591
independent coordinates of the native Zebiak--Cane restart state. It is the
same covariance for warm and cold targets and for every one of the nine
intervention times.

For annual phase `p`, let `X_p` contain one native state from each of the
10,000 training years and let `mu_p` be its row mean. With `q=36` phases and
`n=10,000` states per phase,

```text
C_p = (X_p - 1 mu_p^T)^T (X_p - 1 mu_p^T) / (n - 1)
C   = (1/q) sum_{p=0}^{35} C_p.
```

Equivalently, `C` accumulates all 360,000 training time points after removing
a separate mean at each phase, with denominator `q(n-1) = 359,964`. This is
not the ordinary covariance of the uncentered seasonal mixture and therefore
does not charge or reward the deterministic drift among the 36 phase means.
Equal sample counts make equal-phase and equal-observation weighting identical.

There is no mathematical obstacle to this construction. The pooled centered
sample rank is at most `min(28,591, 36 * 9,999) = 28,591`, so unlike either
nine-phase factor alone it is not forced to be rank deficient by sample count.
The steering code still applies `C` directly and never forms `C^-1`.

## Capture, factors, and dense compilation

The production history supplies 1,000 certified decade-boundary checkpoints
inside the half-open training interval `[0, 360000)`. Replaying each checkpoint
through offset 359 supplies ten complete years and every phase `0,...,35`.
The resulting indices cover all 360,000 training time points exactly once.
Phase is assigned by integer time-step index modulo 36, not by the slightly
drifting stored float32 sine/cosine clock.

The capture is bound to the canonical history, production report, replay
executable, independent-state manifest, and processed-data metadata. Its
overlap audit independently replays the first annual row from every source
checkpoint and checks it bit for bit against the certified decadal cache.

The complete float32 phase arrays contain 10,000 by 28,591 values for each of
36 phases: 41,171,044,608 bytes including NumPy headers (about 38.34 GiB).
The factor manifest records the 36 distinct phase means and exact rectangular
factors. A resumable compiler then accumulates the upper triangle in float64,
mirrors it, and publishes one read-only Fortran-order `28,591 x 28,591` dense
matrix. Its payload is 6,539,562,248 bytes (about 6.09 GiB). This changes
storage and solve-time I/O, not the empirical covariance definition.

The canonical build commands use all-phase defaults:

```bash
python scripts/capture_zc_annual_native_phases.py \
  --independent-overlap-replay

python scripts/build_zc_native_covariance_factors.py

python scripts/build_zc_all_phase_dense_covariance.py
```

The published steering input is:

```text
outputs/zc_native_covariance/training-years10000-phases00-35/dense_manifest.json
```

The dense compiler checkpoints after each completed phase, verifies source
hashes, audits the matrix diagonal and symmetry, and compares independent
matrix-vector products against the factor operator before publication. Its
`--help` output documents resume, memory-block, and overwrite controls.

Python consumers use the same operator protocol as the steering solver:

```python
from zc_xai.native_covariance import DenseNativeCovarianceOperator

common = DenseNativeCovarianceOperator.load(
    "outputs/zc_native_covariance/"
    "training-years10000-phases00-35/dense_manifest.json",
    verify_hashes=True,
)

covariance_direction = common.covariance_apply(native_covector)
all_directions = common.covariance_apply_matrix(native_covectors)
```

The dense manifest records `d=28,591`, `n=10,000`, `q=36`, phase offsets
`0,...,35`, matrix shape/dtype/order/hash, phase-specific centering, the
half-open training interval, and the data and state-layout hashes. The
steering driver validates all of these fields before opening the matrix.

## Reproducibility boundary

The phase arrays are large derived intermediates, not required publication
data. A reproducible release can provide the capture, factor, and dense-build
code plus hashes of the canonical ZC sources, executables, and inputs. No
validation/test state or future outcome enters the training population or the
covariance construction.
