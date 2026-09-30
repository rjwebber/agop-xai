# Balanced controls for fresh ZC core4

This directory contains an isolated empirical control map for perturbing the
independent native Zebiak--Cane (ZC) state in directions that occur in
authentic, same-season model states.  It does not modify or depend on the
run23 final3 build directory, and it does not replace the run23 qualification.

The main artifact exposes two explicit matrices:

- `B`, shape `(28591, r)`: dimensionless reduced controls to native
  independent-control increments (`AKB`, `UB`, `HB`, `V`, and `TO`);
- `G`, shape `(2160, r)`: the same controls to training-standardized fresh
  core4 increments.

`B.T` and `G.T` are used directly, so their transpose behavior is algebraic
rather than an independently approximated reverse map.

## Paired state convention

For a selected input row `i`, the native sample is the complete restart
immediately **before** the model consumes input `i`.  Its paired observation
is fresh core4 row `i`, produced by the update immediately after that
checkpoint.  The fresh `H1`, `U1`, and `V1` fields are carried from the
pre-input checkpoint; SST is advanced once by `SSTA` before output.  Thus `G`
is an empirical paired one-step/secant image, not the exact differentiated
fresh observation operator.

The production history stores a checkpoint every 360 ten-day steps.  The
builder accepts `--reference-input-index` and uses only its annual phase.  For
the neutral member-04 preconditioning checkpoint, the reference index is
`402758`, hence the phase offset is `402758 % 36 = 26`.  Every training-only
sparse checkpoint is authentically advanced by 26 steps and then paired with
fresh row `base_index + 26`.  The concrete rank-36 artifact therefore uses

```text
i_j = 26 + 360 j,  j = 0, ..., 999,
```

all inside the fresh training interval `[0, 360000)`.  No validation/test
state, forecast target, neural-network explanation, or adjoint gradient is
used to fit the map.

## Derivation

Let rows of `X` be the paired compact native states and rows of `Z` be the
paired standardized core4 states.  With column means removed, form the sample
Gram matrix

```text
K = Zc Zc.T / (n - 1) = U Lambda U.T.
```

For the leading `r` positive modes, define sample reduced coordinates

```text
A = sqrt(n - 1) U_r.
```

They are centered (up to eigensolver roundoff) and empirically whitened:

```text
A.T A / (n - 1) = I.
```

The least-squares maps from these coordinates to each paired space are

```text
B = Xc.T U_r / sqrt(n - 1),
G = Zc.T U_r / sqrt(n - 1).
```

Consequently `delta_x = B a`, `delta_z_hat = G a`, and
`G.T G = diag(lambda_1, ..., lambda_r)`.  The action `||a||^2` is an empirical
Mahalanobis action in the retained coordinates.  `B.T B` is not an identity
and must not be interpreted as one: rows of `B` retain their heterogeneous
native model units.

This construction differs from the old minimum-Euclidean-norm `ZAVG` lift.
It cannot exactly prescribe every coarse field value, but its native
increments follow covariance-linked combinations found in authentic model
states instead of independently inverting the diagnostic averaging operator.

## Lifting a requested core4 direction

For a standardized core4 increment `d`, the public lift minimizes

```text
0.5 ||G a - d||^2 + 0.5 alpha ||a||^2,
alpha = ridge_fraction * lambda_1.
```

The default `ridge_fraction` is `1e-3`.  With zero ridge, the result is the
orthogonal projection into `range(G)`; it is exact only when `d` is already in
that range.  A rank-36 `G` cannot invert an arbitrary 2,160-dimensional
direction.  Callers should always retain the reported unresolved residual and
the action `||a||`, rather than presenting `B a` as an exact lift.

For a native packed adjoint covector `g_x`, composition is exact:

```text
delta_J = g_x.T B a = (B.T g_x).T a.
```

Only the manifest's 28,591 independent coordinates contribute.  Active
boundary and atmospheric-memory coordinates excluded by the manifest are
left at zero by `apply_B(..., packed=True)`.

## Public API

```python
from adjoint.balanced_control_map import BalancedControlMap

model, metadata = BalancedControlMap.load("balanced_control_map.npz")

packed_increment = model.apply_B(a, packed=True)
packed_matrix = model.packed_control_matrix()  # explicit E @ B for ZC tangents
reduced_gradient = model.apply_BT(packed_gradient, packed=True)
predicted_core4 = model.apply_G(a)
reduced_core4_covector = model.apply_GT(core4_covector)
solution = model.lift_observation(
    standardized_core4_direction,
    ridge_fraction=1e-3,
    packed=True,
)
```

Reusable state helpers are also exported:

- `load_independent_control_layout(path)`;
- `read_restart_payload(payload, layout)`;
- `read_restart_sample(path, layout)`;
- `layout.compact_to_packed(values)` and `layout.packed_to_compact(values)`;
- `fit_balanced_control_map(...)` and
  `paired_reconstruction_diagnostics(...)`.

The final axis is the state/control axis for all apply and layout methods, so
both individual vectors and batches are accepted.

## Coherent phase-local stack

A distributed nine-step intervention should not reuse the phase-26 map at
every step: the relationship between the native state and the observed fields
changes through the seasonal cycle.  It is also not valid to concatenate nine
separately fitted PCAs, because their signs and orthogonal rotations are
arbitrary.  `PhaseLocalControlStack` resolves both issues with one common
sample-space basis fitted from the same 1,000 training trajectories at offsets
26 through 34.  For phase `k`,

```text
K = mean_k(Zc_k Zc_k.T) / (n - 1) = U Lambda U.T,
B_k = Xc_k.T U_r / sqrt(n - 1),
G_k = Zc_k.T U_r / sqrt(n - 1).
```

Thus coordinate `j` represents the same contrast of matched training
trajectories at every phase, while `B_k` and `G_k` are allowed to evolve.  The
concrete stack is built with:

```bash
python adjoint/balanced_control_map/build_phase_local_stack.py \
  --jobs 4 --overwrite
```

The builder advances each sparse ten-year checkpoint only once, retains the
nine requested intermediate restart records, and can safely resume the large
native capture with `--reuse-native-cache`.  An old cache without a completion
manifest is adopted only after full schema/finiteness checks and bitwise
authentic replay audits of its first, middle, and last rows at every phase.

```python
from adjoint.balanced_control_map import PhaseLocalControlStack

stack, metadata = PhaseLocalControlStack.load("phase_local_control_stack.npz")
packed_B = stack.packed_control_stack()  # (9, 59148, 36)
phase_30_increment = stack.apply_B(a, phase_offset=30, packed=True)
```

The checked rank-36 artifact is
`outputs/zc_balanced_control_map/training-phases26-34-common-rank36/phase_local_control_stack.npz`
(SHA-256
`1d3fd6300e9104154a01e7147257a0fb5456a1554f3e089d83b7e219490e15c9`).
It retains `0.9365191` of pooled standardized core4 variance.  On the
chronological training-only 800/200 diagnostic split, the phasewise median
core4 reconstruction cosine ranges from `0.9634` to `0.9646`.  Its explicit
packed stack has shape `(9, 59148, 36)` and array SHA-256
`f1e0d5129a7a1834f2c92856c0a80b5221f412e5eac31851f40a79d604bab1d9`.

This is the primary map for a phase-26-through-34 intervention.  Repeating the
single phase-26 map remains useful only as a sensitivity experiment.

## Concrete rank-36 result

Build from the repository root with:

```bash
python adjoint/balanced_control_map/build_training_map.py --overwrite
```

The checked artifact is
`outputs/zc_balanced_control_map/training-phase26-rank36/balanced_control_map.npz`
(SHA-256
`825f4ba436cdec0cecffda6c836effd2466057e9586b46a035fc13a1b1756f5b`).
Its adjacent `report.json` binds the production history, manifest,
normalization, executable, and training row indices.

For that artifact:

- retained standardized core4 sample variance: `0.9398906`;
- retained eigenvalue condition number: `186.24`;
- packed `B/B.T` maximum relative dot defect over eight seeded trials:
  `1.74e-13`;
- `G/G.T` maximum relative dot defect: `2.32e-15`;
- chronological training-only 800/200 holdout median core4 cosine: `0.9656`;
- holdout median standardized core4 RMS residual: `0.2288`.

These diagnostics support a useful low-rank phase-local map.  They do not
show that a finite perturbation remains exactly on the nonlinear state
manifold.  Large actions, other annual phases, and any intervention that
requires active memory to be adjusted must be validated by authentic forward
replay.  The packaged run23 gradients remain
`provisional_major_tape_conditioned`; projection by `B.T` preserves that
qualification and adds no certification.

## Tests

```bash
python -m unittest tests.test_zc_balanced_control_map -v
python -m unittest tests.test_zc_phase_local_control_stack -v
python -m ruff check adjoint/balanced_control_map \
  tests/test_zc_balanced_control_map.py \
  tests/test_zc_phase_local_control_stack.py
```

The isolated tests cover the packed independent-control mask, fresh restart
record parsing, `B/B.T` and `G/G.T` identities, exact-vs-unresolved range
behavior, serialization without pickle, and paired reconstruction metrics.
