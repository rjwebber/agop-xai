# Exact dense AGOP computation

For the primary four-ocean-field model, the standardized input has 2,162
coordinates: four `20 x 27` spatial fields followed by annual-phase sine and
cosine. At this dimension, the empirical average gradient outer product

\[
M = \frac{1}{n}\sum_{i=1}^n \nabla f(x_i)\nabla f(x_i)^\mathsf{T}
\]

is small enough to form and diagonalize exactly.

The production solver `build_exact_dense_agop_factor` streams every training
gradient batch into a float64 upper-triangle BLAS `DSYRK` accumulation and
applies a complete float64 symmetric eigendecomposition. It does not retain
the full `n x d` gradient matrix. The matrix and complete eigenvector array use
about 71 MiB at `d=2162`.

For the 10-month experiment, the empirical population is every usable
predictor date wholly contained in the 10,000-year training block: 359,970
inputs. All 360,000 training-block states estimate the input normalization;
the last 30 cannot be predictors for a 10-month target without crossing the
training boundary.

This exact calculation avoids reference subsampling, truncation rank,
oversampling, and iteration-depth choices. Neural-network gradient evaluation,
rather than dense matrix formation or eigendecomposition, is the main cost at
the manuscript dimension. `scripts/benchmark_fresh_agop.py` records gradient,
accumulation, eigendecomposition, integrity-check, and file-publication time
separately. See `docs/FRESH_AGOP_BENCHMARK.md` for the reproducible benchmark
commands and the companion XAI-method timing protocol.

If a future input representation makes a dense `d x d` matrix impractical,
the fallback should preserve at least one complete pass through the training
gradients. A deterministic Frequent Directions sketch is a natural one-pass
choice. When multiple passes or matrix--block products are affordable, a
Nyström block-Krylov method is preferable to basic randomized subspace
iteration. These approximations are not used in the manuscript experiments.
