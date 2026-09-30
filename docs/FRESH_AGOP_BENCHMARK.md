# Fresh-data exact AGOP timing benchmark

`scripts/benchmark_fresh_agop.py` measures the three costs that are otherwise
interleaved in the production exact-dense AGOP solver:

1. loading standardized inputs and evaluating one forecast gradient per
   reference input;
2. accumulating the float64 empirical matrix
   `M = G.T @ G / n` with upper-triangle BLAS `DSYRK` updates; and
3. computing every float64 eigenvalue and eigenvector with SciPy's symmetric
   `EVD` driver.

The default model is the 10-month, seed-42, `core4` CNN. A reference count of
zero means all 359,970 usable predictors wholly contained in the fixed
10,000-year training block. A positive count selects that many predictors from
the chronological beginning of the same block; it is a timing prefix, not a
random scientific subsample.

The benchmark writes a float32 `n x d` gradient cache so that gradient
evaluation and matrix algebra can be timed independently. For the default
`n=359970`, `d=2162` calculation, that diagnostic file is about 2.9 GiB. The
production solver does **not** require this cache: it streams each gradient
batch directly into the roughly 36 MiB dense matrix.

The timing values reported in the manuscript come from four allocated AMD
EPYC 7662 CPU cores, a gradient batch size of 1,024, and all 359,970 training
predictors. The cluster launcher keeps only compact reports and a hardware log
under `outputs/manuscript/benchmarks/xai_cpu4/`; its multi-gigabyte gradient
cache is created in temporary pod storage and deleted automatically:

```bash
bash scripts/launch_xai_timing_cpu_ucsd_cluster.sh
```

For an optional local Apple-MPS comparison, use a separate scratch directory:

```bash
caffeinate -i python scripts/benchmark_fresh_agop.py \
  --data-dir data/processed/zc-v3 \
  --artifacts-dir artifacts/zc-v3 \
  --input-profile core4 \
  --architecture cnn \
  --lead-months 10 \
  --seed 42 \
  --device mps \
  --reference-count 0 \
  --gradient-batch-size 1024 \
  --output-dir scratch/agop_timing_mps
```

The command atomically updates a machine-readable `report.json` after each
completed stage. Repeating the identical command validates and reuses all
completed caches. A changed model, reference population, normalizer, device,
software version, batch size, or source script changes the recorded identity;
use another output directory or explicitly pass `--overwrite` rather than
silently mixing results.

For a quick end-to-end check before committing to all training predictors, use
a separate 2,048-reference output:

```bash
python scripts/benchmark_fresh_agop.py \
  --device mps \
  --reference-count 2048
```

Do not compare a prefix run's gradient time directly with the all-reference
scientific run without labeling it as an extrapolation. The report separately
records standardized-input loading, autograd, gradient-cache writes, float64
cache conversion, `DSYRK`, eigendecomposition, output writes, and integrity
hashing. Thus file checks and diagnostic I/O are not misreported as AGOP
linear-algebra cost.

By default every processed field and auxiliary checksum is verified before a
run. `--skip-data-checksums` exists only for development and is permanently
recorded in the report when used. Reports contain content hashes, portable
artifact identifiers, software versions, and nonidentifying machine details;
they contain no home-directory or host-specific paths.

## End-to-end XAI method benchmark

After the full exact-AGOP benchmark above is complete,
`scripts/benchmark_fresh_xai_methods.py` times explanation generation for the
same `core4` CNN. The default target is the largest true Nino-3 value among
the fixed held-out test targets. The archived manuscript timing run uses one
independently selected nearest-1% neighbor so that it measures the cost of one
explanation, not the separate exhaustive robustness calculation.

The manuscript-production run is the CPU launcher above. Internally it invokes
the method benchmark with:

```bash
python scripts/benchmark_fresh_xai_methods.py \
  --device cpu \
  --neighbor-samples 1 \
  --ig-steps 1024 \
  --gradient-shap-samples 1024 \
  --gradient-batch-size 1024 \
  --fused-pair-batch-size 1024 \
  --execution-modes current
```

The scientific estimator settings are exactly:

- IG uses 1,024 equally spaced right-endpoint path gradients, at
  `alpha = 1/1024, 2/1024, ..., 1`, for every explanation.
- Expected gradients uses 1,024 **distinct** empirical training-predictor
  baselines sampled without replacement. Each baseline is paired once with an
  independently generated `Uniform(0,1)` interpolation alpha.
- The same ordered baseline indices and alpha values are reused for the target
  and every neighbor. This common-random-numbers design makes differences
  across explanations attributable to their query states rather than to new
  Monte Carlo draws.

The report times the target and neighbor population separately for GRAD, IG,
expected gradients (labeled GradientSHAP for manuscript continuity), and cached
exact AGOP. It also records the actual number of gradient rows,
`input_gradients` calls, and device batches. Here one **gradient row** means one
gradient of the scalar CNN forecast with respect to one interpolated,
standardized input state.

The current implementation constructs and evaluates one path or background
set at a time. An optional fused implementation enumerates query-sample pairs
in a canonical query-major order and accumulates each query's attribution
online. It never materializes the full `queries x samples x features`
cartesian array. For a separate 256-neighbor convergence study with fused
batch size 256, the workloads would be:

| Method/mode | Neighbor gradient rows | All 257 gradient rows | Neighbor batches | All 257 batches |
|---|---:|---:|---:|---:|
| GRAD | 256 | 257 | 4 | 5 |
| IG, current (batch 64) | 262,144 | 263,168 | 4,096 | 4,112 |
| IG, fused (batch 256) | 262,144 | 263,168 | 1,024 | 1,028 |
| Expected gradients, current (batch 64) | 262,144 | 263,168 | 4,096 | 4,112 |
| Expected gradients, fused (batch 256) | 262,144 | 263,168 | 1,024 | 1,028 |
| Cached exact AGOP | 0 | 0 | 0 | 0 |

Thus fusion changes scheduling and overhead, not the estimator or number of
model-gradient rows. The report compares current and fused explanation vectors
row by row. Small deterministic CPU tests also verify numerical equivalence to
the current explainers while confirming that no fused tile exceeds its declared
pair batch size.

Reusing expected-gradient baselines does **not** make their gradients globally
reusable. For query `x_j`, baseline `b_i`, and reused alpha `a_i`, the gradient
is evaluated at `b_i + a_i (x_j - b_i)`, which changes with `x_j`. Thus using
all 359,970 training predictors as baselines would require 359,970 model
gradients **per explanation**, or 92,512,290 gradient rows for the target and
256 neighbors. It is not analogous to AGOP's one-time global gradient
accumulation.

Counts remain explicitly configurable for convergence and timing studies. For
example, the former 300/1,000 choice can still be requested directly:

```bash
python scripts/benchmark_fresh_xai_methods.py \
  --ig-steps 300 \
  --gradient-shap-samples 1000 \
  --execution-modes fused
```

A high-sample expected-gradients stress test can likewise be labeled explicitly
rather than confused with the production default:

```bash
caffeinate -i python scripts/benchmark_fresh_xai_methods.py \
  --device mps \
  --ig-steps 1000 \
  --gradient-shap-samples 10000 \
  --execution-modes fused
```

Target and neighbor results are atomically cached independently, so the long
neighbor stage can be resumed without recomputing a completed target stage.
The complete cold AGOP gradient-generation, dense-accumulation, and EVD timing
is copied from the validated exact-AGOP report into a separate
`agop_cold_build` section. Those one-time costs are never conflated with the
cached AGOP explanation-application time.
