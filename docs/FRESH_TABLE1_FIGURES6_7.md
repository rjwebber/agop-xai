# Table 1 and Figures 6--7

These outputs use the `zc-v3` `core4+phase` input: SST anomaly, thermocline
depth, zonal and meridional ocean current on the `20 x 27` active grid, plus
two scalar annual-phase coordinates.

The XAI contract is fixed:

- Integrated Gradients uses 1,024 gradients on the straight path from zero;
- GradientSHAP uses 1,024 distinct training-predictor backgrounds sampled
  without replacement and one independent interpolation coefficient per
  background;
- gradients are evaluated in batches of 1,024;
- AGOP is formed exactly from every lead-valid training predictor and uses its
  complete dense eigendecomposition; and
- robustness averages over every one of the 3,600 predictors in the nearest
  1% of the training population, with no Monte Carlo subsampling.

Validation and test predictors never enter the XAI reference sets.

## Render the archived results

From the repository root after unpacking the Zenodo artifact bundle:

```bash
python scripts/make_fresh_table1.py \
  --table-dir artifacts/zc-v3/manuscript/table1_figure6 \
  --output outputs/manuscript/tables/table1_xai_scores.tex \
  --overwrite

python scripts/make_fresh_figure6.py \
  --data-dir data/processed/zc-v3 \
  --table-dir artifacts/zc-v3/manuscript/table1_figure6 \
  --output outputs/manuscript/figures/figure6_fresh_core4.pdf \
  --overwrite

python scripts/make_fresh_figure7.py \
  --bundle-root artifacts/zc-v3/manuscript/figure7 \
  --output outputs/manuscript/figures/figure7_zcv3_core4.pdf \
  --summary-output outputs/manuscript/figures/figure7_zcv3_core4_scores.csv \
  --overwrite
```

Figure 6 shows the selected extreme event for the independently trained MLP,
CNN, and ViT. Figure 7 evaluates the CNN separately for seeds 42--46 and plots
the arithmetic mean of the five scores. It does not average model checkpoints
or explanation vectors.

## Recompute Table 1 and Figure 6 inputs

The production generator is resumable at the neighbor-chunk boundary. Use a
scratch directory until the complete bundle passes validation:

```bash
python scripts/generate_fresh_table1_data.py \
  --data-dir data/processed/zc-v3 \
  --artifacts-dir artifacts/zc-v3 \
  --agop-benchmark-root scratch/table1_exact_agop \
  --output-dir scratch/table1_figure6 \
  --device auto
```

If interrupted, rerun the identical command. Do not add `--overwrite`; valid
completed chunks are reused. Architectures can instead be filled sequentially
with `--architectures mlp`, `cnn`, and `vit`. Do not write concurrently to one
output directory.

Once complete, compare the generated hashes and scores with
`artifacts/zc-v3/manuscript/table1_figure6` before replacing any archived
artifact.

## Recompute Figure 7 inputs

Every CNN seed is evaluated independently. The following loop is portable to a
local CPU, Apple MPS, or CUDA machine:

```bash
for seed in 42 43 44 45 46; do
  python scripts/generate_fresh_figure7_data.py \
    --data-dir data/processed/zc-v3 \
    --artifacts-dir artifacts/zc-v3 \
    --agop-cache-root scratch/figure7_exact_agop \
    --output-dir "scratch/figure7/seed-$(printf '%06d' "$seed")" \
    --seed "$seed" \
    --device auto \
    --resume-complete
done
```

The exact AGOP calculation is the expensive step. Rerunning the loop validates
and skips complete bundles and reuses valid AGOP caches. Use
`--overwrite-agop` only when intentionally rebuilding those matrices.

After all seeds finish, point `make_fresh_figure7.py` at
`scratch/figure7`. The renderer refuses incomplete or configuration-mismatched
bundles.

## Spatial display convention

The explanation figures show standardized thermocline depth and standardized
ocean-current components. SST and the two phase coefficients remain part of
every full explanation and every applicable score; phase is excluded only from
the spatial maps and spatial-coherence calculation.
