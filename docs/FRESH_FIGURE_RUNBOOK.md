# Final `zc-v3` manuscript runbook

This runbook rerenders Figures 3--12, Table 1, and the released video from
the final archived data and artifacts. The commands below contain no
machine-specific paths or cluster-only instructions.

The scientific configuration is fixed:

- data generation: public `zc-v3`;
- chronological split: 10,000 training years, 1,000 validation years, and
  1,000 test years;
- model input: SST anomaly, thermocline depth, zonal and meridional ocean
  current, plus two scalar annual-phase coordinates (`core4+phase`, `d=2162`);
- XAI reference population: lead-valid training predictors only;
- IG and GradientSHAP: 1,024 gradients per explanation;
- AGOP: gradients at every lead-valid training input; and
- robustness: every member of the nearest 1% training neighborhood.

Figures 1 and 2 are manuscript-native diagrams and are not generated here.

## 1. Prepare a clean environment

Clone the GitHub repository, unpack the companion Zenodo archive at its root,
and then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m ruff check .
```

The unpacked archive must provide:

```text
data/processed/zc-v3/
data/external/noaa/
artifacts/zc-v3/
outputs/manuscript/source_data/
```

Create the final destination directories if they are absent:

```bash
mkdir -p outputs/manuscript/figures
mkdir -p outputs/manuscript/tables
mkdir -p outputs/video
```

The commands below replace existing products deliberately with `--overwrite`.
Their JSON sidecars record the exact inputs and output hashes.

## 2. Figure 3: geographic domains

```bash
python scripts/make_figure3.py \
  --data-dir data/processed/zc-v3 \
  --output outputs/manuscript/figures/figure3_noaa_ersstv5.pdf \
  --overwrite
```

The Zenodo data tree includes the checksum-pinned NOAA ERSSTv5 climatology so
the SST background can be reproduced offline. If it is absent, the script
downloads and verifies the same file. Cartopy may still fetch its Natural
Earth coastline data on first use.

## 3. Figure 4: 100 years of CNN forecasts

```bash
python scripts/make_figure4.py \
  --data-dir data/processed/zc-v3 \
  --artifacts-dir artifacts/zc-v3 \
  --input-profile core4 \
  --segment-years 100 \
  --prediction-batch-size 1024 \
  --device auto \
  --data-output outputs/manuscript/figures/figure4_cnn_forecasts.csv \
  --output outputs/manuscript/figures/figure4_cnn_forecasts.pdf \
  --overwrite
```

This uses the archived seed-42 CNNs at 5- and 10-month lead times. It does not
retrain a model.

## 4. Figure 5: predictive skill

```bash
python scripts/audit_figure5_results.py \
  --results outputs/manuscript/source_data/figure5/figure5_zcv3_results.csv \
  --artifacts-dir artifacts/zc-v3 \
  --expected-left-repetitions 5 \
  --expected-right-repetitions 5

python scripts/make_figure5.py \
  --results outputs/manuscript/source_data/figure5/figure5_zcv3_results.csv \
  --output outputs/manuscript/figures/figure5_predictive_skill.pdf \
  --overwrite
```

The source CSV contains all five seeds, 42--46. The figure plots their
arithmetic mean and does not show uncertainty bars.

## 5. Table 1 and Figure 6: primary XAI comparison

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
```

The archived bundle is the final training-only calculation. It contains exact
nearest-1% neighborhoods and exhaustive robustness comparisons; no
development-period reference points are used.

## 6. Figure 7: five-seed XAI scores

```bash
python scripts/make_fresh_figure7.py \
  --bundle-root artifacts/zc-v3/manuscript/figure7 \
  --output outputs/manuscript/figures/figure7_zcv3_core4.pdf \
  --summary-output outputs/manuscript/figures/figure7_zcv3_core4_scores.csv \
  --overwrite
```

The renderer validates five complete, independent bundles for CNN seeds
42--46 and gives every seed equal weight. It does not average model weights or
explanations.

## 7. Figures 8 and 9: ENSO pathways

```bash
python scripts/make_fresh_figures8_9.py \
  --data-dir data/processed/zc-v3 \
  --bundle-dir artifacts/zc-v3/manuscript/figures8_9 \
  --output-a outputs/manuscript/figures/figure8_el_nino_multilead.pdf \
  --output-b outputs/manuscript/figures/figure9_la_nina_multilead.pdf \
  --overwrite
```

Both figures use the same 1-, 5-, and 10-month CNNs. The warm and cold panels
are target-locked to the selected test-period events rather than selected
independently at each lead.

To recompute the shared numerical bundle rather than render the archived one:

```bash
python scripts/generate_fresh_figures8_9_data.py \
  --data-dir data/processed/zc-v3 \
  --artifacts-dir artifacts/zc-v3 \
  --agop-root scratch/figures8_9_exact_agop \
  --output-dir scratch/figures8_9_bundle \
  --device auto
```

Then pass `--bundle-dir scratch/figures8_9_bundle` to the renderer above.

The field-mass percentages cited in the accompanying text can be regenerated
from the archived 1--12-month exact factors with:

```bash
python scripts/analyze_agop_field_mass.py \
  --data-dir data/processed/zc-v3 \
  --artifacts-dir artifacts/zc-v3 \
  --factor-dir artifacts/zc-v3/manuscript/field_mass_crosslead \
  --output-csv outputs/manuscript/source_data/diagnostics/agop_field_mass_audit.csv \
  --output-json outputs/manuscript/source_data/diagnostics/agop_field_mass_audit.json \
  --overwrite
```

This normal audit is read-only. Use `--build-missing-factors` only for a full,
expensive rebuild from the released training inputs; dense AGOP matrices are
then accumulated transiently and are not archived.

## 8--10. Final steering figures

```bash
python scripts/render_final_steering_figures.py \
  --bundle-dir artifacts/zc-v3/manuscript/steering/final_plot_data \
  --output-dir outputs/manuscript/figures \
  --overwrite
```

This single portable command renders Figure 10 (extreme-event dose response),
Figure 11 (matched AGOP steering cohorts), and Figure 12 (mean response by XAI
or composite direction). The compact bundle contains exactly the accepted
trajectories plotted in the paper and the strict-gate reports that authenticate
them; rerendering does not rerun the nonlinear solves. Missing or failed
members remain absent rather than being imputed.

## 11. Standardized ocean-current video

Install `ffmpeg`, then run:

```bash
python scripts/animate_zc_data.py \
  --data-dir data/processed/zc-v3 \
  --artifacts-dir artifacts/zc-v3 \
  --input-profile core4 \
  --architecture cnn \
  --lead-months 10 \
  --train-years 10000 \
  --seed 42 \
  --vector-kind ocean-current \
  --start-step 0 \
  --frames 360 \
  --stride 1 \
  --fps 6 \
  --dpi 150 \
  --quiver-stride 2 \
  --sst-min -8 \
  --sst-max 8 \
  --output outputs/video/zc_standardized_ocean_currents.mp4 \
  --overwrite
```

SST and both current components are displayed in featurewise standardized
units using the archived training-only normalizer. The animation uses the
even, 256-color `bwr` lookup table recorded in its sidecar.

## 12. Recomputing archived artifacts

The commands above are the fast, exact rendering path. Recomputing every
numerical artifact is substantially more expensive:

- data generation and restart certification:
  [`ZC_DATASET_REPRODUCIBILITY.md`](ZC_DATASET_REPRODUCIBILITY.md);
- exact AGOP and Table 1/Figures 6--7 generation:
  [`FRESH_TABLE1_FIGURES6_7.md`](FRESH_TABLE1_FIGURES6_7.md) and
  [`AGOP_COMPUTATION.md`](AGOP_COMPUTATION.md);
- native observation and covariance operators:
  [`ZC_NATIVE_OBSERVATION.md`](ZC_NATIVE_OBSERVATION.md) and
  [`ZC_NATIVE_COVARIANCE.md`](ZC_NATIVE_COVARIANCE.md); and
- nonlinear action-minimizing steering:
  [`ZC_DIRECT_AGOP_COVARIANCE_ACTION.md`](ZC_DIRECT_AGOP_COVARIANCE_ACTION.md),
  [`ZC_ADJOINT_DESIGN.md`](ZC_ADJOINT_DESIGN.md), and
  [`ZC_MATCHED_ADJOINT_FLOATING_POINT.md`](ZC_MATCHED_ADJOINT_FLOATING_POINT.md).

The archived artifacts are retained precisely so readers can verify every
reported figure and table without repeating the expensive training and
nonlinear optimization campaigns.

## 13. Final release check

After rendering, confirm that `outputs/manuscript/figures` contains only
Figures 3--12 and their sidecars, `outputs/manuscript/tables` contains only
Table 1 and its sidecar, and `outputs/video` contains only the one MP4
and its sidecar. Search the staged tree for credentials and private raw-data
paths. A few immutable raw provenance records retain documented historical
creation-machine paths covered by their SHA-256 identities; final commands and
presentation sidecars do not.

Matplotlib may embed a new PDF creation timestamp when a figure is rerendered.
Consequently, a scientifically identical PDF can receive a different bytewise
hash even when its page geometry and rendered pixels are unchanged; the newly
written sidecar records the new output hash.
