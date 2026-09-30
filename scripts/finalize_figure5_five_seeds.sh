#!/usr/bin/env bash
set -euo pipefail

# Assemble the canonical five-seed Figure 5 grid from completed model caches,
# audit every artifact, and render the canonical IBM-palette PDF.

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_dir}"

python scripts/run_figure5_experiments.py \
    --data-dir data/processed/zc-v3 \
    --input-profile core4 \
    --artifacts-dir artifacts/zc-v3 \
    --output outputs/manuscript/source_data/figure5/figure5_zcv3_results.csv \
    --architectures mlp cnn vit \
    --training-years 50 100 200 500 1000 2000 5000 10000 \
    --lead-months 1 2 3 4 5 6 7 8 9 10 11 12 \
    --right-panel-training-years 50 10000 \
    --left-repetitions 5 \
    --right-repetitions 5 \
    --base-seed 42 \
    --device cuda \
    --batch-size 256 \
    --statistics-batch-size 1024 \
    --maximum-epochs 100 \
    --patience 10 \
    --minimum-improvement 1e-4 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --overwrite

python scripts/audit_figure5_results.py \
    --results outputs/manuscript/source_data/figure5/figure5_zcv3_results.csv \
    --artifacts-dir artifacts/zc-v3 \
    --expected-left-repetitions 5 \
    --expected-right-repetitions 5

python scripts/make_figure5.py \
    --results outputs/manuscript/source_data/figure5/figure5_zcv3_results.csv \
    --output outputs/manuscript/figures/figure5_predictive_skill.pdf \
    --overwrite

echo "Five-seed Figure 5 aggregation, audit, and rendering completed."
