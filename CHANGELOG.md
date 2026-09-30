# Changelog

## 1.0.0 — 2026-09-29

First public manuscript release.

- Provides the PyTorch MLP, CNN, and ViT forecasting implementations used in
  the paper.
- Implements GRAD, Integrated Gradients, GradientSHAP, and exact dense AGOP
  XAI, including the training-only robustness evaluation.
- Reproduces computational Figures 3--12, Table 1, and the standardized
  ocean-current animation when combined with the companion Zenodo archive.
- Includes the project-authored Zebiak--Cane patches, tangent/adjoint code,
  and covariance-weighted steering implementation.
- Separates the lightweight BSD-3-Clause source release on GitHub from the
  CC BY 4.0 processed data, trained artifacts, and generated outputs on
  Zenodo.

The private legacy input `grads_1.data`, downloaded upstream Zebiak--Cane
source, generated compiler products, and unsuccessful exploratory analyses
are intentionally excluded.
