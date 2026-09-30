# Released computational artifacts

These released artifacts are licensed under CC BY 4.0; see
[`../LICENSE-DATA`](../LICENSE-DATA).

Artifacts are immutable numerical inputs needed to verify or rerender the
manuscript. They are separate from `outputs/`, which contains only finished
presentation products and small source-data tables.

The public archive contains one artifact generation, `zc-v3`:

```text
zc-v3/
  models/core4/
  figure5/core4/
  manuscript/
```

## Models

`models/core4/` contains the seed-42 MLP, CNN, and ViT checkpoints used by the
main XAI analysis, plus the 1-, 5-, and 10-month CNNs used by Figures 4, 8, and
9. Each completed experiment includes the training-only normalizer and split
indices required to reconstruct its inputs.

`figure5/core4/` contains the five-seed predictive-skill model grid. It is kept
separately because Figure 5 varies architecture, lead, training length, and
seed, whereas the remainder of the manuscript uses the fixed primary models.

## Manuscript support bundles

- `table1_figure6/`: Table 1 scores, target explanations, exact nearest-1%
  neighbors, and robustness overlaps.
- `figure7/`: independently evaluated XAI bundles for CNN seeds 42--46.
- `figures8_9/`: selected-event, composite, and AGOP arrays for the three lead
  times.
- `field_mass_crosslead/`: exact full-rank seed-42 CNN AGOP eigensystems for
  leads 1--12, used to regenerate the field-mass audit without retaining dense
  AGOP matrices or unrelated XAI robustness bundles.
- `covariance/training-years10000-phases00-35/`: all 36 phase moments and the
  construction and validation manifests for the final pooled covariance. The
  6.1 GB dense matrix and approximately 48 GB native-state cache are
  reproducible scratch space and are not archived.
- `steering/final_plot_data/`: exact plotted trajectories, accepted-member
  identities, and strict-gate reports underlying Figures 10--12, all using the
  single annual all-phase covariance.

The compact steering records are retained because recomputing them requires
the native Fortran model, its adjoint, and many nonlinear constrained solves.
Failed pilots, optimizer scratch files, native phase caches, and compiler
products are not part of this archive. Rebuilding the steering solutions from
source therefore starts by recapturing the annual native states and
reconstructing the phase cache as described in
`docs/ZC_NATIVE_COVARIANCE.md`; the retained bundle is sufficient for auditing
and rerendering the published trajectories.

Do not edit artifact files in place. Provenance sidecars bind them by SHA-256.
If a scientific input changes, create a new versioned data record rather than
silently replacing an existing artifact.

The compact steering manifest and accepted-report archive use portable paths
and are validated by size, shape, dtype, and SHA-256 identities.
