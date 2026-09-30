# Public release contents

This file defines the intended public boundary for the manuscript release.
Anything not listed here is a local working product and should not be included
in the GitHub repository or Zenodo deposit.

## GitHub deposit

Include:

- `.gitignore`, `.gitattributes`, `.github/workflows/ci.yml`, `README.md`, `CHANGELOG.md`,
  `RELEASE_CONTENTS.md`, `THIRD_PARTY_NOTICES.md`, `pyproject.toml`, `LICENSE`,
  `LICENSE-DATA`, and `CITATION.cff`;
- `src/zc_xai/`;
- `examples/`;
- the manuscript-production entry points in `scripts/`;
- their regression tests in `tests/`;
- the current method and reproduction notes in `docs/`; and
- the source-only adjoint/control implementation in `adjoint/` that is needed
  to rebuild the steering results, including the project-authored ZC-derived
  patches in `adjoint/tapenade_toolchain/patches/` and coupled drivers in
  `adjoint/tapenade_toolchain/coupled/`.

Do not include generated executables, object/module files, build directories,
native-state caches, optimizer continuation directories, local logs, or
downloaded upstream source. Redistribution permission has been confirmed for
the project-authored ZC-derived modifications included above. This does not
change the separate-source workflow: users download the checksum-pinned
upstream ZC distribution and apply the released patches locally. Generated
Tapenade source and compiler products remain reproducible build outputs rather
than GitHub source files.

## Zenodo deposit

The data, trained artifacts, generated manuscript products, and video are
released under CC BY 4.0 as stated in `LICENSE-DATA`.

Preserve this layout when packaging the data record:

```text
data/
  README.md
  processed/zc-v3/            # processed 12,000-year simulation and restarts
  external/noaa/              # pinned ERSSTv5 climatology for offline Figure 3
artifacts/
  README.md
  zc-v3/
    SHA256SUMS
    models/core4/             # seed-42 manuscript models and normalizers
    figure5/core4/            # five-seed predictive-skill model grid
    manuscript/               # compact XAI and steering support bundles
outputs/
  README.md
  SHA256SUMS
  manuscript/
    figures/                  # Figures 3--12 (PDF plus provenance sidecars)
    tables/                   # Table 1 (TeX plus provenance sidecar)
    source_data/              # compact values supporting figures and text
    benchmarks/               # reported XAI timing measurements
  video/
    zc_standardized_ocean_currents.mp4
    zc_standardized_ocean_currents.json
```

The final manuscript artifact subtree has the following roles:

| Path | Why it is retained |
| --- | --- |
| `manuscript/table1_figure6/` | Training-only explanations, robustness neighborhoods, and scores underlying Table 1 and Figure 6 |
| `manuscript/figure7/` | Complete seeds 42--46 XAI bundles underlying the five-seed means in Figure 7 |
| `manuscript/figures8_9/` | Multi-lead selected-event, composite, and AGOP arrays for Figures 8 and 9 |
| `manuscript/field_mass_crosslead/` | Exact seed-42 CNN AGOP eigensystems at leads 1--12 used by the field-mass audit; dense matrices are rebuilt transiently |
| `manuscript/covariance/training-years10000-phases00-35/` | All 36 phase moments and manifests for the final pooled covariance; the dense matrix and native cache are regenerated, not archived |
| `manuscript/steering/final_plot_data/` | Compact validated trajectories, accepted-member identities, and strict-gate reports underlying Figures 10--12 |

These artifacts are not duplicate presentation outputs. They are the compact,
hash-bound numerical inputs needed to audit the reported values without
rerunning thousands of model fits or the expensive nonlinear steering solves.
The repository retains the scientific drivers needed to rerun individual
steering experiments, while the Zenodo bundle is the archival source for exact
Figure 10--12 rerendering. Local batch-rescue orchestration and the one-time
curator conversion from raw solver workspaces are intentionally excluded.

## Final output inventory

`outputs/manuscript/figures/` contains:

1. `figure3_noaa_ersstv5.pdf`
2. `figure4_cnn_forecasts.pdf`
3. `figure5_predictive_skill.pdf`
4. `figure6_fresh_core4.pdf`
5. `figure7_zcv3_core4.pdf`
6. `figure8_el_nino_multilead.pdf`
7. `figure9_la_nina_multilead.pdf`
8. `figure10_extreme_agop_dose_response.pdf`
9. `figure11_agop_warm_cold_paired_trajectories.pdf`
10. `figure12_xai_method_mean_trajectories.pdf`

`outputs/manuscript/tables/` contains `table1_xai_scores.tex` and its JSON
provenance sidecar.

The retained source data are limited to the Figure 5 values and the AGOP
field-mass audit cited in the text. The CPU XAI benchmark supports the timing
paragraph. The video directory contains exactly one public `zc-v3` animation.

Figures 1 and 2 are manuscript-native diagrams and are therefore not generated
or archived here.

## Explicit exclusions

Do not release:

- `data/raw/grads_1.data` or any derivative of that private legacy file;
- a downloaded or unpacked `CZ_model_share` tree;
- obsolete `zc-v2` data or models;
- abandoned channel-profile experiments;
- exploratory composites, neighborhood tests, neutral-state pilots, failed
  optimization attempts, or rescue workspaces not used by the final results;
- native phase caches, full dense gradient-row caches, compiler products, and
  intermediate AGOP matrices that are reproducible from retained inputs;
- local `scratch/`, `tmp/`, and non-public `outputs/zc_*` workspaces;
- cluster launch state, shell histories, or logs not needed to support a
  reported timing; or
- `.DS_Store`, `__pycache__`, `.ruff_cache`, and similar local metadata.

## Pre-deposit verification

From a clean checkout with the Zenodo tree staged at the repository root:

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m ruff check .
python scripts/verify_release.py
python scripts/make_fresh_table1.py --help >/dev/null
python scripts/render_final_steering_figures.py --help >/dev/null
```

Then rerender all final products using
[`docs/FRESH_FIGURE_RUNBOOK.md`](docs/FRESH_FIGURE_RUNBOOK.md), verify the
archive checksum manifest, and search the staged JSON/text files for secrets or
private raw-data paths before upload.

Verify the four archive checksum manifests from the repository root:

```bash
(cd data/processed/zc-v3 && shasum -a 256 -c SHA256SUMS)
(cd data/external/noaa && shasum -a 256 -c SHA256SUMS)
(cd artifacts/zc-v3 && shasum -a 256 -c SHA256SUMS)
(cd outputs && shasum -a 256 -c SHA256SUMS)
```

All release commands, presentation sidecars, and compact support bundles use
repository-relative paths. Curated derivative provenance replaces local build
locations with explicit placeholders. No released record contains credentials
or private raw-data paths.
