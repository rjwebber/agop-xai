# AGOP XAI for Zebiak--Cane ENSO forecasts

This repository contains the code used to train neural-network forecasts of
the Zebiak--Cane (ZC) ENSO model, explain those forecasts with the average
gradient outer product (AGOP), and test the explanation directions in the
nonlinear ZC dynamics.

The final analysis uses the public `zc-v3` simulation and the `core4` input
profile: SST anomaly, thermocline depth, zonal ocean current, meridional ocean
current, and two scalar annual-phase coordinates. Each neural-network input
therefore has dimension

```text
4 fields x 20 latitudes x 27 longitudes + 2 phase coordinates = 2,162.
```

The retained simulation contains 12,000 years at three model steps per month.
The first 10,000 years are used for training, the next 1,000 for validation,
and the last 1,000 for testing. Input normalization is fit on the training
block only. The Nino-3 response remains in degrees Celsius.

## Public release

The release is divided deliberately between two services:

- **[GitHub](https://github.com/rjwebber/agop-xai)** contains the Python and
  Fortran code, tests, and documentation.
- **Zenodo** contains the 11 GB processed `zc-v3` data set, the pinned NOAA
  climatology used by Figure 3, trained model checkpoints, compact figure and
  steering artifacts, Figures 3--12, Table 1, source data, the reported XAI
  timing benchmark, and one MP4 animation.

After downloading the companion Zenodo archive, unpack it at the repository
root so that `data/processed/zc-v3`, `artifacts/zc-v3`, and `outputs` retain
their documented relative paths. See [`RELEASE_CONTENTS.md`](RELEASE_CONTENTS.md)
for the exact boundary and the reason each retained directory is needed.
The changes included in each source release are summarized in
[`CHANGELOG.md`](CHANGELOG.md).

Figures 1 and 2 are manuscript-native schematics. The archived computational
release covers Figures 3--12, Table 1, and the standardized-ocean-current
video. Portable rendering commands are in
[`docs/FRESH_FIGURE_RUNBOOK.md`](docs/FRESH_FIGURE_RUNBOOK.md).

## Install and test

Python 3.11 or newer is required:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m ruff check .
```

Generating the ZC data and reproducing the steering experiments additionally
requires GNU Fortran and `make`. MP4 rendering requires `ffmpeg`.

## Reproduce the manuscript products

With the Zenodo data and artifacts unpacked, follow the commands in
[`docs/FRESH_FIGURE_RUNBOOK.md`](docs/FRESH_FIGURE_RUNBOOK.md). Every generated
PDF or TeX table has a JSON provenance sidecar. The presentation sidecars and
the compact Figures 10--12 steering bundle record input and output hashes
using portable paths.

To regenerate the processed data from the public upstream ZC source instead,
follow [`docs/ZC_DATASET_REPRODUCIBILITY.md`](docs/ZC_DATASET_REPRODUCIBILITY.md).
The downloaded source is checksum-pinned and is not redistributed here. The
project-authored ZC-derived patches and coupled adjoint drivers needed by the
release are published under `adjoint/tapenade_toolchain/patches/` and
`adjoint/tapenade_toolchain/coupled/`.

## Repository layout

| Path | Purpose | Release destination |
| --- | --- | --- |
| `src/zc_xai/` | Validated data loading, models, training, XAI, and shared utilities | GitHub |
| `examples/` | Minimal standalone exact AGOP reference implementation | GitHub |
| `scripts/` | Data-generation, training, analysis, rendering, and steering entry points | GitHub |
| `tests/` | Regression and scientific-contract tests | GitHub |
| `docs/` | Reproduction and method documentation | GitHub |
| `adjoint/` | Native ZC tangent/reverse and control code needed by the steering experiments | GitHub |
| `data/processed/zc-v3/` | Public processed simulation and two verified restart checkpoints | Zenodo |
| `data/external/noaa/` | Checksum-pinned ERSSTv5 climatology used by Figure 3 | Zenodo |
| `artifacts/zc-v3/` | Trained models and compact inputs needed to verify or rerender the manuscript | Zenodo |
| `outputs/manuscript/` | Final Figures 3--12, Table 1, and supporting source data | Zenodo |
| `outputs/video/` | The single released animation and its sidecar | Zenodo |
| `scratch/` | Regenerable caches and interrupted-run state | Local only; never released |

`data/raw/` is intentionally empty except for its README. The legacy private
`grads_1.data`, downloaded upstream source, temporary native streams, build
products, caches, and unsuccessful exploratory analyses are not release
materials. Only checksum-validated results promoted from `scratch/` into the
documented artifact or output layout are eligible for release.

## Method documentation

- [`docs/AGOP_COMPUTATION.md`](docs/AGOP_COMPUTATION.md): exact dense AGOP and
  higher-dimensional alternatives.
- [`docs/FRESH_TABLE1_FIGURES6_7.md`](docs/FRESH_TABLE1_FIGURES6_7.md): final
  training-only XAI comparison contract.
- [`docs/ZC_NATIVE_OBSERVATION.md`](docs/ZC_NATIVE_OBSERVATION.md): map between
  native ZC checkpoints and the `core4+phase` neural-network coordinates.
- [`docs/ZC_NATIVE_COVARIANCE.md`](docs/ZC_NATIVE_COVARIANCE.md): pooled native
  covariance used to price steering perturbations.
- [`docs/ZC_DIRECT_AGOP_COVARIANCE_ACTION.md`](docs/ZC_DIRECT_AGOP_COVARIANCE_ACTION.md):
  final nonlinear covariance-action steering experiment.
- [`docs/ZC_ADJOINT_DESIGN.md`](docs/ZC_ADJOINT_DESIGN.md) and
  [`docs/ZC_MATCHED_ADJOINT_FLOATING_POINT.md`](docs/ZC_MATCHED_ADJOINT_FLOATING_POINT.md):
  derivative implementation and numerical scope.

## Release safeguards

Do not publish a ZIP of a working directory. Build public deposits from the
allowlists in `RELEASE_CONTENTS.md`, verify hashes, and inspect the staged file
list. Downloaded ZC source, private inputs, temporary optimization state, and
compiler products must not enter either deposit. Curated derivative provenance
uses explicit placeholders in place of machine-local paths.

## Licensing

The software and documentation are released under the
[BSD 3-Clause License](LICENSE). The Zenodo data, trained artifacts, generated
figures and tables, and video are released under
[CC BY 4.0](LICENSE-DATA), except where a third-party provenance record states
otherwise. Downloaded upstream ZC source is not included in either release.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the ZC source,
NOAA ERSSTv5, and Tapenade boundaries.
