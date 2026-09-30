# Data release

Except where third-party provenance states otherwise, the released data are
licensed under CC BY 4.0; see [`../LICENSE-DATA`](../LICENSE-DATA).

The public data record contains `processed/zc-v3` and the small,
checksum-pinned NOAA climatology used by Figure 3. Raw ZC inputs are not
redistributed in this directory.

## `processed/zc-v3`

`zc-v3` is a 12,000-year Zebiak--Cane simulation sampled three times per month.
The chronological split is fixed:

- years 0--10,000: training;
- years 10,000--11,000: validation; and
- years 11,000--12,000: testing.

The release stores 13 interpretable spatial fields on the `20 x 27` active
grid, annual phase, the center-inclusive Nino-3 index, native time, metadata,
and verified native checkpoints ten months before the selected extreme El Nino
and La Nina events. The neural networks in the manuscript use only four spatial
fields--SST anomaly, thermocline depth, zonal ocean current, and meridional
ocean current--plus sine and cosine annual phase.

All input means and scales are fit on the training block only and are stored
with the corresponding model artifacts. The Nino-3 response is not
standardized and remains in degrees Celsius.

`metadata.json` defines field names, units, grid centers, time convention,
splits, source fingerprints, run settings, and restart checks. The Zenodo
package also contains `generation_provenance.json`, the sanitized generation
reports under `provenance/`, and the checksum manifest described in
[`../docs/ZC_DATASET_REPRODUCIBILITY.md`](../docs/ZC_DATASET_REPRODUCIBILITY.md).

## `external/noaa`

`sst.mon.ltm.1991-2020.nc` is the fixed NOAA ERSSTv5 1991--2020 monthly
climatology used by Figure 3. It is retained because it is small,
checksum-pinned, and allows the figure to be reproduced offline. The figure
generator verifies its SHA-256 before use.

## `raw`

See [`raw/README.md`](raw/README.md). The legacy private data file and the
downloaded upstream source are intentionally absent from the public release.
