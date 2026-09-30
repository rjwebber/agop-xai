# Raw inputs are intentionally omitted

This directory is a local staging location, not a public data product.

The legacy file `grads_1.data` is private and is not used by the final `zc-v3`
analysis. It must not be committed to GitHub or deposited on Zenodo.

The upstream Zebiak--Cane source is publicly downloadable, but the downloaded
archive and unpacked `CZ_model_share` tree are also omitted. Reproduction uses
the checksum-pinned archive from Eli Tziperman's public download page. Follow
[`../../docs/ZC_DATASET_REPRODUCIBILITY.md`](../../docs/ZC_DATASET_REPRODUCIBILITY.md)
to download, verify, build, and run it locally.

Typical temporary local contents are:

```text
data/raw/CZ_model_share.zip
data/raw/CZ_model_share/
```

Both paths are ignored by Git and should be removed from a staged release.
