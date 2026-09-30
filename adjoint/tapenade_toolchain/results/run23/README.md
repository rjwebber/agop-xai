# Run23 Zebiak--Cane adjoint evidence

This is the portable, source-free evidence bundle for the run23 discrete
adjoint.  Its precise status is **`provisional_major_tape_conditioned`**.
It supports a branchwise derivative of the locked Standard Zebiak--Cane
Fortran calculation for the three recorded checkpoints; it is not global
derivative certification.

The authentic explicit-state primal passes 15 bitwise replay cases, three
fresh-process segmented replays, and six omitted-workspace poison tests.  The
same-generation real32 tangent and reverse pass scalar-head and dense-terminal
transpose comparisons.  One-step centered differences are especially strong
on the conservative independent-control mask.  Long-window scalar checks are
useful but limited by float32 rounding and switching: the best major-tape-
stable aligned defects are about 3.36e-4 for the extreme warm case and 1.11e-2
for the neutral case; every tested aligned cold-case perturbation changes NSST
at step 15.  No clean multi-level quadratic Taylor regime was established.

## Contents

- `INDEX.json` is the human- and machine-readable claim boundary and inventory.
- `evidence/` contains sanitized detailed reports and compact summaries.  The
  index records both each original report hash and its published sanitized
  hash, along with the sanitizer rule.
- `provenance/` contains the explicit state manifest, exact hash inventories,
  source-free build summaries, and Tapenade warning counts.
- `cases/*/gradient.npz` contains the packed gradient plus each named manifest
  segment.  `gradient.json` records its coordinate semantics and embeds the
  sanitized v2 producer provenance.  `producer_metadata.json` is the exact v2
  producer document used by the strict packager.
- `MANIFEST.sha256` covers every published file other than itself.  Finder
  metadata is never included.

## What is deliberately absent

The bundle does not publish upstream ZC source, Tapenade-generated source,
executables, checkpoints, or raw trajectory paths. Those artifacts are
identified by hashes in the evidence chain. Project-authored ZC-derived
patches and coupled drivers are published separately in the repository rather
than copied into this evidence bundle.

## Reproducing with authorized ZC source

1. Build and replay the explicit-state primal using the commands in
   `../../README.md` and `../../../fortran_kernel/README.md`.  Require the
   15/3/6 replay gates to pass for the exact three checkpoint hashes in
   `INDEX.json`.
2. Prepare the coupled sources locally, run pinned Tapenade 3.16 revision
   `0449e37b7da896cb2c38b63c8a34f10c87a84534` on Linux, instrument the
   generated reverse, and compile on the Mac.  Compare the resulting three
   compile-input inventories and executable hashes with `provenance/` and
   `INDEX.json`.
3. Run each checkpoint with `run_coupled_case.py`.  The released case record
   names the expected path, gradient, checkpoint, and reverse-executable
   hashes.  It uses 31 coupled ten-day transitions and the final `TO` mean over
   Fortran `I=13:18,J=21:31`.
4. Re-run the numerical validators described in
   `../../../full_tangent_audit/README.md`.  Compare original report hashes in
   `INDEX.json`; do not compare timing fields across machines.
5. From the repository root, re-run
   `python adjoint/tapenade_toolchain/scripts/curate_run23_release.py --overwrite`.
   The curator verifies the producer/evidence chain, rebuilds all named gradient
   packages, strips local paths from published reports, and writes the top-level
   manifest.

The gradient has 59,148 raw packed Fortran coordinates.  It is not a gradient
in the standardized 2,162-feature neural-network space, and raw cross-variable
norms are not feature importance.  A comparison with AGOP additionally needs
the differentiated observation/time-alignment map, the frozen normalization
transpose, and a balanced native-state lift or declared control map.

This warning matters numerically: the largest raw-gradient segment is carried
atmospheric memory for each released case (see `INDEX.json`).  Only 28,591
native entries belong to the first conservative independent-control mask.
Plotting all 59,148 coefficients as a physical precursor pattern would be
misleading.
