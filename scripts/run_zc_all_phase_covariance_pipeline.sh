#!/usr/bin/env bash
# Build the single native-state covariance pooled across all 36 annual phases.

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
ZC_SOURCE_DIR="${ZC_SOURCE_DIR:-$REPOSITORY_ROOT/scratch/zc_adjoint/kernel_build_run23_final3/source}"
ZC_EXECUTABLE="${ZC_EXECUTABLE:-$ZC_SOURCE_DIR/zeqfc1}"
PRODUCTION_REPORT="${PRODUCTION_REPORT:-$REPOSITORY_ROOT/data/processed/zc-v3/provenance/production_report.json}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-$REPOSITORY_ROOT/data/processed/zc-v3/provenance/preflight_report.json}"
HISTORY_DIR="${HISTORY_DIR:-$REPOSITORY_ROOT/scratch/zc_generation/zc-v3-native-history}"
PHASE_CACHE_DIR="${PHASE_CACHE_DIR:-$REPOSITORY_ROOT/outputs/zc_native_phase_cache/training-years10000-phases00-35}"
COVARIANCE_DIR="${COVARIANCE_DIR:-$REPOSITORY_ROOT/outputs/zc_native_covariance/training-years10000-phases00-35}"
PIPELINE_DIR="${PIPELINE_DIR:-$REPOSITORY_ROOT/scratch/zc_all_phase_covariance_pipeline}"
STATUS_FILE="$PIPELINE_DIR/status.log"
LOCK_DIR="$PIPELINE_DIR/active.lock"

mkdir -p "$PIPELINE_DIR"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

status() {
  printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  status "ERROR another all-phase covariance pipeline owns $LOCK_DIR"
  exit 1
fi

finish() {
  code="$1"
  trap - EXIT
  if [[ $code -eq 0 ]]; then
    status "COMPLETE all-phase dense native covariance"
  else
    status "FAILED exit=$code; rerun this script to resume from the last committed stage"
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  return "$code"
}
trap 'finish "$?"' EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
  status "ERROR Python executable is unavailable: $PYTHON_BIN"
  exit 1
fi
if [[ ! -x "$ZC_EXECUTABLE" ]]; then
  status "ERROR certified ZC executable is unavailable: $ZC_EXECUTABLE"
  exit 1
fi

available_kib=$(df -Pk "$REPOSITORY_ROOT" | awk 'NR==2 {print $4}')
minimum_kib=$((80 * 1024 * 1024))
if (( available_kib < minimum_kib )); then
  status "ERROR at least 80 GiB free disk space is required"
  exit 1
fi

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"

status "START exact native-history regeneration/validation"
"$PYTHON_BIN" scripts/regenerate_zc_native_history.py \
  --source-dir "$ZC_SOURCE_DIR" \
  --executable "$ZC_EXECUTABLE" \
  --production-report "$PRODUCTION_REPORT" \
  --preflight-report "$PREFLIGHT_REPORT" \
  --output-dir "$HISTORY_DIR"

capture_resume=0
if [[ -f "$PHASE_CACHE_DIR/capture_complete.json" || -f "$PHASE_CACHE_DIR/capture_progress.json" ]]; then
  capture_resume=1
elif [[ -e "$PHASE_CACHE_DIR" ]]; then
  status "ERROR phase-cache directory exists without resumable provenance: $PHASE_CACHE_DIR"
  exit 1
fi

status "START capture of 360,000 training states at all 36 annual phases"
capture_command=("$PYTHON_BIN" scripts/capture_zc_annual_native_phases.py \
  --history "$HISTORY_DIR/outhst" \
  --production-report "$PRODUCTION_REPORT" \
  --data-dir data/processed/zc-v3 \
  --state-manifest adjoint/fortran_kernel/state_manifest.json \
  --source-dir "$ZC_SOURCE_DIR" \
  --executable "$ZC_EXECUTABLE" \
  --independent-overlap-replay \
  --output-dir "$PHASE_CACHE_DIR" \
  --phase-offsets 0 1 2 3 4 5 6 7 8 9 10 11 \
    12 13 14 15 16 17 18 19 20 21 22 23 \
    24 25 26 27 28 29 30 31 32 33 34 35 \
  --jobs 4)
if (( capture_resume )); then
  capture_command+=(--resume)
fi
"${capture_command[@]}"

factor_manifest="$COVARIANCE_DIR/manifest.json"
if [[ ! -f "$factor_manifest" ]]; then
  if [[ -f "$COVARIANCE_DIR/phase_moments.npz" ]]; then
    status "ERROR incomplete factor artifact exists without its manifest"
    exit 1
  fi
  status "START phase-specific centering and factor publication"
  "$PYTHON_BIN" scripts/build_zc_native_covariance_factors.py \
    --phase-cache-dir "$PHASE_CACHE_DIR" \
    --output-dir "$COVARIANCE_DIR"
else
  status "SKIP factor publication; manifest already exists"
fi

dense_resume=0
if [[ -f "$COVARIANCE_DIR/dense_manifest.json" || -f "$COVARIANCE_DIR/.dense_covariance_work/progress.json" ]]; then
  dense_resume=1
elif [[ -f "$COVARIANCE_DIR/covariance.npy" ]]; then
  status "ERROR dense covariance exists without resumable progress"
  exit 1
fi

status "START exact dense all-phase covariance accumulation"
dense_command=("$PYTHON_BIN" scripts/build_zc_all_phase_dense_covariance.py \
  --factor-manifest "$factor_manifest" \
  --output-dir "$COVARIANCE_DIR")
if (( dense_resume )); then
  dense_command+=(--resume)
fi
"${dense_command[@]}"
