#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
canonical_build="${1:-${repository_root}/adjoint/tapenade_toolchain/build/coupled_run23_final3}"
output_dir="${2:-${repository_root}/adjoint/controlled_window/build}"
compiled="${canonical_build}/coupled_compiled"
prepared="${canonical_build}/coupled_prepared"
driver="${repository_root}/adjoint/controlled_window/zc_one_step_driver.F"
tangent_driver="${repository_root}/adjoint/controlled_window/zc_one_step_tangent_driver.F"

for required in \
  "${compiled}/build_input_manifest.sha256" \
  "${compiled}/primal/zc_kernel_api.o" \
  "${compiled}/support/constc.o" \
  "${prepared}/zc_kernel_state.inc" \
  "${driver}" \
  "${tangent_driver}" \
  "${compiled}/libzc_tangent.a"; do
  test -s "${required}" || {
    printf 'ERROR: required canonical input is missing: %s\n' "${required}" >&2
    exit 2
  }
done

fc="${FC:-gfortran-13}"
if ! command -v "${fc}" >/dev/null 2>&1; then
  fc="${FC:-gfortran}"
fi
fc_path="$(command -v "${fc}")"
flags=(-std=legacy -O0 -g -fcheck=all -fbacktrace
  -fallow-argument-mismatch -ffixed-line-length-none)
platform_flags=()
if [[ "$(uname -s)" == Darwin ]]; then
  sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
  sdk_major="$(xcrun --sdk macosx --show-sdk-version | cut -d. -f1)"
  platform_flags=(-isysroot "${sdk_path}" -mmacosx-version-min="${sdk_major}.0")
fi

mkdir -p "${output_dir}"
object="${output_dir}/zc_one_step_driver.o"
executable="${output_dir}/zc_one_step"
tangent_object="${output_dir}/zc_one_step_tangent_driver.o"
tangent_executable="${output_dir}/zc_one_step_tangent"
"${fc_path}" "${flags[@]}" "${platform_flags[@]}" \
  -I "${prepared}" -c "${driver}" -o "${object}"
"${fc_path}" "${flags[@]}" "${platform_flags[@]}" -o "${executable}" \
  "${object}" \
  "${compiled}/primal/zc_kernel_api.o" \
  "${compiled}/primal/ssta.o" \
  "${compiled}/primal/ztmfc1.o" \
  "${compiled}/primal/cforce.o" \
  "${compiled}/primal/mloop.o" \
  "${compiled}/primal/akcalc.o" \
  "${compiled}/primal/bndary.o" \
  "${compiled}/primal/uhcalc.o" \
  "${compiled}/primal/uhinit.o" \
  "${compiled}/primal/tridag.o" \
  "${compiled}/primal/fft2c.o" \
  "${compiled}/support/"*.o
"${fc_path}" "${flags[@]}" "${platform_flags[@]}" \
  -I "${prepared}" -I "${canonical_build}/coupled_generated/tangent" \
  -c "${tangent_driver}" -o "${tangent_object}"
"${fc_path}" "${flags[@]}" "${platform_flags[@]}" \
  -o "${tangent_executable}" "${tangent_object}" \
  "${compiled}/libzc_tangent.a" \
  "${compiled}/support/"*.o

{
  printf 'driver_sha256='
  shasum -a 256 "${driver}" | awk '{print $1}'
  printf 'tangent_driver_sha256='
  shasum -a 256 "${tangent_driver}" | awk '{print $1}'
  printf 'canonical_build_manifest_sha256='
  shasum -a 256 "${compiled}/build_input_manifest.sha256" | awk '{print $1}'
  printf 'canonical_primal_executable_sha256='
  shasum -a 256 "${compiled}/zc_adjoint_make_path" | awk '{print $1}'
  printf 'compiler=%s\n' "$("${fc_path}" --version | head -n 1)"
  printf 'flags=%s %s\n' "${flags[*]}" "${platform_flags[*]}"
  printf 'executable_sha256='
  shasum -a 256 "${executable}" | awk '{print $1}'
  printf 'tangent_executable_sha256='
  shasum -a 256 "${tangent_executable}" | awk '{print $1}'
} > "${output_dir}/build_provenance.txt"

printf '%s\n%s\n' "${executable}" "${tangent_executable}"
