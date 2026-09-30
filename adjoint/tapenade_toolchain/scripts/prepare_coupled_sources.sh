#!/usr/bin/env bash
set -euo pipefail

toolchain_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kernel_source="${1:?usage: prepare_coupled_sources.sh KERNEL_BUILD_SOURCE [BUILD_DIR]}"
build_dir="${2:-${toolchain_dir}/build/coupled_release}"
# shellcheck disable=SC1091
source "${toolchain_dir}/scripts/build_guard.sh"
kernel_source="$(zc_resolve_path "${kernel_source}")"
build_dir="$(zc_claim_build_dir "${toolchain_dir}" "${build_dir}" \
  "${kernel_source}")"
prepared_dir="${build_dir}/coupled_prepared"

sources=(
  zc_kernel_api.F ssta.F ztmfc1.F cforce.F mloop.F akcalc.F bndary.F
  uhcalc.F uhinit.F tridag.F
)
includes=(zc_kernel_state.inc zeq.common modified_means.common)
passive_sources=(openfl.F setup.F constc.F nrdhist.F initdat.F setup2.F close_files.F)
passive_libraries=(GGNQF.F GGUBFS.F MDNRIS.F MERFI.F UERTST.F UGETIO.F USPKD.F)
kernel_identity_files=(
  "${sources[@]}" "${includes[@]}" "${passive_sources[@]}"
)
for name in "${passive_libraries[@]}" FFT2C.F; do
  kernel_identity_files+=("ZC_lib_routines/${name}")
done
normalization_patches=(
  kernel_api_ad_ready.patch kernel_api_scalar_helpers.patch
  ssta_ad_io_free.patch mloop_io_free.patch mloop_diagnostics_io_free.patch
  cforce_no_hidden_save.patch cforce_ad_ready.patch
  ztmfc1_feedback_loop.patch ztmfc1_ad_constants.patch
  remove_blank_local_saves.patch stress_selective_save.patch
  terminal_push_initialization.patch
)
fft_patch=fft2c_no_equivalence.patch

# Complete read-only preflight before replacing a previously good stage.
command -v patch >/dev/null
command -v cpp >/dev/null
for name in "${sources[@]}" "${includes[@]}"; do
  test -s "${kernel_source}/${name}" || {
    printf 'ERROR: required kernel input is missing: %s\n' \
      "${kernel_source}/${name}" >&2
    exit 2
  }
done
test -s "${kernel_source}/ZC_lib_routines/FFT2C.F"
for name in "${passive_sources[@]}"; do
  test -s "${kernel_source}/${name}" || {
    printf 'ERROR: required passive kernel input is missing: %s\n' \
      "${kernel_source}/${name}" >&2
    exit 2
  }
done
for name in "${passive_libraries[@]}"; do
  test -s "${kernel_source}/ZC_lib_routines/${name}" || {
    printf 'ERROR: required passive kernel library is missing: %s\n' \
      "${kernel_source}/ZC_lib_routines/${name}" >&2
    exit 2
  }
done
for patch_name in "${normalization_patches[@]}" "${fft_patch}"; do
  test -s "${toolchain_dir}/patches/${patch_name}" || {
    printf 'ERROR: required normalization patch is missing: %s\n' \
      "${patch_name}" >&2
    exit 2
  }
done
test -s "${toolchain_dir}/coupled/zc_kernel_nino3.F"
for provenance_input in VERSION.env scripts/build_guard.sh \
  scripts/prepare_coupled_sources.sh; do
  test -s "${toolchain_dir}/${provenance_input}" || {
    printf 'ERROR: preparation provenance input is missing: %s\n' \
      "${toolchain_dir}/${provenance_input}" >&2
    exit 2
  }
done
zc_assert_stage_replacement_allowed "${prepared_dir}" 0
prepared_target_state="$(zc_capture_path_state "${prepared_dir}")"
script_state="$(zc_capture_path_state "${BASH_SOURCE[0]}")"
guard_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/build_guard.sh")"

workspace="$(zc_create_stage_workspace "${build_dir}" prepare)"
workspace_identity="$(zc_path_identity "${workspace}")"
prepared_target="${prepared_dir}"
prepared_dir="${workspace}/payload"
input_snapshot="${workspace}/inputs"
snapshot_kernel="${input_snapshot}/kernel_source"
snapshot_toolchain="${input_snapshot}/toolchain"
mkdir -p "${prepared_dir}/ZC_lib_routines" "${snapshot_kernel}" \
  "${snapshot_toolchain}/scripts" "${snapshot_toolchain}/patches" \
  "${snapshot_toolchain}/coupled"

cleanup_prepare_workspace() {
  local status="$?"
  trap - EXIT HUP INT TERM
  zc_remove_stage_workspace "${build_dir}" "${workspace}" \
    "${workspace_identity}" || status=2
  exit "${status}"
}
trap cleanup_prepare_workspace EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for name in "${kernel_identity_files[@]}"; do
  mkdir -p "$(dirname "${snapshot_kernel}/${name}")"
  cp "${kernel_source}/${name}" "${snapshot_kernel}/${name}"
done
for patch_name in "${normalization_patches[@]}" "${fft_patch}"; do
  cp "${toolchain_dir}/patches/${patch_name}" \
    "${snapshot_toolchain}/patches/${patch_name}"
done
cp "${toolchain_dir}/coupled/zc_kernel_nino3.F" \
  "${snapshot_toolchain}/coupled/zc_kernel_nino3.F"
for provenance_input in VERSION.env scripts/build_guard.sh \
  scripts/prepare_coupled_sources.sh; do
  cp "${toolchain_dir}/${provenance_input}" \
    "${snapshot_toolchain}/${provenance_input}"
done
preparation_input_manifest="${workspace}/preparation_input_manifest.sha256"
(
  cd "${input_snapshot}"
  find . -type f -print | sed 's#^\./##' | LC_ALL=C sort |
    while IFS= read -r relative_name; do
      printf '%s  %s\n' "$(zc_sha256_file "${relative_name}")" \
        "${relative_name}"
    done
) > "${preparation_input_manifest}"
zc_verify_sha256_manifest_exact "${input_snapshot}" \
  "${preparation_input_manifest}"
find "${input_snapshot}" -type f -exec chmod a-w {} +
find "${input_snapshot}" -type d -exec chmod a-w {} +
input_snapshot_state="$(zc_capture_path_state "${input_snapshot}")"

cpp_path="$(command -v cpp)"
if [[ "$(uname -s)" == Darwin ]]; then
  cpp_path=/usr/bin/cpp
fi
cpp_path="$(zc_resolve_path "${cpp_path}")"
patch_path="$(zc_resolve_path "$(command -v patch)")"
cpp_state="$(zc_capture_path_state "${cpp_path}")"
patch_state="$(zc_capture_path_state "${patch_path}")"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

preprocess_fortran() {
  local input="$1"
  local output="$2"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    # Apple's cpp rejects -traditional-cpp, but its ordinary preprocessing
    # preserves fixed-form label/continuation columns.  clang -E -x c does not.
    "${cpp_path}" -P -I"${snapshot_kernel}" "${input}" > "${output}"
  else
    "${cpp_path}" -traditional-cpp -P -I "${snapshot_kernel}" \
      "${input}" "${output}"
  fi
}

for name in "${sources[@]}"; do
  test -f "${snapshot_kernel}/${name}"
  preprocess_fortran \
    "${snapshot_kernel}/${name}" "${prepared_dir}/${name%.F}.f"
done
for name in "${includes[@]}"; do
  cp "${snapshot_kernel}/${name}" "${prepared_dir}/${name}"
done
preprocess_fortran \
  "${snapshot_kernel}/ZC_lib_routines/FFT2C.F" \
  "${prepared_dir}/ZC_lib_routines/fft2c.f"

# This manifest prevents preparing active sources from one certified kernel
# build and later linking passive support from another.
{
  for name in "${kernel_identity_files[@]}"; do
    printf '%s  %s\n' "$(sha256_file "${snapshot_kernel}/${name}")" "${name}"
  done
} | LC_ALL=C sort > \
  "${prepared_dir}/kernel_source_input_manifest.sha256"

manifest_files=(
  zc_kernel_api.f ssta.f ztmfc1.f cforce.f mloop.f akcalc.f bndary.f
  uhcalc.f uhinit.f tridag.f zc_kernel_state.inc zeq.common
  modified_means.common ZC_lib_routines/fft2c.f
)
{
  for name in "${manifest_files[@]}"; do
    printf '%s  %s\n' "$(sha256_file "${prepared_dir}/${name}")" "${name}"
  done
} > "${prepared_dir}/upstream_preprocessed_manifest.sha256"

cforce_pre_sha256="$(sha256_file "${prepared_dir}/cforce.f")"
ztmfc1_pre_sha256="$(sha256_file "${prepared_dir}/ztmfc1.f")"

# Make only semantics-preserving source normalizations needed by Tapenade:
# explicit adjustable bounds, structured control flow, a collision-free local
# name, and removal of a plotting-only external call.  MLOOP's diagnostic
# WRITE is also absent from the differentiated I/O-free kernel.
for patch_name in "${normalization_patches[@]}"; do
  "${patch_path}" --quiet --fuzz=0 -V none \
    --directory "${prepared_dir}" --strip 0 \
    < "${snapshot_toolchain}/patches/${patch_name}"
done

# The atmospheric fixed-point iteration must be structured before AD.  A GOTO
# into the Q-reset loop makes Tapenade attach the target label to a DO terminal,
# yielding invalid generated control flow.
grep -q '^      DO WHILE(.TRUE.)$' "${prepared_dir}/ztmfc1.f"
if grep -Eq 'GO TO (310|380)|^310|^380' "${prepared_dir}/ztmfc1.f"; then
  printf 'ERROR: unstructured ZATMC feedback-loop target remains\n' >&2
  exit 1
fi

# Record the exact transformation boundary without retaining a modified copy
# of the upstream CFORCE source in this release-oriented tree.
cat > "${prepared_dir}/normalization_hashes.txt" <<EOF
cforce_preprocessed_upstream_sha256 ${cforce_pre_sha256}
cforce_ad_prepared_sha256 $(sha256_file "${prepared_dir}/cforce.f")
ztmfc1_preprocessed_upstream_sha256 ${ztmfc1_pre_sha256}
ztmfc1_ad_prepared_sha256 $(sha256_file "${prepared_dir}/ztmfc1.f")
EOF
"${patch_path}" --quiet --fuzz=0 -V none \
  --directory "${prepared_dir}" --strip 0 \
  < "${snapshot_toolchain}/patches/${fft_patch}"
find "${prepared_dir}" -type f \( -name '*.orig' -o -name '*~' \) -delete
if find "${prepared_dir}" -type f -name '*.rej' -print -quit | grep -q .; then
  printf 'ERROR: rejected prepared-source patch remains in %s\n' \
    "${prepared_dir}" >&2
  exit 2
fi
cp "${snapshot_toolchain}/coupled/zc_kernel_nino3.F" \
  "${prepared_dir}/zc_kernel_nino3.f"

manifest_files+=(zc_kernel_nino3.f normalization_hashes.txt)

if [[ "$(uname -s)" == "Darwin" ]]; then
  cpp_flags="-P -I<KERNEL_SOURCE>"
else
  cpp_flags="-traditional-cpp -P -I <KERNEL_SOURCE>"
fi
{
  printf 'stage=prepare_coupled_sources\n'
  printf 'script_sha256=%s\n' \
    "$(sha256_file \
      "${snapshot_toolchain}/scripts/prepare_coupled_sources.sh")"
  printf 'build_guard_sha256=%s\n' \
    "$(sha256_file "${snapshot_toolchain}/scripts/build_guard.sh")"
  printf 'version_env_sha256=%s\n' \
    "$(sha256_file "${snapshot_toolchain}/VERSION.env")"
  printf 'scalar_head_source_sha256=%s\n' \
    "$(sha256_file \
      "${snapshot_toolchain}/coupled/zc_kernel_nino3.F")"
  printf 'cpp_path=%s\n' "${cpp_path}"
  printf 'cpp_version=%s\n' "$("${cpp_path}" --version 2>&1 | head -n 1)"
  printf 'cpp_flags=%s\n' "${cpp_flags}"
  printf 'cpp_sha256=%s\n' "$(sha256_file "${cpp_path}")"
  printf 'patch_program=%s\n' "${patch_path}"
  printf 'patch_sha256=%s\n' "$(sha256_file "${patch_path}")"
  printf 'patch_version=%s\n' \
    "$("${patch_path}" --version 2>&1 | head -n 1)"
  printf 'kernel_source_input_manifest_sha256=%s\n' \
    "$(sha256_file \
      "${prepared_dir}/kernel_source_input_manifest.sha256")"
  printf '%s\n' \
    'patch_flags=--quiet --fuzz=0 -V none --directory <PREPARED> --strip 0'
  for patch_name in "${normalization_patches[@]}" "${fft_patch}"; do
    printf 'patch_%s_sha256=%s\n' "${patch_name}" \
      "$(sha256_file \
        "${snapshot_toolchain}/patches/${patch_name}")"
  done
} > "${prepared_dir}/preparation_provenance.txt"
manifest_files+=(preparation_provenance.txt)
{
  for name in "${manifest_files[@]}"; do
    printf '%s  %s\n' "$(sha256_file "${prepared_dir}/${name}")" "${name}"
  done
} > "${prepared_dir}/prepared_source_manifest.sha256"
{
  printf 'binding_schema=zc-prepared-manifest-binding-v1\n'
  printf 'prepared_source_manifest_sha256=%s\n' \
    "$(sha256_file "${prepared_dir}/prepared_source_manifest.sha256")"
  printf 'preparation_provenance_sha256=%s\n' \
    "$(sha256_file "${prepared_dir}/preparation_provenance.txt")"
  printf 'upstream_preprocessed_manifest_sha256=%s\n' \
    "$(sha256_file \
      "${prepared_dir}/upstream_preprocessed_manifest.sha256")"
  printf 'kernel_source_input_manifest_sha256=%s\n' \
    "$(sha256_file \
      "${prepared_dir}/kernel_source_input_manifest.sha256")"
} > "${prepared_dir}/prepared_manifest_binding.txt"

# Self-check the staged result. The binding is deliberately outside the
# content manifest to avoid a circular hash; downstream stages verify both.
zc_verify_sha256_manifest_exact "${prepared_dir}" \
  "${prepared_dir}/prepared_source_manifest.sha256" \
  prepared_source_manifest.sha256 prepared_manifest_binding.txt \
  upstream_preprocessed_manifest.sha256 kernel_source_input_manifest.sha256
zc_verify_binding_file_hash "${prepared_dir}/prepared_manifest_binding.txt" \
  prepared_source_manifest_sha256 \
  "${prepared_dir}/prepared_source_manifest.sha256"
zc_verify_binding_file_hash "${prepared_dir}/prepared_manifest_binding.txt" \
  preparation_provenance_sha256 \
  "${prepared_dir}/preparation_provenance.txt"
zc_verify_binding_file_hash "${prepared_dir}/prepared_manifest_binding.txt" \
  upstream_preprocessed_manifest_sha256 \
  "${prepared_dir}/upstream_preprocessed_manifest.sha256"
zc_verify_binding_file_hash "${prepared_dir}/prepared_manifest_binding.txt" \
  kernel_source_input_manifest_sha256 \
  "${prepared_dir}/kernel_source_input_manifest.sha256"

# Reverify every immutable copied input and each executing tool/script after
# preprocessing and patching, then publish the completed directory atomically.
zc_test_failpoint prepare_after_tool_execution
zc_verify_sha256_manifest_exact "${input_snapshot}" \
  "${preparation_input_manifest}"
zc_verify_path_state "${input_snapshot}" "${input_snapshot_state}"
zc_verify_path_state "${BASH_SOURCE[0]}" "${script_state}"
zc_verify_path_state "${toolchain_dir}/scripts/build_guard.sh" \
  "${guard_state}"
zc_verify_path_state "${cpp_path}" "${cpp_state}"
zc_verify_path_state "${patch_path}" "${patch_state}"
zc_publish_stage_directory "${build_dir}" "${prepared_target}" \
  "${prepared_dir}" "${prepared_target_state}" standard

trap - EXIT HUP INT TERM
zc_remove_stage_workspace "${build_dir}" "${workspace}" \
  "${workspace_identity}"
workspace=""
prepared_dir="${prepared_target}"
printf 'Prepared coupled kernel sources in %s\n' "${prepared_dir}"
