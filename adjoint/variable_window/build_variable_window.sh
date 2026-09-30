#!/usr/bin/env bash
set -euo pipefail

variable_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd "${variable_dir}/../.." && pwd -P)"
canonical_build="${1:-${repository_root}/adjoint/tapenade_toolchain/build/coupled_run23_final3}"
requested_output="${2:-${variable_dir}/build}"
object_manifest="${variable_dir}/canonical_final3_objects.sha256"

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

if [[ -L "${canonical_build}" || ! -d "${canonical_build}" ]]; then
  printf 'ERROR: canonical final3 build is missing or symbolic: %s\n' \
    "${canonical_build}" >&2
  exit 2
fi
canonical_build="$(cd "${canonical_build}" && pwd -P)"
compiled="${canonical_build}/coupled_compiled"
if [[ -L "${compiled}" || ! -d "${compiled}" ]]; then
  printf 'ERROR: canonical compiled stage is missing or symbolic: %s\n' \
    "${compiled}" >&2
  exit 2
fi

manifest_value() {
  local key="$1"
  local file="$2"
  local count value
  count="$(awk -F= -v wanted="${key}" '$1 == wanted {n += 1} END {print n + 0}' "${file}")"
  value="$(awk -F= -v wanted="${key}" '$1 == wanted {sub(/^[^=]*=/, ""); print}' "${file}")"
  if [[ "${count}" != 1 || -z "${value}" ]]; then
    printf 'ERROR: canonical manifest key is not unique: %s\n' "${key}" >&2
    exit 2
  fi
  printf '%s' "${value}"
}

verify_canonical_objects() {
  local expected relative extra candidate actual count
  count=0
  while read -r expected relative extra; do
    [[ -n "${expected:-}" ]] || continue
    if [[ -n "${extra:-}" || ! "${expected}" =~ ^[0-9a-f]{64}$ || \
          -z "${relative:-}" || "${relative}" = /* || \
          "${relative}" == *".."* ]]; then
      printf 'ERROR: invalid canonical object-manifest entry\n' >&2
      exit 2
    fi
    candidate="${compiled}/${relative}"
    if [[ -L "${candidate}" || ! -f "${candidate}" || ! -s "${candidate}" ]]; then
      printf 'ERROR: canonical final3 input is missing or symbolic: %s\n' \
        "${candidate}" >&2
      exit 2
    fi
    actual="$(sha256_file "${candidate}")"
    if [[ "${actual}" != "${expected}" ]]; then
      printf 'ERROR: canonical final3 input hash mismatch: %s\n' \
        "${candidate}" >&2
      exit 2
    fi
    count=$((count + 1))
  done < "${object_manifest}"
  if [[ "${count}" -ne 45 ]]; then
    printf 'ERROR: canonical object manifest must contain 45 entries\n' >&2
    exit 2
  fi
}

verify_canonical_objects
build_manifest="${compiled}/build_manifest.txt"
for key in build_input_manifest path_executable adjoint_executable \
  tangent_archive; do
  recorded="$(manifest_value "${key}_sha256" "${build_manifest}")"
  case "${key}" in
    build_input_manifest) file="${compiled}/build_input_manifest.sha256" ;;
    path_executable) file="${compiled}/zc_adjoint_make_path" ;;
    adjoint_executable) file="${compiled}/zc_nino3_adjoint" ;;
    tangent_archive) file="${compiled}/libzc_tangent.a" ;;
  esac
  if [[ "$(sha256_file "${file}")" != "${recorded}" ]]; then
    printf 'ERROR: canonical build manifest no longer binds %s\n' "${file}" >&2
    exit 2
  fi
done

expected_flags='-std=legacy -O0 -g -fcheck=all -fbacktrace -fallow-argument-mismatch -ffixed-line-length-none'
if [[ "$(manifest_value derivative_flags "${build_manifest}")" != \
      "${expected_flags}" ]]; then
  printf 'ERROR: canonical final3 derivative flags changed\n' >&2
  exit 2
fi
read -r -a derivative_flags <<< "${expected_flags}"
platform_text="$(manifest_value platform_flags "${build_manifest}")"
platform_flags=()
if [[ -n "${platform_text}" ]]; then
  read -r -a platform_flags <<< "${platform_text}"
fi

requested_fc="${FC:-$(manifest_value compiler_path "${build_manifest}")}"
if [[ "${requested_fc}" == */* ]]; then
  fc_path="${requested_fc}"
else
  fc_path="$(command -v "${requested_fc}" || true)"
fi
if [[ -z "${fc_path}" || -L "${fc_path}" || ! -x "${fc_path}" ]]; then
  printf 'ERROR: exact final3 compiler is unavailable or symbolic: %s\n' \
    "${requested_fc}" >&2
  exit 2
fi
fc_path="$(cd "$(dirname "${fc_path}")" && pwd -P)/$(basename "${fc_path}")"
expected_fc_sha256="$(manifest_value compiler_sha256 "${build_manifest}")"
if [[ "$(sha256_file "${fc_path}")" != "${expected_fc_sha256}" ]]; then
  printf 'ERROR: compiler does not match canonical final3: %s\n' \
    "${fc_path}" >&2
  exit 2
fi

include_dir="${compiled}/build_input_snapshot/prepared"
runtime_source="${variable_dir}/zc_variable_runtime.F"
path_source="${variable_dir}/zc_variable_path_driver.F"
adjoint_source="${variable_dir}/zc_variable_adjoint_driver.F"
tangent_source="${variable_dir}/zc_variable_tangent_driver.F"
for source in "${runtime_source}" "${path_source}" "${adjoint_source}" \
  "${tangent_source}"; do
  if [[ -L "${source}" || ! -f "${source}" || ! -s "${source}" ]]; then
    printf 'ERROR: variable-window source is missing or symbolic: %s\n' \
      "${source}" >&2
    exit 2
  fi
done

output_parent="$(dirname "${requested_output}")"
output_name="$(basename "${requested_output}")"
if [[ -z "${output_name}" || "${output_name}" == . || \
      "${output_name}" == .. ]]; then
  printf 'ERROR: invalid output directory: %s\n' "${requested_output}" >&2
  exit 2
fi
mkdir -p "${output_parent}"
output_parent="$(cd "${output_parent}" && pwd -P)"
output_dir="${output_parent}/${output_name}"
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  printf 'ERROR: output directory already exists: %s\n' "${output_dir}" >&2
  exit 2
fi
staging="$(mktemp -d "${output_parent}/.${output_name}.staging.XXXXXX")"

cleanup() {
  local status="$?"
  trap - EXIT HUP INT TERM
  if [[ -n "${staging:-}" && -d "${staging}" && \
        "${staging}" == "${output_parent}/.${output_name}.staging."* ]]; then
    rm -rf -- "${staging}"
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

source_manifest="${staging}/driver_sources.sha256"
for source in "${runtime_source}" "${path_source}" "${adjoint_source}" \
  "${tangent_source}"; do
  printf '%s  %s\n' "$(sha256_file "${source}")" "$(basename "${source}")"
done > "${source_manifest}"

primal_objects=(
  "${compiled}/primal/zc_kernel_api.o"
  "${compiled}/primal/ssta.o"
  "${compiled}/primal/ztmfc1.o"
  "${compiled}/primal/cforce.o"
  "${compiled}/primal/mloop.o"
  "${compiled}/primal/akcalc.o"
  "${compiled}/primal/bndary.o"
  "${compiled}/primal/uhcalc.o"
  "${compiled}/primal/uhinit.o"
  "${compiled}/primal/tridag.o"
  "${compiled}/primal/fft2c.o"
)
reverse_objects=(
  "${compiled}/reverse/akcalc_b.o"
  "${compiled}/reverse/bndary_b.o"
  "${compiled}/reverse/cforce_b.o"
  "${compiled}/reverse/fft2c_b.o"
  "${compiled}/reverse/mloop_b.o"
  "${compiled}/reverse/ssta_b.o"
  "${compiled}/reverse/tridag_b.o"
  "${compiled}/reverse/uhcalc_b.o"
  "${compiled}/reverse/uhinit_b.o"
  "${compiled}/reverse/zc_kernel_api_b.o"
  "${compiled}/reverse/zc_kernel_nino3_b.o"
  "${compiled}/reverse/ztmfc1_b.o"
)
support_objects=("${compiled}/support/"*.o)

compile_source() {
  local source="$1"
  local object="$2"
  "${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
    -I "${include_dir}" -c "${source}" -o "${object}"
}

compile_source "${runtime_source}" "${staging}/zc_variable_runtime.o"
compile_source "${path_source}" "${staging}/zc_variable_path_driver.o"
compile_source "${adjoint_source}" \
  "${staging}/zc_variable_adjoint_driver.o"
compile_source "${tangent_source}" \
  "${staging}/zc_variable_tangent_driver.o"

"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -o "${staging}/zc_variable_make_path" \
  "${staging}/zc_variable_path_driver.o" \
  "${staging}/zc_variable_runtime.o" \
  "${primal_objects[@]}" "${support_objects[@]}"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -o "${staging}/zc_variable_adjoint" \
  "${staging}/zc_variable_adjoint_driver.o" \
  "${staging}/zc_variable_runtime.o" \
  "${reverse_objects[@]}" "${compiled}/primal/ssta.o" \
  "${support_objects[@]}"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -o "${staging}/zc_variable_tangent" \
  "${staging}/zc_variable_tangent_driver.o" \
  "${staging}/zc_variable_runtime.o" \
  "${compiled}/libzc_tangent.a" "${support_objects[@]}"

verify_canonical_objects
while read -r expected name; do
  if [[ "$(sha256_file "${variable_dir}/${name}")" != "${expected}" ]]; then
    printf 'ERROR: variable-window source changed during build: %s\n' \
      "${name}" >&2
    exit 2
  fi
done < "${source_manifest}"

{
  printf 'schema=zc-variable-window-build-v1\n'
  printf 'canonical_build_manifest_sha256=%s\n' \
    "$(sha256_file "${build_manifest}")"
  printf 'canonical_object_manifest_sha256=%s\n' \
    "$(sha256_file "${object_manifest}")"
  printf 'canonical_build_input_manifest_sha256=%s\n' \
    "$(sha256_file "${compiled}/build_input_manifest.sha256")"
  printf 'compiler_path=%s\n' "${fc_path}"
  printf 'compiler_sha256=%s\n' "${expected_fc_sha256}"
  printf 'compiler=%s\n' "$("${fc_path}" --version | head -n 1)"
  printf 'derivative_flags=%s\n' "${expected_flags}"
  printf 'platform_flags=%s\n' "${platform_text}"
  printf 'driver_sources_sha256=%s\n' "$(sha256_file "${source_manifest}")"
  printf 'path_executable_sha256=%s\n' \
    "$(sha256_file "${staging}/zc_variable_make_path")"
  printf 'adjoint_executable_sha256=%s\n' \
    "$(sha256_file "${staging}/zc_variable_adjoint")"
  printf 'tangent_executable_sha256=%s\n' \
    "$(sha256_file "${staging}/zc_variable_tangent")"
} > "${staging}/build_provenance.txt"

mv "${staging}" "${output_dir}"
staging=''
trap - EXIT HUP INT TERM
printf '%s\n%s\n%s\n' \
  "${output_dir}/zc_variable_make_path" \
  "${output_dir}/zc_variable_adjoint" \
  "${output_dir}/zc_variable_tangent"
