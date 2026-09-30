#!/usr/bin/env bash
set -euo pipefail

toolchain_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${toolchain_dir}/VERSION.env"
# shellcheck disable=SC1091
source "${toolchain_dir}/scripts/build_guard.sh"

requested_install_parent="${1:-${toolchain_dir}/build/tools}"

curl_path="$(command -v curl)" || exit 2
tar_path="$(command -v tar)" || exit 2
command -v python3 >/dev/null
java_path="$(command -v java)" || {
  printf 'ERROR: Java 17 is required to install and run Tapenade\n' >&2
  exit 2
}
curl_path="$(zc_resolve_path "${curl_path}")"
tar_path="$(zc_resolve_path "${tar_path}")"
java_path="$(zc_resolve_path "${java_path}")"
java_version="$("${java_path}" -version 2>&1 | head -n 1)"
if ! grep -Eq 'version "17([.]|")' <<<"${java_version}"; then
  printf 'ERROR: Java 17 is required; observed: %s\n' "${java_version}" >&2
  exit 2
fi
installer_state="$(zc_capture_path_state "${BASH_SOURCE[0]}")"
guard_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/build_guard.sh")"
version_state="$(zc_capture_path_state "${toolchain_dir}/VERSION.env")"
curl_state="$(zc_capture_path_state "${curl_path}")"
tar_state="$(zc_capture_path_state "${tar_path}")"
installer_hash="$(zc_sha256_file "${BASH_SOURCE[0]}")"
guard_hash="$(zc_sha256_file "${toolchain_dir}/scripts/build_guard.sh")"
version_hash="$(zc_sha256_file "${toolchain_dir}/VERSION.env")"

install_parent="$(zc_assert_no_symlink_components \
  "${requested_install_parent}" 'Tapenade install parent')"
mkdir -p "${install_parent}"
install_parent="$(zc_assert_no_symlink_components \
  "${install_parent}" 'Tapenade install parent')"
[[ -d "${install_parent}" && ! -L "${install_parent}" ]] || {
  printf 'ERROR: Tapenade install parent is not a real directory: %s\n' \
    "${install_parent}" >&2
  exit 2
}

requested_archive="${TAPENADE_ARCHIVE:-${install_parent}/tapenade_${TAPENADE_VERSION}.tar}"
archive_parent="$(zc_assert_no_symlink_components \
  "$(dirname "${requested_archive}")" 'Tapenade archive parent')"
mkdir -p "${archive_parent}"
archive_parent="$(zc_assert_no_symlink_components \
  "${archive_parent}" 'Tapenade archive parent')"
archive="$(zc_assert_no_symlink_components \
  "${requested_archive}" 'Tapenade archive')"
[[ "$(dirname "${archive}")" == "${archive_parent}" ]]

download_temp=""
download_identity=""
archive_identity=""
archive_published_by_installer=0
extract_archive=""
extract_archive_identity=""
extract_root=""
extract_root_identity=""
install_complete=0

cleanup_install() {
  local status="$?"
  trap - EXIT HUP INT TERM
  if [[ "${install_complete}" -eq 0 && \
        "${archive_published_by_installer}" -eq 1 && \
        -n "${archive_identity}" ]]; then
    zc_unlink_if_identity "${archive}" "${archive_identity}" || status=2
  fi
  if [[ -n "${download_temp}" && -n "${download_identity}" ]]; then
    zc_unlink_if_identity "${download_temp}" "${download_identity}" || status=2
  fi
  if [[ -n "${extract_archive}" && -n "${extract_archive_identity}" ]]; then
    zc_unlink_if_identity "${extract_archive}" \
      "${extract_archive_identity}" || status=2
  fi
  if [[ -n "${extract_root}" && -d "${extract_root}" && \
        ! -L "${extract_root}" && \
        "$(zc_path_identity "${extract_root}")" == \
          "${extract_root_identity}" && \
        -f "${extract_root}/.zc_tapenade_extract" && \
        ! -L "${extract_root}/.zc_tapenade_extract" && \
        "$(cat "${extract_root}/.zc_tapenade_extract")" == \
          managed-zc-tapenade-extract-v1 ]]; then
    rm -rf -- "${extract_root}"
  fi
  exit "${status}"
}
trap cleanup_install EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -e "${archive}" || -L "${archive}" ]]; then
  actual_sha="$(zc_verify_single_link_regular_sha256 \
    "${archive}" "${TAPENADE_ARCHIVE_SHA256}")"
else
  download_temp="$(mktemp "${archive_parent}/.tapenade-download.XXXXXX")"
  download_identity="$(zc_path_identity "${download_temp}")"
  "${curl_path}" --fail --location --silent --show-error \
    "${TAPENADE_ARCHIVE_URL}" --output "${download_temp}"
  actual_sha="$(zc_verify_single_link_regular_sha256 \
    "${download_temp}" "${TAPENADE_ARCHIVE_SHA256}")"
  zc_test_failpoint installer_after_download
  if [[ -e "${archive}" || -L "${archive}" ]]; then
    printf 'ERROR: Tapenade archive target appeared during download: %s\n' \
      "${archive}" >&2
    exit 2
  fi
  archive_published_by_installer=1
  ln "${download_temp}" "${archive}"
  archive_identity="${download_identity}"
  zc_test_failpoint installer_after_archive_link
  zc_unlink_if_identity "${download_temp}" "${download_identity}"
  download_temp=""
  download_identity=""
  actual_sha="$(zc_verify_single_link_regular_sha256 \
    "${archive}" "${TAPENADE_ARCHIVE_SHA256}")"
fi

# Extract from an owned, checksum-verified copy. Concurrent changes to a valid
# external cache therefore cannot change the bytes consumed by tar.
extract_archive="$(mktemp "${install_parent}/.tapenade-archive.XXXXXX")"
extract_archive_identity="$(zc_path_identity "${extract_archive}")"
cp "${archive}" "${extract_archive}"
zc_verify_single_link_regular_sha256 \
  "${extract_archive}" "${TAPENADE_ARCHIVE_SHA256}" >/dev/null

# Always extract a fresh tree from the checksum-verified archive. Reusing a
# preexisting directory after checking only its launcher/parser would leave
# the remainder of the AD distribution outside the provenance boundary.
extract_root="$(mktemp -d "${install_parent}/.tapenade-extract.XXXXXX")"
printf '%s\n' managed-zc-tapenade-extract-v1 > \
  "${extract_root}/.zc_tapenade_extract"
extract_root_identity="$(zc_path_identity "${extract_root}")"
"${tar_path}" -xzf "${extract_archive}" -C "${extract_root}"
zc_verify_single_link_regular_sha256 \
  "${extract_archive}" "${TAPENADE_ARCHIVE_SHA256}" >/dev/null
zc_unlink_if_identity "${extract_archive}" "${extract_archive_identity}"
extract_archive=""
extract_archive_identity=""
install_dir="${extract_root}/tapenade_${TAPENADE_VERSION}"
for required in bin/tapenade bin/linux/fortranParser \
  ADFirstAidKit/adStack.c ADFirstAidKit/adStack.h \
  ADFirstAidKit/adComplex.h; do
  test -s "${install_dir}/${required}" || {
    printf 'ERROR: pinned Tapenade archive lacks %s\n' "${required}" >&2
    exit 1
  }
done
test -x "${install_dir}/bin/tapenade"
test -x "${install_dir}/bin/linux/fortranParser"

version_output="$("${install_dir}/bin/tapenade" -version 2>&1)"
grep -Fq "Tapenade ${TAPENADE_VERSION}" <<<"${version_output}"
grep -Fq "Revision: ${TAPENADE_REVISION}" <<<"${version_output}"

# The receipt identifies the scripts and pinned constants that actually drove
# this extraction. Refuse publication if any of them, or the resolved download
# and extraction tools, changed while the installer was running.
zc_verify_path_state "${BASH_SOURCE[0]}" "${installer_state}"
zc_verify_path_state "${toolchain_dir}/scripts/build_guard.sh" \
  "${guard_state}"
zc_verify_path_state "${toolchain_dir}/VERSION.env" "${version_state}"
zc_verify_path_state "${curl_path}" "${curl_state}"
zc_verify_path_state "${tar_path}" "${tar_state}"

# Bind the *complete observed extraction*, not just the launcher and parser.
# Any later unmanifested JAR/resource edit makes generation refuse this tree.
tree_manifest="${install_dir}/.zc_tapenade_tree_manifest.sha256"
install_receipt="${install_dir}/.zc_tapenade_install_receipt.txt"
if find "${install_dir}" -type l -print -quit | grep -q .; then
  printf 'ERROR: pinned Tapenade extraction contains a symbolic link\n' >&2
  exit 2
fi
(
  cd "${install_dir}"
  find . -type f \
    ! -name '.zc_tapenade_tree_manifest.sha256' \
    ! -name '.zc_tapenade_install_receipt.txt' -print |
    sed 's#^\./##' | LC_ALL=C sort |
    while IFS= read -r relative_name; do
      printf '%s  %s\n' "$(zc_sha256_file "${relative_name}")" \
        "${relative_name}"
    done
) > "${tree_manifest}"
test -s "${tree_manifest}"
{
  printf 'receipt_schema=zc-tapenade-install-receipt-v1\n'
  printf 'tapenade_version=%s\n' "${TAPENADE_VERSION}"
  printf 'tapenade_revision=%s\n' "${TAPENADE_REVISION}"
  printf 'archive_url=%s\n' "${TAPENADE_ARCHIVE_URL}"
  printf 'expected_archive_sha256=%s\n' "${TAPENADE_ARCHIVE_SHA256}"
  printf 'observed_archive_sha256=%s\n' "${actual_sha}"
  printf 'tree_manifest_sha256=%s\n' "$(zc_sha256_file "${tree_manifest}")"
  printf 'tree_file_count=%s\n' \
    "$(wc -l < "${tree_manifest}" | tr -d '[:space:]')"
  printf 'installer_script_sha256=%s\n' \
    "${installer_hash}"
  printf 'build_guard_sha256=%s\n' \
    "${guard_hash}"
  printf 'version_env_sha256=%s\n' \
    "${version_hash}"
} > "${install_receipt}"
# Keep the installation receipt content-addressed and host-independent so an
# identical archive extraction can be verified on the Linux generation host
# and the macOS compilation host.  The Java executable and version actually
# used for differentiation are recorded separately in generation provenance.
zc_verify_tapenade_install "${install_dir}" "${toolchain_dir}" \
  "${TAPENADE_VERSION}" "${TAPENADE_REVISION}" \
  "${TAPENADE_ARCHIVE_SHA256}"
zc_verify_path_state "${BASH_SOURCE[0]}" "${installer_state}"
zc_verify_path_state "${toolchain_dir}/scripts/build_guard.sh" \
  "${guard_state}"
zc_verify_path_state "${toolchain_dir}/VERSION.env" "${version_state}"
zc_verify_path_state "${curl_path}" "${curl_state}"
zc_verify_path_state "${tar_path}" "${tar_state}"

# Successful callers own this fresh content-addressed extraction through its
# unique parent. The final line is the machine-readable return value used by
# the orchestration scripts.
install_complete=1
trap - EXIT HUP INT TERM
printf '%s\n' "${version_output}"
printf '%s\n' "${install_dir}"
