#!/usr/bin/env bash
set -euo pipefail

project_root="${1:?usage: generate_coupled_on_cluster.sh PROJECT_ROOT RUN_NAME}"
run_name="${2:?usage: generate_coupled_on_cluster.sh PROJECT_ROOT RUN_NAME}"
toolchain_dir="${project_root}/adjoint/tapenade_toolchain"
run_dir="${toolchain_dir}/build/${run_name}"
tools_parent="/tmp/zc_tapenade_tools"

# The cluster launcher creates a fresh container for every job, so /tmp cannot
# be reused across jobs.  Verify and unpack the pinned archive on every run.
export TAPENADE_ARCHIVE="${TAPENADE_ARCHIVE:-/home/${USER}/tapenade_3.16.tar}"
tapenade_home="$("${toolchain_dir}/scripts/install_tapenade_linux.sh" \
  "${tools_parent}" | tail -n 1)"
"${toolchain_dir}/scripts/generate_coupled_nino3.sh" \
  "${tapenade_home}" \
  "${run_dir}"
