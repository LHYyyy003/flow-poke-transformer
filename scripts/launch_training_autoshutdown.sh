#!/usr/bin/env bash
# Launch training detached from SSH. Power off this ordinary AutoDL instance
# locally after either successful completion or a training error.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/lhy/flow-poke-transformer-repro}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
SCRIPT_PATH="$(readlink -f "$0")"
RUNNER="${PROJECT_DIR}/scripts/train_billiards_physics_modelscope.sh"
SHUTDOWN_COMMAND="${SHUTDOWN_COMMAND:-/usr/bin/shutdown -h now}"

if [[ "${AUTODL_LAUNCH_CHILD:-0}" != "1" ]]; then
  [[ -x "${RUNNER}" || -f "${RUNNER}" ]] || { echo "Missing runner: ${RUNNER}" >&2; exit 1; }

  mkdir -p "${LOG_DIR}"
  run_id="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/physics_train_${run_id}.log"
  pid_file="${LOG_DIR}/physics_train_${run_id}.pid"

  export AUTODL_LAUNCH_CHILD=1 LOG_FILE="${log_file}"
  nohup bash "${SCRIPT_PATH}" >"${log_file}" 2>&1 < /dev/null &
  echo $! >"${pid_file}"
  echo "Started in background."
  echo "PID: $(cat "${pid_file}")"
  echo "Log: ${log_file}"
  echo "Follow with: tail -f ${log_file}"
  exit 0
fi

cd "${PROJECT_DIR}"
set +e
bash "${RUNNER}"
train_status=$?
set -e

echo "Training ended with status ${train_status}; powering off this AutoDL instance..."
sync
exec ${SHUTDOWN_COMMAND}
