#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-${PROJECT_ROOT}/outputs/parking_slot_agent_v2_frame_9277/part1_fresh30m/part1_output.json}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/outputs/parking_slot_agent_v2_frame_9277/openai_gpt_5_6_terra}"
KEY_FILE="${PARKING_OPENAI_KEY_FILE:-${PROJECT_ROOT}/.secrets/openai_api_key}"
RUNTIME_DIR="${PARKING_OPENAI_RUNTIME_DIR:-${PROJECT_ROOT}/.openai-runtime}"

if [[ ! -s "${KEY_FILE}" ]]; then
  echo "OpenAI key file is missing or empty: ${KEY_FILE}" >&2
  exit 2
fi
if [[ "$(stat -c '%a' "${KEY_FILE}")" != "600" ]]; then
  echo "OpenAI key file must have permission 600: ${KEY_FILE}" >&2
  exit 2
fi
if [[ ! -d "${RUNTIME_DIR}/openai" ]]; then
  echo "OpenAI runtime is missing. Install it with:" >&2
  echo "python3 -m pip install --target '${RUNTIME_DIR}' -r '${PROJECT_ROOT}/configs/openai_vlm_requirements.txt'" >&2
  exit 2
fi

export PYTHONPATH="${RUNTIME_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PREFLIGHT_ARGS=(
  --input "${INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --key-file "${KEY_FILE}"
)
if [[ "${PARKING_ALLOW_OVERWRITE:-0}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--allow-existing-output)
fi
python3 "${PROJECT_ROOT}/scripts/check_parking_slot_agent_v2_openai.py" "${PREFLIGHT_ARGS[@]}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"
LOCK_DIR="${OUTPUT_DIR}.lock"
if ! mkdir "${LOCK_DIR}"; then
  echo "Another Part2 run may be using this output directory: ${LOCK_DIR}" >&2
  exit 3
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT INT TERM

mkdir -p "${OUTPUT_DIR}"
RUN_ARGS=(
  run-part2
  --input "${INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --openai-model gpt-5.6-terra
  --openai-reasoning-effort medium
  --openai-image-detail original
  --openai-api-key-file "${KEY_FILE}"
  --replay-output "${OUTPUT_DIR}/replay_actions.json"
)
if [[ "${PARKING_EVALUATION_EXHAUSTIVE:-0}" == "1" ]]; then
  RUN_ARGS+=(--evaluation-exhaustive)
fi
python3 -m parking_slot_agent_v2 "${RUN_ARGS[@]}"
