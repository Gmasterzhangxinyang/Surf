#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INPUT="${1:-${PROJECT_ROOT}/outputs/parking_slot_agent_v2_frame_9277/part1_fresh30m/part1_output.json}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/outputs/parking_slot_agent_v2_frame_9277/qwen3_5_0_8b}"
BASE_URL="${PARKING_VLM_BASE_URL:-http://127.0.0.1:8010/v1}"
MODEL_NAME="${PARKING_VLM_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"

if [[ ! -f "${INPUT}" ]]; then
  echo "Part1 input does not exist: ${INPUT}" >&2
  exit 2
fi
if ! MODELS_PAYLOAD="$(curl --fail --silent --show-error "${BASE_URL}/models")"; then
  echo "Local VLM is not ready at ${BASE_URL}" >&2
  exit 3
fi
if ! PARKING_VLM_MODELS_PAYLOAD="${MODELS_PAYLOAD}" python3 - "${MODEL_NAME}" <<'PY'
import json
import os
import sys

expected = sys.argv[1]
try:
    payload = json.loads(os.environ["PARKING_VLM_MODELS_PAYLOAD"])
    model_ids = {
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if expected in model_ids else 1)
PY
then
  echo "Local endpoint does not serve the required model: ${MODEL_NAME}" >&2
  exit 3
fi

cd "${PROJECT_ROOT}"
exec python3 -m parking_slot_agent_v2 run-part2 \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL_NAME}" \
  --replay-output "${OUTPUT_DIR}/replay_actions.json"
