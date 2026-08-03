#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${PARKING_VLM_VENV_DIR:-${PROJECT_ROOT}/.venv-vlm}"
MODEL_DIR="${PARKING_VLM_MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3.5-0.8B}"
SERVED_MODEL_NAME="${PARKING_VLM_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
VLM_HOST="${PARKING_VLM_HOST:-127.0.0.1}"
VLM_PORT="${PARKING_VLM_PORT:-8010}"
GPU_MEMORY_UTILIZATION="${PARKING_VLM_GPU_MEMORY_UTILIZATION:-0.12}"
MAX_MODEL_LEN="${PARKING_VLM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${PARKING_VLM_MAX_NUM_SEQS:-1}"
MAX_IMAGES="${PARKING_VLM_MAX_IMAGES:-4}"
MIN_FREE_MIB="${PARKING_VLM_MIN_FREE_MIB:-7500}"

# FlashInfer's sampling kernels JIT-compile with nvcc on first use. This host has
# the CUDA runtime but no CUDA toolkit, so keep sampling on vLLM's native path.
export VLLM_USE_FLASHINFER_SAMPLER="${PARKING_VLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
  echo "vLLM is missing: ${VENV_DIR}/bin/vllm" >&2
  echo "Install the pinned runtime with: ${VENV_DIR}/bin/python -m pip install vllm==0.25.1" >&2
  exit 2
fi

for required_file in config.json model.safetensors.index.json; do
  if [[ ! -s "${MODEL_DIR}/${required_file}" ]]; then
    echo "Model metadata is incomplete: ${MODEL_DIR}/${required_file}" >&2
    exit 2
  fi
done

while read -r weight_file expected_bytes; do
  weight_path="${MODEL_DIR}/${weight_file}"
  if [[ ! -f "${weight_path}" ]]; then
    echo "Model weight is missing: ${weight_path}" >&2
    exit 2
  fi
  actual_bytes="$(stat -c '%s' "${weight_path}")"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "Model weight has the wrong size: ${weight_path} (${actual_bytes}/${expected_bytes} bytes)" >&2
    exit 2
  fi
done <<'MODEL_WEIGHTS'
model.safetensors-00001-of-00001.safetensors 1746942600
MODEL_WEIGHTS

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the startup memory guard" >&2
  exit 2
fi

FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
if [[ ! "${FREE_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Unable to read free GPU memory" >&2
  exit 2
fi
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "Refusing to start: ${FREE_MIB} MiB GPU memory is free; ${MIN_FREE_MIB} MiB is required." >&2
  echo "Wait for the existing GPU workload to finish or explicitly lower PARKING_VLM_MIN_FREE_MIB." >&2
  exit 3
fi

exec "${VENV_DIR}/bin/vllm" serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${VLM_HOST}" \
  --port "${VLM_PORT}" \
  --generation-config vllm \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES},\"video\":0}" \
  --enforce-eager
