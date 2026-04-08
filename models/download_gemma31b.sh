#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mnt/carles/.cache}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-src/maxtext/configs/base.yml}"
MODEL_NAME="${MODEL_NAME:-gemma3-1b}"
HF_MODEL_PATH="${HF_MODEL_PATH:-google/gemma-3-1b-it}"
HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-${HF_TOKEN:-}}"
SAVE_DTYPE="${SAVE_DTYPE:-bfloat16}"
LAZY_LOAD_TENSORS="${LAZY_LOAD_TENSORS:-false}"

MODE="${1:-both}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
FREE_OUT="${FREE_OUT:-${MODELS_DIR}/gemma31b}"
SCAN_OUT="${SCAN_OUT:-${MODELS_DIR}/gemma31b-scan}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at '${PYTHON_BIN}'." >&2
  echo "Set PYTHON_BIN or create the MaxText venv first." >&2
  exit 1
fi

mkdir -p "${MODELS_DIR}"

run_convert() {
  local scan_layers="$1"
  local out_dir="$2"
  local run_name="$3"

  local -a args=(
    -m maxtext.checkpoint_conversion.to_maxtext
    "${CONFIG_PATH}"
    "model_name=${MODEL_NAME}"
    "base_output_directory=${out_dir}"
    "run_name=${run_name}"
    "hf_access_token=${HF_ACCESS_TOKEN}"
    "hardware=cpu"
    "skip_jax_distributed_system=True"
    "scan_layers=${scan_layers}"
    "--save_dtype=${SAVE_DTYPE}"
  )

  if [[ -n "${HF_MODEL_PATH}" ]]; then
    args+=("--hf_model_path=${HF_MODEL_PATH}")
  fi

  if [[ "${LAZY_LOAD_TENSORS}" == "true" || "${LAZY_LOAD_TENSORS}" == "True" ]]; then
    args+=("--lazy_load_tensors=True")
  fi

  echo "Converting ${MODEL_NAME} -> ${out_dir} (scan_layers=${scan_layers})"
  "${PYTHON_BIN}" "${args[@]}"
}

case "${MODE}" in
  free)
    run_convert "false" "${FREE_OUT}" "${MODEL_NAME}-to-maxtext-free"
    ;;
  scan)
    run_convert "true" "${SCAN_OUT}" "${MODEL_NAME}-to-maxtext-scan"
    ;;
  both)
    run_convert "false" "${FREE_OUT}" "${MODEL_NAME}-to-maxtext-free"
    run_convert "true" "${SCAN_OUT}" "${MODEL_NAME}-to-maxtext-scan"
    ;;
  *)
    echo "Usage: $0 [free|scan|both]" >&2
    exit 1
    ;;
esac

echo "Done."
